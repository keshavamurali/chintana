"""RevGPT: chintana's GPT with a swappable (optionally reversible) trunk and a
memory-lean loss head.

Subclassing chintana.GPT keeps the embeddings, the Block modules, the weight
tying, the GPT-2 initialization and generate() identical to the base model;
only forward() changes, in two independent ways:

1. The trunk. variant='baseline' is the ordinary residual stack (the
   "forward Euler" update p[l+1] = p[l] + f_l(p[l])); any other variant routes
   the same Block modules through a reversible rule from reversible.py.
2. The loss head. With the GPT-2 vocabulary (50304) the logits tensor is by
   far the largest activation in a small model -- B*T*V floats, stored once in
   the compute dtype and again in fp32 inside cross_entropy -- and it is the
   same for every trunk. loss_impl='fused' computes the loss and its gradient
   chunk by chunk (see fused_linear_cross_entropy) so the logits are never
   materialized at full size; loss_impl='standard' is the plain
   lm_head + F.cross_entropy path from chintana.GPT, kept for comparison.
"""

import torch
from torch.nn import functional as F

from chintana.gpt import GPT

from reversible import autocast_state, make_rule, run_stack


def _compute_dtype(device_type, fallback):
    enabled, dtype = autocast_state(device_type)
    return dtype if enabled else fallback


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """mean cross_entropy(h @ W^T, targets), with the gradient computed chunk
    by chunk in the forward pass and only (dL/dh, dL/dW) kept for backward.

    softmax - onehot lies in [-1, 1], so it is cast to the compute dtype
    before the 1/N mean scaling (applied later in fp32), which keeps fp16
    from flushing the small probabilities to zero.
    """

    @staticmethod
    def forward(ctx, h, weight, targets, chunk):
        n = h.shape[0]
        dtype = _compute_dtype(h.device.type, weight.dtype)
        acc = torch.promote_types(torch.float32, weight.dtype)
        w = weight.to(dtype)
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros_like(weight, dtype=acc)
        loss_sum = torch.zeros((), device=h.device, dtype=acc)
        for s in range(0, n, chunk):
            hc, tc = h[s:s + chunk].to(dtype), targets[s:s + chunk]
            logits = (hc @ w.t()).to(acc)
            lse = torch.logsumexp(logits, dim=-1)
            loss_sum += (lse - logits.gather(1, tc[:, None]).squeeze(1)).sum()
            probs = logits.sub_(lse[:, None]).exp_()
            probs[torch.arange(tc.shape[0], device=tc.device), tc] -= 1.0
            g = probs.to(dtype)
            del logits, probs
            grad_h[s:s + chunk] = g @ w
            grad_w += (g.t() @ hc).to(acc)
        grad_h.div_(n)
        grad_w.div_(n)
        ctx.save_for_backward(grad_h, grad_w.to(weight.dtype))
        return loss_sum / n

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_out, grad_w * grad_out, None, None


def fused_linear_cross_entropy(h, weight, targets, chunk=1024):
    if torch.is_grad_enabled():
        return _FusedLinearCrossEntropy.apply(h, weight, targets, chunk)
    acc = torch.promote_types(torch.float32, weight.dtype)
    total = torch.zeros((), device=h.device, dtype=acc)
    for s in range(0, h.shape[0], chunk):
        logits = F.linear(h[s:s + chunk], weight).to(acc)
        total += F.cross_entropy(logits, targets[s:s + chunk], reduction='sum')
    return total / h.shape[0]


class RevGPT(GPT):

    def __init__(self, config, variant='baseline', h=None, a_seed=0, memory_saving=True,
                 loss_impl='fused', loss_chunk=1024):
        super().__init__(config)
        assert config.dropout == 0.0 or variant == 'baseline', \
            "reversible recomputation needs a deterministic block (dropout=0)"
        assert loss_impl in ('fused', 'standard')
        self.variant = variant
        self.memory_saving = memory_saving
        self.loss_impl = loss_impl
        self.loss_chunk = loss_chunk
        self.rule = make_rule(variant, self.transformer.h, h=h, a_seed=a_seed)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        if self.rule is None:
            for block in self.transformer.h:
                x = block(x)
        else:
            x = run_stack(self.rule, x, memory_saving=self.memory_saving)
        x = self.transformer.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        if self.loss_impl == 'fused':
            loss = fused_linear_cross_entropy(x.reshape(b * t, -1), self.lm_head.weight,
                                              targets.reshape(-1), self.loss_chunk)
            return None, loss
        logits = self.lm_head(x)
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
