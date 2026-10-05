"""MoEGPT: chintana's GPT with some (or all) feed-forward MLPs swapped for MoE layers,
and upcycle(), which converts a trained dense GPT into an MoEGPT.

MoEGPT subclasses chintana.GPT, so the embeddings, attention, LayerNorms,
weight tying, GPT-2 init, forward() and generate() are the base model's. The
only change is block.mlp: Block.forward calls self.mlp(self.ln_2(x)), so
replacing that module with an MoE is enough, and with moe.n_experts=0 an
MoEGPT is exactly a chintana GPT (the "dense" / Linear model).

forward() still returns (logits, cross-entropy) like GPT; the router's
auxiliary losses from the last forward are read with aux_losses() and added
to the training objective by common.train_step.
"""

import math
from dataclasses import dataclass

import torch

from chintana.gpt import GPT

from moe import MoE


@dataclass
class MoEConfig:
    n_experts: int = 8  # experts per MoE layer; 0 -> dense model (plain chintana GPT)
    top_k: int = 2  # experts each token is routed to
    expert_hidden: int = 0  # expert hidden width; 0 -> 4 * n_embd (the dense MLP's width; required for upcycling)
    moe_every: int = 1  # every n-th block gets an MoE: 1 = all blocks, 2 = blocks 1, 3, 5, ...
    normalize_topk: bool = True  # renormalize the top-k gates to sum to 1 (needed for function-preserving upcycling)
    capacity_factor: float = 0.0  # 0 -> no token dropping; else expert capacity = ceil(cf * N * k / E)
    router_jitter: float = 0.0  # multiplicative uniform noise on router inputs during training (Switch)
    aux_loss_coef: float = 0.01  # load-balancing loss weight
    z_loss_coef: float = 0.001  # router z-loss weight
    upcycle_noise: float = 0.0  # std of Gaussian noise added to each expert copy when upcycling (relative to weight std)


def moe_layer_ids(n_layer, moe_cfg):
    if moe_cfg.n_experts == 0:
        return []
    return [i for i in range(n_layer) if i % moe_cfg.moe_every == moe_cfg.moe_every - 1]


class MoEGPT(GPT):

    def __init__(self, config, moe_cfg: MoEConfig):
        super().__init__(config)
        self.moe_cfg = moe_cfg
        self.moe_layers = moe_layer_ids(config.n_layer, moe_cfg)
        hidden = moe_cfg.expert_hidden or 4 * config.n_embd
        for i in self.moe_layers:
            moe = MoE(config.n_embd, moe_cfg.n_experts, moe_cfg.top_k, hidden, bias=config.bias,
                      dropout=config.dropout, normalize_topk=moe_cfg.normalize_topk,
                      capacity_factor=moe_cfg.capacity_factor, router_jitter=moe_cfg.router_jitter)
            moe.apply(self._init_weights)
            for e in moe.experts:
                torch.nn.init.normal_(e.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
            self.transformer.h[i].mlp = moe

    def moe_modules(self):
        return [self.transformer.h[i].mlp for i in self.moe_layers]

    def aux_losses(self):
        """(load-balancing, z) losses from the last forward, averaged over MoE layers."""
        mods = self.moe_modules()
        if not mods:
            zero = self.lm_head.weight.new_zeros(())
            return zero, zero
        return (torch.stack([m.aux_loss for m in mods]).mean(),
                torch.stack([m.z_loss for m in mods]).mean())

    def param_counts(self):
        """Total parameters, and parameters a single token actually uses (top_k of n_experts)."""
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for m in self.moe_modules():
            per_expert = sum(p.numel() for p in m.experts[0].parameters())
            inactive += (m.n_experts - m.top_k) * per_expert
        return {'total': total, 'active': total - inactive}


@torch.no_grad()
def upcycle(dense: GPT, moe_cfg: MoEConfig, seed=0):
    """Sparse upcycling (Komatsuzaki et al., 2022): an MoEGPT whose every expert
    is a copy of the dense model's MLP in that block. Everything else
    (embeddings, attention, LayerNorms) is copied as is; routers start from
    the usual N(0, 0.02) init.

    With normalize_topk=True, no capacity dropping and upcycle_noise=0 the
    result computes the same function as `dense`: the gates of the chosen
    experts sum to 1 and every expert is the same MLP.
    """
    cfg = dense.config
    hidden = moe_cfg.expert_hidden or 4 * cfg.n_embd
    assert hidden == 4 * cfg.n_embd, \
        f"upcycling copies the dense MLP, so moe.expert_hidden must be 0 or {4 * cfg.n_embd}"
    torch.manual_seed(seed)
    moe_model = MoEGPT(cfg, moe_cfg)
    dense_sd = dense.state_dict()
    target = moe_model.state_dict()
    copied = set()
    for name in target:
        if name in dense_sd:
            target[name].copy_(dense_sd[name])
            copied.add(name)
    gen = torch.Generator().manual_seed(seed + 1)
    for i in moe_model.moe_layers:
        prefix = f'transformer.h.{i}.mlp.'
        for e in range(moe_cfg.n_experts):
            for pname in ('c_fc.weight', 'c_fc.bias', 'c_proj.weight', 'c_proj.bias'):
                src = dense_sd.get(prefix + pname)
                if src is None:  # bias=False
                    continue
                dst = target[f'{prefix}experts.{e}.{pname}']
                dst.copy_(src)
                if moe_cfg.upcycle_noise > 0 and src.dim() >= 2:
                    dst.add_(torch.randn(src.shape, generator=gen) * moe_cfg.upcycle_noise * src.std())
                copied.add(f'{prefix}experts.{e}.{pname}')
    moe_model.load_state_dict(target)
    fresh = [n for n in target if n not in copied and not n.endswith('.attn.bias')]
    assert all('router' in n for n in fresh), f"unexpected uninitialized params: {fresh}"
    return moe_model
