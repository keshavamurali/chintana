"""Reversible residual stacks for a chintana GPT trunk (Gal et al., 2025,
"Reversing Large Language Models for Efficient Training and Fine-Tuning").

Every rule here keeps a two-tensor state (s0, s1) and maps it layer by layer
through an update that can be inverted exactly (up to float rounding), so the
backward pass can rebuild each layer's input from its output instead of
reading it from memory:

  midpoint     (2.4)  p[l+1] = p[l-1] + 2h f_l(p[l])
  midpoint_a   (3.6)  p[l+1] = a_l p[l-1] + (1 - a_l) p[l] + h f_l(p[l])
  leapfrog     (2.6)  p[l+1] = 2 p[l] - p[l-1] + h^2 f_l(p[l])
  hamiltonian  (2.8)  q[l] = q[l-1] + Attn_l(LN1(p[l-1]))
                      p[l] = p[l-1] + MLP_l(LN2(q[l]))

f_l is the full transformer block update of (2.5), built from the modules of
an ordinary chintana Block. Both states start from the embedding (p[-1] =
p[0] = x, q[0] = p[0] = x), so the first layer of every two-step rule
reduces to a plain residual step when 2h = 1 / a = anything / h^2 = 1.

_ReversibleStack is the memory-saving path: its forward runs under no_grad
and keeps only the final state; its backward walks the layers in reverse,
recomputing f_l(p[l]) once with grad enabled and using that single
evaluation both to invert the step and to back-propagate through it. Peak
activation memory is therefore one layer's worth, independent of depth.
"""

import random
from contextlib import nullcontext

import torch


def block_update(block, x):
    """f of eq. (2.5): the residual increment of a pre-norm transformer block."""
    a = block.attn(block.ln_1(x))
    return a + block.mlp(block.ln_2(x + a))


def autocast_state(device_type):
    """(enabled, dtype) of the autocast region the caller is running in."""
    try:
        return torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)
    except (TypeError, AttributeError):  # torch < 2.4
        if device_type == 'cuda':
            return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()
        return torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()


def _autocast_factory(device_type):
    """Capture the caller's autocast state so backward can replay it.

    Autograd runs CUDA backward on its own thread, where autocast is off, so
    the recomputation of f_l would silently switch precision without this.
    """
    enabled, dtype = autocast_state(device_type)
    if not enabled:
        return nullcontext
    return lambda: torch.autocast(device_type=device_type, dtype=dtype)


def _grads(outputs, inputs, grad_outputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, allow_unused=True)


class _Rule:
    def __init__(self, blocks):
        self.blocks = blocks
        self.n_layer = len(blocks)

    def params(self):
        return [p for b in self.blocks for p in b.parameters()]

    def init(self, x):
        return x, x

    def readout(self, s0, s1):
        return s1


class TwoStepRule(_Rule):
    """p[l+1] = alpha_l p[l-1] + beta_l p[l] + gamma_l f_l(p[l]); state = (p[l-1], p[l])."""

    def __init__(self, blocks, coeffs):
        super().__init__(blocks)
        assert len(coeffs) == len(blocks)
        assert all(a != 0 for a, _, _ in coeffs), "alpha = 0 is plain Euler, which is not invertible"
        self.coeffs = coeffs

    def _linear(self, l, prev, cur):
        alpha, beta, _ = self.coeffs[l]
        out = prev if alpha == 1 else alpha * prev
        if beta != 0:
            out = out + beta * cur
        return out

    def step(self, l, prev, cur):
        gamma = self.coeffs[l][2]
        return cur, self._linear(l, prev, cur) + gamma * block_update(self.blocks[l], cur)

    def _invert(self, l, cur, new, fv):
        alpha, beta, gamma = self.coeffs[l]
        prev = new - gamma * fv
        if beta != 0:
            prev = prev - beta * cur
        return prev if alpha == 1 else prev / alpha

    def inverse(self, l, cur, new):
        return self._invert(l, cur, new, block_update(self.blocks[l], cur)), cur

    def reverse(self, l, cur, new, g_cur, g_new):
        alpha, beta, gamma = self.coeffs[l]
        params = list(self.blocks[l].parameters())
        c = cur.detach().requires_grad_(True)
        fv = block_update(self.blocks[l], c)
        grads = _grads(fv, [c] + params, (gamma * g_new).to(fv.dtype))
        prev = self._invert(l, cur, new, fv.detach())
        g_prev = g_new if alpha == 1 else alpha * g_new
        g_c = g_cur + grads[0]
        if beta != 0:
            g_c = g_c + beta * g_new
        return (prev, cur), (g_prev, g_c), list(zip(params, grads[1:]))


class HamiltonianRule(_Rule):
    """Staggered position/momentum update of (2.8)-(2.9) with a = b = 1; state = (q, p)."""

    @staticmethod
    def _attn(block, p):
        return block.attn(block.ln_1(p))

    @staticmethod
    def _mlp(block, q):
        return block.mlp(block.ln_2(q))

    def step(self, l, q, p):
        b = self.blocks[l]
        q = q + self._attn(b, p)
        return q, p + self._mlp(b, q)

    def inverse(self, l, q, p):
        b = self.blocks[l]
        p = p - self._mlp(b, q)
        return q - self._attn(b, p), p

    def reverse(self, l, q, p, g_q, g_p):
        b = self.blocks[l]
        mlp_params = list(b.ln_2.parameters()) + list(b.mlp.parameters())
        attn_params = list(b.ln_1.parameters()) + list(b.attn.parameters())

        qq = q.detach().requires_grad_(True)
        m = self._mlp(b, qq)
        gm = _grads(m, [qq] + mlp_params, g_p.to(m.dtype))
        p = p - m.detach()
        g_q = g_q + gm[0]

        pp = p.detach().requires_grad_(True)
        a = self._attn(b, pp)
        ga = _grads(a, [pp] + attn_params, g_q.to(a.dtype))
        q = q - a.detach()
        g_p = g_p + ga[0]

        return (q, p), (g_q, g_p), list(zip(mlp_params, gm[1:])) + list(zip(attn_params, ga[1:]))


class _ReversibleStack(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, rule, *params):
        ctx.rule = rule
        ctx.autocast = _autocast_factory(x.device.type)
        s0, s1 = rule.init(x)
        for l in range(rule.n_layer):
            s0, s1 = rule.step(l, s0, s1)
        # for n_layer < 2 a state can still alias the input; outputs must not
        s0 = s0.clone() if s0 is x else s0
        s1 = s1.clone() if s1 is x else s1
        ctx.save_for_backward(s0, s1)
        return s0, s1

    @staticmethod
    def backward(ctx, g0, g1):
        rule = ctx.rule
        s0, s1 = ctx.saved_tensors
        params = rule.params()
        slot = {id(p): i for i, p in enumerate(params)}
        param_grads = [None] * len(params)
        with torch.enable_grad(), ctx.autocast():
            for l in reversed(range(rule.n_layer)):
                (s0, s1), (g0, g1), pg = rule.reverse(l, s0, s1, g0, g1)
                for p, g in pg:
                    if g is None:
                        continue
                    i = slot[id(p)]
                    param_grads[i] = g if param_grads[i] is None else param_grads[i] + g
        return (g0 + g1, None, *param_grads)


def run_stack(rule, x, memory_saving=True):
    """Apply every layer of `rule` to x and return the readout state."""
    if memory_saving and torch.is_grad_enabled():
        s0, s1 = _ReversibleStack.apply(x, rule, *rule.params())
    else:
        s0, s1 = rule.init(x)
        for l in range(rule.n_layer):
            s0, s1 = rule.step(l, s0, s1)
    return rule.readout(s0, s1)


VARIANTS = ('baseline', 'midpoint', 'midpoint_a', 'leapfrog', 'hamiltonian')
DEFAULT_H = {'midpoint': 0.5, 'midpoint_a': 1.0, 'leapfrog': 1.0}


def sample_midpoint_a(n_layer, seed):
    """a_l ~ {1 + U(-1,1)/2, -1 + U(-1,1)/2} with equal odds, per (3.7): zero mean, |a_l| in [0.5, 1.5]."""
    rng = random.Random(seed)
    return [rng.choice((1.0, -1.0)) + 0.5 * rng.uniform(-1.0, 1.0) for _ in range(n_layer)]


def make_rule(variant, blocks, h=None, a_seed=0):
    """Return the rule for `variant`, or None for the ordinary (non-reversible) residual baseline."""
    if variant == 'baseline':
        return None
    if variant == 'hamiltonian':
        return HamiltonianRule(blocks)
    h = DEFAULT_H[variant] if h is None else h
    n = len(blocks)
    if variant == 'midpoint':
        coeffs = [(1.0, 0.0, 2.0 * h)] * n
    elif variant == 'midpoint_a':
        coeffs = [(a, 1.0 - a, h) for a in sample_midpoint_a(n, a_seed)]
    elif variant == 'leapfrog':
        coeffs = [(-1.0, 2.0, h * h)] * n
    else:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return TwoStepRule(blocks, coeffs)
