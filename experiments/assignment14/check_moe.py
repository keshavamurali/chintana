"""Correctness checks for the MoE layer and the dense -> MoE conversion.

  python experiments/assignment14/check_moe.py

1. MoEGPT with moe.n_experts=0 is chintana's GPT: same init, identical logits.
2. MoE dispatch (gather / expert / index_add_) matches a naive per-token loop,
   with and without a capacity limit.
3. upcycle() preserves the function: the converted MoE's logits and loss match
   the dense model's, for several (n_experts, top_k, moe_every) settings.
4. The router receives gradient from the LM loss and from the aux loss.

Writes results/logs/check_moe.json.
"""

import json
import math

import torch

from chintana.config import GPTConfig
from chintana.gpt import GPT

from common import RESULTS_DIR
from model import MoEConfig, MoEGPT, upcycle
from moe import MoE

CFG = GPTConfig(block_size=32, vocab_size=65, n_layer=4, n_head=4, n_embd=64, dropout=0.0, bias=False)


def batch(seed=0, b=4, t=32):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, CFG.vocab_size, (b, t), generator=g),
            torch.randint(0, CFG.vocab_size, (b, t), generator=g))


def check_dense_equals_gpt():
    torch.manual_seed(0)
    ref = GPT(CFG)
    torch.manual_seed(0)
    ours = MoEGPT(CFG, MoEConfig(n_experts=0))
    x, y = batch()
    (l1, loss1), (l2, loss2) = ref(x, y), ours(x, y)
    same_params = all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), ours.state_dict().values()))
    return {'identical_params': same_params, 'identical_logits': torch.equal(l1, l2),
            'loss_gpt': loss1.item(), 'loss_moegpt': loss2.item()}


def naive_moe(m: MoE, x):
    logits = m.router(x).float()
    probs = logits.softmax(-1)
    gate, idx = probs.topk(m.top_k, -1)
    if m.normalize_topk:
        gate = gate / gate.sum(-1, keepdim=True)
    capacity = math.ceil(m.capacity_factor * x.shape[0] * m.top_k / m.n_experts) if m.capacity_factor > 0 else None
    used = [0] * m.n_experts
    out = torch.zeros_like(x)
    for n in range(x.shape[0]):
        for j in range(m.top_k):
            e = idx[n, j].item()
            used[e] += 1
            if capacity is not None and used[e] > capacity:
                continue
            out[n] += gate[n, j] * m.experts[e](x[n:n + 1])[0]
    return out


def check_dispatch():
    rows = []
    for e, k, cf in [(4, 1, 0.0), (8, 2, 0.0), (8, 2, 1.0), (4, 2, 0.5)]:
        torch.manual_seed(1)
        m = MoE(CFG.n_embd, e, k, 4 * CFG.n_embd, bias=True, capacity_factor=cf).eval()
        x = torch.randn(64, CFG.n_embd)
        with torch.no_grad():
            diff = (m(x) - naive_moe(m, x)).abs().max().item()
        rows.append({'n_experts': e, 'top_k': k, 'capacity_factor': cf, 'max_abs_diff': diff,
                     'dropped': m.dropped, 'ok': diff < 1e-5})
    return rows


def check_upcycle():
    torch.manual_seed(0)
    dense = MoEGPT(CFG, MoEConfig(n_experts=0)).eval()
    x, y = batch(1)
    with torch.no_grad():
        ref_logits, ref_loss = dense(x, y)
    rows = []
    for e, k, every in [(8, 2, 1), (8, 1, 1), (4, 2, 2), (16, 4, 1)]:
        moe = upcycle(dense, MoEConfig(n_experts=e, top_k=k, moe_every=every)).eval()
        with torch.no_grad():
            logits, loss = moe(x, y)
        diff = (logits - ref_logits).abs().max().item()
        rows.append({'n_experts': e, 'top_k': k, 'moe_every': every, 'moe_layers': moe.moe_layers,
                     'params_dense': sum(p.numel() for p in dense.parameters()),
                     **{f'params_{n}': v for n, v in moe.param_counts().items()},
                     'loss_dense': ref_loss.item(), 'loss_moe': loss.item(),
                     'max_abs_logit_diff': diff, 'ok': diff < 1e-5})
    # without renormalized gates the top-k weights sum to < 1, so the conversion is no longer exact
    moe = upcycle(dense, MoEConfig(n_experts=8, top_k=2, normalize_topk=False)).eval()
    with torch.no_grad():
        logits, loss = moe(x, y)
    rows.append({'n_experts': 8, 'top_k': 2, 'normalize_topk': False, 'loss_dense': ref_loss.item(),
                 'loss_moe': loss.item(), 'max_abs_logit_diff': (logits - ref_logits).abs().max().item(),
                 'ok': None})
    return rows


def check_router_grad():
    torch.manual_seed(0)
    model = MoEGPT(CFG, MoEConfig(n_experts=4, top_k=2))
    x, y = batch(2)
    out = {}
    for name in ('lm', 'aux'):
        model.zero_grad()
        _, lm = model(x, y)
        loss = lm if name == 'lm' else model.aux_losses()[0]
        loss.backward()
        out[f'router_grad_norm_from_{name}'] = sum(m.router.weight.grad.norm().item()
                                                   for m in model.moe_modules())
    out['ok'] = all(v > 0 for v in out.values())
    return out


def main():
    results = {'dense_equals_gpt': check_dense_equals_gpt(), 'dispatch_vs_naive': check_dispatch(),
               'upcycle_preserves_function': check_upcycle(), 'router_gradients': check_router_grad()}
    out = RESULTS_DIR / 'logs' / 'check_moe.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    ok = (results['dense_equals_gpt']['identical_logits']
          and all(r['ok'] for r in results['dispatch_vs_naive'])
          and all(r['ok'] for r in results['upcycle_preserves_function'] if r['ok'] is not None)
          and results['router_gradients']['ok'])
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
