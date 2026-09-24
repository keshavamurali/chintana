"""Correctness checks, run before trusting any memory or speed number.

1. Gradient equivalence. For every reversible variant, the memory-saving path
   (no activations stored; states rebuilt in backward) must give the same loss
   and the same gradient as ordinary autograd through the same dynamics. The
   fused loss head must likewise match lm_head + F.cross_entropy. Checked in
   float64 on CPU, where any mismatch is a bug, not rounding -- and, when a GPU
   is present, again at the real model size under the training autocast dtype.
2. Reconstruction error. Run the stack forward keeping every state, invert it
   from the top using only the final two states, and report how far each
   rebuilt state drifts from the stored one (relative L2), in fp32 and under
   autocast. This is the numerical price of not storing activations.

Writes results/logs/grad_check.csv and results/logs/reconstruction.csv.
"""

import csv
import sys

import torch

from chintana.configuration import load_config

import common
from reversible import VARIANTS

LOG_DIR = common.RESULTS_DIR / "logs"
TINY = dict(block_size=32, vocab_size=256, n_layer=6, n_head=4, n_embd=64)


def flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def loss_and_grad(cfg, state, x, y, amp, dtype, **model_overrides):
    model = common.build_model(cfg, **model_overrides).to(x.device, dtype)
    model.load_state_dict(state)
    with amp():
        _, loss = model(x, y)
    loss.backward()
    return loss.item(), flat_grad(model).double()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def grad_rows(cfg, device, dtype, amp, amp_label, model_overrides, batch):
    rows = []
    T = model_overrides.get('block_size', cfg.model.block_size)
    V = model_overrides.get('vocab_size', cfg.model.vocab_size)
    x, y = common.random_batch(batch, T, V, device, seed=1)
    torch.manual_seed(0)
    state = common.build_model(cfg, **model_overrides).to(device, dtype).state_dict()
    checks = [(v, 'fused', 'memory_saving vs autograd') for v in VARIANTS if v != 'baseline']
    checks.append(('baseline', 'fused', 'fused vs standard CE'))
    for variant, loss_impl, what in checks:
        cfg.rev.variant, cfg.loss.impl = variant, loss_impl
        if variant == 'baseline':
            cfg.rev.memory_saving = True
            la, ga = loss_and_grad(cfg, state, x, y, amp, dtype, **model_overrides)
            cfg.loss.impl = 'standard'
            lb, gb = loss_and_grad(cfg, state, x, y, amp, dtype, **model_overrides)
        else:
            cfg.rev.memory_saving = True
            la, ga = loss_and_grad(cfg, state, x, y, amp, dtype, **model_overrides)
            cfg.rev.memory_saving = False
            lb, gb = loss_and_grad(cfg, state, x, y, amp, dtype, **model_overrides)
        rows.append({'device': device, 'dtype': str(dtype).replace('torch.', ''), 'autocast': amp_label,
                     'n_layer': model_overrides.get('n_layer', cfg.model.n_layer),
                     'variant': variant, 'check': what, 'loss_a': la, 'loss_b': lb,
                     'loss_abs_diff': abs(la - lb), 'grad_rel_err': rel(ga, gb),
                     'grad_max_abs_err': (ga - gb).abs().max().item()})
        print(f"  {variant:12s} {what:28s} |dloss|={abs(la - lb):.2e} grad rel err={rows[-1]['grad_rel_err']:.2e}")
    cfg.rev.memory_saving, cfg.loss.impl = True, 'fused'
    return rows


def reconstruction_rows(cfg, device, amp, label, batch=4):
    rows = []
    x, _ = common.random_batch(batch, cfg.model.block_size, cfg.model.vocab_size, device, seed=2)
    for variant in VARIANTS:
        if variant == 'baseline':
            continue
        cfg.rev.variant = variant
        torch.manual_seed(0)
        model = common.build_model(cfg).to(device)
        rule = model.rule
        with torch.no_grad(), amp():
            pos = torch.arange(x.shape[1], device=device)
            h = model.transformer.wte(x) + model.transformer.wpe(pos)
            states = [rule.init(h)]
            for l in range(rule.n_layer):
                states.append(rule.step(l, *states[-1]))
            s = states[-1]
            for l in reversed(range(rule.n_layer)):
                s = rule.inverse(l, *s)
                err = max(rel(s[i].double(), states[l][i].double()) for i in range(2))
                rows.append({'setting': label, 'variant': variant, 'layer': l, 'rel_err': err})
        worst = max(r['rel_err'] for r in rows if r['variant'] == variant and r['setting'] == label)
        print(f"  {label:10s} {variant:12s} worst rebuilt-state rel err = {worst:.2e}")
    cfg.rev.variant = 'baseline'
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main(argv=None):
    cfg = load_config(common.A13Config, argv)
    fp32 = common.amp_context('cpu', torch.float32)

    print("gradient equivalence, float64, CPU, tiny model:")
    grads = grad_rows(cfg, 'cpu', torch.float64, fp32, 'none', TINY, batch=4)

    print("reconstruction error, float32, CPU, full-size model:")
    recon = reconstruction_rows(cfg, 'cpu', fp32, 'cpu_fp32')

    if torch.cuda.is_available():
        dtype = common.resolve_dtype('auto', 'cuda')
        amp = common.amp_context('cuda', dtype)
        name = str(dtype).replace('torch.', '')
        print(f"gradient equivalence, fp32 params + {name} autocast, CUDA, full-size model:")
        grads += grad_rows(cfg, 'cuda', torch.float32, amp, name, {}, batch=8)
        print("reconstruction error, CUDA:")
        recon += reconstruction_rows(cfg, 'cuda', fp32, 'cuda_fp32')
        recon += reconstruction_rows(cfg, 'cuda', amp, f'cuda_{name}')

    write(LOG_DIR / 'grad_check.csv', grads)
    write(LOG_DIR / 'reconstruction.csv', recon)
    bad = [r for r in grads if r['dtype'] == 'float64' and r['grad_rel_err'] > 1e-10]
    if bad:
        raise SystemExit(f"float64 gradient mismatch: {bad}")
    print("float64 gradients match for every variant")


if __name__ == '__main__':
    main(sys.argv[1:])
