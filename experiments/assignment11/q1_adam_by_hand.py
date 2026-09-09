"""
Q1 -- Reproduce Adam by hand.

Take one (synthetic) weight and five gradients, compute m, v, m_hat, v_hat and
the resulting parameter update ourselves (manual_adam.AdamState, plain Python
floats, no torch), then check every quantity against torch.optim.Adam applied
to the same weight with the same five gradients fed in by hand (no autograd
involved -- we just set .grad directly each step).

Run:
    uv run python experiments/assignment11/q1_adam_by_hand.py
"""

import csv
import os

import torch

from manual_adam import AdamState
from plotting import plot_series

# -----------------------------------------------------------------------------
W0 = 1.0
GRADS = [0.5, -0.3, 0.8, -0.1, 0.4]
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
TOL = 1e-9  # both sides run in float64, so agreement should be near machine precision
# -----------------------------------------------------------------------------


def run_manual():
    state = AdamState(w=W0, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS)
    return state.run(GRADS)


def run_torch():
    w = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=0.0)
    rows = []
    for g in GRADS:
        w.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        state = opt.state[w]
        rows.append({
            'm': state['exp_avg'].item(),
            'v': state['exp_avg_sq'].item(),
            'w': w.item(),
        })
    return rows


def main():
    manual = run_manual()
    torch_rows = run_torch()
    assert len(manual) == len(torch_rows) == len(GRADS)

    print(f"{'t':>2} {'grad':>6} | {'m (manual)':>12} {'m (torch)':>12} | "
          f"{'v (manual)':>12} {'v (torch)':>12} | {'w (manual)':>14} {'w (torch)':>14} {'|diff|':>10}")
    csv_rows = []
    max_abs_diff = 0.0
    for step, (m, t) in enumerate(zip(manual, torch_rows), start=1):
        diff = abs(m.w - t['w'])
        max_abs_diff = max(max_abs_diff, diff)
        print(f"{step:>2} {m.grad:>6.2f} | {m.m:>12.8f} {t['m']:>12.8f} | "
              f"{m.v:>12.8f} {t['v']:>12.8f} | {m.w:>14.10f} {t['w']:>14.10f} {diff:>10.2e}")
        csv_rows.append({
            't': step, 'grad': m.grad,
            'm_hat': m.m_hat, 'v_hat': m.v_hat, 'update': m.update,
            'm_manual': m.m, 'm_torch': t['m'],
            'v_manual': m.v, 'v_torch': t['v'],
            'w_manual': m.w, 'w_torch': t['w'],
            'abs_diff_w': diff,
        })
        assert abs(m.m - t['m']) < TOL, f"step {step}: m mismatch"
        assert abs(m.v - t['v']) < TOL, f"step {step}: v mismatch"
        assert diff < TOL, f"step {step}: w mismatch"

    print(f"\nmax |manual_w - torch_w| over {len(GRADS)} steps: {max_abs_diff:.2e} (tolerance {TOL:.0e})")
    print("PASS: hand-rolled Adam matches torch.optim.Adam to within tolerance.")

    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'logs', 'q1_adam_by_hand.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"saved table to {csv_path}")

    steps = [0] + [r['t'] for r in csv_rows]
    plot_series(
        steps,
        {
            'manual (hand-rolled)': [W0] + [r['w_manual'] for r in csv_rows],
            'torch.optim.Adam': [W0] + [r['w_torch'] for r in csv_rows],
        },
        title='Q1: weight trajectory, manual Adam vs torch.optim.Adam',
        xlabel='step', ylabel='w',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q1_weight_trajectory.png'),
    )
    # abs_diff_w is exactly 0.0 at every step here (bit-for-bit float64 agreement),
    # so rather than a misleading "zero on a log axis" plot, show the quantities
    # that actually evolve step to step: the bias-corrected moments and the step size.
    plot_series(
        [r['t'] for r in csv_rows],
        {'m_hat': [r['m_hat'] for r in csv_rows], 'v_hat': [r['v_hat'] for r in csv_rows]},
        title='Q1: bias-corrected first/second moment estimates',
        xlabel='step', ylabel='value',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q1_bias_corrected_moments.png'),
    )
    plot_series(
        [r['t'] for r in csv_rows],
        {'update = lr * m_hat / (sqrt(v_hat) + eps)': [r['update'] for r in csv_rows]},
        title='Q1: resulting Adam step size per iteration',
        xlabel='step', ylabel='update',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q1_update_size.png'),
    )


if __name__ == '__main__':
    main()
