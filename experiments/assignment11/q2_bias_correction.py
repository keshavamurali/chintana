"""
Q2 -- Disable bias correction and plot the first twenty steps both ways;
report the number of steps after which the difference stops mattering.

Continues Q1's synthetic setup (one weight, hand-rolled Adam), now run twice
-- once with bias correction (the real Adam algorithm) and once without (m,
v used raw) -- over the same 20 fixed gradients.

The exact multiplicative gap between the two update rules, ignoring eps, is
the closed-form bias-correction factor:

    factor(t) = sqrt(1 - beta2^t) / (1 - beta1^t)

which -> 1 as t -> infinity. Because this is closed-form we don't need to run
training to find where it settles -- we compute it directly, both over the
requested first-20-steps window and further out to find where it actually
converges (beta2 close to 1 gives it a much longer timescale than beta1).

Run:
    uv run python experiments/assignment11/q2_bias_correction.py
"""

import csv
import os
import random

from manual_adam import AdamState
from plotting import plot_series

# -----------------------------------------------------------------------------
W0 = 1.0
N_STEPS = 20
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
SEED = 0
GRAD_STD = 0.5

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
TOLERANCES = [0.10, 0.05, 0.01]  # report the step where |factor - 1| first drops below each
EXTENDED_HORIZON = 6000  # cheap (closed-form / pure python), so look far past 20 steps
# -----------------------------------------------------------------------------


def bias_correction_factor(t, beta1=BETA1, beta2=BETA2):
    return (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)


def first_step_within_tolerance(tol, horizon):
    for t in range(1, horizon + 1):
        if abs(bias_correction_factor(t) - 1.0) < tol:
            return t
    return None


def main():
    rng = random.Random(SEED)
    grads = [rng.gauss(0, GRAD_STD) for _ in range(N_STEPS)]

    corrected = AdamState(w=W0, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, bias_correction=True).run(grads)
    uncorrected = AdamState(w=W0, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, bias_correction=False).run(grads)

    print(f"{'t':>2} {'grad':>7} | {'w (corrected)':>14} {'w (uncorrected)':>16} | "
          f"{'update ratio (unc/corr)':>24} {'closed-form factor':>19}")
    csv_rows = []
    for t, (c, u) in enumerate(zip(corrected, uncorrected), start=1):
        factor = bias_correction_factor(t)
        update_ratio = u.update / c.update if c.update != 0 else float('nan')
        print(f"{t:>2} {c.grad:>7.3f} | {c.w:>14.8f} {u.w:>16.8f} | {update_ratio:>24.4f} {factor:>19.4f}")
        csv_rows.append({
            't': t, 'grad': c.grad,
            'w_corrected': c.w, 'w_uncorrected': u.w,
            'update_corrected': c.update, 'update_uncorrected': u.update,
            'update_ratio_uncorrected_over_corrected': update_ratio,
            'bias_correction_factor': factor,
        })

    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'logs', 'q2_bias_correction.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nsaved table to {csv_path}")

    print(f"\nWithin the requested first {N_STEPS} steps: "
          f"factor(t={N_STEPS}) = {bias_correction_factor(N_STEPS):.4f} "
          f"(still {abs(1 - bias_correction_factor(N_STEPS)) * 100:.0f}% away from 1.0 -- "
          "bias correction has NOT stopped mattering yet at step 20, because beta2=0.999 "
          "has a ~1/(1-beta2) = 1000-step timescale.")

    print("\nExtending the (closed-form, free to compute) factor curve further out:")
    for tol in TOLERANCES:
        step = first_step_within_tolerance(tol, EXTENDED_HORIZON)
        msg = f"step {step}" if step is not None else f"> {EXTENDED_HORIZON} (not reached)"
        print(f"  |factor - 1| < {tol:.0%}: {msg}")

    # beta1's own correction term 1/(1-beta1^t) converges much faster than beta2's;
    # report it separately since it's the one relevant within the 20-step window.
    beta1_only_step = None
    for t in range(1, EXTENDED_HORIZON + 1):
        if (1 / (1 - BETA1 ** t) - 1) < 0.01:
            beta1_only_step = t
            break
    print(f"  (for reference, beta1's own correction term alone is within 1% by step {beta1_only_step})")

    steps = list(range(1, N_STEPS + 1))
    plot_series(
        steps,
        {'bias-corrected (real Adam)': [r['w_corrected'] for r in csv_rows],
         'uncorrected (m, v used raw)': [r['w_uncorrected'] for r in csv_rows]},
        title='Q2: weight trajectory, first 20 steps',
        xlabel='step', ylabel='w',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q2_weight_trajectory_20steps.png'),
    )

    extended_steps = list(range(1, EXTENDED_HORIZON + 1, max(1, EXTENDED_HORIZON // 500)))
    plot_series(
        extended_steps,
        {'bias-correction factor = sqrt(1-b2^t)/(1-b1^t)': [bias_correction_factor(t) for t in extended_steps]},
        title=f'Q2: bias-correction factor out to {EXTENDED_HORIZON} steps (converges to 1.0)',
        xlabel='step', ylabel='factor',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q2_factor_extended.png'),
        markers=False,
    )


if __name__ == '__main__':
    main()
