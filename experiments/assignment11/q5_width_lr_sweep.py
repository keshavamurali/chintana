"""
Q5 -- Sweep the learning rate at widths 256, 512, and 1024; plot loss against
learning rate; mark the three minima; extrapolate to width 4096.

Uses the real chintana.GPT (via experiments/assignment11/common.py), trained
from scratch on shakespeare_char for a short, fixed step budget per (width,
lr) pair, at a constant (non-decayed) learning rate -- isolating the effect of
LR magnitude from any schedule. Depth and head_dim are held fixed across
widths (standard practice for a width sweep); only n_embd/n_head change.

chintana/gpt.py uses GPT-2-style fixed-std init (std=0.02 regardless of
width), i.e. standard parameterization, NOT muP -- so the optimal LR is
expected to shift with width, which is exactly what this sweep measures.

Run:
    uv run python experiments/assignment11/q5_width_lr_sweep.py
"""

import csv
import os

import numpy as np

from chintana import GPTConfig
import common
from plotting import plot_series

# -----------------------------------------------------------------------------
WIDTHS = [256, 512, 1024]
HEAD_DIM = 32  # fixed head_dim -> n_head scales with width
N_LAYER = 2
BLOCK_SIZE = 64
BATCH_SIZE = 16
STEPS = 150
AVG_LAST_N = 10  # final loss = mean of the last N steps' losses (denoise)
LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SEED = 0
EXTRAPOLATE_TO = 4096

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
# -----------------------------------------------------------------------------


def final_loss(losses):
    tail = losses[-AVG_LAST_N:]
    if any(l != l or l == float('inf') for l in tail):  # nan or inf -> diverged
        return float('nan')
    return sum(tail) / len(tail)


def run_sweep():
    rows = []
    for width in WIDTHS:
        cfg = GPTConfig(n_layer=N_LAYER, n_head=width // HEAD_DIM, n_embd=width,
                         block_size=BLOCK_SIZE, dropout=0.0, bias=False)
        for lr in LR_GRID:
            _, losses = common.train(cfg, seed=SEED, steps=STEPS, lr_fn=lambda it, lr=lr: lr,
                                      batch_size=BATCH_SIZE)
            loss = final_loss(losses)
            print(f"width={width:5d} lr={lr:.0e} -> final loss (last {AVG_LAST_N} steps) = {loss:.4f}")
            rows.append({'width': width, 'lr': lr, 'final_loss': loss})
    return rows


def finalize(rows):
    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'logs', 'q5_width_lr_sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['width', 'lr', 'final_loss'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved sweep table to {csv_path}")

    series = {}
    best_lr_per_width = {}
    for width in WIDTHS:
        width_rows = [r for r in rows if r['width'] == width]
        width_rows.sort(key=lambda r: r['lr'])
        series[f'width={width}'] = [r['final_loss'] for r in width_rows]
        finite = [r for r in width_rows if r['final_loss'] == r['final_loss']]  # drop nan
        best = min(finite, key=lambda r: r['final_loss']) if finite else None
        best_lr_per_width[width] = best['lr'] if best else None
        if best is not None:
            edge = " (** at grid edge -- true minimum may lie outside the sweep **)" \
                if best['lr'] in (LR_GRID[0], LR_GRID[-1]) else ""
            print(f"width={width}: best lr in grid = {best['lr']:.0e}, loss={best['final_loss']:.4f}{edge}")
        else:
            print(f"width={width}: every LR in the grid diverged")

    mark_points = {
        f'width={width}': (best_lr_per_width[width], min(r['final_loss'] for r in rows
                                                           if r['width'] == width and r['lr'] == best_lr_per_width[width]))
        for width in WIDTHS if best_lr_per_width[width] is not None
    }
    plot_series(
        LR_GRID, series,
        title='Q5: loss vs learning rate by width (stars = per-width minimum)',
        xlabel='learning rate', ylabel=f'loss (mean of last {AVG_LAST_N} steps)',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q5_loss_vs_lr.png'),
        logx=True, mark_points=mark_points,
    )

    # log-log fit of optimal LR vs width, extrapolated to EXTRAPOLATE_TO
    fit_widths = [w for w in WIDTHS if best_lr_per_width[w] is not None]
    if len(fit_widths) >= 2:
        log_w = np.log10(fit_widths)
        log_lr = np.log10([best_lr_per_width[w] for w in fit_widths])
        slope, intercept = np.polyfit(log_w, log_lr, 1)
        predicted_lr = 10 ** (slope * np.log10(EXTRAPOLATE_TO) + intercept)
        print(f"\nlog-log fit: log10(lr*) = {slope:.3f} * log10(width) + {intercept:.3f}")
        print(f"predicted optimal LR at width={EXTRAPOLATE_TO}: {predicted_lr:.2e}")
        print("(3 points, non-muP init, short-horizon proxy for the true optimum, "
              f"and width={EXTRAPOLATE_TO} is {EXTRAPOLATE_TO / max(fit_widths):.0f}x beyond the largest "
              "width measured -- treat this as a low/moderate-confidence extrapolation.)")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(fit_widths, [best_lr_per_width[w] for w in fit_widths], 'o', markersize=10,
                color='tab:blue', label='measured optimum (grid argmin)')
        fit_x = fit_widths + [EXTRAPOLATE_TO]
        fit_y = [10 ** (slope * np.log10(w) + intercept) for w in fit_x]
        ax.plot(fit_x, fit_y, '--', color='gray', label='log-log fit')
        ax.plot(EXTRAPOLATE_TO, predicted_lr, '*', markersize=18, color='black',
                label=f'extrapolated @ width={EXTRAPOLATE_TO}: {predicted_lr:.1e}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('width (n_embd)')
        ax.set_ylabel('optimal learning rate')
        ax.set_title('Q5: optimal LR vs width, extrapolated')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, 'plots', 'q5_lr_vs_width_extrapolation.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved plot to {out_path}")


def main():
    finalize(run_sweep())


if __name__ == '__main__':
    main()
