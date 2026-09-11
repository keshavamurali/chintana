"""
Q3 -- Log the update-to-weight ratio for every layer, and identify the step
at which warmup stops changing it.

Trains the real chintana.GPT (small config, see below) on shakespeare_char
with cosine-with-warmup (mirroring train.py's own schedule shape, see
schedules.cosine_lr), using the real Adam-based configure_optimizers. After
every optimizer.step(), for a representative set of parameter tensors we log

    ratio = ||delta_w|| / ||w||     where delta_w = w_after_step - w_before_step

(same "how big is the step relative to the weight" diagnostic as Karpathy's
update/data ratio plot). The peak LR here is the one Q5 found to be optimal
for this same width (256) -- "tune both sides" applies here too: there is no
point studying warmup dynamics at a badly chosen LR.

Run:
    uv run python experiments/assignment11/q3_update_ratio.py [--lr 1e-3]
"""

import argparse
import csv
import os

import torch

from chintana import GPTConfig
import common
from plotting import plot_series
from schedules import cosine_lr

# -----------------------------------------------------------------------------
WIDTH = 256
N_LAYER = 2
N_HEAD = 8  # head_dim=32, matching Q5's fixed head_dim
BLOCK_SIZE = 64
BATCH_SIZE = 16
STEPS = 400
WARMUP_ITERS = 30
LR_DECAY_ITERS = STEPS
MIN_LR = 1e-4
SEED = 0
DEFAULT_LR = 1e-3  # overridden by --lr, e.g. with Q5's optimum for width=256

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLATEAU_WINDOW = 10  # steps over which we average the ratio before checking it has flattened
PLATEAU_SLOPE_TOL = 0.01  # "flat" := relative change in the windowed average < 1% per step
# -----------------------------------------------------------------------------


def tracked_param_names(n_layer):
    names = ['transformer.wte.weight']  # tied with lm_head.weight
    names.append(f'transformer.h.0.attn.c_attn.weight')
    names.append(f'transformer.h.0.mlp.c_fc.weight')
    names.append(f'transformer.h.{n_layer - 1}.attn.c_attn.weight')
    names.append(f'transformer.h.{n_layer - 1}.mlp.c_fc.weight')
    names.append('transformer.ln_f.weight')
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    args = parser.parse_args()
    base_lr = args.lr

    cfg = GPTConfig(n_layer=N_LAYER, n_head=N_HEAD, n_embd=WIDTH, block_size=BLOCK_SIZE,
                     dropout=0.0, bias=False)
    tracked = tracked_param_names(N_LAYER)
    ratios = {name: [] for name in tracked}
    lrs = []
    prev = {}

    def on_step(it, loss, lr, model, optimizer):
        lrs.append(lr)
        params = dict(model.named_parameters())
        for name in tracked:
            p = params[name]
            if name in prev:
                update_norm = (p.detach() - prev[name]).norm().item()
                weight_norm = prev[name].norm().item()
                ratios[name].append(update_norm / weight_norm)
            else:
                ratios[name].append(float('nan'))  # no ratio available before the first step
            prev[name] = p.detach().clone()

    lr_fn = lambda it: cosine_lr(it, base_lr, WARMUP_ITERS, LR_DECAY_ITERS, MIN_LR)
    print(f"training width={WIDTH} model for {STEPS} steps, base_lr={base_lr:.2e}, warmup_iters={WARMUP_ITERS}")
    model, losses = common.train(cfg, seed=SEED, steps=STEPS, lr_fn=lr_fn, batch_size=BATCH_SIZE, on_step=on_step)
    print(f"final training loss (last 10 steps mean): {sum(losses[-10:]) / 10:.4f}")

    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'logs', 'q3_update_ratio.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['step', 'lr', 'loss'] + tracked
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in range(STEPS):
            row = {'step': it, 'lr': lrs[it], 'loss': losses[it]}
            row.update({name: ratios[name][it] for name in tracked})
            writer.writerow(row)
    print(f"saved per-step ratios to {csv_path}")

    # find, per tracked tensor, the first step after which the ratio's windowed
    # average stops changing by more than PLATEAU_SLOPE_TOL relative to itself
    plateau_steps = {}
    for name in tracked:
        series = ratios[name]
        plateau_steps[name] = None
        for t in range(WARMUP_ITERS, STEPS - PLATEAU_WINDOW):
            window_now = sum(series[t:t + PLATEAU_WINDOW]) / PLATEAU_WINDOW
            window_next = sum(series[t + PLATEAU_WINDOW:t + 2 * PLATEAU_WINDOW]) / PLATEAU_WINDOW \
                if t + 2 * PLATEAU_WINDOW <= STEPS else window_now
            if window_now > 0 and abs(window_next - window_now) / window_now < PLATEAU_SLOPE_TOL:
                plateau_steps[name] = t
                break

    print(f"\nwarmup_iters (config) = {WARMUP_ITERS}")
    print("empirical step at which each tracked tensor's update/weight ratio plateaus "
          f"(windowed average over {PLATEAU_WINDOW} steps changes by <{PLATEAU_SLOPE_TOL:.0%}):")
    for name in tracked:
        step = plateau_steps[name]
        print(f"  {name:35s}: {'step ' + str(step) if step is not None else 'no plateau found in this run'}")

    plot_series(
        list(range(STEPS)), {'lr': lrs},
        title='Q3: learning-rate schedule (cosine with warmup)',
        xlabel='step', ylabel='learning rate',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q3_lr_schedule.png'),
        markers=False,
    )
    plot_series(
        list(range(1, STEPS)),  # first entry has no ratio yet
        {name: ratios[name][1:] for name in tracked},
        title='Q3: update/weight ratio per layer (log scale)',
        xlabel='step', ylabel='||delta_w|| / ||w||',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q3_update_weight_ratio.png'),
        logy=True, markers=False,
    )


if __name__ == '__main__':
    main()
