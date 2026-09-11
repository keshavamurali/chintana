"""
Q4 -- Train the same model twice for 300 steps, once under cosine and once
under WSD, and stop both at step 200. Report both losses and state which
model you would keep.

Both schedules are built as complete 300-step schedules (i.e. each one is
what you'd actually use if you planned to train for 300 steps), and both
runs use the same seed (same init, same batch order) so the LR schedule is
the only thing that differs. We then look at what each schedule's LR --
and resulting loss -- looks like specifically at step 200, before either
run reaches step 300.

WSD's decay phase is deliberately placed in the last 20% of the *planned*
300 steps (steps 240-300), so at step 200 it is still at peak LR --
un-annealed. Cosine, tuned for the same 300-step horizon, has already
decayed most of the way by step 200. This is the tradeoff WSD makes: you can
stop it *later* with a short decay and get a good checkpoint, but stopping it
*before* its decay phase leaves it undercooked relative to a schedule that
was already decaying by that point.

Peak LR is the one Q5 found to be optimal for this width (256) -- "tune both
sides": comparing an undertuned schedule against a tuned one is not a fair
comparison.

Run:
    uv run python experiments/assignment11/q4_cosine_vs_wsd.py [--lr 1e-3]
"""

import argparse
import csv
import os

from chintana import GPTConfig
import common
from plotting import plot_series
from schedules import cosine_lr, wsd_lr

# -----------------------------------------------------------------------------
WIDTH = 256
N_LAYER = 2
N_HEAD = 8
BLOCK_SIZE = 64
BATCH_SIZE = 16
TOTAL_STEPS = 300  # the horizon both schedules are designed for
STOP_AT = 200  # both runs are physically halted here
WARMUP_ITERS = 15
MIN_LR = 1e-4
WSD_DECAY_START = int(0.8 * TOTAL_STEPS)  # last 20% of the planned 300 steps
SEED = 0
DEFAULT_LR = 1e-3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    args = parser.parse_args()
    base_lr = args.lr

    cfg = GPTConfig(n_layer=N_LAYER, n_head=N_HEAD, n_embd=WIDTH, block_size=BLOCK_SIZE,
                     dropout=0.0, bias=False)

    cosine_fn = lambda it: cosine_lr(it, base_lr, WARMUP_ITERS, TOTAL_STEPS, MIN_LR)
    wsd_fn = lambda it: wsd_lr(it, base_lr, WARMUP_ITERS, WSD_DECAY_START, TOTAL_STEPS, MIN_LR)

    print(f"width={WIDTH}, base_lr={base_lr:.2e}, stopping both runs at step {STOP_AT} "
          f"of a {TOTAL_STEPS}-step schedule (WSD decay starts at step {WSD_DECAY_START})")

    print("\ntraining under cosine...")
    _, cosine_losses = common.train(cfg, seed=SEED, steps=STOP_AT, lr_fn=cosine_fn, batch_size=BATCH_SIZE)
    print("training under WSD...")
    _, wsd_losses = common.train(cfg, seed=SEED, steps=STOP_AT, lr_fn=wsd_fn, batch_size=BATCH_SIZE)

    cosine_final = sum(cosine_losses[-10:]) / 10
    wsd_final = sum(wsd_losses[-10:]) / 10
    print(f"\nat step {STOP_AT}:")
    print(f"  cosine lr = {cosine_fn(STOP_AT - 1):.2e}, loss (mean of last 10 steps) = {cosine_final:.4f}")
    print(f"  wsd    lr = {wsd_fn(STOP_AT - 1):.2e}, loss (mean of last 10 steps) = {wsd_final:.4f}")
    winner = 'cosine' if cosine_final < wsd_final else 'wsd'
    print(f"\nlower loss at step {STOP_AT}: {winner} (margin={abs(cosine_final - wsd_final):.4f}); "
          "see README for the fuller interpretation")

    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'logs', 'q4_cosine_vs_wsd.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'cosine_lr', 'wsd_lr', 'cosine_loss', 'wsd_loss'])
        writer.writeheader()
        for it in range(STOP_AT):
            writer.writerow({
                'step': it, 'cosine_lr': cosine_fn(it), 'wsd_lr': wsd_fn(it),
                'cosine_loss': cosine_losses[it], 'wsd_loss': wsd_losses[it],
            })
    print(f"saved per-step log to {csv_path}")

    steps = list(range(STOP_AT))
    plot_series(
        steps, {'cosine': [cosine_fn(it) for it in steps], 'wsd': [wsd_fn(it) for it in steps]},
        title=f'Q4: LR schedules (both designed for {TOTAL_STEPS} steps, shown up to step {STOP_AT})',
        xlabel='step', ylabel='learning rate',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q4_lr_schedules.png'),
        markers=False,
    )
    plot_series(
        steps, {'cosine': cosine_losses, 'wsd': wsd_losses},
        title=f'Q4: training loss, cosine vs WSD (stopped at step {STOP_AT})',
        xlabel='step', ylabel='training loss',
        out_path=os.path.join(RESULTS_DIR, 'plots', 'q4_loss_curves.png'),
        markers=False,
    )


if __name__ == '__main__':
    main()
