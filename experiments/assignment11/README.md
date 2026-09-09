# Assignment 11 -- optimizer internals

This directory holds the assignment work on top of Chintana. It is entirely
additive: nothing here is imported by `train.py`, `sample.py`, `bench.py`, or
`chintana/`, so the base model and training pipeline behave exactly as before
except for the one deliberate change described below.

## Base-model change: AdamW -> Adam

`chintana/gpt.py`'s `GPT.configure_optimizers` now builds a `torch.optim.Adam`
instead of a `torch.optim.AdamW`. This was requested specifically so the "real"
optimizer used by actual training runs is the same algorithm (plain Adam) as
the one reproduced by hand in Q1 below -- comparing hand-rolled Adam against
AdamW would be comparing two different update rules.

This is a real behavior change, not just a rename: `torch.optim.Adam`'s
`weight_decay` is coupled L2 regularization (added to the gradient before the
moment updates), whereas `AdamW`'s `weight_decay` is decoupled (subtracted
from the weight directly, independent of the adaptive learning rate). Any
config with `optim.weight_decay != 0` (e.g. the default `0.1` inherited by
`config/train_shakespeare_char.yaml`) will now train slightly differently than
it did before this change. Everything else (param grouping by tensor
dimension, fused-kernel selection, betas) is unchanged.

Verified after the swap: `train.py` still runs end-to-end
(`uv run python train.py config/train_shakespeare_char.yaml system.device=cpu system.compile=false`),
printing `using fused Adam: False` where it used to print
`using fused AdamW: False`, with training proceeding normally.

## Q1 -- reproduce Adam by hand

Files:
- `manual_adam.py` -- a dependency-free, pure-Python reimplementation of the
  Adam update (`AdamState.step`), operating on plain floats.
- `q1_adam_by_hand.py` -- runs one synthetic weight through 5 fixed gradients,
  once through `manual_adam.AdamState`, once through `torch.optim.Adam`
  (float64, gradients set by hand via `w.grad = ...` -- no autograd involved),
  and checks every intermediate quantity against each other.
- `plotting.py` -- a small shared `plot_series` helper used by this and future
  questions.

Run it:
```bash
uv run python experiments/assignment11/q1_adam_by_hand.py
```

Setup: `w0 = 1.0`, `grads = [0.5, -0.3, 0.8, -0.1, 0.4]`, `lr = 1e-3`,
`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8` (Adam's textbook defaults).

Result: for every one of the 5 steps, the manually computed `m`, `v`, and
resulting weight `w` matched `torch.optim.Adam`'s internal state
(`opt.state[w]['exp_avg']`, `['exp_avg_sq']`) and output exactly, to
float64 precision (`max |w_manual - w_torch| = 0.00e+00` across all 5 steps --
see `results/logs/q1_adam_by_hand.csv` for the full per-step table).

| step | grad | m | v | m_hat | v_hat | update | w |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.05000000 | 0.00025000 | 0.50000000 | 0.25000000 | 0.00100000 | 0.99900000 |
| 2 | -0.30 | 0.01500000 | 0.00033975 | 0.07894737 | 0.16995998 | 0.00019150 | 0.99880850 |
| 3 | 0.80 | 0.09350000 | 0.00097941 | 0.34501845 | 0.32679677 | 0.00060354 | 0.99820497 |
| 4 | -0.10 | 0.07415000 | 0.00098843 | 0.21561500 | 0.24747868 | 0.00043342 | 0.99777154 |
| 5 | 0.40 | 0.10673500 | 0.00114744 | 0.26064077 | 0.22994792 | 0.00054354 | 0.99722801 |

Plots:
- `results/plots/q1_weight_trajectory.png` -- manual vs torch weight trajectory
  (the two lines are exactly on top of each other).
- `results/plots/q1_bias_corrected_moments.png` -- `m_hat`/`v_hat` per step.
- `results/plots/q1_update_size.png` -- the resulting step size per iteration.

## Not yet implemented

Q2 (bias-correction ablation), Q3 (update/weight ratio through warmup), Q4
(cosine vs WSD), and Q5 (width LR sweep) are not part of this change -- only
Q1, the AdamW->Adam swap, and the shared plotting support were requested so
far. `manual_adam.AdamState` already supports `bias_correction=False` for Q2.
