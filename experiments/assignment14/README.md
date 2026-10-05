# Assignment 14: from a dense ("Linear") model to a Mixture of Experts

This directory holds the assignment work on top of Chintana, a modular refactor of
[nanoGPT](https://github.com/karpathy/nanoGPT) (see the repo root `README.md`). Nothing under
`chintana/` was changed. `MoEGPT` subclasses `chintana.GPT` and only swaps the `mlp` module
inside chosen `Block`s; embeddings, attention, LayerNorms, weight tying, init, `forward()` and
`generate()` are the base model's.

**The assignment:** train a Linear model and convert it into a Mixture of Experts (MoE).
Provide both models and compare them. The MoE size is left open, but it must be configurable.

**What "Linear model" means here.** A transformer block's feed-forward network (FFN) in
chintana is a dense stack of `nn.Linear` layers: `Linear(C, 4C) → GELU → Linear(4C, C)`
(`chintana/mlp.py`). That FFN is what an MoE replaces, so the "Linear" model is the ordinary
dense GPT, and the MoE is the same GPT with each FFN turned into several FFN *experts* plus a
router. Everything else stays identical, so any difference comes from the FFN alone.

**How the conversion works.** The dense model is converted by *sparse upcycling*
([Komatsuzaki et al., 2022](https://arxiv.org/abs/2212.05055)). Every expert starts as a copy of
the trained dense FFN, and a new router is added. The converted model computes **exactly** the
same function as the dense model at the moment of conversion (section 2.2), and training
then continues as an MoE.

**Results are in section 4.**

---

## 1. Quick start

Run everything from the repo root (the scripts locate their data and configs relative to
this directory).

```bash
uv run python experiments/assignment14/run_all.py          # all stages, ~2 h on a laptop CPU
uv run python experiments/assignment14/check_moe.py        # correctness checks only (seconds)

# or step by step:
uv run python experiments/assignment14/train.py experiments/assignment14/config/dense.yaml
uv run python experiments/assignment14/train.py experiments/assignment14/config/moe_upcycle.yaml
uv run python experiments/assignment14/train.py experiments/assignment14/config/dense_continue.yaml
uv run python experiments/assignment14/report.py
```

`run_all.py --stages ...` picks stages (`check dense dense_continue upcycle moe_scratch
dense_wide sweep report`). `--phase1-steps` / `--phase2-steps` change the budgets. Any extra
`key=value` argument is passed to every training run of the selected stages.

## 2. Implementation

| file | what it does |
|---|---|
| [`moe.py`](moe.py) | `Expert` (chintana's MLP with a configurable hidden width, same `c_fc`/`c_proj` names) and `MoE`: top-k router, dispatch, capacity limit, load-balancing and z-losses |
| [`model.py`](model.py) | `MoEConfig`; `MoEGPT(chintana.GPT)`, which swaps `block.mlp` for an `MoE` in the chosen blocks; `upcycle()`, the dense → MoE conversion |
| [`common.py`](common.py) | the `A14Config` schema (loaded with `chintana.load_config`, like the root `train.py`), data sampling, checkpoints, AdamW, LR schedule, `train_step`, `evaluate` (with routing stats) |
| [`train.py`](train.py) | one run (`init.mode` = `scratch`, `resume` or `upcycle`); writes `results/runs/<name>/{log.jsonl,summary.json,ckpt.pt}` |
| [`check_moe.py`](check_moe.py) | correctness checks (section 2.4) |
| [`run_all.py`](run_all.py) | the full pipeline |
| [`report.py`](report.py) | writes `results/summary.md` and `results/plots/*.png` |
| [`config/`](config) | YAML for every run |

### 2.1 The MoE layer

For the `N = B·T` tokens entering a block's FFN (hidden size `C`), with `E` experts and top-`k` routing:

```
probs  = softmax(x @ W_router)            (N, E)   router: Linear(C, E, bias=False), fp32 softmax
gate, idx = topk(probs, k)                (N, k)
gate   = gate / gate.sum(-1)              (renormalize, moe.normalize_topk=true)
y[n]   = Σ_j gate[n, j] · Expert_{idx[n, j]}(x[n])
```

Dispatch is a gather and scatter: each expert gathers the rows routed to it, runs on just
those rows, and adds its gated output back with `index_add_`. An expert's compute is therefore
proportional to the tokens it receives. That is the point of an MoE: total parameters grow with
`E`, while compute per token grows only with `k`.

The training loss is

```
loss = CE + aux_loss_coef · L_balance + z_loss_coef · L_z
L_balance = E · Σ_i f_i · P_i        (Switch Transformer; f_i = share of assignments to expert i,
                                       P_i = mean router prob of expert i; = 1.0 when perfectly uniform)
L_z       = mean(logsumexp(router logits)²)   (ST-MoE router z-loss)
```

Both are averaged over the MoE layers. Validation loss is always the plain cross-entropy, so
dense and MoE numbers are directly comparable.

### 2.2 The conversion: sparse upcycling

`model.upcycle(dense, moe_cfg)` builds an `MoEGPT` and copies every dense weight over by name:
embeddings, attention, LayerNorms, and the FFNs of blocks that stay dense. In each MoE block it
copies the dense FFN's `c_fc` / `c_proj` into **every** expert. Only the routers are new
(`N(0, 0.02)`, the GPT-2 init).

With renormalized top-k gates (`Σ_j gate_j = 1`) and identical experts,
`y = Σ_j gate_j · FFN(x) = FFN(x)`. Whatever the router picks, the converted model is
**function-preserving** (identical up to float rounding). It starts phase 2 at the dense model's
loss rather than from scratch. The experts then diverge because the router sends them different
tokens, so each one receives different gradients. `moe.upcycle_noise` optionally perturbs the
copies to break the symmetry faster (off by default).

The conversion requires `expert_hidden = 4C`, the dense FFN's width. Without renormalization
(`moe.normalize_topk=false`) the top-2 gates sum to less than 1, and the conversion is no longer
exact (section 2.4 measures this).

### 2.3 Everything is configurable

All MoE knobs live in `MoEConfig` (`model.py`) and are set from YAML or the command line:

| key | default | meaning |
|---|---|---|
| `moe.n_experts` | 8 | experts per MoE layer; **0 = dense model** (a plain chintana GPT) |
| `moe.top_k` | 2 | experts per token |
| `moe.expert_hidden` | 0 (→ 4C) | hidden width of each expert (must be 4C to upcycle) |
| `moe.moe_every` | 1 | 1 = every block's FFN is an MoE; 2 = blocks 1, 3, … (GShard-style interleaving) |
| `moe.normalize_topk` | true | renormalize the chosen gates to sum to 1 |
| `moe.capacity_factor` | 0.0 | 0 = no token dropping; else each expert takes at most `ceil(cf·N·k/E)` assignments |
| `moe.router_jitter` | 0.0 | multiplicative noise on router inputs while training |
| `moe.aux_loss_coef` | 0.01 | load-balancing loss weight |
| `moe.z_loss_coef` | 0.001 | router z-loss weight |
| `moe.upcycle_noise` | 0.0 | relative noise added to each expert copy when upcycling |

The model (`model.n_layer`, `model.n_head`, `model.n_embd`, `model.block_size`,
`model.dropout`, `model.bias`), the data (`data.dir`, `data.batch_size`) and the optimizer
(`train.*`) are configurable the same way. For example:

```bash
# a 16-expert, top-1 MoE in every other block, upcycled from the dense checkpoint
uv run python experiments/assignment14/train.py experiments/assignment14/config/moe_upcycle.yaml \
    moe.n_experts=16 moe.top_k=1 moe.moe_every=2 run.name=moe_e16_k1_every2

# a bigger dense model, then convert it
uv run python experiments/assignment14/run_all.py --stages dense upcycle dense_continue report \
    model.n_layer=6 model.n_embd=192 model.n_head=6
```

### 2.4 Correctness checks (`check_moe.py`)

Results are in `results/logs/check_moe.json`; all checks pass.

| check | result |
|---|---|
| `MoEGPT` with `moe.n_experts=0` vs `chintana.GPT`, same seed | identical parameters and **bit-identical logits** (loss 4.193335 for both) |
| `MoE` dispatch vs a naive per-token loop, `(E, k, capacity_factor)` = (4,1,0), (8,2,0), (8,2,1.0), (4,2,0.5) | max abs diff ≤ 4.2e-7; with capacity limits the same 15 / 64 assignments are dropped |
| `upcycle()` vs the dense model, `(E, k, moe_every)` = (8,2,1), (8,1,1), (4,2,2), (16,4,1) | same loss to 6 decimals (4.211341), max abs logit diff ≤ 3.0e-7 |
| `upcycle()` with `normalize_topk=false` (E=8, k=2) | **not** function-preserving: logits differ by up to 0.28 |
| router gradient | non-zero from both the LM loss and the load-balancing loss |

The same equivalence shows up in the real runs: every upcycled model's step-0 val loss
(1.6603) equals the dense checkpoint's final val loss exactly (section 4.1).

## 3. Setup

All numbers below come from one complete `run_all.py` run on the laptop CPU (Intel
i7-9750H, fp32, torch 2.2.2). `results/summary.md` holds the generated tables and samples,
`results/runs/<name>/` the per-run logs, summaries and checkpoints, and
`results/logs/run_all.log` the full console output.

| | |
|---|---|
| data | Tiny Shakespeare, character level (`data/shakespeare_char`: 1.0M train / 0.11M val characters, 65-symbol vocabulary) |
| dense model | chintana GPT, `n_layer=4, n_head=4, n_embd=128, block_size=128, bias=False, dropout=0`: **0.812M parameters** |
| MoE model | the same GPT with **every** block's FFN replaced by **8 experts, top-2** routing: **4.486M total / 1.341M active parameters per token** |
| batch | 32 × 128 = 4,096 tokens per step, no gradient accumulation |
| optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on ≥2-D tensors, grad clip 1.0 |
| phase 1 (dense) | 2,000 steps (8.2M tokens), lr 1e-3, 100 warmup steps, cosine to 1e-4 |
| phase 2 (after conversion) | 1,500 steps (6.1M tokens), lr 5e-4, 50 warmup steps, cosine to 5e-5, fresh optimizer state, same data order for every phase-2 run |
| router losses | load-balance 0.01, z-loss 0.001, no jitter, no capacity limit (no token dropped) |
| eval | the same 20 × 64 val windows (164k characters) for every run, every 250 steps |

**The runs.** The main comparison is `moe_upcycle` vs `dense_continue`. Both start from the
same dense checkpoint and get the same steps, schedule and batches. The only difference is
whether the FFNs were converted. The two from-scratch runs put that result in context:

| run | what it is |
|---|---|
| `dense` | phase 1: the dense ("Linear") model, 2,000 steps from scratch |
| `dense_continue` | phase 2 control: the `dense` checkpoint, trained 1,500 more steps as a dense model |
| `moe_upcycle` | phase 2: the `dense` checkpoint **converted to an MoE**, trained 1,500 steps |
| `moe_scratch` | the same MoE architecture from random init, 3,500 steps (phase 1 + phase 2) |
| `dense_wide` | a dense model with the MoE's *active* parameters: one 8C-wide FFN per block (1.337M parameters), 3,500 steps |
| `sweep_*` | phase 2 upcycling with other `(E, k, moe_every)` settings |

## 4. Results

### 4.1 Dense vs MoE

| run | model | steps | total params | active params | start val | final train | **final val** | tokens/s | wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| `dense` | dense | 2000 | 0.812M | 0.812M | 4.1809 | 1.4772 | **1.6603** | 15,782 | 8.9 |
| `dense_continue` | dense | +1500 | 0.812M | 0.812M | 1.6603 | 1.3680 | **1.5775** | 20,112 | 5.3 |
| `moe_upcycle` | E=8 k=2 | +1500 | 4.486M | 1.341M | 1.6603 | 1.3106 | **1.5367** | 9,668 | 11.1 |
| `moe_scratch` | E=8 k=2 | 3500 | 4.486M | 1.341M | 4.1842 | 1.2317 | **1.5016** | 10,457 | 23.9 |
| `dense_wide` | dense, 8C FFN | 3500 | 1.337M | 1.337M | 4.1716 | 1.3232 | **1.5474** | 13,039 | 26.0 |

(Val loss is per-character cross-entropy in nats. "start val" is the val loss at the run's
step 0. "final train" is the mean of the last 50 training losses.)

![validation loss](results/plots/val_loss.png)

Val loss during phase 2 (same steps for both runs):

| phase-2 step | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|---|
| `dense_continue` | 1.6603 | 1.6920 | 1.6698 | 1.6328 | 1.6118 | 1.5883 | 1.5775 |
| `moe_upcycle` | 1.6603 | 1.6835 | 1.6405 | 1.5985 | 1.5702 | 1.5510 | **1.5367** |
| gap | 0 | 0.009 | 0.029 | 0.034 | 0.042 | 0.037 | **0.041** |

Both curves rise at phase-2 step 250. That is the LR re-warmup: phase 1 had decayed to 1e-4,
and phase 2 warms back up to 5e-4. It affects both runs equally.

**Approximate training compute.** Training FLOPs ≈ 6 × active params × tokens; the attention
`T²` term is ignored. In units of "active params (M) × tokens (M)":

| path | compute | final val |
|---|---|---|
| dense, 3,500 steps (`dense` → `dense_continue`) | 0.812 × 14.3 = **11.6** | 1.5775 |
| dense 2,000 steps, then upcycled MoE 1,500 steps (`dense` → `moe_upcycle`) | 0.812 × 8.2 + 1.341 × 6.1 = **14.9** | 1.5367 |
| compute-matched dense, 3,500 steps (`dense_wide`) | 1.337 × 14.3 = **19.2** | 1.5474 |
| MoE from scratch, 3,500 steps (`moe_scratch`) | 1.341 × 14.3 = **19.2** | **1.5016** |

### 4.2 Different MoE sizes (all upcycled from the same dense checkpoint, same phase 2)

| run | E | k | MoE blocks | total params | active params | **final val** | Δ vs `dense_continue` | tokens/s |
|---|---|---|---|---|---|---|---|---|
| `moe_upcycle` | 8 | 2 | all 4 | 4.486M | 1.341M | **1.5367** | −0.041 | 9,668 |
| `sweep_e16_k2` | 16 | 2 | all 4 | 8.685M | 1.345M | 1.5427 | −0.035 | 7,615 |
| `sweep_e8_k2_every2` | 8 | 2 | 1, 3 | 2.649M | 1.076M | 1.5496 | −0.028 | 12,197 |
| `sweep_e4_k1` | 4 | 1 | all 4 | 2.387M | 0.814M | 1.5671 | −0.010 | 10,730 |
| `sweep_e8_k1` | 8 | 1 | all 4 | 4.486M | 0.816M | 1.5703 | −0.007 | 8,286 |
| `dense_continue` | – | – | – | 0.812M | 0.812M | 1.5775 | 0 | 20,112 |

![upcycling sweep](results/plots/upcycle_sweep.png)

(`sweep_e8_k1`'s wall time in `results/summary.md` reads 114 min because the laptop was
suspended during that run. Its tokens/s only counts timed training steps and is unaffected.)

### 4.3 Routing

The load per expert at the final eval: each expert's share of the routing assignments, × E,
so 1.0 is perfectly uniform. With top-k routing an expert can reach at most E/k (4.0 for E=8, k=2).

![expert utilization](results/plots/expert_utilization.png)

| run | max load (× uniform) | min load | final load-balance loss (1.0 = uniform) |
|---|---|---|---|
| `moe_upcycle` (E=8, k=2) | 4.00 | 0.19 | 1.403 |
| `moe_scratch` (E=8, k=2) | 2.88 | 0.22 | 1.077 |
| `sweep_e16_k2` | 8.00 | 0.02 | 1.495 |
| `sweep_e8_k2_every2` | 2.99 | 0.34 | 1.060 |
| `sweep_e4_k1` | 1.11 | 0.91 | 1.000 |
| `sweep_e8_k1` | 1.19 | 0.91 | 1.001 |

## 5. Findings

1. **The conversion is exact, and the MoE keeps improving after it.** The upcycled model
   starts phase 2 at exactly the dense model's loss (1.6603). After 1,500 steps it ends **0.041
   nats lower than the same dense model trained for the same 1,500 steps** (1.5367 vs 1.5775).
   The gap opens within the first 500 steps and holds. That gain cost about 2× the FFN FLOPs
   per token (top-2) and 5.5× the parameters (8 FFN copies per block).

2. **Upcycling is a cheap route to a good MoE, though not the best one.** The dense → MoE
   path used about 78% of the compute of the two 3,500-step references. It beat the
   compute-matched dense model (`dense_wide`, 1.5474), even though `dense_wide` had more
   compute, started from scratch and trained 3,500 steps at full active size. It did not
   reach the MoE trained from scratch for the full budget (1.5016). With the same total
   steps, an MoE from scratch is the best of these models; upcycling wins when a trained
   dense model already exists and the extra budget is small. This is consistent with the
   upcycling paper, where MoEs trained from scratch catch up once the extra budget is large
   enough.

3. **For a fixed active size, sparsity helps.** `moe_scratch` and `dense_wide` have the same
   active parameters (1.341M vs 1.337M) and the same steps. The MoE ends 0.046 lower (1.5016
   vs 1.5474), because its 4.5M total parameters hold more knowledge at the same per-token
   compute. This is the core MoE argument, and it shows up even at this tiny scale.

4. **top-2 helps far more than more experts.** Among the upcycled variants, the gain over dense
   tracks the *active* parameters (k), not the total (E). With top-1, both E=4 and E=8 gain only
   0.007-0.010 over dense. Going to top-2 gives 0.041. Doubling to 16 experts at top-2 did not
   help (1.5427 vs 1.5367) despite 2× the parameters. Upcycling only half the blocks keeps 70%
   of the gain (0.028) with 59% of the parameters and runs 26% faster.

5. **Upcycled top-2 routers grow an "always-on" expert.** In the first two blocks of the
   upcycled E=8/k=2 model, one expert receives every token (load 4.00, the E/k maximum).
   The router's second slot is spread over the other seven. The 16-expert run does the same
   (8.00 in blocks 0 and 1). The preference was already there at step 0 (loads
   2.62 and 3.55 for those same two experts, which is likely because a random router projects the
   residual stream's shared mean direction onto one expert), and a 0.01 load-balancing weight
   never undid it. Since all experts start equal, this costs
   little. The model ends up with a shared expert plus a routed one, a split some MoE designs
   (e.g. DeepSeekMoE) build in on purpose. The MoE trained from scratch spreads its load much
   more evenly (max 2.88, in block 0 only), and top-1 routing stays almost perfectly balanced.
   Raising `moe.aux_loss_coef`, adding `moe.router_jitter` or `moe.upcycle_noise`, or setting
   a `moe.capacity_factor` are the knobs for changing this.

6. **On this CPU, the MoE is about 2× slower than dense per token.** It ran at 9.7k tok/s
   vs 20.1k. Its active parameters are 1.65× the dense model's, and the per-expert
   gather/scatter loop in Python adds the rest. A grouped GEMM or an expert-parallel kernel on
   a GPU removes most of that overhead. tokens/s varied by about 25% between runs of the same
   model (dense: 15.8k in phase 1, 20.1k in phase 2), so treat speed differences under that
   as noise.

7. **Generalization.** The gap between train and val loss is larger for the bigger models
   (`dense_continue` 0.21, `moe_upcycle` 0.23, `moe_scratch` 0.27). The 1M-character dataset was
   seen about 14 times in 3,500 steps, so the val numbers already include some overfitting.
   With more data the larger models would likely gain more, but this run doesn't test that.

Samples (prompt `ROMEO:`, 200 characters, temperature 0.8) are in `results/summary.md`.

## 6. Limitations and next steps

- This is a 0.8M-parameter character-level model on 1M characters. The trends (exact conversion,
  MoE > dense at equal steps, k matters more than E) are standard results, but the margins
  depend on scale. To scale up, pass larger `model.*` values and a BPE dataset via `data.dir`
  (any directory with nanoGPT-style `train.bin`/`val.bin`; without a `meta.pkl` the GPT-2
  tokenizer is used).
- One seed per configuration. Differences under about 0.01 nats (e.g. E=4 vs E=8 at top-1)
  are within the run-to-run noise one would expect.
- Phase 2 starts with fresh AdamW state for both the dense and the MoE run, so the comparison
  is fair. Carrying over the dense moments into the copied experts is a possible refinement.
- Dispatch loops over experts in Python. It is correct and simple, but slow at large E.

