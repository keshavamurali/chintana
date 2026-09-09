# chintana

A small, modular GPT implementation (multi-file refactor of nanoGPT).

## Setup

```bash
uv sync
```

## Prepare data

```bash
uv run python data/shakespeare_char/prepare.py   # already prepared, included in repo
uv run python data/shakespeare/prepare.py        # BPE-tokenized shakespeare
```

## Train

```bash
uv run python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

Override any config value from the command line, e.g.:

```bash
uv run python train.py config/train_shakespeare_char.py --max_iters=1000 --batch_size=32
```

Checkpoints are written to `out_dir` (see the config file) as `ckpt.pt`.

## Sample from a checkpoint

```bash
uv run python sample.py --out_dir=out-shakespeare-char --device=cpu
```

## Project layout

```
chintana/       model code (config, norm, attention, mlp, block, gpt)
data/           per-dataset prepare.py scripts + prepared .bin files
config/         training run configs
train.py        training loop
sample.py       generate text from a checkpoint
bench.py        benchmarking script
```
