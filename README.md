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
uv run python train.py config/train_shakespeare_char.yaml system.device=cpu system.compile=false
```

Config files are nested YAML, loaded and merged with `OmegaConf` (see
`chintana/train_config.py` for the schema). Override any value from the
command line with a dotted `key=value`, e.g.:

```bash
uv run python train.py config/train_shakespeare_char.yaml optim.max_iters=1000 data.batch_size=32
```

Checkpoints are written to `io.out_dir` (see the config file) as `ckpt.pt`.

## Sample from a checkpoint

```bash
uv run python sample.py out_dir=out-shakespeare-char device=cpu
```

## Project layout

```
chintana/       model code (config, norm, attention, mlp, block, gpt) +
                 nested config schema and loader (train_config.py, configuration.py)
data/           per-dataset prepare.py scripts + prepared .bin files
config/         training run configs (YAML)
train.py        training loop
sample.py       generate text from a checkpoint
bench.py        benchmarking script
```
