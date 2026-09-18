"""Shared helpers: build the demo model, synthesize data, flatten/shard params."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chintana.config import GPTConfig
from chintana.gpt import GPT

# Deliberately tiny + self-contained (synthetic byte-level data, no data/ prep
# step needed) so the simulation runs in seconds on a laptop CPU. n_layer=4,
# n_embd=128 gives ~0.8M parameters -- enough to have a real params/grads/optim
# split worth plotting, small enough that 32 virtual GPUs (see virtual_gpu.py)
# each doing a real forward+backward is fast.
DEMO_CONFIG = GPTConfig(
    block_size=64,
    vocab_size=256,  # byte-level, so no tokenizer/data prep dependency
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=False,
)


def build_model(seed=0):
    torch.manual_seed(seed)
    return GPT(DEMO_CONFIG)


def synthetic_batch(n_sequences, block_size, vocab_size, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (n_sequences, block_size), generator=g)
    y = torch.randint(0, vocab_size, (n_sequences, block_size), generator=g)
    return x, y


def split_batch(x, y, n_shards):
    """Split one global batch into n_shards equal-size micro-batches (one per virtual GPU)."""
    assert x.size(0) % n_shards == 0, "global batch must divide evenly across virtual GPUs"
    return list(zip(x.chunk(n_shards), y.chunk(n_shards)))


def flatten(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def flat_param_vector(model):
    return flatten(list(model.parameters()))


def load_flat_into_model(model, flat):
    """Copy a flat fp32 vector into a model's parameters in-place (no grad tracking)."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n
    assert offset == flat.numel()


def flat_grad_vector(model):
    """Flatten .grad of every parameter (zeros for any that is None)."""
    parts = []
    for p in model.parameters():
        parts.append(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel()))
    return torch.cat(parts)


def num_params(model):
    return sum(p.numel() for p in model.parameters())


def shard_bounds(total, n_shards):
    """Even shard boundaries over a (possibly padded) length divisible by n_shards."""
    assert total % n_shards == 0
    size = total // n_shards
    return [(i * size, (i + 1) * size) for i in range(n_shards)]


def padded_len(total, n_shards):
    if total % n_shards == 0:
        return total
    return total + (n_shards - total % n_shards)
