"""
Learning-rate schedules used by the assignment-11 experiments. Kept separate
from train.py (which still only implements cosine-with-warmup inline).
"""

import math


def cosine_lr(it, base_lr, warmup_iters, lr_decay_iters, min_lr):
    """Same shape as train.py's get_lr: linear warmup, then cosine decay to min_lr."""
    if it < warmup_iters:
        return base_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def wsd_lr(it, base_lr, warmup_iters, decay_start_iter, total_iters, min_lr):
    """Warmup -> Stable (constant at base_lr) -> Decay (linear to min_lr over the tail)."""
    if it < warmup_iters:
        return base_lr * (it + 1) / (warmup_iters + 1)
    if it < decay_start_iter:
        return base_lr
    if it >= total_iters:
        return min_lr
    decay_ratio = (it - decay_start_iter) / (total_iters - decay_start_iter)
    assert 0 <= decay_ratio <= 1
    return base_lr + decay_ratio * (min_lr - base_lr)
