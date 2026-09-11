"""
Minimal, self-contained training harness shared by the Q3/Q4/Q5 experiments.
Deliberately separate from train.py -- it reuses chintana.GPT/GPTConfig (the
real model and its real Adam-based configure_optimizers), but none of
train.py's DDP/checkpointing/wandb machinery, so these experiments stay cheap
and easy to instrument for a specific question.

All experiments here train on data/shakespeare_char (already prepared).
"""

import os
import pickle

import numpy as np
import torch

from chintana import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'shakespeare_char')


def load_data():
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data, meta['vocab_size']


def get_batch(data, batch_size, block_size, device, generator=None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def build_model(model_cfg: GPTConfig, seed: int, device='cpu') -> GPT:
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    model.to(device)
    return model


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def train(
    model_cfg: GPTConfig,
    seed: int,
    steps: int,
    lr_fn,
    batch_size: int,
    weight_decay: float = 0.0,
    betas=(0.9, 0.999),
    device: str = 'cpu',
    on_step=None,
):
    """
    Trains `model_cfg` from scratch for `steps` steps on shakespeare_char,
    using chintana's own GPT.configure_optimizers (real Adam, see
    chintana/gpt.py). `lr_fn(it) -> float` sets the LR each step.

    `on_step(step, loss, lr, model, optimizer)`, if given, is called after
    every optimizer.step() -- used by Q3 to log update/weight ratios.

    Returns the list of per-step training losses.
    """
    train_data, _, vocab_size = load_data()
    model_cfg = GPTConfig(**{**model_cfg.__dict__, 'vocab_size': vocab_size})
    model = build_model(model_cfg, seed=seed, device=device)
    optimizer = model.configure_optimizers(weight_decay, lr_fn(0), betas, device_type=device)

    gen = torch.Generator().manual_seed(seed)
    losses = []
    for it in range(steps):
        lr = lr_fn(it)
        for group in optimizer.param_groups:
            group['lr'] = lr
        X, Y = get_batch(train_data, batch_size, model_cfg.block_size, device, generator=gen)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if on_step is not None:
            on_step(it, loss.item(), lr, model, optimizer)
    return model, losses
