"""Config schema, data loading, and the training step shared by every script.

The config is an OmegaConf structured config loaded with chintana.load_config,
exactly like the root train.py: dataclass defaults -> YAML file(s) given as
bare arguments -> dotted key=value overrides, e.g.

  python experiments/assignment13/train.py experiments/assignment13/config/run2_reversible.yaml \
      rev.variant=leapfrog data.batch_size=64
"""

import inspect
import math
import pickle
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from chintana.config import GPTConfig

from model import RevGPT

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEFAULT_DATA_DIR = str(HERE / "data" / "fineweb_edu")


def default_model_config():
    # 9 x (d=256, 8 heads) + tied 50304-token GPT-2 vocabulary = 20.1M parameters
    return GPTConfig(block_size=512, vocab_size=50304, n_layer=9, n_head=8, n_embd=256,
                     dropout=0.0, bias=False)


@dataclass
class RevConfig:
    variant: str = 'baseline'  # baseline | midpoint | midpoint_a | leapfrog | hamiltonian
    h: Optional[float] = None  # step size; None -> reversible.DEFAULT_H for the variant
    a_seed: int = 0  # seed for midpoint_a's per-layer a_l
    memory_saving: bool = True  # False: same dynamics, but ordinary autograd (stores activations)


@dataclass
class LossConfig:
    impl: str = 'fused'  # fused | standard
    chunk: int = 1024  # tokens per logits chunk for the fused head


@dataclass
class DataConfig:
    dir: str = DEFAULT_DATA_DIR
    batch_size: int = 64  # sequences per optimizer step (no gradient accumulation)
    total_tokens: int = 50_000_000  # training budget; steps = ceil(total_tokens / (batch_size * block_size))


@dataclass
class OptimConfig:
    lr: float = 1e-3  # peak lr at ref_batch_size
    lr_scaling: str = 'none'  # none | sqrt | linear -- rescales lr by batch_size / ref_batch_size
    ref_batch_size: int = 64
    lr_cap: float = 3e-3  # upper bound after scaling
    min_lr_frac: float = 0.1  # cosine decays to lr * min_lr_frac
    warmup_frac: float = 0.03
    min_warmup_steps: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class RunConfig:
    name: str = 'run'
    out_dir: str = str(RESULTS_DIR / "runs")
    device: str = 'auto'  # auto | cpu | cuda | cuda:N
    dtype: str = 'auto'  # auto (bf16 on Ampere+, fp16 on older GPUs, fp32 on CPU) | float32 | float16 | bfloat16
    seed: int = 1337
    log_interval: int = 10
    n_evals: int = 10  # evenly spaced evals over the run, plus one at step 0
    eval_batches: int = 50
    eval_batch_size: int = 32
    speed_warmup_steps: int = 5  # steps excluded from the tokens/s measurement
    save_checkpoint: bool = False
    sample_tokens: int = 64
    sample_prompt: str = 'The meaning of'


@dataclass
class A13Config:
    model: GPTConfig = field(default_factory=default_model_config)
    rev: RevConfig = field(default_factory=RevConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)


def build_model(cfg: A13Config, **model_overrides):
    model_cfg = GPTConfig(**{**cfg.model.__dict__, **model_overrides})
    return RevGPT(model_cfg, variant=cfg.rev.variant, h=cfg.rev.h, a_seed=cfg.rev.a_seed,
                  memory_saving=cfg.rev.memory_saving, loss_impl=cfg.loss.impl,
                  loss_chunk=cfg.loss.chunk)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return {'total': total, 'non_embedding': total - model.transformer.wpe.weight.numel()
            - model.transformer.wte.weight.numel()}


def resolve_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def resolve_dtype(name, device):
    if name != 'auto':
        return getattr(torch, name)
    if not device.startswith('cuda'):
        return torch.float32
    # is_bf16_supported() reports True on pre-Ampere GPUs via emulation, which is slow
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def amp_context(device, dtype):
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    if dtype == torch.float32:
        return nullcontext
    return lambda: torch.autocast(device_type=device_type, dtype=dtype)


def make_scaler(device, dtype):
    enabled = device.startswith('cuda') and dtype == torch.float16
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model, cfg: OptimConfig, lr, device):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {'params': [p for p in params if p.dim() >= 2], 'weight_decay': cfg.weight_decay},
        {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},
    ]
    extra = {}
    if device.startswith('cuda') and 'fused' in inspect.signature(torch.optim.AdamW).parameters:
        extra['fused'] = True
    return torch.optim.AdamW(groups, lr=lr, betas=(cfg.beta1, cfg.beta2), **extra)


def peak_lr(cfg: OptimConfig, batch_size):
    ratio = batch_size / cfg.ref_batch_size
    scale = {'none': 1.0, 'sqrt': math.sqrt(ratio), 'linear': ratio}[cfg.lr_scaling]
    return min(cfg.lr * scale, cfg.lr_cap) if scale != 1.0 else cfg.lr


def lr_at(step, total_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))) * (lr - min_lr)


def train_step(model, optimizer, scaler, x, y, amp, grad_clip):
    with amp():
        _, loss = model(x, y)
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def load_codec(data_dir):
    """(vocab_size, encode, decode) for the tokens in data_dir.

    A nanoGPT-style meta.pkl (e.g. data/shakespeare_char) means a character
    vocabulary; anything else is GPT-2 BPE, whose vocab_size is left to the
    model config (50304, padded from 50257).
    """
    meta = Path(data_dir) / 'meta.pkl'
    if meta.exists():
        with open(meta, 'rb') as f:
            m = pickle.load(f)
        stoi, itos = m['stoi'], m['itos']
        return m['vocab_size'], (lambda s: [stoi[c] for c in s if c in stoi]), \
            (lambda ids: ''.join(itos[i] for i in ids))
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    return None, enc.encode_ordinary, enc.decode


class TokenData:
    """uint16 token streams written by prepare_data.py, sampled as random windows."""

    def __init__(self, data_dir, block_size):
        d = Path(data_dir)
        self.train = np.memmap(d / 'train.bin', dtype=np.uint16, mode='r')
        self.val = np.memmap(d / 'val.bin', dtype=np.uint16, mode='r')
        self.block_size = block_size

    def _windows(self, data, starts, device):
        idx = np.asarray(starts)[:, None] + np.arange(self.block_size + 1)
        chunk = torch.from_numpy(data[idx].astype(np.int64))
        x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        if device.startswith('cuda'):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def train_batch(self, batch_size, generator, device):
        starts = torch.randint(len(self.train) - self.block_size - 1, (batch_size,), generator=generator)
        return self._windows(self.train, starts.numpy(), device)

    def val_batches(self, n_batches, batch_size, device):
        """The same evenly spaced windows for every run, so val losses are comparable."""
        n = n_batches * batch_size
        starts = np.linspace(0, len(self.val) - self.block_size - 2, n).astype(np.int64)
        return [self._windows(self.val, starts[i:i + batch_size], device) for i in range(0, n, batch_size)]


def random_batch(batch_size, block_size, vocab_size, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch_size, block_size), generator=g)
    y = torch.randint(0, vocab_size, (batch_size, block_size), generator=g)
    return x.to(device), y.to(device)
