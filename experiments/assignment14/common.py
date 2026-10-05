"""Config schema, data loading, checkpoints and the training step shared by every script.

The config is an OmegaConf structured config loaded with chintana.load_config,
exactly like the root train.py: dataclass defaults -> YAML file(s) given as
bare arguments -> dotted key=value overrides, e.g.

  python experiments/assignment14/train.py experiments/assignment14/config/moe_upcycle.yaml \
      moe.n_experts=16 moe.top_k=1 train.steps=2000
"""

import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from chintana.config import GPTConfig

from model import MoEConfig, MoEGPT, upcycle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS_DIR = HERE / "results"
DEFAULT_DATA_DIR = str(REPO / "data" / "shakespeare_char")


def default_model_config():
    # 4 x (d=128, 4 heads), 128-character context; vocab_size is read from the data's meta.pkl
    return GPTConfig(block_size=128, vocab_size=65, n_layer=4, n_head=4, n_embd=128,
                     dropout=0.0, bias=False)


@dataclass
class InitConfig:
    mode: str = 'scratch'  # scratch | resume (load a checkpoint as is) | upcycle (dense checkpoint -> MoE)
    from_ckpt: str = ''  # checkpoint for resume / upcycle (a run's ckpt.pt)


@dataclass
class DataConfig:
    dir: str = DEFAULT_DATA_DIR
    batch_size: int = 32


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 1e-3
    min_lr_frac: float = 0.1  # cosine decays to lr * min_lr_frac
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class RunConfig:
    name: str = 'run'
    out_dir: str = str(RESULTS_DIR / "runs")
    device: str = 'auto'  # auto | cpu | cuda | cuda:N | mps
    seed: int = 1337
    log_interval: int = 25
    eval_interval: int = 250
    eval_batches: int = 20
    eval_batch_size: int = 64
    speed_warmup_steps: int = 5  # steps excluded from the tokens/s measurement
    save_checkpoint: bool = True
    sample_tokens: int = 200
    sample_prompt: str = 'ROMEO:'


@dataclass
class A14Config:
    model: GPTConfig = field(default_factory=default_model_config)
    moe: MoEConfig = field(default_factory=MoEConfig)
    init: InitConfig = field(default_factory=InitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)


def resolve_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def save_checkpoint(path, model, step):
    torch.save({'model': model.state_dict(), 'model_config': asdict(model.config),
                'moe_config': asdict(model.moe_cfg), 'step': step}, path)


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model = MoEGPT(GPTConfig(**ckpt['model_config']), MoEConfig(**ckpt['moe_config']))
    model.load_state_dict(ckpt['model'])
    return model


def build_model(cfg: A14Config):
    """The model a run starts from, per init.mode. For resume/upcycle the
    architecture comes from the checkpoint, and cfg.model is updated to match."""
    mode = cfg.init.mode
    if mode == 'scratch':
        torch.manual_seed(cfg.run.seed)
        return MoEGPT(cfg.model, cfg.moe)
    assert cfg.init.from_ckpt, f"init.mode={mode} needs init.from_ckpt"
    src = load_checkpoint(cfg.init.from_ckpt)
    cfg.model = src.config
    if mode == 'resume':
        cfg.moe = src.moe_cfg
        return src
    if mode == 'upcycle':
        assert not src.moe_layers, "upcycle expects a dense checkpoint (moe.n_experts=0)"
        return upcycle(src, cfg.moe, seed=cfg.run.seed)
    raise ValueError(f"unknown init.mode {mode!r}")


def build_optimizer(model, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {'params': [p for p in params if p.dim() >= 2], 'weight_decay': cfg.weight_decay},
        {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def lr_at(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    min_lr = cfg.lr * cfg.min_lr_frac
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))) * (cfg.lr - min_lr)


def train_step(model, optimizer, x, y, grad_clip):
    _, lm_loss = model(x, y)
    aux, z = model.aux_losses()
    loss = lm_loss + model.moe_cfg.aux_loss_coef * aux + model.moe_cfg.z_loss_coef * z
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return lm_loss.item(), aux.item(), z.item()


@torch.no_grad()
def evaluate(model, batches):
    """Mean val cross-entropy, plus per-MoE-layer routing statistics."""
    model.eval()
    losses = []
    counts = [torch.zeros(m.n_experts, dtype=torch.long) for m in model.moe_modules()]
    aux, dropped, assigned = [], 0, 0
    for x, y in batches:
        losses.append(model(x, y)[1].item())
        for c, m in zip(counts, model.moe_modules()):
            c += m.counts.cpu()
            dropped += m.dropped
            assigned += int(m.counts.sum())
        if counts:
            aux.append(model.aux_losses()[0].item())
    model.train()
    stats = {'val_loss': sum(losses) / len(losses)}
    if counts:
        stats['expert_frac'] = [(c.float() / c.sum()).tolist() for c in counts]
        stats['aux_loss'] = sum(aux) / len(aux)
        stats['drop_frac'] = dropped / max(1, assigned)
    return stats


def load_codec(data_dir):
    """(vocab_size, encode, decode) from a nanoGPT-style meta.pkl, or GPT-2 BPE."""
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
    """uint16 token streams (train.bin / val.bin), sampled as random windows."""

    def __init__(self, data_dir, block_size):
        d = Path(data_dir)
        self.train = np.memmap(d / 'train.bin', dtype=np.uint16, mode='r')
        self.val = np.memmap(d / 'val.bin', dtype=np.uint16, mode='r')
        self.block_size = block_size

    def _windows(self, data, starts, device):
        idx = np.asarray(starts)[:, None] + np.arange(self.block_size + 1)
        chunk = torch.from_numpy(data[idx].astype(np.int64))
        return chunk[:, :-1].contiguous().to(device), chunk[:, 1:].contiguous().to(device)

    def train_batch(self, batch_size, generator, device):
        starts = torch.randint(len(self.train) - self.block_size - 1, (batch_size,), generator=generator)
        return self._windows(self.train, starts.numpy(), device)

    def val_batches(self, n_batches, batch_size, device):
        """The same evenly spaced windows for every run, so val losses are comparable."""
        n = n_batches * batch_size
        starts = np.linspace(0, len(self.val) - self.block_size - 2, n).astype(np.int64)
        return [self._windows(self.val, starts[i:i + batch_size], device) for i in range(0, n, batch_size)]
