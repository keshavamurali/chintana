from dataclasses import dataclass, field

from .config import GPTConfig


@dataclass
class IOConfig:
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'


@dataclass
class WandbConfig:
    log: bool = False
    project: str = 'owt'
    run_name: str = 'gpt2'


@dataclass
class DataConfig:
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size


@dataclass
class OptimConfig:
    # adam optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


@dataclass
class SystemConfig:
    backend: str = 'nccl'  # 'nccl', 'gloo', etc. (DDP)
    device: str = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'auto'  # 'auto', 'float32', 'bfloat16', or 'float16' ('auto' picks bf16 if supported, else fp16)
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster


def _default_model_config() -> GPTConfig:
    # train.py has historically defaulted to bias=False ("a bit better and
    # faster" per GPTConfig's own docstring), which differs from GPTConfig's
    # own dataclass default of bias=True (needed so GPTConfig() alone still
    # matches GPT-2 checkpoint shapes for GPT.from_pretrained).
    return GPTConfig(bias=False)


@dataclass
class TrainConfig:
    io: IOConfig = field(default_factory=IOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: GPTConfig = field(default_factory=_default_model_config)
    optim: OptimConfig = field(default_factory=OptimConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
