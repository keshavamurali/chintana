"""
Nested, type-checked config loading built on OmegaConf structured configs.

Replaces the old "poor man's configurator" (exec(open('configurator.py').read())):
instead of a flat namespace of globals mutated by exec-ing a config file, each
entrypoint (train.py, sample.py, bench.py) declares a dataclass describing its
own config, and load_config() layers overrides onto its defaults:

  1) the dataclass's own field defaults
  2) zero or more YAML files, given as bare (no "=") command-line arguments,
     merged in the order given
  3) "dotted.key=value" command-line overrides, applied last

Example:
$ python train.py config/train_shakespeare_char.yaml system.device=cpu optim.learning_rate=5e-4
"""

import sys
from typing import Type, TypeVar

from omegaconf import OmegaConf

T = TypeVar('T')


def load_config(schema: Type[T], argv=None) -> T:
    argv = sys.argv[1:] if argv is None else argv
    file_args = [a for a in argv if '=' not in a]
    override_args = [a for a in argv if '=' in a]

    cfg = OmegaConf.structured(schema)
    for path in file_args:
        print(f"Overriding config with {path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if override_args:
        print(f"Overriding config with CLI args: {override_args}")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_args))

    return OmegaConf.to_object(cfg)
