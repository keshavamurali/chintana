"""A VirtualGPU: one simulated device in a 32-way data-parallel cluster.

There's no real hardware behind this -- it's a CPU process pretending to be
one rank of a DP group. What makes it a useful stand-in rather than a toy is
that its memory footprint is *measured*, not typed in by hand: every number
reported below comes from `tensor.numel() * tensor.element_size()` on the
actual buffers a given ZeRO stage lets that GPU hold onto, at the moment it
holds them. See zero_engine.py for how the four stages differ in exactly
which buffers that is.
"""

from dataclasses import dataclass, field

import torch

from chintana.gpt import GPT

import common


@dataclass
class VirtualGPU:
    rank: int
    model: GPT  # this GPU's private copy of the model (its own autograd graph)
    resident: dict = field(default_factory=dict)  # name -> tensor currently "on device"

    def put(self, name, tensor):
        self.resident[name] = tensor

    def drop(self, name):
        self.resident.pop(name, None)

    def bytes_by_bucket(self):
        """Group resident tensors into {params, grads, optimizer} byte totals.

        Bucket is inferred from the tensor's key prefix (see zero_engine.py's
        naming: 'params*', 'grad*', 'exp_avg*'/'exp_avg_sq*').
        """
        buckets = {"params": 0, "grads": 0, "optimizer": 0}
        for name, t in self.resident.items():
            n_bytes = t.numel() * t.element_size()
            if name.startswith("param"):
                buckets["params"] += n_bytes
            elif name.startswith("grad"):
                buckets["grads"] += n_bytes
            elif name.startswith("exp_avg"):
                buckets["optimizer"] += n_bytes
            else:
                raise ValueError(f"unrecognized resident buffer: {name}")
        return buckets

    def total_bytes(self):
        return sum(self.bytes_by_bucket().values())


def make_cluster(n_gpus, reference_model_builder, seed=0):
    """Build n_gpus VirtualGPUs, each with its own model instance sharing the
    same initialization (so, before any step, every GPU is bit-for-bit identical
    -- exactly like real DP training starts from a broadcast/shared init)."""
    ref = reference_model_builder(seed=seed)
    ref_flat = common.flat_param_vector(ref)
    gpus = []
    for rank in range(n_gpus):
        m = reference_model_builder(seed=seed)
        common.load_flat_into_model(m, ref_flat)
        gpus.append(VirtualGPU(rank=rank, model=m))
    return gpus, ref_flat
