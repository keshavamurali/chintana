"""Simulated collective communication ops, costed the way a ring implementation
actually moves bytes (this is what NCCL/gloo do under the hood, and it's the
reason ZeRO-1/2's extra scatter+gather round trip turns out to cost exactly the
same as a plain all-reduce -- see README).

For a buffer of P elements split ring-style across N ranks:
  - all_reduce(P):      2*(N-1)/N * P elements move, in total, across the cluster
  - reduce_scatter(P):    (N-1)/N * P elements move (each rank ends with P/N)
  - all_gather(P):        (N-1)/N * P elements move (each rank starts with P/N)
so all_reduce == reduce_scatter + all_gather, exactly -- not a coincidence.

Each call performs the real tensor math (so results are correct, not just
costed) and appends the bytes actually moved to a shared CommMeter.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class CommMeter:
    events: list = field(default_factory=list)  # (op_name, bytes_moved)

    def record(self, op_name, n_elements, element_size, n_ranks):
        bytes_moved = int(round(2 * (n_ranks - 1) / n_ranks * n_elements * element_size))
        self.events.append((op_name, bytes_moved))

    def record_half(self, op_name, n_elements, element_size, n_ranks):
        """Cost of a reduce-scatter or an all-gather alone (half an all-reduce)."""
        bytes_moved = int(round((n_ranks - 1) / n_ranks * n_elements * element_size))
        self.events.append((op_name, bytes_moved))

    def total_bytes(self):
        return sum(b for _, b in self.events)

    def by_op(self):
        totals = {}
        for name, b in self.events:
            totals[name] = totals.get(name, 0) + b
        return totals


def all_reduce_mean(shard_tensors, meter: "CommMeter", op_name="all_reduce"):
    """Average a list of per-GPU tensors (all the same shape) and return the
    single averaged tensor every GPU ends up with."""
    n_ranks = len(shard_tensors)
    stacked = sum(shard_tensors) / n_ranks
    meter.record(op_name, shard_tensors[0].numel(), shard_tensors[0].element_size(), n_ranks)
    return stacked


def reduce_scatter_mean(full_tensors, n_shards_bounds, meter: "CommMeter", op_name="reduce_scatter"):
    """Average n_ranks full-length tensors, but return only each rank's shard
    of the result (like NCCL reduce-scatter)."""
    n_ranks = len(full_tensors)
    averaged = sum(full_tensors) / n_ranks
    shards = [averaged[lo:hi] for lo, hi in n_shards_bounds]
    meter.record_half(op_name, full_tensors[0].numel(), full_tensors[0].element_size(), n_ranks)
    return shards


def all_gather(shards, meter: "CommMeter", op_name="all_gather"):
    """Concatenate per-rank shards into the full tensor every rank ends up with."""
    n_ranks = len(shards)
    full = torch.cat(shards)
    meter.record_half(op_name, full.numel(), shards[0].element_size(), n_ranks)
    return full
