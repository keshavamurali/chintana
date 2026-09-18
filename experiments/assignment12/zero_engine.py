"""The four stages this assignment simulates: plain data parallelism (the
baseline every ZeRO stage is measured against), ZeRO-1 (Pos, optimizer-state
partitioning), ZeRO-2 (Pos+g, + gradient partitioning), ZeRO-3 (Pos+g+p, +
parameter partitioning).

Every stage runs the *same* computation -- a real forward+backward of a real
chintana.GPT on 32 VirtualGPUs (see virtual_gpu.py), each with its own
micro-batch, executed concurrently via a thread pool. What differs between
stages is bookkeeping only: which of {params, gradients, optimizer state} each
GPU is allowed to keep a full copy of vs. only its 1/32 shard of, and how many
collective communication ops (see collectives.py) that requires. Memory and
communication numbers reported below are measured off the real tensors moved
through that bookkeeping, not typed-in constants -- see README for how they
compare to the textbook formulas.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

import common
from collectives import CommMeter, all_gather, all_reduce_mean, reduce_scatter_mean

STAGES = ("dp", "zero1", "zero2", "zero3")

STAGE_LABELS = {
    "dp": "Baseline DP (no ZeRO)",
    "zero1": "ZeRO-1 (P_os)",
    "zero2": "ZeRO-2 (P_os+g)",
    "zero3": "ZeRO-3 (P_os+g+p)",
}

ADAM_LR = 1e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


@dataclass
class StepResult:
    stage: str
    n_gpus: int
    per_gpu_bytes: list  # one {"params":..,"grads":..,"optimizer":..} dict per GPU
    peak_extra_bytes: int  # transient bytes above steady state, on top of one GPU's steady total (0 unless zero3)
    comm_events: list  # [(op_name, bytes), ...]
    wall_time_sec: float
    updated_params: torch.Tensor  # flat, unpadded length -- for correctness checks
    mean_loss: float


def _local_backward_all_gpus(gpus, batches):
    """Run forward+backward on all GPUs concurrently (each its own thread --
    real autograd graphs, real matmuls; PyTorch's C++ kernels release the GIL
    during the large ops so this does get some real overlap, though a single
    CPU obviously isn't 32 independent accelerators -- see README)."""

    def _one(gpu, xy):
        x, y = xy
        gpu.model.zero_grad(set_to_none=True)
        _, loss = gpu.model(x, y)
        loss.backward()
        return common.flat_grad_vector(gpu.model), float(loss.item())

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = list(pool.map(_one, gpus, batches))
    grads, losses = zip(*results)
    return list(grads), list(losses)


def _adam_update(param_slice, grad_slice):
    """One plain-Adam step from a fresh (m=v=0) optimizer state -- this demo
    measures one step's mechanics/memory, not a training curve, so there is no
    state persisted across steps. At t=1 the bias-corrected m_hat/v_hat reduce
    to grad and grad**2 exactly; m/v below are the (uncorrected) raw moments,
    which is what actually occupies optimizer-state memory."""
    m = (1 - ADAM_BETAS[0]) * grad_slice
    v = (1 - ADAM_BETAS[1]) * grad_slice.pow(2)
    m_hat = m / (1 - ADAM_BETAS[0])
    v_hat = v / (1 - ADAM_BETAS[1])
    update = ADAM_LR * m_hat / (v_hat.sqrt() + ADAM_EPS)
    return param_slice - update, m, v


def _shard_adam_updates(ref_flat_padded, grad_shards, bounds):
    """Apply one Adam step independently on each shard -- this is exactly
    what ZeRO-1/2/3 do: every rank only ever runs the optimizer math on the
    parameter shard it owns, using its own private m/v shard."""
    updated_shards, m_shards, v_shards = [], [], []
    for (lo, hi), grad_shard in zip(bounds, grad_shards):
        upd, m_s, v_s = _adam_update(ref_flat_padded[lo:hi], grad_shard)
        updated_shards.append(upd)
        m_shards.append(m_s)
        v_shards.append(v_s)
    return updated_shards, m_shards, v_shards


def run_step(stage, gpus, ref_flat, x, y):
    assert stage in STAGES
    n_gpus = len(gpus)
    total = ref_flat.numel()
    padded_total = common.padded_len(total, n_gpus)
    pad = padded_total - total
    bounds = common.shard_bounds(padded_total, n_gpus)
    meter = CommMeter()

    batches = common.split_batch(x, y, n_gpus)

    t0 = time.perf_counter()

    # Every GPU starts each step holding the full (synced) parameter vector --
    # true for all four stages at the moment forward/backward runs; ZeRO-3 is
    # the one that normally *doesn't* keep this resident between steps, so for
    # it we treat this as the transient, just-gathered copy (see below).
    for gpu in gpus:
        gpu.put("params_full", ref_flat)

    local_grads, losses = _local_backward_all_gpus(gpus, batches)
    local_grads_padded = [torch.nn.functional.pad(g, (0, pad)) for g in local_grads]
    ref_flat_padded = torch.nn.functional.pad(ref_flat, (0, pad))

    if stage == "dp":
        full_grad = all_reduce_mean([g[:total] for g in local_grads], meter, "all_reduce(grad)")
        updated, m, v = _adam_update(ref_flat, full_grad)
        for gpu in gpus:
            gpu.put("grad_full", full_grad)
            gpu.put("exp_avg_full", m)
            gpu.put("exp_avg_sq_full", v)
        peak_extra = 0

    elif stage == "zero1":
        # Full local grad stays resident everywhere (ZeRO-1 doesn't touch
        # gradient memory) -- reduce-scatter is only used to get the reduced
        # value for each GPU's own parameter shard.
        grad_shards = reduce_scatter_mean(local_grads_padded, bounds, meter, "reduce_scatter(grad)")
        updated_shards, m_shards, v_shards = _shard_adam_updates(ref_flat_padded, grad_shards, bounds)
        updated = all_gather(updated_shards, meter, "all_gather(params)")
        for gpu, local_grad, m_s, v_s in zip(gpus, local_grads_padded, m_shards, v_shards):
            gpu.put("grad_full", local_grad)  # unreduced local copy, retained (this is the point of stage 1)
            gpu.put("exp_avg_shard", m_s)
            gpu.put("exp_avg_sq_shard", v_s)
        peak_extra = 0

    elif stage == "zero2":
        # Same comm pattern as ZeRO-1, but each GPU frees its full local grad
        # right after reduce-scatter and keeps only the reduced shard.
        grad_shards = reduce_scatter_mean(local_grads_padded, bounds, meter, "reduce_scatter(grad)")
        updated_shards, m_shards, v_shards = _shard_adam_updates(ref_flat_padded, grad_shards, bounds)
        updated = all_gather(updated_shards, meter, "all_gather(params)")
        for gpu, grad_shard, m_s, v_s in zip(gpus, grad_shards, m_shards, v_shards):
            gpu.put("grad_shard", grad_shard)
            gpu.put("exp_avg_shard", m_s)
            gpu.put("exp_avg_sq_shard", v_s)
        peak_extra = 0

    else:  # zero3
        # Params are only gathered just-in-time for forward and (again) for
        # backward, and freed back down to a shard the rest of the time.
        # This demo gathers the *whole* model in one shot rather than
        # layer-by-layer (production ZeRO-3 overlaps a per-layer gather with
        # compute to keep the transient bump far below a full 4*Psi -- see
        # README), so the peak reported here is the worst case, not the
        # typical one.
        meter.record_half("all_gather(params, forward)", padded_total, 4, n_gpus)
        meter.record_half("all_gather(params, backward)", padded_total, 4, n_gpus)
        grad_shards = reduce_scatter_mean(local_grads_padded, bounds, meter, "reduce_scatter(grad)")
        updated_shards, m_shards, v_shards = _shard_adam_updates(ref_flat_padded, grad_shards, bounds)
        updated = torch.cat(updated_shards)  # each GPU only needs its own shard, no post-step all_gather
        param_shards = [ref_flat_padded[lo:hi] for lo, hi in bounds]
        for gpu, p_shard, grad_shard, m_s, v_s in zip(gpus, param_shards, grad_shards, m_shards, v_shards):
            gpu.drop("params_full")
            gpu.put("params_shard", p_shard)
            gpu.put("grad_shard", grad_shard)
            gpu.put("exp_avg_shard", m_s)
            gpu.put("exp_avg_sq_shard", v_s)
        # peak transient: a full gathered param buffer + a full local grad
        # buffer resident at once during the backward pass, on top of the
        # steady-state shards every GPU always keeps.
        peak_extra = padded_total * 4 + padded_total * 4

    dt = time.perf_counter() - t0
    updated_unpadded = updated[:total]

    per_gpu_bytes = [gpu.bytes_by_bucket() for gpu in gpus]

    # sync every GPU's live model to the new params for the next call
    for gpu in gpus:
        common.load_flat_into_model(gpu.model, updated_unpadded)

    return StepResult(
        stage=stage,
        n_gpus=n_gpus,
        per_gpu_bytes=per_gpu_bytes,
        peak_extra_bytes=peak_extra,
        comm_events=meter.events,
        wall_time_sec=dt,
        updated_params=updated_unpadded,
        mean_loss=sum(losses) / len(losses),
    )
