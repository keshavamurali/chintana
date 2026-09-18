"""Q2 -- how does per-GPU memory change across dp/zero1/zero2/zero3, and how
does it scale as the cluster grows from 1 to 32 virtual GPUs?

All numbers here are measured bytes (tensor.numel() * element_size()) off the
buffers each VirtualGPU actually holds after run_step -- not the textbook
formula typed in by hand. The README compares the two and they match exactly.
"""

import csv
from pathlib import Path

import common
import plotting
import virtual_gpu
import zero_engine

RESULTS_DIR = Path(__file__).parent / "results"
N_GPUS_DEMO = 32
GLOBAL_BATCH = 128
N_SWEEP = (1, 2, 4, 8, 16, 32)


def run_stage(stage, n_gpus, x, y):
    gpus, ref_flat = virtual_gpu.make_cluster(n_gpus, common.build_model, seed=0)
    return zero_engine.run_step(stage, gpus, ref_flat, x, y)


def main():
    x, y = common.synthetic_batch(GLOBAL_BATCH, common.DEMO_CONFIG.block_size, common.DEMO_CONFIG.vocab_size, seed=1)

    # --- per-component memory breakdown at N=32 ---
    per_stage_bytes = {}
    peak_extra = {}
    for stage in zero_engine.STAGES:
        result = run_stage(stage, N_GPUS_DEMO, x, y)
        per_stage_bytes[stage] = result.per_gpu_bytes[0]  # every GPU is identical (even shard split)
        peak_extra[stage] = result.peak_extra_bytes

    rows = []
    for stage in zero_engine.STAGES:
        b = per_stage_bytes[stage]
        rows.append({
            "stage": stage,
            "n_gpus": N_GPUS_DEMO,
            "params_bytes": b["params"],
            "grads_bytes": b["grads"],
            "optimizer_bytes": b["optimizer"],
            "steady_state_total_bytes": sum(b.values()),
            "peak_extra_bytes": peak_extra[stage],
            "peak_total_bytes": sum(b.values()) + peak_extra[stage],
        })

    out_csv = RESULTS_DIR / "logs" / "q2_memory_breakdown.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")
    for r in rows:
        print(f"{r['stage']:>6}: params={r['params_bytes']/1e6:.2f}MB grads={r['grads_bytes']/1e6:.2f}MB "
              f"optim={r['optimizer_bytes']/1e6:.2f}MB steady={r['steady_state_total_bytes']/1e6:.2f}MB "
              f"peak={r['peak_total_bytes']/1e6:.2f}MB")

    plotting.stacked_memory_bar(
        per_stage_bytes, list(zero_engine.STAGES), zero_engine.STAGE_LABELS,
        RESULTS_DIR / "plots" / "q2_memory_breakdown.png",
        f"Per-GPU memory at N={N_GPUS_DEMO} virtual GPUs (steady state)",
    )

    # --- scaling sweep: per-GPU memory vs N, for each stage ---
    sweep_rows = []
    series = {stage: [] for stage in zero_engine.STAGES}
    for n_gpus in N_SWEEP:
        for stage in zero_engine.STAGES:
            result = run_stage(stage, n_gpus, x, y)
            total = sum(result.per_gpu_bytes[0].values())
            series[stage].append(total)
            sweep_rows.append({"stage": stage, "n_gpus": n_gpus, "bytes_per_gpu": total})

    out_csv2 = RESULTS_DIR / "logs" / "q2_memory_vs_n.csv"
    with open(out_csv2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "n_gpus", "bytes_per_gpu"])
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"wrote {out_csv2}")

    plotting.memory_vs_n_lines(
        series, N_SWEEP, list(zero_engine.STAGES), zero_engine.STAGE_LABELS,
        RESULTS_DIR / "plots" / "q2_memory_vs_n.png",
        "Per-GPU memory vs. cluster size (log-log)",
    )


if __name__ == "__main__":
    main()
