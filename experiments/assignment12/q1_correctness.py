"""Q1 -- does sharding change the answer?

ZeRO's whole premise is that it changes *where things are stored*, never *what
gets computed*. This script is the check on that claim: run the same global
batch through (a) a plain single-device full-batch step, and (b) each of
dp/zero1/zero2/zero3 split across 32 virtual GPUs, and compare the resulting
parameter vectors.
"""

import csv
from pathlib import Path

import common
import virtual_gpu
import zero_engine

RESULTS_DIR = Path(__file__).parent / "results"
N_GPUS = 32
GLOBAL_BATCH = 128


def single_device_reference_step(x, y, seed):
    model = common.build_model(seed=seed)
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    p = common.flat_param_vector(model)
    g = common.flat_grad_vector(model)
    updated, _, _ = zero_engine._adam_update(p, g)
    return updated, float(loss.item())


def main():
    x, y = common.synthetic_batch(GLOBAL_BATCH, common.DEMO_CONFIG.block_size, common.DEMO_CONFIG.vocab_size, seed=1)
    ref_updated, ref_loss = single_device_reference_step(x, y, seed=0)

    rows = []
    per_stage_updated = {}
    for stage in zero_engine.STAGES:
        gpus, ref_flat = virtual_gpu.make_cluster(N_GPUS, common.build_model, seed=0)
        result = zero_engine.run_step(stage, gpus, ref_flat, x, y)
        per_stage_updated[stage] = result.updated_params
        vs_single_device = (result.updated_params - ref_updated).abs().max().item()
        rows.append({
            "stage": stage,
            "max_abs_diff_vs_single_device": vs_single_device,
            "mean_loss": result.mean_loss,
        })

    # cross-check every ZeRO stage against every other -- should be exactly 0
    stages = list(zero_engine.STAGES)
    cross_rows = []
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            d = (per_stage_updated[stages[i]] - per_stage_updated[stages[j]]).abs().max().item()
            cross_rows.append({"stage_pair": f"{stages[i]}_vs_{stages[j]}", "max_abs_diff": d})

    out_csv = RESULTS_DIR / "logs" / "q1_correctness.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "max_abs_diff_vs_single_device", "mean_loss"])
        writer.writeheader()
        writer.writerows(rows)

    out_csv2 = RESULTS_DIR / "logs" / "q1_cross_check.csv"
    with open(out_csv2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage_pair", "max_abs_diff"])
        writer.writeheader()
        writer.writerows(cross_rows)

    print(f"N={N_GPUS} virtual GPUs, single-device reference loss: {ref_loss:.6f}")
    print(f"{'stage':>12} {'max|diff| vs single-device':>28} {'mean_loss':>12}")
    for row in rows:
        print(f"{row['stage']:>12} {row['max_abs_diff_vs_single_device']:>28.3e} {row['mean_loss']:>12.6f}")

    print("\ncross-check between ZeRO stages (should be ~0, floating-point only):")
    for row in cross_rows:
        print(f"  {row['stage_pair']:>24} {row['max_abs_diff']:>12.3e}")

    print(f"\nwrote {out_csv}")
    print(f"wrote {out_csv2}")


if __name__ == "__main__":
    main()
