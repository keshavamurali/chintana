"""Q3 -- how does computation/communication change across dp/zero1/zero2/zero3?

Two different things are measured here, and the README is explicit that they
answer different questions:
  - communication volume: bytes actually moved through the simulated
    collectives (collectives.py), summed across the whole 32-GPU cluster.
    This is the number that generalizes to real hardware.
  - wall-clock time: how long this one CPU actually took running 32 Python
    threads. This mostly reflects GIL/thread-pool overhead, not real
    multi-accelerator speedup -- there is no 32-way parallel hardware here,
    just 32 threads timesharing one CPU. Reported for completeness, not as a
    performance claim.
"""

import csv
import statistics
from pathlib import Path

import common
import plotting
import virtual_gpu
import zero_engine

RESULTS_DIR = Path(__file__).parent / "results"
N_GPUS = 32
GLOBAL_BATCH = 128
N_REPEATS = 5  # wall-clock timing is noisy on a shared CPU; repeat and take the median


def main():
    x, y = common.synthetic_batch(GLOBAL_BATCH, common.DEMO_CONFIG.block_size, common.DEMO_CONFIG.vocab_size, seed=1)

    comm_by_stage = {}
    time_rows = []
    for stage in zero_engine.STAGES:
        times = []
        events = None
        for rep in range(N_REPEATS):
            gpus, ref_flat = virtual_gpu.make_cluster(N_GPUS, common.build_model, seed=0)
            result = zero_engine.run_step(stage, gpus, ref_flat, x, y)
            times.append(result.wall_time_sec)
            events = result.comm_events  # identical every rep, deterministic given the stage
            time_rows.append({"stage": stage, "rep": rep, "wall_time_sec": result.wall_time_sec})
        comm_by_stage[stage] = events
        print(f"{stage:>6}: median wall time {statistics.median(times):.3f}s "
              f"(min {min(times):.3f}s, max {max(times):.3f}s), "
              f"comm {sum(b for _, b in events)/1e6:.2f}MB")

    out_csv = RESULTS_DIR / "logs" / "q3_wall_time.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "rep", "wall_time_sec"])
        writer.writeheader()
        writer.writerows(time_rows)
    print(f"wrote {out_csv}")

    comm_csv_rows = []
    for stage, events in comm_by_stage.items():
        for op, b in events:
            comm_csv_rows.append({"stage": stage, "op": op, "bytes": b})
    out_csv2 = RESULTS_DIR / "logs" / "q3_communication.csv"
    with open(out_csv2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "op", "bytes"])
        writer.writeheader()
        writer.writerows(comm_csv_rows)
    print(f"wrote {out_csv2}")

    plotting.stacked_comm_bar(
        comm_by_stage, list(zero_engine.STAGES), zero_engine.STAGE_LABELS,
        RESULTS_DIR / "plots" / "q3_communication.png",
        f"Communication volume per training step (N={N_GPUS}, whole cluster)",
    )

    median_times = {
        stage: statistics.median([r["wall_time_sec"] for r in time_rows if r["stage"] == stage])
        for stage in zero_engine.STAGES
    }
    plotting.stage_bar(
        median_times, list(zero_engine.STAGES), zero_engine.STAGE_LABELS,
        RESULTS_DIR / "plots" / "q3_wall_time.png",
        f"Median wall-clock time per step (N={N_GPUS}, {N_REPEATS} reps)",
        "seconds", fmt=lambda v: f"{v:.2f}s",
    )

    total_comm = {stage: sum(b for _, b in events) for stage, events in comm_by_stage.items()}
    baseline = total_comm["dp"]
    print("\ncommunication volume relative to baseline DP:")
    for stage in zero_engine.STAGES:
        print(f"  {stage:>6}: {total_comm[stage] / baseline:.3f}x")


if __name__ == "__main__":
    main()
