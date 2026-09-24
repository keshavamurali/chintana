"""Run the whole assignment end to end: the script version of assignment13_colab.ipynb.

Stages, in order (each one calls the same script the notebook calls):

  data      prepare_data.py             ~56M FineWeb-Edu tokens
  check     check_reversible.py         reversible grads == autograd grads
  screen    train.py x 5                10M-token runs of every trunk; picks the variant
  probe     memory_probe.py             max batch, memory vs depth, throughput vs batch
                                        (cpu_probe.py with --preset shakespeare_char)
  run1      train.py                    baseline, batch 64, 50M tokens
  run2      train.py                    chosen reversible variant, batch 64, 50M tokens
  run3      train.py                    chosen reversible variant at its max batch, 50M tokens
  run4      train.py                    (optional) baseline at its max batch
  report    report.py                   results/summary.md + results/plots/

  python experiments/assignment13/run_all.py                          # everything except run4
  python experiments/assignment13/run_all.py --stages run1,run2,report --chosen leapfrog
  python experiments/assignment13/run_all.py --stages all --with-run4
  python experiments/assignment13/run_all.py --preset shakespeare_char   # small, CPU-friendly (~1.5 h)

--preset shakespeare_char trains on the repo's data/shakespeare_char (already
tokenized, 65-char vocab, 7.2M-param model) with 2M-token runs, batch 16, and
0.5M-token screening. It skips 'data', runs cpu_probe.py as its 'probe' stage
(memory_probe.py needs CUDA), and run3 uses --run3-batch (default 64 for this
preset) instead of a probed max.

--chosen skips the automatic pick from the screening runs, and --run3-batch the
one from max_batch.csv; otherwise both are read from results written by the
earlier stages, so later stages can be re-run on their own.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
STAGES = ["data", "check", "screen", "probe", "run1", "run2", "run3", "run4", "report"]
VARIANTS = ["baseline", "midpoint", "midpoint_a", "leapfrog", "hamiltonian"]


def sh(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def pick_variant():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("runs/screen_*/summary.json"))]
    for r in rows:
        print(f"  {r['variant']:12s} val {r['final_val_loss']:.4f}  {r['tokens_per_s']:>9,.0f} tok/s  "
              f"diverged={r['diverged']}")
    ok = [r for r in rows if r["variant"] != "baseline" and not r["diverged"]]
    if not ok:
        raise SystemExit("no finished, non-diverged reversible screening run; run --stages screen or pass --chosen")
    return min(ok, key=lambda r: r["final_val_loss"])["variant"]


def max_batch(variant, headroom):
    path = RESULTS / "logs" / "max_batch.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing; run --stages probe or pass --run3-batch")
    for r in csv.DictReader(open(path)):
        if r["variant"] == variant and r["loss_impl"] == "fused":
            # the probe measures one step; a long run fragments the allocator a little
            return max(8, int(int(r["max_batch"]) * headroom) // 8 * 8)
    raise SystemExit(f"no fused max-batch row for {variant} in {path}; re-run the probe with it")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all", help=f"comma list of {STAGES}, or 'all'")
    ap.add_argument("--with-run4", action="store_true", help="include run4 when --stages all")
    ap.add_argument("--preset", default="fineweb", choices=["fineweb", "shakespeare_char"])
    ap.add_argument("--source", default="fineweb_edu", choices=["fineweb_edu", "tinystories"])
    ap.add_argument("--screen-tokens", type=int, default=None, help="default 10M (fineweb) / 0.5M (shakespeare_char)")
    ap.add_argument("--batch", type=int, default=None, help="fixed batch for screening and runs 1-2; default 64 / 16")
    ap.add_argument("--chosen", default=None, choices=VARIANTS[1:])
    ap.add_argument("--run3-batch", type=int, default=None)
    ap.add_argument("--headroom", type=float, default=0.95, help="fraction of the probed max batch to use")
    args = ap.parse_args()

    small = args.preset == "shakespeare_char"
    args.batch = args.batch or (16 if small else 64)
    args.screen_tokens = args.screen_tokens or (500_000 if small else 10_000_000)
    if small and args.run3_batch is None:
        args.run3_batch = 64
    # appended after each run's own YAML, so its data/model/eval settings win over the 50M-token defaults
    preset = [HERE / "config" / "shakespeare_char.yaml"] if small else []

    skip = {"data"} if small else set()
    stages = ([s for s in STAGES if (s != "run4" or args.with_run4) and s not in skip] if args.stages == "all"
              else args.stages.split(","))
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages {sorted(unknown)}")

    chosen = args.chosen
    cfg = HERE / "config"

    def need_chosen():
        nonlocal chosen
        if chosen is None:
            print("picking the reversible variant from the screening runs:")
            chosen = pick_variant()
            print(f"  -> {chosen}")
        return chosen

    for stage in stages:
        print(f"\n===== {stage} =====", flush=True)
        if stage == "data":
            sh("prepare_data.py", "--source", args.source, "--train-tokens", 55_000_000, "--val-tokens", 1_000_000)
        elif stage == "check":
            sh("check_reversible.py")
        elif stage == "screen":
            for v in VARIANTS:
                sh("train.py", *preset, f"rev.variant={v}", f"run.name=screen_{v}", f"data.batch_size={args.batch}",
                   f"data.total_tokens={args.screen_tokens}", "run.log_interval=50", "run.n_evals=5")
        elif stage == "probe" and small:
            probe = ["cpu_probe.py", preset[0]]
            sh(*probe, "--modes", "memory", "--variants", ",".join(VARIANTS), "--batches", "16,64,128")
            sh(*probe, "--modes", "depth", "--variants", f"baseline,{need_chosen()}")
            sh(*probe, "--modes", "speed", "--variants", f"baseline,{need_chosen()}", "--speed-batches",
               f"{args.batch},{args.run3_batch}")
        elif stage == "probe":
            sh("memory_probe.py", "--modes", "max,throughput,depth", "--variants", f"baseline,{need_chosen()}",
               "--loss-impls", "fused,standard")
        elif stage == "run1":
            sh("train.py", cfg / "run1_baseline.yaml", *preset, f"data.batch_size={args.batch}", "run.log_interval=50")
        elif stage == "run2":
            sh("train.py", cfg / "run2_reversible.yaml", *preset, f"rev.variant={need_chosen()}",
               f"data.batch_size={args.batch}", "run.log_interval=50")
        elif stage == "run3":
            b = args.run3_batch or max_batch(need_chosen(), args.headroom)
            sh("train.py", cfg / "run3_reversible_maxbatch.yaml", *preset, f"rev.variant={need_chosen()}",
               f"data.batch_size={b}", f"optim.ref_batch_size={args.batch}", "run.log_interval=5")
        elif stage == "run4":
            b = args.run3_batch if small else max_batch("baseline", args.headroom)
            sh("train.py", cfg / "run1_baseline.yaml", *preset, "run.name=run4_baseline_maxbatch", f"data.batch_size={b}",
               "optim.lr_scaling=sqrt", f"optim.ref_batch_size={args.batch}", "run.log_interval=10")
        elif stage == "report":
            sh("report.py")

    print(f"\ndone. results: {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
