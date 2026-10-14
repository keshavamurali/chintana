"""Run the whole assignment end to end: the script version of assignment13_colab.ipynb.

Stages, in order (each one calls the same script the notebook calls):

  data      prepare_data.py             ~56M FineWeb-Edu tokens
  check     check_reversible.py         reversible grads == autograd grads
  screen    train.py x 5                5M-token runs of every trunk; picks the variant
  probe     memory_probe.py             max batch, memory vs depth, throughput vs batch
                                        (cpu_probe.py with --preset shakespeare_char)
  run1      train.py                    baseline, batch 64, 50M tokens
  run2      train.py                    chosen reversible variant, batch 64, 50M tokens
  run3      train.py                    chosen reversible variant at its sustained max batch, 50M tokens
  run4      train.py                    baseline at its sustained max batch, 50M tokens
  report    report.py                   results/summary.md + results/plots/

  python experiments/assignment13/run_all.py                          # everything
  python experiments/assignment13/run_all.py --stages run1,run2,report --chosen leapfrog
  python experiments/assignment13/run_all.py --preset shakespeare_char   # small, CPU-friendly (~1.5 h)

Sustained max batch (run3, run4): memory_probe.py finds the largest batch that
survives one training step. A 50M-token run then starts at exactly that batch;
if it runs out of GPU memory at any point (train.py exits with status 75), it
is restarted from scratch at 5% smaller, until a whole run completes. Every
attempt is logged to results/logs/sustained_batch.csv, so the reported batch
is the largest one that sustained a full run, not a one-step estimate.

Resuming: a training stage whose results/runs/<name>/summary.json already
exists is skipped (and 'data' is skipped when train.bin exists), so after a
Colab disconnect the same command picks up where it stopped. --force redoes
everything. Set A13_RESULTS_DIR to write results somewhere else (e.g. Drive).

--preset shakespeare_char trains on the repo's data/shakespeare_char (65-char
vocab, 7.2M-param model) with 2M-token runs, batch 16, and 0.5M-token
screening. Its 'data' stage runs data/shakespeare_char/prepare.py if the
(git-ignored) .bin files are missing, its 'probe' stage runs cpu_probe.py
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
RESULTS = Path(os.environ.get("A13_RESULTS_DIR", HERE / "results"))
OOM_EXIT = 75  # train.py's exit status when a batch does not fit
STAGES = ["data", "check", "screen", "probe", "run1", "run2", "run3", "run4", "report"]
VARIANTS = ["baseline", "midpoint", "midpoint_a", "leapfrog", "hamiltonian"]


def sh(script, *args, allowed=()):
    """Run one of the stage scripts; raises on failure unless the exit status is in `allowed`."""
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rc = subprocess.run(cmd, cwd=REPO, env=env).returncode
    if rc != 0 and rc not in allowed:
        raise SystemExit(f"{script} failed with exit status {rc}")
    return rc


def finished(name):
    return (RESULTS / "runs" / name / "summary.json").exists()


def train_sustained(name, variant, start, args):
    """Train at `start`, stepping the batch down 5% after each out-of-memory exit."""
    path = RESULTS / "logs" / "sustained_batch.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    b = start
    while b >= 8:
        rc = sh("train.py", *args, f"run.name={name}", f"data.batch_size={b}", allowed=(OOM_EXIT,))
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run", "variant", "probe_max_batch", "batch_size", "outcome"])
            w.writerow([name, variant, start, b, "completed" if rc == 0 else "out_of_memory"])
        if rc == 0:
            return b
        print(f"{name}: out of memory at batch {b}; retrying 5% smaller", flush=True)
        b = int(b * 0.95) // 8 * 8
    raise SystemExit(f"{name}: no batch >= 8 completed")


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
            return max(8, int(int(r["max_batch"]) * headroom))
    raise SystemExit(f"no fused max-batch row for {variant} in {path}; re-run the probe with it")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all", help=f"comma list of {STAGES}, or 'all'")
    ap.add_argument("--skip-run4", action="store_true", help="leave run4 out of --stages all")
    ap.add_argument("--force", action="store_true", help="redo stages whose results already exist")
    ap.add_argument("--preset", default="fineweb", choices=["fineweb", "shakespeare_char"])
    ap.add_argument("--source", default="fineweb_edu", choices=["fineweb_edu", "tinystories"])
    ap.add_argument("--screen-tokens", type=int, default=None, help="default 5M (fineweb) / 0.5M (shakespeare_char)")
    ap.add_argument("--batch", type=int, default=None, help="fixed batch for screening and runs 1-2; default 64 / 16")
    ap.add_argument("--chosen", default=None, choices=VARIANTS[1:])
    ap.add_argument("--run3-batch", type=int, default=None)
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="fraction of the probed max batch to start the sustained search from")
    args = ap.parse_args()

    small = args.preset == "shakespeare_char"
    args.batch = args.batch or (16 if small else 64)
    args.screen_tokens = args.screen_tokens or (500_000 if small else 5_000_000)
    if small and args.run3_batch is None:
        args.run3_batch = 64
    # appended after each run's own YAML, so its data/model/eval settings win over the 50M-token defaults
    preset = [HERE / "config" / "shakespeare_char.yaml"] if small else []

    stages = ([s for s in STAGES if s != "run4" or not args.skip_run4] if args.stages == "all"
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

    def already_done(name):
        if finished(name) and not args.force:
            print(f"{name}: already finished ({RESULTS / 'runs' / name}), skipping; --force redoes it")
            return True
        return False

    for stage in stages:
        print(f"\n===== {stage} =====", flush=True)
        if stage == "data" and small:
            # the .bin files are git-ignored, so a fresh clone has to build them
            if (REPO / "data" / "shakespeare_char" / "train.bin").exists() and not args.force:
                print("data/shakespeare_char/train.bin exists, skipping")
                continue
            prep = REPO / "data" / "shakespeare_char" / "prepare.py"
            print(f"\n$ {sys.executable} {prep}", flush=True)
            subprocess.run([sys.executable, str(prep)], check=True, cwd=prep.parent)
        elif stage == "data":
            if (HERE / "data" / args.source / "train.bin").exists() and not args.force:
                print(f"data/{args.source}/train.bin exists, skipping; --force re-tokenizes")
                continue
            sh("prepare_data.py", "--source", args.source, "--train-tokens", 55_000_000, "--val-tokens", 1_000_000)
        elif stage == "check":
            sh("check_reversible.py")
        elif stage == "screen":
            for v in VARIANTS:
                if already_done(f"screen_{v}"):
                    continue
                sh("train.py", *preset, f"rev.variant={v}", f"run.name=screen_{v}", f"data.batch_size={args.batch}",
                   f"data.total_tokens={args.screen_tokens}", "run.log_interval=50", "run.n_evals=5")
        elif stage == "probe" and small:
            probe = ["cpu_probe.py", preset[0]]
            sh(*probe, "--modes", "memory", "--variants", ",".join(VARIANTS), "--batches", "16,64,128")
            sh(*probe, "--modes", "depth", "--variants", f"baseline,{need_chosen()}")
            sh(*probe, "--modes", "speed", "--variants", f"baseline,{need_chosen()}", "--speed-batches",
               f"{args.batch},{args.run3_batch}")
        elif stage == "probe":
            if (RESULTS / "logs" / "max_batch.csv").exists() and not args.force:
                print("logs/max_batch.csv exists, skipping the probe; --force re-probes")
                continue
            sh("memory_probe.py", "--modes", "max,throughput,depth", "--variants", f"baseline,{need_chosen()}",
               "--loss-impls", "fused,standard")
        elif stage == "run1":
            if already_done("run1_baseline"):
                continue
            sh("train.py", cfg / "run1_baseline.yaml", *preset, f"data.batch_size={args.batch}", "run.log_interval=50")
        elif stage == "run2":
            if already_done("run2_reversible"):
                continue
            sh("train.py", cfg / "run2_reversible.yaml", *preset, f"rev.variant={need_chosen()}",
               f"data.batch_size={args.batch}", "run.log_interval=50")
        elif stage == "run3":
            if already_done("run3_reversible_maxbatch"):
                continue
            b = args.run3_batch or max_batch(need_chosen(), args.headroom)
            train_sustained("run3_reversible_maxbatch", need_chosen(), b,
                            [cfg / "run3_reversible_maxbatch.yaml", *preset, f"rev.variant={need_chosen()}",
                             f"optim.ref_batch_size={args.batch}", "run.log_interval=5"])
        elif stage == "run4":
            if already_done("run4_baseline_maxbatch"):
                continue
            b = args.run3_batch if small else max_batch("baseline", args.headroom)
            train_sustained("run4_baseline_maxbatch", "baseline", b,
                            [cfg / "run1_baseline.yaml", *preset, "optim.lr_scaling=sqrt",
                             f"optim.ref_batch_size={args.batch}", "run.log_interval=10"])
        elif stage == "report":
            sh("report.py")

    print(f"\ndone. results: {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
