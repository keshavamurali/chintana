"""Memory and speed measurements for CPU-only machines, where memory_probe.py
(which reads CUDA allocator stats) cannot run.

Two memory numbers per configuration, each from one training step (forward,
backward, AdamW) in a fresh subprocess:

  saved_mb   bytes of every tensor autograd holds for the backward pass,
             counted with torch.autograd.graph.saved_tensors_hooks (parameters
             excluded, shared storages counted once). Exact and deterministic:
             this is the activation memory that reversibility removes.
  peak_mb    growth of the process's peak resident set (getrusage ru_maxrss)
             across the step, measured after model/optimizer setup. It is the
             CPU analogue of torch.cuda.max_memory_allocated minus the static
             state, and it also sees the transient memory of the reversible
             backward's one-layer recomputation, which saved_mb does not.

Speed is measured in one process per repetition, alternating the variants
(A B A B ...) so that background load on a laptop hits all of them equally;
the median over repetitions is reported.

  uv run python experiments/assignment13/cpu_probe.py \
      experiments/assignment13/config/shakespeare_char.yaml --variants baseline,midpoint_a

Writes results/logs/cpu_memory.csv, cpu_memory_vs_depth.csv, cpu_speed.csv.
"""

import argparse
import csv
import json
import resource
import statistics
import subprocess
import sys
import time

import torch

from chintana.configuration import load_config

import common

LOG_DIR = common.RESULTS_DIR / "logs"
MB = 2**20


def _maxrss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # macOS reports bytes, Linux KiB


def measure(cfg, batch, n_layer, speed_steps):
    """Runs inside the child process; returns one JSON-able dict."""
    vocab, _, _ = common.load_codec(cfg.data.dir)
    if vocab is not None:
        cfg.model.vocab_size = vocab
    torch.manual_seed(0)
    model = common.build_model(cfg, n_layer=n_layer)
    opt = common.build_optimizer(model, cfg.optim, cfg.optim.lr, 'cpu')
    scaler = common.make_scaler('cpu', torch.float32)
    amp = common.amp_context('cpu', torch.float32)
    T, V = cfg.model.block_size, cfg.model.vocab_size

    def step(b):
        x, y = common.random_batch(b, T, V, 'cpu')
        common.train_step(model, opt, scaler, x, y, amp, cfg.optim.grad_clip)

    step(1)  # allocates AdamW state, so the measured step only adds activations
    # peak first: ru_maxrss is a high-water mark, so anything run before this
    # step at the same batch would hide its growth
    before = _maxrss_bytes()
    step(batch)
    peak = _maxrss_bytes() - before

    x, y = common.random_batch(batch, T, V, 'cpu')
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        if ptr not in param_ptrs:
            seen[ptr] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = model(x, y)
    saved = sum(seen.values())
    del loss
    opt.zero_grad(set_to_none=True)

    tps = None
    if speed_steps:
        step(batch)
        t0 = time.perf_counter()
        for _ in range(speed_steps):
            step(batch)
        tps = batch * T * speed_steps / (time.perf_counter() - t0)
    return {'saved_mb': saved / MB, 'peak_mb': peak / MB, 'tokens_per_s': tps}


def child(argv):
    spec = json.loads(argv[0])
    cfg = load_config(common.A13Config, spec['overrides'])
    print(json.dumps(measure(cfg, spec['batch'], spec['n_layer'], spec['speed_steps'])))


def run_child(overrides, batch, n_layer, speed_steps=0):
    spec = json.dumps({'overrides': overrides, 'batch': batch, 'n_layer': n_layer, 'speed_steps': speed_steps})
    out = subprocess.run([sys.executable, __file__, '--child', spec], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def write(name, rows):
    if not rows:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {LOG_DIR / name}")


def main(argv):
    if argv and argv[0] == '--child':
        return child(argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='baseline,midpoint_a')
    ap.add_argument('--batches', default='16,64,128')
    ap.add_argument('--depths', default='3,6,9,18,36')
    ap.add_argument('--depth-batch', type=int, default=16)
    ap.add_argument('--speed-batches', default='16,64')
    ap.add_argument('--speed-reps', type=int, default=3)
    ap.add_argument('--speed-steps', type=int, default=5)
    ap.add_argument('--modes', default='memory,depth,speed')
    args, overrides = ap.parse_known_args(argv)
    cfg = load_config(common.A13Config, overrides)
    variants = args.variants.split(',')
    modes = set(args.modes.split(','))
    L = cfg.model.n_layer

    mem, depth, speed = [], [], []
    if 'memory' in modes:
        for b in map(int, args.batches.split(',')):
            for v in variants:
                r = run_child(overrides + [f'rev.variant={v}'], b, L)
                mem.append({'variant': v, 'n_layer': L, 'batch_size': b, 'block_size': cfg.model.block_size,
                            'saved_mb': round(r['saved_mb'], 1), 'peak_mb': round(r['peak_mb'], 1)})
                print(f"  memory {v:12s} B={b:4d}  saved {r['saved_mb']:8.1f} MB  peak +{r['peak_mb']:8.1f} MB", flush=True)
    if 'depth' in modes:
        for n in map(int, args.depths.split(',')):
            for v in variants:
                r = run_child(overrides + [f'rev.variant={v}'], args.depth_batch, n)
                depth.append({'variant': v, 'n_layer': n, 'batch_size': args.depth_batch,
                              'saved_mb': round(r['saved_mb'], 1), 'peak_mb': round(r['peak_mb'], 1)})
                print(f"  depth  {v:12s} L={n:3d}  saved {r['saved_mb']:8.1f} MB  peak +{r['peak_mb']:8.1f} MB", flush=True)
    if 'speed' in modes:
        for b in map(int, args.speed_batches.split(',')):
            samples = {v: [] for v in variants}
            for rep in range(args.speed_reps):
                for v in variants:
                    tps = run_child(overrides + [f'rev.variant={v}'], b, L, args.speed_steps)['tokens_per_s']
                    samples[v].append(tps)
                    print(f"  speed  {v:12s} B={b:4d} rep {rep}  {tps:8,.0f} tok/s", flush=True)
            for v in variants:
                speed.append({'variant': v, 'batch_size': b, 'reps': args.speed_reps,
                              'median_tokens_per_s': round(statistics.median(samples[v])),
                              'min': round(min(samples[v])), 'max': round(max(samples[v]))})

    write('cpu_memory.csv', mem)
    write('cpu_memory_vs_depth.csv', depth)
    write('cpu_speed.csv', speed)


if __name__ == '__main__':
    main(sys.argv[1:])
