"""GPU memory and throughput probes (the analogues of the paper's Table 3,
Figure 3 and Table 4), run on random tokens so they need no data.

  max         largest batch that survives a full training step (forward,
              backward, AdamW step, fp16 GradScaler) for each variant x loss
              head; exponential search then bisection. Every attempt is logged
              with its peak memory, which doubles as a memory-vs-batch curve.
  throughput  tokens/s for each variant at powers of two up to its max batch.
  depth       peak memory vs n_layer at a fixed batch (Figure 3).

  python experiments/assignment13/memory_probe.py --modes max,throughput,depth \
      --variants baseline,midpoint,leapfrog,hamiltonian,midpoint_a
  python experiments/assignment13/memory_probe.py --modes depth --variants baseline,midpoint rev.h=0.5

Extra key=value arguments are A13Config overrides, as in train.py. Writes
results/logs/{max_batch,memory_vs_batch,throughput_vs_batch,memory_vs_depth}.csv
(existing files are replaced only for the modes that ran).
"""

import argparse
import csv
import gc
import sys
import time

import torch

from chintana.configuration import load_config

import common

LOG_DIR = common.RESULTS_DIR / "logs"
MB = 2**20


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e)


class Trainer:
    """One model + optimizer kept alive across batch sizes, so each attempt only
    pays for activations; AdamW state is allocated up front by a tiny step."""

    def __init__(self, cfg, device, dtype, **model_overrides):
        torch.manual_seed(0)
        self.cfg, self.device = cfg, device
        self.model = common.build_model(cfg, **model_overrides).to(device)
        self.T = self.model.config.block_size
        self.V = self.model.config.vocab_size
        self.opt = common.build_optimizer(self.model, cfg.optim, cfg.optim.lr, device)
        self.dtype = dtype
        self.scaler = common.make_scaler(device, dtype)
        self.amp = common.amp_context(device, dtype)
        self.step(1)

    def step(self, batch):
        x, y = common.random_batch(batch, self.T, self.V, self.device)
        common.train_step(self.model, self.opt, self.scaler, x, y, self.amp, self.cfg.optim.grad_clip)

    def try_batch(self, batch):
        """(fits, peak MB) for one training step at this batch size."""
        self.opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        try:
            self.step(batch)
            torch.cuda.synchronize()
            ok = True
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            ok = False
        peak = torch.cuda.max_memory_allocated() / MB
        if not ok:
            # an OOM between unscale_() and update() would leave the scaler mid-step
            self.scaler = common.make_scaler(self.device, self.dtype)
            self.opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        return ok, peak

    def tokens_per_s(self, batch, steps=5, warmup=2):
        for _ in range(warmup):
            self.step(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            self.step(batch)
        torch.cuda.synchronize()
        return batch * self.T * steps / (time.perf_counter() - t0)

    def close(self):
        del self.model, self.opt
        gc.collect()
        torch.cuda.empty_cache()


def search_max(trainer, start, cap, label, trials):
    def attempt(b):
        ok, peak = trainer.try_batch(b)
        trials.append({**label, 'batch_size': b, 'fits': ok, 'peak_mb': round(peak, 1)})
        print(f"  {label['variant']:12s} {label['loss_impl']:8s} B={b:5d} {'ok ' if ok else 'OOM'} peak {peak:8.0f} MB",
              flush=True)
        return ok

    lo, hi = 0, None
    b = start
    while b <= cap:
        if not attempt(b):
            hi = b
            break
        lo, b = b, b * 2
    if hi is not None:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if attempt(mid):
                lo = mid
            else:
                hi = mid
    peak = next((t['peak_mb'] for t in reversed(trials) if t['fits'] and t['batch_size'] == lo), float('nan'))
    return lo, hi is None, peak


def write(name, rows):
    if not rows:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / name
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='max,throughput,depth')
    ap.add_argument('--variants', default='baseline,midpoint,midpoint_a,leapfrog,hamiltonian')
    ap.add_argument('--loss-impls', default='fused,standard', help="loss heads for the max-batch search")
    ap.add_argument('--start', type=int, default=8)
    ap.add_argument('--cap', type=int, default=8192)
    ap.add_argument('--depths', default='2,4,8,12,16,24,32,48,64')
    ap.add_argument('--depth-batch', type=int, default=32)
    ap.add_argument('--depth-variants', default=None, help="default: --variants")
    args, overrides = ap.parse_known_args(argv)
    cfg = load_config(common.A13Config, overrides)

    assert torch.cuda.is_available(), "memory_probe.py measures CUDA memory; run it on a GPU"
    device = 'cuda'
    dtype = common.resolve_dtype(cfg.run.dtype, device)
    modes = set(args.modes.split(','))
    variants = args.variants.split(',')
    gpu = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory / MB
    print(f"{gpu}, {total_mb:.0f} MB, dtype {dtype}, model {cfg.model}")

    max_rows, trials, tput_rows, depth_rows = [], [], [], []
    for variant in variants:
        cfg.rev.variant = variant
        if modes & {'max', 'throughput'}:
            for loss_impl in args.loss_impls.split(','):
                cfg.loss.impl = loss_impl
                trainer = Trainer(cfg, device, dtype)
                label = {'variant': variant, 'loss_impl': loss_impl}
                best, capped, peak = search_max(trainer, args.start, args.cap, label, trials)
                max_rows.append({'gpu': gpu, 'gpu_total_mb': round(total_mb), **label, 'max_batch': best,
                                 'hit_cap': capped, 'peak_mb_at_max': peak})
                if 'throughput' in modes and loss_impl == 'fused' and best > 0:
                    sizes = sorted({b for b in (2**k for k in range(3, 14)) if b <= best} | {best})
                    for b in sizes:
                        try:
                            tps = trainer.tokens_per_s(b)
                        except RuntimeError as e:
                            if not _is_oom(e):
                                raise
                            print(f"  {variant:12s} B={b:5d} OOM when repeated (fragmentation), skipped")
                            trainer.opt.zero_grad(set_to_none=True)
                            trainer.scaler = common.make_scaler(device, dtype)
                            torch.cuda.empty_cache()
                            continue
                        tput_rows.append({'gpu': gpu, 'variant': variant, 'batch_size': b,
                                          'tokens_per_s': round(tps)})
                        print(f"  {variant:12s} B={b:5d} {tps:10,.0f} tok/s", flush=True)
                trainer.close()
        cfg.loss.impl = 'fused'

    depth_variants = (args.depth_variants or args.variants).split(',')
    if 'depth' in modes:
        for variant in depth_variants:
            cfg.rev.variant = variant
            for n_layer in map(int, args.depths.split(',')):
                try:
                    trainer = Trainer(cfg, device, dtype, n_layer=n_layer)
                    ok, peak = trainer.try_batch(args.depth_batch)
                    trainer.close()
                except RuntimeError as e:
                    if not _is_oom(e):
                        raise
                    ok, peak = False, float('nan')
                    gc.collect()
                    torch.cuda.empty_cache()
                depth_rows.append({'gpu': gpu, 'variant': variant, 'n_layer': n_layer,
                                   'batch_size': args.depth_batch, 'fits': ok, 'peak_mb': round(peak, 1)})
                print(f"  depth {variant:12s} L={n_layer:3d} {'ok ' if ok else 'OOM'} peak {peak:8.0f} MB", flush=True)

    write('max_batch.csv', max_rows)
    write('memory_vs_batch.csv', trials)
    write('throughput_vs_batch.csv', tput_rows)
    write('memory_vs_depth.csv', depth_rows)


if __name__ == '__main__':
    main(sys.argv[1:])
