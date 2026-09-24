"""Train one RevGPT for a fixed token budget and record loss, speed and memory.

Writes <run.out_dir>/<run.name>/log.jsonl (one line per logged step / eval)
and summary.json (the numbers report.py tabulates). Examples:

  python experiments/assignment13/train.py experiments/assignment13/config/run1_baseline.yaml
  python experiments/assignment13/train.py experiments/assignment13/config/run2_reversible.yaml rev.variant=midpoint
  python experiments/assignment13/train.py experiments/assignment13/config/run3_reversible_maxbatch.yaml \
      rev.variant=midpoint data.batch_size=1024

Tokens/s counts every training step's wall-clock (batch sampling, forward,
backward, optimizer) after run.speed_warmup_steps, and excludes evals.
Peak memory is torch.cuda.max_memory_allocated over the whole run, so it
includes parameters, gradients and AdamW state as well as activations.
"""

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from chintana.configuration import load_config

import common


def evaluate(model, batches, amp):
    model.eval()
    with torch.no_grad():
        losses = []
        for x, y in batches:
            with amp():
                losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def sample_text(model, device, amp, n_tokens, prompt, encode, decode):
    idx = torch.tensor([encode(prompt)], device=device)
    torch.manual_seed(0)
    model.eval()
    with amp():
        out = model.generate(idx, n_tokens, temperature=0.8, top_k=50)
    model.train()
    return decode(out[0].tolist())


def main(argv=None):
    cfg = load_config(common.A13Config, argv)
    vocab_size, encode, decode = common.load_codec(cfg.data.dir)
    if vocab_size is not None and vocab_size != cfg.model.vocab_size:
        print(f"{cfg.data.dir} has a {vocab_size}-symbol vocabulary; setting model.vocab_size={vocab_size}")
        cfg.model.vocab_size = vocab_size
    device = common.resolve_device(cfg.run.device)
    dtype = common.resolve_dtype(cfg.run.dtype, device)
    cuda = device.startswith('cuda')
    amp = common.amp_context(device, dtype)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.run.out_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / 'log.jsonl', 'w')

    def log(rec):
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()

    torch.manual_seed(cfg.run.seed)
    model = common.build_model(cfg).to(device)
    n_params = common.count_params(model)
    B, T = cfg.data.batch_size, cfg.model.block_size
    tokens_per_step = B * T
    steps = math.ceil(cfg.data.total_tokens / tokens_per_step)
    lr = common.peak_lr(cfg.optim, B)
    min_lr = lr * cfg.optim.min_lr_frac
    warmup = min(steps, max(cfg.optim.min_warmup_steps, int(cfg.optim.warmup_frac * steps)))
    optimizer = common.build_optimizer(model, cfg.optim, lr, device)
    scaler = common.make_scaler(device, dtype)

    data = common.TokenData(cfg.data.dir, T)
    val = data.val_batches(cfg.run.eval_batches, cfg.run.eval_batch_size, device)
    gen = torch.Generator().manual_seed(cfg.run.seed)
    eval_at = {round(i * steps / cfg.run.n_evals) for i in range(1, cfg.run.n_evals + 1)}

    print(f"[{cfg.run.name}] variant={cfg.rev.variant} params={n_params['total'] / 1e6:.2f}M "
          f"B={B} T={T} steps={steps} lr={lr:.2e} warmup={warmup} device={device} dtype={dtype}")
    if getattr(model.rule, 'coeffs', None) is not None:
        print(f"  (alpha, beta, gamma) per layer: {model.rule.coeffs}")

    def run_eval(step):
        loss = evaluate(model, val, amp)
        log({'type': 'eval', 'step': step, 'tokens': step * tokens_per_step, 'val_loss': loss})
        print(f"step {step} | val {loss:.4f}")
        return loss

    val_loss, last_eval = run_eval(0), 0

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated()

    train_losses, timed_tokens, timed_seconds = [], 0, 0.0
    diverged, nan_streak = False, 0
    t_start = time.time()
    for step in range(steps):
        for g in optimizer.param_groups:
            g['lr'] = common.lr_at(step, steps, warmup, lr, min_lr)
        t0 = time.perf_counter()
        x, y = data.train_batch(B, gen, device)
        loss = common.train_step(model, optimizer, scaler, x, y, amp, cfg.optim.grad_clip).item()
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= cfg.run.speed_warmup_steps:
            timed_tokens += tokens_per_step
            timed_seconds += dt
        train_losses.append(loss)

        nan_streak = nan_streak + 1 if not math.isfinite(loss) else 0
        if nan_streak >= 10:
            diverged = True
            print(f"step {step + 1}: loss non-finite for 10 steps, stopping")
            break

        done = step + 1
        if done % cfg.run.log_interval == 0 or done == steps:
            rec = {'type': 'train', 'step': done, 'tokens': done * tokens_per_step, 'loss': loss,
                   'lr': optimizer.param_groups[0]['lr'], 'dt': dt, 'tok_per_s': tokens_per_step / dt}
            if cuda:
                rec['mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
            log(rec)
            print(f"step {done}/{steps} | loss {loss:.4f} | {rec['tok_per_s']:,.0f} tok/s | lr {rec['lr']:.2e}")
        if done in eval_at:
            val_loss, last_eval = run_eval(done), done
    wall = time.time() - t_start
    if last_eval != len(train_losses):  # stopped early: report the model as it ended, not an older eval
        val_loss = run_eval(len(train_losses))

    tail = train_losses[-max(1, min(50, len(train_losses) // 10)):]
    summary = {
        'name': cfg.run.name,
        'variant': cfg.rev.variant,
        'coeffs': getattr(model.rule, 'coeffs', None),
        'memory_saving': cfg.rev.memory_saving,
        'loss_impl': cfg.loss.impl,
        'params_total': n_params['total'],
        'params_non_embedding': n_params['non_embedding'],
        'batch_size': B,
        'block_size': T,
        'steps': len(train_losses),
        'tokens': len(train_losses) * tokens_per_step,
        'peak_lr': lr,
        'final_train_loss': sum(tail) / len(tail),
        'final_val_loss': val_loss,
        'tokens_per_s': timed_tokens / timed_seconds if timed_seconds else float('nan'),
        'wall_s': wall,
        'diverged': diverged,
        'device': torch.cuda.get_device_name(device) if cuda else 'cpu',
        'dtype': str(dtype).replace('torch.', ''),
    }
    if cuda:
        summary['mem_start_mb'] = mem_start / 2**20
        summary['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
        summary['peak_reserved_mb'] = torch.cuda.max_memory_reserved() / 2**20
        summary['gpu_total_mb'] = torch.cuda.get_device_properties(device).total_memory / 2**20
    if not diverged and cfg.run.sample_tokens > 0:
        summary['sample_prompt'] = cfg.run.sample_prompt
        summary['sample'] = sample_text(model, device, amp, cfg.run.sample_tokens, cfg.run.sample_prompt,
                                        encode, decode)
    summary['config'] = OmegaConf.to_container(OmegaConf.structured(cfg))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    if cfg.run.save_checkpoint:
        torch.save({'model': model.state_dict(), 'config': dataclasses.asdict(cfg)}, out_dir / 'ckpt.pt')
    log_f.close()

    keys = ('final_train_loss', 'final_val_loss', 'tokens_per_s', 'peak_mem_mb', 'wall_s')
    print(json.dumps({k: summary.get(k) for k in keys}, indent=2))
    if 'sample' in summary:
        print(summary['sample'])


if __name__ == '__main__':
    main(sys.argv[1:])
