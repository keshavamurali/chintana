"""Train one dense or MoE model for a fixed number of steps.

Writes <run.out_dir>/<run.name>/log.jsonl (one line per logged step / eval),
summary.json (the numbers report.py tabulates) and ckpt.pt. Examples:

  # 1. the dense ("Linear" FFN) model
  python experiments/assignment14/train.py experiments/assignment14/config/dense.yaml
  # 2. convert it into an MoE and keep training
  python experiments/assignment14/train.py experiments/assignment14/config/moe_upcycle.yaml
  # any knob from common.A14Config / model.MoEConfig can be overridden:
  python experiments/assignment14/train.py experiments/assignment14/config/moe_upcycle.yaml \
      moe.n_experts=4 moe.top_k=1 moe.moe_every=2 run.name=moe_e4_k1

Tokens/s counts each training step's wall-clock (batch sampling, forward,
backward, optimizer) after run.speed_warmup_steps, and excludes evals.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from chintana.configuration import load_config

import common


def sample_text(model, device, n_tokens, prompt, encode, decode):
    idx = torch.tensor([encode(prompt)], device=device)
    torch.manual_seed(0)
    model.eval()
    with torch.no_grad():
        out = model.generate(idx, n_tokens, temperature=0.8, top_k=50)
    model.train()
    return decode(out[0].tolist())


def main(argv=None):
    cfg = load_config(common.A14Config, argv)
    vocab_size, encode, decode = common.load_codec(cfg.data.dir)
    if vocab_size is not None:
        cfg.model.vocab_size = vocab_size
    device = common.resolve_device(cfg.run.device)

    out_dir = Path(cfg.run.out_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / 'log.jsonl', 'w')

    def log(rec):
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()

    model = common.build_model(cfg).to(device)
    counts = model.param_counts()
    B, T, steps = cfg.data.batch_size, cfg.model.block_size, cfg.train.steps
    tokens_per_step = B * T
    optimizer = common.build_optimizer(model, cfg.train)

    data = common.TokenData(cfg.data.dir, T)
    val = data.val_batches(cfg.run.eval_batches, cfg.run.eval_batch_size, device)
    gen = torch.Generator().manual_seed(cfg.run.seed)

    kind = f"MoE E={cfg.moe.n_experts} k={cfg.moe.top_k} layers={model.moe_layers}" if model.moe_layers else "dense"
    print(f"[{cfg.run.name}] {kind} init={cfg.init.mode} params total={counts['total'] / 1e6:.3f}M "
          f"active={counts['active'] / 1e6:.3f}M B={B} T={T} steps={steps} lr={cfg.train.lr:.1e} device={device}")

    def run_eval(step):
        stats = common.evaluate(model, val)
        log({'type': 'eval', 'step': step, 'tokens': step * tokens_per_step, **stats})
        extra = f" | aux {stats['aux_loss']:.3f}" if 'aux_loss' in stats else ''
        print(f"step {step} | val {stats['val_loss']:.4f}{extra}")
        return stats

    first = last = run_eval(0)
    train_losses, timed_tokens, timed_seconds = [], 0, 0.0
    t_start = time.time()
    for step in range(steps):
        for g in optimizer.param_groups:
            g['lr'] = common.lr_at(step, cfg.train)
        t0 = time.perf_counter()
        x, y = data.train_batch(B, gen, device)
        loss, aux, z = common.train_step(model, optimizer, x, y, cfg.train.grad_clip)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= cfg.run.speed_warmup_steps:
            timed_tokens += tokens_per_step
            timed_seconds += dt
        train_losses.append(loss)
        if not math.isfinite(loss):
            print(f"step {step + 1}: non-finite loss, stopping")
            break

        done = step + 1
        if done % cfg.run.log_interval == 0 or done == steps:
            rec = {'type': 'train', 'step': done, 'tokens': done * tokens_per_step, 'loss': loss,
                   'aux_loss': aux, 'z_loss': z, 'lr': optimizer.param_groups[0]['lr'],
                   'tok_per_s': tokens_per_step / dt}
            log(rec)
            print(f"step {done}/{steps} | loss {loss:.4f} | aux {aux:.3f} | {rec['tok_per_s']:,.0f} tok/s")
        if done % cfg.run.eval_interval == 0 or done == steps:
            last = run_eval(done)
    wall = time.time() - t_start

    tail = train_losses[-max(1, min(50, len(train_losses) // 10)):]
    summary = {
        'name': cfg.run.name,
        'init': cfg.init.mode,
        'from_ckpt': cfg.init.from_ckpt,
        'moe_layers': model.moe_layers,
        'n_experts': cfg.moe.n_experts if model.moe_layers else 0,
        'top_k': cfg.moe.top_k if model.moe_layers else 0,
        'params_total': counts['total'],
        'params_active': counts['active'],
        'batch_size': B,
        'block_size': T,
        'steps': len(train_losses),
        'tokens': len(train_losses) * tokens_per_step,
        'peak_lr': cfg.train.lr,
        'initial_val_loss': first['val_loss'],
        'final_train_loss': sum(tail) / len(tail),
        'final_val_loss': last['val_loss'],
        'tokens_per_s': timed_tokens / timed_seconds if timed_seconds else float('nan'),
        'wall_s': wall,
        'device': device,
    }
    for k in ('expert_frac', 'aux_loss', 'drop_frac'):
        if k in last:
            summary[f'final_{k}'] = last[k]
    if cfg.run.sample_tokens > 0:
        summary['sample_prompt'] = cfg.run.sample_prompt
        summary['sample'] = sample_text(model, device, cfg.run.sample_tokens, cfg.run.sample_prompt,
                                        encode, decode)
    summary['config'] = OmegaConf.to_container(OmegaConf.structured(cfg))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    if cfg.run.save_checkpoint:
        common.save_checkpoint(out_dir / 'ckpt.pt', model, len(train_losses))
    log_f.close()

    keys = ('initial_val_loss', 'final_train_loss', 'final_val_loss', 'tokens_per_s', 'wall_s')
    print(json.dumps({k: summary.get(k) for k in keys}, indent=2))
    return summary


if __name__ == '__main__':
    main(sys.argv[1:])
