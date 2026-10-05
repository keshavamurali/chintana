"""The whole assignment pipeline: dense model -> MoE conversion -> comparison runs -> report.

  python experiments/assignment14/run_all.py                      # every stage
  python experiments/assignment14/run_all.py --stages dense upcycle report
  python experiments/assignment14/run_all.py --phase1-steps 500 --phase2-steps 300   # quick smoke run
  python experiments/assignment14/run_all.py --stages upcycle moe.n_experts=16 moe.top_k=1

Stages (in order):
  check           check_moe.py: dispatch and function-preserving conversion
  dense           phase 1: train the dense model (config/dense.yaml)
  dense_continue  phase 2 control: the same dense model, trained phase2 more steps
  upcycle         phase 2: convert the phase-1 dense model to an MoE, train phase2 steps
  moe_scratch     reference: the same MoE trained from random init for phase1 + phase2 steps
  dense_wide      reference: a dense model with the MoE's *active* parameter count
                  (one 8C-wide FFN = moe.n_experts=1, expert_hidden=8C), phase1 + phase2 steps
  sweep           phase 2 upcycling with other (n_experts, top_k, moe_every) settings
  report          report.py: results/summary.md and results/plots/*.png

Extra dotted key=value arguments are passed to every training run of the
selected stages (e.g. moe.n_experts=16 changes the upcycle and moe_scratch runs).
"""

import argparse

import check_moe
import report
import train
from common import HERE

CFG = HERE / "config"
RUNS = HERE / "results" / "runs"
STAGES = ['check', 'dense', 'dense_continue', 'upcycle', 'moe_scratch', 'dense_wide', 'sweep', 'report']

# name -> overrides on top of config/moe_upcycle.yaml
SWEEP = {
    'sweep_e4_k1': ['moe.n_experts=4', 'moe.top_k=1'],
    'sweep_e8_k1': ['moe.n_experts=8', 'moe.top_k=1'],
    'sweep_e16_k2': ['moe.n_experts=16', 'moe.top_k=2'],
    'sweep_e8_k2_every2': ['moe.n_experts=8', 'moe.top_k=2', 'moe.moe_every=2'],
}


def run(config, name, *overrides):
    train.main([str(CFG / config), f'run.name={name}', f'run.out_dir={RUNS}', *overrides])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    p.add_argument('--phase1-steps', type=int, default=2000)
    p.add_argument('--phase2-steps', type=int, default=1500)
    args, extra = p.parse_known_args()
    bad = [a for a in extra if '=' not in a]
    if bad:
        p.error(f"unrecognized arguments: {bad}")

    dense_ckpt = f'init.from_ckpt={RUNS / "dense" / "ckpt.pt"}'
    p1, p2 = f'train.steps={args.phase1_steps}', f'train.steps={args.phase2_steps}'
    both = f'train.steps={args.phase1_steps + args.phase2_steps}'

    for stage in args.stages:
        print(f"\n===== {stage} =====")
        if stage == 'check':
            assert check_moe.main(), "check_moe failed"
        elif stage == 'dense':
            run('dense.yaml', 'dense', p1, *extra)
        elif stage == 'dense_continue':
            run('dense_continue.yaml', 'dense_continue', dense_ckpt, p2, *extra)
        elif stage == 'upcycle':
            run('moe_upcycle.yaml', 'moe_upcycle', dense_ckpt, p2, *extra)
        elif stage == 'moe_scratch':
            run('moe_scratch.yaml', 'moe_scratch', both, *extra)
        elif stage == 'dense_wide':
            run('dense_wide.yaml', 'dense_wide', both, *extra)
        elif stage == 'sweep':
            for name, ov in SWEEP.items():
                run('moe_upcycle.yaml', name, dense_ckpt, p2, *ov, *extra)
        elif stage == 'report':
            report.main()


if __name__ == '__main__':
    main()
