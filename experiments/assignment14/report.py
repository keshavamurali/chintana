"""Tabulate results/runs/*/summary.json into results/summary.md and draw results/plots/*.png.

  python experiments/assignment14/report.py
"""

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR  # noqa: E402

RUNS = RESULTS_DIR / "runs"
PLOTS = RESULTS_DIR / "plots"

# fixed categorical slots, one per model (never cycled)
COLORS = {'dense': '#2a78d6', 'dense_continue': '#2a78d6', 'moe_upcycle': '#eb6834',
          'moe_scratch': '#1baf7a', 'dense_wide': '#eda100'}
LABELS = {'dense': 'dense (phase 1)', 'dense_continue': 'dense, continued',
          'moe_upcycle': 'MoE, upcycled from dense', 'moe_scratch': 'MoE from scratch',
          'dense_wide': 'dense 2x-wide FFN (MoE-active-matched)'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load_runs():
    runs = {}
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if (d / 'summary.json').exists():
            s = json.loads((d / 'summary.json').read_text())
            s['log'] = [json.loads(line) for line in (d / 'log.jsonl').read_text().splitlines()]
            runs[d.name] = s
    return runs


def offset(name, runs):
    """Phase-2 runs start where the dense run stopped, so plot them on one step axis."""
    return runs['dense']['steps'] if runs[name]['init'] in ('resume', 'upcycle') and 'dense' in runs else 0


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc='left', color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_val_curves(runs):
    names = [n for n in COLORS if n in runs]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for zoom, ax in zip((False, True), axes):
        for n in names:
            evals = [r for r in runs[n]['log'] if r['type'] == 'eval']
            off = offset(n, runs)
            xs = [off + r['step'] for r in evals]
            ys = [r['val_loss'] for r in evals]
            dashed = n == 'dense_continue'
            ax.plot(xs, ys, color=COLORS[n], linewidth=2, linestyle='--' if dashed else '-',
                    marker='o', markersize=4, label=LABELS[n])
        if 'dense' in runs:
            ax.axvline(runs['dense']['steps'], color=MUTED, linewidth=1, linestyle=':')
        if zoom and 'dense' in runs:
            p1 = runs['dense']['steps']
            ax.set_xlim(p1 - 100, max(offset(n, runs) + runs[n]['steps'] for n in names) + 50)
            lo = min(min(r['val_loss'] for r in runs[n]['log'] if r['type'] == 'eval') for n in names)
            ax.set_ylim(lo - 0.03, runs['dense']['final_val_loss'] + 0.06)
        style(ax, 'Phase 2 (after conversion), zoomed' if zoom else 'Validation loss, all runs',
              'training step (phase-2 runs continue from step ' + str(runs.get('dense', {}).get('steps', 0)) + ')',
              'val cross-entropy (nats/char)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = PLOTS / 'val_loss.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_utilization(runs):
    names = [n for n in ('moe_upcycle', 'moe_scratch') if n in runs and 'final_expert_frac' in runs[n]]
    if not names:
        return None
    # one shared scale; with top-k routing an expert's load can reach at most E/k x uniform
    vmax = max(f * len(row) for n in names for row in runs[n]['final_expert_frac'] for f in row)
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 3.2), squeeze=False)
    for n, ax in zip(names, axes[0]):
        frac = runs[n]['final_expert_frac']
        e = len(frac[0])
        im = ax.imshow([[f * e for f in row] for row in frac], cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        for i, row in enumerate(frac):
            for j, f in enumerate(row):
                v = f * e
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.55 * vmax else INK)
        ax.set_xticks(range(e))
        ax.set_yticks(range(len(frac)))
        ax.set_yticklabels([f'block {i}' for i in runs[n]['moe_layers']])
        style(ax, f'{LABELS[n]}: load per expert', 'expert', '')
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9)
    cb.set_label('share of assignments x E (1.0 = uniform)', color=MUTED)
    path = PLOTS / 'expert_utilization.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_sweep(runs):
    names = ['moe_upcycle'] + sorted(n for n in runs if n.startswith('sweep_'))
    names = [n for n in names if n in runs]
    if len(names) < 2 or 'dense_continue' not in runs:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = [config_label(runs[n]) for n in names]
    vals = [runs[n]['final_val_loss'] for n in names]
    # dots, not bars: the axis is zoomed, and bar length from a truncated baseline would mislead
    ax.scatter(vals, range(len(names)), color='#eb6834', s=60, zorder=3, label='upcycled MoE')
    ref = runs['dense_continue']['final_val_loss']
    ax.axvline(ref, color='#2a78d6', linewidth=2, linestyle='--', label=f'dense, continued ({ref:.4f})')
    for i, v in enumerate(vals):
        ax.text(v + 0.0025, i, f'{v:.4f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels)
    lo = min(vals + [ref])
    ax.set_xlim(lo - 0.04, max(vals + [ref]) + 0.03)
    style(ax, 'Upcycled MoE settings: final val loss (same phase-2 budget)', 'val cross-entropy', '')
    ax.legend(frameon=False, fontsize=8, loc='center left')
    fig.tight_layout()
    path = PLOTS / 'upcycle_sweep.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def config_label(s):
    if s['n_experts'] == 0:
        return 'dense'
    if s['n_experts'] == 1:
        return f"dense ({s['config']['moe']['expert_hidden'] // s['config']['model']['n_embd']}C FFN)"
    every = s['config']['moe']['moe_every']
    return f"E={s['n_experts']} k={s['top_k']}" + (f" every {every}" if every > 1 else '')


def table(runs, names):
    rows = ['| run | model | init | steps | total params | active params | start val | final train | '
            'final val | tokens/s | wall (min) |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for n in names:
        s = runs[n]
        rows.append(f"| {n} | {config_label(s)} | {s['init']} | {s['steps']} | {s['params_total'] / 1e6:.3f}M | "
                    f"{s['params_active'] / 1e6:.3f}M | {s['initial_val_loss']:.4f} | {s['final_train_loss']:.4f} | "
                    f"**{s['final_val_loss']:.4f}** | {s['tokens_per_s']:,.0f} | {s['wall_s'] / 60:.1f} |")
    return rows


def main():
    runs = load_runs()
    PLOTS.mkdir(parents=True, exist_ok=True)
    main_names = [n for n in COLORS if n in runs]
    sweep_names = [n for n in ['moe_upcycle'] + sorted(n for n in runs if n.startswith('sweep_')) if n in runs]
    lines = ['# Assignment 14 results', '', 'Generated by `report.py` from `results/runs/`.', '',
             '## Main runs', '', *table(runs, main_names), '']
    if len(sweep_names) > 1:
        lines += ['## Upcycling sweep (phase 2, same budget as `moe_upcycle`)', '', *table(runs, sweep_names), '']
    lines += ['## Routing at the final eval', '',
              '| run | aux (load-balance) loss | max expert load (x uniform) | min expert load (x uniform) | dropped |',
              '|---|---|---|---|---|']
    for n in main_names + sweep_names[1:]:
        s = runs[n]
        if 'final_expert_frac' not in s or s['n_experts'] < 2:
            continue
        e = s['n_experts']
        loads = [f * e for row in s['final_expert_frac'] for f in row]
        lines.append(f"| {n} | {s['final_aux_loss']:.4f} | {max(loads):.2f} | {min(loads):.2f} | "
                     f"{s['final_drop_frac']:.1%} |")
    lines += ['', '## Samples', '']
    for n in main_names:
        lines.append(f"- **{n}** (prompt {runs[n]['sample_prompt']!r}): {runs[n]['sample']!r}")
    lines.append('')
    plots = [p for p in (plot_val_curves(runs), plot_utilization(runs), plot_sweep(runs)) if p]
    lines += ['## Plots', ''] + [f'![{p.stem}](plots/{p.name})' for p in plots] + ['']
    (RESULTS_DIR / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
