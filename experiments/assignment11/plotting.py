"""
Small shared plotting helper for the assignment-11 experiment scripts.
Keeps a consistent look without pulling every script into a big shared
framework -- each question's script stays runnable on its own.
"""

import os

import matplotlib
matplotlib.use('Agg')  # headless: these scripts run from the CLI, not a notebook
import matplotlib.pyplot as plt


LINESTYLES = ['-', '--', '-.', ':']
MARKER_SHAPES = ['o', 's', '^', 'D', 'v', 'P']


def plot_series(x, series, title, xlabel, ylabel, out_path, logy=False, logx=False, markers=True,
                 mark_points=None):
    """
    series: dict mapping a label -> list of y-values (same length as x)
    mark_points: optional dict mapping a series label -> (x, y) to highlight
                 (e.g. that series' minimum) with a black star + annotation.

    Series are drawn with a different linestyle/marker/linewidth each, cycling
    through LINESTYLES/MARKER_SHAPES -- so two series with identical (or
    near-identical) values stay visually distinguishable instead of one
    solid line silently hiding another underneath it.

    Saves a PNG to out_path (parent directories created as needed).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, label=label,
                 linestyle=LINESTYLES[i % len(LINESTYLES)],
                 linewidth=2.5 - 0.4 * min(i, 3),
                 marker=MARKER_SHAPES[i % len(MARKER_SHAPES)] if markers else None,
                 markersize=5, markevery=max(1, len(x) // 25) if markers else None,
                 alpha=0.85)
    if mark_points:
        for label, (px, py) in mark_points.items():
            ax.plot(px, py, marker='*', markersize=16, color='black', linestyle='None', zorder=5)
            ax.annotate(f'{px:.0e}', (px, py), textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8)
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved plot to {out_path}")
