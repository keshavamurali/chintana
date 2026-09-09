"""
Small shared plotting helper for the assignment-11 experiment scripts.
Keeps a consistent look without pulling every script into a big shared
framework -- each question's script stays runnable on its own.
"""

import os

import matplotlib
matplotlib.use('Agg')  # headless: these scripts run from the CLI, not a notebook
import matplotlib.pyplot as plt


def plot_series(x, series, title, xlabel, ylabel, out_path, logy=False, markers=True):
    """
    series: dict mapping a label -> list of y-values (same length as x)
    Saves a PNG to out_path (parent directories created as needed).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, y in series.items():
        ax.plot(x, y, marker='o' if markers else None, markersize=4, label=label)
    if logy:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved plot to {out_path}")
