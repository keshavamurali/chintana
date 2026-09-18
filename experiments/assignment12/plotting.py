"""Shared matplotlib styling + chart builders for the ZeRO demo.

Colors follow the project's categorical palette (fixed hue order, not cycled):
blue/orange/aqua/yellow for the four pipeline stages (dp, zero1, zero2, zero3),
and a separate blue/orange/aqua trio for the three memory components
(params/grads/optimizer) stacked within a stage's bar.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STAGE_COLORS = {
    "dp": "#2a78d6",
    "zero1": "#eb6834",
    "zero2": "#1baf7a",
    "zero3": "#eda100",
}
COMPONENT_COLORS = {
    "params": "#2a78d6",
    "grads": "#eb6834",
    "optimizer": "#1baf7a",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def stacked_memory_bar(per_stage_bytes, stage_order, stage_labels, outfile, title):
    """per_stage_bytes: {stage: {"params":B, "grads":B, "optimizer":B}}"""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    _style_axes(ax)
    x = range(len(stage_order))
    bottoms = [0] * len(stage_order)
    for component in ("params", "grads", "optimizer"):
        heights = [per_stage_bytes[s][component] / (1024 ** 2) for s in stage_order]
        ax.bar(
            x, heights, bottom=bottoms, width=0.6, label=component,
            color=COMPONENT_COLORS[component], edgecolor=SURFACE, linewidth=2, zorder=3,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    for xi, s in zip(x, stage_order):
        total = sum(per_stage_bytes[s].values())
        ax.text(xi, bottoms[xi] + max(bottoms) * 0.02, _human_bytes(total),
                 ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=9)
    ax.set_ylabel("memory per GPU (MB)")
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)


def memory_vs_n_lines(series, n_values, stage_order, stage_labels, outfile, title):
    """series: {stage: [bytes_per_gpu for each n in n_values]}"""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    _style_axes(ax)
    for s in stage_order:
        ys = [v / (1024 ** 2) for v in series[s]]
        ax.plot(n_values, ys, marker="o", markersize=5, linewidth=2,
                 color=STAGE_COLORS[s], label=stage_labels[s], zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("number of virtual GPUs (N)")
    ax.set_ylabel("memory per GPU (MB, log scale)")
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, labelcolor=INK_SECONDARY, fontsize=8, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)


def stage_bar(values, stage_order, stage_labels, outfile, title, ylabel, fmt=None):
    """values: {stage: number}"""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    _style_axes(ax)
    x = range(len(stage_order))
    heights = [values[s] for s in stage_order]
    ax.bar(x, heights, width=0.55, color=[STAGE_COLORS[s] for s in stage_order],
           edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, s in zip(x, stage_order):
        label = fmt(values[s]) if fmt else f"{values[s]:.3g}"
        ax.text(xi, values[s] + max(heights) * 0.02, label, ha="center", va="bottom",
                 fontsize=9, color=INK_PRIMARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)


def stacked_comm_bar(per_stage_events, stage_order, stage_labels, outfile, title):
    """per_stage_events: {stage: [(op_name, bytes), ...]}"""
    all_ops = []
    for events in per_stage_events.values():
        for name, _ in events:
            if name not in all_ops:
                all_ops.append(name)
    op_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    _style_axes(ax)
    x = range(len(stage_order))
    bottoms = [0] * len(stage_order)
    for i, op in enumerate(all_ops):
        heights = []
        for s in stage_order:
            byname = dict(per_stage_events[s])
            heights.append(byname.get(op, 0) / (1024 ** 2))
        ax.bar(x, heights, bottom=bottoms, width=0.6, label=op,
               color=op_colors[i % len(op_colors)], edgecolor=SURFACE, linewidth=2, zorder=3)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    for xi, s in zip(x, stage_order):
        ax.text(xi, bottoms[xi] + max(bottoms) * 0.02, f"{bottoms[xi]:.1f}MB",
                 ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=9)
    ax.set_ylabel("communication volume (MB, whole cluster)")
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY, fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
