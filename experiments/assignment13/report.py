"""Collect every run and probe result into results/summary.md and results/plots/.

Runs are read from results/runs/*/summary.json (+ log.jsonl); runs whose name
starts with 'screen_' are the short variant-screening runs, everything else is
a main run. Probe CSVs come from memory_probe.py and check_reversible.py.
Missing inputs are skipped, so this can be re-run after every stage.

  python experiments/assignment13/report.py
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import common

RUNS_DIR = common.RESULTS_DIR / "runs"
LOG_DIR = common.RESULTS_DIR / "logs"
PLOT_DIR = common.RESULTS_DIR / "plots"

# fixed categorical order (validated: CVD-adjacent dE >= 9, normal >= 19.6);
# three of these sit under 3:1 on white, so every line is direct-labeled and
# every chart has a table in summary.md
VARIANT_COLORS = {
    "baseline": "#2a78d6",
    "midpoint": "#eb6834",
    "midpoint_a": "#1baf7a",
    "leapfrog": "#eda100",
    "hamiltonian": "#e87ba4",
}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)


def _end_labels(ax, items, min_gap_frac=0.055):
    """Direct labels at the right end of each line, nudged apart vertically so
    converging lines stay legible. items: [(x, y, text)]."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    gap = min_gap_frac * (hi - lo)
    placed = []
    for x, y, text in sorted(items, key=lambda t: t[1]):
        ty = y if not placed else max(y, placed[-1][1] + gap)
        placed.append((x, ty, text, y))
    # shift the whole stack down if it ran off the top
    overflow = placed[-1][1] - hi
    if overflow > 0 and not log:
        placed = [(x, ty - overflow, t, y) for x, ty, t, y in placed]
    for x, ty, text, y in placed:
        ax.annotate(text, (x, y), xytext=(x, ty), textcoords="data", va="center", fontsize=9,
                    color=INK_2, annotation_clip=False)


def _save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"wrote {PLOT_DIR / name}")


def _read_csv(name):
    path = LOG_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_runs():
    runs = []
    for d in sorted(RUNS_DIR.glob("*")):
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        recs = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines() if line]
        s["train_curve"] = [(r["tokens"], r["loss"]) for r in recs if r["type"] == "train"]
        s["val_curve"] = [(r["tokens"], r["val_loss"]) for r in recs if r["type"] == "eval"]
        runs.append(s)
    return runs


def _smooth(ys, k=9):
    out = []
    for i in range(len(ys)):
        w = ys[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def _line_style(run):
    return "--" if run["name"].startswith("run3") or "maxbatch" in run["name"] else "-"


def plot_curves(runs, key, title, ylabel, name, smooth=False, skip_initial=False):
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    for r in runs:
        pts = r[key][1:] if skip_initial else r[key]
        if not pts:
            continue
        xs = [t / 1e6 for t, _ in pts]
        ys = _smooth([v for _, v in pts]) if smooth else [v for _, v in pts]
        ax.plot(xs, ys, color=VARIANT_COLORS.get(r["variant"], INK_MUTED), linewidth=2,
                linestyle=_line_style(r), label=f"{r['name']} (B={r['batch_size']})",
                marker=None if smooth else "o", markersize=4)
        labels.append((xs[-1], ys[-1], f"  {r['name']}  {ys[-1]:.3f}"))
    _style(ax, title, "tokens seen (millions)", ylabel)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    ax.margins(x=0.35)
    _end_labels(ax, labels)
    _save(fig, name)


def plot_bars(runs, key, title, ylabel, name, fmt):
    runs = [r for r in runs if r.get(key) is not None]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['name']}\nB={r['batch_size']}" for r in runs]
    vals = [r[key] for r in runs]
    bars = ax.bar(labels, vals, width=0.55, edgecolor=SURFACE, linewidth=2,
                  color=[VARIANT_COLORS.get(r["variant"], INK_MUTED) for r in runs])
    for b, v in zip(bars, vals):
        ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK_2)
    _style(ax, title, "", ylabel)
    ax.grid(axis="x", visible=False)
    _save(fig, name)


def plot_grouped_speed(rows, name):
    if not rows:
        return
    batches = sorted({int(r["batch_size"]) for r in rows})
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, v in enumerate(variants):
        vals = [next((float(r["median_tokens_per_s"]) for r in rows
                      if r["variant"] == v and int(r["batch_size"]) == b), 0.0) for b in batches]
        xs = [j + (i - (len(variants) - 1) / 2) * width for j in range(len(batches))]
        bars = ax.bar(xs, vals, width=width, color=VARIANT_COLORS.get(v, INK_MUTED), edgecolor=SURFACE,
                      linewidth=2, label=v)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:,.0f}", (bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(batches)), [f"batch {b}" for b in batches])
    _style(ax, "Training speed, variants alternated (median of repetitions)", "", "tokens / s")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _save(fig, name)


def plot_xy(rows, xkey, ykey, group, title, xlabel, ylabel, name, logx=False, filt=None, series_style=None):
    rows = [r for r in rows if (filt is None or filt(r))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups, labels = {}, []
    for r in rows:
        groups.setdefault(group(r), []).append(r)
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: float(r[xkey]))
        xs, ys = [float(r[xkey]) for r in rs], [float(r[ykey]) for r in rs]
        variant, style = series_style(g) if series_style else (g, "-")
        ax.plot(xs, ys, color=VARIANT_COLORS.get(variant, INK_MUTED), linewidth=2, linestyle=style,
                marker="o", markersize=5, label=g)
        labels.append((xs[-1], ys[-1], f"  {g}"))
    _style(ax, title, xlabel, ylabel)
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    ax.margins(x=0.25)
    _end_labels(ax, labels)
    _save(fig, name)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _gb(mb):
    return f"{float(mb) / 1024:.2f}" if mb not in (None, "", "nan") else "-"


def run_table(runs):
    rows = []
    for r in runs:
        rows.append([r["name"], r["variant"], r["batch_size"], r["steps"], f"{r['tokens'] / 1e6:.1f}M",
                     f"{r['peak_lr']:.2e}", f"{r['final_train_loss']:.4f}", f"{r['final_val_loss']:.4f}",
                     f"{r['tokens_per_s']:,.0f}", _gb(r.get("peak_mem_mb")), f"{r['wall_s'] / 60:.1f}",
                     "yes" if r["diverged"] else "no"])
    return _table(["run", "variant", "batch", "steps", "tokens", "peak lr", "final train loss",
                   "final val loss", "tokens/s", "peak mem (GB)", "wall (min)", "diverged"], rows)


def main():
    runs = load_runs()
    screen = [r for r in runs if r["name"].startswith("screen_")]
    main_runs = [r for r in runs if not r["name"].startswith("screen_")]

    plot_curves(main_runs, "val_curve", "Validation loss, main runs (from the first eval after step 0)",
                "val loss", "val_loss.png", skip_initial=True)
    plot_curves(main_runs, "train_curve", "Training loss (9-point moving average)", "train loss",
                "train_loss.png", smooth=True)
    plot_curves(screen, "val_curve", "Variant screening: validation loss (from the first eval after step 0)",
                "val loss", "screen_val_loss.png", skip_initial=True)
    plot_bars(main_runs, "tokens_per_s", "Training throughput", "tokens / s", "throughput.png",
              lambda v: f"{v:,.0f}")
    plot_bars(main_runs, "peak_mem_mb", "Peak GPU memory (torch max_memory_allocated)", "MB",
              "peak_memory.png", lambda v: f"{v / 1024:.2f} GB")

    depth = _read_csv("memory_vs_depth.csv")
    plot_xy(depth, "n_layer", "peak_mb", lambda r: r["variant"],
            f"Peak memory vs depth (batch {depth[0]['batch_size'] if depth else ''})",
            "n_layer", "peak MB", "memory_vs_depth.png", filt=lambda r: r["fits"] == "True")
    trials = _read_csv("memory_vs_batch.csv")
    plot_xy(trials, "batch_size", "peak_mb", lambda r: f"{r['variant']} / {r['loss_impl']}",
            "Peak memory vs batch size (successful steps)", "batch size", "peak MB", "memory_vs_batch.png",
            logx=True, filt=lambda r: r["fits"] == "True",
            series_style=lambda g: (g.split(" / ")[0], "-" if g.endswith("fused") else ":"))
    tput = _read_csv("throughput_vs_batch.csv")
    plot_xy(tput, "batch_size", "tokens_per_s", lambda r: r["variant"], "Throughput vs batch size",
            "batch size", "tokens / s", "throughput_vs_batch.png", logx=True)

    # charts compare the baseline with the reversible variant the main runs used;
    # the tables in summary.md keep every variant
    shown = {"baseline"} | {r["variant"] for r in main_runs}
    cpu_mem = _read_csv("cpu_memory.csv")
    plot_xy(cpu_mem, "batch_size", "peak_mb", lambda r: r["variant"],
            "Peak memory added by one training step vs batch (CPU)", "batch size", "MB",
            "cpu_memory_vs_batch.png", logx=True, filt=lambda r: r["variant"] in shown)
    cpu_depth = _read_csv("cpu_memory_vs_depth.csv")
    plot_xy(cpu_depth, "n_layer", "peak_mb", lambda r: r["variant"],
            f"Peak memory added by one training step vs depth (CPU, batch {cpu_depth[0]['batch_size'] if cpu_depth else ''})",
            "n_layer", "MB", "cpu_memory_vs_depth.png")
    cpu_speed = _read_csv("cpu_speed.csv")
    plot_grouped_speed(cpu_speed, "cpu_speed.png")

    md = ["# Assignment 13 results", "",
          "Generated by `report.py` from `results/runs/` and `results/logs/`.", ""]
    if main_runs:
        md += ["## Main runs", "", run_table(main_runs), ""]
        md += ["### Samples", ""]
        md += [f"- **{r['name']}** (prompt {r.get('sample_prompt', 'The meaning of')!r}): "
               f"{r.get('sample', '').strip()!r}" for r in main_runs]
        md += [""]
    if screen:
        md += ["## Variant screening", "", run_table(screen), ""]
    mb = _read_csv("max_batch.csv")
    if mb:
        md += ["## Maximum batch size (one full training step)", "",
               _table(["gpu", "variant", "loss head", "max batch", "peak mem at max (GB)", "hit search cap"],
                      [[r["gpu"], r["variant"], r["loss_impl"], r["max_batch"], _gb(r["peak_mb_at_max"]),
                        r["hit_cap"]] for r in mb]), ""]
    sustained = _read_csv("sustained_batch.csv")
    if sustained:
        md += ["## Sustained maximum batch (full training runs)", "",
               "Each run starts at the one-step maximum from `max_batch.csv` and steps down 5% after "
               "every out-of-memory exit; the completed row is the batch that sustained a whole run.", "",
               _table(["run", "variant", "one-step max", "batch tried", "outcome"],
                      [[r["run"], r["variant"], r["probe_max_batch"], r["batch_size"], r["outcome"]]
                       for r in sustained]), ""]
    if tput:
        md += ["## Throughput vs batch size", "",
               _table(["variant", "batch", "tokens/s"],
                      [[r["variant"], r["batch_size"], f"{int(r['tokens_per_s']):,}"] for r in tput]), ""]
    if depth:
        md += ["## Peak memory vs depth", "",
               _table(["variant", "n_layer", "batch", "fits", "peak mem (GB)"],
                      [[r["variant"], r["n_layer"], r["batch_size"], r["fits"], _gb(r["peak_mb"])]
                       for r in depth]), ""]
    if cpu_mem:
        md += ["## CPU memory for one training step", "",
               "`saved` = bytes autograd holds for backward (exact); `peak` = growth of process peak RSS "
               "during the step (includes the reversible backward's transient one-layer recompute).", "",
               _table(["variant", "n_layer", "batch", "saved for backward (MB)", "peak RSS growth (MB)"],
                      [[r["variant"], r["n_layer"], r["batch_size"], r["saved_mb"], r["peak_mb"]] for r in cpu_mem]), ""]
    if cpu_depth:
        md += ["## CPU memory vs depth", "",
               _table(["variant", "n_layer", "batch", "saved for backward (MB)", "peak RSS growth (MB)"],
                      [[r["variant"], r["n_layer"], r["batch_size"], r["saved_mb"], r["peak_mb"]] for r in cpu_depth]), ""]
    if cpu_speed:
        md += ["## CPU speed (variants alternated, median of repetitions)", "",
               _table(["variant", "batch", "reps", "median tokens/s", "min", "max"],
                      [[r["variant"], r["batch_size"], r["reps"], f"{int(r['median_tokens_per_s']):,}",
                        f"{int(r['min']):,}", f"{int(r['max']):,}"] for r in cpu_speed]), ""]
    grads = _read_csv("grad_check.csv")
    if grads:
        md += ["## Gradient equivalence", "",
               _table(["device", "dtype", "autocast", "variant", "check", "|loss diff|", "grad rel err"],
                      [[r["device"], r["dtype"], r["autocast"], r["variant"], r["check"],
                        f"{float(r['loss_abs_diff']):.1e}", f"{float(r['grad_rel_err']):.1e}"] for r in grads]), ""]
    recon = _read_csv("reconstruction.csv")
    if recon:
        worst = {}
        for r in recon:
            k = (r["setting"], r["variant"])
            worst[k] = max(worst.get(k, 0.0), float(r["rel_err"]))
        md += ["## Worst rebuilt-state error over all layers", "",
               _table(["setting", "variant", "max relative L2 error"],
                      [[s, v, f"{e:.1e}"] for (s, v), e in worst.items()]), ""]
    (common.RESULTS_DIR / "summary.md").write_text("\n".join(md))
    print(f"wrote {common.RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
