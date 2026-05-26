"""
plot_metric_criterion_overall_heatmap.py

Corpus-level companion to plot_metric_criterion_split_heatmap.py.

Pools all top-30-tasks papers (no per-task filter) into a single
split-cell heatmap: top-K metrics (rows) x top-K criteria (cols),
each cell split into LaaJ (upper-left, orange) and Human (lower-right,
blue) triangles.

Cell value = within-method pair-association LR (same machinery as the
per-task version), computed by `compute_per_method`: ratio of the
metric-rate among papers reporting that criterion under the method to
the metric-rate among papers not reporting it under the method. The
test (G^2 + BH-FDR, q<=0.05, co-occurrence >= MIN_PAIR_CO_OCC) and the
colour scale are shared with the per-task figure for direct visual
comparison.

Top-K selection is by raw corpus frequency for both axes; no freq vs
distinct band split, because at corpus level there is no "other task"
to compare against.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from data_loader import load_data, short_label
from plot_metric_criterion_split_heatmap import (
    load_metric_display,
    compute_per_method,
    draw_split_cell,
    LABEL_ABBREV,
    LAAJ_COLOR,
    HUMAN_COLOR,
    LR_LOW,
    LR_HIGH,
)

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_criterion_overall_heatmap.png"
OUT_PDF = OUT_DIR / "metric_criterion_overall_heatmap.pdf"

TOP_METRICS = 12
TOP_CRITERIA = 12


def select_overall(papers, top_metrics_n, top_criteria_n):
    """Top-K metrics and criteria, unioned across the two methods.

    Computes the top-K metrics within LaaJ-using papers (by frequency)
    and the top-K within Human-using papers, then returns the UNION so
    method-distinctive items (e.g., BARTScore for LaaJ; CIDER, Distinct
    for Human; safety-related criteria for LaaJ) survive even if they
    sit below the pooled top-K. The union is ordered by combined
    cross-method frequency so the headline items still anchor the top
    of the axis.

    Returns (laaj_papers, human_papers, top_metrics, top_criteria).
    `top_metrics` are lowercased canonical strings; `top_criteria` keep
    their QCET original case."""
    laaj_papers = [p for p in papers if p.get("laaj_criteria")]
    human_papers = [p for p in papers if p.get("human_criteria")]

    # --- Metrics: per-method top-K, then union ----------------------
    def metric_freq(subset):
        c: Counter = Counter()
        for p in subset:
            for m in (p.get("auto_metrics") or []):
                c[m.lower().strip()] += 1
        return c
    laaj_mc = metric_freq(laaj_papers)
    human_mc = metric_freq(human_papers)
    union_m = (
        {m for m, _ in laaj_mc.most_common(top_metrics_n)}
        | {m for m, _ in human_mc.most_common(top_metrics_n)}
    )
    # Order the union by combined frequency over papers that use either
    # method (each paper counted once across the two subsets to avoid
    # double-counting dual-method papers).
    method_papers = [p for p in papers if p.get("laaj_criteria") or p.get("human_criteria")]
    pooled_mc: Counter = Counter()
    for p in method_papers:
        for m in (p.get("auto_metrics") or []):
            pooled_mc[m.lower().strip()] += 1
    top_m = [m for m, _ in pooled_mc.most_common() if m in union_m]

    # --- Criteria: per-method top-K from each method's own field, union
    def crit_freq(subset, field):
        c: Counter = Counter()
        for p in subset:
            for x in (p.get(field) or []):
                c[x.strip()] += 1
        return c
    laaj_cc = crit_freq(laaj_papers, "laaj_criteria")
    human_cc = crit_freq(human_papers, "human_criteria")
    union_c = (
        {x for x, _ in laaj_cc.most_common(top_criteria_n)}
        | {x for x, _ in human_cc.most_common(top_criteria_n)}
    )
    pooled_cc = laaj_cc + human_cc
    top_c = [x for x, _ in pooled_cc.most_common() if x in union_c]

    return laaj_papers, human_papers, top_m, top_c


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")

    metric_display_map = load_metric_display()

    def metric_label(m: str) -> str:
        return metric_display_map.get(m, m.title())

    def crit_label(c: str) -> str:
        short = short_label(c, prefixed=False)
        return LABEL_ABBREV.get(short, short)

    laaj_papers, human_papers, top_m, top_c = select_overall(
        papers, TOP_METRICS, TOP_CRITERIA,
    )
    print(f"n_LaaJ={len(laaj_papers)} n_Human={len(human_papers)}")
    print(f"metrics  ({len(top_m)}): {[metric_label(m) for m in top_m]}")
    print(f"criteria ({len(top_c)}): {[crit_label(c) for c in top_c]}")

    lr_l, lr_h, sig_l, sig_h, co_l, co_h = compute_per_method(
        laaj_papers, human_papers, top_m, top_c,
    )
    print(f"sig LaaJ cells:  {int(sig_l.sum())} / {sig_l.size}")
    print(f"sig Human cells: {int(sig_h.sum())} / {sig_h.size}")

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    n_m, n_c = len(top_m), len(top_c)
    for mi in range(n_m):
        for ci in range(n_c):
            draw_split_cell(
                ax, ci, mi,
                lr_l[mi, ci], lr_h[mi, ci],
                bool(sig_l[mi, ci]), bool(sig_h[mi, ci]),
            )

    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(n_m - 0.5, -0.5)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(
        [crit_label(c) for c in top_c],
        rotation=35, ha="right", rotation_mode="anchor", fontsize=8,
    )
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([metric_label(m) for m in top_m], fontsize=8)
    ax.set_xlabel("Criterion (top-12 by corpus frequency)", fontsize=9)
    ax.set_ylabel("Metric (top-12 by corpus frequency)", fontsize=9)
    ax.tick_params(axis="both", length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    LEGEND_Y = 0.015  # vertical position of legend strip
    def add_method_legend(fig, x_tri, side, x_text, label,
                          x_bar, base_color, bar_width=0.10):
        tri_ax = fig.add_axes([x_tri, LEGEND_Y, 0.022, 0.040])
        if side == "upper":
            tri = mpatches.Polygon(
                [(0, 1), (1, 1), (0, 0)],
                facecolor=base_color, edgecolor="#333",
                linewidth=0.5, alpha=0.9,
            )
        else:
            tri = mpatches.Polygon(
                [(1, 1), (1, 0), (0, 0)],
                facecolor=base_color, edgecolor="#333",
                linewidth=0.5, alpha=0.9,
            )
        tri_ax.add_patch(tri)
        tri_ax.set_xlim(-0.05, 1.05)
        tri_ax.set_ylim(-0.05, 1.05)
        tri_ax.set_xticks([])
        tri_ax.set_yticks([])
        tri_ax.set_aspect("equal")
        for s in tri_ax.spines.values():
            s.set_visible(False)

        fig.text(x_text, LEGEND_Y + 0.018, label, fontsize=8, va="center")

        ramp_ax = fig.add_axes([x_bar, LEGEND_Y + 0.015, bar_width, 0.013])
        n = 256
        gradient = np.linspace(0, 1, n).reshape(1, -1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"white_{label}", ["#ffffff", base_color], N=n,
        )
        ramp_ax.imshow(gradient, aspect="auto", cmap=cmap)
        ramp_ax.set_xticks([0, n - 1])
        ramp_ax.set_xticklabels(
            [f"LR={LR_LOW:g}", f"$\\geq${LR_HIGH:g}"], fontsize=6.5,
        )
        ramp_ax.set_yticks([])
        for s in ramp_ax.spines.values():
            s.set_visible(False)
        ramp_ax.tick_params(length=0, pad=1)

    add_method_legend(fig, 0.14, "upper", 0.175, "LaaJ-LR:", 0.235, LAAJ_COLOR)
    add_method_legend(fig, 0.52, "lower", 0.555, "Human-LR:", 0.615, HUMAN_COLOR)

    fig.subplots_adjust(left=0.21, right=0.99, top=0.97, bottom=0.30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
