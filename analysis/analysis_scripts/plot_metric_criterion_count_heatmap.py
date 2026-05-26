"""
plot_metric_criterion_count_heatmap.py

Appendix companion to Figure 5 (plot_metric_criterion_split_heatmap.py),
showing raw co-occurrence percentages instead of LR.

Layout and item selection are identical to Figure 5: per-task 1x4 grid,
10 metrics x 10 criteria per panel (top-5 by frequency + top-5 by
task-LR among non-frequent candidates). Each cell is split diagonally:
upper-left orange triangle = LaaJ % (= papers in the LaaJ-using subset
that report BOTH metric and criterion, divided by the LaaJ-using
subset size, times 100), lower-right blue triangle = Human %. Colour
saturates linearly from 0 % (white) to 100 % (full method colour).

No significance testing is performed: every cell renders at full
opacity. This figure complements Figure 5 by exposing the raw frequency
landscape that LR is computed from, making the case for LR visible by
contrast (frequent-vs-frequent cells dominate raw counts even where
they show no special association under LR).
"""

from __future__ import annotations

import os
import sys
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
    select_for_task,
    compute_per_method,
    load_metric_display,
    LABEL_ABBREV,
    DISTINCT_LABEL_COLOR,
    LAAJ_COLOR,
    HUMAN_COLOR,
    TASKS_ORDERED,
    TOP_METRICS_PER_TASK,
    TOP_CRITERIA_PER_TASK,
    N_FREQ,
)

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_criterion_count_heatmap.png"
OUT_PDF = OUT_DIR / "metric_criterion_count_heatmap.pdf"


def pct_to_color(pct: float, base_color: str) -> tuple:
    """Map percentage (0-100) to a colour from white to base_color via
    linear interpolation."""
    if pct is None or pct <= 0:
        return (1.0, 1.0, 1.0)
    sat = float(np.clip(pct / 100.0, 0.0, 1.0))
    base_rgb = np.array(mcolors.to_rgb(base_color))
    rgb = (1 - sat) * np.ones(3) + sat * base_rgb
    return tuple(rgb)


def draw_split_cell_pct(ax, x, y, pct_laaj, pct_human):
    """Two triangles at cell (x, y): upper-left = LaaJ %, lower-right
    = Human %. Same triangle orientation as Figure 5 (the cells use an
    inverted y-axis so (x-0.5, y-0.5) is the visual upper-left corner)."""
    upper = mpatches.Polygon(
        [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x - 0.5, y + 0.5)],
        facecolor=pct_to_color(pct_laaj, LAAJ_COLOR),
        edgecolor="white", linewidth=0.4, alpha=1.0, zorder=2,
    )
    lower = mpatches.Polygon(
        [(x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)],
        facecolor=pct_to_color(pct_human, HUMAN_COLOR),
        edgecolor="white", linewidth=0.4, alpha=1.0, zorder=2,
    )
    ax.add_patch(upper)
    ax.add_patch(lower)


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")
    metric_display_map = load_metric_display()

    def metric_label(m: str) -> str:
        return metric_display_map.get(m, m.title())

    task_data = []
    for task_name, label in TASKS_ORDERED:
        laaj, human, top_m, top_c, m_groups, c_groups = select_for_task(
            papers, task_name, TOP_METRICS_PER_TASK, TOP_CRITERIA_PER_TASK,
        )
        _, _, _, _, co_l, co_h = compute_per_method(
            laaj, human, top_m, top_c,
        )
        n_laaj = max(len(laaj), 1)
        n_human = max(len(human), 1)
        pct_l = 100.0 * co_l / n_laaj
        pct_h = 100.0 * co_h / n_human
        task_data.append({
            "task": task_name, "label": label,
            "n_laaj": len(laaj), "n_human": len(human),
            "metrics": top_m, "criteria": top_c,
            "m_groups": m_groups, "c_groups": c_groups,
            "pct_l": pct_l, "pct_h": pct_h,
        })
        print(f"\n{label}: n_LaaJ={len(laaj)} n_Human={len(human)}")
        print(f"  LaaJ % range: {pct_l.min():.1f}-{pct_l.max():.1f}")
        print(f"  Human % range: {pct_h.min():.1f}-{pct_h.max():.1f}")

    plt.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.8))

    for ax, td in zip(axes.flat, task_data):
        n_m = len(td["metrics"])
        n_c = len(td["criteria"])
        for mi in range(n_m):
            for ci in range(n_c):
                draw_split_cell_pct(
                    ax, ci, mi,
                    td["pct_l"][mi, ci], td["pct_h"][mi, ci],
                )
        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(n_m - 0.5, -0.5)

        def crit_label(c):
            short = short_label(c, prefixed=False)
            return LABEL_ABBREV.get(short, short)

        ax.set_xticks(range(n_c))
        x_text = ax.set_xticklabels(
            [crit_label(c) for c in td["criteria"]],
            rotation=35, ha="right", rotation_mode="anchor", fontsize=7,
        )
        ax.set_yticks(range(n_m))
        y_text = ax.set_yticklabels(
            [metric_label(m) for m in td["metrics"]], fontsize=7,
        )
        for label, grp in zip(y_text, td["m_groups"]):
            if grp == "distinct":
                label.set_color(DISTINCT_LABEL_COLOR)
        for label, grp in zip(x_text, td["c_groups"]):
            if grp == "distinct":
                label.set_color(DISTINCT_LABEL_COLOR)

        if N_FREQ < n_m:
            ax.axhline(N_FREQ - 0.5, color="#888888", linewidth=0.6,
                       linestyle=":", zorder=4)
        if N_FREQ < n_c:
            ax.axvline(N_FREQ - 0.5, color="#888888", linewidth=0.6,
                       linestyle=":", zorder=4)
        ax.set_title(
            f"{td['label']}  (LaaJ {td['n_laaj']}, Human {td['n_human']})",
            fontsize=8.5, pad=4,
        )
        ax.tick_params(axis="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    # Bottom legend: two method elements + red-text key. Same layout
    # convention as Figure 5; scale runs 0--100 % instead of LR=1--20.
    def add_method_legend(fig, x_tri, side, x_text, label, x_bar,
                           base_color, bar_width=0.08):
        tri_ax = fig.add_axes([x_tri, 0.040, 0.022, 0.040])
        if side == "upper":
            filled = mpatches.Polygon(
                [(0, 1), (1, 1), (0, 0)],
                facecolor=base_color, edgecolor="#333",
                linewidth=0.5, alpha=0.9)
        else:
            filled = mpatches.Polygon(
                [(1, 1), (1, 0), (0, 0)],
                facecolor=base_color, edgecolor="#333",
                linewidth=0.5, alpha=0.9)
        tri_ax.add_patch(filled)
        tri_ax.set_xlim(-0.05, 1.05); tri_ax.set_ylim(-0.05, 1.05)
        tri_ax.set_xticks([]); tri_ax.set_yticks([])
        tri_ax.set_aspect("equal")
        for s in tri_ax.spines.values():
            s.set_visible(False)

        fig.text(x_text, 0.058, label, fontsize=7, va="center")

        ramp_ax = fig.add_axes([x_bar, 0.052, bar_width, 0.012])
        n = 256
        gradient = np.linspace(0, 1, n).reshape(1, -1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"white_{label}", ["#ffffff", base_color], N=n,
        )
        ramp_ax.imshow(gradient, aspect="auto", cmap=cmap)
        ramp_ax.set_xticks([0, n - 1])
        ramp_ax.set_xticklabels(["0\\%", "100\\%"], fontsize=6)
        ramp_ax.set_yticks([])
        for s in ramp_ax.spines.values():
            s.set_visible(False)
        ramp_ax.tick_params(length=0, pad=1)

    add_method_legend(fig, 0.10, "upper", 0.13, "LaaJ \\%:", 0.18, LAAJ_COLOR)
    add_method_legend(fig, 0.32, "lower", 0.35, "Human \\%:", 0.40, HUMAN_COLOR)

    fig.text(0.54, 0.058, "Red text", fontsize=7, va="center",
             color=DISTINCT_LABEL_COLOR)
    fig.text(0.575, 0.058,
             ": top-5 by task-LR (others: top-5 by frequency)",
             fontsize=7, va="center")

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.32,
                        wspace=0.55)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
