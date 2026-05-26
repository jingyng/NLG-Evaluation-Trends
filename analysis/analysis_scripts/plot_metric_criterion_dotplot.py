"""
plot_metric_criterion_dotplot.py

Side-by-side dot-plot companion to Figure 5 (the corpus-level split-cell
heatmap). Two panels --- LaaJ on the left (orange), Human on the right
(blue) --- both using the same top-12 metrics x top-12 criteria
selection (by raw corpus frequency, no per-panel clustering). The row
and column order is identical across panels so the eye can compare any
(metric, criterion) cell across the two methods at a glance.

Each circle encodes one cell:
  - radius = sqrt(co-occurrence count) (so area is linear in count)
  - colour saturation = log_10(LR) clipped to [0, log_10(LR_HIGH)]
  - bold dark edge if (G^2 + BH-FDR, q<=SIG_Q, co>=MIN_PAIR_CO_OCC)
    significant; faint edge otherwise.

Reading: a large saturated circle = a metric-criterion pair that is
both common (large) and tightly associated (saturated) under that
method; bold outlines mark cells that survive multiple-testing
correction.
"""

from __future__ import annotations

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
from plot_metric_criterion_overall_heatmap import select_overall
from plot_metric_criterion_split_heatmap import (
    load_metric_display,
    compute_per_method,
    LABEL_ABBREV,
    LAAJ_COLOR,
    HUMAN_COLOR,
    SIG_EDGE_COLOR,
    SIG_EDGE_WIDTH,
)

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_criterion_dotplot.png"
OUT_PDF = OUT_DIR / "metric_criterion_dotplot.pdf"

TOP_METRICS = 12
TOP_CRITERIA = 12

MIN_RADIUS = 0.10  # floor: smallest non-zero count is still visible
MAX_RADIUS = 0.45  # ceiling: largest count fits inside a 1x1 cell

# Diverging LR colour scale centred at LR=1.
# LR<1  => negative association: white --> NEG_COLOR
# LR>=1 => positive association: white --> POS_COLOR (method-specific)
LR_DIV_LOW = 0.1     # LR at which the negative side saturates
LR_PIVOT = 1.0       # white pivot
LR_DIV_HIGH = 20.0   # LR at which the positive side saturates
NEG_COLOR = "#c0392b"   # muted red for under-association


def diverging_lr_color(lr: float, pos_color: str) -> tuple:
    if lr is None or lr <= 0:
        return (1.0, 1.0, 1.0)
    log_lr = np.log10(lr)
    if log_lr >= 0:
        log_hi = np.log10(LR_DIV_HIGH)
        sat = float(np.clip(log_lr / log_hi, 0.0, 1.0))
        base_rgb = np.array(mcolors.to_rgb(pos_color))
    else:
        log_lo = -np.log10(LR_DIV_LOW)  # positive magnitude
        sat = float(np.clip(-log_lr / log_lo, 0.0, 1.0))
        base_rgb = np.array(mcolors.to_rgb(NEG_COLOR))
    rgb = (1 - sat) * np.ones(3) + sat * base_rgb
    return tuple(rgb)


def radius_for_count(co: float, max_co: float) -> float:
    if co <= 0 or max_co <= 0:
        return 0.0
    return MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * np.sqrt(co / max_co)


def draw_panel(ax, lr, sig, co, top_m, top_c, method_color,
               metric_label, crit_label, title, max_co, show_ylabel):
    n_m, n_c = len(top_m), len(top_c)
    for mi in range(n_m):
        for ci in range(n_c):
            co_val = co[mi, ci]
            if co_val <= 0:
                continue
            r = radius_for_count(co_val, max_co)
            face = diverging_lr_color(lr[mi, ci], method_color)
            if sig[mi, ci]:
                edge_color = SIG_EDGE_COLOR
                edge_width = SIG_EDGE_WIDTH
            else:
                edge_color = "#cccccc"
                edge_width = 0.3
            ax.add_patch(mpatches.Circle(
                (ci, mi), r, facecolor=face, edgecolor=edge_color,
                linewidth=edge_width, alpha=0.95, zorder=3,
            ))
    ax.set_xlim(-0.6, n_c - 0.4)
    ax.set_ylim(n_m - 0.4, -0.6)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(
        [crit_label(c) for c in top_c],
        rotation=35, ha="right", rotation_mode="anchor", fontsize=8,
    )
    ax.set_yticks(range(n_m))
    if show_ylabel:
        ax.set_yticklabels([metric_label(m) for m in top_m], fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, pad=6)
    ax.tick_params(axis="both", length=0)
    ax.grid(True, color="#eeeeee", linewidth=0.4, zorder=0)
    ax.set_aspect("equal")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")

    metric_display_map = load_metric_display()

    def metric_label(m: str) -> str:
        return metric_display_map.get(m, m.title())

    def crit_label(c: str) -> str:
        short = short_label(c, prefixed=False)
        return LABEL_ABBREV.get(short, short)

    laaj, human, top_m, top_c = select_overall(
        papers, TOP_METRICS, TOP_CRITERIA,
    )
    lr_l, lr_h, sig_l, sig_h, co_l, co_h = compute_per_method(
        laaj, human, top_m, top_c,
    )
    max_co = float(max(co_l.max(), co_h.max()))
    print(f"n_LaaJ={len(laaj)}  n_Human={len(human)}")
    print(f"max co-occurrence (any cell, any method): {int(max_co)}")
    print(f"sig cells: LaaJ={int(sig_l.sum())}/{sig_l.size}, "
          f"Human={int(sig_h.sum())}/{sig_h.size}")

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8))
    draw_panel(
        axes[0], lr_l, sig_l, co_l, top_m, top_c, LAAJ_COLOR,
        metric_label, crit_label,
        f"LLM-as-a-Judge ({len(laaj)} papers)", max_co, show_ylabel=True,
    )
    draw_panel(
        axes[1], lr_h, sig_h, co_h, top_m, top_c, HUMAN_COLOR,
        metric_label, crit_label,
        f"Human Evaluation ({len(human)} papers)", max_co, show_ylabel=False,
    )

    # ---- Bottom legend strip: size | LaaJ-LR | Human-LR | significance ----
    # Each of the four sub-legends has the same column-title row at
    # LEGEND_Y_TITLE, with the graphic element sitting below it at
    # LEGEND_Y_GFX. Reserve enough bottom margin (subplots_adjust) so
    # nothing is cropped under bbox_inches="tight".
    LEGEND_Y_TITLE = 0.085
    LEGEND_Y_GFX = 0.040

    # (1) Co-occurrence size scale ----------------------------------
    SIZE_X0 = 0.060
    SIZE_W = 0.22
    fig.text(SIZE_X0 + SIZE_W / 2, LEGEND_Y_TITLE,
             "Co-occurrence (papers)", fontsize=8.5, ha="center",
             fontweight="bold")
    size_ax = fig.add_axes([SIZE_X0, LEGEND_Y_GFX - 0.030, SIZE_W, 0.060])
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    size_ax.set_aspect("auto")
    legend_co = [10, 50, 150, 350]
    for i, c in enumerate(legend_co):
        x = 0.12 + 0.26 * i
        full_r = radius_for_count(c, max_co)
        r_scaled = full_r * 0.38
        circle = mpatches.Circle(
            (x, 0.62), r_scaled, facecolor="#bbbbbb", edgecolor="#444",
            linewidth=0.5, alpha=0.90,
        )
        size_ax.add_patch(circle)
        size_ax.text(x, 0.06, f"{c}", ha="center", va="center",
                     fontsize=7.5)
    size_ax.set_xticks([])
    size_ax.set_yticks([])
    for s in size_ax.spines.values():
        s.set_visible(False)

    # (2) & (3) LR colour ramps (diverging, log-symmetric around LR=1)
    def add_ramp(x0, w, color, title):
        fig.text(x0 + w / 2, LEGEND_Y_TITLE, title, fontsize=8.5,
                 ha="center", fontweight="bold")
        ramp_ax = fig.add_axes([x0, LEGEND_Y_GFX, w, 0.020])
        n = 256
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"div_{title}", [NEG_COLOR, "#ffffff", color], N=n,
        )
        gradient = np.linspace(0, 1, n).reshape(1, -1)
        ramp_ax.imshow(gradient, aspect="auto", cmap=cmap)
        ramp_ax.set_xticks([0, n // 2, n - 1])
        ramp_ax.set_xticklabels(
            [f"LR={LR_DIV_LOW:g}", "1", f"$\\geq${LR_DIV_HIGH:g}"],
            fontsize=7.5,
        )
        ramp_ax.set_yticks([])
        for s in ramp_ax.spines.values():
            s.set_visible(False)
        ramp_ax.tick_params(length=0, pad=2)

    add_ramp(0.31, 0.19, LAAJ_COLOR, "LaaJ-LR")
    add_ramp(0.52, 0.19, HUMAN_COLOR, "Human-LR")

    # (4) Significance marker ---------------------------------------
    SIG_X0 = 0.75
    SIG_W = 0.22
    fig.text(SIG_X0 + SIG_W / 2, LEGEND_Y_TITLE, "Significance",
             fontsize=8.5, ha="center", fontweight="bold")
    sig_ax = fig.add_axes([SIG_X0, LEGEND_Y_GFX - 0.030, SIG_W, 0.060])
    sig_ax.set_xlim(0, 1)
    sig_ax.set_ylim(0, 1)
    sig_ax.set_aspect("auto")
    sig_ax.add_patch(mpatches.Circle(
        (0.25, 0.62), 0.13, facecolor="#dddddd",
        edgecolor=SIG_EDGE_COLOR, linewidth=SIG_EDGE_WIDTH, alpha=0.95,
    ))
    sig_ax.text(0.25, 0.06, "significant", ha="center", va="center",
                fontsize=7.5)
    sig_ax.add_patch(mpatches.Circle(
        (0.70, 0.62), 0.13, facecolor="#dddddd",
        edgecolor="#cccccc", linewidth=0.3, alpha=0.95,
    ))
    sig_ax.text(0.70, 0.06, "not significant", ha="center", va="center",
                fontsize=7.5)
    sig_ax.set_xticks([])
    sig_ax.set_yticks([])
    for s in sig_ax.spines.values():
        s.set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.28,
                        wspace=0.06)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
