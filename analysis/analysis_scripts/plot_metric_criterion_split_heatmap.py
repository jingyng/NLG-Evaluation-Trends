"""
plot_metric_criterion_split_heatmap.py

Combined §4.2 figure: per-task metric x criterion heatmap where each cell is
split diagonally to encode BOTH the LaaJ-conditioned LR (upper-left
triangle, orange ramp) and the Human-conditioned LR (lower-right triangle,
blue ramp). This fuses the per-task metric-method dot plot
(plot_metric_method_task_dotplot.py) with the per-task metric-criterion
heatmap (plot_metric_criterion_task_diff.py) into a single figure: the
reader sees, in one cell, both how strongly the metric associates with the
criterion in LaaJ-using papers and in human-eval-using papers, and the
diagonal split makes LaaJ-vs-Human divergence visible at a glance.

Encoding:
  - color = method (orange = LaaJ; blue = human eval)
  - intensity = log_10(LR) clipped to [0, log_10(LR_HIGH)]; LR <= 1
    renders near-white. So strongly associated = saturated; near-baseline =
    pale.
  - hatched cell halves = not significant after G^2 + BH-FDR (per-method
    test) OR low support; the corresponding triangle is overlaid with
    diagonal hatching in light grey rather than coloured.
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
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from data_loader import load_data, short_label
from association_measures import compute_all, bh_fdr

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "metric_criterion_split_heatmap.png"
OUT_PDF = OUT_DIR / "metric_criterion_split_heatmap.pdf"
NORMALIZATION_CSV = BASE / "metadata_unique_counts" / "automatic_metrics_normalization_merges.csv"

TASKS_ORDERED = [
    ("Dialogue Generation",   "Dialogue Generation"),
    ("Machine Translation",   "Machine Translation"),
    ("Text Summarization",    "Text Summarization"),
    ("Question Answering",    "Question Answering"),
]

N_FREQ = 5     # top-5 by frequency in the task
N_DISTINCT = 5 # top-5 by task-LR among items not in the frequency set
TOP_METRICS_PER_TASK = N_FREQ + N_DISTINCT
TOP_CRITERIA_PER_TASK = N_FREQ + N_DISTINCT
MIN_METRIC_COUNT = 10   # metric must appear in >= this many task papers
MIN_CRIT_COUNT = 8      # criterion must appear in >= this many criteria slots
SIG_Q = 0.05
MIN_PAIR_CO_OCC = 5
NONSIG_ALPHA = 0.18
SIG_EDGE_COLOR = "#333333"
SIG_EDGE_WIDTH = 0.7

# Tick-label styling: frequent items stay default black (no special
# handling); task-distinctive items get a red label + leading star, so
# the band membership is signalled by a single marked-exception style
# rather than two custom colours.
DISTINCT_LABEL_COLOR = "#c0392b"    # red

LAAJ_COLOR = "#ff7f0e"   # orange
HUMAN_COLOR = "#1f77b4"  # blue

LR_LOW = 1.0     # LR at which colour starts (white)
LR_HIGH = 20.0   # LR at which colour saturates

# Figure-only abbreviations for criterion labels. The first group mirrors
# LABEL_ABBREV in plot_task_dashboard_rich_v2.py (Figure 4) so the same
# criteria get the same short names across figures. The second group is
# specific to this figure: long criterion names that appear in the per-
# task task-distinctive set but are too long to fit a heatmap column.
LABEL_ABBREV = {
    # Shared with Figure 4
    "Overall Quality / Preference": "Overall Quality",
    "Detectability of Author Trait": "Author Trait",
    "Internal Consistency of Outputs": "Internal Consistency",
    "Absence of Toxic / Harmful Content": "Non-Toxicity",
    "Empathy / Emotional Appropriateness": "Empathy",
    "Engagingness/Interestingness": "Engagingness",
    "Relative Factual Accuracy": "Rel. Factual Acc.",
    "Translation Accuracy": "Translation Acc.",
    "Consistency with Input": "Input Consistency",
    "Usefulness": "Helpfulness",
    "Usefulness for Task": "Task Usefulness",
    "Quality as Explanation of Input": "Explanation Quality",
    "Output Answers Question": "Answers the Question",
    # Figure-5-specific (long task-distinctive criteria)
    "Appropriateness of System Response Type": "Response Approp.",
    "Adherence to Style Guide": "Style Adherence",
    "Control over Style": "Style Control",
    "Effect on User Stance": "User Stance Effect",
    "Effect on User Emotion": "User Emotion Effect",
    "Similarity to Target Outputs (content)": "Target Similarity",
    "Absence of Bias / Stereotypes": "Non-Bias",
    "Coverage of Topics": "Topic Coverage",
    "Similarity to Target Outputs": "Target Similarity",
}


def load_metric_display() -> dict[str, str]:
    out: dict[str, str] = {}
    if not NORMALIZATION_CSV.exists():
        return out
    import csv
    with open(NORMALIZATION_CSV) as f:
        for row in csv.DictReader(f):
            normalized = (row.get("normalized") or "").strip()
            variants = (row.get("variants_with_counts") or "").split(";")
            if not normalized or not variants:
                continue
            top = variants[0].strip().rsplit("(", 1)[0].strip()
            out[normalized.lower()] = top
    return out


def select_for_task(papers, task_name, top_metrics, top_criteria):
    """Pick metrics & criteria as the union of:
      (A) the top N_FREQ items by frequency in the task (the "popular" set)
      (B) the top N_DISTINCT items by task-LR (LR pooled with task) among
          items NOT in (A).

    Returns laaj_papers, human_papers, top_m, top_c, m_groups, c_groups
    where *_groups label each item as 'freq' or 'distinct' so the renderer
    can mark them differently. Items are returned ordered (A first, then
    B) so the figure displays a clear 5+5 banded layout per axis.
    """
    task_low = task_name.lower().strip()
    task_papers = [p for p in papers if any(t.lower().strip() == task_low for t in (p.get("tasks") or []))]
    other_papers = [p for p in papers if p not in task_papers]
    n_task = len(task_papers)
    n_other = len(other_papers)

    laaj_papers = [p for p in task_papers if p.get("laaj_criteria")]
    human_papers = [p for p in task_papers if p.get("human_criteria")]

    metric_counts: Counter = Counter()
    for p in task_papers:
        for m in (p.get("auto_metrics") or []):
            metric_counts[m.lower().strip()] += 1
    cand_m = [m for m, c in metric_counts.items() if c >= MIN_METRIC_COUNT]

    crit_counts: Counter = Counter()
    for p in task_papers:
        for c in (p.get("laaj_criteria") or []):
            crit_counts[c.strip()] += 1
        for c in (p.get("human_criteria") or []):
            crit_counts[c.strip()] += 1
    cand_c = [c for c, n in crit_counts.items() if n >= MIN_CRIT_COUNT]

    if not cand_m or not cand_c:
        return laaj_papers, human_papers, cand_m[:top_metrics], cand_c[:top_criteria], [], []

    def metric_in(p, m_low):
        return m_low in {x.lower().strip() for x in (p.get("auto_metrics") or [])}

    def crit_in(p, c):
        return c in {x.strip() for x in (p.get("laaj_criteria") or [])} \
            or c in {x.strip() for x in (p.get("human_criteria") or [])}

    metric_lrs = []
    for m in cand_m:
        k11 = sum(1 for p in task_papers if metric_in(p, m))
        k21 = sum(1 for p in other_papers if metric_in(p, m))
        stats = compute_all(k11, n_task - k11, k21, n_other - k21)
        lr = stats.get("lr")
        metric_lrs.append(lr if lr is not None and np.isfinite(lr) else 1.0)

    crit_lrs = []
    for c in cand_c:
        k11 = sum(1 for p in task_papers if crit_in(p, c))
        k21 = sum(1 for p in other_papers if crit_in(p, c))
        stats = compute_all(k11, n_task - k11, k21, n_other - k21)
        lr = stats.get("lr")
        crit_lrs.append(lr if lr is not None and np.isfinite(lr) else 1.0)

    def split_freq_distinct(items, counts, lrs, n_freq, n_distinct,
                             distinct_first=False):
        n = len(items)
        freq_idx = sorted(range(n), key=lambda i: -counts[items[i]])[:n_freq]
        freq_set = set(freq_idx)
        rest = [i for i in range(n) if i not in freq_set]
        rest_sorted = sorted(rest, key=lambda i: -lrs[i])[:n_distinct]
        if distinct_first:
            # Full reversal of the row order under the inverted y-axis:
            # the distinct band sits at the top with items in ASCENDING
            # task-LR order (lowest LR at the very top of the panel,
            # highest LR just above the band divider); the frequency
            # band sits at the bottom with items in ASCENDING frequency
            # order (least-frequent just below the divider, most-frequent
            # at the very bottom). So the strongest items in each band
            # meet in the middle, and the dominant items (highest LR
            # distinct, most-frequent) anchor the centre and bottom.
            chosen = list(reversed(rest_sorted)) + list(reversed(freq_idx))
            groups = ["distinct"] * len(rest_sorted) + ["freq"] * len(freq_idx)
        else:
            # Frequency band first (low index = displayed at left under
            # the normal x-axis); used for the column axis. Within-band
            # order is descending (most-frequent on the far left of the
            # frequency band; highest-LR on the far left of the distinct
            # band).
            chosen = list(freq_idx) + list(rest_sorted)
            groups = ["freq"] * len(freq_idx) + ["distinct"] * len(rest_sorted)
        return [items[i] for i in chosen], groups

    # Rows (metrics) — distinct band on top, frequency band on bottom.
    top_m, m_groups = split_freq_distinct(
        cand_m, metric_counts, metric_lrs, N_FREQ, N_DISTINCT,
        distinct_first=True)
    # Columns (criteria) — frequency band on left, distinct on right.
    top_c, c_groups = split_freq_distinct(
        cand_c, crit_counts, crit_lrs, N_FREQ, N_DISTINCT,
        distinct_first=False)
    return laaj_papers, human_papers, top_m, top_c, m_groups, c_groups


def compute_per_method(laaj_papers, human_papers, top_metrics, top_criteria):
    """Per-method LR matrix + significance mask + co-occ count for each
    (metric, criterion) cell, separately for LaaJ- and human-criteria
    subsets."""
    n_m, n_c = len(top_metrics), len(top_criteria)
    lr_l = np.ones((n_m, n_c)); lr_h = np.ones((n_m, n_c))
    p_l = np.ones((n_m, n_c));  p_h = np.ones((n_m, n_c))
    co_l = np.zeros((n_m, n_c)); co_h = np.zeros((n_m, n_c))

    def fill(papers, lr, p, co, crit_field):
        if not papers:
            return
        metric_sets = [{x.lower().strip() for x in (q.get("auto_metrics") or [])} for q in papers]
        crit_sets = [{x.lower().strip() for x in (q.get(crit_field) or [])} for q in papers]
        for ci, c in enumerate(top_criteria):
            cl = c.lower().strip()
            with_idx = [i for i, s in enumerate(crit_sets) if cl in s]
            without_idx = [i for i, s in enumerate(crit_sets) if cl not in s]
            n_w, n_wo = len(with_idx), len(without_idx)
            if n_w == 0 or n_wo == 0:
                continue
            for mi, m in enumerate(top_metrics):
                ml = m.lower().strip()
                k_w = sum(1 for i in with_idx if ml in metric_sets[i])
                k_wo = sum(1 for i in without_idx if ml in metric_sets[i])
                stats = compute_all(k_w, n_w - k_w, k_wo, n_wo - k_wo)
                lr_val = stats.get("lr"); p_val = stats.get("p_value", 1.0)
                if lr_val is None or not np.isfinite(lr_val):
                    lr_val = 1.0
                lr[mi, ci] = lr_val
                p[mi, ci] = p_val
                co[mi, ci] = k_w

    fill(laaj_papers,  lr_l, p_l, co_l, "laaj_criteria")
    fill(human_papers, lr_h, p_h, co_h, "human_criteria")

    flat_q_l, _ = bh_fdr(p_l.flatten().tolist())
    flat_q_h, _ = bh_fdr(p_h.flatten().tolist())
    q_l = np.array(flat_q_l).reshape(p_l.shape)
    q_h = np.array(flat_q_h).reshape(p_h.shape)

    sig_l = (q_l <= SIG_Q) & (co_l >= MIN_PAIR_CO_OCC)
    sig_h = (q_h <= SIG_Q) & (co_h >= MIN_PAIR_CO_OCC)
    return lr_l, lr_h, sig_l, sig_h, co_l, co_h


def lr_to_color(lr: float, base_color: str) -> tuple:
    """Map LR to a color: white at LR<=LR_LOW, base_color at LR>=LR_HIGH,
    log-linear interpolation between."""
    if lr is None or lr <= 0:
        return (1.0, 1.0, 1.0)
    log_lr = np.log10(max(lr, 1e-3))
    log_lo = np.log10(LR_LOW)
    log_hi = np.log10(LR_HIGH)
    sat = np.clip((log_lr - log_lo) / (log_hi - log_lo), 0.0, 1.0)
    base_rgb = np.array(mcolors.to_rgb(base_color))
    rgb = (1 - sat) * np.ones(3) + sat * base_rgb
    return tuple(rgb)


def draw_split_cell(ax, x, y, lr_laaj, lr_human, sig_laaj, sig_human):
    """Draw two triangles at cell (x, y).
    Upper-left triangle = LaaJ; lower-right triangle = Human.

    Significant triangles get a dark outline + full opacity; non-
    significant triangles get no outline and are faded to NONSIG_ALPHA,
    so the eye picks up significance via the crisp border."""
    laaj_color = lr_to_color(lr_laaj, LAAJ_COLOR)
    if sig_laaj:
        upper = mpatches.Polygon(
            [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x - 0.5, y + 0.5)],
            facecolor=laaj_color, edgecolor=SIG_EDGE_COLOR,
            linewidth=SIG_EDGE_WIDTH, alpha=1.0, zorder=3,
        )
    else:
        upper = mpatches.Polygon(
            [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x - 0.5, y + 0.5)],
            facecolor=laaj_color, edgecolor="white", linewidth=0.4,
            alpha=NONSIG_ALPHA, zorder=2,
        )

    human_color = lr_to_color(lr_human, HUMAN_COLOR)
    if sig_human:
        lower = mpatches.Polygon(
            [(x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)],
            facecolor=human_color, edgecolor=SIG_EDGE_COLOR,
            linewidth=SIG_EDGE_WIDTH, alpha=1.0, zorder=3,
        )
    else:
        lower = mpatches.Polygon(
            [(x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)],
            facecolor=human_color, edgecolor="white", linewidth=0.4,
            alpha=NONSIG_ALPHA, zorder=2,
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
        lr_l, lr_h, sig_l, sig_h, co_l, co_h = compute_per_method(
            laaj, human, top_m, top_c,
        )
        task_data.append({
            "task": task_name, "label": label,
            "n_laaj": len(laaj), "n_human": len(human),
            "metrics": top_m, "criteria": top_c,
            "m_groups": m_groups, "c_groups": c_groups,
            "lr_l": lr_l, "lr_h": lr_h,
            "sig_l": sig_l, "sig_h": sig_h,
            "co_l": co_l, "co_h": co_h,
        })
        print(f"\n{label}: n_LaaJ={len(laaj)} n_Human={len(human)}")
        print(f"  metrics: {[metric_label(m) for m in top_m]}")
        print(f"  criteria: {[short_label(c, prefixed=False) for c in top_c]}")
        print(f"  sig LaaJ cells: {int(sig_l.sum())} / {sig_l.size}")
        print(f"  sig Human cells: {int(sig_h.sum())} / {sig_h.size}")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.4))

    for ax, td in zip(axes.flat, task_data):
        n_m = len(td["metrics"]); n_c = len(td["criteria"])

        # Quadrant highlight: shade the frequency-by-frequency quadrant.
        # Data layout: rows 0..N_DISTINCT-1 are distinct metrics, rows
        # N_DISTINCT..n_m-1 are frequent metrics; cols 0..N_FREQ-1 are
        # frequent criteria, cols N_FREQ..n_c-1 are distinct criteria.
        # With y-axis inverted at render time, the freq metric rows sit
        # at the visual BOTTOM of each panel, so the freq x freq quadrant
        # is the bottom-left of the rendered panel.
        n_dist_m = n_m - N_FREQ  # distinct metric rows (top of data array)
        if N_FREQ <= n_m and N_FREQ <= n_c:
            ax.add_patch(mpatches.Rectangle(
                (-0.5, n_dist_m - 0.5),  # data coords: x left, y at freq-metric boundary
                N_FREQ,                  # width  = freq criteria span
                n_m - n_dist_m,          # height = freq metric span
                facecolor="#FFE188", alpha=0.20,
                edgecolor="none", zorder=0,
            ))

        for mi in range(n_m):
            for ci in range(n_c):
                draw_split_cell(
                    ax, ci, mi,
                    td["lr_l"][mi, ci], td["lr_h"][mi, ci],
                    bool(td["sig_l"][mi, ci]), bool(td["sig_h"][mi, ci]),
                )

        ax.set_xlim(-0.5, n_c - 0.5)
        ax.set_ylim(n_m - 0.5, -0.5)  # invert y
        def crit_label(c):
            short = short_label(c, prefixed=False)
            return LABEL_ABBREV.get(short, short)
        ax.set_xticks(range(n_c))
        x_text = ax.set_xticklabels(
            [crit_label(c) for c in td["criteria"]],
            rotation=35, ha="right", rotation_mode="anchor", fontsize=8.5,
        )
        ax.set_yticks(range(n_m))
        y_text = ax.set_yticklabels(
            [metric_label(m) for m in td["metrics"]], fontsize=8.5,
        )
        # Visual cue: frequent items stay default black; task-distinctive
        # items get a red label. One marked-exception style; no star
        # prefix (red alone is sufficient distinction).
        for label, grp in zip(y_text, td["m_groups"]):
            if grp == "distinct":
                label.set_color(DISTINCT_LABEL_COLOR)
        for label, grp in zip(x_text, td["c_groups"]):
            if grp == "distinct":
                label.set_color(DISTINCT_LABEL_COLOR)
        # Divider lines between the freq and distinct bands
        if N_FREQ < n_m:
            ax.axhline(N_FREQ - 0.5, color="#888888", linewidth=0.6,
                       linestyle=":", zorder=4)
        if N_FREQ < n_c:
            ax.axvline(N_FREQ - 0.5, color="#888888", linewidth=0.6,
                       linestyle=":", zorder=4)
        ax.set_title(
            f"{td['label']}  (LaaJ {td['n_laaj']}, Human {td['n_human']})",
            fontsize=10, pad=4,
        )
        ax.tick_params(axis="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    # Single-row legend. Each method has its own triangle icon (showing
    # WHICH triangle in a cell corresponds to that method) followed by a
    # text label and a colour ramp.
    def add_method_legend(fig, x_tri, side, x_text, label,
                          x_bar, base_color, bar_width=0.08):
        # Triangle icon: just the filled triangle, no outline of the
        # other half. The triangle's position alone (upper-left vs
        # lower-right within the small square axis) signals geometry.
        tri_ax = fig.add_axes([x_tri, 0.040, 0.022, 0.040])
        if side == "upper":
            filled = mpatches.Polygon(
                [(0, 1), (1, 1), (0, 0)],
                facecolor=base_color, edgecolor="#333",
                linewidth=0.5, alpha=0.9)
        else:  # "lower"
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

        # Text label
        fig.text(x_text, 0.058, label, fontsize=8.5, va="center")

        # Colour ramp
        ramp_ax = fig.add_axes([x_bar, 0.052, bar_width, 0.012])
        n = 256
        gradient = np.linspace(0, 1, n).reshape(1, -1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"white_{label}", ["#ffffff", base_color], N=n,
        )
        ramp_ax.imshow(gradient, aspect="auto", cmap=cmap)
        ramp_ax.set_xticks([0, n - 1])
        ramp_ax.set_xticklabels([f"LR={LR_LOW:g}", f"$\\geq${LR_HIGH:g}"],
                                 fontsize=7.5)
        ramp_ax.set_yticks([])
        for s in ramp_ax.spines.values():
            s.set_visible(False)
        ramp_ax.tick_params(length=0, pad=1)
        # Anchor the LR=1 label leftward and the >=20 label rightward so
        # they do not spill into adjacent text on either side.
        rlabels = ramp_ax.get_xticklabels()
        rlabels[0].set_horizontalalignment("left")
        rlabels[1].set_horizontalalignment("right")

    # Centred single-row legend: [LaaJ tri][LaaJ-LR:][bar] [Human tri][Human-LR:][bar] [Red-text key]
    add_method_legend(fig, 0.165, "upper", 0.195, "LaaJ-LR:", 0.255, LAAJ_COLOR)
    add_method_legend(fig, 0.365, "lower", 0.395, "Human-LR:", 0.455, HUMAN_COLOR)

    # Task-distinctive label key. Frequent items use default black, so no
    # separate entry is needed. No star prefix --- red alone is the cue.
    fig.text(0.565, 0.058, "Red text", fontsize=8.5, va="center",
             color=DISTINCT_LABEL_COLOR)
    fig.text(0.612, 0.058,
             ": top-5 by task-LR (others: top-5 by frequency)",
             fontsize=8.5, va="center")

    # Significance encoding (outlined sig cells vs faded non-sig) is
    # described in the caption rather than in an inline legend.

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.32,
                        wspace=0.55)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
