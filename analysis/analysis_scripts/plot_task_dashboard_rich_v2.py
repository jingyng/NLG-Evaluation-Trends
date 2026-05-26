#!/usr/bin/env python3
"""Redesigned task dashboard (v2).

Improvements over plot_task_dashboard_rich.py:
  - Top-5 items per panel instead of top-10, selected by per-task frequency
    (i.e., what the field actually uses most often when evaluating this
    task).
  - LR (relative risk) and G² log-likelihood test computed per (task, item)
    over the full corpus, with BH-FDR within each (task, field) family.
    The significance verdict only affects rendering: items whose LR is
    significant after BH-FDR (q <= 0.05) are drawn with solid line + filled
    markers; popular-but-not-distinctive items get dashed line + hollow
    markers + LR shown in italic — flagging "this is widely used here but
    not specific to the task".
  - Labels at line endpoints only (start + end), no per-year cluttering.
  - Drop bubble-size encoding for prevalence (uniform marker).
  - Color by QCET L1 axis ([QO]/[QI]/[QT]/[QE] + auto-metrics + AUX) so the
    figure tells a categorical story instead of asking the reader to track
    30 individual lines.

Layout: 4 columns (DG / MT / TS / QA) x 3 rows (auto metrics / human criteria
/ LaaJ criteria), 12 panels total. Each panel: rank-over-time bump chart of
the top-5 most-frequent items in that task, with G² significance shown by
line style.

Run from `paper_code/`:

    python 07_figures/plot_task_dashboard_rich_v2.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data, short_label
from association_measures import compute_all, bh_fdr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_N = 8

# Figure-only abbreviations to keep endpoint labels short. The map is keyed
# on the *short_label()* output (the bare short name from qcet_labels.py),
# since `short_label()` is applied to the raw criterion before this map is
# consulted.
LABEL_ABBREV = {
    "Overall Quality / Preference": "Overall Quality",
    "Detectability of Author Trait": "Author Trait",
    "Internal Consistency of Outputs": "Internal Consistency",
    "Absence of Toxic / Harmful Content": "Non-Toxicity",
    "Empathy / Emotional Appropriateness": "Empathy",
    "Engagingness/Interestingness": "Engagingness",
    "Nonredundancy (form)": "Nonredundancy",
    "Relative Factual Accuracy": "Rel. Factual Acc.",
    "Translation Accuracy": "Translation Acc.",
    "Consistency with Input": "Input Consistency",
    "Usefulness": "Helpfulness",
    "Usefulness for Task": "Task Usefulness",
    "Quality as Explanation of Input": "Explanation Quality",
    "Output Answers Question": "Answers the Question",
}
TOP_TASKS = [
    ("dialogue generation",   "Dialogue Generation"),
    ("machine translation",   "Machine Translation"),
    ("text summarization",    "Text Summarization"),
    ("question answering",    "Question Answering"),
]
FIELDS = [
    ("auto_metrics",    "Automatic Metrics"),
    ("human_criteria",  "Human Criteria"),
    ("laaj_criteria",   "LaaJ Criteria"),
]
YEARS = list(range(2020, 2026))                   # 2020..2025 inclusive

# QCET L1 → color (color = role on the lattice). Auto-metrics and AUX are off-lattice.
L1_COLORS = {
    "QO": "#3b6ea8",     # blue   — Own right
    "QI": "#2a8a55",     # green  — Input-relative
    "QT": "#cc7a00",     # orange — Target-relative
    "QE": "#8a3a8a",     # purple — External-relative
    "AUX": "#6f6f6f",    # grey   — auxiliary (off-lattice)
    "AM":  "#444444",    # dark grey — automatic metrics
}
L1_NAMES = {
    "QO": "QO: Quality (intrinsic)",
    "QI": "QI: Quality (vs. input)",
    "QT": "QT: Quality (vs. target)",
    "QE": "QE: Quality (vs. external)",
    "AUX": "AUX: QCET extension",
    "AM":  "Auto. metric",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _l1_axis(item: str, field_key: str) -> str:
    """Best-effort L1 classification for a display name. The criteria fields
    contain QCET full names (which start with the leaf 'name' but we don't
    have the id directly here), so we infer L1 from a small lookup over the
    short_label table. For auto_metrics, return 'AM'."""
    if field_key == "auto_metrics":
        return "AM"
    return _NAME_TO_L1.get(item.strip().lower(), "AUX")


def _build_name_to_l1() -> dict[str, str]:
    """Build a name → L1 lookup using the shared QCET label module."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root / "src" / "qcet_normalization"))
    from qcet_labels import build_label_map

    classifications_csv = (repo_root / "analysis" / "intermediate_results"
                           / "qcet" / "criteria_classifications_final.csv")
    return {name_lower: (l1 if l1 else "AUX")
            for name_lower, (_name, l1, _bare, _prefixed)
            in build_label_map(classifications_csv).items()}


_NAME_TO_L1 = _build_name_to_l1()


def _pure_task_papers(papers, task_lower: str):
    return [p for p in papers
            if len(p["tasks"]) == 1
            and p["tasks"][0].strip().lower() == task_lower]


def _items(paper, field_key: str) -> list[str]:
    raw = paper.get(field_key, []) or []
    return [str(x).strip() for x in raw if x]


def _select_top_by_frequency(
    all_papers,
    papers_for_task,
    field_key: str,
    top_n: int,
    min_count: int = 3,
    fdr_q: float = 0.05,
) -> tuple[list[str], dict[str, dict]]:
    """Pick the top-N most-frequent items in the task and annotate each
    with its task-vs-other LR plus G² significance (BH-FDR within the
    (task, field) family).

    Returns:
        (chosen_terms, stats) where stats[term] = {
            'lr': float, 'g2': float, 'q': float, 'reject_q05': bool,
            'k11': int, 'count_in_task': int,
        }
    """
    n_task  = len(papers_for_task)
    other   = [p for p in all_papers if p not in papers_for_task]
    n_other = len(other)
    if n_task == 0 or n_other == 0:
        return [], {}

    c_task:  Counter = Counter()
    c_other: Counter = Counter()
    for p in papers_for_task:
        c_task.update(set(_items(p, field_key)))
    for p in other:
        c_other.update(set(_items(p, field_key)))

    # Candidate pool = items above min_count in this task. We compute LR/G²
    # over the candidate pool so BH-FDR has an honest family size; we then
    # show only the top_n most-frequent items.
    cands = [t for t, n in c_task.items() if n >= min_count]
    if not cands:
        return [], {}

    k11 = np.array([c_task[t] for t in cands], dtype=float)
    k12 = np.array([n_task - c_task[t] for t in cands], dtype=float)
    k21 = np.array([c_other.get(t, 0) for t in cands], dtype=float)
    k22 = np.array([n_other - c_other.get(t, 0) for t in cands], dtype=float)
    out = compute_all(k11, k12, k21, k22, fdr_q=fdr_q)
    lr  = np.atleast_1d(out["lr"])
    g2v = np.atleast_1d(out["g2"])
    q   = np.atleast_1d(out["q_value"])

    # Pick top_n by raw frequency in the task (popularity).
    by_freq = sorted(range(len(cands)), key=lambda i: -k11[i])[:top_n]
    chosen = [cands[i] for i in by_freq]
    stats = {
        cands[i]: {
            "lr":   float(lr[i]),
            "g2":   float(g2v[i]),
            "q":    float(q[i]),
            "reject_q05": bool(q[i] <= 0.05),
            "k11":  int(k11[i]),
            "count_in_task": int(c_task[cands[i]]),
        }
        for i in by_freq
    }
    return chosen, stats


def _ranks_per_year(papers_for_task, field_key: str, chosen_terms: list[str]):
    """Per-year frequency ranks AMONG THE CHOSEN TERMS ONLY.

    Returns ranks {year: {term: rank}} (rank 1 = most-mentioned that year
    among the 5). Terms with 0 mentions in a year get rank=None.
    """
    by_year: dict[int, Counter] = {y: Counter() for y in YEARS}
    for p in papers_for_task:
        y = p.get("year")
        if not isinstance(y, int) or y not in by_year:
            continue
        for t in _items(p, field_key):
            if t in chosen_terms:
                by_year[y][t] += 1

    ranks: dict[int, dict[str, int]] = {y: {} for y in YEARS}
    for y, c in by_year.items():
        present = [t for t in chosen_terms if c.get(t, 0) > 0]
        present.sort(key=lambda t: -c[t])
        for i, t in enumerate(present, start=1):
            ranks[y][t] = i
    return ranks


def _per_year_lr(
    all_papers,
    papers_for_task,
    field_key: str,
    chosen_terms: list[str],
) -> dict[int, dict[str, float]]:
    """LR per (year, term) for the chosen terms, with k21=0 clipped to a
    finite ceiling so marker sizes stay sane.

        LR = (k11/n_task_year) / max(k21/n_other_year, 1/n_other_year)
    """
    chosen_set = set(chosen_terms)
    # All papers grouped by year (for the "other" denominator).
    by_year_other: dict[int, list] = {y: [] for y in YEARS}
    by_year_task:  dict[int, list] = {y: [] for y in YEARS}
    task_set_id = {id(p) for p in papers_for_task}
    for p in all_papers:
        y = p.get("year")
        if not isinstance(y, int) or y not in by_year_other:
            continue
        if id(p) in task_set_id:
            by_year_task[y].append(p)
        else:
            by_year_other[y].append(p)

    out: dict[int, dict[str, float]] = {y: {} for y in YEARS}
    for y in YEARS:
        n_task_y  = len(by_year_task[y])
        n_other_y = len(by_year_other[y])
        if n_task_y == 0:
            continue
        c_t: Counter = Counter()
        c_o: Counter = Counter()
        for p in by_year_task[y]:
            c_t.update(set(_items(p, field_key)) & chosen_set)
        for p in by_year_other[y]:
            c_o.update(set(_items(p, field_key)) & chosen_set)
        for t in chosen_terms:
            k11 = c_t.get(t, 0)
            if k11 == 0:
                continue
            p_t = k11 / n_task_y
            p_o = max(c_o.get(t, 0) / max(n_other_y, 1), 1.0 / max(n_other_y, 1))
            out[y][t] = p_t / p_o
    return out


def _lr_to_size(lr: float) -> float:
    """Map per-year LR to scatter marker area in points^2.
    Clipped so max scatter is ~4x min."""
    if lr is None or lr <= 0:
        return 22.0
    s = 22.0 + 28.0 * np.log10(max(lr, 0.5))
    return max(18.0, min(s, 110.0))


def _cell_totals(papers_for_task, field_key: str) -> dict[int, int]:
    """Per-year count of papers in this (task, paradigm) cell, i.e., papers
    in `papers_for_task` for that year that have at least one entry in
    `field_key`. Used to annotate the x-axis so a reader can see how thin
    the early-year columns are relative to recent ones."""
    out = {y: 0 for y in YEARS}
    for p in papers_for_task:
        y = p.get("year")
        if isinstance(y, int) and y in out and _items(p, field_key):
            out[y] += 1
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _label_positions(chosen, ranks, final_year, top_n):
    """Return per-term label placement.

    All labels live at the right edge (x = final_year), so the eye can scan
    them in a single column.

    - Items present in `final_year` sit at their final-year rank.
    - Items that dropped out before `final_year` are assigned an empty
      final-year row (in increasing-last_rank order, filling empty rows
      top-to-bottom) and we record their actual last marker so the caller
      can draw a connector from the marker to the label.

    Returns: dict[term -> {label_x, label_y, marker_x, marker_y, is_dropped}].
    """
    in_range: list[tuple[str, int]] = []
    dropped: list[tuple[str, int, int]] = []  # (term, last_year, last_rank)
    used_rows: set[int] = set()
    for term in chosen:
        r_final = ranks[final_year].get(term)
        if r_final is not None:
            in_range.append((term, r_final))
            used_rows.add(r_final)
            continue
        last_year = None
        last_rank = None
        for y in sorted(ranks.keys(), reverse=True):
            if y < final_year and ranks[y].get(term) is not None:
                last_year = y
                last_rank = ranks[y][term]
                break
        if last_year is not None:
            dropped.append((term, last_year, last_rank))

    out: dict[str, dict] = {}
    for term, rank in in_range:
        out[term] = {
            "label_x": float(final_year),
            "label_y": float(rank),
            "marker_x": None,
            "marker_y": None,
            "is_dropped": False,
        }
    empty_rows = sorted(r for r in range(1, top_n + 1) if r not in used_rows)
    dropped.sort(key=lambda x: x[2])  # by last_rank ascending
    for (term, last_year, last_rank), label_row in zip(dropped, empty_rows):
        out[term] = {
            "label_x": float(final_year),
            "label_y": float(label_row),
            "marker_x": float(last_year),
            "marker_y": float(last_rank),
            "is_dropped": True,
        }
    return out


def _plot_panel(ax, all_papers, papers_for_task, field_key: str,
                label_right: bool = True):
    # Pick top-N by frequency; annotate with overall LR + G² significance.
    min_count = 5 if field_key == "auto_metrics" else 3
    chosen, stats = _select_top_by_frequency(
        all_papers, papers_for_task, field_key, TOP_N, min_count=min_count
    )
    ranks = _ranks_per_year(papers_for_task, field_key, chosen)
    lr_y  = _per_year_lr(all_papers, papers_for_task, field_key, chosen)
    cell_n = _cell_totals(papers_for_task, field_key)
    x = YEARS
    label_pos = _label_positions(chosen, ranks, max(YEARS), TOP_N)

    drawn_l1: set = set()
    for term in chosen:
        l1 = _l1_axis(term, field_key)
        color = L1_COLORS.get(l1, "#999999")
        drawn_l1.add(l1)

        sig = stats.get(term, {}).get("reject_q05", False)
        ys = [ranks[y].get(term) for y in x]

        # Bridge line: a faint, dotted line connecting all non-None points
        # across gaps so a trajectory that drops out of the top-N for a year
        # and returns can still be traced. Pairs of bridge points that span
        # more than one year (a missing year between them) are drawn as a
        # quadratic Bézier whose apex is displaced toward smaller y (up in
        # the inverted axis), so the curve arcs over any in-between markers
        # at the same row instead of passing straight through them.
        xs_bridge = [xi for xi, yi in zip(x, ys) if yi is not None]
        ys_bridge = [yi for yi in ys if yi is not None]
        for j in range(len(xs_bridge) - 1):
            x0, y0 = xs_bridge[j], ys_bridge[j]
            x1, y1 = xs_bridge[j + 1], ys_bridge[j + 1]
            if x1 - x0 > 1:
                bulge = 0.45  # in y-units; subtracted because y is inverted
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2 - bulge
                t = np.linspace(0, 1, 40)
                bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                ax.plot(bx, by, linestyle=(0, (1.5, 1.5)),
                        color=color, linewidth=0.9, alpha=0.55, zorder=1)
            else:
                ax.plot([x0, x1], [y0, y1], linestyle=(0, (1.5, 1.5)),
                        color=color, linewidth=0.9, alpha=0.55, zorder=1)

        # Line: solid for significant, dashed for non-significant.
        line_style = "-" if sig else (0, (3, 2))   # dash pattern when not sig
        line_alpha = 0.9 if sig else 0.55
        seg_x: list[int] = []
        seg_y: list[int] = []
        for xi, yi in zip(x, ys):
            if yi is None:
                if seg_x:
                    ax.plot(seg_x, seg_y, linestyle=line_style, color=color,
                            linewidth=1.6, alpha=line_alpha, zorder=2)
                    seg_x, seg_y = [], []
                continue
            seg_x.append(xi)
            seg_y.append(yi)
        if seg_x:
            ax.plot(seg_x, seg_y, linestyle=line_style, color=color,
                    linewidth=1.6, alpha=line_alpha, zorder=2)

        # Markers: size = log(per-year LR), filled if significant, hollow
        # otherwise. Per-year cell volumes are shown via the x-axis tick
        # labels (`'YY n=K`) rather than marker size.
        for xi, yi in zip(x, ys):
            if yi is None:
                continue
            size = _lr_to_size(lr_y.get(xi, {}).get(term))
            if sig:
                ax.scatter([xi], [yi], color=color, s=size, zorder=3,
                           edgecolors="white", linewidths=1.0)
            else:
                ax.scatter([xi], [yi], facecolors="white", s=size, zorder=3,
                           edgecolors=color, linewidths=1.4)

        # Endpoint labels: all anchored at the right edge.
        #   - In-range items sit at their final-year rank.
        #   - Dropped items sit in an empty final-year row (assigned by
        #     `_label_positions`); we draw a thin connector from the actual
        #     last marker to the label so the trajectory stays traceable.
        last = label_pos.get(term)
        lr_val = stats.get(term, {}).get("lr", 0.0)
        if field_key == "auto_metrics":
            label_term = term
        else:
            label_term = short_label(term, prefixed=False)
        # Apply figure-only abbreviations
        label_term = LABEL_ABBREV.get(label_term, label_term)

        if last is not None and label_right:
            label_x = last["label_x"] + 0.25
            label_y = last["label_y"]
            label_va = "center"
            if last["is_dropped"]:
                ax.plot(
                    [last["marker_x"], last["label_x"]],
                    [last["marker_y"], last["label_y"]],
                    color=color, alpha=0.45, linewidth=0.7,
                    linestyle=(0, (3, 2)), zorder=2, clip_on=True,
                )
            # Single-line endpoint label: name in normal/italic at 7.5pt,
            # followed by a smaller "(LR=X.X)" suffix at 6pt placed via
            # an offset_points annotation that uses the rendered text width
            # of the name (so it stays right after the name regardless of
            # length).
            name_artist = ax.text(
                label_x, label_y,
                label_term,
                fontsize=7.5, ha="left", va=label_va,
                color=color,
                fontweight="bold" if sig else "normal",
                style="normal" if sig else "italic",
                zorder=4, clip_on=True,
            )
            k_papers = stats.get(term, {}).get("k11", 0)
            ax.annotate(
                f"  ({lr_val:.1f}, {int(k_papers)})",
                xycoords=name_artist,
                xy=(1.0, 0.5),
                xytext=(2, 0), textcoords="offset points",
                fontsize=6.5, ha="left", va="center",
                color="#6e6e6e",
                zorder=4, clip_on=True,
                annotation_clip=False,
            )

    # Axes styling — right margin sized so the longest criteria label plus
    # "  (LR=X.X, n.s.)" fits, while keeping data area readable. Bare short
    # labels (no [QO] prefix) are shorter, so margin can be tighter.
    ax.set_xlim(min(YEARS) - 0.4, max(YEARS) + 7.0)
    ax.set_ylim(0.5, TOP_N + 0.5)
    ax.invert_yaxis()
    ax.set_xticks(YEARS)
    ax.set_xticklabels([f"'{str(y)[2:]}" for y in YEARS], fontsize=8.5)
    ax.set_yticks(range(1, TOP_N + 1))
    ax.set_yticklabels([f"{i}" for i in range(1, TOP_N + 1)], fontsize=8)
    ax.tick_params(axis="both", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=1)

    return drawn_l1


def make_dashboard(papers, output_path: Path) -> None:
    n_tasks   = len(TOP_TASKS)
    n_fields  = len(FIELDS)
    fig, axes = plt.subplots(
        n_fields, n_tasks,
        figsize=(4.6 * n_tasks, 2.0 * n_fields),
        squeeze=False,
        gridspec_kw={"hspace": 0.30, "wspace": 0.10},      # tight rows + columns
    )

    # Per-task pure-paper sets (cached)
    task_papers = {tname_lc: _pure_task_papers(papers, tname_lc) for tname_lc, _ in TOP_TASKS}

    drawn_l1_global: set = set()
    for j, (tname_lc, tname_disp) in enumerate(TOP_TASKS):
        n_pure = len(task_papers[tname_lc])
        for i, (fkey, fdisp) in enumerate(FIELDS):
            ax = axes[i][j]
            _plot_panel(
                ax, papers, task_papers[tname_lc], fkey,
                label_right=True,
            )
            # Track which L1 axes appeared (for legend) — re-run selection
            # to keep the figure code path consistent.
            min_count = 5 if fkey == "auto_metrics" else 3
            chosen, _ = _select_top_by_frequency(
                papers, task_papers[tname_lc], fkey, TOP_N, min_count=min_count
            )
            for term in chosen:
                drawn_l1_global.add(_l1_axis(term, fkey))

            if i == 0:
                ax.set_title(f"{tname_disp}  (n={n_pure})",
                             fontsize=10.5, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(fdisp, fontsize=9.5, fontweight="bold")

    # Legend (bottom). Show L1 colours + the sig/non-sig style key.
    legend_order = ["AM", "QO", "QI", "QT", "QE", "AUX"]
    handles, labels = [], []
    for k in legend_order:
        if k in drawn_l1_global:
            handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=L1_COLORS[k],
                                      markeredgecolor="white",
                                      markersize=10, linewidth=0))
            labels.append(L1_NAMES[k])
    # Significance key (LR values are printed next to each item, so we only
    # need to explain the line-style / marker-fill encoding here).
    # Only the colour legend lives in-figure; line-style / marker-fill /
    # bridge-curve / connector encodings are described in the caption.
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.4, columnspacing=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    papers = load_data()
    out_dir = Path(__file__).parent.parent / "outputs" / "figures" / "task_dashboard"
    make_dashboard(papers, out_dir / "task_dashboard_rich_v2.png")


if __name__ == "__main__":
    main()
