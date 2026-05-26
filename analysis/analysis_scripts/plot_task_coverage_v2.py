"""
plot_task_coverage_v2.py

Figure 3 (`coverage_summary.png`): per-task small multiples showing how each of
the top four NLG tasks has shifted between automatic metrics, human evaluation,
and LaaJ over 2020-2025.

Layout: 2x2 grid, one panel per task (DG / MT / TS / QA). Each panel plots
percentage of that task's papers each year that use each method.

A paper is counted toward a task's panel if the task appears in its `tasks`
list, regardless of whether the paper covers other tasks too. (We deliberately
do NOT restrict to single-task papers, because multi-task papers are
increasingly common in the LLM era and a single-task filter would
systematically deflate the recent-year trends.) The same multi-task paper can
therefore contribute to several panels; this is the standard convention for
paper-level corpus studies and is flagged as a known limitation in the prose.

Encoding (B&W- and colour-blind-safe):
  - method = line style + marker:
        Auto metrics  : dotted + triangle
        Human eval    : dashed + square
        LaaJ          : solid  + circle
  - method = colour:
        Auto metrics  : grey
        Human eval    : blue
        LaaJ          : orange
  - panels share y-axis (0-100%) so the eye can compare paradigm shifts directly
    across tasks; x-axis is shared too.

Reads the corpus JSONs through `data_loader.load_data` and writes the PNG/PDF
into `my_paper/.../imgs/coverage_summary.png`/`.pdf`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from data_loader import load_data

OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "coverage_summary.png"
OUT_PDF = OUT_DIR / "coverage_summary.pdf"
VALIDATION_DIR = BASE / "data" / "laaj_human_validation_results_normalized"

TASKS_ORDERED = [
    "Dialogue Generation",
    "Machine Translation",
    "Text Summarization",
    "Question Answering",
]

METHOD_STYLES = {
    "Auto metrics":  {"color": "#888888", "ls": ":",  "marker": "^", "alpha": 0.85},
    "Human eval":    {"color": "#1f77b4", "ls": "--", "marker": "s", "alpha": 0.95},
    "LaaJ":          {"color": "#ff7f0e", "ls": "-",  "marker": "o", "alpha": 1.00},
    "LaaJ--Human validation": {
        "color": "#c92a2a", "ls": "-.", "marker": "D", "alpha": 1.00,
    },
}
TASK_SHARE_FILL = {"color": "#222222", "alpha": 0.10}

YEARS = list(range(2020, 2026))


def _load_validation_paper_ids() -> set[str]:
    """Return the set of paper_ids that explicitly validate LaaJ against
    human evaluation (`explicit_validation.answer == 'yes'`). Same
    convention as Figure 1 (`plot_evaluation_method_adoption.py`)."""
    ids: set[str] = set()
    if not VALIDATION_DIR.exists():
        return ids
    for root, _dirs, files in os.walk(VALIDATION_DIR):
        for fn in files:
            if not fn.endswith(".json") or "summary" in fn:
                continue
            try:
                with open(os.path.join(root, fn)) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            ans = (d.get("explicit_validation", {}).get("answer", "") or "").strip().lower()
            if ans == "yes":
                pid = d.get("paper_id") or ""
                if pid:
                    ids.add(pid)
    return ids


def main() -> None:
    papers = load_data()
    print(f"Loaded {len(papers)} papers")
    validation_ids = _load_validation_paper_ids()
    print(f"Loaded {len(validation_ids)} explicit-validation paper IDs")

    # Total NLG papers per year (denominator for "task share").
    nlg_per_year = {}
    for p in papers:
        y = p.get("year")
        if isinstance(y, int) and y in YEARS:
            nlg_per_year[y] = nlg_per_year.get(y, 0) + 1
    print(f"NLG papers per year: " + ", ".join(f"{y}={nlg_per_year.get(y,0)}" for y in YEARS))

    # For each (task, year), aggregate any paper whose `tasks` list contains
    # the task. Multi-task papers therefore count toward multiple panels.
    rows = []
    for task in TASKS_ORDERED:
        task_low = task.lower().strip()
        matched = [p for p in papers
                   if any((t.lower().strip() == task_low) for t in (p.get("tasks") or []))]
        for p in matched:
            year = p.get("year")
            if not isinstance(year, int) or year not in YEARS:
                continue
            pid = p.get("paper_id")
            rows.append({
                "task": task,
                "year": year,
                "paper_id": pid,
                "has_auto_metrics": bool(p.get("auto_metrics")),
                "has_llm_judge":    bool(p.get("laaj_criteria") or p.get("laaj_models")),
                "has_human_eval":   bool(p.get("human_criteria")),
                "has_validation":   bool(pid and pid in validation_ids),
            })
    full = pd.DataFrame(rows)
    print(f"Multi-task aware totals: " + ", ".join(
        f"{t}={int((full['task']==t).sum())}" for t in TASKS_ORDERED))

    yearly = (
        full.groupby(["task", "year"])
            .agg(total_papers=("paper_id", "nunique"),
                 papers_with_auto_metrics=("has_auto_metrics", "sum"),
                 papers_with_llm_judge=("has_llm_judge", "sum"),
                 papers_with_human_eval=("has_human_eval", "sum"),
                 papers_with_validation=("has_validation", "sum"))
            .reset_index()
    )

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6),
                             sharex=True, sharey=True)
    panels = [
        (axes[0, 0], "Dialogue Generation"),
        (axes[0, 1], "Machine Translation"),
        (axes[1, 0], "Text Summarization"),
        (axes[1, 1], "Question Answering"),
    ]

    method_to_col = {
        "Auto metrics":            "papers_with_auto_metrics",
        "Human eval":              "papers_with_human_eval",
        "LaaJ":                    "papers_with_llm_judge",
        "LaaJ--Human validation":  "papers_with_validation",
    }

    for ax, task in panels:
        sub = yearly[yearly["task"] == task].sort_values("year")
        for method, src_col in method_to_col.items():
            sty = METHOD_STYLES[method]
            pcts = []
            for y in YEARS:
                row = sub[sub["year"] == y]
                if row.empty:
                    pcts.append(0.0)
                    continue
                tot = int(row["total_papers"].iloc[0])
                v = int(row[src_col].iloc[0])
                pcts.append(100.0 * v / tot if tot else 0.0)
            ax.plot(YEARS, pcts,
                    color=sty["color"], linestyle=sty["ls"],
                    marker=sty["marker"], markersize=5,
                    linewidth=1.7, alpha=sty["alpha"],
                    markeredgewidth=0.0)
        # Task share: % of all NLG papers in that year that involve this task.
        # Distinct denominator from the method lines, so we draw it as a
        # translucent filled area (rather than a line) to visually separate it
        # from the method-adoption curves.
        share_pcts = []
        for y in YEARS:
            row = sub[sub["year"] == y]
            v = int(row["total_papers"].iloc[0]) if not row.empty else 0
            denom = nlg_per_year.get(y, 0)
            share_pcts.append(100.0 * v / denom if denom else 0.0)
        ax.fill_between(YEARS, 0, share_pcts,
                        color="#222222", alpha=0.10, linewidth=0, zorder=0)
        ax.plot(YEARS, share_pcts,
                color="#222222", linestyle="-", linewidth=0.8,
                alpha=0.45, zorder=1)
        ax.set_title(task, pad=4)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y)[-2:] for y in YEARS])
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]])
        ax.grid(axis="y", alpha=0.25, linewidth=0.6, linestyle="--")

    # Y-axis label on left column
    axes[0, 0].set_ylabel("% of papers")
    axes[1, 0].set_ylabel("% of papers")
    # X-axis label on bottom row only
    for ax in axes[1, :]:
        ax.set_xlabel("Year")

    legend_handles = [
        Line2D([0], [0],
               color=METHOD_STYLES[m]["color"],
               linestyle=METHOD_STYLES[m]["ls"],
               marker=METHOD_STYLES[m]["marker"],
               markersize=5, linewidth=1.7, label=m)
        for m in ("Auto metrics", "Human eval", "LaaJ", "LaaJ--Human validation")
    ]
    from matplotlib.patches import Patch
    legend_handles.append(
        Patch(facecolor="#222222", alpha=0.10, edgecolor="#222222",
              linewidth=0.8, label="Task share (% of NLG)")
    )
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=5, frameon=False, columnspacing=1.3,
               handletextpad=0.5, fontsize=9)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
