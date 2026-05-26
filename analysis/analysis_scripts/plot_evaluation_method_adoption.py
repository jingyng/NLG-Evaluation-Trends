"""
plot_evaluation_method_adoption.py

Figure 1 (page-1 teaser): per-year share of NLG papers using each evaluation
method paradigm — Automatic metrics, Human evaluation, LaaJ, and explicit
LaaJ↔Human validation — over 2020-2025.

A paper is counted as using a method if its corresponding extraction field is
non-empty:
  - Automatic:  answer_2.automatic_metrics
  - Human:      answer_4.criteria
  - LaaJ:       answer_3.criteria OR answer_3.models
  - Validation: explicit_validation.answer == 'yes'
                (in the curated `laaj_human_validation_results_normalized/`
                 dataset; that dataset itself is restricted to papers that use
                 BOTH LaaJ AND human evaluation).

Reads from `data/llm-merged-results-top30-tasks/` (the 3,334-paper top-30-tasks
subset that the rest of the paper analyzes). The validation dataset is curated
only over this subset, so using the full 8,665-paper corpus as denominator
would unfairly deflate the validation line.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "llm-merged-results-top30-tasks"
VALIDATION_DIR = HERE.parent / "data" / "laaj_human_validation_results_normalized"
OUT_DIR = HERE.parent / "figures"  # analysis/figures/ (was: paper imgs/)
OUT_PNG = OUT_DIR / "evaluation_method_adoption.png"
OUT_PDF = OUT_DIR / "evaluation_method_adoption.pdf"

YEARS = list(range(2020, 2026))


def get_year(d: dict, root: str) -> int | None:
    pid = d.get("paper_id", "") or ""
    parts = pid.split(".")
    if parts and parts[0].isdigit() and len(parts[0]) == 4:
        return int(parts[0])
    folder = os.path.basename(root)
    if "-" in folder:
        try:
            return int(folder.split("-")[-1])
        except ValueError:
            return None
    return None


def load_counts() -> dict[int, dict[str, int]]:
    counts = defaultdict(lambda: {"total": 0, "auto": 0, "human": 0, "laaj": 0, "validation": 0})
    for root, _dirs, files in os.walk(DATA_DIR):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(root, fn)) as fh:
                d = json.load(fh)
            y = get_year(d, root)
            if y not in YEARS:
                continue
            auto = bool(d.get("answer_2", {}).get("automatic_metrics"))
            human = bool(d.get("answer_4", {}).get("criteria"))
            laaj = bool(
                d.get("answer_3", {}).get("criteria")
                or d.get("answer_3", {}).get("models")
            )
            counts[y]["total"] += 1
            counts[y]["auto"] += int(auto)
            counts[y]["human"] += int(human)
            counts[y]["laaj"] += int(laaj)

    # Validation papers: from the curated LaaJ↔Human validation dataset.
    # Count only papers where `explicit_validation.answer == 'yes'` —
    # i.e., the paper explicitly validates LaaJ against human evaluation
    # (not just uses both methods).
    if VALIDATION_DIR.exists():
        for root, _dirs, files in os.walk(VALIDATION_DIR):
            for fn in files:
                if not fn.endswith(".json") or "summary" in fn:
                    continue
                with open(os.path.join(root, fn)) as fh:
                    d = json.load(fh)
                y = get_year(d, root)
                if y not in YEARS:
                    continue
                ans = (d.get("explicit_validation", {}).get("answer", "") or "").strip().lower()
                if ans == "yes":
                    counts[y]["validation"] += 1
    return counts


def main() -> None:
    counts = load_counts()
    totals = [counts[y]["total"] for y in YEARS]
    auto_pct = [counts[y]["auto"] / counts[y]["total"] * 100 for y in YEARS]
    human_pct = [counts[y]["human"] / counts[y]["total"] * 100 for y in YEARS]
    laaj_pct = [counts[y]["laaj"] / counts[y]["total"] * 100 for y in YEARS]
    valid_pct = [counts[y]["validation"] / counts[y]["total"] * 100 for y in YEARS]

    n_papers = sum(totals)
    print(f"Loaded {n_papers} papers across {YEARS[0]}-{YEARS[-1]}")
    print(f"{'year':6} {'n':>5}  {'auto%':>6} {'laaj%':>6} {'human%':>7} {'valid%':>7}")
    for i, y in enumerate(YEARS):
        print(f"{y:6} {totals[i]:>5}  {auto_pct[i]:>5.1f}% {laaj_pct[i]:>5.1f}% {human_pct[i]:>6.1f}% {valid_pct[i]:>6.2f}%")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(4.6, 2.8))

    AUTO_C  = "#888888"
    HUMAN_C = "#1f77b4"
    LAAJ_C  = "#ff7f0e"
    VALID_C = "#c92a2a"   # dark red — emphasizes the gap

    ax.plot(YEARS, auto_pct,  marker="o", color=AUTO_C,  lw=2.0, label="Automatic metrics")
    ax.plot(YEARS, human_pct, marker="s", color=HUMAN_C, lw=2.0, label="Human evaluation")
    ax.plot(YEARS, laaj_pct,  marker="^", color=LAAJ_C,  lw=2.4, label="LaaJ (LLM-as-a-Judge)")
    ax.plot(YEARS, valid_pct, marker="D", color=VALID_C, lw=2.0, label="LaaJ with Human validation")

    # Endpoint annotations
    for ys, color, vals, dy in [
        ("auto",  AUTO_C,  auto_pct,  3),
        ("human", HUMAN_C, human_pct, -10),
        ("laaj",  LAAJ_C,  laaj_pct,  3),
        ("valid", VALID_C, valid_pct, 0),
    ]:
        ax.annotate(f"{vals[-1]:.1f}%" if vals[-1] < 10 else f"{vals[-1]:.0f}%",
                    xy=(YEARS[-1], vals[-1]),
                    xytext=(6, dy), textcoords="offset points",
                    color=color, fontsize=9.5, fontweight="bold",
                    va="center")

    ax.set_xticks(YEARS)
    ax.set_xlim(YEARS[0] - 0.3, YEARS[-1] + 1.0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylabel("Share of papers using method")
    ax.set_xlabel("Year")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.02, 0.90), labelspacing=0.25, fontsize=9)

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
