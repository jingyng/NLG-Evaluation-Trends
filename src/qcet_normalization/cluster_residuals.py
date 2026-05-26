"""Cluster initial-classification residuals to surface aux-category candidates.

Input: outputs/criteria_classifications_initial.csv (produced by classify_criteria.py).
Only rows with qcet_fit in {"none", "partial"} are considered residuals; they
are clustered by the `construct` phrase the classifier returned.

Method:
  1. Select residual rows.
  2. Embed `construct` with sentence-transformers (default: all-MiniLM-L6-v2).
  3. Cluster with HDBSCAN (density-based; auto-selects k; flags noise).
  4. Report: cluster id, size (variants), total_occurrences, representative
     construct, top-10 raw strings, closest QCET leaf suggested by the
     classifier (if any), top partial-fit QCET leaves seen in the cluster.

This is the input to the human-judgment step where we apply the
prevalence + non-reducibility tests to decide which clusters become aux
categories.

Outputs:
  outputs/residual_cluster_assignments.csv   per-variant cluster id
  outputs/residual_cluster_report.csv        per-cluster summary
  outputs/residual_cluster_report.md         human-readable summary

Usage:
  pip install sentence-transformers hdbscan
  python cluster_residuals.py
  python cluster_residuals.py --min-cluster-size 15  # tune granularity
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
INITIAL_CLASSIFICATIONS_CSV = OUT_DIR / "criteria_classifications_initial.csv"
ASSIGN_CSV = OUT_DIR / "residual_cluster_assignments.csv"
REPORT_CSV = OUT_DIR / "residual_cluster_report.csv"
REPORT_MD = OUT_DIR / "residual_cluster_report.md"


def load_residuals() -> list[dict[str, Any]]:
    if not INITIAL_CLASSIFICATIONS_CSV.exists():
        print(f"ERROR: {INITIAL_CLASSIFICATIONS_CSV} not found. Run classify_criteria.py first.")
        sys.exit(2)
    rows: list[dict[str, Any]] = []
    with open(INITIAL_CLASSIFICATIONS_CSV) as f:
        for row in csv.DictReader(f):
            fit = row.get("qcet_fit", "")
            if fit in ("none", "partial") and row.get("construct", "").strip():
                rows.append({
                    "raw_string": row["raw_string"],
                    "construct": row["construct"].strip(),
                    "qcet_fit": fit,
                    "qcet_id": row.get("qcet_id", "") or "",
                    "qcet_name": row.get("qcet_name", "") or "",
                    "occ_llm": int(row.get("occurrences_llm", 0) or 0),
                    "occ_human": int(row.get("occurrences_human", 0) or 0),
                    "source": row.get("source", ""),
                    "justification": row.get("justification", ""),
                })
    if not rows:
        print("No residuals found; every variant mapped strong to QCET. residual clustering is unnecessary.")
        sys.exit(0)
    return rows


def embed(texts: list[str], model_name: str) -> "np.ndarray":
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("Need sentence-transformers + numpy. pip install sentence-transformers numpy")
        sys.exit(2)
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs)


def cluster(vecs, min_cluster_size: int, min_samples: int | None):
    try:
        import hdbscan
    except ImportError:
        print("Need hdbscan. pip install hdbscan")
        sys.exit(2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vecs)
    return labels, clusterer


def summarize_cluster(rows: list[dict[str, Any]]) -> dict[str, Any]:
    occ_total = sum(r["occ_llm"] + r["occ_human"] for r in rows)
    occ_llm = sum(r["occ_llm"] for r in rows)
    occ_human = sum(r["occ_human"] for r in rows)

    top_raw = sorted(rows, key=lambda r: -(r["occ_llm"] + r["occ_human"]))[:10]
    top_raw_list = [(r["raw_string"], r["occ_llm"] + r["occ_human"]) for r in top_raw]

    top_constructs = Counter(r["construct"] for r in rows).most_common(5)

    partial_qcet = Counter(
        (r["qcet_id"], r["qcet_name"]) for r in rows
        if r["qcet_fit"] == "partial" and r["qcet_id"]
    ).most_common(5)

    none_count = sum(1 for r in rows if r["qcet_fit"] == "none")
    partial_count = sum(1 for r in rows if r["qcet_fit"] == "partial")

    return {
        "n_variants": len(rows),
        "occ_total": occ_total,
        "occ_llm": occ_llm,
        "occ_human": occ_human,
        "top_raw": top_raw_list,
        "top_constructs": top_constructs,
        "partial_qcet_candidates": partial_qcet,
        "none_count": none_count,
        "partial_count": partial_count,
    }


def write_report(clusters: dict[int, dict[str, Any]], output_md: Path, output_csv: Path, p_thr: int, o_thr: int) -> None:
    rows = sorted(
        clusters.items(),
        key=lambda kv: (kv[0] == -1, -kv[1]["occ_total"]),
    )

    with open(output_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "cluster_id", "n_variants", "occ_total", "occ_llm", "occ_human",
            "none_count", "partial_count",
            "top_raw", "top_constructs", "partial_qcet_candidates",
            "passes_occ_threshold",
        ])
        for cid, s in rows:
            writer.writerow([
                cid, s["n_variants"], s["occ_total"], s["occ_llm"], s["occ_human"],
                s["none_count"], s["partial_count"],
                "; ".join(f"{r}({c})" for r, c in s["top_raw"]),
                "; ".join(f"{c}({n})" for c, n in s["top_constructs"]),
                "; ".join(f"{cid_}/{name}({n})" for (cid_, name), n in s["partial_qcet_candidates"]),
                str(s["occ_total"] >= o_thr),
            ])

    with open(output_md, "w") as fout:
        fout.write("# Residual cluster report\n\n")
        fout.write(
            "Each cluster is a candidate aux category. Apply the prevalence + "
            "non-reducibility tests manually to decide which survive.\n\n"
        )
        fout.write(f"Prevalence heuristic: occurrences >= {o_thr} "
                   f"(paper-level threshold P = {p_thr} must also be checked "
                   "after joining to paper-level counts).\n\n")
        for cid, s in rows:
            label = "NOISE (HDBSCAN -1)" if cid == -1 else f"Cluster {cid}"
            flag = " PASSES occ threshold" if s["occ_total"] >= o_thr else " below occ threshold"
            fout.write(f"## {label}  [variants={s['n_variants']}, occ={s['occ_total']} "
                       f"(llm={s['occ_llm']}, human={s['occ_human']}), {flag}]\n\n")
            fout.write(f"- none/partial split: {s['none_count']} none, {s['partial_count']} partial\n")
            fout.write("- Top representative constructs:\n")
            for c, n in s["top_constructs"]:
                fout.write(f"  - *{c}* ({n})\n")
            fout.write("- Top raw strings:\n")
            for r, c in s["top_raw"]:
                fout.write(f"  - `{r}` ({c})\n")
            if s["partial_qcet_candidates"]:
                fout.write("- Closest QCET leaves flagged by classifier as partial fit:\n")
                for (cid_, name), n in s["partial_qcet_candidates"]:
                    fout.write(f"  - {cid_} {name} ({n})\n")
            fout.write("\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--min-cluster-size", type=int, default=8)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--occ-threshold", type=int, default=50, help="Prevalence: total occurrences for a cluster to be flagged")
    ap.add_argument("--paper-threshold", type=int, default=20, help="Paper-prevalence threshold (informational; joined later)")
    args = ap.parse_args(argv[1:])

    OUT_DIR.mkdir(exist_ok=True)
    residuals = load_residuals()
    print(f"Residuals: {len(residuals)} variants.")

    texts = [r["construct"] for r in residuals]
    vecs = embed(texts, args.model)
    print(f"Embedded {len(vecs)} vectors, dim={vecs.shape[1]}")

    labels, _ = cluster(vecs, args.min_cluster_size, args.min_samples)
    import numpy as np
    unique_labels = sorted(set(labels))
    print(f"Clusters: {len(unique_labels)} (including noise {int((labels == -1).sum())} items)")

    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r, label in zip(residuals, labels):
        clusters[int(label)].append(r)

    with open(ASSIGN_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "raw_string", "construct", "qcet_fit", "qcet_id", "qcet_name",
            "occ_llm", "occ_human", "cluster_id",
        ])
        for r, label in zip(residuals, labels):
            writer.writerow([
                r["raw_string"], r["construct"], r["qcet_fit"], r["qcet_id"],
                r["qcet_name"], r["occ_llm"], r["occ_human"], int(label),
            ])

    summaries = {cid: summarize_cluster(rows) for cid, rows in clusters.items()}
    write_report(summaries, REPORT_MD, REPORT_CSV, args.paper_threshold, args.occ_threshold)
    print(f"Wrote: {ASSIGN_CSV}")
    print(f"Wrote: {REPORT_CSV}")
    print(f"Wrote: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
