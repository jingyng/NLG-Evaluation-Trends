"""Non-reducibility verdicts on the residual clusters.

For each cluster, the LLM is asked: given the cluster's central construct,
its representative raw strings, and the QCET leaves the initial classifier
already flagged as partial fits, render a verdict:

  KEEP_AUX        — the cluster's construct is not adequately captured by any
                    QCET leaf; it warrants a new auxiliary category.
  FOLD_INTO_QCET  — one QCET leaf does cover the construct; cluster members
                    should be re-classified to that leaf in the final reclassification.
  SPLIT           — the cluster mixes >=2 distinct constructs that need
                    different destinations.

Why one call per cluster (not per variant):
  - The decision is about CONCEPT reducibility, which is a cluster-level
    property. Per-variant verdicts at the aux-decision step would just duplicate the final
    classification.
  - 19 calls vs ~600 calls is the difference between $0.02 and $1.50.
  - The final reclassification still does per-variant judgement against {QCET + surviving aux}
    so any per-variant errors here can be corrected later.

The model also proposes a name + definition for KEEP_AUX clusters; these
populate aux_taxonomy_candidates.json which the final reclassification uses.

Output:
  outputs/aux_category_decisions.csv      one row per cluster with verdict
  outputs/aux_taxonomy_candidates.json  surviving aux categories for the final reclassification
  outputs/aux_category_decisions.md       human-readable summary

Usage:
  export OPENROUTER_API_KEY=...
  python decide_aux_categories.py
  python decide_aux_categories.py --dry-run    # print prompt for cluster 13 and exit
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from deepseek_client import DeepSeekClient

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
QCET_JSON = HERE / "qcet_taxonomy.json"
ASSIGN_CSV = OUT_DIR / "residual_cluster_assignments.csv"
INITIAL_CLASSIFICATIONS_CSV = OUT_DIR / "criteria_classifications_initial.csv"
DECISIONS_CSV = OUT_DIR / "aux_category_decisions.csv"
DECISIONS_MD = OUT_DIR / "aux_category_decisions.md"
AUX_OUT_JSON = OUT_DIR / "aux_taxonomy_candidates.json"



# Loading / aggregation

def load_qcet_leaves() -> list[dict[str, Any]]:
    tax = json.load(open(QCET_JSON))
    return [n for n in tax["nodes"] if n.get("is_leaf")]


def load_clusters() -> dict[int, list[dict[str, Any]]]:
    """Group Stage-2 cluster assignments by cluster_id, merging Stage-1 occurrences."""
    if not ASSIGN_CSV.exists():
        sys.exit(f"ERROR: {ASSIGN_CSV} not found. Run cluster_residuals.py first.")
    if not INITIAL_CLASSIFICATIONS_CSV.exists():
        sys.exit(f"ERROR: {INITIAL_CLASSIFICATIONS_CSV} not found. Run classify_criteria.py first.")

    # Pull Stage-1's full row (we need source + justification + ALL the qcet
    # info, which the smaller residual-clusters file dropped on writing).
    initial_by_raw: dict[str, dict[str, Any]] = {}
    with open(INITIAL_CLASSIFICATIONS_CSV) as f:
        for r in csv.DictReader(f):
            initial_by_raw[r["raw_string"]] = r

    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with open(ASSIGN_CSV) as f:
        for r in csv.DictReader(f):
            cid = int(r["cluster_id"])
            s1 = initial_by_raw.get(r["raw_string"], {})
            by_cluster[cid].append({
                "raw_string":    r["raw_string"],
                "construct":     r["construct"],
                "qcet_fit":      r["qcet_fit"],
                "qcet_id":       r["qcet_id"] or "",
                "qcet_name":     r["qcet_name"] or "",
                "occ_llm":       int(r["occ_llm"]),
                "occ_human":     int(r["occ_human"]),
                "source":        s1.get("source", ""),
                "justification": s1.get("justification", ""),
            })
    return by_cluster


def cluster_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    occ_total = sum(r["occ_llm"] + r["occ_human"] for r in rows)
    top_raw = sorted(rows, key=lambda r: -(r["occ_llm"] + r["occ_human"]))[:15]
    top_constructs = Counter(r["construct"] for r in rows).most_common(8)
    qcet_partial = Counter(
        (r["qcet_id"], r["qcet_name"]) for r in rows
        if r["qcet_fit"] == "partial" and r["qcet_id"]
    ).most_common(5)
    return {
        "n_variants":        len(rows),
        "occ_total":         occ_total,
        "occ_llm":           sum(r["occ_llm"]   for r in rows),
        "occ_human":         sum(r["occ_human"] for r in rows),
        "none_count":        sum(1 for r in rows if r["qcet_fit"] == "none"),
        "partial_count":     sum(1 for r in rows if r["qcet_fit"] == "partial"),
        "top_raw":           [(r["raw_string"], r["occ_llm"] + r["occ_human"])
                              for r in top_raw],
        "top_constructs":    top_constructs,
        "qcet_partial_hits": qcet_partial,
    }



# Prompt construction

def build_system_prompt(leaves: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(
        "You are evaluating whether a cluster of NLG-evaluation criteria should "
        "be a NEW auxiliary category or whether it should fold into the existing "
        "QCET taxonomy (Belz et al., 2025)."
    )
    lines.append("")
    lines.append("## QCET leaves available")
    lines.append("")
    for n in leaves:
        defn = (n.get("definition") or "").strip()
        if defn:
            lines.append(f"- `{n['id']}` **{n['name']}** — {defn[:160]}")
        else:
            lines.append(f"- `{n['id']}` **{n['name']}**")
    lines.append("")
    lines.append("## Decision criteria")
    lines.append(
        "1. **FOLD_INTO_QCET** — pick this if a SINGLE QCET leaf substitutes "
        "for the cluster's central construct without semantic loss. The criteria "
        "in the cluster could be replaced by the QCET leaf name in any paper "
        "without changing what is being measured. Provide `target_qcet_id`."
    )
    lines.append(
        "2. **KEEP_AUX** — pick this if the cluster's central construct is NOT "
        "captured by any single QCET leaf (i.e., the cluster is non-reducible). "
        "Propose `aux_id` (CamelCase, prefixed `AUX-`), `aux_name` (human title), "
        "and `aux_definition` (one sentence, explicit about scope and what it "
        "EXCLUDES). Be conservative: only propose a new aux if you can write a "
        "definition that NO QCET leaf already covers."
    )
    lines.append(
        "3. **SPLIT** — pick this if the cluster contains >=2 distinct "
        "constructs that need different destinations. Provide `subgroups` as "
        "an array, each with its own verdict (FOLD or KEEP) and parameters."
    )
    lines.append(
        "4. **DROP** — pick this for clusters dominated by data-quality issues "
        "(single-character extractions, fragments, parsing artifacts) that are "
        "not measurable criteria at all."
    )
    lines.append("")
    lines.append("## Output (STRICT JSON, no prose, no markdown)")
    lines.append(
        "Schema (the root MUST be a JSON object starting with `{`):\n"
        "{\n"
        "  \"verdict\": \"FOLD_INTO_QCET\" | \"KEEP_AUX\" | \"SPLIT\" | \"DROP\",\n"
        "  \"target_qcet_id\": <QCET id like \"QOC-w-1\", required iff verdict=FOLD_INTO_QCET>,\n"
        "  \"aux_id\":         <string like \"AUX-Safety\", required iff verdict=KEEP_AUX>,\n"
        "  \"aux_name\":       <string, required iff verdict=KEEP_AUX>,\n"
        "  \"aux_definition\": <one sentence, max 35 words, required iff verdict=KEEP_AUX>,\n"
        "  \"subgroups\": [   // required iff verdict=SPLIT, array length >=2\n"
        "    { \"description\": <which variants this subgroup covers>,\n"
        "      \"verdict\": \"FOLD_INTO_QCET\" | \"KEEP_AUX\",\n"
        "      \"target_qcet_id\": <if FOLD>, \"aux_id\": <if KEEP>,\n"
        "      \"aux_name\": <if KEEP>, \"aux_definition\": <if KEEP> } ],\n"
        "  \"justification\": <one sentence explaining why, max 35 words>,\n"
        "  \"confidence\": \"high\" | \"medium\" | \"low\"\n"
        "}"
    )
    lines.append(
        "Do NOT wrap the output in a Markdown code fence. Do NOT include "
        "`<think>` tags. Output the JSON object only."
    )
    return "\n".join(lines)


def build_user_prompt(cid: int, summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Cluster {cid}")
    lines.append(
        f"variants={summary['n_variants']}, "
        f"total_occurrences={summary['occ_total']} "
        f"(llm={summary['occ_llm']}, human={summary['occ_human']}), "
        f"none={summary['none_count']}, partial={summary['partial_count']}"
    )
    lines.append("")
    lines.append("Top representative constructs (Stage-1 LLM-generated, count):")
    for c, n in summary["top_constructs"]:
        lines.append(f"  - {c}  [{n}]")
    lines.append("")
    lines.append("Top raw criterion strings (raw, total_occurrences):")
    for raw, occ in summary["top_raw"]:
        lines.append(f"  - {raw!r}  [{occ}]")
    lines.append("")
    if summary["qcet_partial_hits"]:
        lines.append("QCET leaves the Stage-1 classifier flagged as partial fit "
                     "for variants in this cluster (consider these first when "
                     "choosing FOLD_INTO_QCET):")
        for (qid, qname), n in summary["qcet_partial_hits"]:
            lines.append(f"  - {qid}  {qname}  [{n} variants]")
    else:
        lines.append("(No QCET leaves were flagged as partial fit for any "
                     "variant in this cluster — strongly suggests KEEP_AUX or DROP.)")
    lines.append("")
    lines.append("Render your verdict now as JSON.")
    return "\n".join(lines)



# Verdict parsing + validation

VALID_VERDICTS = {"FOLD_INTO_QCET", "KEEP_AUX", "SPLIT", "DROP"}


def validate_verdict(parsed: dict[str, Any], leaf_ids: set[str]) -> tuple[bool, str]:
    """Returns (ok, error_message). Performs minimum schema sanity."""
    if not isinstance(parsed, dict):
        return False, f"top-level not a dict: {type(parsed).__name__}"
    v = parsed.get("verdict")
    if v not in VALID_VERDICTS:
        return False, f"invalid verdict: {v!r}"
    if v == "FOLD_INTO_QCET":
        tid = parsed.get("target_qcet_id")
        if not tid or tid not in leaf_ids:
            return False, f"FOLD_INTO_QCET requires valid target_qcet_id; got {tid!r}"
    elif v == "KEEP_AUX":
        aid = parsed.get("aux_id") or ""
        if not isinstance(aid, str) or not aid.startswith("AUX-"):
            return False, f"KEEP_AUX requires aux_id starting with 'AUX-'; got {aid!r}"
        for k in ("aux_name", "aux_definition"):
            if not parsed.get(k):
                return False, f"KEEP_AUX requires non-empty {k}"
    elif v == "SPLIT":
        sg = parsed.get("subgroups")
        if not isinstance(sg, list) or len(sg) < 2:
            return False, "SPLIT requires subgroups array of length >=2"
        for i, s in enumerate(sg):
            if not isinstance(s, dict) or s.get("verdict") not in {"FOLD_INTO_QCET", "KEEP_AUX"}:
                return False, f"SPLIT subgroup {i} verdict invalid: {s.get('verdict')!r}"
            if s["verdict"] == "FOLD_INTO_QCET":
                tid = s.get("target_qcet_id")
                if not tid or tid not in leaf_ids:
                    return False, f"SPLIT subgroup {i}: invalid target_qcet_id {tid!r}"
            else:
                aid = s.get("aux_id") or ""
                if not aid.startswith("AUX-"):
                    return False, f"SPLIT subgroup {i}: bad aux_id {aid!r}"
    return True, ""



# Main

def main(argv: list[str]) -> int:
    # Line-buffer stdout so progress prints surface immediately when piped to
    # `tee` or redirected to a file. Each LLM call can take 20-40s on R1 with
    # reasoning; without this the user sees nothing until the script finishes.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts for cluster 13 (largest by occ) and exit")
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip-clusters", default="",
                    help="comma-separated cluster ids to skip (e.g. '0' for garbage)")
    args = ap.parse_args(argv[1:])

    OUT_DIR.mkdir(exist_ok=True)
    leaves = load_qcet_leaves()
    leaf_ids = {n["id"] for n in leaves}
    by_cluster = load_clusters()

    # Drop noise (-1) from candidate set; we only judge the 19 induced clusters.
    candidate_cids = sorted([cid for cid in by_cluster.keys() if cid != -1])
    skip_set = {int(x) for x in args.skip_clusters.split(",") if x.strip()}
    candidate_cids = [c for c in candidate_cids if c not in skip_set]
    print(f"Clusters to judge: {candidate_cids} (skipping {sorted(skip_set)})")

    summaries = {cid: cluster_summary(by_cluster[cid]) for cid in candidate_cids}

    system = build_system_prompt(leaves)
    print(f"System prompt length: {len(system)} chars / ~{len(system)//4} tokens")

    if args.dry_run:
        # Pick the largest cluster as the demonstration target
        cid = max(candidate_cids, key=lambda c: summaries[c]["occ_total"])
        user = build_user_prompt(cid, summaries[cid])
        print("\n--- DRY-RUN SYSTEM PROMPT (head) ---")
        print(system[:1500] + "\n...(truncated)\n")
        print(f"\n--- DRY-RUN USER PROMPT (cluster {cid}) ---")
        print(user)
        return 0

    kwargs: dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model
    client = DeepSeekClient(**kwargs)

    decisions: list[dict[str, Any]] = []
    aux_categories: list[dict[str, Any]] = []

    t_overall = time.time()
    for idx, cid in enumerate(candidate_cids):
        s = summaries[cid]
        user = build_user_prompt(cid, s)
        print(f"\n[{idx+1}/{len(candidate_cids)}] cluster {cid}  "
              f"variants={s['n_variants']}, occ={s['occ_total']} -> calling LLM...")
        t0 = time.time()
        try:
            resp = client.chat(system=system, user=user, json_mode=True)
        except Exception as e:
            print(f"  ERROR after {time.time()-t0:.1f}s: {e}")
            decisions.append({
                "cluster_id": cid, "verdict": "ERROR",
                "error": str(e), "raw_response": "",
                "n_variants": s["n_variants"], "occ_total": s["occ_total"],
            })
            continue

        parsed = resp.get("parsed")
        ok, err = validate_verdict(parsed if isinstance(parsed, dict) else {},
                                    leaf_ids)
        if not ok:
            print(f"  SCHEMA ERROR: {err}")
            decisions.append({
                "cluster_id": cid, "verdict": "SCHEMA_ERROR",
                "error": err, "raw_response": (resp.get("content", "") or "")[:500],
                "n_variants": s["n_variants"], "occ_total": s["occ_total"],
            })
            continue

        verdict = parsed["verdict"]
        dt = time.time() - t0
        cache_str = " (cached)" if resp.get("cache_hit") else ""
        print(f"  [{dt:5.1f}s{cache_str}] verdict={verdict}  "
              f"confidence={parsed.get('confidence')}  "
              f"prompt_tk={resp.get('prompt_tokens')}  "
              f"completion_tk={resp.get('completion_tokens')}")
        if verdict == "FOLD_INTO_QCET":
            print(f"    -> {parsed['target_qcet_id']}: "
                  f"{parsed.get('justification','')[:80]}")
        elif verdict == "KEEP_AUX":
            print(f"    -> {parsed['aux_id']}: {parsed['aux_name']}")
            print(f"       def: {parsed['aux_definition'][:120]}")
            aux_categories.append({
                "cluster_id": cid,
                "aux_id": parsed["aux_id"],
                "aux_name": parsed["aux_name"],
                "aux_definition": parsed["aux_definition"],
                "n_variants": s["n_variants"],
                "occ_total": s["occ_total"],
                "top_raw": [r for r, _ in s["top_raw"][:8]],
            })
        elif verdict == "SPLIT":
            print(f"    -> {len(parsed['subgroups'])} subgroups:")
            for sg in parsed["subgroups"]:
                if sg["verdict"] == "FOLD_INTO_QCET":
                    print(f"       FOLD -> {sg['target_qcet_id']}: {sg['description'][:60]}")
                else:
                    print(f"       AUX -> {sg['aux_id']}: {sg['description'][:60]}")
                    aux_categories.append({
                        "cluster_id": f"{cid}.split",
                        "aux_id": sg["aux_id"],
                        "aux_name": sg["aux_name"],
                        "aux_definition": sg["aux_definition"],
                        "n_variants": -1,  # subgroup size unknown without re-parse
                        "occ_total": -1,
                        "top_raw": [],
                    })
        elif verdict == "DROP":
            print(f"    -> drop: {parsed.get('justification','')[:80]}")

        decisions.append({
            "cluster_id": cid,
            "verdict": verdict,
            "n_variants": s["n_variants"],
            "occ_total": s["occ_total"],
            "target_qcet_id": parsed.get("target_qcet_id", ""),
            "aux_id":         parsed.get("aux_id", ""),
            "aux_name":       parsed.get("aux_name", ""),
            "aux_definition": parsed.get("aux_definition", ""),
            "subgroups_json": json.dumps(parsed.get("subgroups", []),
                                         ensure_ascii=False) if parsed.get("subgroups") else "",
            "confidence":    parsed.get("confidence", ""),
            "justification": parsed.get("justification", ""),
            "error":         "",
            "raw_response":  "",
        })

    print(f"\nAll {len(candidate_cids)} clusters judged in "
          f"{time.time()-t_overall:.1f}s wall time.")

    # Write decisions CSV
    headers = [
        "cluster_id", "verdict", "n_variants", "occ_total",
        "target_qcet_id", "aux_id", "aux_name", "aux_definition",
        "subgroups_json", "confidence", "justification", "error", "raw_response",
    ]
    with open(DECISIONS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for d in decisions:
            w.writerow({h: d.get(h, "") for h in headers})
    print(f"\nWrote: {DECISIONS_CSV}")

    # Write surviving aux taxonomy JSON
    aux_payload = {
        "method": "Stage-3 non-reducibility test on Stage-2 HDBSCAN clusters",
        "n_clusters_judged": len(candidate_cids),
        "n_aux_kept": len(aux_categories),
        "categories": aux_categories,
    }
    with open(AUX_OUT_JSON, "w") as f:
        json.dump(aux_payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {AUX_OUT_JSON}")

    # Markdown summary
    with open(DECISIONS_MD, "w") as f:
        f.write("# Non-reducibility verdicts\n\n")
        f.write(f"Judged {len(candidate_cids)} clusters; "
                f"kept {sum(1 for d in decisions if d['verdict']=='KEEP_AUX')} aux, "
                f"folded {sum(1 for d in decisions if d['verdict']=='FOLD_INTO_QCET')}, "
                f"split {sum(1 for d in decisions if d['verdict']=='SPLIT')}, "
                f"dropped {sum(1 for d in decisions if d['verdict']=='DROP')}.\n\n")
        for d in decisions:
            f.write(f"## Cluster {d['cluster_id']} — {d['verdict']}  "
                    f"(variants={d['n_variants']}, occ={d['occ_total']}, "
                    f"confidence={d.get('confidence','')})\n\n")
            if d["verdict"] == "FOLD_INTO_QCET":
                f.write(f"- target QCET: `{d['target_qcet_id']}`\n")
            elif d["verdict"] == "KEEP_AUX":
                f.write(f"- aux: `{d['aux_id']}` — **{d['aux_name']}**\n")
                f.write(f"- definition: {d['aux_definition']}\n")
            elif d["verdict"] == "SPLIT" and d["subgroups_json"]:
                f.write("- subgroups:\n")
                for sg in json.loads(d["subgroups_json"]):
                    if sg["verdict"] == "FOLD_INTO_QCET":
                        f.write(f"  - FOLD `{sg['target_qcet_id']}` — {sg['description']}\n")
                    else:
                        f.write(f"  - AUX `{sg['aux_id']}` ({sg['aux_name']}) — {sg['description']}\n")
            if d.get("justification"):
                f.write(f"- justification: {d['justification']}\n")
            if d.get("error"):
                f.write(f"- ERROR: {d['error']}\n")
            f.write("\n")
    print(f"Wrote: {DECISIONS_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
