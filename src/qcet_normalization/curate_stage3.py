"""Apply human curation on top of the Stage-3 LLM verdicts.

Stage-3 is automated (LLM judges each cluster), but the LLM occasionally
splits a single construct across multiple clusters. We don't edit the raw
verdict file (it stays as a reproducibility artifact); instead, all human
overrides are declared HERE in `CURATION_RULES`, and this script derives
`stage3_aux_taxonomy_FINAL.json` from `stage3_aux_taxonomy.json` + the rules.

This keeps the audit trail clean:
  - `stage3_aux_taxonomy.json`        what the LLM said
  - this file                          what we changed and why
  - `stage3_aux_taxonomy_FINAL.json`  what Stage 4 actually uses

Rule kinds:
  MERGE    — combine N clusters into one aux with chosen id/name/definition.
  RENAME   — rename an aux_id (e.g. clarify naming).
  DROP     — discard an aux entry post-hoc.
  REDIRECT — change a KEEP_AUX into a FOLD_INTO_QCET. (Not implemented yet
             because it would need to also touch stage3_decisions.csv.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RAW_JSON   = HERE / "outputs" / "stage3_aux_taxonomy.json"
FINAL_JSON = HERE / "outputs" / "stage3_aux_taxonomy_FINAL.json"


# ----------------------------------------------------------------------------
# Curation rules — append new entries here as we curate.
# Each rule is a dict with a `kind` field. They are applied in declaration
# order; later rules see the result of earlier rules.
# ----------------------------------------------------------------------------

CURATION_RULES: list[dict[str, Any]] = [
    {
        "kind":   "MERGE",
        "rationale": (
            "Cluster 1 (abstract bias/fairness/objectivity, 11 variants/16 occ) "
            "and Cluster 2 (gender/stereotype/demographic bias, 22 variants/30 occ) "
            "are the same construct at two levels of specificity. The LLM split "
            "them because the embeddings cluster the demographic-specific "
            "instances separately, but the substitutability test is symmetric: "
            "any 'gender bias' criterion measures bias-and-fairness, and any "
            "'unfair bias' criterion subsumes the demographic case."
        ),
        "merge_aux_ids":  ["AUX-BiasFairness", "AUX-Fairness"],
        "into_aux_id":    "AUX-BiasFairness",
        "into_aux_name":  "Bias and Fairness",
        "into_aux_definition": (
            "The extent to which the output is free from unfair bias, "
            "stereotypes, or discriminatory content (including bias regarding "
            "sensitive attributes such as gender, race, or age), maintains "
            "objectivity, or adheres to a specified bias level."
        ),
    },
]


# ----------------------------------------------------------------------------
# Rule executor
# ----------------------------------------------------------------------------

def apply_merge(categories: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    src_ids = set(rule["merge_aux_ids"])
    src = [c for c in categories if c["aux_id"] in src_ids]
    others = [c for c in categories if c["aux_id"] not in src_ids]
    if len(src) != len(src_ids):
        found = {c["aux_id"] for c in src}
        missing = src_ids - found
        raise ValueError(f"MERGE rule references unknown aux_id(s): {missing}")
    merged = {
        "cluster_id":     [c["cluster_id"] for c in src],
        "aux_id":         rule["into_aux_id"],
        "aux_name":       rule["into_aux_name"],
        "aux_definition": rule["into_aux_definition"],
        "n_variants":     sum(c["n_variants"] for c in src),
        "occ_total":      sum(c["occ_total"]  for c in src),
        "top_raw":        [s for c in src for s in c.get("top_raw", [])][:16],
        "_curated_from":  [{"cluster_id": c["cluster_id"], "aux_id": c["aux_id"],
                            "aux_name": c["aux_name"]} for c in src],
        "_curation_rationale": rule["rationale"],
    }
    return others + [merged]


def apply_rename(categories: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for c in categories:
        if c["aux_id"] == rule["from_aux_id"]:
            c = {**c,
                 "aux_id":   rule["to_aux_id"],
                 "aux_name": rule.get("to_aux_name", c["aux_name"])}
        out.append(c)
    return out


def apply_drop(categories: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in categories if c["aux_id"] != rule["drop_aux_id"]]


def main() -> int:
    if not RAW_JSON.exists():
        raise FileNotFoundError(f"{RAW_JSON} not found. Run decide_stage3.py first.")
    raw = json.load(open(RAW_JSON))
    cats = list(raw["categories"])
    print(f"Loaded {len(cats)} aux categories from {RAW_JSON.name}.")

    curation_log: list[dict[str, Any]] = []
    for i, rule in enumerate(CURATION_RULES, 1):
        kind = rule["kind"]
        before = len(cats)
        if kind == "MERGE":
            cats = apply_merge(cats, rule)
        elif kind == "RENAME":
            cats = apply_rename(cats, rule)
        elif kind == "DROP":
            cats = apply_drop(cats, rule)
        else:
            raise ValueError(f"Unknown rule kind: {kind}")
        after = len(cats)
        print(f"  rule {i}: {kind}  ({before} -> {after} categories)")
        curation_log.append({"index": i, "kind": kind, "before_n": before,
                              "after_n": after,
                              "rationale": rule.get("rationale", "")})

    # Sort by occ_total desc for stable output order
    cats.sort(key=lambda c: -c["occ_total"])

    final = {
        "method":            "Stage-3 LLM verdicts + curate_stage3.py merges",
        "source_file":       RAW_JSON.name,
        "n_aux_pre_curation":  len(raw["categories"]),
        "n_aux_post_curation": len(cats),
        "curation_log":      curation_log,
        "categories":        cats,
    }
    FINAL_JSON.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"\nWrote: {FINAL_JSON}")
    print(f"Final aux taxonomy ({len(cats)} categories):")
    for c in cats:
        print(f"  - {c['aux_id']:30s}  variants={c['n_variants']:4d}  occ={c['occ_total']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
