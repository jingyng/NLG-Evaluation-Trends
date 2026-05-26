"""classify_stage4_simple.py — simplified single-pass Stage 4 classification.

Target set (119 entries):
  - 117 QCET leaves (Belz et al., 2025, extended from 111 to 117)
  - AUX-OverallQuality  — holistic overall-quality / preference judgements
  - AUX-Other          — catch-all for metrics, noise, out-of-scope items

Routing (deterministic where Stage 3 already gave a clear verdict):
  1. Stage-1 strong-fit → carry forward unchanged.
  2. Stage-3 FOLD_INTO_QCET → remap to target QCET leaf.
  3. Stage-3 SPLIT, Stage-1 fit present → keep Stage-1 per-variant assignment.
  4. Stage-3 DROP → AUX-Other.
  5. All other variants (KEEP_AUX clusters + Stage-2 noise + Stage-1 errors)
     → single LLM call against the 119-entry target set.

This replaces the previous two-pass design (classify_stage4.py + stage4b_reclassify.py)
and the string-match rescue layer (subdivide_aux_other.py, apply_overrides_stage4.py).

Usage:
  export OPENROUTER_API_KEY=sk-or-v1-...
  python classify_stage4_simple.py --dry-run    # prompts + routing summary, no LLM
  python classify_stage4_simple.py --plan-only  # deterministic rows + plan, no LLM
  python classify_stage4_simple.py              # full run
  python classify_stage4_simple.py --resume     # resume after partial run
  python classify_stage4_simple.py --workers 4  # N concurrent API workers
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from deepseek_client import DeepSeekClient

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
QCET_JSON  = HERE / "qcet_taxonomy.json"
STAGE1_CSV = OUT_DIR / "stage1_classifications.csv"
STAGE2_CSV = OUT_DIR / "stage2_cluster_assignments.csv"
STAGE3_CSV = OUT_DIR / "stage3_decisions.csv"
OUT_CSV    = OUT_DIR / "stage4_classifications_simple.csv"
PLAN_CSV   = OUT_DIR / "stage4_simple_routing_plan.csv"

# The two AUX entries that survive in the simplified taxonomy.
_AUX_ENTRIES: list[dict[str, str]] = [
    {
        "aux_id":   "AUX-OverallQuality",
        "aux_name": "Overall Quality / Preference",
        "aux_definition": (
            "Holistic overall judgment of output quality without decomposition "
            "into specific sub-criteria. Covers 'overall quality', 'overall "
            "rating', pairwise preference judgements, and 'which is better' "
            "comparisons. Use when the paper explicitly declines to decompose "
            "into specific aspects."
        ),
    },
    {
        "aux_id":   "AUX-Other",
        "aux_name": "Other / Unclassifiable",
        "aux_definition": (
            "Catch-all for strings that are (a) not criteria at all "
            "(extraction noise, stopwords, paper-section headers), "
            "(b) automatic-metric names (BLEU, ROUGE-L, BERTScore, F1), "
            "or (c) plausibly criteria but not mappable to any QCET leaf "
            "or AUX-OverallQuality."
        ),
    },
]

OUTPUT_HEADER = [
    "raw_string", "source", "occurrences_llm", "occurrences_human",
    "chosen_id", "chosen_name", "chosen_type", "chosen_source",
    "stage1_qcet_id", "stage1_qcet_fit", "stage2_cluster_id",
    "fit", "construct", "justification", "cache_hit", "error",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_qcet_leaves() -> list[dict[str, Any]]:
    data = json.load(open(QCET_JSON))
    leaves = [n for n in data["nodes"] if n["is_leaf"]]
    assert leaves, "No leaves in qcet_taxonomy.json"
    return leaves


def load_stage1() -> list[dict[str, str]]:
    if not STAGE1_CSV.exists():
        sys.exit(f"ERROR: {STAGE1_CSV} not found. Run classify_stage1.py first.")
    seen: dict[str, dict[str, str]] = {}
    with open(STAGE1_CSV) as f:
        for row in csv.DictReader(f):
            raw = row["raw_string"]
            if raw in seen and seen[raw].get("error"):
                if not row.get("error"):
                    seen[raw] = row
            elif raw not in seen:
                seen[raw] = row
    return list(seen.values())


def load_stage2_clusters() -> dict[str, int]:
    out: dict[str, int] = {}
    if not STAGE2_CSV.exists():
        return out
    with open(STAGE2_CSV) as f:
        for row in csv.DictReader(f):
            out[row["raw_string"]] = int(row["cluster_id"])
    return out


def load_stage3_decisions() -> dict[int, dict[str, Any]]:
    if not STAGE3_CSV.exists():
        sys.exit(f"ERROR: {STAGE3_CSV} not found.")
    out: dict[int, dict[str, Any]] = {}
    with open(STAGE3_CSV) as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cluster_id"])
            except ValueError:
                continue
            out[cid] = row
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_variant(
    v: dict[str, str],
    cluster_id: int | None,
    decisions: dict[int, dict[str, Any]],
) -> dict[str, str]:
    """Return routing decision. chosen_source=='stage4_llm' means LLM is needed."""

    s1_fit  = v.get("qcet_fit", "")
    s1_id   = v.get("qcet_id", "")
    s1_name = v.get("qcet_name", "")

    # 1. Stage-1 strong-fit: carry forward.
    if s1_fit == "strong" and s1_id:
        return {
            "chosen_id": s1_id, "chosen_name": s1_name,
            "chosen_type": "qcet", "chosen_source": "stage1_exact",
            "fit": "strong",
            "construct": v.get("construct", ""),
            "justification": v.get("justification", ""),
        }

    # Variants absent from Stage 2 that weren't strong-fits go to LLM.
    if cluster_id is None:
        return {"chosen_source": "stage4_llm"}

    d = decisions.get(cluster_id)

    # 2. DROP cluster → AUX-Other.
    if d and d["verdict"] == "DROP":
        return {
            "chosen_id": "AUX-Other", "chosen_name": "Other / Unclassifiable",
            "chosen_type": "aux", "chosen_source": "stage3_drop",
            "fit": "",
            "construct": v.get("construct", ""),
            "justification": d.get("justification", ""),
        }

    # 3. FOLD_INTO_QCET: send to LLM with a fold hint so it can override to
    #    AUX-OverallQuality when appropriate (e.g. "quality" → AUX-OQ not QOC-w-1).
    if d and d["verdict"] == "FOLD_INTO_QCET" and d.get("target_qcet_id"):
        return None  # falls through to LLM

    # 4. SPLIT: trust Stage-1 per-variant QCET id when present.
    if d and d["verdict"] == "SPLIT":
        if s1_id and s1_fit in ("strong", "partial"):
            return {
                "chosen_id": s1_id, "chosen_name": s1_name,
                "chosen_type": "qcet",
                "chosen_source": "stage3_split_keep_stage1",
                "fit": s1_fit,
                "construct": v.get("construct", ""),
                "justification": "Stage-3 SPLIT verdict; per-variant target from Stage-1.",
            }
        return {"chosen_source": "stage4_llm"}

    # 5. KEEP_AUX clusters and noise → LLM.
    return {"chosen_source": "stage4_llm"}


def fill_qcet_names(rows: list[dict[str, Any]], leaves: list[dict[str, Any]]) -> None:
    name_for = {n["id"]: n["name"] for n in leaves}
    for r in rows:
        if r.get("chosen_type") == "qcet" and r.get("chosen_id") and not r.get("chosen_name"):
            r["chosen_name"] = name_for.get(r["chosen_id"], "")


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_FIT_LEVELS_LINES: list[str] = [
    "## Fit levels",
    "- \"strong\": the raw string is a near-synonym or paraphrase of the chosen "
    "target; a reviewer would agree without effort.",
    "- \"partial\": the chosen target is the closest fit but is broader, "
    "narrower, or shifts emphasis; a reviewer might prefer something else.",
]

_RULES_LINES: list[str] = [
    "## Rules",
    "1. The raw string is the primary signal. Interpret it at face value. "
    "Do not over-infer.",
    "2. If the raw string is a metric name (BLEU, ROUGE-L, BERTScore, F1, etc.), "
    "a paper-section header, or not a quality criterion at all, choose "
    "\"AUX-Other\" with fit=\"partial\" and explain in construct.",
    "3. Prefer a specific QCET leaf over AUX-OverallQuality or AUX-Other when "
    "both fit. AUX-OverallQuality is only for holistic judgements that "
    "explicitly decline to commit to a specific aspect.",
    "4. Only choose AUX-Other if the criterion is genuinely outside the scope "
    "of all 117 QCET leaves and AUX-OverallQuality.",
    "5. \"construct\" must describe the measured property (not the raw string "
    "tokens). E.g. for \"engagingness\" write \"user engagement with output\"; "
    "for \"BLEU\" write \"automatic n-gram metric\".",
]


def _qcet_leaves_block(leaves: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for leaf in leaves:
        grouped[(leaf["level1"], leaf["level2"] or "")].append(leaf)
    lines: list[str] = ["## QCET leaves (id | name | short definition)"]
    for (lvl1, lvl2), items in sorted(grouped.items()):
        if not items:
            continue
        header = f"### {lvl1} {items[0]['level1_name']}"
        if items[0]["level2_name"]:
            header += f"  /  {lvl2} {items[0]['level2_name']}"
        lines.append(header)
        for leaf in items:
            short = (leaf.get("short_definition") or "").strip().replace("\n", " ")
            if len(short) > 260:
                short = short[:257] + "..."
            lines.append(f"- {leaf['id']} | {leaf['name']} | {short}")
        lines.append("")
    return lines


def _aux_block() -> list[str]:
    lines: list[str] = ["## Auxiliary categories (outside QCET; use only when no QCET leaf fits)"]
    for a in _AUX_ENTRIES:
        lines.append(f"- {a['aux_id']} | {a['aux_name']} | {a['aux_definition']}")
    lines.append("")
    return lines


def build_system_prompt(leaves: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "You normalize evaluation-criterion strings from NLG research papers "
        "into a consolidated taxonomy: 117 QCET leaves (Belz et al., 2025, "
        "extended) plus two auxiliary categories.",
        "",
        "## Task",
        "Given ONE raw criterion string, choose the SINGLE best target. "
        "Every input must be assigned a target.",
        "",
    ]
    lines.extend(_FIT_LEVELS_LINES)
    lines.append("")
    lines.append("## Output (STRICT JSON, no prose, no markdown)")
    lines.append(
        "{\n"
        "  \"chosen_id\":     <id like \"QOG-c-3\" OR \"AUX-OverallQuality\" OR \"AUX-Other\">,\n"
        "  \"chosen_name\":   <matching name>,\n"
        "  \"chosen_type\":   \"qcet\" | \"aux\",\n"
        "  \"fit\":           \"strong\" | \"partial\",\n"
        "  \"construct\":     <2-6 word description of what the criterion measures>,\n"
        "  \"justification\": <one sentence, max 25 words, explaining the choice>\n"
        "}"
    )
    lines.append("")
    lines.extend(_qcet_leaves_block(leaves))
    lines.extend(_aux_block())
    lines.extend(_RULES_LINES)
    return "\n".join(lines)


def build_batched_system_prompt(leaves: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "You normalize evaluation-criterion strings from NLG research papers "
        "into a consolidated taxonomy: 117 QCET leaves (Belz et al., 2025, "
        "extended) plus two auxiliary categories.",
        "",
        "## Task",
        "You will receive a SHORT LIST of raw criterion strings, numbered 1..N. "
        "Classify each one INDEPENDENTLY. Every input must be assigned a target.",
        "",
    ]
    lines.extend(_FIT_LEVELS_LINES)
    lines.append("")
    lines.append("## Output (STRICT JSON, no prose, no markdown)")
    lines.append(
        "The ROOT of your response MUST be a JSON OBJECT (starts with `{`). "
        "It has exactly one key, \"classifications\", whose value is an array:"
    )
    lines.append(
        "{\n"
        "  \"classifications\": [\n"
        "    {\n"
        "      \"n\": <1-based integer index matching input>,\n"
        "      \"raw\": <the raw criterion string, echoed verbatim>,\n"
        "      \"chosen_id\":     <id like \"QOG-c-3\" or \"AUX-OverallQuality\" or \"AUX-Other\">,\n"
        "      \"chosen_name\":   <matching name>,\n"
        "      \"chosen_type\":   \"qcet\" | \"aux\",\n"
        "      \"fit\":           \"strong\" | \"partial\",\n"
        "      \"construct\":     <2-6 word description>,\n"
        "      \"justification\": <one sentence, max 25 words>\n"
        "    },\n"
        "    ... one object per input item, in the same order ...\n"
        "  ]\n"
        "}"
    )
    lines.append(
        "The array length MUST equal the number of input items. Every \"n\" "
        "must appear exactly once. Do NOT wrap the output in a Markdown code fence."
    )
    lines.append("")
    lines.extend(_qcet_leaves_block(leaves))
    lines.extend(_aux_block())
    lines.extend(_RULES_LINES)
    lines.append(
        "6. Treat each numbered item as a standalone classification. Do not "
        "assume items in the same batch are related or share a common context."
    )
    return "\n".join(lines)


def build_user_prompt(v: dict[str, Any]) -> str:
    lines = [f"Criterion: {v['raw_string']}"]
    s1_id  = v.get("qcet_id", "")
    s1_fit = v.get("qcet_fit", "")
    if s1_id and s1_fit == "partial":
        lines.append(
            f"(Prior partial-fit hint, may be wrong: {s1_id} '{v.get('qcet_name','')}'. "
            f"Re-evaluate against the full consolidated set.)"
        )
    fold_hint = v.get("stage3_fold_hint", "")
    if fold_hint:
        lines.append(
            f"(Cluster-level hint from Stage 3: similar variants mapped to {fold_hint}. "
            f"Override if a better match exists, e.g. AUX-OverallQuality.)"
        )
    return "\n".join(lines)


def build_batched_user_prompt(variants: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Classify each of the following criteria:"]
    for i, v in enumerate(variants, 1):
        line = f"{i}. {v['raw_string']}"
        s1_id  = v.get("qcet_id", "")
        s1_fit = v.get("qcet_fit", "")
        if s1_id and s1_fit == "partial":
            line += f"   [hint: prior partial-fit was {s1_id}]"
        fold_hint = v.get("stage3_fold_hint", "")
        if fold_hint:
            line += f"   [stage3 cluster hint: {fold_hint}, override if needed]"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing / validation
# ---------------------------------------------------------------------------

def _build_valid_target_index(leaves: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    idx: dict[str, dict[str, str]] = {}
    for n in leaves:
        idx[n["id"]] = {"name": n["name"], "type": "qcet"}
    for a in _AUX_ENTRIES:
        idx[a["aux_id"]] = {"name": a["aux_name"], "type": "aux"}
    return idx


def validate_response(
    parsed: dict[str, Any],
    valid_targets: dict[str, dict[str, str]],
) -> tuple[bool, str]:
    required = ["chosen_id", "chosen_name", "chosen_type", "fit", "construct", "justification"]
    for k in required:
        if k not in parsed:
            return False, f"missing field: {k}"
    cid = parsed["chosen_id"]
    if cid not in valid_targets:
        return False, f"unknown chosen_id: {cid!r}"
    if parsed["chosen_type"] != valid_targets[cid]["type"]:
        return False, (f"chosen_type mismatch for {cid}: "
                       f"got {parsed['chosen_type']!r}, "
                       f"expected {valid_targets[cid]['type']!r}")
    if parsed["fit"] not in {"strong", "partial"}:
        return False, f"bad fit: {parsed['fit']!r}"
    return True, ""


def parse_batched_response(
    parsed: Any,
    raws: list[str],
    valid_targets: dict[str, dict[str, str]],
) -> list[tuple[dict[str, Any] | None, str | None]]:
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = (parsed.get("classifications")
                 or parsed.get("results")
                 or parsed.get("items"))
        if items is None:
            raise ValueError("dict top-level but no classifications/results/items key")
    else:
        raise ValueError(f"top-level neither dict nor list: {type(parsed).__name__}")
    if not isinstance(items, list):
        raise ValueError(f"items payload not a list: {type(items).__name__}")

    out: list[tuple[dict[str, Any] | None, str | None]] = [
        (None, "no response slot") for _ in raws
    ]
    seen_n: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        n = item.get("n")
        if not isinstance(n, int) or not (1 <= n <= len(raws)):
            continue
        if n in seen_n:
            continue
        seen_n.add(n)
        ok, why = validate_response(item, valid_targets)
        out[n - 1] = (item, None) if ok else (None, why)
    return out


# ---------------------------------------------------------------------------
# LLM unit processors
# ---------------------------------------------------------------------------

def _row_from_llm_item(v: dict[str, Any], item: dict[str, Any],
                       cache_hit: bool) -> dict[str, Any]:
    return {
        **v,
        "chosen_id":     item["chosen_id"],
        "chosen_name":   item["chosen_name"],
        "chosen_type":   item["chosen_type"],
        "chosen_source": "stage4_llm",
        "fit":           item["fit"],
        "construct":     item.get("construct", ""),
        "justification": item.get("justification", ""),
        "cache_hit":     str(cache_hit),
        "error":         "",
    }


def _row_from_error(v: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **v,
        "chosen_id": "", "chosen_name": "", "chosen_type": "",
        "chosen_source": "stage4_llm",
        "fit": "", "construct": "", "justification": "",
        "cache_hit": "", "error": error[:200],
    }


def _classify_singleton(
    client: DeepSeekClient,
    system: str,
    v: dict[str, Any],
    valid_targets: dict[str, dict[str, str]],
) -> dict[str, Any]:
    try:
        resp = client.chat(system=system, user=build_user_prompt(v), json_mode=True)
        parsed = resp["parsed"]
        if parsed is None:
            raise ValueError(f"non-JSON: {resp['content'][:120]!r}")
        ok, why = validate_response(parsed, valid_targets)
        if not ok:
            raise ValueError(f"schema: {why}")
        return _row_from_llm_item(v, parsed, resp["cache_hit"])
    except Exception as e:
        return _row_from_error(v, f"singleton: {e}")


def _classify_batch(
    client: DeepSeekClient,
    system_batched: str,
    system_singleton: str,
    batch: list[dict[str, Any]],
    valid_targets: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    raws = [v["raw_string"] for v in batch]
    try:
        resp = client.chat(
            system=system_batched,
            user=build_batched_user_prompt(batch),
            json_mode=True,
        )
        parsed = resp["parsed"]
        if parsed is None:
            raise ValueError(f"non-JSON: {resp['content'][:120]!r}")
        slots = parse_batched_response(parsed, raws, valid_targets)
    except Exception:
        rows = [_classify_singleton(client, system_singleton, v, valid_targets)
                for v in batch]
        return rows, len(batch)

    rows: list[dict[str, Any]] = []
    n_fallbacks = 0
    for i, (item, _why) in enumerate(slots):
        if item is not None:
            rows.append(_row_from_llm_item(batch[i], item, resp["cache_hit"]))
        else:
            rows.append(_classify_singleton(client, system_singleton,
                                             batch[i], valid_targets))
            n_fallbacks += 1
    return rows, n_fallbacks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N unique LLM representatives (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="skip variants already in output CSV")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts + routing summary; no LLM calls")
    ap.add_argument("--plan-only", action="store_true",
                    help="write deterministic rows + routing plan; no LLM calls")
    ap.add_argument("--provider", default=None,
                    help="LLM backend: 'novita' or 'openrouter' "
                         "(default: LLM_PROVIDER env var, else 'novita')")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv[1:])

    OUT_DIR.mkdir(exist_ok=True)

    leaves          = load_qcet_leaves()
    stage1          = load_stage1()
    stage2_clusters = load_stage2_clusters()
    stage3          = load_stage3_decisions()
    valid_targets   = _build_valid_target_index(leaves)

    print(f"Loaded {len(leaves)} QCET leaves, {len(_AUX_ENTRIES)} aux entries, "
          f"{len(stage1)} Stage-1 rows, {len(stage2_clusters)} Stage-2 assignments, "
          f"{len(stage3)} Stage-3 verdicts.")

    system_singleton = build_system_prompt(leaves)
    system_batched   = build_batched_system_prompt(leaves)
    print(f"Singleton prompt: {len(system_singleton)} chars / ~{len(system_singleton)//4} tokens")
    print(f"Batched   prompt: {len(system_batched)} chars / ~{len(system_batched)//4} tokens")

    deterministic_rows: list[dict[str, Any]] = []
    llm_pending: list[dict[str, Any]] = []
    src_counter: Counter[str] = Counter()

    for v in stage1:
        cluster_id = stage2_clusters.get(v["raw_string"])
        base = {
            "raw_string":        v["raw_string"],
            "source":            v.get("source", ""),
            "occurrences_llm":   v.get("occurrences_llm", "0"),
            "occurrences_human": v.get("occurrences_human", "0"),
            "stage1_qcet_id":    v.get("qcet_id", ""),
            "stage1_qcet_fit":   v.get("qcet_fit", ""),
            "stage2_cluster_id": "" if cluster_id is None else str(cluster_id),
            "qcet_id":           v.get("qcet_id", ""),
            "qcet_name":         v.get("qcet_name", ""),
            "qcet_fit":          v.get("qcet_fit", ""),
            "construct":         v.get("construct", ""),
            "justification":     v.get("justification", ""),
        }
        verdict = route_variant(v, cluster_id, stage3)
        if verdict is None:
            fold_target = stage3.get(cluster_id, {}).get("target_qcet_id", "") if cluster_id is not None else ""
            verdict = {"chosen_source": "stage4_llm", "stage3_fold_hint": fold_target}
        src_counter[verdict["chosen_source"]] += 1

        if verdict["chosen_source"] == "stage4_llm":
            base["stage3_fold_hint"] = verdict.get("stage3_fold_hint", "")
            llm_pending.append(base)
        else:
            row = {**base, **verdict}
            row.pop("qcet_id", None)
            row.pop("qcet_name", None)
            row.pop("qcet_fit", None)
            row["error"]     = ""
            row["cache_hit"] = ""
            deterministic_rows.append(row)

    fill_qcet_names(deterministic_rows, leaves)

    print("\n=== Routing summary ===")
    for src, n in src_counter.most_common():
        print(f"  {src:35s} {n:5d}")
    print(f"  {'TOTAL':35s} {sum(src_counter.values()):5d}")

    # Case-insensitive deduplication: group llm_pending by lowercased raw_string.
    # One LLM call per unique lowercase key; the result is broadcast to all
    # case variants in the group. Each case variant keeps its own row in the
    # output with its own occurrence counts.
    case_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in llm_pending:
        case_groups[v["raw_string"].strip().lower()].append(v)

    def _pick_rep(group: list[dict[str, Any]]) -> dict[str, Any]:
        """Representative = variant with highest total occurrences."""
        return max(group, key=lambda v: (
            int(v.get("occurrences_llm", 0) or 0) +
            int(v.get("occurrences_human", 0) or 0)
        ))

    all_reps: list[dict[str, Any]] = [_pick_rep(g) for g in case_groups.values()]
    n_llm_variants = len(llm_pending)
    n_llm_unique   = len(all_reps)
    print(f"\nCase deduplication: {n_llm_variants} LLM-routed variants → "
          f"{n_llm_unique} unique ({n_llm_variants - n_llm_unique} calls saved)")

    with open(PLAN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chosen_source", "n_variants"])
        for src, n in src_counter.most_common():
            w.writerow([src, n])
    print(f"Wrote: {PLAN_CSV}")

    if args.dry_run:
        print("\n--- SINGLETON SYSTEM PROMPT (head, 1500 chars) ---")
        print(system_singleton[:1500] + "\n...(truncated)\n")
        if all_reps:
            print(f"\n--- SAMPLE BATCHED USER PROMPT (first {min(5,len(all_reps))} representatives) ---")
            print(build_batched_user_prompt(all_reps[:5]))
        print(f"\nLLM calls needed: {n_llm_unique} unique representatives "
              f"(covers {n_llm_variants} total variants incl. case duplicates) "
              f"= {(n_llm_unique + args.batch_size - 1) // args.batch_size} batches "
              f"at batch_size={args.batch_size}")
        return 0

    done_keys: set[str] = set()
    skipped_errors = 0
    if args.resume and OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("error"):
                    skipped_errors += 1
                    continue
                done_keys.add(row["raw_string"])
        print(f"Resume: {len(done_keys)} already in output, "
              f"{skipped_errors} prior errors will be retried.")

    mode = "a" if (args.resume and OUT_CSV.exists()) else "w"
    with open(OUT_CSV, mode, newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_HEADER)
        if mode == "w":
            writer.writeheader()
            fout.flush()
        for row in deterministic_rows:
            if row["raw_string"] in done_keys:
                continue
            writer.writerow({k: row.get(k, "") for k in OUTPUT_HEADER})
        fout.flush()
    det_written = sum(1 for r in deterministic_rows if r["raw_string"] not in done_keys)
    print(f"Wrote {det_written} deterministic rows to {OUT_CSV}.")

    if args.plan_only:
        print(f"\n--plan-only: skipping LLM phase. "
              f"{n_llm_unique} unique representatives pending "
              f"(covering {n_llm_variants} total variants).")
        return 0

    # Filter representatives: a group is considered done if its representative
    # raw_string is already in done_keys (all group members were written together).
    pending_reps = [v for v in all_reps if v["raw_string"] not in done_keys]
    if args.limit:
        pending_reps = pending_reps[: args.limit]

    bs = max(1, args.batch_size)
    workers = max(1, args.workers)
    n_batches = (len(pending_reps) + bs - 1) // bs
    n_pending_variants = sum(
        len(case_groups[v["raw_string"].strip().lower()]) for v in pending_reps
    )
    print(f"\nLLM phase: {len(pending_reps)} unique representatives "
          f"(covering {n_pending_variants} total variants); "
          f"batch_size={bs}, workers={workers}, batches={n_batches}.")

    if not pending_reps:
        print("Nothing to classify with LLM. Done.")
        return 0

    if args.provider:
        client = DeepSeekClient.for_provider(
            args.provider, **({"model": args.model} if args.model else {}))
    else:
        client = DeepSeekClient(**({"model": args.model} if args.model else {}))

    units: list[list[dict[str, Any]]]
    if bs == 1:
        units = [[v] for v in pending_reps]
    else:
        units = [pending_reps[i : i + bs] for i in range(0, len(pending_reps), bs)]

    csv_lock = threading.Lock()
    errors = 0
    item_fallbacks = 0
    hard_fallbacks = 0
    processed = 0
    completed = 0
    t0 = time.time()

    def process_unit(unit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        if len(unit) == 1:
            return [_classify_singleton(client, system_singleton, unit[0], valid_targets)], 0
        return _classify_batch(client, system_batched, system_singleton, unit, valid_targets)

    _CLS_FIELDS = ("chosen_id", "chosen_name", "chosen_type", "chosen_source",
                   "fit", "construct", "justification", "cache_hit", "error")

    with open(OUT_CSV, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_HEADER)

        def write_group_rows(rows: list[dict[str, Any]]) -> None:
            """For each representative result, write one output row per case
            variant in its group, keeping each variant's own occurrence counts
            and pipeline metadata but sharing the LLM classification fields."""
            expanded: list[dict[str, Any]] = []
            for r in rows:
                lower_key = r["raw_string"].strip().lower()
                cls = {k: r.get(k, "") for k in _CLS_FIELDS}
                for v in case_groups[lower_key]:
                    if v["raw_string"] in done_keys:
                        continue
                    expanded.append({
                        "raw_string":        v["raw_string"],
                        "source":            v.get("source", ""),
                        "occurrences_llm":   v.get("occurrences_llm", "0"),
                        "occurrences_human": v.get("occurrences_human", "0"),
                        "stage1_qcet_id":    v.get("stage1_qcet_id", ""),
                        "stage1_qcet_fit":   v.get("stage1_qcet_fit", ""),
                        "stage2_cluster_id": v.get("stage2_cluster_id", ""),
                        **cls,
                    })
            with csv_lock:
                for mr in expanded:
                    writer.writerow({k: mr.get(k, "") for k in OUTPUT_HEADER})
                fout.flush()

        def progress_line() -> str:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(pending_reps) - processed) / rate if rate > 0 else 0
            return (f"[{completed}/{n_batches}] batches; "
                    f"{processed}/{len(pending_reps)} reps; "
                    f"errors={errors}; fallbacks={item_fallbacks}; "
                    f"hard_fallbacks={hard_fallbacks}; "
                    f"{rate:.1f} it/s; ETA={eta/60:.1f}m")

        if workers == 1:
            for unit in units:
                rows, n_fb = process_unit(unit)
                if n_fb >= len(unit) and len(unit) > 1:
                    hard_fallbacks += 1
                else:
                    item_fallbacks += n_fb
                for row in rows:
                    if row.get("error"):
                        errors += 1
                write_group_rows(rows)
                processed += len(unit)
                completed += 1
                if completed % 10 == 0 or completed == n_batches:
                    print(progress_line())
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_unit, unit): unit for unit in units}
                for fut in as_completed(futures):
                    rows, n_fb = fut.result()
                    unit = futures[fut]
                    if n_fb >= len(unit) and len(unit) > 1:
                        hard_fallbacks += 1
                    else:
                        item_fallbacks += n_fb
                    for row in rows:
                        if row.get("error"):
                            errors += 1
                    write_group_rows(rows)
                    processed += len(unit)
                    completed += 1
                    if completed % 10 == 0 or completed == n_batches:
                        print(progress_line())

    print(f"\nDone. {processed} representatives classified by LLM in "
          f"{(time.time()-t0)/60:.1f}m "
          f"(covering {n_pending_variants} total variants). "
          f"errors={errors}, fallbacks={item_fallbacks}, "
          f"hard_fallbacks={hard_fallbacks}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
