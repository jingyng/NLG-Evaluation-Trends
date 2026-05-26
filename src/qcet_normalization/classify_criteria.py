"""Initial pure-QCET classification of raw criterion variants.

This is the *discovery* pass. The classifier is told about QCET ONLY (no aux
taxonomy, no priors). For each raw criterion string it must either:

  (a) pick the best QCET leaf with a confidence signal, or
  (b) return "none" and describe the construct in a short phrase.

The point is to let the residuals — everything that doesn't fit QCET — surface
naturally. Clustering then drives the auxiliary taxonomy induction.

Three important methodological choices:
  - No paper context at classification time. Classification depends only on the raw string
    and QCET. This maximizes reproducibility of the discovery and avoids
    injecting corpus-specific task information into the decision of which
    categories exist.
  - Per-variant, not per-occurrence. We have ~8.7K unique raw variants. Each
    gets exactly one classification. A later audit can resample per-occurrence
    for variants that look task-sensitive.
  - Batched classification by default (batch_size=5). Stage-0 calibration
    (calibrate_batch_size.py) showed 86.7% EXACT on 30 gold pairs at
    batch=5 vs 83.3% at batch=1, for ~5x lower per-item input cost and
    ~3.5x lower per-item output cost. The batched prompt explicitly
    instructs the model to classify each item independently; the parser
    aligns outputs by an integer index `n` to tolerate reordering.
    If a batched call fails parsing, the runner falls back to
    singleton classification for every item in that batch — no lost rows.

Outputs:
  outputs/criteria_classifications_initial.csv
    raw_string, source (llm/human/both), occurrences_llm, occurrences_human,
    qcet_id, qcet_name, qcet_fit, construct, justification, cache_hit, error

Usage:
  export OPENROUTER_API_KEY=sk-or-v1-...
  python classify_criteria.py --dry-run            # print prompts, no calls
  python classify_criteria.py --limit 50           # 50 most-frequent variants
  python classify_criteria.py                      # full run, batch_size=5
  python classify_criteria.py --batch-size 1       # singleton mode (fallback)
  python classify_criteria.py --resume             # continue from existing CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from deepseek_client import DeepSeekClient

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent  # repo root
QCET_JSON = HERE / "qcet_taxonomy.json"
LLM_CSV = BASE / "metadata_unique_counts" / "criteria" / "llm_criteria_stats.csv"
HUMAN_CSV = BASE / "metadata_unique_counts" / "criteria" / "human_criteria_stats.csv"
OUT_DIR = HERE / "outputs"
OUT_CSV = OUT_DIR / "criteria_classifications_initial.csv"


def load_qcet_leaves() -> list[dict[str, Any]]:
    with open(QCET_JSON) as f:
        data = json.load(f)
    leaves = [n for n in data["nodes"] if n["is_leaf"]]
    assert leaves, "No leaves found in qcet_taxonomy.json"
    return leaves


_FIT_LEVELS_LINES: list[str] = [
    "## Fit levels",
    "- \"strong\": the raw string is a near-synonym or paraphrase of the QCET "
    "leaf; a reviewer would agree without effort.",
    "- \"partial\": the leaf is the closest QCET fit but is broader, narrower, "
    "or shifts emphasis; a reviewer might prefer a different leaf or none at all.",
    "- \"none\": no QCET leaf fits; the construct is genuinely outside QCET's "
    "scope. Use this freely when warranted; do not force a fit.",
]

_RULES_LINES: list[str] = [
    "## Rules",
    "1. The raw string is the primary signal. Interpret it at face value. "
    "Do not over-infer.",
    "2. If the raw string is a metric name (BLEU, ROUGE-L, BERTScore, F1, etc.), "
    "a paper-section header, or not a quality criterion at all, return fit=\"none\" "
    "and say so in the construct field.",
    "3. If the raw string is ambiguous between two leaves, pick the best and mark "
    "fit=\"partial\". Do not return multiple candidates.",
    "4. \"construct\" must describe the measured property (not the raw string "
    "tokens). E.g. for \"engagingness\" write \"user engagement with output\"; "
    "for \"BLEU\" write \"automatic n-gram metric\"; for \"helpfulness\" write "
    "\"usefulness to user\".",
]


def _qcet_leaves_block(leaves: list[dict[str, Any]]) -> list[str]:
    """Lines for the '## QCET leaves' body, shared by singleton and batched."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for leaf in leaves:
        key = (leaf["level1"], leaf["level2"] or "")
        grouped[key].append(leaf)

    lines: list[str] = ["## QCET leaves (id | name | short definition)"]
    for (lvl1, lvl2), items in sorted(grouped.items()):
        if not items:
            continue
        lvl1_name = items[0]["level1_name"]
        lvl2_name = items[0]["level2_name"] or ""
        header = f"### {lvl1} {lvl1_name}"
        if lvl2_name:
            header += f"  /  {lvl2} {lvl2_name}"
        lines.append(header)
        for leaf in items:
            short = (leaf.get("short_definition") or "").strip().replace("\n", " ")
            if len(short) > 260:
                short = short[:257] + "..."
            lines.append(f"- {leaf['id']} | {leaf['name']} | {short}")
        lines.append("")
    return lines


def build_system_prompt(leaves: list[dict[str, Any]]) -> str:
    """Singleton system prompt: one raw criterion per call."""
    lines: list[str] = []
    lines.append(
        "You normalize evaluation-criterion strings from NLG research papers "
        "into the QCET taxonomy (Belz et al., 2025)."
    )
    lines.append("")
    lines.append("## Task")
    lines.append(
        "Given ONE raw criterion string, decide which QCET leaf it maps to. "
        "If no QCET leaf captures the construct, say so explicitly and describe "
        "the construct in your own words. Do not invent new categories."
    )
    lines.append("")
    lines.extend(_FIT_LEVELS_LINES)
    lines.append("")
    lines.append("## Output (STRICT JSON, no prose, no markdown)")
    lines.append(
        "{\n"
        "  \"qcet_id\": <QCET leaf id string like \"QOG-c-3\", or null if fit==\"none\">,\n"
        "  \"qcet_name\": <matching leaf name, or null if fit==\"none\">,\n"
        "  \"qcet_fit\": \"strong\" | \"partial\" | \"none\",\n"
        "  \"construct\": <2-6 word description of what the criterion measures>,\n"
        "  \"justification\": <one sentence, max 25 words, explaining the decision>\n"
        "}"
    )
    lines.append("")
    lines.extend(_qcet_leaves_block(leaves))
    lines.extend(_RULES_LINES)
    return "\n".join(lines)


def build_batched_system_prompt(leaves: list[dict[str, Any]]) -> str:
    """Batched system prompt: classify N raw criteria per call, INDEPENDENTLY.

    The expected output is a JSON object with key "classifications" holding
    an array of one object per input criterion, in the same order as the
    input. Each output object echoes the "raw" string so the caller can
    verify alignment.
    """
    lines: list[str] = []
    lines.append(
        "You normalize evaluation-criterion strings from NLG research papers "
        "into the QCET taxonomy (Belz et al., 2025)."
    )
    lines.append("")
    lines.append("## Task")
    lines.append(
        "You will receive a SHORT LIST of raw criterion strings, numbered 1..N. "
        "Classify each one INDEPENDENTLY of the others: reasoning about one "
        "item must not be affected by the identity of the others in the batch. "
        "For each item, decide which QCET leaf it maps to. If no QCET leaf "
        "captures the construct, say so explicitly and describe the construct "
        "in your own words. Do not invent new categories."
    )
    lines.append("")
    lines.extend(_FIT_LEVELS_LINES)
    lines.append("")
    lines.append("## Output (STRICT JSON, no prose, no markdown)")
    lines.append(
        "The ROOT of your response MUST be a JSON OBJECT (starts with `{`), "
        "NOT a bare JSON array (starts with `[`). The object has exactly one "
        "key, \"classifications\", whose value is an array. Schema:"
    )
    lines.append(
        "{\n"
        "  \"classifications\": [\n"
        "    {\n"
        "      \"n\": <1-based integer index matching input>,\n"
        "      \"raw\": <the raw criterion string, echoed verbatim>,\n"
        "      \"qcet_id\": <QCET leaf id like \"QOG-c-3\", or null if fit==\"none\">,\n"
        "      \"qcet_name\": <matching leaf name, or null if fit==\"none\">,\n"
        "      \"qcet_fit\": \"strong\" | \"partial\" | \"none\",\n"
        "      \"construct\": <2-6 word description of what the criterion measures>,\n"
        "      \"justification\": <one sentence, max 25 words>\n"
        "    },\n"
        "    ... one object per input item, in the same order ...\n"
        "  ]\n"
        "}"
    )
    lines.append(
        "The array length MUST equal the number of input items. Every \"n\" "
        "must appear exactly once. Do not merge, reorder, or drop items. "
        "Do NOT wrap the output in a Markdown code fence."
    )
    lines.append("")
    lines.extend(_qcet_leaves_block(leaves))
    lines.extend(_RULES_LINES)
    lines.append(
        "5. Treat each numbered item as a standalone classification. Do not "
        "assume items in the same batch are related, belong to the same paper, "
        "or share a common context."
    )
    return "\n".join(lines)


def build_user_prompt(raw: str) -> str:
    return f"Criterion: {raw}"


def build_batched_user_prompt(raws: list[str]) -> str:
    lines = ["Classify each of the following criteria:"]
    for i, r in enumerate(raws, 1):
        lines.append(f"{i}. {r}")
    return "\n".join(lines)


def parse_batched_response(
    parsed: Any, raws: list[str]
) -> list[dict[str, Any]]:
    """Pull a list of per-item dicts aligned to `raws`, or raise.

    Alignment strategy: trust `n` over `raw` over list position, in that
    order. If any item is missing we raise. If `raw` echoes disagree with
    the input we tolerate it but log a warning via the `_raw_mismatch` flag.

    Tolerated top-level shapes:
      1. Canonical:  {"classifications": [ ... ]}
      2. Bare array: [ ... ]  (R1 sometimes skips the wrapper)
      3. Alt key:    {"results": [...]} or {"items": [...]}
    """
    items: Any
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = (
            parsed.get("classifications")
            or parsed.get("results")
            or parsed.get("items")
        )
        if items is None:
            raise ValueError(
                "dict top-level but no 'classifications'/'results'/'items' key"
            )
    else:
        raise ValueError(f"top-level neither dict nor list: {type(parsed).__name__}")

    if not isinstance(items, list):
        raise ValueError(f"items payload not a list: {type(items).__name__}")
    if len(items) != len(raws):
        raise ValueError(
            f"batch size mismatch: got {len(items)} classifications for {len(raws)} inputs"
        )

    out: list[dict[str, Any] | None] = [None] * len(raws)
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"classification item not a dict: {type(item).__name__}")
        n = item.get("n")
        if not isinstance(n, int) or not (1 <= n <= len(raws)):
            raise ValueError(f"bad 'n' field: {n!r}")
        ok, why = validate_response(item)
        if not ok:
            raise ValueError(f"item n={n}: {why}")
        out[n - 1] = item

    missing = [i + 1 for i, x in enumerate(out) if x is None]
    if missing:
        raise ValueError(f"missing indices: {missing}")

    # Soft check on echoed raw (don't raise on mismatch; just attach)
    for i, (r_in, item) in enumerate(zip(raws, out)):
        echo = (item or {}).get("raw", "")
        item["_raw_mismatch"] = (echo != r_in)  # type: ignore[index]
    return [x for x in out if x is not None]


def load_criteria() -> list[dict[str, Any]]:
    llm_counts: dict[str, int] = {}
    if LLM_CSV.exists():
        with open(LLM_CSV) as f:
            for row in csv.DictReader(f):
                llm_counts[row["criterion"]] = int(row["count"])
    human_counts: dict[str, int] = {}
    if HUMAN_CSV.exists():
        with open(HUMAN_CSV) as f:
            for row in csv.DictReader(f):
                human_counts[row["criterion"]] = int(row["count"])

    all_raw = sorted(set(llm_counts) | set(human_counts))
    variants: list[dict[str, Any]] = []
    for raw in all_raw:
        l = llm_counts.get(raw, 0)
        h = human_counts.get(raw, 0)
        source = "both" if (l > 0 and h > 0) else ("llm" if l > 0 else "human")
        variants.append(
            {"raw_string": raw, "source": source, "occurrences_llm": l, "occurrences_human": h}
        )
    variants.sort(key=lambda d: -(d["occurrences_llm"] + d["occurrences_human"]))
    return variants


def validate_response(parsed: dict[str, Any]) -> tuple[bool, str]:
    required = ["qcet_id", "qcet_name", "qcet_fit", "construct", "justification"]
    for k in required:
        if k not in parsed:
            return False, f"missing field: {k}"
    if parsed["qcet_fit"] not in {"strong", "partial", "none"}:
        return False, f"bad qcet_fit: {parsed['qcet_fit']!r}"
    if parsed["qcet_fit"] == "none":
        if parsed["qcet_id"] not in (None, ""):
            return False, "qcet_fit=none but qcet_id is not null"
    else:
        if not parsed["qcet_id"]:
            return False, f"qcet_fit={parsed['qcet_fit']} but qcet_id is null"
    return True, ""


def _row_from_item(v: dict[str, Any], item: dict[str, Any], cache_hit: bool, error: str = "") -> dict[str, Any]:
    return {
        **v,
        "qcet_id": item.get("qcet_id") or "",
        "qcet_name": item.get("qcet_name") or "",
        "qcet_fit": item.get("qcet_fit", ""),
        "construct": item.get("construct", ""),
        "justification": item.get("justification", ""),
        "cache_hit": str(cache_hit),
        "error": error,
    }


def _row_from_error(v: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **v,
        "qcet_id": "", "qcet_name": "", "qcet_fit": "", "construct": "",
        "justification": "", "cache_hit": "", "error": error[:200],
    }


def _classify_singleton(
    client: DeepSeekClient, system: str, v: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Return (row, error_short). Used for batch-failure fallback too."""
    try:
        resp = client.chat(system=system, user=build_user_prompt(v["raw_string"]), json_mode=True)
        parsed = resp["parsed"]
        if parsed is None:
            raise ValueError(f"non-JSON: {resp['content'][:120]!r}")
        ok, why = validate_response(parsed)
        if not ok:
            raise ValueError(f"schema: {why}")
        return _row_from_item(v, parsed, resp["cache_hit"]), ""
    except Exception as e:
        return _row_from_error(v, f"singleton: {e}"), str(e)


def _classify_batch(
    client: DeepSeekClient, system_batched: str, system_singleton: str,
    batch: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Try batched call. On any failure, fall back to singleton per item."""
    raws = [v["raw_string"] for v in batch]
    try:
        resp = client.chat(
            system=system_batched,
            user=build_batched_user_prompt(raws),
            json_mode=True,
        )
        parsed = resp["parsed"]
        if parsed is None:
            raise ValueError(f"non-JSON: {resp['content'][:120]!r}")
        items = parse_batched_response(parsed, raws)
        rows = [_row_from_item(v, item, resp["cache_hit"]) for v, item in zip(batch, items)]
        return rows, ""
    except Exception as e:
        err = f"batch_fallback: {e}"
        rows: list[dict[str, Any]] = []
        for v in batch:
            row, _ = _classify_singleton(client, system_singleton, v)
            rows.append(row)
        return rows, err


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the top-N variants")
    ap.add_argument("--resume", action="store_true", help="skip rows already in output CSV")
    ap.add_argument("--dry-run", action="store_true", help="print the system prompt and exit")
    ap.add_argument("--model", default=None, help="override model name")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="items per API call (1 = singleton; default 5, validated in calibration)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent API workers (default 1). Each worker processes "
                         "one batch at a time. With batch_size=5 and workers=4, up to "
                         "20 items are in flight at once.")
    args = ap.parse_args(argv[1:])

    leaves = load_qcet_leaves()
    system_singleton = build_system_prompt(leaves)
    system_batched = build_batched_system_prompt(leaves)

    if args.dry_run:
        print(f"--- SINGLETON SYSTEM PROMPT ({len(system_singleton)} chars) ---")
        print(system_singleton[:1200] + "\n...(truncated)\n")
        print(f"--- BATCHED SYSTEM PROMPT ({len(system_batched)} chars) ---")
        print(system_batched[:1200] + "\n...(truncated)\n")
        print(f"leaves: {len(leaves)}  batch_size: {args.batch_size}")
        return 0

    variants = load_criteria()
    if args.limit:
        variants = variants[: args.limit]

    OUT_DIR.mkdir(exist_ok=True)
    done_keys: set[str] = set()
    skipped_errors = 0
    if args.resume and OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                # Only consider a raw_string "done" if its classification
                # succeeded. Rows with a non-empty `error` column are
                # retried on resume; this lets us recover from transient
                # failures (e.g. missing API key, rate limit) without
                # hand-editing the CSV. The retry will append a new row
                # for the same raw_string, which downstream consumers
                # should de-duplicate on (raw_string, error=="").
                if row.get("error"):
                    skipped_errors += 1
                    continue
                done_keys.add(row["raw_string"])
        msg = f"Resume: {len(done_keys)} already done."
        if skipped_errors:
            msg += f" {skipped_errors} prior errors will be retried."
        print(msg)

    # Drop already-done variants, preserve order
    pending = [v for v in variants if v["raw_string"] not in done_keys]
    bs = max(1, args.batch_size)
    workers = max(1, args.workers)
    n_batches = (len(pending) + bs - 1) // bs
    print(
        f"Scheduled {len(pending)} variants (total {len(variants)}, "
        f"done {len(done_keys)}), batch_size={bs}, workers={workers}, "
        f"batches={n_batches}. Out: {OUT_CSV}"
    )

    kwargs: dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model
    client = DeepSeekClient(**kwargs)

    header = [
        "raw_string", "source", "occurrences_llm", "occurrences_human",
        "qcet_id", "qcet_name", "qcet_fit", "construct", "justification",
        "cache_hit", "error",
    ]
    mode = "a" if (args.resume and OUT_CSV.exists()) else "w"

    # Build the job list up front so futures can be mapped back to their unit.
    # A "unit" is a batch (bs>1) or a singleton (bs==1) of variants.
    units: list[list[dict[str, Any]]]
    if bs == 1:
        units = [[v] for v in pending]
    else:
        units = [pending[i : i + bs] for i in range(0, len(pending), bs)]

    def process_unit(unit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        if len(unit) == 1:
            row, err = _classify_singleton(client, system_singleton, unit[0])
            return [row], ("singleton_err: " + err) if err else ""
        return _classify_batch(client, system_batched, system_singleton, unit)

    csv_lock = threading.Lock()
    errors = 0
    batch_fallbacks = 0
    processed = 0
    completed_units = 0

    with open(OUT_CSV, mode, newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=header)
        if mode == "w":
            writer.writeheader()
            fout.flush()

        # Sequential fast path when workers==1. Also the default, so we don't
        # pay thread-pool overhead for small/default runs.
        if workers == 1:
            for ui, unit in enumerate(units, 1):
                rows, unit_err = process_unit(unit)
                if unit_err and "batch_fallback" in unit_err:
                    batch_fallbacks += 1
                for row in rows:
                    if row.get("error"):
                        errors += 1
                    writer.writerow(row)
                processed += len(unit)
                completed_units += 1
                if completed_units % 10 == 0 or completed_units == len(units):
                    print(
                        f"  [{completed_units}/{len(units)}] units; "
                        f"{processed}/{len(pending)} items; "
                        f"errors={errors}; batch_fallbacks={batch_fallbacks}"
                    )
                    fout.flush()
        else:
            # Concurrent path: submit all units, stream rows back in completion
            # order. Row order in the CSV will not match input order, but
            # raw_string uniquely identifies each row, and --resume is robust
            # to any order because it filters on raw_string.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_unit, u): u for u in units}
                for fut in as_completed(futures):
                    try:
                        rows, unit_err = fut.result()
                    except Exception as e:
                        # A _classify_unit wrapper should have caught these,
                        # but defensively: fail the unit's items with error.
                        u = futures[fut]
                        rows = [_row_from_error(v, f"worker_crash: {e}") for v in u]
                        unit_err = f"worker_crash: {e}"

                    if unit_err and "batch_fallback" in unit_err:
                        batch_fallbacks += 1
                    with csv_lock:
                        for row in rows:
                            if row.get("error"):
                                errors += 1
                            writer.writerow(row)
                        processed += len(rows)
                        completed_units += 1
                        if completed_units % 10 == 0 or completed_units == len(units):
                            print(
                                f"  [{completed_units}/{len(units)}] units; "
                                f"{processed}/{len(pending)} items; "
                                f"errors={errors}; batch_fallbacks={batch_fallbacks}"
                            )
                            fout.flush()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
