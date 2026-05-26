"""Stage 0 (batched) — calibration of the batched QCET classifier.

Same 30-pair gold set as calibrate_stage0.py, but criteria are packed into
batches of BATCH_SIZE (default 5) items per API call. Tests whether batched
mode preserves accuracy; if EXACT% >= ACCEPT_THRESHOLD (default 80%),
batching is green-lit for Stage 1.

What this tests:
  - Accuracy: does reasoning-in-batch degrade classification vs singleton?
  - Alignment: does the model emit one output per input in the right order?
  - Cost: how do per-item input/output tokens scale with batch size?

Output:
  outputs/stage0_calibration_results_batched.csv

Usage:
  export OPENROUTER_API_KEY=sk-or-v1-...
  python calibrate_stage0_batched.py              # default BATCH_SIZE=5
  python calibrate_stage0_batched.py --batch-size 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from calibrate_stage0 import leaf_index, load_pairs, score
from classify_stage1 import (
    build_batched_system_prompt,
    build_batched_user_prompt,
    load_qcet_leaves,
    parse_batched_response,
)
from deepseek_client import DeepSeekClient

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_CSV_TMPL = "stage0_calibration_results_batched_bs{bs}.csv"

DEFAULT_BATCH_SIZE = 5
ACCEPT_THRESHOLD = 80.0  # %  (moderate gate chosen in the plan)
SINGLETON_BASELINE = 83.3  # % EXACT from the AtlasCloud-only re-score


def chunks(lst: list[Any], n: int) -> list[list[Any]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = ap.parse_args(argv[1:])

    bs = args.batch_size
    pairs = load_pairs()
    leaves_list = load_qcet_leaves()
    leaves = leaf_index(leaves_list)
    system = build_batched_system_prompt(leaves_list)
    client = DeepSeekClient()

    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / OUT_CSV_TMPL.format(bs=bs)

    batches = chunks(pairs, bs)
    print(
        f"Batched Stage-0: n_pairs={len(pairs)}, batch_size={bs}, "
        f"n_batches={len(batches)}, system_prompt_chars={len(system)}"
    )

    header = [
        "raw_string", "gold_target_id", "gold_target_name",
        "pred_qcet_id", "pred_qcet_name", "pred_fit", "pred_construct",
        "pred_justification", "verdict", "batch_idx", "within_batch_idx",
        "raw_echo_mismatch", "cache_hit", "notes",
    ]
    counts = {"EXACT": 0, "SIBLING": 0, "WRONG": 0}
    total_prompt_tk = 0
    total_completion_tk = 0
    total_latency = 0.0

    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=header)
        writer.writeheader()

        for bi, batch in enumerate(batches, 1):
            raws = [p["raw_string"] for p in batch]
            user = build_batched_user_prompt(raws)
            t0 = time.time()
            try:
                resp = client.chat(system=system, user=user, json_mode=True)
                dt = time.time() - t0
                total_latency += dt
                total_prompt_tk += resp["prompt_tokens"] or 0
                total_completion_tk += resp["completion_tokens"] or 0

                parsed = resp["parsed"]
                if parsed is None:
                    raise ValueError(f"non-JSON: {resp['content'][:200]!r}")
                items = parse_batched_response(parsed, raws)
            except Exception as e:
                # Whole-batch failure: mark every item WRONG with the error.
                print(f"  batch {bi}/{len(batches)} FAILED: {e}")
                for wi, p in enumerate(batch, 1):
                    counts["WRONG"] += 1
                    writer.writerow({
                        "raw_string": p["raw_string"],
                        "gold_target_id": p["gold_target_id"],
                        "gold_target_name": p.get("gold_target_name", ""),
                        "pred_qcet_id": "", "pred_qcet_name": "",
                        "pred_fit": "", "pred_construct": "",
                        "pred_justification": "",
                        "verdict": "WRONG",
                        "batch_idx": bi, "within_batch_idx": wi,
                        "raw_echo_mismatch": "",
                        "cache_hit": "",
                        "notes": f"BATCH_ERROR: {e}"[:200],
                    })
                fout.flush()
                continue

            # Score each item
            print(
                f"  batch {bi}/{len(batches)}  prov={resp.get('provider')}  "
                f"pt={resp['prompt_tokens']}  ct={resp['completion_tokens']}  "
                f"dt={dt:.1f}s"
            )
            for wi, (p, item) in enumerate(zip(batch, items), 1):
                gold = p["gold_target_id"]
                verdict = score(
                    gold, item.get("qcet_id"), item.get("qcet_fit", ""), leaves
                )
                counts[verdict] += 1
                mark = {"EXACT": "OK", "SIBLING": "~ ", "WRONG": "X "}[verdict]
                print(
                    f"    [{wi}] {mark} {p['raw_string']!r:<32} "
                    f"gold={gold!r:<12} pred={(item.get('qcet_id') or 'NONE')!r}"
                )
                writer.writerow({
                    "raw_string": p["raw_string"],
                    "gold_target_id": gold,
                    "gold_target_name": p.get("gold_target_name", ""),
                    "pred_qcet_id": item.get("qcet_id") or "",
                    "pred_qcet_name": item.get("qcet_name") or "",
                    "pred_fit": item.get("qcet_fit", ""),
                    "pred_construct": item.get("construct", ""),
                    "pred_justification": item.get("justification", ""),
                    "verdict": verdict,
                    "batch_idx": bi,
                    "within_batch_idx": wi,
                    "raw_echo_mismatch": str(bool(item.get("_raw_mismatch"))),
                    "cache_hit": str(resp["cache_hit"]),
                    "notes": p.get("notes", ""),
                })
            fout.flush()

    # Summary
    total = sum(counts.values())
    ex_pct = 100 * counts["EXACT"] / total if total else 0
    si_pct = 100 * counts["SIBLING"] / total if total else 0
    wr_pct = 100 * counts["WRONG"] / total if total else 0
    print("")
    print("=" * 70)
    print(f"Total items:       {total}")
    print(f"EXACT:             {counts['EXACT']} ({ex_pct:.1f}%)")
    print(f"SIBLING:           {counts['SIBLING']} ({si_pct:.1f}%)")
    print(f"WRONG:             {counts['WRONG']} ({wr_pct:.1f}%)")
    print(f"Singleton baseline (AtlasCloud-only): {SINGLETON_BASELINE}%")
    print(f"Drop vs singleton: {SINGLETON_BASELINE - ex_pct:+.1f} pp")
    print("-" * 70)
    print(f"Total prompt_tk:   {total_prompt_tk}")
    print(f"Total compl_tk:    {total_completion_tk}")
    if total:
        print(
            f"Per-item input:    {total_prompt_tk / total:.0f} tk "
            f"(singleton ~4760)"
        )
        print(
            f"Per-item output:   {total_completion_tk / total:.0f} tk "
            f"(singleton ~980)"
        )
    print(f"Total API time:    {total_latency:.1f}s")
    print("=" * 70)

    if ex_pct >= ACCEPT_THRESHOLD:
        print(f"GATE PASS: batched EXACT {ex_pct:.1f}% >= {ACCEPT_THRESHOLD:.0f}%.")
        print("Batching is green-lit for Stage 1.")
    else:
        print(f"GATE FAIL: batched EXACT {ex_pct:.1f}% < {ACCEPT_THRESHOLD:.0f}%.")
        print("Do not proceed to batched Stage 1; inspect WRONGs.")
    print(f"\nDetails: {out_csv}")
    return 0 if ex_pct >= ACCEPT_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
