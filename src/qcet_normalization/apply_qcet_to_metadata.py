#!/usr/bin/env python3
"""
Regenerate metadata_unique_counts/*_criteria_*.csv files using the QCET
classification results from stage4_classifications_simple.csv.

Produces (overwrites):
  metadata_unique_counts/
    llm_criteria_normalization_mapping.csv      (original,normalized,count,qcet_id)
    human_criteria_normalization_mapping.csv
    llm_criteria_normalization_merges.csv       (normalized,total_count,num_variants,variants_with_counts,qcet_id)
    human_criteria_normalization_merges.csv
    llm_criteria_stats_normalized.csv           (criterion,count,qcet_id)
    human_criteria_stats_normalized.csv
    criteria_qcet_short_labels.csv              (qcet_id,qcet_name,short_label)

The "normalized" column carries the QCET full name (canonical, resolved across
chosen_id ↔ chosen_name conflicts).  Rows whose chosen_id == 'AUX-Other' use
the sentinel "__DROP__" so the JSON-normalization step can drop them.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BASE = Path(__file__).parent.parent           # paper_code/
QCET_CSV = BASE / "05_criteria_normalization" / "outputs" / "stage4_classifications_simple.csv"
POLY_OVERRIDES_CSV = BASE / "05_criteria_normalization" / "polysemous_overrides.csv"
OUT_DIR = BASE / "metadata_unique_counts"

DROP_SENTINEL = "__DROP__"

# Hand-fixes for chosen_id ↔ chosen_name conflicts in stage4 (resolved by
# (a) preferring the dominant name, (b) fixing typos, (c) keeping a single
# name per QCET id).  See apply_qcet_to_metadata diagnostic output for
# the conflict table.
CANONICAL_NAME_OVERRIDES: Dict[str, str] = {
    "QEF-w-11": "Empathy / Emotional Appropriateness",   # fix 'Emppathy' typo
    "QOC-c-2":  "Absence of Toxic / Harmful Content",    # 2 stray "Informativeness" rows are mis-classified
    "QOG-c-2":  "Informativeness",                       # canonical Informativeness
    # QIC-c-1 and QIF-c-1 both produced "Absence of Omissions (relative to input)";
    # disambiguate so qcet_name stays unique (used as primary key in figures).
    "QIF-c-1":  "Absence of Omissions (relative to input, feature)",
}

# Short labels for figures.  We are deliberately QCET-faithful: no paraphrasing,
# no reordering of word order, no substituting community terms (e.g. we keep
# "Usefulness" rather than "Helpfulness"). The only changes from the QCET full
# name are:
#   (1) drop the "(outputs as a whole)" suffix on QXG-w / QXC-w / QXF-w leaves
#       (it is implicit at the w-aspect level);
#   (2) collapse "(content/meaning)" → "(content)" when the parenthetical is
#       still needed for sibling disambiguation;
#   (3) drop verbose tail clauses that are redundant given the L1 axis
#       (e.g. "from Input" / "to Input" / "relative to input" / "as Affected
#       by Outputs") since the [QI]/[QE] prefix already conveys frame-of-
#       reference.
# We add a QCET L1 prefix ([QO]/[QI]/[QT]/[QE]) when writing the CSV so figures
# display e.g. "[QO] Coherence" or "[QE] Factual Truth".
SHORT_LABELS: Dict[str, str] = {
    # ── QO: Quality of outputs in their own right ──────────────────────────
    "QOG-w-3": "Fluency",
    "QOG-w-2": "Readability",
    "QOG-w-1": "Nonredundancy",
    "QOG-w-4": "Humanlikeness",
    "QOG-w-4.1": "Native Speaker Likeness",
    "QOG-w-4.2": "Non-AI Likeness",
    "QOG-w-5": "Understandability",
    "QOG-w-5.1": "Clarity",
    "QOG-w-5.1.1": "Speed of Understanding",
    "QOG-c-1": "Nonredundancy (content)",
    "QOG-c-2": "Informativeness",
    "QOG-c-3": "Coherence",
    "QOG-c-3.1": "Wellorderedness",
    "QOG-c-3.2": "Cohesiveness",
    "QOG-c-4": "Internal Consistency of Outputs",
    "QOG-f-1": "Nonredundancy (form)",
    "QOG-f-2": "Speech Quality",

    "QOC-w-1": "Correctness of Outputs",
    "QOC-c-1": "Semantic Correctness",
    "QOC-c-2": "Absence of Toxic / Harmful Content",
    "QOC-f-1": "Grammaticality",
    "QOC-f-2": "Spelling Accuracy",
    "QOC-f-3": "Pronunciation Accuracy",

    "QOF-w-1": "Diversity",
    "QOF-w-2": "Poeticness",
    "QOF-w-3": "Complexity",
    "QOF-w-4": "Conversationality",
    "QOF-w-5": "Humorousness",
    "QOF-w-6": "Creativity",
    "QOF-c-1": "Diversity (content)",
    "QOF-c-2": "Poeticness (content)",
    "QOF-c-3": "Complexity (content)",
    "QOF-f-1": "Diversity (form)",
    "QOF-f-2": "Poeticness (form)",
    "QOF-f-3": "Complexity (form)",
    "QOF-f-4": "Formality",
    "QOF-f-5": "Output Length",

    # ── QI: Quality relative to Input ──────────────────────────────────────
    "QIG-c-1": "Answerability from Input",
    "QIG-c-3": "Relevance to Input",
    "QIG-f-1": "Appropriateness of System Response Type",
    "QIG-f-2": "Success of Style Transfer from Sample",
    "QIG-w-1": "Parse Accuracy",
    "QIG-w-2": "Output Answers Question",
    "QIG-w-3": "Quality as Explanation of Input",

    "QIC-c-1": "Absence of Omissions",
    "QIC-c-2": "Absence of Additions",
    "QIC-c-3": "Consistency with Input",
    "QIC-c-4": "Coverage of Topics",
    "QIC-f-1": "Conformance to Syntactic Structure",
    "QIC-f-2": "Inclusion of Keywords",
    "QIC-w-1": "Translation Accuracy",

    "QIF-c-1": "Absence of Omissions (feature)",
    "QIF-c-2": "Similarity to Input (content)",
    "QIF-c-3": "Specificity",
    "QIF-f-1": "Control over Complexity (form)",
    "QIF-f-2": "Control over Style",
    "QIF-f-2.1": "Control over Formality",
    "QIF-f-3": "Output Size Relative to Input",
    "QIF-f-4": "Similarity to Input (form)",
    "QIF-w-1": "Control over Complexity",
    "QIF-w-2": "Control over Sentiment",
    "QIF-w-3": "Bias Inversion",
    "QIF-w-4": "Similarity to Input",
    "QIF-w-5": "Control over Multiple Attributes",

    # ── QT: Quality relative to Target outputs ─────────────────────────────
    "QTC-c-1": "Meaning Accuracy",
    "QTC-f-1": "Form Accuracy",
    "QTC-w-1": "Classification Accuracy",
    "QTC-w-2": "Sequence Labelling Accuracy",
    "QTC-w-3": "Complete Target Output Matching",
    "QTC-w-3.1": "Complete Word Matching",
    "QTC-w-3.2": "Character Matching",
    "QTC-w-4": "Retrieval Accuracy",
    "QTC-w-5": "Sequence Alignment Accuracy",
    "QTC-w-6": "Parse Accuracy (with refs)",

    "QTG-c-1": "Similarity to Target Outputs (content)",
    "QTG-f-1": "Similarity to Target Outputs (form)",
    "QTG-w-1": "Similarity to Target Outputs",
    "QTG-w-3": "Cross-Dataset Generalisation",

    # ── QE: Quality relative to External frame ─────────────────────────────
    "QEC-c-1": "Factual Truth",
    "QEC-c-2": "Relative Factual Accuracy",
    "QEC-c-3": "Absence of Bias / Stereotypes",
    "QEC-f-1": "Adherence to Style Guide",
    "QEC-f-2": "Adherence to Syntactic Rules",
    "QEC-w-1": "Functional Correctness",
    "QEC-w-2": "Refusal Appropriateness",

    "QEG-c-1": "Naturalness (content)",
    "QEG-c-2": "Appropriateness (content)",
    "QEG-f-1": "Naturalness (form)",
    "QEG-f-2": "Appropriateness (form)",
    "QEG-w-1": "Naturalness",
    "QEG-w-2": "Appropriateness",
    "QEG-w-3": "Usefulness",
    "QEG-w-3.1": "Usefulness for Task",
    "QEG-w-4": "Goodness as Explanation of System Behaviour",
    "QEG-w-5": "System Usability",
    "QEG-w-5.1": "Ease of Communication",
    "QEG-w-5.2": "Task Completion Speed",
    "QEG-w-6": "User Satisfaction",
    "QEG-w-7": "Clarity of Referents",
    "QEG-w-8": "Performance of Embedding/Downstream System",
    "QEG-w-9": "Multi-task Performance",
    "QEG-w-10": "Win Rate",

    "QEF-c-1": "Similarity to Non-target Reference (content)",
    "QEF-f-1": "Similarity to Non-target Reference (form)",
    "QEF-w-1": "Similarity to Non-target Reference",
    "QEF-w-2": "Effect on User Behaviour",
    "QEF-w-3": "Effect on User Emotion",
    "QEF-w-4": "Detectability of Author Stance",
    "QEF-w-5": "Detectability of Author Trait",
    "QEF-w-6": "Effect on User Opinion",
    "QEF-w-7": "Effect on User Stance",
    "QEF-w-8": "Interaction Completion Speed",
    "QEF-w-9": "Likelihood per External Model",
    "QEF-w-10": "Engagingness/Interestingness",
    "QEF-w-11": "Empathy / Emotional Appropriateness",

    # ── Auxiliary (not on the QCET lattice) ────────────────────────────────
    "AUX-OverallQuality": "Overall Quality / Preference",
    "AUX-TaskSpecificPerformance": "Task-Specific Performance",
    "AUX-Safety": "Safety",
    "AUX-Toxicity": "Toxicity",
    "AUX-Bias": "Bias",
    "AUX-InstructionFollowing": "Instruction Following",
    "AUX-Empathy": "Empathy",
    "AUX-Creativity": "Creativity",
}


def _l1_prefix(qcet_id: str) -> str:
    """Return the QCET L1 frame-of-reference prefix in [...] form, or '' for AUX."""
    if not qcet_id or qcet_id.startswith("AUX-"):
        return ""
    # QCET ids start with QO, QI, QT, or QE
    return f"[{qcet_id[:2]}] "


def load_qcet_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with QCET_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_polysemous_overrides() -> Dict[str, Tuple[str, str]]:
    """Return {raw_lowercase: (new_qcet_id, new_qcet_name)}.

    The overrides file is produced by `extract_polysemous_overrides.py` from
    a hand-annotated review workbook. They take precedence over the QCET
    classifier's verdict, and apply to ALL case variants of the lowercased
    raw string (matching how normalize_merged_results.py looks up mappings).
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not POLY_OVERRIDES_CSV.exists():
        return out
    with POLY_OVERRIDES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("raw_lowercase") or "").strip().lower()
            new_id = (row.get("new_qcet_id") or "").strip()
            new_name = (row.get("new_qcet_name") or "").strip()
            if key and new_id and new_name:
                out[key] = (new_id, new_name)
    return out


def apply_polysemous_overrides(
    rows: List[Dict[str, str]],
    overrides: Dict[str, Tuple[str, str]],
) -> int:
    """Apply per-lowercase-key overrides in place.  Returns count of rows changed."""
    if not overrides:
        return 0
    n = 0
    for row in rows:
        key = row["raw_string"].strip().lower()
        if key in overrides:
            new_id, new_name = overrides[key]
            if row["chosen_id"] != new_id:
                row["chosen_id"]   = new_id
                row["chosen_name"] = new_name
                row["chosen_source"] = "polysemous_override"
                n += 1
    return n


def resolve_canonical_names(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Return {chosen_id: canonical_name}.  When stage4 has multiple chosen_name
    values for the same chosen_id, prefer the override, else the dominant name."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cid = row["chosen_id"]
        name = row["chosen_name"]
        n = int(row["occurrences_llm"] or 0) + int(row["occurrences_human"] or 0)
        counts[cid][name] += n
    canonical: Dict[str, str] = {}
    for cid, name_counts in counts.items():
        if cid in CANONICAL_NAME_OVERRIDES:
            canonical[cid] = CANONICAL_NAME_OVERRIDES[cid]
        else:
            canonical[cid] = max(name_counts.items(), key=lambda x: x[1])[0]
    return canonical


def build_mapping(rows: List[Dict[str, str]], canonical: Dict[str, str], occ_field: str) -> List[Tuple[str, str, int, str]]:
    """Return list of (original, normalized, count, qcet_id) sorted by count desc."""
    entries: List[Tuple[str, str, int, str]] = []
    for row in rows:
        n = int(row[occ_field] or 0)
        if n == 0:
            continue
        cid = row["chosen_id"]
        if cid == "AUX-Other":
            normalized = DROP_SENTINEL
        else:
            normalized = canonical[cid]
        entries.append((row["raw_string"], normalized, n, cid))
    entries.sort(key=lambda x: -x[2])
    return entries


def write_mapping(path: Path, entries: List[Tuple[str, str, int, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "normalized", "count", "qcet_id"])
        for orig, norm, n, cid in entries:
            w.writerow([orig, norm, n, cid])


def write_merges(path: Path, entries: List[Tuple[str, str, int, str]]) -> None:
    """Aggregate by normalized name, recording variants with their counts."""
    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    name_to_id: Dict[str, str] = {}
    for orig, norm, n, cid in entries:
        if norm == DROP_SENTINEL:
            continue
        grouped[norm].append((orig, n))
        name_to_id.setdefault(norm, cid)
    rows = []
    for norm, variants in grouped.items():
        variants.sort(key=lambda x: -x[1])
        total = sum(n for _, n in variants)
        rows.append((norm, total, len(variants), variants, name_to_id[norm]))
    rows.sort(key=lambda x: -x[1])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalized", "total_count", "num_variants", "variants_with_counts", "qcet_id"])
        for norm, total, k, variants, cid in rows:
            variants_str = "; ".join(f"{v} ({n})" for v, n in variants)
            w.writerow([norm, total, k, variants_str, cid])


def write_stats_normalized(path: Path, entries: List[Tuple[str, str, int, str]]) -> None:
    """Aggregate counts by normalized name (drops the sentinel entries)."""
    counts: Dict[str, int] = defaultdict(int)
    name_to_id: Dict[str, str] = {}
    for _orig, norm, n, cid in entries:
        if norm == DROP_SENTINEL:
            continue
        counts[norm] += n
        name_to_id.setdefault(norm, cid)
    rows = sorted(counts.items(), key=lambda x: -x[1])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "count", "qcet_id"])
        for name, n in rows:
            w.writerow([name, n, name_to_id[name]])


def write_short_labels(path: Path, canonical: Dict[str, str]) -> None:
    """Write four columns:
        qcet_id              e.g. 'QOG-c-3'
        qcet_l1              e.g. 'QO' (or '' for AUX)
        qcet_name            full QCET leaf name
        short_label          bare short label (e.g. 'Coherence')
        short_label_prefixed bracketed L1 prefix + bare label (e.g. '[QO] Coherence')
    Figures should consume `short_label_prefixed` by default; reserve
    `short_label` for tables/text where the lattice axis is mentioned in prose.
    """
    rows = []
    for cid, full in sorted(canonical.items()):
        if cid == "AUX-Other":
            continue
        bare = SHORT_LABELS.get(cid)
        if bare is None:
            bare = full          # fall back to full name
        prefix = _l1_prefix(cid)
        l1 = "" if cid.startswith("AUX-") else cid[:2]
        rows.append((cid, l1, full, bare, f"{prefix}{bare}"))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qcet_id", "qcet_l1", "qcet_name", "short_label", "short_label_prefixed"])
        for cid, l1, full, bare, prefixed in rows:
            w.writerow([cid, l1, full, bare, prefixed])


def main() -> None:
    rows = load_qcet_rows()
    overrides = load_polysemous_overrides()
    n_overridden = apply_polysemous_overrides(rows, overrides)
    canonical = resolve_canonical_names(rows)

    print(f"Loaded {len(rows)} stage4 classifications.")
    if overrides:
        print(f"Loaded {len(overrides)} polysemous overrides; "
              f"applied to {n_overridden} stage4 rows.")
    print(f"Canonical QCET names: {len(canonical)} (excluding AUX-Other).")

    for source, occ_field in [("llm", "occurrences_llm"), ("human", "occurrences_human")]:
        entries = build_mapping(rows, canonical, occ_field)
        n_dropped = sum(1 for e in entries if e[1] == DROP_SENTINEL)
        n_kept    = len(entries) - n_dropped
        print(f"  {source}: {len(entries)} mapping rows  ({n_kept} kept, {n_dropped} AUX-Other → __DROP__)")

        write_mapping(OUT_DIR / f"{source}_criteria_normalization_mapping.csv", entries)
        write_merges(OUT_DIR / f"{source}_criteria_normalization_merges.csv", entries)
        write_stats_normalized(OUT_DIR / f"{source}_criteria_stats_normalized.csv", entries)

    write_short_labels(OUT_DIR / "criteria_qcet_short_labels.csv", canonical)
    print(f"Wrote short-label table.")


if __name__ == "__main__":
    main()
