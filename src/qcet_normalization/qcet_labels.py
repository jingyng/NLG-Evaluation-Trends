"""Author-edited short labels for QCET leaves + auxiliary categories.

These labels are deliberately QCET-faithful: no paraphrasing, no community
synonyms.  The bare form drops the verbose tail clauses that the QCET L1
prefix already conveys (e.g. "relative to input"); the prefixed form
adds the QCET L1 axis tag at the front (e.g. "[QO] Coherence") so figure
legends can show the frame of reference inline.

`build_label_map()` joins these labels with the canonical (chosen_id ↔
chosen_name) pairs found in `outputs/criteria_classifications_final.csv`
and is the single source consulted by both `apply_qcet_to_metadata.py`
and the figure scripts under `analysis/`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple


# Hand-fixes for chosen_id ↔ chosen_name conflicts emitted by the final
# classifier (typos, sibling-ambiguity).  Applied when resolving canonical
# names per QCET id.
CANONICAL_NAME_OVERRIDES: Dict[str, str] = {
    "QEF-w-11": "Empathy / Emotional Appropriateness",   # fix 'Emppathy' typo
    "QOC-c-2":  "Absence of Toxic / Harmful Content",    # stray "Informativeness" mis-classifications
    "QOG-c-2":  "Informativeness",                       # canonical Informativeness
    # QIC-c-1 and QIF-c-1 both produced "Absence of Omissions (relative to input)";
    # disambiguate so qcet_name stays unique (used as primary key in figures).
    "QIF-c-1":  "Absence of Omissions (relative to input, feature)",
}


# Bare short labels per QCET id.  Edits from the QCET full name:
#   (1) drop the "(outputs as a whole)" suffix on QXG-w / QXC-w / QXF-w leaves
#       (implicit at the w-aspect level);
#   (2) collapse "(content/meaning)" → "(content)" when the parenthetical is
#       still needed for sibling disambiguation;
#   (3) drop tail clauses redundant given the L1 axis (e.g. "from Input" /
#       "to Input" / "relative to input" / "as Affected by Outputs") since
#       the [QI]/[QE] prefix already conveys frame-of-reference.
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


def l1_axis(qcet_id: str) -> str:
    """Return the QCET L1 axis tag ('QO'/'QI'/'QT'/'QE') or '' for AUX/empty."""
    if not qcet_id or qcet_id.startswith("AUX-"):
        return ""
    return qcet_id[:2]


def _l1_prefix(qcet_id: str) -> str:
    """Bracketed L1 prefix used in figure-friendly labels."""
    l1 = l1_axis(qcet_id)
    return f"[{l1}] " if l1 else ""


def build_label_map(
    classifications_csv: Path | str,
) -> Dict[str, Tuple[str, str, str, str]]:
    """Read the final classifications CSV and return
    `{qcet_name_lower: (qcet_name, l1_tag, bare_short, prefixed_short)}`.

    The CSV must have `chosen_id` and `chosen_name` columns (what
    `reclassify_criteria.py` + `apply_polysemous_overrides.py` produce).
    Names are canonicalised via `CANONICAL_NAME_OVERRIDES`.
    """
    path = Path(classifications_csv)
    out: Dict[str, Tuple[str, str, str, str]] = {}
    if not path.exists():
        return out

    seen_per_id: Dict[str, Dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("chosen_id") or "").strip()
            name = (row.get("chosen_name") or "").strip()
            if not cid or not name:
                continue
            seen_per_id.setdefault(cid, {})
            seen_per_id[cid][name] = seen_per_id[cid].get(name, 0) + 1

    for cid, name_counts in seen_per_id.items():
        if cid in CANONICAL_NAME_OVERRIDES:
            name = CANONICAL_NAME_OVERRIDES[cid]
        else:
            # dominant name wins
            name = max(name_counts.items(), key=lambda kv: kv[1])[0]
        bare = SHORT_LABELS.get(cid, name)
        prefixed = _l1_prefix(cid) + bare if l1_axis(cid) else bare
        out[name.lower()] = (name, l1_axis(cid), bare, prefixed)

    return out
