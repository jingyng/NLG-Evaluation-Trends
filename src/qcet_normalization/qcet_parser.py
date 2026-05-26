"""Parse the QCET taxonomy markdown into structured JSON.

Source: https://nlp-qcet.github.io/  (Belz et al., 2025)
Output: qcet_taxonomy.json (already committed in this directory; only re-run
this script if you want to re-parse the upstream markdown source from scratch).

Each leaf node carries:
  id, name, definition, level1, level2, level3, parent_id,
  short_definition (one line for the classifier prompt),
  full_prose (elicitation + notes, kept for reference).

Non-leaf category nodes (Q, QO, QOC, QOC-f, ...) are also emitted with
hierarchy metadata but are never used as classification targets.

Usage
-----
At runtime, classifier scripts (classify_stage1.py, classify_stage4_simple.py)
read `qcet_taxonomy.json` directly; this parser is only needed to regenerate
that JSON from a fresh markdown dump of the QCET site. To re-parse:

    python qcet_parser.py path/to/nlp-qcet.github.io-0.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# Default path is a placeholder pointing at a local-only QCET markdown dump;
# this script is run only to regenerate qcet_taxonomy.json from a fresh source.
DEFAULT_INPUT = HERE / "nlp-qcet.github.io-0.md"
OUTPUT = HERE / "qcet_taxonomy.json"


HEADER_RE = re.compile(r"^#####\s+(?P<id>\S+)\s*:\s*(?P<name>.+?)\s*$")


LEVEL1_LABELS = {
    "QO": "Quality of outputs in their own right",
    "QI": "Quality of outputs relative to input",
    "QT": "Quality of outputs relative to target outputs",
    "QE": "Quality of outputs relative to a specified external frame of reference",
}

LEVEL2_LABELS = {
    "C": "Correctness",
    "G": "Goodness",
    "F": "Feature",
}

LEVEL3_LABELS = {
    "f": "Form",
    "c": "Content",
    "w": "Outputs as a whole",
}


def classify_node(node_id: str) -> dict[str, Any]:
    """Return level metadata for a QCET node id.

    IDs observed:
        Q                     -> root
        QO, QI, QT, QE        -> level 1 (frame-of-reference)
        QOC, QOG, QOF         -> level 2 (Correctness/Goodness/Feature)
        QOC-f, QOC-c, QOC-w   -> level 3 (Form/Content/Whole)
        QOC-f-1, QOG-c-3.1, QOG-w-5.1.1 -> leaves
    """
    meta: dict[str, Any] = {
        "id": node_id,
        "depth": None,
        "is_leaf": False,
        "level1": None,
        "level1_name": None,
        "level2": None,
        "level2_name": None,
        "level3": None,
        "level3_name": None,
        "parent_id": None,
    }

    if node_id == "Q":
        meta["depth"] = 0
        return meta

    parts = node_id.split("-")
    head = parts[0]
    if not head.startswith("Q") or len(head) < 2:
        return meta

    # Level 1 = QO/QI/QT/QE (head length 2)
    if len(head) == 2:
        meta["depth"] = 1
        meta["level1"] = head
        meta["level1_name"] = LEVEL1_LABELS.get(head)
        meta["parent_id"] = "Q"
        if len(parts) == 1:
            return meta

    # Level 2 = QOC/QOG/QOF (head length 3). The third letter maps to C/G/F.
    if len(head) == 3:
        meta["depth"] = 2
        meta["level1"] = head[:2]
        meta["level1_name"] = LEVEL1_LABELS.get(head[:2])
        meta["level2"] = head[2]
        meta["level2_name"] = LEVEL2_LABELS.get(head[2])
        meta["parent_id"] = head[:2]
        if len(parts) == 1:
            return meta

    # Level 3 = QOC-f etc. (one segment after head, a single letter from {f,c,w})
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1] in LEVEL3_LABELS:
        meta["depth"] = 3
        meta["level1"] = head[:2]
        meta["level1_name"] = LEVEL1_LABELS.get(head[:2])
        meta["level2"] = head[2] if len(head) >= 3 else None
        meta["level2_name"] = LEVEL2_LABELS.get(meta["level2"])
        meta["level3"] = parts[1]
        meta["level3_name"] = LEVEL3_LABELS[parts[1]]
        meta["parent_id"] = head
        return meta

    # Leaves: >=3 segments, last segment starts with a digit (possibly x.y)
    if len(parts) >= 3 and re.match(r"^\d", parts[-1]):
        meta["is_leaf"] = True
        meta["depth"] = 4 + parts[-1].count(".")
        meta["level1"] = head[:2]
        meta["level1_name"] = LEVEL1_LABELS.get(head[:2])
        meta["level2"] = head[2] if len(head) >= 3 else None
        meta["level2_name"] = LEVEL2_LABELS.get(meta["level2"])
        if len(parts) >= 2 and parts[1] in LEVEL3_LABELS:
            meta["level3"] = parts[1]
            meta["level3_name"] = LEVEL3_LABELS[parts[1]]
        parent = node_id.rsplit(".", 1)[0] if "." in parts[-1] else "-".join(parts[:-1])
        meta["parent_id"] = parent
        return meta

    return meta


def parse_markdown(md_text: str) -> list[dict[str, Any]]:
    """Split the markdown into blocks by ##### headers and extract definitions."""

    lines = md_text.splitlines()
    blocks: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            if current is not None:
                blocks.append(current)
            current = (m.group("id"), m.group("name").strip(), [])
        else:
            if current is not None:
                current[2].append(line)
    if current is not None:
        blocks.append(current)

    nodes: list[dict[str, Any]] = []
    for node_id, name, body_lines in blocks:
        meta = classify_node(node_id)
        body_text = "\n".join(body_lines).strip()

        short_def, notes = extract_definition(body_text)

        meta["name"] = name
        meta["short_definition"] = short_def
        meta["notes"] = notes
        meta["raw_body"] = body_text
        nodes.append(meta)

    return nodes


DEF_LABEL_RE = re.compile(r"^\s*Definition:\s*$", re.IGNORECASE | re.MULTILINE)
NOTES_LABEL_RE = re.compile(
    r"^\s*Additional notes and information:\s*$", re.IGNORECASE | re.MULTILINE
)


def extract_definition(body: str) -> tuple[str, str]:
    """Extract the Definition line and the notes block.

    Some category nodes have no Definition; in that case we fall back to the
    first non-empty prose line that is not a "show details" marker.
    """

    match = DEF_LABEL_RE.search(body)
    if match:
        after = body[match.end() :]
        notes_match = NOTES_LABEL_RE.search(after)
        if notes_match:
            def_block = after[: notes_match.start()].strip()
            notes_block = after[notes_match.end() :].strip()
        else:
            def_block = after.strip()
            notes_block = ""
        first_para = re.split(r"\n\s*\n", def_block, maxsplit=1)[0].strip()
        short = " ".join(first_para.split())
        short = re.sub(r"\s+Elicitation:.*$", "", short, flags=re.IGNORECASE | re.DOTALL)
        return short.strip(), notes_block

    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.lower() == "show details":
            continue
        return stripped, ""
    return "", ""


def main(argv: list[str]) -> int:
    input_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_INPUT
    if not input_path.exists():
        print(f"Input markdown not found: {input_path}", file=sys.stderr)
        return 1

    md = input_path.read_text()
    nodes = parse_markdown(md)

    leaves = [n for n in nodes if n["is_leaf"]]
    categories = [n for n in nodes if not n["is_leaf"] and n["depth"] is not None]

    summary = {
        "source": str(input_path),
        "n_total_nodes": len(nodes),
        "n_leaves": len(leaves),
        "n_categories": len(categories),
        "level1_counts": {lv: sum(1 for n in leaves if n["level1"] == lv) for lv in LEVEL1_LABELS},
        "level2_counts": {lv: sum(1 for n in leaves if n["level2"] == lv) for lv in LEVEL2_LABELS},
        "level3_counts": {lv: sum(1 for n in leaves if n["level3"] == lv) for lv in LEVEL3_LABELS},
    }

    missing_def = [n["id"] for n in leaves if not n["short_definition"]]
    if missing_def:
        print(f"WARN: {len(missing_def)} leaves missing definition: {missing_def[:5]}...",
              file=sys.stderr)

    OUTPUT.write_text(json.dumps(
        {"summary": summary, "nodes": nodes},
        indent=2,
        ensure_ascii=False,
    ))

    print(f"Wrote {OUTPUT}")
    print(f"  total nodes: {len(nodes)}")
    print(f"  leaves:      {len(leaves)}")
    print(f"  categories:  {len(categories)}")
    print(f"  level1 distribution: {summary['level1_counts']}")
    print(f"  level2 distribution: {summary['level2_counts']}")
    print(f"  level3 distribution: {summary['level3_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
