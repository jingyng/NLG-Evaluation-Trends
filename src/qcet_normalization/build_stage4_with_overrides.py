"""
build_stage4_with_overrides.py

Apply `polysemous_overrides.csv` on top of `outputs/stage4_classifications_simple.csv`
and write `outputs/stage4_classifications_simple_with_overrides.csv`.

The override-corrected frame is the canonical input for Stage 5 sampling
(`sample_stage5_validation.py`), so the validation sample reflects the
final routing reported in the paper rather than the pre-override classifier
output.

For each row whose lowercased raw_string matches an override key:

    chosen_id      <- override.new_qcet_id
    chosen_name    <- override.new_qcet_name
    chosen_type    <- 'aux' if new_qcet_id starts with 'AUX-' else 'qcet'
    chosen_source  <- 'polysemous_override(was:<old_qcet_id>)'
    override_applied <- 'Y'  (new tracking column)

All other columns are preserved unchanged. Rows not matched by any override
get override_applied=''.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "outputs" / "stage4_classifications_simple.csv"
OVR = HERE / "polysemous_overrides.csv"
DST = HERE / "outputs" / "stage4_classifications_simple_with_overrides.csv"


def main() -> None:
    overrides: dict[str, tuple[str, str]] = {}
    with OVR.open(newline="") as f:
        for r in csv.DictReader(f):
            overrides[r["raw_lowercase"].strip().lower()] = (
                r["new_qcet_id"].strip(),
                r["new_qcet_name"].strip(),
            )
    print(f"Loaded {len(overrides)} polysemous overrides from {OVR.name}")

    with SRC.open(newline="") as fin, DST.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        fields = list(reader.fieldnames or [])
        if "override_applied" not in fields:
            fields.append("override_applied")
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        n_total = n_changed = 0
        for row in reader:
            n_total += 1
            key = (row.get("raw_string") or "").strip().lower()
            if key in overrides:
                new_id, new_name = overrides[key]
                old_id = row["chosen_id"]
                row["chosen_id"] = new_id
                row["chosen_name"] = new_name
                row["chosen_type"] = "aux" if new_id.startswith("AUX") else "qcet"
                row["chosen_source"] = f"polysemous_override(was:{old_id})"
                row["override_applied"] = "Y"
                n_changed += 1
            else:
                row["override_applied"] = ""
            writer.writerow(row)

    print(f"Read {n_total} stage4 rows; rewrote {n_changed} via overrides.")
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
