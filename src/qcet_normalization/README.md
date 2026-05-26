# QCET-based criterion classification

Re-maps the normalised criterion strings produced by Step 3 of the main
pipeline onto the QCET taxonomy (Belz et al., 2024) plus a small auxiliary
category set induced from residuals.  The output is a per-variant
`(chosen_id, chosen_name)` assignment for every unique raw criterion in
the corpus, which is then folded back into the per-paper criterion CSVs
under `metadata_unique_counts/criteria/`.

All commands below assume `cwd` is the repository root.

## Pipeline

### 1. Calibrate batch size

```bash
python src/qcet_normalization/calibrate_batch_size.py
```

Calibrates the batched classifier against 30 gold pairs before the full
run; batching is green-lit only if EXACT% stays at or above 80%.

### 2. Initial classification + residual clustering

```bash
python src/qcet_normalization/classify_criteria.py     # 111 QCET leaves
python src/qcet_normalization/cluster_residuals.py     # HDBSCAN on partial / no-fit rows
```

`classify_criteria.py` is task-blind: each variant is classified against
the 111 QCET leaves with a fit level of `strong`, `partial`, or `none`.
Residuals (partial / none) feed HDBSCAN clustering, which surfaces
candidate auxiliary categories.

### 3. Auxiliary-category decisions

```bash
python src/qcet_normalization/decide_aux_categories.py    # LLM verdicts per cluster
python src/qcet_normalization/curate_aux_categories.py    # manual merges + keeps
```

Each residual cluster gets a verdict — KEEP_AUX, FOLD_INTO_QCET, SPLIT,
or DROP. The decisions are then manually curated (merging near-duplicates,
dropping data-quality clusters).

### 4. Final classification

```bash
python src/qcet_normalization/reclassify_criteria.py            # 117 QCET leaves + 2 aux categories
python src/qcet_normalization/apply_polysemous_overrides.py     # hand-edited polysemous overrides
```

The final classifier runs against the consolidated taxonomy: 111 QCET
leaves plus 6 surviving aux extensions, plus 2 catch-alls
(`AUX-OverallQuality`, `AUX-Other`).  Polysemous overrides correct cases
where the same raw string was misclassified due to ambiguity in context.

### 5. Fold into the per-family CSVs

```bash
python src/qcet_normalization/apply_qcet_to_metadata.py
```

Rewrites
`metadata_unique_counts/criteria/{llm,human}_criteria_normalization_{mapping,merges,stats_normalized}.csv`
with QCET ids attached.

### 6. Two-annotator Likert validation

```bash
python src/qcet_normalization/sample_validation_set.py         # stratified random sample (155 criteria)
python src/qcet_normalization/export_validation_to_excel.py    # build annotator xlsx
python src/qcet_normalization/score_validation.py              # after annotation: Likert summary
```

A 155-criterion stratified sample is sent to two annotators for a
five-point Likert evaluation. Strata cover strong agreements, rescued-by-
final-pass items, partial fits, top movers between initial and final
passes, and the new aux / extension nodes.

## Files

| File | Purpose |
|---|---|
| `qcet_taxonomy.json`            | The 111-leaf QCET taxonomy from Belz et al. (2024) |
| `aux_taxonomy_initial.json`     | A-priori auxiliary categories used at both the initial and final classification passes |
| `qcet_labels.py`                | Author-edited short labels for figures (single source of truth, also used by `analysis/analysis_scripts/data_loader.py`) |
| `calibration_pairs.csv`         | 30 gold pairs for batch-size calibration |
| `polysemous_overrides.csv`      | Hand-edited polysemous-string overrides applied after the final classification |
| `deepseek_client.py`            | LLM API client used by every stage that calls a model |
| `qcet_parser.py`                | Parses the published QCET markdown into `qcet_taxonomy.json` |
| `count_constructs_in_corpus.py` | Diagnostic: counts how many corpus criteria map to each QCET leaf |
| `outputs/`                      | Intermediate artefacts written by each stage (not committed) |

## Prompts

| Prompt | Used by |
|---|---|
| [`../../prompts_guidelines/classify_criteria_prompt.md`](../../prompts_guidelines/classify_criteria_prompt.md)     | `classify_criteria.py` (initial pass) |
| [`../../prompts_guidelines/reclassify_criteria_prompt.md`](../../prompts_guidelines/reclassify_criteria_prompt.md) | `reclassify_criteria.py` (final pass) |

## Final outputs

Committed under `analysis/intermediate_results/qcet/`:

| File | Contents |
|---|---|
| `criteria_classifications_final.csv` | Final per-variant classification (`chosen_id`, `chosen_name`, `chosen_source`, fit, etc.) |
| `aux_taxonomy_curated.json`          | The curated aux taxonomy (6 surviving categories + 2 catch-alls) |
| `validation_sample.csv` + `validation_annotator_{1,2}.xlsx` + `validation_summary.md` | Two-annotator validation sample + Likert scorecard |
