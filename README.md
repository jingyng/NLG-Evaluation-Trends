This repository contains the code and data for the paper **"Order in the Evaluation Court: A Critical Analysis of NLG Evaluation Trends"**.

## Overview

This project analyzes evaluation trends in Natural Language Generation (NLG) research by extracting and analyzing metadata from academic papers. The workflow covers paper collection, LLM-based annotation, term and QCET-based criterion normalization, and quantitative analysis of evaluation practices.

## Usage

The workflow follows these main steps:

1. **Paper Collection**: Extract URLs and download PDFs
2. **Text Extraction**: Convert PDFs to structured JSON
3. **LLM Annotation**: Extract metadata using multiple LLMs
4. **Harmonization**: Merge and harmonize LLM results
5. **Normalization**: Normalize tasks, metrics, models, languages, datasets, and evaluation criteria
6. **QCET-based criterion normalization**: Classify each criterion into the QCET quality taxonomy (4-stage LLM pipeline + human validation)
7. **Analysis**: Compute association measures and generate figures
8. **LaaJ–Human validation extraction**: Extract reported LaaJ-vs-human comparison values from dual-method papers

See individual scripts in `src/` and `analysis/` for detailed usage instructions.

## Repository Structure

### `src/`

Source code organized into four modules:

#### `pdf2text/`
Pipeline for converting paper PDFs to structured text:
- `extract_url.py`: Extracts paper URLs from ACL Anthology and saves them in `paper_sources/`
- `download_pdf.py`: Downloads PDFs from the extracted URLs
- `batch_pdf2xml.sh`: Converts PDFs to XML format using GROBID
- `xml2json.py`: Extracts structured JSON text with sections from XML files

#### `llm_annotation/`
LLM-based metadata extraction and processing:
- `run_llm_annotation.py`: Extracts metadata from papers using APIs with three different LLMs (DeepSeek-R1, GPT-OSS-120B, Qwen3-235B)
- `merge_extractions.py`: Merges results from the three LLMs (binary answers use majority voting, others are concatenated)
- `compute_agreement.py`: Computes Krippendorff α inter-LLM agreement on the four binary extraction questions
- `run_llm_harmonization.py`: Harmonizes merged results using a fourth LLM
- `normalize_merged_results.py`: Normalizes merged results from LLM harmonization
- `run_laaj_human_validation.py`: Extracts validation-related metadata for papers containing both LLM-as-a-Judge (LaaJ) and human evaluation

#### `term_normalization/`
Term normalization pipeline for tasks, metrics, models, languages, datasets, and (raw) evaluation criteria:
- `create_item_stats_csv.py`: Builds the unique-term statistics CSVs (one per term family) used as input to the normalisers
- `normalize_automatic_metrics.py`, `normalize_datasets.py`, `normalize_languages_stats.py`, `normalize_models.py`, `normalize_tasks_stats.py`: Family-specific normalisers (preprocessing + fuzzy matching + manual mapping)
- `normalize_human_criteria.py`, `normalize_llm_criteria.py`: Raw criterion-string normalisers (these are then re-mapped via QCET in `src/qcet_normalization/`)

#### `qcet_normalization/`
4-stage QCET-based criterion classification (Belz et al., 2024 taxonomy):
- `calibrate_stage0_batched.py`: Stage 0 — calibrate batch size against 30 gold pairs.
- `classify_stage1.py`: Stage 1 — task-blind classification of every raw variant against 111 QCET leaves.
- `cluster_stage2.py`: Stage 2 — HDBSCAN clustering of Stage-1 residuals (`fit ∈ {partial, none}`) to surface aux-category candidates.
- `decide_stage3.py` + `curate_stage3.py`: Stage 3 — non-reducibility verdicts (KEEP_AUX / FOLD / SPLIT / DROP) per cluster, followed by manual curation merging.
- `classify_stage4_simple.py` + `build_stage4_with_overrides.py`: Stage 4 — per-variant final classification against the 117 QCET leaves + 2 meta-categories, with manual polysemous overrides applied.
- `apply_qcet_to_metadata.py`: Folds the Stage-4 mapping back into the per-paper criterion-normalization CSVs.
- `sample_stage5_validation.py` + `score_stage5_validation.py` + `export_stage5_excel.py`: Stage 5 — stratified random validation sample (155 criteria) for two-annotator Likert-scale evaluation.
- `qcet_parser.py`, `qcet_taxonomy.json`, `aux_taxonomy_APRIORI.json`: QCET taxonomy + a-priori auxiliary categories used by Stages 1 and 4.
- `calibration_pairs.csv`, `polysemous_overrides.csv`, `polysemous_mappings_review.csv`: Curation artefacts.
- `docs/CLASSIFIER_DESIGN.md`, `docs/WALKTHROUGH.md`: Design rationale + 20-criterion walkthrough.

### `analysis/`

Analysis scripts, intermediate results, and figures:
- `analysis_scripts/`: Python scripts for aggregations, association measures, and figure generation. Includes per-task aggregators (`aggregate_{dialogue,mt,qa,summarization}_data.py`), task-dashboard plots (`plot_task_dashboard_rich_v2.py`), the metric–criterion split-heatmap (`plot_metric_criterion_split_heatmap.py`), the LaaJ–human alignment boxplot (`plot_validation_human_annotated.py`), and a shared `data_loader.py`.
- `association_analysis/`: LR + NPMI + G² robustness analysis (`analyze_metrics_vs_{human,llm}_eval.py`, `association_robustness.py`); generated CSVs under `association_robustness/`.
- `figures/`: Rendered PNG/PDF figures used in the paper.
- `intermediate_results/`: Intermediate aggregated files used as inputs for figure generation.
- `intermediate_results/qcet/`: Final QCET mapping (`stage4_classifications_simple_with_overrides.csv`), the final aux taxonomy (`stage3_aux_taxonomy_FINAL.json`), and the Stage-5 validation sample + two annotator spreadsheets.

### `results/`

Processed data at various stages:
- `llm-annotations/`: LLM annotations from each of the three LLMs, plus majority-merged results
- `llm-merged-results/`: LLM-harmonized results by a fourth LLM (filtered NLG papers with majority vote "yes" as NLG)
- `llm-merged-results-normalized/`: LLM-harmonized results after term + QCET-criterion normalization
- `llm-merged-results-top30-tasks/`: Top-30 filtered NLG papers (the 3,334-paper analysis corpus)
- `llm-merged-results-top30-a3a4-yes/`: 433 papers that contain both LaaJ and human evaluation
- `laaj_human_validation_results/`: 433 papers with extracted LaaJ-vs-human comparison values
- `laaj_human_validation_results_normalized/`: same, with QCET-criterion normalization applied

### Additional Directories

- `human_annotation/`: Human-annotation guidelines and Excel/markdown files. Includes the 90-paper LaaJ-vs-human validation annotation spreadsheet (`LaaJ against Human Validation.xlsx`) used by `plot_validation_human_annotated.py`.
- `metadata_unique_counts/`: Original and normalized terms with their mapping CSVs (one per term family) and `criteria_qcet_short_labels.csv` (QCET full name → short label table used by figure scripts).
- `prompts_guidelines/`: Prompts for the four extraction stages — initial annotation, verification, normalisation, and LaaJ-vs-human validation extraction.
- `paper_sources/`: Extracted paper URLs and metadata from ACL Anthology.

## Reproducing the paper

Figure-by-figure: each figure in the paper is produced by a single script in `analysis/analysis_scripts/`. The data pipeline is:

1. Run the QCET-normalisation pipeline once (`src/qcet_normalization/`, Stages 0–4) to produce `analysis/intermediate_results/qcet/stage4_classifications_simple_with_overrides.csv`.
2. Run `src/qcet_normalization/apply_qcet_to_metadata.py` to refresh the per-paper JSONs under `results/llm-merged-results-normalized/` and `results/llm-merged-results-top30-tasks/`.
3. Run the per-task aggregators (`aggregate_*_data.py`) to produce `analysis/intermediate_results/*_analysis_data/`.
4. Run any `plot_*.py` script in `analysis/analysis_scripts/` to render the corresponding figure into `analysis/figures/`.

## Data Availability

Due to GitHub storage limitations, PDFs, XMLs, and intermediate JSONs of all papers will be provided via external link after the anonymous review period. The repository ships with the post-normalisation JSON corpora needed to regenerate every figure.
