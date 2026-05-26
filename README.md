This repository contains the code and data for the paper **"What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020-2025"**.

## Overview

This project analyzes evaluation trends in Natural Language Generation (NLG) research by extracting and analyzing metadata from academic papers. The workflow covers paper collection, LLM-based annotation, term and QCET-based criterion normalization, and quantitative analysis of evaluation practices.

## Usage

The workflow follows these main steps:

1. **Paper Collection**: Extract URLs and download PDFs
2. **Text Extraction**: Convert PDFs to structured JSON
3. **LLM Annotation**: Extract metadata using multiple LLMs, and suplimentary validation results of LaaJ agaist human evaluations
4. **Harmonization**: Merge and harmonize LLM results
5. **Normalization**: Normalize tasks, metrics, models, languages, datasets, and evaluation criteria
6. **Analysis**: Compute association measures and generate figures

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
QCET-based criterion classification (Belz et al., 2024 taxonomy). Scripts are listed in pipeline order:
- `calibrate_batch_size.py`: calibrate the batched classifier against 30 gold pairs before the full run.
- `classify_criteria.py`: task-blind classification of every raw variant against the 111 QCET leaves; residuals (`fit ∈ {partial, none}`) get flagged for clustering.
- `cluster_residuals.py`: HDBSCAN clustering of the residuals to surface aux-category candidates.
- `decide_aux_categories.py` + `curate_aux_categories.py`: non-reducibility verdicts (KEEP_AUX / FOLD / SPLIT / DROP) per cluster, followed by manual curation.
- `reclassify_criteria.py` + `apply_polysemous_overrides.py`: per-variant final classification against 117 QCET leaves + 2 auxiliary categories, with manual polysemous overrides applied.
- `apply_qcet_to_metadata.py`: folds the final classification back into the per-paper criterion-normalization CSVs under `metadata_unique_counts/criteria/`.
- `sample_validation_set.py` + `export_validation_to_excel.py` + `score_validation.py`: stratified random validation sample (155 criteria) for two-annotator Likert-scale evaluation.
- `qcet_parser.py`, `qcet_taxonomy.json`, `aux_taxonomy_initial.json`: QCET taxonomy + a priori auxiliary categories shared by the initial and final classification passes.

### `analysis/`

Analysis scripts, intermediate results, and figures:
- `analysis_scripts/`: Python scripts for aggregations, association measures, and figure plotting. 
- `association_analysis/`: LR + G² robustness analysis (`analyze_metrics_vs_{human,llm}_eval.py`, `association_robustness.py`); generated CSVs under `association_robustness/`.
- `figures/`: Figures used in the paper.
- `intermediate_results/`: Intermediate aggregated files used as inputs for the figures.

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

- `human_annotation/`: Human-annotation guidelines and annotation results. 
- `metadata_unique_counts/`: Per-term-family subfolders (`criteria/`, `automatic_metrics/`, `datasets/`, `languages/`, `models/`), each containing the family's `*_normalization_mapping.csv` (raw → normalized lookup) and, where applicable, `*_normalization_merges.csv` (multi-variant groupings). `criteria/` additionally contains `qcet_short_labels.csv` (QCET full name → short label used by figure scripts).
- `prompts_guidelines/`: Prompts used by the LLM pipeline — initial verification + term normalisation (`verify_and_normalize_prompt.md`), LaaJ-vs-human validation extraction (`extract_laaj_human_validation_prompt.md`), and QCET criterion classification at Stage 1 / Stage 4 (`qcet_classification_stage{1,4}_prompt.md`).
- `paper_sources/`: Extracted paper URLs and metadata from ACL Anthology.

## Data Availability

Due to GitHub storage limitations, PDFs, XMLs, and intermediate JSONs of all papers will be provided via external link after the anonymous review period. The repository ships with the post-normalisation JSON corpora needed to regenerate every figure.
