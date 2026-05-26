# What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020–2025

Code and data for **"What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020–2025"**. Run the pipeline below to reproduce the 3,334-paper analysis corpus and every figure and association analysis reported in the paper.

## Quick overview

The pipeline runs in six steps, each implemented by one module under `src/` (or `analysis/`):

| # | Step | Module |
|---|---|---|
| 1 | Paper collection + text extraction        | `src/pdf2text/` |
| 2 | LLM-based metadata extraction              | `src/llm_annotation/` |
| 3 | Normalise each metadata category (tasks, automatic metrics, datasets, languages, models, criteria) separately | `src/term_normalization/` |
| 4 | QCET criterion classification              | `src/qcet_normalization/` |
| 5 | Apply normalisation to the corpus          | `src/llm_annotation/normalize_merged_results.py` |
| 6 | Aggregations, association analysis, figures | `analysis/` |

The post-normalisation JSON corpora needed for every figure are committed; PDFs, XMLs, and per-paper JSONs are too large for GitHub and will be released via an external link after anonymous review.

## Reproducing the pipeline

### Step 1 — Paper collection (`src/pdf2text/`)

```bash
python src/pdf2text/extract_url.py     # ACL Anthology → paper_sources/*.csv
python src/pdf2text/download_pdf.py    # → PDFs            (external link)
bash   src/pdf2text/batch_pdf2xml.sh   # → XML via GROBID  (external link)
python src/pdf2text/xml2json.py        # → per-paper JSON  (external link)
```

### Step 2 — Metadata extraction (`src/llm_annotation/`)

Triple-LLM extraction, majority-voted merge, fourth-LLM harmonisation, and a separate LaaJ ↔ human-evaluation extraction.

```bash
python src/llm_annotation/run_llm_annotation.py         # DeepSeek-R1 + GPT-OSS-120B + Qwen3-235B
python src/llm_annotation/merge_extractions.py          # majority vote on binary fields; concatenate the rest
python src/llm_annotation/compute_agreement.py          # Krippendorff α on the four binary fields
python src/llm_annotation/run_llm_harmonization.py      # fourth-LLM verification + normalisation pass
python src/llm_annotation/run_laaj_human_validation.py  # → results/laaj_human_validation_results/
```

Prompts: [`verify_and_normalize_prompt.md`](prompts_guidelines/verify_and_normalize_prompt.md), [`extract_laaj_human_validation_prompt.md`](prompts_guidelines/extract_laaj_human_validation_prompt.md).

### Step 3 — Term normalisation (`src/term_normalization/`)

Each metadata category (tasks, automatic metrics, datasets, languages, models, and human / LLM-as-a-Judge criteria) has its own normaliser — same shape (preprocessing + fuzzy matching + manual mappings) but category-specific rules and a separate output CSV. Criteria are stub-normalised here and re-mapped by the QCET pipeline in Step 4.

```bash
python src/term_normalization/create_item_stats_csv.py       # builds {family}_stats.csv
python src/term_normalization/normalize_tasks_stats.py
python src/term_normalization/normalize_automatic_metrics.py
python src/term_normalization/normalize_datasets.py
python src/term_normalization/normalize_languages_stats.py
python src/term_normalization/normalize_models.py
python src/term_normalization/normalize_human_criteria.py
python src/term_normalization/normalize_llm_criteria.py
```

Outputs for each category land in `metadata_unique_counts/{category}/`:
`{category}_normalization_mapping.csv` (raw → normalised) and `{category}_normalization_merges.csv` (multi-variant groupings).

### Step 4 — QCET criterion classification (`src/qcet_normalization/`)

Re-maps the normalised criterion strings onto the QCET taxonomy (Belz et al., 2024) plus a small auxiliary category set induced from residuals. The full six-stage workflow (calibration → initial classification → residual clustering → aux-category decisions → final classification → two-annotator validation), with inputs, outputs, and prompt references, is documented in [`src/qcet_normalization/README.md`](src/qcet_normalization/README.md).

### Step 5 — Apply normalisation to the corpus

```bash
python src/llm_annotation/normalize_merged_results.py
```

Rewrites every harmonised paper JSON with the normalised terms + QCET criterion mappings. Produces, in order:

| Folder | Contents |
|---|---|
| `results/llm-annotations/`                       | Per-LLM raw extractions + majority-merged result |
| `results/llm-merged-results/`                    | Harmonised result, papers voted Yes-NLG |
| `results/llm-merged-results-normalized/`         | Same, with term + QCET normalisation applied |
| `results/llm-merged-results-top30-tasks/`        | 3,334-paper analysis corpus (Top-30 task filter) |
| `results/llm-merged-results-top30-a3a4-yes/`     | 433 papers with both LaaJ and human evaluation |
| `results/laaj_human_validation_results/`         | 433 papers with extracted LaaJ-vs-human values |
| `results/laaj_human_validation_results_normalized/` | Same, with QCET-criterion normalisation |

### Step 6 — Analysis and figures (`analysis/`)

```bash
# Aggregations + LR + G² (Dunning) robustness with BH-FDR
python analysis/association_analysis/analyze_metrics_vs_human_eval.py
python analysis/association_analysis/analyze_metrics_vs_llm_eval.py
python analysis/association_analysis/association_robustness.py

# Paper figures (each script writes to analysis/figures/)
python analysis/analysis_scripts/plot_<figure_name>.py
```

## Repository layout

```
acl2026-nlg-eval/
├── src/
│   ├── pdf2text/                # Step 1
│   ├── llm_annotation/          # Steps 2 + 5
│   ├── term_normalization/      # Step 3
│   └── qcet_normalization/      # Step 4 (+ qcet_taxonomy.json, aux_taxonomy_initial.json, qcet_labels.py)
├── analysis/
│   ├── analysis_scripts/        # Figure plotters + shared data_loader.py
│   ├── association_analysis/    # LR + G² robustness
│   ├── intermediate_results/    # Aggregations + QCET-pipeline outputs (under qcet/)
│   └── figures/                 # Final figures used in the paper
├── results/                     # See the table in Step 5
├── metadata_unique_counts/      # Per-category normalisation lookups (criteria, automatic_metrics,
│                                # datasets, languages, models — each with mapping + merges CSVs)
├── prompts_guidelines/          # LLM prompts for Steps 2 and 4
├── human_annotation/            # Annotation guidelines + results
└── paper_sources/               # ACL Anthology URL extracts
```

## Data availability

PDFs, XMLs, and per-paper JSONs are released via an external link after the anonymous-review period. The post-normalisation JSON corpora needed to regenerate every figure are committed.

## Setup

See [`requirements.txt`](requirements.txt). Python 3.10+; the only non-standard dependencies are `sentence-transformers`, `hdbscan`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `openpyxl`.
