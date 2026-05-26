# What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020–2025

This repository accompanies the paper **"What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020–2025"** and contains the code and data required to reproduce the 3,334-paper analysis corpus together with all figures and association analyses reported in the paper.

## Pipeline overview

The pipeline is organised into six sequential steps, each implemented by one module under `src/` or `analysis/`:

| # | Step | Module |
|---|---|---|
| 1 | Paper collection and text extraction | `src/pdf2text/` |
| 2 | LLM-based metadata extraction | `src/llm_annotation/` |
| 3 | Normalisation of each metadata category (tasks, automatic metrics, datasets, languages, models, evaluation criteria) | `src/term_normalization/` |
| 4 | QCET-based criterion classification | `src/qcet_normalization/` |
| 5 | Propagation of normalisation results to the corpus | `src/llm_annotation/normalize_merged_results.py` |
| 6 | Aggregation, association analysis, and figure generation | `analysis/` |

The post-normalisation JSON corpora required to regenerate every figure are committed to this repository. PDFs, XMLs, and per-paper JSONs exceed GitHub's file-size limits and will be released through an external link once the anonymous-review period concludes.

## Reproducing the pipeline

### Step 1 — Paper collection (`src/pdf2text/`)

```bash
python src/pdf2text/extract_url.py     # ACL Anthology → paper_sources/*.csv
python src/pdf2text/download_pdf.py    # → PDFs            (external link)
bash   src/pdf2text/batch_pdf2xml.sh   # → XML via GROBID  (external link)
python src/pdf2text/xml2json.py        # → per-paper JSON  (external link)
```

### Step 2 — Metadata extraction (`src/llm_annotation/`)

Metadata is extracted independently by three open-weight LLMs (DeepSeek-R1, GPT-OSS-120B, and Qwen3-235B) and aggregated using majority voting on binary fields together with concatenation of free-text fields. A fourth LLM then performs a verification and normalisation pass on the merged output. A separate extraction pipeline produces the LaaJ-vs-human-evaluation validation set used in the paper's validation analyses.

```bash
python src/llm_annotation/run_llm_annotation.py         # DeepSeek-R1 + GPT-OSS-120B + Qwen3-235B
python src/llm_annotation/merge_extractions.py          # majority vote on binary fields; concatenate the rest
python src/llm_annotation/compute_agreement.py          # Krippendorff α on the four binary fields
python src/llm_annotation/run_llm_harmonization.py      # fourth-LLM verification + normalisation pass
python src/llm_annotation/run_laaj_human_validation.py  # → results/laaj_human_validation_results/
```

Prompts used in this step: [`verify_and_normalize_prompt.md`](prompts_guidelines/verify_and_normalize_prompt.md) and [`extract_laaj_human_validation_prompt.md`](prompts_guidelines/extract_laaj_human_validation_prompt.md).

### Step 3 — Term normalisation (`src/term_normalization/`)

This step normalises six metadata categories (tasks, automatic metrics, datasets, languages, models, and evaluation criteria) using category-specific rules. Each normaliser combines orthographic preprocessing, fuzzy matching of near-duplicate variants, and a hand-curated mapping table. Evaluation criteria are normalised separately by source (human evaluation vs LLM-as-a-Judge) and receive only a preliminary canonicalisation at this stage; their final assignment to the QCET taxonomy is performed in Step 4.

```bash
python src/term_normalization/create_item_stats_csv.py       # builds {category}_stats.csv
python src/term_normalization/normalize_tasks_stats.py
python src/term_normalization/normalize_automatic_metrics.py
python src/term_normalization/normalize_datasets.py
python src/term_normalization/normalize_languages_stats.py
python src/term_normalization/normalize_models.py
python src/term_normalization/normalize_human_criteria.py
python src/term_normalization/normalize_llm_criteria.py
```

For each category, the outputs are written to `metadata_unique_counts/{category}/` as `{category}_normalization_mapping.csv` (raw → normalised lookup) and `{category}_normalization_merges.csv` (groupings of multiple variants under the same canonical form).

### Step 4 — QCET criterion classification (`src/qcet_normalization/`)

This step re-maps the criterion strings normalised in Step 3 onto the QCET taxonomy (Belz et al., 2024), together with a small auxiliary category set induced from residual variants that no QCET leaf adequately captures. The full six-stage workflow — calibration, initial classification, residual clustering, auxiliary-category decisions, final classification, and two-annotator validation — together with all inputs, outputs, and prompt references, is documented in [`src/qcet_normalization/README.md`](src/qcet_normalization/README.md).

### Step 5 — Propagation to the corpus

```bash
python src/llm_annotation/normalize_merged_results.py
```

This step propagates the normalised terms and QCET criterion mappings into every harmonised per-paper JSON, producing the following intermediate corpora:

| Folder | Contents |
|---|---|
| `results/llm-annotations/`                          | Per-LLM raw extractions and the majority-merged result |
| `results/llm-merged-results/`                       | Harmonised result, restricted to papers voted as NLG |
| `results/llm-merged-results-normalized/`            | Harmonised result with term and QCET normalisation applied |
| `results/llm-merged-results-top30-tasks/`           | 3,334-paper analysis corpus (Top-30 task filter) |
| `results/llm-merged-results-top30-a3a4-yes/`        | 433 papers reporting both LaaJ and human evaluation |
| `results/laaj_human_validation_results/`            | 433 papers with extracted LaaJ-vs-human comparison values |
| `results/laaj_human_validation_results_normalized/` | The same set with QCET-criterion normalisation applied |

### Step 6 — Analysis and figures (`analysis/`)

```bash
# Aggregations and likelihood-ratio + G² (Dunning) robustness analysis with BH-FDR correction
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
│   ├── llm_annotation/          # Steps 2 and 5
│   ├── term_normalization/      # Step 3
│   └── qcet_normalization/      # Step 4 (taxonomy, auxiliary categories, label helpers)
├── analysis/
│   ├── analysis_scripts/        # Figure plotters and the shared data_loader.py
│   ├── association_analysis/    # Likelihood-ratio and G² robustness analyses
│   ├── intermediate_results/    # Aggregations and QCET-pipeline outputs (under qcet/)
│   └── figures/                 # Final figures used in the paper
├── results/                     # See the table in Step 5
├── metadata_unique_counts/      # Per-category normalisation lookups (criteria, automatic_metrics,
│                                # datasets, languages, models; each with mapping and merges CSVs)
├── prompts_guidelines/          # LLM prompts for Steps 2 and 4
├── human_annotation/            # Annotation guidelines and results
└── paper_sources/               # ACL Anthology URL extracts
```

## Data availability

PDFs, XMLs, and per-paper JSONs will be released through an external link once the anonymous-review period concludes. The post-normalisation JSON corpora required to regenerate every figure are committed to this repository.

## Requirements

Python 3.10 or later. The principal third-party dependencies are `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`, `sentence-transformers`, `hdbscan`, `openai`, and `krippendorff`; see [`requirements.txt`](requirements.txt) for the complete list.
