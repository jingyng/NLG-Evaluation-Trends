# Order in the Evaluation Court: A Critical Analysis of NLG Evaluation Trends

This repository contains the code and data for the paper **"Order in the Evaluation Court: A Critical Analysis of NLG Evaluation Trends"**.

## Overview

This project analyzes evaluation trends in Natural Language Generation (NLG) research by extracting and analyzing metadata from academic papers. The workflow includes paper collection, LLM-based annotation, term normalization, and results analysis.

## Usage

The workflow follows these main steps:

1. **Paper Collection**: Extract URLs and download PDFs
2. **Text Extraction**: Convert PDFs to structured JSON
3. **LLM Annotation**: Extract metadata using multiple LLMs
4. **Harmonization**: Merge and harmonize LLM results
5. **Normalization**: Normalize terms across categories
6. **Analysis**: Generate figures and analyses

See individual scripts in `src/` for detailed usage instructions.

## Repository Structure

### `src/`

Source code organized into three main modules:

#### `pdf2text/`
Pipeline for converting paper PDFs to structured text:
- `extract_url.py`: Extracts paper URLs from ACL Anthology and saves them in `paper_sources/`
- `download_pdf.py`: Downloads PDFs from the extracted URLs
- `batch_pdf2xml.py`: Converts PDFs to XML format using GROBID
- `xml2json.py`: Extracts structured JSON text with sections from XML files

#### `llm_annotation/`
LLM-based metadata extraction and processing:
- `run_llm_annotation.py`: Extracts metadata from papers using APIs with three different LLMs
- `merge_extractions.py`: Merges results from the three LLMs (binary answers use majority voting, others are concatenated)
- `run_llm_harmonization.py`: Harmonizes merged results using a fourth LLM
- `normalize_merged_results.py`: Normalizes merged results from LLM harmonization
- `run_laaj_human_validation.py`: Extracts validation-related metadata for papers containing both LLM-as-a-Judge (LaaJ) and human evaluation

#### `term_normalization/`
Term normalization pipeline:
- Creates a complete list of all unique terms (tasks, datasets, languages, models, criteria)
- Normalizes the list for each term category
- Generates normalization mappings and statistics

### `analysis/`

Analysis scripts and generated outputs:
- `analysis_scripts/`: Python scripts for generating figures and analyses
- `figures/`: Generated figures from the normalized extractions
- `intermediate_results/`: Intermediate saved files used as inputs for figure generation

### `results/`

Processed data at various stages:
- `llm-annotations/`: LLM annotations from each of the three LLMs, plus majority-merged results
- `llm-merged-results/`: LLM-harmonized results by a fourth LLM (filtered NLG papers with majority vote "yes" as NLG)
- `llm-merged-results-normalized/`: LLM-harmonized results after term normalization
- `llm-merged-results-top30-tasks/`: Top-30 filtered NLG papers results after normalization
- `llm-merged-results-top30-a3a4-yes/`: 433 papers that contain both LaaJ and human evaluation
- `laaj_human_validation_results/`: 433 papers with extracted validation results by an LLM
- `laaj_human_validation_results_normalized/`: 433 papers with extracted validation results, normalized with term normalization

### Additional Directories

- `human_annotation/`: Human annotation guidelines and annotation results
- `metadata_unique_counts/`: Original and normalized terms with their mappings
- `prompts_guidelines/`: Full prompts for harmonization and validation between LaaJ and humans (LLM annotation prompt is directly coded in `run_llm_annotation.py`)
- `paper_sources/`: Extracted paper URLs and metadata from ACL Anthology

## Data Availability

Due to GitHub storage limitations, PDFs, XMLs, and JSONs of all papers will be provided via external link after the anonymous review period.

