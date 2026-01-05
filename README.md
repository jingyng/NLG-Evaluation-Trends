This is the code for the paper "Order in the Evaluation Court:\\A Critical Analysis of NLG Evaluation Trends".

We organize our code and results in the following structure:

src:
- pdf2text: converting paper pdfs to full text
    - extract_url.py: We first get the paper urls from ACLantology, saved in paper_sources
    - download_pdf.py: Then we download the pdfs from the urls
    - batch_pdf2xml.py: We convert the pdfs into xmls using grobid
    - xml2json.py: We extract structured text with sections from xmls

- llm_annotations
    - run_llm_annotation.py: extracting metadata from papers using API, for three of the LLMs
    - merge_extractions.py: merging the results from three LLMs, binary answers are with majority merging, and the others are concatenated into a list.
    - run_llm_harmonization.py: harmonizing the mereged results with another LLM.
    - normalize_merged_results.py: normalizing merged results from LLM harmonization
    - run laaj_human_validation: extracting validation related metadata with papers having both LaaJ and human evaluation 

- term_normalization: We first create a complete list of all unique terms (tasks, datasets, languages, models, criteria), the normalize the list for each term category.

analysis:
- figures: figures produced with the normalized extractions
- intermediate_results: intermediate saved files for figure inputs
- results_analysis: code for the figures produced in the paper

results:
llm-annotations: llm annotations from each of the three LLMs, and the majority merged ones.
llm-merged-results: llm harmonized results by a fourth LLM (filtered NLG papers with majority voted "yes" as NLG)
llm-merged-results-normalized: llm harmonized results after term normalization
llm-merged-results-top30-tasks: top-30 filtered NLG papers results after normalization
llm-merged-results-top30-a3a4-yes: 433 papers that contain both LaaJ and human evaluation. 
laaj_human_validation_results: 433 papers with extracted validation results by an LLM. 
laaj_human_validation_results_normalized: 433 papers with extracted validation results and then normalized with term normalization. 

human_annotation: containing human annotation guidelines and annotation results

metadata_unique_counts: original and normalized terms and their mappings.

prompts_guidelines: full prompt for harmonization and validation between LaaJ and humans (LLM annotation prompt is directly coded in run_llm_annotation.py)

Due to limited space in github, we will provide a link to pdfs, xmls and jsons of of all papers after anonymous period. 