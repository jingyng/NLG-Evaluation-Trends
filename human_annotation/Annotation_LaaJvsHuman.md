# **Annotation Guideline: LaaJ vs. Human**

## **Overview**

You are analyzing a research paper that uses both LLM-as-a-judge (LaaJ) and human evaluation to evaluate natural language generation (NLG) outputs. Your task is to extract detailed information about how (or whether) the paper validates LaaJ evaluation against human evaluation. 

## **Annotation Process**

### **Step 0: Read the Paper: download and read the paper** using the PDF link provided in the spreadsheet

### **Step 1: Answer a Yes or No Question:** Does the paper explicitly compare LLM-as-a-Judge evaluation and humans?

### **How to Answer:**

Answer "Yes" only if the paper:  
\- Compares LLM and human judgments on the same set of instances  
\- Reports quantitative metrics of agreement/correlation between LLM and human  
\- Discusses the relationship between LLM and human evaluation results

Answer "No" if:  
\- Both LLM and human evaluations are conducted but never compared  
\- LLM and human evaluate different sets of instances or different aspects  
\- Only qualitative discussion without any comparison  
\- Other: no LaaJ or human evaluation

### **Step 2: Extract relevant metadata (if answer is yes).**

## **How to Answer:**

For **each** result/score reported, extract:  
\- **Value** (numerical \- extract exact value as reported)  
\- **Metric name** (e.g., "Spearman correlation", "Pearson r", "Cohen's kappa"; one row for each metric).   
\- Which **criterion** it applies to (e.g., "fluency", "coherence", or "overall"; one row for each criterion)  
\- Which **LLM** used as the judge (one row for each model)  
\- **Statistical significance (**if reported, whether result is significantly different between LaaJ and Human)  
\- **Evaluation Sample Size** (How many instances/examples were used for validation?)

How to extract: 

- **Validation Metrics** (list **all** metrics used to compare LLM vs human).   
- **Shared Evaluation Criteria** (criteria evaluated by **both** LLM and human): List **only criteria that both LLM and human evaluate**. This may be different from the full criteria listed for LaaJ or Human.  
- **Statistical significance**, select from one of the options:   
  - **Yes**: Significance is reported and the results are statistically significant	\*   
  - **No**: Significance is reported and the results are not statistically significant	\*   
  - **N/A**: Significance is not reported

**Important**: **One Value \= One Row.** If a paper reports scores for 3 different criteria across 2 different LLMs, you should fill out **6 separate rows** in the spreadsheet. Do not list multiple values in a single cell – you should **copy the row for the paper you are annotating 5 times**.  
