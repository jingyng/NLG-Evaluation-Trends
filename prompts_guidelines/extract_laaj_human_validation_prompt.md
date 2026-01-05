# LLM-as-a-Judge vs Human Evaluation Validation Extraction

You are analyzing a research paper that uses **both** LLM-as-a-judge (LLM evaluators) and human evaluation to assess natural language generation outputs. Your task is to extract detailed information about how (or whether) the paper validates LLM evaluation against human evaluation.

## Paper Information

**Paper ID:** {paper_id}
**Title:** {title}
**Abstract:** {abstract}

**Full Paper Text:**
{full_text}

---

## Previously Extracted Metadata

### LLM-as-a-Judge (Answer 3)
- **Models:** {answer_3_models}
- **Methods:** {answer_3_methods}
- **Criteria:** {answer_3_criteria}

### Human Evaluation (Answer 4)
- **Guideline:** {answer_4_guideline}
- **Criteria:** {answer_4_criteria}

---

## Your Task

Extract information about **validation** of LLM-as-a-judge against human evaluation. Answer the following questions based on the full paper text:

### Question 1: Is there explicit validation?

**Does the paper explicitly compare LLM-as-a-judge results with human evaluation results?**

Answer "Yes" only if the paper:
- Compares LLM and human judgments on the same set of instances
- Reports quantitative metrics of agreement/correlation between LLM and human
- Discusses the relationship between LLM and human evaluation results

Answer "No" if:
- Both LLM and human evaluations are conducted but never compared
- LLM and human evaluate different sets of instances or different aspects
- Only qualitative discussion without any comparison

---

### Question 2: LLM-as-a-Judge Details

Extract detailed information about how LLMs were used as judges:

**A. Number of LLM Models**:
- How many different LLM models were used as judges?
- List the models (from Answer 3 metadata)

**B. LLM Prompts** (CRITICAL - Extract exact prompts if available):
- Does the paper show the exact prompt(s) used for LLM evaluation?
- If yes, extract the full prompt text verbatim (in appendix, main text, or figures)
- If no, describe what information is provided about the prompts
- Note: Look for prompts in main text, appendices, figures, or supplementary materials

---

### Question 3: Human Evaluation Details

Extract detailed information about how human evaluation was conducted:

**A. Number of Human Evaluators**:
- How many human annotators/evaluators were used?

**B. Evaluator Type**:
- "expert": Domain experts, researchers, or trained annotators
- "crowdsourced": Crowd workers (MTurk, Prolific, etc.)
- "mixed": Combination of both
- "unclear": Not specified

**C. Inter-Annotator Agreement** (CRITICAL):
- Was inter-annotator agreement (IAA) reported?
- If yes, extract:
  - Metric used (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, percentage agreement, etc.)
  - Value(s) reported
  - Interpretation if provided (e.g., "substantial agreement")
- If no, note "Not reported"

**D. Human Evaluation Guidelines**:
- Does the paper provide detailed evaluation guidelines/instructions?
- Are example annotations or scoring rubrics shown?
- Where are guidelines described (main text, appendix, supplementary)?

---

### Question 4: Validation Setup (if Q1 = Yes)

Extract the following information about how validation was conducted:

**A. Validation Type** (select all that apply):
- "correlation_analysis": Correlation between LLM and human scores (continuous values)
- "agreement_analysis": Agreement between LLM and human labels/judgments (categorical values)
- "ranking_comparison": Compare rankings produced by LLM vs human
- "error_analysis": Analyze disagreements between LLM and human
- "other": Other types of validation

**B. Validation Metrics** (list ALL metrics used to compare LLM vs human):
Examples: "Pearson correlation", "Spearman correlation", "Kendall's tau", "Cohen's kappa", "Accuracy", "F1", "Krippendorff's alpha", "Percentage agreement", etc.
- Note which metrics are correlation-based vs agreement-based

**C. Shared Evaluation Criteria** (criteria evaluated by BOTH LLM and human):
List only criteria that both LLM and human evaluate. This may be different from the full criteria lists in Answer 3/4.

**D. Sample Size**:
- How many instances/examples were used for validation?
- If multiple validation sets, list all sizes
- Note if validation uses subset or all evaluated instances

---

### Question 5: Validation Results (if Q1 = Yes)

Extract quantitative results comparing LLM and human:

**A. Correlation/Agreement Scores** (CRITICAL - Extract ALL reported values):
For EACH metric reported, extract:
- Metric name (e.g., "Spearman correlation", "Pearson r", "Cohen's kappa")
- Value (numerical - extract exact value as reported)
- Which criterion it applies to (e.g., "fluency", "coherence", or "overall")
- Which LLM model (if multiple LLMs were compared)
- Statistical significance if reported (p-value, confidence intervals)

Example: [{{"metric": "Spearman correlation", "value": 0.87, "criterion": "fluency", "llm_model": "GPT-4", "significance": "p<0.001"}}]

**Important**:
- Extract values for EACH criterion separately if reported
- Extract values for EACH LLM model separately if multiple models were compared
- Note if results are reported in tables, figures, or main text
- If ranges are given (e.g., "0.75-0.85"), note the range

**B. Correlation Strength Interpretation**:
- Does the paper interpret correlation strength (e.g., "strong", "moderate", "weak")?
- What threshold do they use for "strong" correlation?
- Do they compare to prior work?

---

## Output Format

Return ONLY a JSON object with this structure:

```json
{{
  "paper_id": "{{paper_id}}",
  "explicit_validation": {{
    "answer": "Yes"/"No",
    "explanation": "Brief explanation of why validation is or isn't present"
  }},
  "llm_judge_details": {{
    "num_models": 1,
    "models": ["GPT-4", ...],
    "prompts": {{
      "provided": "yes"/"no"/"partial",
      "location": "appendix"/"main_text"/"figure"/"supplementary"/"not_provided",
      "rubric_or_examples_shown": "yes"/"no",
      "notes": "Additional notes about prompts"
    }}
  }},
  "human_eval_details": {{
    "num_evaluators": 3,
    "evaluator_type": "expert"/"crowdsourced"/"mixed"/"unclear",
    "inter_annotator_agreement": {{
      "reported": "yes"/"no",
      "metric": "Fleiss' kappa",
      "value": 0.68,
      "interpretation": "substantial agreement",
      "per_criterion": [
        {{"criterion": "fluency", "metric": "Cohen's kappa", "value": 0.75}}
      ]
    }},
    "guidelines": {{
      "detailed_guidelines_provided": "yes"/"no"/"partial",
      "location": "appendix"/"main_text"/"supplementary"/"not_provided",
      "rubric_or_examples_shown": "yes"/"no",
      "notes": "Additional notes about guidelines"
    }}
  }},
  "validation_setup": {{
    "validation_types": ["correlation_analysis", "agreement_analysis", ...],
    "validation_metrics": ["Pearson correlation", "Cohen's kappa", ...],
    "shared_criteria": ["fluency", "coherence", ...],
    "sample_size": {{
      "total_generated": 100,
      "validated_by_both": 50,
      "notes": "Additional notes about sample size"
    }}
  }},
  "validation_results": {{
    "quantitative_scores": [
      {{
        "metric": "Spearman correlation",
        "value": 0.87,
        "criterion": "fluency",
        "llm_model": "GPT-4",
        "significance": "p<0.001",
        "notes": "Optional notes"
      }}
    ],
    "correlation_interpretation": {{
      "strength_described": "yes"/"no",
      "interpretation": "strong"/"moderate"/"weak",
      "threshold_used": "Correlation > 0.7 considered strong",
      "compared_to_prior_work": "yes"/"no"
    }},
    "summary_finding": "1-2 sentence summary of LLM-human alignment"
  }},
  "criteria_mapping": {{
    "llm_only_criteria": ["criterion1", ...],
    "human_only_criteria": ["criterion2", ...],
    "shared_criteria": ["criterion3", ...],
    "notes": "Explanation of any criterion differences"
  }}
}}
```

## Important Notes

**If explicit_validation.answer = "No":**
- Set validation_setup and validation_results to `null`
- Still fill out llm_judge_details and human_eval_details (these are independent of validation)
- Still fill out criteria_mapping to show which criteria are used by each method

**Always extract (regardless of validation):**
- llm_judge_details: Always extract LLM setup information
- human_eval_details: Always extract human evaluation information
- criteria_mapping: Always show which criteria each method uses

**Guidelines:**
- **Be precise**: Only mark as validated if there's explicit comparison
- **Extract exact values**: Copy numerical results exactly as reported
- **Distinguish correlation types**: Pearson vs Spearman vs Kendall
- **Note statistical significance**: If p-values or confidence intervals are reported
- **Consider multi-criterion scenarios**: LLM and human might evaluate different criteria even in same paper

**Common Scenarios:**

1. **Full validation**: Paper uses LLM to evaluate all outputs, validates on human-annotated subset, reports correlation
2. **Parallel evaluation**: Both LLM and human evaluate the same outputs, direct comparison
3. **Sequential validation**: Human labels used as ground truth, LLM accuracy measured
4. **Independent streams**: Both methods used but never compared (answer "No")
5. **Qualitative only**: Paper discusses differences but no quantitative comparison (answer "No")

---

## Examples

### Example 1: Explicit Validation (Yes)

Paper text excerpt: *"We validate GPT-4 judgments against human ratings on a subset of 200 summaries. The Spearman correlation for coherence is ρ=0.81 (p<0.001) and for fluency ρ=0.75 (p<0.001). This suggests GPT-4 can reliably replicate human judgments..."*

```json
{{
  "explicit_validation": {{
    "answer": "Yes",
    "explanation": "Paper explicitly validates GPT-4 against human on 200 summaries with correlation analysis"
  }},
  "validation_results": {{
    "quantitative_scores": [
      {{"metric": "Spearman correlation", "value": 0.81, "criterion": "coherence", "notes": "p<0.001"}},
      {{"metric": "Spearman correlation", "value": 0.75, "criterion": "fluency", "notes": "p<0.001"}}
    ],
    "summary_finding": "GPT-4 judgments show strong correlation with human ratings (ρ=0.75-0.81), suggesting reliable replication of human judgments."
  }}
}}
```

### Example 2: No Validation

Paper text excerpt: *"We evaluate outputs using GPT-4 to assess relevance and factuality. Additionally, we conduct human evaluation focusing on overall quality and user satisfaction..."*

```json
{{
  "explicit_validation": {{
    "answer": "No",
    "explanation": "Both LLM and human evaluations conducted but they assess different criteria and no comparison is made between the two methods"
  }},
  "criteria_mapping": {{
    "llm_only_criteria": ["relevance", "factuality"],
    "human_only_criteria": ["overall quality", "user satisfaction"],
    "shared_criteria": [],
    "notes": "LLM and human evaluate complementary aspects without overlap"
  }}
}}
```
---

**Read the full paper text carefully** and extract all validation-related information accurately.
