# Verification Prompt for NLG Evaluation Paper Extraction

You are verifying the extracted metadata from a research paper about natural language generation (NLG) evaluation methods. Your task is to carefully review the original paper and validate whether the extracted information is accurate.

## Paper Information
**Paper ID:** {paper_id}
**Title:** {title}
**Abstract:** {abstract}

## Extracted Metadata to Verify

### Answer 1: NLG Task
**Question:** Does this paper study or evaluate natural language generation (NLG) tasks?
**Extracted Answer:** {answer_1_answer}
**Extracted Quote:** {answer_1_quote}
**Extracted Metadata:**
- Tasks: {answer_1_tasks}
- Datasets: {answer_1_datasets}
- Languages: {answer_1_languages}
- Models: {answer_1_models}
- Outputs: {answer_1_outputs}

### Answer 2: Automatic Metrics
**Question:** Does this paper use automatic metrics to evaluate NLG outputs?
**Extracted Answer:** {answer_2_answer}
**Extracted Quote:** {answer_2_quote}
**Extracted Metadata:**
- Automatic Metrics: {answer_2_metrics}

### Answer 3: LLM-as-Evaluator
**Question:** Does this paper use large language models (LLMs) as evaluators for NLG outputs?
**Extracted Answer:** {answer_3_answer}
**Extracted Quote:** {answer_3_quote}
**Extracted Metadata:**
- Models: {answer_3_models}
- Methods: {answer_3_methods}
- Criteria: {answer_3_criteria}

### Answer 4: Human Evaluation
**Question:** Does this paper conduct human evaluation of NLG outputs?
**Extracted Answer:** {answer_4_answer}
**Extracted Quote:** {answer_4_quote}
**Extracted Metadata:**
- Guideline: {answer_4_guideline}
- Criteria: {answer_4_criteria}

## Verification Instructions

For each of the four answers above, please:

1. **Verify the Yes/No Answer**: Is the extracted answer (Yes/No) correct based on the paper content?
2. **Verify the Quote**: Are the provided quotes accurate and support the answer?
3. **Verify the Metadata**: Are the extracted metadata fields (tasks, datasets, metrics, models, etc.) accurate and complete?

## Output Format

Please provide your verification in the following JSON format:

```json
{{
  "paper_id": "PAPER_ID_HERE",
  "verification": {{
    "answer_1": {{
      "answer_correct": true/false,
      "answer_should_be": "Yes"/"No" (if incorrect),
      "quote_accurate": true/false,
      "metadata_accurate": true/false,
      "missing_items": ["list any missing tasks/datasets/languages/models/outputs"],
      "incorrect_items": ["list any incorrect items"],
      "comments": "Any additional observations"
    }},
    "answer_2": {{
      "answer_correct": true/false,
      "answer_should_be": "Yes"/"No" (if incorrect),
      "quote_accurate": true/false,
      "metadata_accurate": true/false,
      "missing_items": ["list any missing automatic metrics"],
      "incorrect_items": ["list any incorrect metrics"],
      "comments": "Any additional observations"
    }},
    "answer_3": {{
      "answer_correct": true/false,
      "answer_should_be": "Yes"/"No" (if incorrect),
      "quote_accurate": true/false,
      "metadata_accurate": true/false,
      "missing_items": ["list any missing models/methods/criteria"],
      "incorrect_items": ["list any incorrect items"],
      "comments": "Any additional observations"
    }},
    "answer_4": {{
      "answer_correct": true/false,
      "answer_should_be": "Yes"/"No" (if incorrect),
      "quote_accurate": true/false,
      "metadata_accurate": true/false,
      "missing_items": ["list any missing guidelines/criteria"],
      "incorrect_items": ["list any incorrect items"],
      "comments": "Any additional observations"
    }}
  }},
  "overall_quality": "excellent/good/fair/poor",
  "overall_comments": "General comments about the extraction quality"
}}
```

## Important Notes

- Focus on **accuracy** over **completeness** - it's acceptable if some minor details are missing, but extracted information must be correct
- For quotes, verify they are actual text from the paper (or close paraphrases)
- For metadata lists, check if items are relevant and correctly extracted
- If the paper is not about NLG evaluation at all, all answers should be "No"
- Be strict but fair in your assessment
