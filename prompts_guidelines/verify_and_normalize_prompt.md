# Verification and Normalization Prompt for NLG Evaluation Paper

You are verifying and improving the extracted metadata from a research paper about natural language generation (NLG) evaluation. Your task is to:

1. **Verify** the extracted yes/no answers are correct
2. **Normalize** metadata to use canonical forms (e.g., "BLEU" instead of "bleu")
3. **Correct** any incorrect items
4. **Add** any missing important items
5. **Remove** any irrelevant or incorrect items

## Paper Information

**Paper ID:** {paper_id}
**Title:** {title}
**Abstract:** {abstract}

**Full Paper Text:**
{full_text}

---

## Extracted Metadata to Review

### Question 1: Does the paper address NLG tasks?

**Extracted Answer:** {answer_1_answer}

**Extracted Metadata:**
- **Tasks:** {answer_1_tasks}
- **Datasets:** {answer_1_datasets}
- **Languages:** {answer_1_languages}
- **Models:** {answer_1_models}
- **Outputs:** {answer_1_outputs}

---

### Question 2: Does the paper use automatic metrics to evaluate the generated outputs?

**Extracted Answer:** {answer_2_answer}

**Extracted Metadata:**
- **Automatic Metrics:** {answer_2_metrics}

---

### Question 3: Does the paper use Large-Language Models (LLMs) as judges (i.e., *after* generation, an LLM is used to judge/assess the outputs)?

**Extracted Answer:** {answer_3_answer}

**Extracted Metadata:**
- **Models:** {answer_3_models}
- **Methods:** {answer_3_methods}
- **Criteria:** {answer_3_criteria}

---

### Question 4: Does the paper conduct *human* evaluations of the generated outputs?

**Extracted Answer:** {answer_4_answer}

**Extracted Metadata:**
- **Guideline:** {answer_4_guideline}
- **Criteria:** {answer_4_criteria}

---

## Your Task

For each question above:

1. **Verify the Yes/No answer** - Is it correct based on the full paper text?
2. **Review the metadata lists** - For each item:
   - Is it correctly extracted from the paper?
   - Is it relevant to the specific question?
   - Should it be normalized? (e.g., "BLEU" vs "bleu", "GPT-3" vs "gpt-3")
3. **Add missing items** - Are there important items mentioned in the paper that are missing?
4. **Remove incorrect items** - Are there items that shouldn't be there?

## Guidelines

### Normalization Rules
- Use canonical/standard forms (e.g., "BLEU" not "bleu", "GPT-3" not "gpt-3")
- Use consistent capitalization for metrics, models
- Use title case for tasks (e.g., "Machine Translation")
- **For metrics**: Simplify to base form (e.g., "BLEU-1", "BLEU-2", "BLEU-4" → all become "BLEU"; "ROUGE-1", "ROUGE-2", "ROUGE-L" → all become "ROUGE")
- **For models**: Keep version numbers distinct (e.g., "GPT-3", "GPT-4", "BERT-base", "BERT-large" are different)
- Merge case variations and abbreviations that refer to the same thing

### Verification Rules
- Only include items **explicitly mentioned** in the paper
- Focus on the **main contributions** - don't include every model/dataset mentioned in passing
- For tasks: Only include NLG tasks that are actually evaluated/studied
- For metrics: Include all automatic metrics used to evaluate NLG outputs
- For criteria: Include evaluation criteria used for human eval or LLM-as-evaluator
- Be accurate over complete - it's better to miss minor details than include wrong information

### Answer-Specific Guidelines

**Question 1 (Does the paper address NLG tasks?):**
- Answer "Yes" only if the paper studies/evaluates natural language GENERATION (not just understanding/classification)
- **Tasks**: Choose from {{"Text Summarization", "Dialogue Generation", "Paraphrase Generation", "Machine Translation", "Image Captioning", "Code Generation"}}. If none apply, use "Other:<task name>".
- **Datasets**: NLG datasets used
- **Languages**: Languages of the generated outputs (e.g., "English", "Chinese", "German")
- **Models**: NLG models being evaluated
- **Outputs**: Description of what is being generated

**Question 2 (Does the paper use automatic metrics to evaluate the generated outputs?):**
- Answer "Yes" if the paper uses any automatic metrics to evaluate generated text
- **Automatic Metrics**: List of automatic evaluation metrics (e.g., BLEU, ROUGE, METEOR, BERTScore)
  - **Important**: Simplify metric variants to base form (e.g., "BLEU-1", "BLEU-2", "BLEU-4" should all be normalized to just "BLEU"; "ROUGE-1", "ROUGE-2", "ROUGE-L" should all be "ROUGE")

**Question 3 (Does the paper use LLMs as judges?):**
- Answer "Yes" only if an LLM is used *after* generation to assess the outputs (not just as generation model)
- **Models**: Which LLMs are used for evaluation (e.g., "GPT-4", "Claude-3")
- **Methods**: Short name/description of the evaluation procedure or prompt (e.g., "pairwise evaluation", "direct scoring")
- **Criteria**: List the rubric properties the LLM is asked to score (e.g., "fluency", "relevance", "helpfulness"). If not specified, use empty list.

**Question 4 (Does the paper conduct human evaluations?):**
- Answer "Yes" if humans, annotators, raters, or crowdsourcing are used to evaluate generated outputs
- **Guideline**: Description of questions or criteria for the evaluation
- **Criteria**: List all criteria explicitly mentioned (e.g., "fluency", "coherence", "relevance"). If not specified, use empty list.

## Output Format

Please return ONLY a JSON object with the following structure:

```json
{{
  "paper_id": "{paper_id}",
  "answer_1": {{
    "answer": "Yes"/"No",
    "answer_changed": true/false,
    "tasks": ["normalized_task1", "normalized_task2", ...],
    "datasets": ["normalized_dataset1", "normalized_dataset2", ...],
    "languages": ["normalized_language1", "normalized_language2", ...],
    "models": ["normalized_model1", "normalized_model2", ...],
    "outputs": ["output_description1", "output_description2", ...],
    "changes_made": {{
      "added": {{"tasks": [...], "datasets": [...], "languages": [...], "models": [...], "outputs": [...]}},
      "removed": {{"tasks": [...], "datasets": [...], "languages": [...], "models": [...], "outputs": [...]}},
      "normalized": {{"original_item": "normalized_item", ...}},
      "explanation": "Brief explanation of major changes"
    }}
  }},
  "answer_2": {{
    "answer": "Yes"/"No",
    "answer_changed": true/false,
    "automatic_metrics": ["normalized_metric1", "normalized_metric2", ...],
    "changes_made": {{
      "added": {{"automatic_metrics": [...]}},
      "removed": {{"automatic_metrics": [...]}},
      "normalized": {{"original_metric": "normalized_metric", ...}},
      "explanation": "Brief explanation of major changes"
    }}
  }},
  "answer_3": {{
    "answer": "Yes"/"No",
    "answer_changed": true/false,
    "models": ["normalized_model1", "normalized_model2", ...],
    "methods": ["normalized_method1", "normalized_method2", ...],
    "criteria": ["normalized_criterion1", "normalized_criterion2", ...],
    "changes_made": {{
      "added": {{"models": [...], "methods": [...], "criteria": [...]}},
      "removed": {{"models": [...], "methods": [...], "criteria": [...]}},
      "normalized": {{"original_item": "normalized_item", ...}},
      "explanation": "Brief explanation of major changes"
    }}
  }},
  "answer_4": {{
    "answer": "Yes"/"No",
    "answer_changed": true/false,
    "guideline": ["guideline1", "guideline2", ...],
    "criteria": ["normalized_criterion1", "normalized_criterion2", ...],
    "changes_made": {{
      "added": {{"guideline": [...], "criteria": [...]}},
      "removed": {{"guideline": [...], "criteria": [...]}},
      "normalized": {{"original_item": "normalized_item", ...}},
      "explanation": "Brief explanation of major changes"
    }}
  }},
  "overall_notes": "Any general observations about the extraction quality or paper content"
}}
```

## Important Notes

- **Be conservative with changes** - Only modify if you're confident
- **Prioritize accuracy** - Better to keep existing correct items than to add uncertain ones
- **Normalize consistently** - Use standard naming conventions
- **Document major changes** - Explain why you added/removed important items
- **Use the full paper text** - Read the complete paper to verify all metadata is accurate and complete
