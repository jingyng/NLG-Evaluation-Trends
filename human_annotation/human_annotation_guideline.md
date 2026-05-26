# Human Annotation Guidelines for NLG Evaluation Papers

## Overview

This document provides detailed instructions for manually annotating research papers about natural language generation (NLG) evaluation. You will read each paper and extract structured metadata to answer four main questions about the paper's approach to NLG evaluation.

**Important:** The papers you are annotating have been **pre-filtered as potential NLG papers** (with an initial "Yes" answer to Question 1). However, this filtering may not be perfect. **You should verify this classification** and change the answer to "No" if, after reading the paper, you determine it does not actually address NLG tasks.

---

## Annotation Process

### Step 1: Read the Paper

1. **Download and read the paper** using the PDF link provided in the spreadsheet
2. **Focus on the following sections**:
   - Abstract and Introduction (for understanding the task and goals)
   - Methodology/Approach (for understanding the models and datasets)
   - Experimental Setup/Evaluation (for understanding metrics and evaluation procedures)
   - Results section (to confirm what was actually evaluated)
3. **Take notes** as you read to identify relevant information for each question

### Step 2: Answer Four Main Questions

For each paper, you will answer four yes/no questions and extract relevant metadata. Work through each question in order.

---

## Question 1: Does the paper address NLG tasks?

### Definition

Natural Language Generation (NLG) refers to tasks where a system **produces/generates** natural language text as output. This is distinct from Natural Language Understanding (NLU) tasks where the system only reads/analyzes text.

### Important Context

**These papers have been pre-filtered as NLG papers** (initially classified as "Yes"). However, the automatic filtering may have made mistakes. Your job is to **verify this classification** by carefully reading the paper.

### How to Answer

**Answer "Yes" if:**
- The paper generates text, sentences, or natural language as output
- The paper evaluates or studies systems that produce natural language
- Examples: summarization systems, dialogue systems, machine translation, paraphrase generation, image captioning

**Answer "No" if:**
- The paper only does classification, tagging, or understanding tasks
- No text is generated as output
- Examples: sentiment analysis, named entity recognition, question answering with extractive answers (just selecting existing text)
- The paper was incorrectly classified during pre-filtering

**If you change the answer to "No":**
- **Skip this paper entirely** and move to the next paper
- You do not need to fill in any other fields (Q1 metadata, Q2, Q3, Q4)
- We are only interested in papers that address NLG tasks

### Metadata to Extract (if answer is "Yes")

#### 1. **Tasks** (List of NLG task types)

**What to include:**
- The main NLG task(s) that the paper addresses
- Use **standardized task names** from this list:
  - Text Summarization
  - Dialogue Generation
  - Paraphrase Generation
  - Machine Translation
  - Image Captioning
  - Code Generation
  - Data-to-Text Generation
  - Question Generation
  - Story Generation
  - News Generation
- If the task doesn't fit any category, use: "Other: [specific task name]"

**Examples:**
- ✅ "Text Summarization", "Dialogue Generation"
- ❌ Don't include: "NLP", "Generation" (too vague)

**Instructions:**
- Include only the **primary task(s)** being studied/evaluated
- Don't include tasks mentioned only in related work or background
- If a paper studies multiple NLG tasks, list all of them

---

#### 2. **Datasets** (List of dataset names)

**What to include:**
- Names of NLG datasets used for experiments or evaluation
- Include datasets that are central to the paper's contribution
- Use the **official dataset name** as cited in the paper

**Examples:**
- ✅ "CNN/DailyMail", "XSum", "WMT14", "MultiWOZ"
- ❌ Don't include: Generic terms like "news articles", "dialogue data"

**Instructions:**
- Only include datasets that are **actually used** in the paper's experiments
- Don't include datasets only mentioned in related work
- If a paper creates a new dataset, include its name
- Use the exact name from the paper, but normalize capitalization

---

#### 3. **Languages** (List of languages)

**What to include:**
- The language(s) of the **generated outputs**
- Use standard language names in English

**Examples:**
- ✅ "English", "Chinese", "German", "French"
- ❌ Don't use: ISO codes like "en", "zh" (use full names)

**Instructions:**
- Include all target languages for generation
- For multilingual papers, list all languages mentioned
- If the paper doesn't specify but uses English datasets, annotate as "English"

---

#### 4. **Models** (List of model names)

**What to include:**
- Names of NLG **models used or proposed for GENERATION** (not evaluation)
- These are models that **produce/generate the text outputs**
- Include both models proposed by the authors and baseline generation models

**IMPORTANT:** This is for **generation models only**, NOT evaluation models:
- ✅ Include: Models that generate the summaries, translations, dialogue responses, etc.
- ❌ Don't include: Models used to evaluate/judge outputs (those go in Q3)
- Example: If GPT-4 generates text → Q1. If GPT-4 judges/evaluates text → Q3.

**Examples:**
- ✅ "GPT-3", "BART", "T5", "Seq2Seq", "Transformer" (if used for generation)
- ✅ "GPT-3.5", "GPT-4" (keep versions distinct when specified)
- ❌ Don't include: Models only used for evaluation/judging outputs

**Normalization rules:**
- Use **canonical model names** with proper capitalization
- **Keep version numbers distinct**: "GPT-3" vs "GPT-4" are different models
- Normalize case variations: "gpt-3" → "GPT-3", "bert" → "BERT"
- Include size variants if specified: "BERT-base", "BERT-large"

**Instructions:**
- Focus on models that **generate the NLG outputs being evaluated**
- Don't list every model mentioned in passing in related work
- If a paper proposes a new generation model with a name, include it
- For papers proposing unnamed approaches, describe briefly: "Proposed model"
- Remember: Evaluation models go in Q3, not here!

---

#### 5. **Outputs** (List of output descriptions)

**What to include:**
- Brief descriptions of **what text is being generated**
- Focus on the actual output artifacts, not the process

**Examples:**
- ✅ "News article summaries", "Task-oriented dialogue responses", "English-to-German translations", "Image captions"
- ❌ Don't be too technical: "decoder hidden states", "token embeddings"

**Instructions:**
- Use natural language descriptions
- Be specific but concise (3-7 words typically)
- If multiple types of outputs, list the main ones
- Focus on what is generated, not how

---

## Question 2: Does the paper use automatic metrics to evaluate the generated outputs?

### Definition

Automatic metrics are computational measures that evaluate the quality of generated text without human involvement. These metrics compare generated text against reference texts or use learned models to score outputs.

### How to Answer

**Answer "Yes" if:**
- The paper reports scores from any automatic evaluation metrics
- Metrics are used to compare different systems or configurations
- Common examples: BLEU, ROUGE, METEOR, BERTScore, BLEURT, ChrF

**Answer "No" if:**
- The paper only uses human evaluation
- Only intrinsic metrics (like perplexity during training) are reported
- No evaluation of generated outputs is performed

### Metadata to Extract (if answer is "Yes")

#### **Automatic Metrics** (List of metric names)

**What to include:**
- All automatic evaluation metrics used to assess the generated outputs
- Use **standardized metric names** with proper capitalization

**Examples:**
- ✅ "BLEU", "ROUGE", "METEOR", "BERTScore", "BLEURT", "ChrF", "TER", "PARENT"
- ❌ Don't use: "bleu", "Blue" (wrong capitalization)

**Critical normalization rule:**
- **Simplify metric variants to their base form**:
  - "BLEU-1", "BLEU-2", "BLEU-4" → all become **"BLEU"**
  - "ROUGE-1", "ROUGE-2", "ROUGE-L" → all become **"ROUGE"**
  - "F1-score", "Exact Match" → keep as separate metrics
  - "BERT-F1", "BERTScore" → use **"BERTScore"**

**Why normalize variants?**
- We want to know which **metric families** are used, not every variant
- This simplifies analysis and prevents overcounting similar metrics

**Instructions:**
- Include all metrics actually used in the evaluation section
- Don't include metrics only mentioned in related work
- Use the base metric name (BLEU not BLEU-4)
- Standard capitalizations:
  - BLEU, ROUGE, METEOR (all caps)
  - BERTScore, BARTScore (title case with acronym prefix)
  - ChrF, TER, WER (all caps for acronyms)

---

## Question 3: Does the paper use Large Language Models (LLMs) as judges?

### Definition

This refers to using LLMs **after generation** to automatically evaluate or judge the quality of generated outputs. The LLM is used as an evaluator, not as the generation model itself.

### How to Answer

**Answer "Yes" if:**
- An LLM (like GPT-4, Claude, PaLM) is used to **score, rank, or judge** generated outputs
- The paper describes using LLM prompts to assess quality
- Examples: "GPT-4 as a judge", "LLM-based evaluation", "using ChatGPT to rate fluency"

**Answer "No" if:**
- LLMs are only used for **generation**, not evaluation
- No LLM-based evaluation is performed
- Only traditional automatic metrics or human evaluation is used

**Important distinction:**
- If GPT-4 **generates** the text → this doesn't count for Q3 (but counts for Q1)
- If GPT-4 **evaluates** text generated by another system → this **does** count for Q3

### Metadata to Extract (if answer is "Yes")

#### 1. **Models** (List of LLM names used as judges)

**What to include:**
- Names of specific LLMs used for evaluation
- Include version numbers when specified

**Examples:**
- ✅ "GPT-4", "GPT-3.5", "Claude-3", "PaLM-2", "Llama-2-70B"
- ❌ Don't use: "ChatGPT" (use "GPT-3.5" or "GPT-4" if version is known)

**Normalization rules:**
- Use official model names with proper capitalization
- Keep versions distinct: "GPT-3" vs "GPT-4"
- If paper says "ChatGPT" without version, keep as "ChatGPT"
- Format: "ModelName-Version" (e.g., "Claude-3-Opus", "Llama-2-70B")

---

#### 2. **Methods** (List of evaluation methods/approaches)

**What to include:**
- Brief description or name of the evaluation procedure
- How the LLM is prompted or used

**Examples:**
- ✅ "Pairwise comparison", "Direct scoring", "Likert scale rating", "Binary preference", "Multi-aspect scoring"
- ✅ "Chain-of-thought evaluation", "Self-consistency"
- ❌ Don't include: Full prompt text (too detailed)

**Instructions:**
- Use short descriptive names (2-5 words)
- If the paper gives a name to their method, use it
- If not, describe the approach briefly
- Focus on the evaluation procedure, not the scoring scale

---

#### 3. **Criteria** (List of evaluation criteria)

**What to include:**
- The specific aspects or dimensions that the LLM is asked to evaluate
- The rubric properties being scored

**Examples:**
- ✅ "Fluency", "Relevance", "Coherence", "Factuality", "Helpfulness", "Safety"
- ❌ Don't include: The scores themselves (like "1-5 scale")

**If criteria are not specified:**
- Use an **empty list** if the paper doesn't mention specific criteria
- Don't guess or infer criteria

**Instructions:**
- Use the exact terminology from the paper when possible
- Normalize capitalization: lowercase first letter ("fluency" not "Fluency")
- List all criteria mentioned
- If paper uses very general terms like "quality", still include it

---

## Question 4: Does the paper conduct human evaluations of the generated outputs?

### Definition

Human evaluation means that real people (not LLMs or automatic metrics) are asked to read and assess the generated outputs. This includes crowdsourcing, expert annotations, or user studies.

### How to Answer

**Answer "Yes" if:**
- Human annotators, raters, or judges evaluate the generated text
- Crowdsourcing platforms (MTurk, Prolific) are used
- User studies with human participants assess outputs
- Expert evaluations of generated text are conducted

**Answer "No" if:**
- Only automatic metrics or LLM judges are used
- No human feedback on generated outputs is collected
- Humans are only used for data collection, not evaluation

### Metadata to Extract (if answer is "Yes")

#### 1. **Methods** (List of evaluation methods/approaches)

**What to include:**
- Brief description or name of the evaluation procedure
- How human evaluators are asked to assess the outputs
- The type of evaluation task (rating, ranking, comparison, etc.)

**Examples:**
- ✅ "Pairwise comparison", "Direct scoring", "Likert scale rating", "Binary preference", "Multi-aspect rating"
- ✅ "Ranking", "Best-worst scaling", "Magnitude estimation"
- ✅ "A/B testing", "Adequacy and fluency rating"
- ❌ Don't include: Full instruction text or specific questions (too detailed)

**Instructions:**
- Use short descriptive names (2-5 words)
- If the paper gives a name to their evaluation method, use it
- If not, describe the approach briefly (e.g., "5-point Likert scale rating")
- Focus on the evaluation procedure, not the specific criteria being evaluated
- If multiple different evaluation methods are used, list them separately

---

#### 2. **Criteria** (List of evaluation criteria)

**What to include:**
- The specific aspects or dimensions that humans are asked to evaluate
- Evaluation categories or rubric items

**Examples:**
- ✅ "Fluency", "Adequacy", "Coherence", "Informativeness", "Naturalness", "Relevance"
- ✅ "Grammaticality", "Readability", "Factual accuracy"
- ❌ Don't include: The scores themselves

**If criteria are not specified:**
- Use an **empty list** if the paper only describes a general "quality" rating without specific dimensions
- Don't infer criteria if not explicitly stated

**Instructions:**
- Use exact terminology from the paper
- Normalize capitalization: lowercase first letter ("fluency" not "Fluency")
- List all criteria mentioned in the evaluation setup
- If the paper uses "overall quality" as the only criterion, include it

---

## General Annotation Guidelines

### Quality Standards

1. **Be accurate**: Only annotate information that is **explicitly stated** in the paper
2. **Be complete**: Try to find all relevant information for each question
3. **Be consistent**: Use standardized names and formats
4. **Be conservative**: When in doubt, don't guess—leave it out or mark as uncertain

### Handling Edge Cases

**If you're unsure about something:**
- Add a comment/note in your annotation
- Mark items you're uncertain about
- It's better to be cautious than incorrect

**If information is ambiguous:**
- Use your best judgment based on context
- Add a note explaining your interpretation

**If a field should be empty:**
- Leave it as an empty list, don't put placeholder text
- Empty lists are valid annotations

### Normalization Standards

**Capitalization:**
- **Metrics**: Follow standard conventions (BLEU, ROUGE, BERTScore)
- **Models**: Use official capitalization (GPT-4, BERT, T5)
- **Tasks**: Use title case (Text Summarization, Machine Translation)
- **Criteria**: Use lowercase (fluency, coherence, relevance)
- **Languages**: Capitalize (English, Chinese, German)

**Naming:**
- Use official, canonical names when available
- Be consistent across all annotations
- Merge obvious duplicates (e.g., "MT" and "Machine Translation" → use "Machine Translation")

### Common Mistakes to Avoid

❌ **Don't include:**
- Information from related work sections (unless actually used in the paper)
- Background or motivation content (focus on what the paper does)
- Every model/dataset mentioned (focus on what's evaluated)
- Your own interpretations or assumptions

✅ **Do include:**
- Information from experiments and evaluation sections
- Main contributions and findings
- All metrics, criteria, and methods actually used
- Clear, specific terminology from the paper

---

## Annotation Workflow Summary

For each paper:

1. ✅ **Read the paper** (especially abstract, methodology, and evaluation sections)
2. ✅ **Answer Question 1**: Does it address NLG tasks?
   - **Verify the pre-filtered classification** - the paper was initially classified as "Yes"
   - Change to "No" if it doesn't actually address NLG tasks
   - **If No: Skip to the next paper** (we only annotate NLG papers)
   - If Yes: Continue to extract all metadata
3. ✅ **Extract Q1 metadata**: tasks, datasets, languages, models, outputs
4. ✅ **Answer Question 2**: Does it use automatic metrics?
   - If Yes: Extract and normalize metric names
5. ✅ **Answer Question 3**: Does it use LLMs as judges?
   - If Yes: Extract LLM models, methods, and criteria
6. ✅ **Answer Question 4**: Does it conduct human evaluation?
   - If Yes: Extract evaluation guidelines and criteria
7. ✅ **Review your annotations** for completeness and consistency
8. ✅ **Add any notes** about difficult decisions or uncertainties

---

## Example Annotation

**Paper**: "Evaluating Neural Abstractive Summarization with BLEU and Human Ratings"

**Question 1: Does the paper address NLG tasks?**
- Answer: **Yes**
- Tasks: ["Text Summarization"]
- Datasets: ["CNN/DailyMail", "XSum"]
- Languages: ["English"]
- Models: ["BART", "PEGASUS", "T5"]
- Outputs: ["News article summaries"]

**Question 2: Does it use automatic metrics?**
- Answer: **Yes**
- Automatic Metrics: ["BLEU", "ROUGE", "BERTScore"]
  - Note: Paper reported BLEU-1, BLEU-2, BLEU-4 → normalized to "BLEU"

**Question 3: Does it use LLMs as judges?**
- Answer: **No**

**Question 4: Does it conduct human evaluation?**
- Answer: **Yes**
- Methods: ["Likert scale rating", "5-point scale"]
- Criteria: ["informativeness", "fluency"]

---

## Questions or Issues?

If you encounter any problems during annotation:
- Document unclear cases in the notes section
- Flag papers that are ambiguous or difficult to categorize
- Ask for clarification on systematic issues

Good luck with your annotations!
