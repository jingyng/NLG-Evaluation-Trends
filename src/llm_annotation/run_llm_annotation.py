from openai import OpenAI
import pandas as pd
import json
import os
import time
# DeepSeek API
client = OpenAI(base_url="https://api.deepseek.com", api_key="Your API Key")

def extract_eval(texts):
  
  system_prompt=f"""
  """
  user_prompt = f"""
  You are an expert NLP researcher with deep experience in Natural-Language Generation (NLG).

  TASK
  Read the paper provided below and answer the four numbered questions.
  Return **only** a single, valid JSON object (no markdown, no comments, no trailing commas).

  PAPER
  { " ".join(texts) }

  QUESTIONS

  1. Does the paper address NLG tasks?
  2. Does the paper use automatic metrics to evaluate the generated outputs?
  3. Does the paper use Large-Language Models (LLMs) as judges (i.e., *after* generation, an LLM is used to judge/assess the outputs)?
  4. Does the paper conduct *human* evaluations of the generated outputs?

  ANSWER FORMAT (strict)

  {{
    "answer_1": {{
      "answer": "Yes|No",
      "quote": "...",
      "tasks": ["Text Summarization", "Machine Translation", "Other:<task>"],
      "datasets": ["..."],
      "languages": ["English","Chinese","German","..."],
      "models": ["..."],
      "outputs": "..."
    }},
    "answer_2": {{
      "answer": "Yes|No",
      "quote": "...",
      "automatic_metrics": ["..."]
    }},
    "answer_3": {{
      "answer": "Yes|No",
      "quote": "...",
      "models": ["..."],                
      "methods": ["pairwise evaluation", "..."]
      "criteria": ["fluency","coherence","..."]
    }},
    "answer_4": {{
      "answer": "Yes|No",
      "quote": "...",
      "guideline": "...",
      "criteria": ["fluency", "coherence", "..."]
    }}
  }}

  INSTRUCTIONS & CONSTRAINTS

  * If the answer is "No", set all other fields in that section to an empty string ("") or an empty list ([]).

  * For **answer_1.tasks** choose one or more from:
    {{"Text Summarization","Dialogue Generation","Paraphrase Generation","Machine Translation","Image Captioning","Code Generation"}}.
    If none apply, use "Other:<task name>".

  * **Answer-2 guidance (automatic evaluation metrics)**
    * The **automatic_metrics** must be a list of automatic metrics used to evaluate the generated outputs.

  * **Answer-3 guidance (LLM as judge)**
    1. Answer **Yes** only if an LLM is used *after* generation to assess the outputs.
    2. **methods** – short name/description of the evaluation procedure or prompt.
    3. **criteria** – list the rubric properties the LLM is asked to score (e.g., "fluency","relevance","helpfulness").  If the prompt does not specify criteria, leave as an empty list [].

  * **Answer-4 guidance (human evaluation)**
    * The **quote** must mention humans, annotators, raters, a crowdsourcing platform, or a similar human-evaluation indicator, 
    * The **guideline** must mention questions or criteria for the evaluation.
    * The **criteria** must be explicitly mentioned in the human evaluation, list all criteria. If the paper does not specify criteria, leave as an empty list [].

  * The **quote** fields must be verbatim excerpts from the paper (use ellipses … to shorten if needed).
  * Use double quotes for all JSON strings; do **not** use backticks.
  * Do not add any keys, text, or formatting other than the JSON object.
  """

  response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
      {"role": "user", "content": user_prompt}
    ],
    response_format={
      'type': 'json_object'
    },
    temperature=1.0,
  )

  result=json.loads(response.choices[0].message.content)
  return result

start_time = time.time()  # Record the start time
max_duration = 240 * 60    # 60 minutes in seconds

for idx in range(0,600):
  elapsed = time.time() - start_time
  print(idx, elapsed)
  paper_id = "2025.acl-long."+str(idx+1)
  print(f"Processing paper: {paper_id}")
  with open("../papers/ACL_raw/" + paper_id + ".json", "r", encoding="utf-8") as f:
    data = json.load(f)
  texts = [sec.get("text", "") for sec in data.get("sections", [])]
  result= extract_eval(texts)
  with open("../papers/ACL-2025/"+paper_id+".json", "w") as file:
      json.dump(result, file, indent=4)
