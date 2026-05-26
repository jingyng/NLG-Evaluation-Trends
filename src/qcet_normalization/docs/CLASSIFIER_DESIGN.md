# Classifier design: conditional task context

Per the decision "use tasks only when needed", the classifier runs a two-pass
routine for each raw criterion variant.

## Pass 1 — task-blind

Inputs:
- raw criterion string
- a small sample of co-occurring criteria from papers that used this raw
  string (default: up to 10 other criterion strings, deduped)
- the full extended QCET lattice + aux taxonomy (definitions + tree
  placements + disambiguation rules)

Returned fields (JSON):
- `target_id` : a QCET leaf id (`QXX-y-n`) or an aux id (`AUX-X`)
- `confidence` : `high` | `medium` | `low`
- `is_ambiguous` : bool — true if more than one target could plausibly fit
- `alt_candidates` : list of other plausible target ids
- `rationale` : one-sentence paraphrase
- `sub_reason` : only for AUX-Other (`not_a_criterion`, `metric_name`,
  `judgement_bias`, `unclassifiable`)

Accept Pass-1 output as final IF:
- `confidence == "high"` AND `is_ambiguous == false`

Otherwise, escalate to Pass 2.

## Pass 2 — task-informed (escalation only)

Triggered when Pass 1 returns `low`/`medium` confidence OR
`is_ambiguous == true`.

Inputs are the same as Pass 1, PLUS:
- `paper_tasks` : a small set (up to 5) of tasks reported by papers that used
  this raw criterion string (from the `tasks` field in each paper's
  `answer_1`). Deduped and presented as a list.

Explicit instruction to the model: "Use task information only to break ties
between equally-plausible target ids from Pass 1. Do not re-route variants that
Pass 1 classified with high confidence."

Returned fields: same as Pass 1 plus a `task_context_was_used` boolean.

## Audit outputs

For each variant we store both Pass-1 and (if triggered) Pass-2 outputs, so we
can later answer:
- What fraction of variants required task context?
- Among escalated variants, how often did the Pass-2 label differ from Pass-1?
  (High disagreement => task context genuinely disambiguated; low
  disagreement => task context only confirmed.)
- For any downstream paper claim about task x criterion associations, is that
  claim supported by task-blind classification? (Robustness check.)

## Reporting in paper

> "Criteria were classified into the extended QCET taxonomy in two passes: a
> task-blind pass for all variants, followed by a task-informed pass triggered
> only for variants Pass 1 flagged as low-confidence or ambiguous. Among N
> unique variants, K (K/N = X%) required task context for disambiguation. For
> findings that report associations between tasks and criteria, we repeated
> the classification with task information hidden and confirmed that headline
> claims survive (see Appendix B.Y for the per-finding robustness table)."
