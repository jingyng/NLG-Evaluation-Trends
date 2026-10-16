"""Compile the project-page dataset (data/corpus.json) from the final paper corpus.

Inputs (read-only):
  paper_code/data/llm-merged-results-top30-tasks/      3,334 final NLG papers (QCET-normalized)
  paper_code/data/llm-merged-results-normalized/       8,665 NLG papers before task filtering
  paper_code/data/laaj_human_validation_results_normalized/  433 LaaJ+human papers
  paper_code/07_figures/validation_analysis_data/validation_human_deduped.csv  (Fig. 7 values)
  papers/<VENUE-YEAR>/*_papers.json + merged/          titles and collected-paper counts

Run:  python project_page/build_data.py   (from nlg-eval-llm/)
"""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # nlg-eval-llm
PC = ROOT / "paper_code"
sys.path.insert(0, str(PC))
from data_loader import short_label  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "corpus.json"
VENUES = ["ACL", "EMNLP", "NAACL", "INLG"]
YEARS = list(range(2020, 2026))

# Display names for upper-cased normalized metric strings.
METRIC_NAMES = {
    "BLEU": "BLEU", "ROUGE": "ROUGE", "ACCURACY": "Accuracy", "BERTSCORE": "BERTScore",
    "F1": "F1", "METEOR": "METEOR", "RECALL": "Recall", "PERPLEXITY": "Perplexity",
    "EXACT MATCH": "Exact Match", "COMET": "COMET", "PRECISION": "Precision",
    "BLEURT": "BLEURT", "DISTINCT": "Distinct-n", "CIDER": "CIDEr", "CHRF": "chrF",
    "BARTSCORE": "BARTScore", "TER": "TER", "MOVERSCORE": "MoverScore",
    "PEARSON CORRELATION": "Pearson corr.", "SACREBLEU": "SacreBLEU", "CHRF++": "chrF++",
    "COSINE SIMILARITY": "Cosine similarity", "FACTCC": "FactCC", "SPICE": "SPICE",
    "MRR": "MRR", "SUCCESS RATE": "Success rate", "NDCG": "nDCG", "SELF BLEU": "Self-BLEU",
    "SPEARMAN CORRELATION": "Spearman corr.", "QUESTEVAL": "QuestEval", "F0": "F0.5",
    "SARI": "SARI", "WIN RATE": "Win rate", "AUROC": "AUROC", "UNIEVAL": "UniEval",
    "AUC": "AUC", "NIST": "NIST", "ERRANT": "ERRANT", "ENTROPY": "Entropy", "GLEU": "GLEU",
    "XCOMET": "xCOMET", "METRICX": "MetricX", "SUMMAC": "SummaC", "ALIGNSCORE": "AlignScore",
    "QAFACTEVAL": "QAFactEval", "USR": "USR", "FED": "FED", "GPTSCORE": "GPTScore",
    "G EVAL": "G-Eval", "EMBEDDING AVERAGE": "Embedding average", "MAUVE": "MAUVE",
    "ATTACK SUCCESS RATE": "Attack success rate", "HIT RATE": "Hit rate",
    "KENDALL TAU": "Kendall tau", "WER": "WER", "CER": "CER", "SELF-BLEU": "Self-BLEU",
}
ACRONYM = re.compile(r"^[A-Z0-9+\-]{2,6}$")


def metric_name(raw: str) -> str:
    if raw in METRIC_NAMES:
        return METRIC_NAMES[raw]
    return " ".join(w if ACRONYM.match(w) and len(w) <= 4 else w.capitalize() for w in raw.split())


def model_name(raw: str) -> str:
    """gpt_4o_mini -> GPT-4o-mini, llama_2_7b -> Llama-2-7b."""
    s = raw.replace("_", "-")
    for k, v in (("gpt-", "GPT-"), ("chatgpt", "ChatGPT"), ("llama", "Llama"), ("t-5", "T5"), ("bart", "BART"),
                 ("bert", "BERT"), ("claude", "Claude"), ("gemini", "Gemini"), ("qwen", "Qwen"), ("mistral", "Mistral"),
                 ("deepseek", "DeepSeek"), ("palm", "PaLM"), ("flan", "Flan"), ("transformer", "Transformer")):
        if s.lower().startswith(k):
            s = v + s[len(k):]
            break
    return s.replace("-t-5", "-T5").replace("seq-2seq", "seq2seq")


def yes(a) -> bool:
    return str((a or {}).get("answer", "")).strip().lower() == "yes"


class Vocab:
    def __init__(self):
        self.items, self.idx = [], {}

    def __call__(self, s):
        if s not in self.idx:
            self.idx[s] = len(self.items)
            self.items.append(s)
        return self.idx[s]


def short_authors(authors):
    if isinstance(authors, str):
        authors = re.findall(r"'([^']+)'", authors) or [authors]
    if not authors:
        return ""
    last = lambda n: n.split()[-1]
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{last(authors[0])} & {last(authors[1])}"
    return f"{last(authors[0])} et al."


def main():
    # --- metadata (titles) and collected counts per venue-year
    meta = {}
    for f in glob.glob(str(ROOT / "papers" / "*" / "*_papers.json")):
        for p in json.load(open(f)):
            meta[p["paper_id"]] = p
    collected = {d.name: len(list((d / "merged").glob("*.json")))
                 for d in (ROOT / "papers").iterdir() if (d / "merged").is_dir()}
    nlg = Counter(Path(f).parent.name for f in glob.glob(str(PC / "data/llm-merged-results-normalized/*/*.json")))

    # --- LaaJ-human validation flags
    validated = {}
    for f in glob.glob(str(PC / "data/laaj_human_validation_results_normalized/*/*.json")):
        d = json.load(open(f))
        validated[d["paper_id"]] = yes(d.get("explicit_validation"))

    # --- QCET dimension for each criterion (by full name)
    crit_l1 = {}
    with open(PC / "metadata_unique_counts/criteria_qcet_short_labels.csv") as fh:
        for r in csv.DictReader(fh):
            l1 = r["qcet_l1"].strip() or ("AUX" if r["qcet_id"].startswith("AUX") else "")
            crit_l1[r["qcet_name"].strip().lower()] = (r["short_label"].strip(), l1 or "AUX")

    V = {k: Vocab() for k in ("task", "metric", "crit", "model", "dataset", "lang", "gmodel", "method")}
    papers, n_laaj_yes = [], 0
    files = sorted(glob.glob(str(PC / "data/llm-merged-results-top30-tasks/*/*.json")))
    for f in files:
        d = json.load(open(f))
        vy = Path(f).parent.name
        venue, year = vy.split("-")[0], int(vy.split("-")[1])
        pid = d.get("paper_id") or Path(f).stem.removesuffix("_verified")
        m = meta.get(pid, {})
        a1, a2, a3, a4 = (d.get(f"answer_{i}") or {} for i in range(1, 5))
        n_laaj_yes += yes(a3)  # the paper's "679 LaaJ-using papers" counts Q3 = Yes
        uniq = lambda xs: list(dict.fromkeys(x.strip() for x in (xs or []) if x and x.strip()))
        # Same definitions as paper_code/07_figures/plot_evaluation_method_adoption.py (Fig. 1):
        # a method counts when its extracted list is non-empty.
        mets, lcr, hcr, lmo = (uniq(a2.get("automatic_metrics")), uniq(a3.get("criteria")),
                               uniq(a4.get("criteria")), uniq(a3.get("models")))
        flags = (1 if mets else 0) | (2 if (lcr or lmo) else 0) | (4 if hcr else 0)
        if validated.get(pid):
            flags |= 8
        papers.append([
            pid, VENUES.index(venue), year, m.get("title", ""), short_authors(m.get("authors", [])),
            [V["task"](t) for t in uniq(a1.get("tasks"))],
            [V["metric"](x) for x in mets],
            [V["crit"](x) for x in lcr],
            [V["crit"](x) for x in hcr],
            [V["model"](x) for x in lmo],
            flags,
            [V["dataset"](x) for x in uniq(a1.get("datasets"))],
            [V["lang"](x) for x in uniq(a1.get("languages"))],
            [V["gmodel"](x) for x in uniq(a1.get("models"))],
            list(dict.fromkeys(V["method"](x.lower()) for x in uniq(a3.get("methods")))),
        ])

    crits = []
    for c in V["crit"].items:
        lab, l1 = crit_l1.get(c.lower(), (c, "AUX"))
        crits.append([lab, l1, c])

    # --- Fig. 7 values (human-annotated 90-paper subset)
    vrows = []
    with open(PC / "07_figures/validation_analysis_data/validation_human_deduped.csv") as fh:
        for r in csv.DictReader(fh):
            q = r["criterion_norm"].strip()  # already the QCET name used in Fig. 7
            try:
                v = float(r["value_clean"])
            except ValueError:
                continue
            if not q:
                continue
            vrows.append([r["paper_id"], r["metric_norm"], short_label(q, prefixed=False), round(v, 3), r["LaaJ_Model"]])

    out = {
        "venues": VENUES, "years": YEARS,
        "funnel": {"collected": 14171, "nlg": sum(nlg.values()), "final": len(papers), "laaj": n_laaj_yes,
                   "dual": len(validated), "validated": sum(validated.values())},
        "byVenueYear": {vy: {"collected": collected.get(vy, 0), "nlg": nlg.get(vy, 0)} for vy in sorted(collected)},
        "tasks": V["task"].items,
        "metrics": [metric_name(x) for x in V["metric"].items],
        "crits": crits,
        "models": [model_name(x) for x in V["model"].items],
        "datasets": V["dataset"].items,
        "languages": V["lang"].items,
        "genModels": [model_name(x) for x in V["gmodel"].items],
        "methods": [x[:1].upper() + x[1:] for x in V["method"].items],
        "papers": papers,
        "validation": vrows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), {len(papers)} papers, "
          f"{len(V['metric'].items)} metrics, {len(crits)} criteria, {len(vrows)} validation values")
    sanity(out)


def sanity(out):
    """Print the paper's headline numbers so they can be checked against the text."""
    P = out["papers"]
    tidx = {t: i for i, t in enumerate(out["tasks"])}
    print("funnel", out["funnel"])
    for y in (2020, 2023, 2025):
        ps = [p for p in P if p[2] == y]
        r = lambda b: round(100 * sum(1 for p in ps if p[10] & b) / len(ps), 1)
        print(y, len(ps), "auto", r(1), "laaj", r(2), "human", r(4), "validated", r(8))
    bleu = out["metrics"].index("BLEU")
    mt = tidx["Machine Translation"]
    for single in (False, True):
        for y in (2020, 2025):
            ps = [p for p in P if p[2] == y and mt in p[5] and (not single or len(p[5]) == 1)]
            print("MT BLEU", "single" if single else "any", y, round(100 * sum(bleu in p[6] for p in ps) / len(ps), 1))
    top = Counter(out["crits"][c][0] for p in P for c in p[8]).most_common(8)
    print("top human criteria", top)


if __name__ == "__main__":
    main()
