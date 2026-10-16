# Project page: What Are We Measuring in NLG? A Meta-Analysis of Evaluation Trends 2020–2025 (EMNLP 2026)

Live at https://jingyng.github.io/NLG-Evaluation-Trends/

A static page: `index.html`, `data/corpus.json` (paper-level metadata for the 3,334 NLG papers)
and `static/images/`. D3 loads from cdnjs. Every interactive chart is computed in the browser
from `data/corpus.json`, using the definitions in the paper (likelihood ratio, Dunning's G²,
Benjamini–Hochberg-adjusted p ≤ 0.05).

| Section | Paper figure |
|---|---|
| Evaluation trends (all tasks / per task) | Figures 1 and 3 |
| Paper annotation pipeline | Figure 2, Table 1 |
| Metric inertia (bump chart, single-task papers) | Figure 4 |
| Metric–criteria mapping (split heatmap) | Figure 5 |
| Validation gap (LaaJ–human agreement, 70 papers) | Figure 6 |
| Papers, Corpus explorer | Interactive browsing and custom analyses of the corpus |

## Preview locally

```bash
cd docs && python -m http.server 8000   # open http://localhost:8000
```

Opening `index.html` directly from disk does not work because the page fetches `data/corpus.json`.

## Data

`build_data.py` documents how `data/corpus.json` was compiled from the extraction results
(QCET-normalized top-30-task papers, LaaJ–human validation results and the human-annotated
validation values). It expects the authors' working-tree layout, so its input paths may need
adjusting. Its sanity output should show 2025: automatic metrics 95.0%, LaaJ 46.4%,
human 43.1%, validated 15.8%; and MT papers using BLEU 88.6% (2020) → 61.8% (2025).

Link buttons (Paper, Code, arXiv, Data) are set in the `LINKS` object at the top of the
`<script>` in `index.html`; a `null` entry hides its button.
