# evals/ — the parser and chunking evaluation

Everything here produces one artefact: **[Reports/Report.md](Reports/Report.md)**, which
covers both rounds. Nothing is kept unless that report reads it or a script that
feeds it writes it.

| | Round 1 | Round 2 |
|---|---|---|
| Chunk shape | 5 parsers x 4 chunkers, 15 papers | 3 parsers x 3 chunkers, 514 papers |
| Retrieval | 12 cells, 100 questions, 103 papers | 9 cells, **400 questions**, 514 papers |
| Questions written from | each parser's own output | neutral `pdftotext`, which no parser under test uses |
| Answers judged by | the answering model, judging itself | an entailment classifier and Gemma 3 4B, both local and validated |
| Settled | semantic chunking is worst; fixed-k scoring flatters big chunks | Grain-Growth beats `fixed_token`, not `recursive_char`; parsers ranked |

Round 2 answers questions round 1 could not, because its questions favour no
parser and its answer scores come from something other than the model being
scored. What it could **not** measure is stated in the report: faithfulness and
context sufficiency failed their checks under both methods.

## How a claim gets made

1. **Pre-registration.** `Reports/round2_preregistration.json` fixes the primary
   metrics, their priority order, the tests, the thresholds and the verdict rules,
   with a timestamp and the git commit, **before** the run. It records the
   SHA-256 of the frozen question set; the run refuses to start if `dataset.json`
   no longer matches.
2. **Amendments.** Every later change is appended to
   `Reports/round2_amendments.json` with the date, what results existed at the
   time, and who decided. Four were needed; two demoted metrics that failed
   their checks.
3. **Analysis.** `scripts/round2_analysis.py` applies the pre-registered rules to
   the per-question records and writes `Reports/round2_analysis.json`. No verdict
   is written by hand.
4. **Report.** `scripts/make_round2_report.py` renders that file. Round-1 inputs
   are named explicitly, because round 2 overwrote the shared paths
   (`questions/dataset.json`, `corpus/manifest.json`).

## Layout

```
evals/
  corpus/
    manifest.json                 the 514 round-2 papers (95 arXiv + 419 never
                                  used in YOLO training), with pages, columns,
                                  tables, SHA-256
    manifest_103.json             the round-1 corpus, kept so round 1 stays readable
    unused_training_pdfs.json     PDFs in the YOLO folder that no labelled round used
    excluded_ids.json             arXiv ids of every training PDF
    *.pdf                         cached or symlinked (git-ignored)
  text/                           pdftotext output, the neutral question source (git-ignored)
  parsed/<parser>/<stem>.json     one parse per parser per paper (git-ignored)
  parsed/<parser>/_stats.json     cumulative timing and VRAM; _stats_round1.json is the round-1 split
  questions/
    dataset.json                  the frozen 400, read-only, hashed in the pre-registration
    candidates.json               all 1,016 generated, with reject reasons
    round1/                       round 1's 100-question set
  scripts/                        see below
  Reports/
    Report.md                     the report
    round2_preregistration.json   the plan, written before the run
    round2_amendments.json        every change since, with evidence
    round2_analysis.json          tests, effect sizes, intervals and verdicts
    rag_results_round2.json       per-cell aggregates
    rag_round2_rows/              per question: retrieval scores, the 10 retrieved chunks, the answer
    rag_round2_scores/            per question: string, NLI and judge scores
    shape_round2.json             chunk shape on the same 514 papers
    rag_results.json, results.json, parser_quality.json   round 1
    assets/*.svg                  figures for both rounds
```

## Running it

```bash
bash evals/scripts/round2_resume.sh     # parse, questions, 9 retrieval cells (~30 h)
python evals/scripts/score_answers.py --stage string
python evals/scripts/score_answers.py --stage nli --device cuda --batch 8
python evals/scripts/score_answers.py --stage gemma            # ~12 h on a 4 GB card
python evals/scripts/shape_round2.py
python evals/scripts/round2_analysis.py
python evals/scripts/make_round2_figures.py
python evals/scripts/make_round2_report.py --final
```

`round2_resume.sh` is also the **restart** command: cached parses, papers that
already have questions, finished cells and answered questions are all skipped,
and every file is written through a temporary file, so a crash cannot leave a
half-written one. A machine failure during round 2 cost two hours, not the run.

**Status at any time:** `python evals/scripts/eval_status.py` shows finished
cells, the cell in flight, an ETA, and scoring progress. It only reads.

## Scripts

| Script | Does |
|---|---|
| `build_corpus.py` | arXiv download under the published rate limit, or `--offline` to manifest what is cached; `--max-pages` caps paper length |
| `unused_training_pdfs.py` | finds PDFs the YOLO model never trained on |
| `parse_cache.py` | parses each paper once per parser into `parsed/` |
| `build_questions.py` | generates, verifies and shortlists questions; two text engines must agree the answer is in the paper |
| `run_rag_eval.py` | the 9 retrieval cells, each in its own vector collection |
| `score_answers.py` | offline scoring: string metrics, NLI, Gemma judge |
| `round2_stats.py` | paired tests, Holm, bootstrap by paper, the verdict rules |
| `round2_analysis.py` | applies the pre-registration, writes `round2_analysis.json` |
| `shape_round2.py` | chunk shape on the round-2 corpus |
| `make_round2_figures.py`, `make_round2_report.py` | figures and the report |
| `run_matrix.py`, `make_figures.py`, `make_main_report.py`, `parser_quality_probe.py` | round 1 |

## Things worth knowing

- **The production store is never touched.** Retrieval runs against
  `evals/vector_db_eval`, set before any embedding code loads, with one
  collection per cell, dropped between cells and deleted at the end.
- **Every span metric is computed twice**, exact and near match (90% of the
  span's words). Exact matching penalises a parser whose text engine differs
  from the one the answers were verified against; verdicts use the near match.
- **Two of the three parsers read text through PyMuPDF**, so answers were
  verified against `pdftotext` (poppler) as well, and questions were written
  from it.
- **The corpus is disjoint from YOLO training.** arXiv ids are read from each
  training PDF's page-1 stamp and excluded; the 419 folder papers appear in no
  round's labels.
- **Downloads honour the [arXiv Terms of Use](https://info.arxiv.org/help/api/tou.html)**:
  single connection, no parallelism, and fifteen seconds between requests rather
  than the published three-second floor after arXiv returned 429 and 503.
- **Mistakes found and fixed while running** are in the report and the
  amendments: an NLI metric that scored a cell higher for having its context cut
  into more windows, a judge that gave one verdict three times, a fact set the
  reference answers did not support, and an effect-size function that called a
  perfectly consistent gain "no effect".

## Related

- `scripts/chunking_bench.py` — the earlier, narrower chunking benchmark this
  directory supersedes. Still referenced from `docs/USER_GUIDE.md`.
- `scripts/retrieval_eval.py` — the earlier single-configuration retrieval check
  against the live store; `evals/scripts/run_rag_eval.py` supersedes it.
