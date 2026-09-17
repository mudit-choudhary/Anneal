# Anneal

To anneal is to relieve the internal stresses a forming process leaves behind —
heat the material past recrystallisation, hold it, cool it slowly, and let a
strained structure settle back into a sound one.

Extracting text from a PDF is such a forming process. The document itself is
intact: columns sit where the author placed them, a figure interrupts the page
without interrupting the argument. Extraction flattens that two-dimensional
arrangement into a single stream and, in doing so, fractures it — paragraphs
severed at column boundaries, sentences broken across pages, words split by wrap
hyphens, captions welded into the middle of a thought. Nothing is missing. It is
the structure that is damaged, and every chunk cut from that stream inherits the
damage.

Anneal relieves it. A fine-tuned YOLOv11 model recovers twelve region types per
page, and a parser-agnostic assembler rebuilds reading order and rejoins prose
across columns, pages and interruptions. The clearest measure of the assembler's
worth is what it does to *someone else's* layout: run Docling's output through
it and Docling's own mid-sentence paragraph rate falls from **17.4% to 2.9%**
(5 papers, `evals/Reports/parser_quality.json`) — the gain belongs to the
assembler, not to any one detector. Chunking then grows along the document's own
grain instead of a character count: `grain_growth` nucleates at section headings
and expands until it meets a structural barrier, keeping tables and equations
whole. All of it runs on a 4 GB consumer GPU with no cloud dependency in the
parse or embed path.

Whether any of that helps is measured, not claimed.

## Results

**→ [evals/Reports/Report.md](evals/Reports/Report.md)** — two rounds over a
514-paper corpus, the second with metrics, tests and thresholds fixed in writing
**before** the run: 3 parsers × 3 chunkers, 400 questions, every comparison
paired and corrected for multiplicity.

- The fine-tuned YOLOv11 parser **retrieves significantly better** than Docling
  and PyMuPDF4LLM under every chunker tested, and loses the fewest answers in
  parsing (24 of 400, against 48 and 61).
- `grain_growth` beats a fixed-token window; against a stock recursive character
  splitter the difference is **not established** — it did not clear its own
  pre-set threshold, and the report says so rather than moving it.
- Faithfulness and context sufficiency **could not be measured reliably** by
  either method tried; both were demoted before any comparison was run.

## Quick start

```bash
python3.12 -m venv virtual_environments/annealenv                      # once
virtual_environments/annealenv/bin/pip install -r app/requirements.txt
virtual_environments/annealenv/bin/pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
ln -s "$PWD/app/bin/anneal" ~/.local/bin/anneal                        # once
anneal                                        # start everything, open the app
anneal status                                 # what is running, what is indexed
anneal stop                                   # stop everything
```

The first run builds the React app if needed and opens
<http://127.0.0.1:4002>. `anneal fresh-start --yes` purges and re-ingests every
PDF in `app/data/raw_pdfs/`. Full detail in [app/docs/OPERATIONS.md](app/docs/OPERATIONS.md).

The last install line matters: see the note at the top of `app/requirements.txt`.

> `npm run dev` serves the frontend **only**, with no backend behind it. Use
> `anneal`.

## Documentation

- [app/docs/OPERATIONS.md](app/docs/OPERATIONS.md) — install, run, ingest, rebuild, troubleshoot
- [app/docs/PARSING.md](app/docs/PARSING.md) — layout detection, the assembler, fine-tuning the model
- [app/docs/SYSTEM_WALKTHROUGH.md](app/docs/SYSTEM_WALKTHROUGH.md) — every module and store: what, why, when, what follows
- [app/docs/USER_GUIDE.md](app/docs/USER_GUIDE.md) — testing and tuning each RAG stage
- [app/docs/BUILD_LOG.md](app/docs/BUILD_LOG.md) — how the system got here: decisions, benchmarks, what was rejected
- [app/docs/PENDING_IMPROVEMENTS.md](app/docs/PENDING_IMPROVEMENTS.md) — known gaps, deliberately deferred
- [evals/README.md](evals/README.md) — how the evaluation was run and how to reproduce it
- [app/diagrams/system_architecture_detailed.drawio](app/diagrams/system_architecture_detailed.drawio) — multi-page: system + one page per module

## Layout

```
app/          the application: services, UI, scripts, docs, diagrams, bin/anneal,
              plus data/* models/* vector_db/* run/*          (* git-ignored)
evals/        the evaluation harness and its report
tests/        the test suite
virtual_environments/annealenv/                               (git-ignored)
```

Tests, from the repository root:

```bash
virtual_environments/annealenv/bin/pip install -r tests/requirements.txt
virtual_environments/annealenv/bin/python -m pytest tests/ -q
```
