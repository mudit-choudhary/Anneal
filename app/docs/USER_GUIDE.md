# Anneal — User Guide: testing and tuning the RAG cycle

> **Paths.** Code paths are relative to `app/`; the virtualenv and `tests/` sit
> at the repository root. Everything the app *writes* lives outside the
> repository, in the data home — `~/.local/share/anneal/`, or `$ANNEAL_HOME` if
> set — so `data/raw_pdfs/`, `vector_db/`, `models/`, `registry/` and `run/`
> below all mean `<data home>/…`.

This guide walks the RAG cycle stage by stage so you can inspect the output
quality at each point, tune it, and only then move to the next stage. The
companion tool for everything here is:

```bash
python scripts/rag_inspect.py <stage> ...
```

Run it from anywhere with `annealenv` active. Stages: `parse`,
`tables`, `chunks`, `retrieve`, `answer`.

## Is the whole thing working? — one command

```bash
python scripts/smoke_test.py            # every stage it can reach
python scripts/smoke_test.py -v         # also print sample output per stage
python scripts/smoke_test.py --offline  # stages 1-3 only, no services needed
```

It walks the entire pipeline and prints a pass/fail line per stage:

| # | Stage | Needs | Checks |
|---|---|---|---|
| 1 | parse | nothing | regions found, blocks produced, body text present |
| 2 | chunk | nothing | chunks produced, none over the 512-token window, mid-sentence cuts counted |
| 3 | embed | nothing | chunks embedded and retrievable |
| 4 | retrieval | embedding service | live store returns chunks, best distance is sane |
| 5 | answer | + Ollama | model answers, cites `[n]` |
| 6 | UI | UI service | all endpoints respond, React build is being served |
| 7 | query | UI + Ollama | full ndjson stream completes, chat persisted, memory works |

Stages 1–3 touch **no project state** — parsing writes to
`data/debug/smoke/` and embedding goes to a throwaway store in a temp
directory, so it is safe to run at any time, including mid-ingestion.
Stages 4–7 are skipped with a reason if the services are down. Exit code is
0 unless something actually failed.

Use it after any change, before a re-ingest, or when something feels wrong
and you want to know *which* stage broke. For tuning a specific stage, use
the per-stage commands below.

```
PDF ──stage 1──▶ tagged text ──stage 2──▶ chunks ──stage 3──▶ retrieved excerpts ──stage 4──▶ answer
     parsing              chunking              retrieval                   generation
```

A defect at any stage poisons every stage after it, so always fix the
earliest broken stage first.

---

## Stage 1 — Parsing quality

**No services needed.** Parses one PDF through YOLO layout detection and
paragraph assembly, then prints a quality summary:

```bash
python scripts/rag_inspect.py parse Some_Paper.pdf            # bare filename: looked up in data/raw_pdfs/
python scripts/rag_inspect.py parse Some_Paper.pdf --pages 3  # quick look at first 3 pages
python scripts/rag_inspect.py parse Some_Paper.pdf --show 20  # print more blocks
```

**What to check in the summary:**

| Signal | Healthy | Suspicious |
|---|---|---|
| Region detections | mostly `Text`, one `Title`, several `Section-header` | many `fallback` regions → YOLO missing content |
| Fallback Text regions | 0–2 per doc | high count → consider lowering `YOLO_CONF` (0.30 → 0.25) |
| Words swallowed | small (figure labels, running headers) | huge count → body text eaten by an oversized `Picture` box |
| Blocks of text | headings marked `##`, abstract one paragraph | paragraphs split mid-sentence, figure text in body, interleaved columns |

**Knobs** (`parse_manager/config.py`): `YOLO_CONF`, `RENDER_DPI`,
`FULL_WIDTH_FRACTION`, `SINGLE_COLUMN_FRACTION`, `MODEL_CANDIDATES`. See
[PARSING.md](PARSING.md).

Outputs: `data/parsed/<name>.json` (raw regions) and
`data/processed/<name>.txt` + `.json` (assembled blocks plus everything
that was dropped — check `dropped` if you suspect lost content).

### Stage 1b — Tables

```bash
python scripts/rag_inspect.py tables Some_Paper.pdf
```

For every `Table` region: saves a crop image to
`data/debug/tables/<paper>/table_p<page>_<n>.png` and checks that every
numeric token PyMuPDF sees inside the box survived into the extracted text.
`MISSING` lines name the lost numbers. Open the crop next to the printed
rows to judge column alignment by eye — that is what the number check
can't see.

### Trying Docling instead of YOLO

```bash
python scripts/docling_compare.py Paper_A.pdf Paper_B.pdf --docling-native
```

Runs both backends on the same PDFs and reports pages/s, peak VRAM, block
and table counts, and the first table's text from each; outputs land in
`data/debug/docling_compare/{yolo,docling}/`. Docling's TableFormer emits
tables as Markdown rows, which is the main thing to compare. To switch the
pipeline: `PARSER_BACKEND = "docling"` in `parse_manager/config.py`, then
a fresh start.

---

## Stage 2 — Chunking quality

**No services needed.** Previews exactly the chunks the embedder would
produce from a processed paper — without embedding anything:

```bash
python scripts/rag_inspect.py chunks Some_Paper               # bare stem: looked up in data/processed/
python scripts/rag_inspect.py chunks Some_Paper --target 1000 # experiment without touching config
python scripts/rag_inspect.py chunks Some_Paper --max 2400 --show 30
```

Chunking is **structure-aware** (`embedding_manager/chunking.py`): it reads
the typed blocks from `data/processed/<name>.json`, not the flat text.
Headings never form a chunk — the heading path (`Paper › 1. Introduction`)
is prefixed to every chunk and stored as metadata. Whole paragraphs are
packed up to `target` chars; a paragraph is split only if it alone exceeds
`max`, at sentence boundaries with one-sentence overlap. Captions and tables
are standalone chunks (a caption travels with its table), formulas are
packed inline with the prose explaining them, lists split at item
boundaries, and authors/footnotes are never embedded.

**What to check in the summary:**

- **Prose chunks ending mid-sentence** should be ~0 — a non-zero count is
  usually a stage-1 problem.
- **Chunks near/over the 512-token window** must be 0 — anything beyond is
  silently truncated by the embedder. Lower `--max` if not.
- **Sections covered** should match the paper's section count.
- **Body length distribution** — very small prose chunks (< ~150 chars)
  are usually stray fragments worth tracing back to the parse.
- **By block type** — tables should show up as `caption,table`.

**Knobs**: `CHUNK_TARGET_CHARS` / `CHUNK_MAX_CHARS` in
`embedding_manager/config.py` (defaults 1500 / 2000 — bge-base's 512-token
window holds ~2,600 chars of paper text).

> After changing chunking (or re-parsing papers), the vector store must be
> rebuilt — see "Re-ingesting" below.

---

## Stage 3 — Retrieval quality

**Needs the embedding service** with papers embedded:

```bash
python scripts/rag_inspect.py retrieve "How does AgentReuse evaluate request similarity?"
python scripts/rag_inspect.py retrieve "..." -k 12
python scripts/rag_inspect.py retrieve "..." --files Paper_A Paper_B
python scripts/rag_inspect.py retrieve "..." --chats        # also saved conversations
```

**How to read the output:** each hit shows its cosine distance
(0 = identical, ~1 = unrelated). Judge with questions you know the answers to:

- **Is the right paper in the top 3?** If not, the problem is upstream.
- **Distance cliff** — 0.42, 0.45, 0.48, **0.71** … means only the first
  three hits are real; lower "paper chunks per question" in Settings.
- **Same paper dominating all k slots** with near-duplicate chunks →
  consider more diverse retrieval.
- **Right content, wrong granularity** → revisit chunk size.

---

## Stage 4 — Answer quality

**Needs the embedding service + the answering model** (Ollama, or the
OpenAI-compatible API configured in Settings):

```bash
python scripts/rag_inspect.py answer "How does AgentReuse evaluate request similarity?"
python scripts/rag_inspect.py answer "..." --web              # add web pages to the excerpts
python scripts/rag_inspect.py answer "..." --backend openai   # override the configured backend
```

Sources are printed before the answer streams, so you can verify each `[n]`.

**What to check:** every claim cited? faithful to the excerpts? "The
excerpts do not contain…" when retrieval looked right → chunks too
fragmented. Compare backends on the same question to separate model limits
from retrieval problems.

**Knobs**: Settings page (retrieval counts, model, temperature, context),
and `SYSTEM_PROMPT` in `rag_setup/rag.py`.

---

## Re-ingesting after parser/chunking changes

```bash
scripts/fresh_start.sh          # purge + register PDFs + start all services
```

See [OPERATIONS.md](OPERATIONS.md). Re-embedding a single paper is
idempotent (`python scripts/register_pdfs.py --force Paper.pdf`).

## Suggested iteration loop

1. `parse` and `tables` on 3–5 representative papers → fix parsing knobs
   until the output reads clean.
2. `chunks` on the same papers → tune size until few mid-sentence cuts.
3. Re-ingest.
4. `retrieve` with ~10 questions you can verify.
5. `answer` the same questions; compare backends.
6. Only then trust day-to-day answers in the UI.

---

## Measuring retrieval quality against the original pipeline

`scripts/retrieval_eval.py` answers "did layout parsing and structure-aware
chunking actually improve retrieval?" with numbers instead of impressions.

```bash
python scripts/retrieval_eval.py --device cpu all --limit 12 --per-paper 3
python scripts/retrieval_eval.py --device cpu run --judge     # the slow, meaningful metric
```

### Three arms, so the changes are not confounded

| arm | text | chunking | embedder |
|---|---|---|---|
| `original` | PyMuPDF page dump | recursive 512/103 chars | all-MiniLM-L6-v2 |
| `rechunked` | PyMuPDF page dump | recursive 512/103 chars | bge-base-en-v1.5 |
| `current` | YOLO layout parse | structure-aware | bge-base-en-v1.5 |

`original` and `rechunked` are reconstructed from commit `c41e756`, so the
baseline is the code that actually ran, not an approximation of it.
`original → rechunked` isolates the embedding-model change;
`rechunked → current` isolates parsing plus chunking.

Everything is written to `data/eval/` and indexed into a throwaway vector
store at `data/eval/vector_db`. **The live corpus is never touched.**

### Ground truth without hand labelling

Questions are generated by the local model from the **raw PDF text**, never
from either pipeline's chunks, so neither arm is favoured by construction.
Each question carries a verbatim answer span, and is kept only if that span
survives in *every* arm's text. Questions whose span one arm lost are set
aside and counted separately — that is a coverage finding about the pipeline,
not a retrieval one, and scoring on it would measure the wrong thing.

### Which metric to believe

| metric | what it means | trust it? |
|---|---|---|
| paper recall@k | was the right paper retrieved | yes, comparable across arms |
| answer recall@k | verbatim span in the top *k* chunks | **no** — see below |
| answer recall by chars | verbatim span within an equal character budget | partly |
| context sufficient (`--judge`) | can the model answer from what was retrieved | yes, this is the point |
| chunk shape | fragment rate, page-marker noise | yes, no ground truth needed |

**Why `answer recall@k` is not comparable.** The baseline's 512-char chunks
with 103 chars of overlap give the same passage two or three chances to land
in a top-*k* list; the current arm's 1500-char chunks give it one. Scoring at
an equal *character* budget removes that artifact.

**Why even the verbatim metric misleads.** It rewards finding an exact string,
not the right passage. Observed in a real run: for "what plan reuse rate does
AgentReuse achieve", the current pipeline correctly retrieved section 6.3
(Result Analysis), but the generator had copied its span from the Abstract, so
it scored zero. The `--judge` metric exists because of this.

### Sample size

The report prints a 95% margin of error. At 24 questions it is about ±20
points, which is wider than most differences you will see. **A pilot run is
not a verdict.** For a real answer use the whole corpus:

```bash
python scripts/retrieval_eval.py --device cpu all --per-paper 3
python scripts/retrieval_eval.py --device cpu run --judge
```

Free the GPU first (`ollama stop`, stop the parser) — question generation and
judging both need the answering model, and embedding runs on CPU by default so
they do not compete.


---

## Comparing chunking strategies (shape, not retrieval)

`scripts/chunking_bench.py` answers a different question from
`scripts/retrieval_eval.py`. The eval harness measures *retrieval quality*
and needs a question set. This one measures the **shape** of what each
pipeline produces, which needs no ground truth, so it is fast and repeatable.

```bash
python scripts/chunking_bench.py --n 12
python scripts/chunking_bench.py --n 12 --pdf-dir /path/to/PDFs --seed 7
python scripts/chunking_bench.py --report
```

**Fully sandboxed.** It reads PDFs from an external directory and writes only
under `/tmp/chunking_bench` (override with `BENCH_DIR`). It never touches
`data/`, the registry or the vector store, and starts no services.

### Two parsers, one interface

| class | what it is |
|---|---|
| `LegacyParser` | commit `057ce0e9`: PyMuPDF page dump, then the regex paragraph builder |
| `CurrentParser` | today: YOLO layout detection, then typed blocks |

Both return the same two artefacts — the **parsed JSON** a pipeline would
store, and the **final chunks** — written to
`/tmp/chunking_bench/output/<parser>/`. The legacy paragraph rules are ported
verbatim, including the hardcoded running-header regex that matched one
specific paper; removing it would improve the baseline rather than reproduce it.

### Three strategies and their variations

| strategy | variations swept | unique knob |
|---|---|---|
| `recursive` | 512 and 1024 chars, 0 / 10 / 20% overlap | size and overlap |
| `semantic` | percentile, standard deviation, interquartile | **the breakpoint threshold type** |
| `structure` | target 1500 and 2000, heading prefix on/off | **the heading-path prefix** |

`recursive` and `semantic` consume flat text, so they run on both parsers.
`structure` consumes typed blocks; running it on legacy blocks (which carry no
types or headings) is itself part of the comparison.

### Metrics

Total pages, corpus size, chunk count, and mean / median / min / max chunk
size, plus four that say whether the chunks are any good:

| metric | meaning |
|---|---|
| `ov=0` | share of neighbouring chunk pairs sharing no text |
| `ov med` | median shared characters where they do overlap |
| `mid-start` | chunk begins mid-sentence |
| `mid-end` | chunk ends without terminal punctuation |

Overlap is **measured** from the text, not read off the configuration, so
strategies that never declare an overlap are still comparable.

---

## Catching a partly-parsed paper

A parse that dies half way still writes its partial output, still reports
`processed`, then `embedded`. The registry tracks which *stage* a paper
reached, never whether that stage covered the whole document, so nothing
notices. Two papers sat in the corpus like that for days holding 9% and 29%
of their content.

`common/coverage.py` is the check for it. A paper is flagged when the parse
reached fewer pages than the PDF has, when it produced almost no blocks, or
when its text is implausibly thin for its page count.

- **Automatically**, at the end of every `ops.py daily-ingest`, written to
  `run/logs/daily-ingest.log`.
- **In the app**, on the Ingestion tab. The card is quiet when everything is
  complete and turns red when it is not, listing each paper as
  "parsed 1 of 11 pages".
- **From a terminal**:
  ```bash
  python -c "import sys; sys.path.insert(0,'.'); from common import coverage; print(coverage.format_report(coverage.audit()))"
  ```

The **Re-ingest** button performs the repair by hand-verified steps: delete
the partial intermediates so the parser cannot skip them, then set the
registry row back to `downloaded` so the normal pipeline picks it up.
Re-embedding removes a filename's old chunks, so the vector store needs no
separate cleanup. The parser must be running for anything to happen, and the
button says so when it is not.

Papers parsed from this release on record `pdf_pages` in their own artefacts,
so coverage stays checkable even after the PDF has been pruned.
