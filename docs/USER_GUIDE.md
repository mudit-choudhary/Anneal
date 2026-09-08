# User Guide — Testing & Tuning the RAG Cycle

This guide walks the RAG cycle stage by stage so you can inspect the output
quality at each point, tune it, and only then move to the next stage. The
companion tool for everything here is:

```bash
python scripts/rag_inspect.py <stage> ...
```

Run it from anywhere with `globalragsetup_env` active. Stages: `parse`,
`tables`, `chunks`, `retrieve`, `answer`.

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
[parse_manager.md](parse_manager.md).

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

See [FRESH_START.md](FRESH_START.md). Re-embedding a single paper is
idempotent (`python scripts/register_pdfs.py --force Paper.pdf`).

## Suggested iteration loop

1. `parse` and `tables` on 3–5 representative papers → fix parsing knobs
   until the output reads clean.
2. `chunks` on the same papers → tune size until few mid-sentence cuts.
3. Re-ingest.
4. `retrieve` with ~10 questions you can verify.
5. `answer` the same questions; compare backends.
6. Only then trust day-to-day answers in the UI.
