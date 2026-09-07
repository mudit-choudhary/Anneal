# User Guide — Testing & Tuning the RAG Cycle

This guide walks the RAG cycle stage by stage so you can inspect the output
quality at each point, tune it, and only then move to the next stage. The
companion tool for everything here is:

```bash
python scripts/rag_inspect.py <stage> ...
```

Run it from anywhere with `globalragsetup_env` active. Stages: `parse`,
`chunks`, `retrieve`, `answer`.

```
PDF ──stage 1──▶ tagged text ──stage 2──▶ chunks ──stage 3──▶ retrieved chunks ──stage 4──▶ answer
     parsing              chunking              retrieval                 generation
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
| Region detections | mostly `Text`, one `Title`, several `Section-header` | many `fallback` regions → YOLO missing content (raise `--pages`? check `YOLO_CONF`) |
| Fallback Text regions | 0–2 per doc | high count → model misses; consider lowering `YOLO_CONF` (0.30 → 0.25) |
| Words swallowed | small (figure labels, running headers) | huge count → body text being eaten by an oversized `Picture` box |
| Blocks of text | headings marked `##`, abstract one paragraph | paragraphs split mid-sentence, figure text in body, interleaved columns |

**Knobs** (in `parse_manager/config.py`): `YOLO_CONF`, `RENDER_DPI`,
`FULL_WIDTH_FRACTION`, `SINGLE_COLUMN_FRACTION`. See
[parse_manager.md](parse_manager.md) for what each does.

Full outputs land in `data/parsed/<name>.json` (raw regions) and
`data/processed/<name>.txt` + `.json` (assembled blocks, plus everything that
was dropped — check `dropped` in the JSON if you suspect lost content).

---

## Stage 2 — Chunking quality

**No services needed.** Previews exactly how the embedder would split a
processed file — without embedding anything:

```bash
python scripts/rag_inspect.py chunks Some_Paper                  # bare stem: looked up in data/processed/
python scripts/rag_inspect.py chunks Some_Paper --chunk-size 768 # experiment without touching config
python scripts/rag_inspect.py chunks Some_Paper --overlap 0.1 --show 30
```

**What to check:**

- **Mid-sentence endings** — the summary counts chunks ending mid-word or
  mid-sentence. High counts mean the splitter is cutting inside paragraphs;
  a paragraph that fits whole in a chunk embeds much better.
- **Orphaned headings** — a chunk that ends with `## Method` puts the heading
  in one chunk and its content in the next; retrieval for "method" then
  fetches the useless one.
- **Tiny chunks** (< ~100 chars) — usually a lone heading or `[CAPTION]`;
  they embed poorly and waste retrieval slots.
- **Tag noise** — `[TABLE]` rows split across chunks lose meaning.

**Knobs**: `CHUNK_SIZE` and `CHUNK_OVERLAP_PCT` in
`embedding_manager/config.py` (defaults 512 / 0.2). Experiment via the CLI
flags first; when you settle on values, write them into the config —
`chunk_and_embed` uses the same constants.

> After changing chunking (or re-parsing papers), the vector store must be
> rebuilt — see "Re-ingesting" below.

---

## Stage 3 — Retrieval quality

**Needs the embedding service** (`cd embedding_manager && python main.py`)
with papers already embedded:

```bash
python scripts/rag_inspect.py retrieve "How does AgentReuse evaluate request similarity?"
python scripts/rag_inspect.py retrieve "..." -k 12
python scripts/rag_inspect.py retrieve "..." --files Paper_A.txt Paper_B.txt
```

**How to read the output:** each hit shows its cosine distance
(0 = identical, ~1 = unrelated). Judge with a handful of questions you know
the answers to:

- **Is the right paper in the top 3?** If not, the problem is upstream:
  noisy parse or bad chunk boundaries — go back a stage.
- **Distance cliff** — a jump like 0.42, 0.45, 0.48, **0.71** … means only
  the first three hits are real; consider whether `N_RESULTS` (in
  `rag_setup/config.py`) should be lowered so junk never reaches the LLM.
- **Same paper dominating all k slots** with near-duplicate chunks →
  overlap too high, or you may want more diverse retrieval.
- **Right content, wrong granularity** (a caption instead of the explaining
  paragraph) → revisit chunk size.

---

## Stage 4 — Answer quality

**Needs the embedding service + Ollama** (daemon runs automatically after
install; the model loads on first use):

```bash
python scripts/rag_inspect.py answer "How does AgentReuse evaluate request similarity?"
python scripts/rag_inspect.py answer "..." --backend gemini   # quality comparison
```

The sources are printed before the answer streams, so you can verify each
`[n]` citation against what was actually retrieved.

**What to check:**

- **Every claim cited?** Uncited claims from a 4B model deserve suspicion.
- **Faithful to the chunks?** Spot-check a citation: does chunk `[2]` really
  say that?
- **"The excerpts do not contain..."** answers — if retrieval (stage 3)
  looked good for the same question, the chunks may be too fragmented for
  the model to connect; consider larger chunks.
- **Local vs Gemini** — run both backends on the same question. If Gemini
  answers well from the same chunks and Qwen doesn't, it's a model limit
  (acceptable for lookups, use `--backend gemini` for synthesis). If both
  fail, the problem is retrieval or parsing, not the LLM.

**Knobs**: `N_RESULTS`, `OLLAMA_MODEL`, `temperature` (in
`rag_setup/rag.py`'s `_local_payload`), and the `SYSTEM_PROMPT` itself.

The same stage-4 experience is available in the web UI
(`cd UI && python main.py` → http://127.0.0.1:4002) with clickable
citations.

---

## Re-ingesting after parser/chunking changes

Embeddings are snapshots of whatever the parser + chunker produced at embed
time. After improving either, rebuild:

```bash
# 1. Stop the embedding service. Then:
rm -rf vector_db/
# 2. Reset registry statuses so the pipeline re-runs (with registry service stopped):
sqlite3 registry_manager/rag_registry.db \
  "UPDATE file_status_table SET status='downloaded', parsed_at=NULL, embedded_at=NULL;"
# 3. Clear stale intermediate outputs:
rm -f data/parsed/* data/processed/*
# 4. Start registry, parse_manager, and embedding_manager and let them run
#    (overnight for a large corpus — GPU is used by YOLO + the embedder).
```

## Suggested iteration loop

1. `parse` 3–5 representative papers → fix parsing knobs until block output
   reads clean.
2. `chunks` on the same papers → tune `CHUNK_SIZE`/overlap until few
   mid-sentence cuts.
3. Re-ingest the corpus overnight.
4. `retrieve` with ~10 questions you can verify → confirm the right chunks
   surface in the top k.
5. `answer` the same questions → judge grounding; compare `--backend gemini`.
6. Only then trust day-to-day answers in the UI.
