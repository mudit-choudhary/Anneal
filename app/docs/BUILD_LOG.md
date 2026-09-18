# Anneal — Build Log: start to finish

A chronological account of how this repository went from a working-but-poor
RAG pipeline to the current system: what was built, what was measured, what
was deliberately rejected, and what is still open.

Companion documents: [SYSTEM_WALKTHROUGH.md](SYSTEM_WALKTHROUGH.md) (the full account),
[SYSTEM_WALKTHROUGH.md](SYSTEM_WALKTHROUGH.md) (per-module reference),
[PENDING_IMPROVEMENTS.md](PENDING_IMPROVEMENTS.md) (open items in detail).

| | |
|---|---|
| Papers searchable | 48 |
| Parse throughput | 10–14 pages/s |
| Peak VRAM | 883 MB |
| Automated tests | 161 passing, smoke test 7/7 |

---

## 1. The chunks were the problem

The system already worked end to end: download, extract text, cut into
pieces, embed, retrieve. Answers came back; they were just not good. The
reason was visible the moment we printed the *chunks* instead of the answers.

Spacing-heuristic text extraction cannot answer four questions that decide
chunk quality:

- Are these two pieces of text the same paragraph?
- Is this body prose, or a label drawn *inside* a figure?
- When a paragraph runs onto the next page, is the first line a continuation
  or a page header?
- In a two-column paper, which column does this belong to, and in what order
  should the columns be read?

Get those wrong and every downstream stage inherits the damage. A chunk that
splices a running header into the middle of a sentence embeds to a vector
that means nothing in particular, and no amount of prompt work at the far end
recovers it. So the work started at the parser, not the model.

## 2. Teaching the parser to see a page

Rebuilt around a fine-tuned document-layout detection model. Every page is
rendered to an image, the model draws boxes, and each box carries one of
twelve labels: Title, Authors, Section-header, Text, List-item, Table,
Picture, Caption, Formula, Footnote, Page-header, Page-footer.

Those boxes govern the text. Words are extracted with coordinates and
assigned to the region containing them; regions are ordered by reading order
rather than PDF storage order. Three labels are **swallowed** — their words
never reach the body flow at all: `Picture`, `Page-header`, `Page-footer`
(`SWALLOW_LABELS` in [parse_manager/config.py](../parse_manager/config.py)).
That single rule fixed figure-internal text leaking into paragraphs, which
was the original motivation for the whole approach.

### The column splitter, and why it is so reluctant

Occasionally the model returns one wide box where a human sees two columns —
a three-across author block, typically. Splitting is dangerous: applied to a
table it separates row labels from their numbers, which happened once during
the Docling benchmark and produced a table consisting only of its left column.

The splitter now runs behind four gates:

1. the label is on a prose-only allowlist (`SPLITTABLE_LABELS`);
2. the region is at least half the page wide — a normal body column in a
   two-column paper cannot be;
3. a gutter crosses the *whole* region with words on both sides across
   **3 or more** rows (two align by coincidence too easily);
4. every resulting column is at least **15%** of the region width — a
   pseudocode line-number margin is ~3%.

Measured over the whole corpus (46 PDFs × 8 pages, 4,430 regions with words):

| Stage | Regions |
|---|---|
| have words | 4,430 |
| blocked: label not splittable | 1,157 |
| blocked: narrower than half the page | 1,996 |
| passed both gates | 1,277 |
| of those, no qualifying gutter | 1,276 |
| **actually split** | **1** |

Zero splits in 45 of 46 papers, once in the paper with the three-across
author block. That is the behaviour you want from a repair that can do harm.

**Known residual:** a definition list (term | meaning) labelled `Text` rather
than `Table` would still be split. Not observed in this corpus; recorded
rather than papered over.

## 3. Chunking that follows the structure

Because the parser emits typed blocks rather than a wall of text, chunking
respects the document instead of counting characters. Whole paragraphs are
packed up to `CHUNK_TARGET_CHARS` (1,500); a single block is only cut apart
beyond `CHUNK_MAX_CHARS` (2,000). Chunks never straddle a section boundary.

Each chunk carries its heading path as a prefix, so a passage from deep
inside a paper reaches the embedding model already knowing it belongs to
*Methods → Graph construction*. That context is what lets retrieval separate
two paragraphs using identical vocabulary for different purposes.

Embeddings come from `bge-base-en-v1.5` (768-d, 512-token window) into
ChromaDB with cosine similarity. The chunk sizes were chosen against that
token window, not picked round.

## 4. Answering without the cloud

Answering runs on a locally hosted model via Ollama. The choice that mattered
was the exact variant: the hybrid reasoning build ignores the instruction to
stop thinking out loud and emits its scratchpad into the answer; the
instruct-only build honours it. Any OpenAI-compatible endpoint can be
configured instead.

Conversation memory was added at the **retrieval** end, not the generation
end. A follow-up like "and what about the second one?" is meaningless as a
search query, so the previous user turn is prepended before the vector search
runs (`search_text` in [rag_setup/rag.py](../rag_setup/rag.py)). History fed
to the model is capped at 4 turns / 1,200 characters, because context window
is the scarcest resource on this machine.

Answers cite sources by number, clickable back to the retrieved passage. Web
search is per-question and its results go straight to the answering model
without being chunked or embedded — the vector store stays a record of
papers, not of pages visited.

## 5. Getting out of the terminal

A React application replaced the command line. Chats persist with dates,
renameable titles, exchange counts, and the response time recorded when the
answer finished so it survives closing the window. Answers render Markdown
and Mermaid. Papers can be filtered before a question runs, grouped by the
date they were added.

Chats on the left, conversation in the middle, paper filter on the right.
Light and dark themes. Logs moved to their own tab and stream live.

## 6. Cutting the AGPL dependency

The layout model ran through an AGPL library. That licence carries a network
clause: software using it and reachable over a network must offer its whole
source to users. Irrelevant for private use; expensive to leave in place if
distribution ever happens.

Inference was rewritten directly on ONNX Runtime (MIT), reimplementing what
the library did invisibly: letterboxing to the model input size, DFL box
decoding, and per-class non-maximum suppression.

The bar was behavioural identity, not "close enough": across 3 papers /
30 pages, **407 of 407 regions matched** exactly. Three corrections were
needed, each looking like nothing and each moving detections:

1. **Resizing filter.** Pillow's bilinear resize antialiases; OpenCV's does
   not. Confidence scores shifted by up to **0.35**.
2. **Padding geometry.** Padding to a full square instead of the next
   multiple of the model stride changes the scale factor, and therefore every
   box coordinate.
3. **Border rounding.** The inverse transform used the float border offset
   where the forward transform used an integer one.

**The trap worth remembering.** Enabling the GPU meant adding the CUDA
libraries shipped inside the Python packages to the loader path. Loading that
directory wholesale also loads `libnvblas`, which hijacks CPU BLAS
**process-wide** — unrelated CPU code silently routes through the GPU. The
fix is `_CUDA_LIB_PREFIXES`, an allowlist of the six library prefixes
actually needed. `scripts/smoke_test.py` is what surfaced it.

**Performance detail.** CUDA re-tunes kernels whenever the input shape
changes, and the last batch of a page run is usually short. Padding that tail
batch to a fixed size took throughput from **4.2 to 17 pages/s**
(`predict(fixed_batch=…)`). End to end the pipeline runs at 10–14 pages/s,
unchanged by the migration.

## 7. Three evaluations that ended in "no"

Recording *why* is the point; otherwise these get proposed again in six months.

### Replacing PyMuPDF with pypdfium2

Benchmarked rather than assumed (`scripts/pdf_backend_compare.py`, two
papers / 20 pages):

| Metric | PyMuPDF | pypdfium2 |
|---|---|---|
| Rendering | 105 / 32 pages/s | 55 / 33 pages/s (**1.18× slower**) |
| Word extraction | baseline | **1.8× slower** |
| Word boxes | baseline | mean IoU **0.89** |
| **Final processed text** | baseline | **87.2% similar** |

Three root causes, each needing compensation code:

1. **No word API** — pdfium exposes characters; words must be rebuilt.
2. **Box semantics differ** — default boxes are the glyph *ink* extent, so
   height varies by letter, where PyMuPDF returns the font line height. The
   pipeline derives *both* the line-grouping tolerance and the column-gutter
   threshold from median word height, so this silently retunes both.
3. **Hyphenated breaks merge** — one word whose box spans two lines, which
   breaks line clustering outright.

**Verdict:** keep PyMuPDF. If distribution happens, the Artifex commercial
licence is cheaper and carries zero code risk.

### Replacing the parser with Docling

Docling brings TableFormer, real table structure — the one thing this
pipeline genuinely lacks. Integrated behind a backend switch for a like-for-
like comparison. The first measurement said 30× slower and **was wrong,
twice over**: model load was being counted as parse time, and CPU OCR ran by
default on born-digital PDFs that already carry their text layer.

Corrected (both backends warmed, two papers × 10 pages, complete pipelines):

| | PyMuPDF + YOLO | Docling |
|---|---|---|
| Throughput | **13.1 pages/s** | 2.5 pages/s |
| Peak VRAM | **883 MB** | 913–1,043 MB |
| Tables | word rows, no structure | **Markdown, aligned cells** |

**Verdict:** a default, not a deletion. YOLO stays — a 46-paper ingest is
minutes either way. Docling stays available for a table-heavy corpus.

### A vision model over figures and tables

Figures contribute only their captions. A VLM pass over `Picture` and `Table`
regions would fix that, and was shelved deliberately: on 4 GB it competes for
the exact memory the rest of the pipeline needs. Written up with the concrete
failure case in [PENDING_IMPROVEMENTS.md](PENDING_IMPROVEMENTS.md#1-tables-and-figures-lose-structuremeaning-in-the-parsed-output).

## 8. What the testing actually caught

`scripts/smoke_test.py` runs all seven stages against a throwaway store, so
parsing through to the web layer can be exercised without touching real data.
It and the 161 unit tests found a specific class of bug: **things failing
silently while reporting success.**

| Defect | What was wrong |
|---|---|
| Fallback that never fired | The layout model's GPU→CPU fallback tested whether CPU was *in the provider list*. It always is. The fallback never triggered; OOM propagated as a hard failure. Now checks the provider actually in use. |
| Embedder with no fallback | The embedding model had no CPU path at all. With the LLM holding VRAM, two papers failed and were marked `error`. Fallback added at load time and per call. |
| Silent device downgrade | Restarting the embedder on GPU reported success even when it landed on CPU, because the call returned as soon as the process spawned. It now waits for the model to load, polls `/v1/health` for the real device, and warns on mismatch. |
| Fabricated time estimate | The ETA averaged stage durations. One paper had sat in the queue 28 hours; that outlier pushed the mean to a predicted 74 min/parse — for two files, with no parser running. Medians and measured throughput replaced it. |
| Misleading disk counts | 96 processed files for 48 papers, unexplained. Each paper writes two: a readable `.txt` and the `.json` blocks the embedder consumes. Now stated, and pipeline *status* is distinguished from files awaiting the pruner. |
| A number with no unit | "Sweep every: 1800". Now reads *1800 seconds (30 min)*. |

## 9. One screen for the whole pipeline

Six services cooperate through a shared `data/` directory and a registry
tracking each paper's lifecycle. All are startable, stoppable and restartable
from the app, plus two presets: **set up for questions** and **set up for
adding papers**.

That distinction exists because of the hardware. The three models total
roughly 5.5 GB; the card holds 4 GB. So the pipeline time-shares: while
ingesting, the layout model and embedder hold the GPU; while answering, the
LLM does and the embedder moves to CPU, fast enough for one query at a time.

A GPU tab makes it visible — total VRAM, a bar segmented by process, a live
trace, and a row per model showing where it actually ended up. The LLM splits
across GPU and CPU **by layers**. The layout model splits **per operator**.
The embedder **cannot** — all or nothing, which is exactly why its device had
to become honest about itself.

## 10. The 22 questions, settled

Twenty-two questions came out of reading the architecture documents. They
drove most of the work above. Nineteen are done; three were answered with a
deliberate *no*; one is genuinely unfinished.

| # | Question | Status | Where it landed |
|---|---|---|---|
| | **Data management and storage** | | |
| 1 | Raw PDF pruning | done | Archive/delete policies with newest-N or oldest-N kept back. Default leaves PDFs untouched — deleting them is the only irreversible action in the system. |
| 2 | Postgres + pgvector | declined | Measured first: the vector store is ~11 MB across ~3,700 chunks. SQLite + Chroma are right-sized for one machine. Revisit at multiple users or a second writer. |
| 3 | Hugging Face cache | answered | Keep it. The embedder loads from it on every start; deleting costs a 440 MB re-download and fails offline. |
| | **Ingestion and processing** | | |
| 4 | Config and scheduling | done | `--url`, `--domain`, `--max`, `--once`, and a **systemd user timer** (not cron) — `Persistent=true` + `OnBootSec=5min` covers the missed-run requirement. Also driveable from the app. |
| 5 | Layout model candidates | done | Three in `MODEL_CANDIDATES`, first present wins: fine-tuned small, fine-tuned nano, pretrained nano. |
| 6 | Column detection | done | Reworked, then audited across the corpus. 1 split in 46 papers. |
| 7 | Docling | done | Wired as an alternative backend and benchmarked. ~5× slower, so not the default; one env var away. |
| 8 | Images and tables | shelved | Your call. Mermaid was rejected on merits anyway: fine for simple block diagrams, unreliable for plots, and it embeds poorly. Prose description remains the candidate. |
| 9 | Table verification | done | `rag_inspect.py tables` — crops each table from the PDF and checks extracted numbers against it. |
| | **Hardware and performance** | | |
| 10 | Concurrent VRAM cost | answered | ~5.5 GB for all three against a 4 GB card; hence time-sharing. Now live in the GPU tab. |
| 11 | Hardware-aware setup | **partial** | `scripts/hardware_check.py` reports VRAM and recommends models per tier, but is **CLI-only — never surfaced in the app**. The one item left incomplete. |
| | **Models and routing** | | |
| 12 | Model routing / gateway | done | Gemini removed; any OpenAI-compatible endpoint configurable. |
| 13 | Web search | done | Per question; pages go to the model unembedded. |
| | **Frontend and experience** | | |
| 14 | React rebuild | done | React + TypeScript, Mermaid, streamed output. |
| 15 | Configurable response model | done | Backend, base URL, key, model, temperature, max tokens in Settings. Key masked on read, never overwritten by its own mask. |
| 16 | Status, ETA, logs | done | Ingestion view with per-stage counts, ETA and live logs in their own tab. The first ETA was wrong; now median-based. |
| | **API and backend** | | |
| 17 | Endpoint naming | done | Versioned REST nouns under `/v1/`. |
| 18 | Embedding saved chats | done | Into a **separate collection**, so a past model answer is never retrieved as if it were evidence from a paper. |
| | **Quality and platform** | | |
| 19 | Logging and error handling | done | Real logging replaced `print`. Failures record `status = error` instead of retrying forever; errored papers are listed and resettable in a batch. |
| 20 | Cross-platform | mostly | `scripts/ops.py` (psutil) replaced the bash launchers and works on macOS/Windows. Only the daily timer is Linux-specific. |
| | **Licensing and commercialization** | | |
| 21 | License audit | done | Every dependency audited, then acted on. ultralytics gone. PyMuPDF is the one remaining AGPL dependency, deliberately kept. |
| 22 | Ownership enforcement | shelved | Answered and parked with the rest of commercialization. |

## 11. Where it stands

48 papers parsed, chunked, embedded and searchable. 161 tests pass; the
staged smoke test passes 7/7. The layout pipeline runs on ONNX Runtime with
no AGPL inference library. The whole system is operable from one browser tab.
Work is committed and pushed. **The exposed API key has been revoked.**

Open items:

| Item | State |
|---|---|
| **Hardware-aware setup never reached the UI** | Unfinished. `hardware_check.py` runs only from the command line, so on a different machine the app gives no guidance on which models fit. |
| **A parsing artefact with no confirmed cause** | The symptom is real; the recorded page-number explanation is **wrong**. Entry 2 of PENDING_IMPROVEMENTS is now marked disputed so nobody acts on it. Needs a fresh reproduction with a specific PDF and page. |
| **Figures and tables lose structure** | Deferred. VLM pass designed and shelved on memory grounds; Docling is the cheaper partial answer for tables and is already wired in. |
| **PyMuPDF is still AGPL** | Deferred. Only matters on distribution. Both exits costed; the deciding benchmark is committed and re-runnable. |
| **Commercialization** | Deferred. Retraining on a permissive architecture and the ownership question are parked until distribution stops being hypothetical. |
