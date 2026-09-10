# Pending Improvements

Ideas identified during testing that are worth doing but deliberately not
started yet — either because they need more design thought, a dependency
(model, library) not yet evaluated, or just aren't the current priority.
Add to this list as more surface; each entry should carry enough context
that revisiting it cold (weeks later, or a fresh session) doesn't require
re-deriving the problem.

**Format per entry**: what's wrong today (with a concrete example), why it
matters, and a candidate direction — not a committed design.

---

## 1. Tables and figures lose structure/meaning in the parsed output

**Status:** Partly addressed 2026-09-09 — decision: **no VLM** for now (user
choice). Tables: `scripts/rag_inspect.py tables` verifies number
preservation and saves crops; the Docling backend
(`parse_manager/docling_backend.py`, `PARSER_BACKEND = "docling"`) emits
tables as Markdown via TableFormer — see `scripts/docling_compare.py` for
the side-by-side benchmark. Figures still contribute only their captions.

### What's confirmed working

Text drawn *inside* a `Picture` region (figure labels, diagram text) is
correctly excluded from the body text flow — this was the original goal of
the YOLO-based parser and it works. See `SWALLOW_LABELS` in
[parse_manager/config.py](../parse_manager/config.py) and the `swallowed`
handling in [pdf_parser.py](../parse_manager/pdf_parser.py).

### What's not good enough

**Pictures** are dropped entirely from the text flow — only their `[CAPTION]`
survives (see `LayoutAssembler.add_region` in
[txt_processor.py](../parse_manager/txt_processor.py), the `Picture` branch
is a no-op). A figure like a model architecture diagram (e.g. Fig. 2 in "A
Community-Enhanced Graph Representation Model for Link Prediction" — a block
diagram of the CELP framework with labeled modules) contributes *nothing* to
the embedded text except its caption. If a question depends on understanding
what the figure shows, retrieval has nothing to work with beyond the caption
sentence.

**Tables** are parsed as raw text, not structure. `Table` regions go through
the same `words_to_lines` grouping as prose (`pdf_parser.py`) — words
clustered into y-bands and sorted left-to-right within each band — then
joined with `\n` (`txt_processor.py:163-164`). For a table like:

```
                            Cora          Citeseer         Pubmed
Degree centrality       91.67 ± 0.92    92.24 ± 1.62    82.70 ± 1.76
Betweenness centrality  91.88*± 0.74    94.20 ± 1.98    83.12*± 1.96
Closeness centrality    91.24 ± 1.36    93.46 ± 1.52    82.22 ± 2.12
PageRank centrality     92.41 ± 1.06    93.70*± 1.83    83.41 ± 1.64
```

the pipeline currently emits:

```
[TABLE]
Cora Citeseer Pubmed
Degree centrality 91.67 ± 0.92 92.24 ± 1.62 82.70 ± 1.76
Betweenness centrality 91.88*± 0.74 94.20 ± 1.98 83.12*± 1.96
Closeness centrality 91.24 ± 1.36 93.46 ± 1.52 82.22 ± 2.12
PageRank centrality 92.41 ± 1.06 93.70*± 1.83 83.41 ± 1.64
[/TABLE]
```

This happens to read okay for a simple, evenly-spaced 4-column table, but it
is fragile by construction:

- No reliable column boundaries — irregular spacing, merged cells, or
  multi-line headers will misalign values with the wrong header.
- Semantic markup is lost: **bolding** (best result) and `*` (second-best)
  survive only as a literal asterisk character with no explanation of what
  it means — a model reading the chunk has no way to know "PageRank
  centrality is the best result on Cora and Pubmed."
- No indication a block *is* a comparison table with two axes vs. an
  incidental list of numbers.

### Why it matters

Research papers put a lot of their actual findings in tables and figures.
Right now those either vanish (pictures) or become ambiguous number soup
(tables) — both are silent quality ceilings on retrieval and answer quality
that no amount of chunking or prompt tuning can fix, because the information
was never captured faithfully in the first place.

### Candidate direction: a VLM pass over Picture/Table regions

Rather than trying to reconstruct table structure from word coordinates
(fragile, diminishing returns), crop the region's bounding box from the
already-rendered page image (the pipeline already rasterizes every page for
YOLO at `RENDER_DPI` — see `layout_detector.py`) and send the crop to a small
vision-language model for:

- **Tables** → a markdown table (preserves column alignment properly) or a
  short prose summary of the key comparison, ideally capturing what
  bold/asterisk annotations mean if a caption or footnote explains them.
- **Pictures** → a 1-3 sentence description of what the figure shows,
  concatenated with its existing `[CAPTION]` text.

This would run as a **third overnight batch model** alongside YOLO and the
embedder — same GPU-sharing pattern already established
(`docs/architecture.md`'s day/night split), so it doesn't touch the daytime
Qwen budget.

**Open questions to resolve before implementing** (why this is shelved, not
just "todo"):

1. **Model choice** — needs to be small enough to coexist with YOLO in the
   overnight batch window on a 4GB card. Candidates to evaluate: Qwen2-VL-2B,
   moondream2, Florence-2-base, InternVL2-1B/2B. Table-reading accuracy on
   scientific papers (dense numbers, small fonts) needs an actual eval, not
   just a general VLM benchmark.
2. **VRAM/time budget** — running a VLM per Table/Picture region per paper
   adds meaningfully to overnight processing time; needs a rough throughput
   estimate against a real corpus size.
3. **Output representation** — does the VLM output replace the raw
   line-grouped text in `[TABLE]`, sit alongside it, or become a new tag
   (e.g. `[TABLE-DESC]`)? Losing the raw numbers entirely seems wrong (a VLM
   description won't reliably reproduce exact figures like "83.41 ± 1.64");
   more likely both should coexist so retrieval can match on either the
   literal numbers or a natural-language description of what they mean.
4. **Failure mode** — small VLMs hallucinate on dense tables. Needs a sanity
   check (e.g. do extracted numbers in the VLM's markdown table roughly match
   the OCR'd numbers?) before trusting the output over the raw text.
5. **Where in the pipeline** — a new stage in `parse_manager` (own module,
   e.g. `visual_describer.py`) that runs after YOLO detection, on the same
   rendered page images, only for `Table`/`Picture` region crops.

---

## 2. Text and page attribution get corrupted in parsed output (cause unconfirmed)

**Status:** Shelved, and the diagnosis below is **disputed**. Reviewed
2026-09-10: the page-number explanation is not what was actually observed,
so do not act on the root cause as written. The symptom (corrupted text and
page attribution in parsed output) is real but currently **uncharacterised** —
it needs a fresh reproduction with a specific PDF and page before any fix.
Everything below is kept as the original hypothesis, not as a finding.

### What's wrong

During YOLOv11 fine-tuning/testing, printed page numbers are usually caught
correctly by the `Page-header`/`Page-footer` classes (which is the *correct*
behavior — those regions are dropped, see `NOISE_LABELS` in
[txt_processor.py](../parse_manager/txt_processor.py)). But often enough,
the model instead classifies a bare page-number digit as `Text` — i.e. body
content — and that misclassification has a worse effect than just one stray
digit ending up in the output.

**Mechanism** (`LayoutAssembler.add_region`,
[txt_processor.py:173-178](../parse_manager/txt_processor.py#L173-L178)):

- A `Text` region merges into the currently *open* paragraph unless that
  paragraph already ends with terminal punctuation (`ends_terminally`,
  same file). A misclassified page number like `"6"` is exactly the kind of
  text that never satisfies this — it's a bare digit, no `.`/`!`/`?` — so it
  becomes (or extends) an open paragraph that is never allowed to close on
  its own.
- Regions are fed into the assembler **one page at a time, in order**
  (`process_layout_json`'s main loop). If that stray `"6"` is the last `Text`
  region on its page, it is still the *open* paragraph when the loop moves
  to the next page — so the real first paragraph of the next page gets
  silently merged onto the end of it via `merge_paragraph`.
- A block's `"page"` field is set once, when the paragraph is opened
  (`_open`, [txt_processor.py:138-142](../parse_manager/txt_processor.py#L138-L142))
  and never updated as more text merges in. So the resulting block reports
  the **earlier** page, while its actual (merged-in) content is from the
  **next** page — a lag of exactly one page. This matches what you saw:
  it isn't a global off-by-one (that would show up from block 1), it only
  appears on the pages where YOLO happens to mislabel the page number, which
  is intermittent — consistent with "very often, but not every time."

### Concrete failure shape

```
Page 6, last Text region on the page:  "6"        (should have been Page-footer)
Page 7, first real paragraph:          "Recent work has shown that..."
```

Instead of two blocks (a dropped page-footer, then a normal `page: 7`
paragraph), you get one corrupted block:

```json
{"type": "paragraph", "page": 6, "text": "6 Recent work has shown that..."}
```

The stray digit is prepended to real content, and everything in that
paragraph is now attributed to page 6 even though its content is from
page 7 — and this persists until the paragraph finally hits terminal
punctuation.

### Why it matters

Two separate harms: **(1)** page provenance in the processed JSON becomes
unreliable, which matters if page numbers are ever surfaced to the user
(e.g. "see page 7") or used for citation/debugging; **(2)** the text itself
is corrupted with a leading stray digit glued onto an unrelated sentence,
which is a small but real embedding-quality hit repeated wherever this
happens across a corpus.

### Candidate direction

The underlying problem is that the assembler fully trusts YOLO's label and
has no independent sanity check for "this doesn't look like real body text."
A cheap heuristic safety net, applied regardless of the model's label:

- Detect page-number-shaped `Text` regions — short (e.g. ≤ 4 chars),
  purely numeric (optionally with `Page N` / `N of M` patterns), positioned
  near the top or bottom margin of the page — and swallow them the same way
  `SWALLOW_LABELS` regions are swallowed in `pdf_parser.py`, instead of
  letting them enter the paragraph assembler as `Text`.
- Belt-and-braces alternative: never let a *single, very short, non-alphabetic*
  region become or extend an open paragraph across a page boundary — treat it
  as untrustworthy content and isolate it into its own dropped/flagged block
  rather than merging.
- Longer-term: track more than one page per block (e.g. `"pages": [6, 7]` or
  a `start_page`/`end_page` pair) so that even legitimate cross-page
  paragraph merges — which are an intentional feature, not a bug — don't
  silently misattribute content to the wrong page.

Any of these should be validated against real fine-tuning output (the
"page number classified as Text" failure rate) before picking one, since a
heuristic tuned too aggressively could start swallowing legitimate short
body text (e.g. a lone equation numeral, a table row label).

---

## 3b. A failed parse can leave a truncated paper indexed as complete

**Status:** ✅ Root cause understood and the two affected papers re-ingested
2026-09-11. The guard below is **not** implemented.

### What happened

`scripts/retrieval_eval.py` compared the parsed text against the raw PDFs and
found two papers holding a fraction of their content:

| paper | pages parsed | actual pages | text kept |
|---|---|---|---|
| A_Plan_Reuse_Mechanism_for_LLM-Driven_Agent | 1 | 11 | 9% |
| A_Community-Enhanced_Graph_Representation_Model | 10 | 34 | 29% |

Both were embedded and reported `status = embedded`. Nothing in the pipeline
or the UI indicated a problem: the registry tracks *stage*, not *completeness*.
Re-parsing them today produced all 11 and all 34 pages, so the parser is
correct — these were partial artifacts written during the VRAM incident that
also cost two papers their embeddings, and they were never re-done.

### Why it matters

A silently truncated paper is worse than a missing one. It answers questions
about its first page confidently and cannot answer anything else, and there is
no signal that the corpus is incomplete.

### Candidate direction

`pdf_parser.parse` already knows `len(doc)`. Compare it with the number of
pages that produced blocks and refuse to write a processed file that covers
materially fewer — post `status = error` instead, which
`register_pdfs.py --retry-errors` already knows how to reset. A cheap
corpus-wide audit is worth having too: parsed page count versus PDF page
count, per paper, surfaced on the Ingestion tab.

---

## 3. No service ever records `status = error`

**Status:** ✅ Resolved 2026-09-09. Every stage loop (`parse_manager/main.py`,
`embedding_manager/main.py`) now reports failures through
`RegistryClient.report_error`, which sets `status = error` with the message
and increments `error_count`; loops select work by status so errored papers
are skipped; `scripts/register_pdfs.py --retry-errors` resets them; the
Ingestion page and `pipeline_status.py` list them. Kept below for the record.

### What's wrong

The registry schema, `STATUS_TYPE`, and `FileRegistry.update_status` all
support an `error` status with `error_count` and `last_error` — but no
stage ever posts it. `parse_manager/main.py`'s loops catch exceptions and
`print` them; `embedding_manager/main.py` logs a failed `chunk_and_embed`
and moves on. The paper's status stays where it was, so it is **retried on
every poll, forever** (every 5 s for parsing), and `pipeline_status.py`'s
`error` row is always 0 even when a PDF is permanently broken (encrypted,
corrupt, scanned-image-only).

### Why it matters

A single bad PDF wastes GPU time continuously and spams the log, and there
is no way to see from the status view that something needs attention.

### Candidate direction

On exception, post `{status: "error", error_msg: str(e)}`; the registry
already increments `error_count`. Loops then skip papers with status
`error`, and `pipeline_status.py` already lists them. Optionally retry
errors up to N times (the `error_count` column exists for exactly this)
before giving up. Requires deciding how a paper gets *out* of `error` —
simplest: `register_pdfs.py --retry-errors` resets them to `downloaded`.

---

## 4. AGPL dependencies block commercial use as-is

**Status:** **ultralytics removed 2026-09-09** — layout inference now runs on
onnxruntime (MIT), verified as an exact behavioural swap (407/407 regions,
mean IoU 0.9999). ultralytics is uninstalled and needed only to re-run
`scripts/export_onnx.py` after retraining. **PyMuPDF remains** and is the
next swap (planned: `pypdfium2`), deferred until the current work is
committed. Commercialization is still not a current goal, but this is
recorded because it constrains library choices made *now*.

### What's wrong

Licenses of the installed dependencies (from `pip show`):

| Component | License |
|---|---|
| ~~**ultralytics**~~ (removed — build-time only now) | ~~AGPL-3.0~~ |
| **PyMuPDF** — the one remaining copyleft dependency | **AGPL-3.0** or a paid Artifex commercial licence |
| onnxruntime, opencv-python | MIT / Apache-2.0 |
| chromadb, sentence-transformers, transformers, requests, huggingface-hub | Apache-2.0 |
| torch, uvicorn, numpy | BSD |
| fastapi, pydantic, arxiv, loguru, Ollama, Docling | MIT |
| bge-base-en-v1.5 weights | MIT |
| Qwen3 weights | Apache-2.0 |

AGPL is network copyleft: shipping or *hosting* a product built on these
requires releasing the whole work under AGPL, or buying commercial licences.
The two AGPL components sit at the heart of parsing, so nothing else being
permissive helps.

### Why it matters now

Every additional parsing dependency chosen today either preserves or
forecloses the option. DocLayout-YOLO, for instance, is also AGPL, whereas
Docling is MIT — that difference is worth knowing before standardising on a
backend.

### Does fine-tuning your own weights avoid it?

**No** — for two independent reasons, and the first is the decisive one:

1. **The runtime dependency.** Inference calls `import ultralytics`. AGPL
   §13 covers network use, so serving answers from software that loads an
   AGPL library triggers the obligation regardless of whose weights it runs.
2. **Weights lineage.** The fine-tunes descend from
   `Armaggheddon/yolo11-document-layout`, itself a fine-tune of Ultralytics'
   AGPL-licensed checkpoints. Ultralytics' own position is that models
   trained from their weights are derivative works. This point is more
   arguable than (1), but it doesn't need to be won — (1) already applies.

(Not legal advice; a lawyer should confirm before anything commercial.)

### Candidate direction

Only if commercialization becomes real:

**Replacing ultralytics — done.** The ONNX route was taken: export once at
build time, run with onnxruntime. The shipped pipeline imports no AGPL code
for layout detection. What remains is only the *weights-lineage* argument
(the fine-tunes descend from Ultralytics checkpoints), which is genuinely
unsettled rather than a certainty. If that ambiguity ever needs to go:

| Option | Licence | Effort | Note |
|---|---|---|---|
| Retrain on **YOLOX** | Apache-2.0 | medium — convert the YOLO-format dataset to COCO, retrain | genuinely permissive YOLO family |
| Retrain on **RT-DETR** (HF `RTDetrForObjectDetection`) | Apache-2.0 | medium | strong on document layout |
| Use **Docling's** layout model | MIT | already wired (`PARSER_BACKEND="docling"`) | loses the fine-tuning and the `Authors` class |
| Ultralytics Enterprise licence | paid | none | only needed if you want to ship ultralytics itself |

Avoid **DocLayout-YOLO** (AGPL, built on ultralytics) and **LayoutLMv3**
(CC-BY-NC, non-commercial).

**Replacing PyMuPDF — evaluated 2026-09-09, and NOT done.** `pypdfium2`
(Apache-2.0/BSD-3) was benchmarked against PyMuPDF with
`scripts/pdf_backend_compare.py` on two papers / 20 pages. It is a real
option but measurably worse today, so switching now would trade output
quality for a licence benefit that only matters if commercialization
happens:

| Metric | PyMuPDF | pypdfium2 |
|---|---|---|
| Rendering | 105 / 32 pages/s | 55 / 33 pages/s (**1.18x slower**) |
| Word extraction | baseline | **1.8x slower** |
| Rendered pixels | — | 8–13% of pixels differ by >8/255 (different rasterizer) |
| Word boxes | — | mean IoU **0.89** with `loose=True`; only 0.73 with default ink boxes |
| Layout detections | 78 / 237 regions | 78 / 239, up to 11 regions differ |
| **Final processed text** | baseline | **87.2% similar** |

Three findings explain the gap, and each would need compensation code:

1. **No word API.** pdfium exposes characters; words must be rebuilt
   (~40 lines to write and keep correct).
2. **Box semantics differ.** Default char boxes are the glyph *ink* extent,
   so height varies by letter ("A" 11.3pt vs "Agent" 15.3pt) where PyMuPDF
   returns the font line height (uniform 17.22pt). The pipeline derives its
   line-grouping tolerance *and* column-gutter threshold from the median word
   height, so this silently retunes both (line tolerance 5.38 -> 3.80).
   `get_charbox(loose=True)` mostly fixes it — x then matches PyMuPDF almost
   exactly — but not entirely.
3. **Hyphenated line breaks are merged.** pdfium returns `degrad￾ing`
   as one word (U+FFFE marking the break) whose **bounding box spans both
   lines** — 19.4pt tall against a typical 8.9pt. That breaks line clustering
   outright, and the de-hyphenation vocabulary logic (which deliberately
   keeps `Edge-centric` intact) would have to be rebuilt around a different
   input shape.

**Recommendation:** keep PyMuPDF while commercialization is hypothetical. If
it becomes real, the cheapest path is the **Artifex commercial licence**
(zero code risk); the pypdfium2 route is viable but needs the compensation
work above plus a full re-ingest and re-validation. Re-run the benchmark any
time with `python scripts/pdf_backend_compare.py`.

For reference, the mapping if that work is ever done — only three files
import `fitz`, for three jobs:

| Job | Where | PyMuPDF | Replacement |
|---|---|---|---|
| Page → image | `layout_detector.py` | `page.get_pixmap(dpi=)` | `pypdfium2` `page.render(scale=)` |
| Word boxes | `pdf_parser.py` | `page.get_text("words")` | `pypdfium2` char boxes grouped into words, or `pdfplumber.extract_words()` (MIT) |
| Region crops | `rag_inspect.py` | `get_pixmap(clip=)`, `fitz.Rect` | render page, crop with Pillow |

Note `pdf2image`'s backend, poppler, is **GPL-2** — the wrapper being MIT
doesn't change that, so it isn't an escape route.

The separate practical point: software running on a customer's machine has
no technical anti-piracy guarantee — obfuscation and licence keys raise
effort, only a hosted service actually prevents redistribution.

---

## Template for new entries

```markdown
## N. Short title

**Status:** Shelved / Not started / In progress.

### What's wrong
(concrete example — a real block/output, not a hypothetical)

### Why it matters

### Candidate direction
(and open questions that need resolving before implementation starts)
```
