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

**Status:** Shelved, not started.

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
                Cora           Citeseer         Pubmed
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

## 2. A misclassified page number corrupts text and page attribution for everything until the next paragraph break

**Status:** Shelved, not started. Root cause identified, not yet fixed.

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

## 3. No service ever records `status = error`

**Status:** Not started. Small.

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
