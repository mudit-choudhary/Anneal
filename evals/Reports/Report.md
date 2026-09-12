# Parser and chunking harness — results

Generated 2026-09-12 12:20:06 by `evals/scripts/make_main_report.py` from `results.json`, `parser_quality.json` and the corpus manifest. Every number is read from those files.

## In one paragraph

Five parsers crossed with four chunking strategies over 15 fresh arXiv papers (401 pages), all 20 cells run. **Grain-Growth is the chunker to keep** by a wide margin over everything except semantic, which only scores well because it cuts between sentences by construction and produces chunks too large for the embedder. **On layout detection our parser and Docling sit within noise of each other** (2.0% against 1.3% mid-start), but ours is **2.87x faster**, uses 336 MB less VRAM, and recovers two document classes Docling does not deliver. Docling keeps roughly twice as many tables intact.

## Corpus

15 papers, 401 pages, 50.2 MB, 10 two-column, 6 with a detectable table. Seed `20260912`, pool of 73, accepted on draw 1. Downloaded one request per three seconds over a single connection, per the [arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html), and cached.

**Disjoint from the YOLO training set.** All arXiv ids were read off the page-1 stamp of every training PDF, giving 1,030 ids dated 2512, 2601. The corpus is dated 2609, so the exclusion filter removed **0** candidates — the sets never overlapped. Column count and table presence come from PyMuPDF geometry, never from our own layout model.

## Parsers

| Parser | Pages | Text | Sec/page | VRAM rise | Section barriers | What it is |
|---|---|---|---|---|---|---|
| `raw_dump` | 401 | 1.22 MB | 0.0319 | 0 MB | 0 | PyMuPDF `page.get_text()`, no structure |
| `legacy` | 401 | 1.07 MB | 0.0320 | 0 MB | 0 | commit `3a577c0`, page dump plus regex paragraph rules |
| `oss_docling` | 401 | 1.34 MB | 0.4462 | 1,380 MB | 406 | Docling layout, our stage-2 assembler |
| `oss_pymupdf4llm` | 401 | 1.22 MB | 0.9146 | 0 MB | 393 | PyMuPDF4LLM markdown mapped to typed blocks |
| `current` | 401 | 1.20 MB | 0.1557 | 1,044 MB | 399 | YOLOv11 layout, column-aware reading order |

`VRAM rise` is peak memory above what was already resident when the parser started; models loaded earlier stay on the card, so an absolute peak would credit every later parser with their footprint. Three of the five use no GPU.

![parser trade-off](assets/fig1_parser_tradeoff.svg)

### Time gain against Docling

| Corpus | `current` | `oss_docling` | Saved |
|---|---|---|---|
| 401 pages | 1.0 min | 3.0 min | 1.9 min |
| 1,000 pages | 2.6 min | 7.4 min | 4.8 min |
| 5,000 pages | 13.0 min | 37.2 min | 24.2 min |

**2.87x faster per page.** Parsing is a one-off cost per paper, so the absolute saving on a small ingest is minutes. The 336 MB of VRAM is the more durable advantage on a 4 GB card shared with the embedder.

`oss_pymupdf4llm` is the slowest arm at 0.9146 s/page despite using no GPU. Rule-based markdown conversion is not automatically cheap.

## The matrix: 5 parsers x 4 chunkers

| Parser | Chunker | Chunks | Avg | Median | Min | Max | Ov=0 | Ov med | Ov max | Mid-start | Mid-end | Tables whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `raw_dump` | `fixed_token` | 1,661 | 875 | 868 | 113 | 1,530 | 17.0% | 108 | 224 | 56.7% | 93.1% | 0.0% |
| `raw_dump` | `recursive_char` | 1,072 | 1,220 | 1,447 | 99 | 1,499 | 37.5% | 117 | 149 | 42.1% | 83.2% | 23.5% |
| `raw_dump` | `semantic` | 593 | 2,086 | 2,073 | 2 | 5,079 | 99.8% | 8 | 8 | 0.0% | 1.0% | 23.5% |
| `raw_dump` | `grain_growth` ⚠ | 1,033 | 1,265 | 1,394 | 49 | 1,990 | 39.4% | 115 | 470 | 5.5% | 31.3% | 23.5% |
| `legacy` | `fixed_token` | 1,330 | 957 | 986 | 110 | 1,590 | 14.6% | 124 | 225 | 64.9% | 92.6% | 0.0% |
| `legacy` | `recursive_char` | 859 | 1,343 | 1,439 | 156 | 1,499 | 27.7% | 104 | 149 | 40.7% | 65.9% | 0.0% |
| `legacy` | `semantic` | 529 | 2,069 | 2,075 | 42 | 2,892 | 100.0% | 0 | 0 | 0.0% | 0.8% | 0.0% |
| `legacy` | `grain_growth` ⚠ | 694 | 1,573 | 1,570 | 106 | 2,000 | 99.3% | 2 | 8 | 13.7% | 22.5% | 0.0% |
| `oss_docling` | `fixed_token` | 1,756 | 865 | 833 | 139 | 1,542 | 10.8% | 103 | 228 | 57.6% | 92.9% | 0.0% |
| `oss_docling` | `recursive_char` | 1,247 | 1,130 | 1,275 | 9 | 1,499 | 55.0% | 75 | 149 | 10.3% | 35.9% | 23.5% |
| `oss_docling` | `semantic` | 623 | 2,190 | 2,071 | 100 | 16,295 | 91.2% | 3 | 3 | 0.3% | 0.5% | 23.5% |
| `oss_docling` | `grain_growth` | 1,243 | 1,091 | 1,147 | 8 | 2,000 | 91.3% | 3 | 443 | 1.3% | 6.0% | 23.5% |
| `oss_pymupdf4llm` | `fixed_token` | 1,824 | 856 | 825 | 81 | 1,538 | 14.0% | 99 | 237 | 49.6% | 93.8% | 0.0% |
| `oss_pymupdf4llm` | `recursive_char` | 1,070 | 1,205 | 1,315 | 18 | 1,499 | 54.6% | 72 | 149 | 11.9% | 53.7% | 5.9% |
| `oss_pymupdf4llm` | `semantic` | 588 | 2,136 | 2,098 | 86 | 6,949 | 99.3% | 2 | 2 | 0.0% | 1.7% | 5.9% |
| `oss_pymupdf4llm` | `grain_growth` | 1,103 | 1,123 | 1,245 | 2 | 2,000 | 97.5% | 3 | 243 | 6.3% | 24.0% | 5.9% |
| `current` | `fixed_token` | 1,633 | 885 | 875 | 205 | 1,547 | 15.4% | 108 | 237 | 57.0% | 92.7% | 0.0% |
| `current` | `recursive_char` | 1,094 | 1,149 | 1,262 | 9 | 1,499 | 53.9% | 79 | 149 | 7.4% | 32.9% | 11.8% |
| `current` | `semantic` | 587 | 2,071 | 2,061 | 100 | 4,662 | 93.2% | 3 | 3 | 0.7% | 1.5% | 11.8% |
| `current` | `grain_growth` | 1,163 | 1,023 | 1,068 | 1 | 2,000 | 94.4% | 3 | 210 | 2.0% | 7.1% | 11.8% |

⚠ **DEGENERATE** — `raw_dump`, `legacy` with `grain_growth`. The strategy nucleates at section headings and stops at structural barriers; these parsers recover none, so it collapses to next-fit packing over blank-line-separated blocks. Run for completeness, **not a fair comparison**.

**Overlap** is the longest common suffix of chunk *i* and prefix of chunk *i+1*, on whitespace-normalised text — one definition used everywhere. **Mid-start** and **Mid-end** use the pipeline's own boundary rules, so the evaluation cannot disagree with the chunker: only `.`, `?` and `!` close a sentence, and a fence bar is a real boundary. **Tables whole** is the share of PyMuPDF-detected tables whose distinctive cell values land at least 80% inside one chunk.

Budgets: target 1500 characters, maximum 2000; fixed-token window 256 with 32 stride; embedding model `BAAI/bge-base-en-v1.5`.

![starts mid-sentence](assets/fig2_mid_start_pct.svg)

![ends mid-sentence](assets/fig2_mid_end_pct.svg)

The two heatmaps carry the main result. Reading down a column shows how little the parser matters once the chunker is fixed; reading across a row shows how much the chunker matters. `fixed_token` is above 90% mid-end on every parser, because a token window cuts wherever it runs out.

![overlap](assets/fig3_overlap.svg)

![chunk size ranges](assets/fig4_size_ranges.svg)

## The class gap: what Docling has no name for

The assembler is written against our twelve YOLO classes, and it runs on Docling because `LABEL_MAP` in `parse_manager/docling_backend.py` translates Docling's taxonomy into those names before anything downstream sees it. On the classes both models have, the two agree closely. Two come back empty, for different reasons: Docling *has* a `title` label and the map reads it, but its model never emitted one here; it has no `authors` concept at all.

> **A bug of mine, found while writing this.** Docling leaves `.text` empty on formula items and puts the content in `.orig`. The adapter read only `.text`, so it discarded every equation and this table showed Docling with zero formulas. Fixed, and the Docling arm re-run: it now recovers 1,388 formula blocks and its fragmentation figures improved. Same lesson as the fence-join bug earlier in this project: a defect in an integration is indistinguishable from a defect in the tool until you go and look.

| Block type | `raw_dump` | `legacy` | `oss_docling` | `oss_pymupdf4llm` | `current` |
|---|---|---|---|---|---|
| **title** | **0** | **0** | **0** | **0** | **16** |
| **authors** | **0** | **0** | **0** | **0** | **27** |
| paragraph | 401 | 11,148 | 1,356 | 4,587 | 1,322 |
| list | 0 | 0 | 112 | 0 | 102 |
| caption | 0 | 0 | 156 | 0 | 181 |
| table | 0 | 0 | 75 | 78 | 73 |
| formula | 0 | 0 | 1,388 | 0 | 1,440 |
| footnote | 0 | 0 | 35 | 0 | 36 |

- `raw_dump`: title found in 0 of 15 documents
- `legacy`: title found in 0 of 15 documents
- `oss_docling`: title found in 0 of 15 documents
- `oss_pymupdf4llm`: title found in 0 of 15 documents
- `current`: title found in 15 of 15 documents

Two consequences the fragmentation metrics never see:

**The heading path breaks.** `chunk_document` takes the title block or falls back to the filename. Share of chunks whose heading path is the arXiv id rather than the paper title: `current` 0.0%, `oss_docling` 100.0%. A chunk prefixed `2609.10065 › Introduction` has lost the title, which carries much of a paper's topical signal for retrieval.

**Author names get embedded as body prose.** `SKIP_TYPES` excludes `authors` from embedding. Docling has no authors label, so names and affiliations fall through as `Text` and become searchable content.

## Does Docling do the assembly work on its own?

The matrix cannot answer this: its `oss_docling` arm feeds Docling's layout into *our* assembler, the very component that merges paragraphs and drops page furniture. `evals/scripts/parser_quality_probe.py` separates the three, over 5 papers.

| Pipeline | Paragraph blocks | Begin mid-sentence |
|---|---|---|
| `docling_native` — Docling's own document model | 409 | 17.4% |
| `docling_plus_ours` — Docling layout, our assembler | 245 | 2.9% |
| `ours` — YOLO layout, our assembler | 257 | 4.3% |

**No.** Docling alone leaves 17.4% of its paragraphs beginning mid-sentence, roughly 6x worse than either pipeline running our assembler, and it emits 409 blocks where the assembler produces 245 from the same layout. It detects layout well; it does not reassemble prose.

Note the ordering between `docling_plus_ours` and `ours` on this 5-paper subset is the reverse of the matrix's 15-paper result. That flip is the signature of two layout detectors being equivalent, not of one leading.

**Header and footer removal is untested here.** Both parsers detected essentially no page furniture on this corpus (6 such items from Docling in total), because arXiv preprints carry little beyond a page number. Both taxonomies support the classes; this corpus does not exercise them. Published IEEE or Elsevier PDFs, with running heads on every page, would.

## Recommendation

| Chunker | Mean of mid-start and mid-end |
|---|---|
| `semantic` | 0.6% |
| `grain_growth` | 7.8% |
| `recursive_char` | 38.4% |
| `fixed_token` | 75.1% |

`semantic` tops that table only because it cuts between sentences by construction and so cannot end mid-sentence — it is scored on its own defining assumption. Its median chunk is 2,061 characters against 1,068, with a maximum of 4,662, and the embedder holds about 2,600. **Keep `grain_growth`.**

On the layout stage, keep `current`, for three reasons that are not the one I first credited it with:

1. It recovers `title` and `authors`. Docling's taxonomy *has* a title label and our map reads it, but its model emitted none on any of the 15 papers, classifying the title as a section header instead; it has no authors concept at all. The paper title therefore sits on every one of our chunks and none of Docling's, and author names stay out of our embedded text.
2. It is 2.87x faster and uses 336 MB less VRAM, which matters when the answering model wants the card.
3. On fragmentation it is level with Docling: 2.0% against 1.3% mid-start and 7.1% against 6.0% mid-end, a difference of +1.8 points summed. On 15 papers that is not a real gap either way.

### The strongest argument against it

**Docling beats us where it counts for tables and costs nothing to own.** It keeps 23.5% of tables whole against our 11.8% — TableFormer doing what it was built for — and it needs no annotation, no training runs, no ONNX export and no AGPL audit. The class gap is real but it is a mapping problem, not a modelling one: a dozen lines that infer the title from the first large text block on page 1, and authors from the block between title and abstract, would close most of it without a fine-tuned model at all.

If that were done, our remaining advantage would be parse speed and VRAM, and the case for maintaining a fine-tuned model would rest on the 4 GB card alone.

**Caveat.** 15 papers is a sample. The seed and the API queries in the manifest make it reconstructable, but exact percentages will move on a different draw, and gaps of a point or two are not real.

## Reproducing

```bash
python evals/scripts/build_corpus.py          # cached after the first run
python evals/scripts/run_matrix.py            # checkpointed per cell
python evals/scripts/parser_quality_probe.py
python evals/scripts/make_figures.py
python evals/scripts/make_main_report.py
```
