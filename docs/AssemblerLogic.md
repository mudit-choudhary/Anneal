# The Assembler — input, logic, output

The assembler is **stage 2** of parsing. Stage 1 decides *where* things are on
a page; the assembler decides *what order they are read in and where one
thought ends*. It is the component that turns a bag of labelled boxes into
prose a reader would recognise.

It is also, on the evidence of `evals/Reports/Report.md`, the highest-value
piece of the pipeline. Feeding Docling's layout through it cut Docling's
mid-sentence paragraph rate from 17.4% to 2.9%. The assembler is parser-
agnostic: it improves any layout detector's output, including ones we did not
write.

```
PDF ──▶ stage 1: layout detection ──▶ layout JSON ──▶ ASSEMBLER ──▶ blocks JSON ──▶ chunker
        (YOLOv11 / Docling)                                       + tagged .txt
```

---

## 1. Input format

The assembler consumes a **layout JSON**: one entry per page, each holding
labelled regions with pixel geometry and the words that fall inside them.

```jsonc
{
  "source_pdf": "…/2609.10065.pdf",
  "backend": "yolo",              // or "docling"
  "num_pages": 8,                 // pages the parser actually read
  "pdf_pages": 8,                 // pages the document has (coverage check)
  "pages": [
    {
      "page": 0,                  // zero-based
      "width": 612.0,             // PDF points, used for column geometry
      "height": 792.0,
      "regions": [
        {
          "label": "Text",        // one of the twelve classes below
          "conf": 0.94,
          "bbox": [72.0, 310.4, 300.1, 480.9],   // x0, y0, x1, y1, top-left origin
          "lines": ["First line of the region,", "second line contin-"]
        }
      ],
      "swallowed_text": [         // words dropped inside Picture/header/footer
        {"label": "Picture", "text": "Fig. 2 axis label"}
      ]
    }
  ]
}
```

### The twelve classes

| Label | Assembler treatment |
|---|---|
| `Title` | own block, forces a paragraph break |
| `Authors` | own block, forces a break, **excluded from embedding** |
| `Section-header` | own block, forces a break |
| `Text` | body prose, participates in paragraph merging |
| `List-item` | consecutive items accumulate into one list block |
| `Caption` | own block, does **not** break an open paragraph |
| `Table` | own block, lines preserved verbatim |
| `Formula` | own block, does **not** break an open paragraph |
| `Footnote` | own block, **excluded from embedding** |
| `Picture` | anchor only, contributes no text |
| `Page-header` | dropped from the flow, kept as metadata |
| `Page-footer` | dropped from the flow, kept as metadata |

### Two contracts the input must honour

**Geometry is top-left origin, in PDF points.** `bbox` is `[x0, y0, x1, y1]`
with `y` increasing downward, and `width` must be the page width in the same
units. Column detection is pure geometry, so a bottom-left origin silently
inverts reading order.

**`lines` are visual lines, not paragraphs.** The assembler expects the region
split at the line breaks the PDF actually has, because de-hyphenation works on
line ends. A region handed one pre-joined string still works, but wrap hyphens
inside it will never be repaired.

`swallowed_text` is optional and informational. Stage 1 already excluded those
words; the assembler only copies them into the output as evidence of what was
discarded.

---

## 2. The logic

### 2.1 Reading order — `order_regions`

Runs per page, before anything is read.

1. **Drop the furniture.** `Page-header` and `Page-footer` regions leave the
   flow immediately.
2. **Decide single- or two-column.** If at least `SINGLE_COLUMN_FRACTION`
   (0.7) of the regions that contain text are wider than
   `FULL_WIDTH_FRACTION` (0.6) of the page, the page is single-column and is
   read top-to-bottom, left-to-right. Done.
3. **Otherwise band the page.** A region is a **separator** if it is
   full-width, or if it is labelled `Authors`. Separators interrupt the column
   flow: everything accumulated above them is emitted as a two-column band
   first, then the separator, then accumulation resumes below.
4. **Within a band**, regions whose horizontal centre is left of the page
   midline are read top-to-bottom, then the right column top-to-bottom.
5. **Runs of separators that overlap vertically** are read left-to-right
   rather than top-to-bottom. This is the three-across author block: three
   narrow boxes on one row that would otherwise be read as a column.

The `Authors` special case earns its place: those boxes sit in the title band
and are narrow, so without naming them explicitly they would be sorted into a
body column and appear in the middle of the abstract.

### 2.2 Line joining and de-hyphenation — `join_lines`, `join_hyphenated`

A region's lines are joined with single spaces, except where a line ends in a
hyphen. Then:

- Continuation starts **uppercase** → keep the hyphen. `non-` + `Euclidean` →
  `non-Euclidean`.
- Continuation starts **lowercase** → drop it. `construc-` + `tion` →
  `construction`.
- **Unless the compound exists intact elsewhere in the document.** Before
  assembling, `collect_hyphenated_vocab` scans every line for hyphenated
  compounds appearing mid-line and lower-cases them into a set. If
  `edge-centric` appears intact anywhere, then `Edge-` + `centric` keeps its
  hyphen.

That vocabulary pass is what stops the de-hyphenator destroying real compound
terms, which in a technical corpus are common and load-bearing.

### 2.3 Paragraph reconstruction — `LayoutAssembler`

This is the core, and the part that produced the 17.4% → 2.9% improvement.

The assembler keeps **one open paragraph buffer** and walks regions in reading
order. A paragraph stays open across region, column and page boundaries. It
closes when either:

- its text **ends terminally** — `.`, `!` or `?`, optionally followed by a
  closing quote or bracket — and a new `Text` region arrives; or
- a `Title`, `Section-header` or `Authors` region forces a flush.

Everything else leaves it open. A `Caption`, `Formula`, `Footnote` or `Table`
appearing mid-paragraph is emitted as its own block **without closing the
buffer**, so a sentence interrupted by a figure caption is rejoined afterwards:

```
…the interplay between recombination and                 ← region ends, no full stop
[CAPTION] Fig. 1: The process of alert life-cycle…       ← emitted, buffer stays open
hydrodynamical instabilities has received…               ← merged into the same paragraph
```

The block records the page where the paragraph **started**, not where it ended.

`List-item` regions have their own rule: consecutive ones accumulate into a
single `list` block joined by newlines, and any other region type closes it.

### 2.4 What never reaches the output

- `Page-header`, `Page-footer` — collected into `dropped.page_headers` /
  `dropped.page_footers`
- text inside `Picture` boxes — collected into `dropped.picture_text` from
  stage 1's `swallowed_text`
- `Picture` regions themselves contribute nothing; they exist only as reading-
  order anchors

Note that `Authors` and `Footnote` blocks **are** emitted here. They are
excluded later, by the chunker's `SKIP_TYPES`, not by the assembler.

---

## 3. Output

Two files per document.

### 3.1 `data/processed/<stem>.json` — what the chunker consumes

```jsonc
{
  "source": "…/2609.10065.pdf",
  "num_pages": 8,
  "pdf_pages": 8,
  "dropped": {
    "page_headers": ["Cost-Aware Vision-Language Model Arbitration"],
    "page_footers": [],
    "picture_text": ["Fig. 2 axis label", "…"]
  },
  "blocks": [
    {"type": "title",     "page": 0, "text": "Cost-Aware Vision–Language…"},
    {"type": "authors",   "page": 0, "text": "A. Author, B. Author"},
    {"type": "section",   "page": 0, "text": "1 Introduction"},
    {"type": "paragraph", "page": 0, "text": "Cloud systems, such as…"},
    {"type": "caption",   "page": 3, "text": "Table 1: Results by dataset."},
    {"type": "table",     "page": 3, "text": "Cora 91.6\nCiteseer 92.2"}
  ]
}
```

Block types are lower-case and fewer than the input labels: `title`,
`authors`, `section`, `paragraph`, `list`, `caption`, `table`, `formula`,
`footnote`. Blocks with empty text are dropped.

`pdf_pages` alongside `num_pages` is what makes a truncated parse detectable
later — see `common/coverage.py`.

### 3.2 `data/processed/<stem>.txt` — for humans

The same blocks rendered with tags, for reading and for
`scripts/rag_inspect.py`:

```
# Cost-Aware Vision–Language Model Arbitration

[AUTHORS] A. Author, B. Author

## 1 Introduction

Cloud systems, such as Microsoft Azure…

[CAPTION] Table 1: Results by dataset.

[TABLE]
Cora 91.6
Citeseer 92.2
[/TABLE]
```

This file is **not** what gets embedded. The chunker reads the JSON. The `.txt`
tags and the chunker's `~~~table` fences are different conventions serving
different readers.

---

## 4. Files

| File | Role |
|---|---|
| **`parse_manager/txt_processor.py`** | **the assembler itself** — `order_regions`, `join_lines`, `join_hyphenated`, `collect_hyphenated_vocab`, `LayoutAssembler`, `render_txt`, `process_layout_json` |
| `parse_manager/config.py` | `FULL_WIDTH_FRACTION` (0.6), `SINGLE_COLUMN_FRACTION` (0.7), `SWALLOW_LABELS`, `PROCESSED_DIR` |
| `parse_manager/pdf_parser.py` | stage 1: renders pages, assigns words to regions, produces the layout JSON and `swallowed_text` |
| `parse_manager/layout_detector.py` | the YOLOv11 detector behind stage 1 |
| `parse_manager/onnx_detector.py` | ONNX Runtime inference: letterboxing, DFL decode, per-class NMS |
| `parse_manager/docling_backend.py` | the adapter that lets Docling feed this assembler (section 5) |
| `parse_manager/main.py` | the service loop that calls stage 1 then stage 2 |
| `embedding_manager/chunking.py` | the consumer — Grain-Growth chunking over these blocks |
| `common/coverage.py` | uses `num_pages` vs `pdf_pages` to detect a truncated parse |
| `tests/test_txt_processor.py` | unit tests for ordering, de-hyphenation and paragraph merging |
| `scripts/rag_inspect.py` | `parse` stage renders the `.txt` for inspection |

---

## 5. Reformatting Docling for this assembler

**`parse_manager/docling_backend.py`** is the adapter. It is the answer to
"the assembler is written for twelve YOLO classes, so how does it run on
Docling" — Docling's output is translated into that vocabulary before the
assembler sees anything.

Entry point:

```python
docling_backend.detect_pdf(pdf_path, max_pages=None)   # -> the same layout JSON
```

Enable it in production with `PARSER_BACKEND=docling`; `pdf_parser.parse()`
dispatches on it and everything downstream is unchanged.

What the adapter does:

1. **Translates labels** through `LABEL_MAP`, roughly twenty Docling labels
   onto our twelve. `section_header` → `Section-header`, `list_item` →
   `List-item`, `code` → `Text`, and so on; anything unrecognised falls back
   to `Text`.
2. **Flips the coordinate origin.** Docling boxes are bottom-left origin; ours
   are top-left. The adapter converts with `page_height - y`, which is
   essential because column detection is pure geometry.
3. **Extracts table structure** through TableFormer, exporting each table to
   markdown and splitting it into `lines`, rather than taking a flat text dump.
4. **Reads text from `.text`, falling back to `.orig`.** Docling leaves
   `.text` empty on formula items and puts the content in `.orig`. Reading
   only `.text` silently discarded every equation in the corpus — a bug that
   survived a full benchmark run before it was caught. See
   `evals/README.md`.
5. **Disables OCR** by default, since arXiv PDFs are born-digital and their
   text layer is already present. `DOCLING_OCR=1` re-enables it for scanned
   documents.

### What the adapter cannot supply

Two of the twelve classes come back empty, and no mapping fixes it:

| Class | Why | Consequence |
|---|---|---|
| `Title` | Docling *has* the label, but its model classified titles as section headers on all 15 evaluation papers | the chunker's heading path falls back to the filename, so **100%** of Docling chunks are prefixed with the arXiv id instead of the paper title |
| `Authors` | Docling has no such concept | author names fall through as `Text` and get embedded as body prose |

Closing that gap does not need a fine-tuned model. Inferring the title from the
largest text block on page 1, and authors from the block between title and
abstract, would recover both — see the recommendation in
`evals/Reports/Report.md`.
