# evals/ — the parser and chunking harness

Everything here exists to produce one artefact: **[Reports/Report.md](Reports/Report.md)**.
Nothing in this directory is kept unless that report reads it or a script that
feeds it writes it.

The harness crosses **5 parsers x 4 chunking strategies** over a corpus of 15
research papers drawn fresh from arXiv, and measures the shape of what comes
out: chunk sizes, overlap, sentence fragmentation, table integrity, parse time
and GPU memory.

## Layout

```
evals/
  README.md                       this file
  corpus/
    manifest.json                 the 15 papers: id, title, category, pages,
                                  size, columns, tables, SHA-256
    excluded_ids.json             arXiv ids read off every YOLO training PDF,
                                  used to guarantee the corpus is disjoint
    <arxiv-id>.pdf                cached downloads (git-ignored, ~490 MB)
  scripts/
    build_corpus.py               query arXiv, download, sample, write manifest
    run_matrix.py                 the 5x4 run; writes results.json
    parser_quality_probe.py       isolates Docling native vs our assembler
    make_figures.py               results.json -> assets/*.svg
    make_main_report.py           everything -> Report.md
  Reports/
    Report.md                     the report
    results.json                  every parser and cell measurement
    parser_quality.json           the paragraph-integrity probe
    assets/*.svg                  five figures, all regenerated from results.json
```

## Running it

```bash
python evals/scripts/build_corpus.py          # cached; never re-downloads
python evals/scripts/run_matrix.py            # ~70 min, checkpointed per cell
python evals/scripts/parser_quality_probe.py  # ~10 min
python evals/scripts/make_figures.py
python evals/scripts/make_main_report.py
```

`run_matrix.py` writes `results.json` after every cell, so a crash costs at most
one cell. `--only <parser>` re-runs a single arm and merges it into the existing
results rather than replacing them. `--limit N` runs a pilot on N papers.

The report is generated, never hand-edited. Every number in it is read from
`results.json`, `parser_quality.json` or the manifest.

## What is measured

**Per parser:** pages read, bytes of extracted text, seconds per page, peak VRAM
above the level already resident when it started, section barriers recovered,
and counts of each block type including `title` and `authors`.

**Per cell (20 of them):** chunk count, mean, median, min and max chunk size,
the share of adjacent pairs sharing no text, median and maximum overlap where it
exists, the share of chunks starting or ending mid-sentence, and the share of
tables landing wholly inside one chunk.

**Overlap** is defined once and used everywhere: the longest common suffix of
chunk *i* and prefix of chunk *i+1*, on whitespace-normalised text.

**Mid-start and mid-end** use the pipeline's own boundary predicates from
`embedding_manager/chunking.py`, imported rather than reimplemented, so the
evaluation cannot drift from what the chunker considers a clean edge.

## Corpus integrity

The corpus is **disjoint from the YOLO training set**. Every PDF in the training
directory was opened and its arXiv id read off the page-1 stamp; those ids are
cached in `corpus/excluded_ids.json` and filtered out of the candidate pool. On
this run the filter removed nothing, because the training set is dated 2512 and
2601 while the corpus is 2609 — the sets never overlapped. The filter stays as a
guard for future runs.

Column count and table presence in the manifest come from PyMuPDF geometry, not
from our own layout model. Using the thing under test to select the corpus would
make the sampling depend on it.

Downloads honour the
[arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html): one request
every three seconds, single connection, no parallelism. Every metadata query and
every PDF fetch goes through one function that enforces the floor.

## Reproducibility

The manifest records the seed, every API query string, and a SHA-256 per PDF.
Re-running `build_corpus.py` reuses the cache and never re-downloads.

One caveat worth knowing: the arXiv queries sort by submission date, so building
a corpus **from scratch** on a different day returns different papers. The
cached PDFs in `corpus/` are what make this specific run reproducible. Deleting
them does not just free space, it retires the corpus.

## Things this run got wrong before it got them right

Both were mine, not the tools', and both are worth remembering because they look
identical to a genuine finding until you check.

- **The Docling adapter discarded every formula.** Docling leaves `.text` empty
  on formula items and puts the content in `.orig`; the adapter read only
  `.text`. Fixed in `parse_manager/docling_backend.py`. Docling went from 0 to
  1,388 formula blocks and its fragmentation figures improved.
- **VRAM was measuring leftover resident models.** Models loaded by an earlier
  parser stay on the card, so an absolute peak credited every later parser with
  their footprint. Now reported as the rise above a pre-parse baseline, which is
  why the three CPU-only parsers correctly read 0 MB.

A third, in the corpus builder: `cat:astro-ph` returns only pre-2009 papers,
because arXiv split those archives into subcategories in 2009. The first build
produced a corpus dated 2008 before that was caught.

## Related

- `scripts/chunking_bench.py` — the earlier, narrower chunking benchmark that
  this harness supersedes. Still referenced from `docs/USER_GUIDE.md`.
- `scripts/retrieval_eval.py` — measures *retrieval quality* against a generated
  question set, which is the question this harness does not answer. Chunk shape
  is necessary but not sufficient.
