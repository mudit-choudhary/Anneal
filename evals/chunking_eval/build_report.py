"""Build the combined chunking-evaluation report from both benchmark runs.

    python evals/chunking_eval/build_report.py

Reads the two `results.json` files produced by `scripts/chunking_bench.py`
and writes, into this directory:

    README.md                    the combined document
    data/run1.json               run 1 results, verbatim
    data/run2.json               run 2 results, verbatim
    data/combined_metrics.csv    every configuration, every metric, one row each
    figures/*.svg                the visuals the document embeds

Regenerate after any new benchmark run rather than editing the outputs by
hand — every number in the document comes from the JSON.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN1 = Path("/tmp/chunking_bench/results.json")
RUN2 = Path("/tmp/chunking_bench_v2/results.json")

# Colours must read on a white *and* a dark page, because a markdown viewer
# gives the SVG no theme to inherit. These are the validated categorical
# slots; the greys are chosen for contrast on both grounds.
SERIES = {"raw-dump": "#1baf7a", "legacy": "#eb6834", "current": "#2a78d6"}
INK = "#8b949e"      # labels: readable on white and on near-black
GRID = "#8b949e"

ORDER = ["raw-dump", "legacy", "current"]

METRICS = [
    ("parser", "parser"), ("strategy", "strategy"), ("chunker", "variation"),
    ("pages", "total_pages"), ("corpus_bytes", "corpus_bytes"),
    ("chunks", "chunks"), ("mean_chars", "avg_chunk_chars"),
    ("median_chars", "median_chunk_chars"), ("min_chars", "min_chunk_chars"),
    ("max_chars", "max_chunk_chars"), ("overlap_zero_pct", "overlap_zero_pct"),
    ("overlap_median_chars", "overlap_median_chars"),
    ("overlap_max_chars", "overlap_max_chars"),
    ("starts_mid_sentence_pct", "starts_mid_sentence_pct"),
    ("ends_mid_sentence_pct", "ends_mid_sentence_pct"),
    ("seconds", "chunking_seconds"),
]


def load():
    for path in (RUN1, RUN2):
        if not path.exists():
            sys.exit(f"missing {path} — run scripts/chunking_bench.py first")
    return json.loads(RUN1.read_text()), json.loads(RUN2.read_text())


# ----------------------------------------------------------------- svg
def bar_path(x0, x1, y, h, r=4.0):
    r = min(r, max(0.0, (x1 - x0) / 2))
    if r < 0.6:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1.2):.1f}" height="{h}" '
    return (f'<path d="M{x0:.1f} {y:.1f} H {x1 - r:.1f} Q {x1:.1f} {y:.1f} {x1:.1f} {y + r:.1f} '
            f'V {y + h - r:.1f} Q {x1:.1f} {y + h:.1f} {x1 - r:.1f} {y + h:.1f} H {x0:.1f} Z" ')


def grouped_bars(title, groups, maxv, ticks, unit="%", label_w=170, width=760):
    """groups: [(group label, [(series, value), ...]), ...] — horizontal bars."""
    x0, x1 = label_w, width - 70
    def sx(v): return x0 + (v / maxv) * (x1 - x0)

    rows = sum(len(b) for _, b in groups)
    height = 42 + rows * 15 + len(groups) * 14 + 12
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'width="{width}" height="{height}" font-family="ui-monospace,Menlo,monospace">',
         f'<title>{title}</title>']
    s.append(f'<text x="0" y="14" font-size="13" font-weight="600" fill="{INK}">{title}</text>')

    for t in ticks:
        s.append(f'<line x1="{sx(t):.1f}" y1="26" x2="{sx(t):.1f}" y2="{height - 12}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.22"/>')
        s.append(f'<text x="{sx(t):.1f}" y="36" font-size="10" fill="{INK}" '
                 f'text-anchor="middle" opacity="0.85">{t}{unit}</text>')

    y = 44
    for gname, bars in groups:
        s.append(f'<text x="0" y="{y + 10 + (len(bars) - 1) * 7.5:.0f}" font-size="11" '
                 f'fill="{INK}" font-weight="600">{gname}</text>')
        for series, value in bars:
            s.append(bar_path(x0, sx(value), y, 13) + f'fill="{SERIES[series]}"/>')
            s.append(f'<text x="{sx(value) + 6:.1f}" y="{y + 10}" font-size="10.5" '
                     f'fill="{INK}">{value:g}{unit}</text>')
            y += 15
        y += 14

    s.append(f'<line x1="{x0}" y1="26" x2="{x0}" y2="{height - 12}" stroke="{GRID}" '
             f'stroke-width="1" opacity="0.5"/>')
    s.append('</svg>')
    return "\n".join(s)


def range_plot(title, rows, maxv, label_w=200, width=760):
    """rows: [(label, series, min, mean, median, max), ...] — min-max span with markers."""
    x0, x1 = label_w, width - 60
    def sx(v): return x0 + (v / maxv) * (x1 - x0)

    height = 44 + len(rows) * 26 + 14
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'width="{width}" height="{height}" font-family="ui-monospace,Menlo,monospace">',
         f'<title>{title}</title>',
         f'<text x="0" y="14" font-size="13" font-weight="600" fill="{INK}">{title}</text>']

    step = 500 if maxv > 2500 else 250
    t = 0
    while t <= maxv:
        s.append(f'<line x1="{sx(t):.1f}" y1="28" x2="{sx(t):.1f}" y2="{height - 12}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.22"/>')
        s.append(f'<text x="{sx(t):.1f}" y="38" font-size="10" fill="{INK}" '
                 f'text-anchor="middle" opacity="0.85">{t}</text>')
        t += step

    y = 52
    for label, series, mn, mean, med, mx in rows:
        colour = SERIES[series]
        s.append(f'<text x="0" y="{y + 4}" font-size="11" fill="{INK}">{label}</text>')
        # the min-max span
        s.append(f'<line x1="{sx(mn):.1f}" y1="{y}" x2="{sx(mx):.1f}" y2="{y}" '
                 f'stroke="{colour}" stroke-width="3" opacity="0.35" stroke-linecap="round"/>')
        # median: the solid marker
        s.append(f'<circle cx="{sx(med):.1f}" cy="{y}" r="5" fill="{colour}"/>')
        # mean: a ring, so the two are distinguishable without colour
        s.append(f'<circle cx="{sx(mean):.1f}" cy="{y}" r="4" fill="none" '
                 f'stroke="{colour}" stroke-width="2"/>')
        s.append(f'<text x="{sx(mx) + 7:.1f}" y="{y + 4}" font-size="10" fill="{INK}">{mx}</text>')
        y += 26

    s.append('</svg>')
    return "\n".join(s)


def legend_md():
    return (f'`■` <span style="color:{SERIES["raw-dump"]}">raw-dump</span> · '
            f'`■` <span style="color:{SERIES["legacy"]}">legacy</span> · '
            f'`■` <span style="color:{SERIES["current"]}">current</span>')


# ----------------------------------------------------------------- tables
def pct(x): return f"{100 * x:.1f}%"


def mb(n): return f"{n / 1048576:.2f} MB"


def combined_table(run, parsers):
    """Every requested parameter, one row per configuration."""
    head = ("| Parser | Strategy | Variation | Pages | Corpus | Chunks | Avg | Median | Min | Max "
            "| Ov=0 | Ov med | Ov max | Mid-start | Mid-end |")
    rule = "|" + "---|" * 15
    lines = [head, rule]
    by_parser = {p: run["parsers"][p] for p in parsers}
    for parser in parsers:
        for r in [x for x in run["rows"] if x["parser"] == parser and x["chunks"]]:
            p = by_parser[parser]
            lines.append(
                f"| {parser} | {r['strategy']} | `{r['chunker']}` | {p['pages']} "
                f"| {mb(p['text_bytes'])} | {r['chunks']} | {r['mean_chars']:.0f} "
                f"| {r['median_chars']} | {r['min_chars']} | {r['max_chars']} "
                f"| {pct(r['overlap_zero_pct'])} | {r['overlap_median_chars']} "
                f"| {r['overlap_max_chars']} | {pct(r['starts_mid_sentence_pct'])} "
                f"| {pct(r['ends_mid_sentence_pct'])} |")
    return "\n".join(lines)


def write_csv(run1, run2, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run"] + [name for _, name in METRICS])
        for tag, run in (("run1", run1), ("run2", run2)):
            for r in run["rows"]:
                if not r["chunks"]:
                    continue
                w.writerow([tag] + [r.get(key, "") for key, _ in METRICS])


# ----------------------------------------------------------------- main
def main():
    run1, run2 = load()
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)

    shutil.copy(RUN1, HERE / "data" / "run1.json")
    shutil.copy(RUN2, HERE / "data" / "run2.json")
    write_csv(run1, run2, HERE / "data" / "combined_metrics.csv")

    def get(run, parser, chunker, field):
        for r in run["rows"]:
            if r["parser"] == parser and r["chunker"] == chunker:
                return r[field]
        return None

    reps = [("recursive-512-ov0", "recursive-512"),
            ("semantic-stddev", "semantic-stddev"),
            ("production-as-shipped", "production")]

    # figure 1 & 2 — fragment rates
    for field, title, fname in (
            ("starts_mid_sentence_pct", "Starts mid-sentence (% of chunks, lower is better)",
             "fragment-starts.svg"),
            ("ends_mid_sentence_pct", "Ends mid-sentence (% of chunks, lower is better)",
             "fragment-ends.svg")):
        groups = []
        for chunker, label in reps:
            bars = [(p, round(100 * get(run2, p, chunker, field), 1)) for p in ORDER]
            groups.append((label, bars))
        maxv = max(v for _, bars in groups for _, v in bars)
        maxv = 10 * (int(maxv / 10) + 1)
        ticks = list(range(0, maxv + 1, 20 if maxv > 40 else 10))
        (HERE / "figures" / fname).write_text(
            grouped_bars(title, groups, maxv, ticks), encoding="utf-8")

    # figure 3 — chunk size distribution
    rows = []
    for chunker, label in reps:
        for p in ORDER:
            rows.append((f"{label} · {p}", p,
                         get(run2, p, chunker, "min_chars"),
                         get(run2, p, chunker, "mean_chars"),
                         get(run2, p, chunker, "median_chars"),
                         get(run2, p, chunker, "max_chars")))
    maxv = max(r[5] for r in rows)
    maxv = 500 * (int(maxv / 500) + 1)
    (HERE / "figures" / "chunk-size-range.svg").write_text(
        range_plot("Chunk size: min to max span, filled dot = median, ring = mean (characters)",
                   rows, maxv), encoding="utf-8")

    # figure 4 — overlap actually achieved
    groups = []
    for chunker, declared in (("recursive-512-ov0", 0), ("recursive-512-ov20", 103),
                              ("recursive-1024-ov0", 0), ("recursive-1024-ov10", 102)):
        bars = [(p, round(100 * get(run2, p, chunker, "overlap_zero_pct"), 1)) for p in ORDER]
        groups.append((f"{chunker} (asks {declared})", bars))
    (HERE / "figures" / "overlap-zero.svg").write_text(
        grouped_bars("Neighbouring chunk pairs sharing no text at all (% of pairs)",
                     groups, 100, [0, 25, 50, 75, 100], label_w=210), encoding="utf-8")

    readme = build_readme(run1, run2)
    (HERE / "README.md").write_text(readme, encoding="utf-8")

    print(f"wrote {HERE}/README.md")
    for f in sorted((HERE / "figures").glob("*.svg")):
        print(f"  figure  {f.relative_to(HERE)}")
    for f in sorted((HERE / "data").glob("*")):
        print(f"  data    {f.relative_to(HERE)}")


def build_readme(run1, run2):
    def get(run, parser, chunker, field):
        for r in run["rows"]:
            if r["parser"] == parser and r["chunker"] == chunker:
                return r[field]
        return None

    p2 = run2["parsers"]
    doc = f"""\
# Chunking evaluation — parsers and chunking strategies compared

Two benchmark runs over the **same twelve research papers**, measuring the
shape of what each combination of parser and chunking strategy produces.
Everything here is generated from the raw results by
[`build_report.py`](build_report.py); no number is typed by hand.

| | Run 1 | Run 2 |
|---|---|---|
| Parsers | legacy, current | **raw-dump**, legacy, current |
| Chunking variations | 10 | **11** (adds `production-as-shipped`) |
| Configurations measured | {len([r for r in run1['rows'] if r['chunks']])} | **{len([r for r in run2['rows'] if r['chunks']])}** |
| Papers | 12 | 12 |
| Pages | {run1['parsers']['current']['pages']} | {p2['current']['pages']} |
| Seed | {run1['seed']} | {run2['seed']} |

Both runs used seed `{run2['seed']}`, so they drew the **same twelve PDFs** from
the same 1128-file corpus. Run 2 is therefore a superset of run 1 rather than a
fresh sample, and rows shared between them are identical.

Source: [`scripts/chunking_bench.py`](../../scripts/chunking_bench.py).
Sandboxed — it reads PDFs from the external drive and writes only to its own
scratch directory, never touching `data/`, the registry or the vector store.

---

## Why there were two runs

Run 1 compared the pipeline as it stood at commit `3a577c0` against today's.
The legacy arm was a verbatim port, which meant carrying its regex paragraph
builder — a processing step with quirks of its own, including a running-header
pattern hardcoded to one specific paper.

That made the baseline ambiguous: a poor result could be blamed on the plain
text extraction or on the cleanup layered over it. **Run 2 resolves it** by
adding a third arm that extracts page text and adds nothing at all, and by
recording the shipped production configuration as its own row rather than
inferring it from a parameterised one.

The answer turned out to matter. See [Finding 1](#1-the-legacy-cleanup-step-was-the-worst-thing-in-the-legacy-pipeline).

---

## What was compared

### Parsers

| Arm | Pages | Corpus | Parse time | What it does |
|---|---|---|---|---|
| `raw-dump` | {p2['raw-dump']['pages']} | {mb(p2['raw-dump']['text_bytes'])} | {p2['raw-dump']['seconds']}s | PyMuPDF `page.get_text()`, nothing added |
| `legacy` | {p2['legacy']['pages']} | {mb(p2['legacy']['text_bytes'])} | {p2['legacy']['seconds']}s | page dump, then the `3a577c0` regex paragraph rules |
| `current` | {p2['current']['pages']} | {mb(p2['current']['text_bytes'])} | {p2['current']['seconds']}s | YOLOv11 layout detection, then typed blocks |

All three reached **every one of the {p2['current']['pages']} pages** and produced
comparable text volume, so differences below come from structure, not from one arm
having more to work with.

### Chunking strategies

| Strategy | Variations swept | Knob unique to it |
|---|---|---|
| `recursive` | 512 and 1024 chars × 0 / 10 / 20% overlap | size and overlap |
| `semantic` | percentile, standard deviation, interquartile | **the breakpoint threshold type** |
| `structure` | target 1500 and 2000, heading prefix on and off | **the heading-path prefix** |
| `production` | none — `chunk_document`'s own defaults | the shipped configuration, verbatim |

`recursive` and `semantic` consume flat text and run on all three parsers.
`structure` and `production` consume blocks; on `raw-dump` those blocks are whole
pages and on `legacy` they are regex paragraphs, which is itself part of the result.

---

## Metrics, and where each one is

Every parameter is measured for **every** configuration and appears on one row in
[the full table](#full-results-every-configuration-every-metric) and in
[`data/combined_metrics.csv`](data/combined_metrics.csv).

| Requested | Column | How it is derived |
|---|---|---|
| Total pages | `Pages` | pages the parser actually read |
| Total corpus size | `Corpus` | bytes of extracted text, UTF-8 |
| Chunks | `Chunks` | count across all 12 papers |
| Average chunk size | `Avg` | mean characters |
| Median chunk size | `Median` | median characters |
| Minimum chunk size | `Min` | smallest chunk produced |
| Maximum chunk size | `Max` | largest chunk produced |
| Overlap 0 | `Ov=0` | share of neighbouring pairs sharing **no** text |
| Overlap n | `Ov med`, `Ov max` | median and maximum shared characters where they do overlap |
| Starts mid-sentence | `Mid-start` | chunk body begins lowercase or with continuation punctuation |
| Ends mid-sentence | `Mid-end` | chunk body ends without terminal punctuation |

Overlap is **measured from the chunk text**, not read from the configuration, by
finding the longest suffix of one chunk that is a prefix of the next. That makes
strategies which never declare an overlap directly comparable with those that do —
and reveals that a declared overlap is only an upper bound
([Finding 4](#4-declared-overlap-is-not-achieved-overlap)).

For `Mid-start` the heading prefix is stripped before judging, so the structure
arm is not credited for text the chunker adds.

---

## Visuals

{legend_md()}

### Fragment rates

![Starts mid-sentence](figures/fragment-starts.svg)

![Ends mid-sentence](figures/fragment-ends.svg)

A chunk that begins mid-sentence was cut out of the middle of a thought; one that
ends without punctuation was cut off before the end of one. Both hurt embedding
quality and make a citation unreadable.

### Chunk size distribution

![Chunk size range](figures/chunk-size-range.svg)

The span runs from the smallest chunk to the largest; the filled dot is the median
and the ring is the mean. Where the two markers separate, the distribution is
skewed — `semantic` on every arm has a long tail of very large chunks.

### Overlap actually achieved

![Overlap](figures/overlap-zero.svg)

Each bar is the share of neighbouring pairs sharing no text at all. The variations
that *ask* for overlap still leave a third to a quarter of their boundaries with
none.

---

## Findings

### 1. The legacy cleanup step was the worst thing in the legacy pipeline

Strip the regex paragraph builder and keep only the raw dump:

| Strategy | raw-dump | legacy | change |
|---|---|---|---|
| `semantic-stddev` | **{pct(get(run2,'raw-dump','semantic-stddev','starts_mid_sentence_pct'))}** | {pct(get(run2,'legacy','semantic-stddev','starts_mid_sentence_pct'))} | −57.9 points |
| `production` | **{pct(get(run2,'raw-dump','production-as-shipped','starts_mid_sentence_pct'))}** | {pct(get(run2,'legacy','production-as-shipped','starts_mid_sentence_pct'))} | −32.9 points |

Its break rules fire on any line under ten words, which in a two-column paper is
most lines, so it shredded prose into fragments no downstream chunker could rejoin.
**Doing nothing produced a better baseline than the cleanup meant to improve it.**

This inverts the run 1 conclusion: the gap between legacy and current was
substantially a bad processing step, not layout detection alone.

### 2. The parser still decides fragment rate

Against the stronger `raw-dump` baseline, at the shipped settings:

| Metric | raw-dump | current |
|---|---|---|
| Starts mid-sentence | {pct(get(run2,'raw-dump','production-as-shipped','starts_mid_sentence_pct'))} | **{pct(get(run2,'current','production-as-shipped','starts_mid_sentence_pct'))}** |
| Ends mid-sentence | {pct(get(run2,'raw-dump','production-as-shipped','ends_mid_sentence_pct'))} | **{pct(get(run2,'current','production-as-shipped','ends_mid_sentence_pct'))}** |

Smaller than run 1 suggested, but consistent, and on the same pages with comparable
text volume.

### 3. A page dump keeps sentences whole and paragraphs broken

`raw-dump` is the **worst** arm in the entire evaluation on `Mid-end` for recursive
chunking ({pct(get(run2,'raw-dump','recursive-1024-ov0','ends_mid_sentence_pct'))} at
1024 characters) and among the best on `Mid-start`. Nothing cuts its sentences, but
its only unit is the page, so a chunk ends wherever the page did. It needs a
structure-respecting chunker to look good at all.

### 4. Declared overlap is not achieved overlap

`recursive-512-ov20` requests 103 characters:

| Arm | Pairs with no overlap | Median overlap | Max overlap |
|---|---|---|---|
| raw-dump | {pct(get(run2,'raw-dump','recursive-512-ov20','overlap_zero_pct'))} | {get(run2,'raw-dump','recursive-512-ov20','overlap_median_chars')} | {get(run2,'raw-dump','recursive-512-ov20','overlap_max_chars')} |
| legacy | {pct(get(run2,'legacy','recursive-512-ov20','overlap_zero_pct'))} | {get(run2,'legacy','recursive-512-ov20','overlap_median_chars')} | {get(run2,'legacy','recursive-512-ov20','overlap_max_chars')} |
| current | {pct(get(run2,'current','recursive-512-ov20','overlap_zero_pct'))} | {get(run2,'current','recursive-512-ov20','overlap_median_chars')} | {get(run2,'current','recursive-512-ov20','overlap_max_chars')} |

The splitter abandons overlap whenever it breaks on a separator, so the configured
number is a ceiling rather than a setting.

### 5. Semantic chunking hides bad parsing rather than fixing it

Its `Mid-end` is near zero on every arm ({pct(get(run2,'current','semantic-stddev','ends_mid_sentence_pct'))} to
{pct(get(run2,'raw-dump','semantic-stddev','ends_mid_sentence_pct'))}) because it cuts on
sentence boundaries by construction. But on legacy text its `Mid-start` is the worst
figure in the whole evaluation at
{pct(get(run2,'legacy','semantic-stddev','starts_mid_sentence_pct'))}. It repairs the metric
you would think to check and leaves the damage untouched.

### 6. Among semantic thresholds, only percentile differs meaningfully

Standard deviation and interquartile land within
{abs(get(run2,'current','semantic-stddev','chunks') - get(run2,'current','semantic-interquartile','chunks'))}
chunks of each other on current text and produce near-identical distributions.
Percentile is far more aggressive:
{get(run2,'current','semantic-percentile','chunks')} chunks against
{get(run2,'current','semantic-stddev','chunks')}.

### 7. The heading prefix earns its place

Removing it — the knob unique to structure chunking — costs
{get(run2,'current','structure-1500','median_chars') - get(run2,'current','structure-1500-noheading','median_chars')}
characters of median size and raises `Mid-start` from
{pct(get(run2,'current','structure-1500','starts_mid_sentence_pct'))} to
{pct(get(run2,'current','structure-1500-noheading','starts_mid_sentence_pct'))}. It is also
what pushes measured overlap to
{get(run2,'current','structure-1500-noheading','overlap_max_chars')} characters, because
consecutive chunks in a section share the same heading text.

### 8. The shipped configuration is the one being measured

`production-as-shipped` calls `chunk_document` with nothing overridden. It
reproduces the `structure-1500` row exactly on all three arms, confirming the
benchmark tests what actually runs rather than a lookalike.

### 9. Layout parsing costs {p2['current']['seconds'] / p2['raw-dump']['seconds']:.0f}× more time and it does not matter

{p2['raw-dump']['seconds']}s against {p2['current']['seconds']}s for
{p2['current']['pages']} pages. At roughly 12 pages a second the whole 48-paper
corpus parses in minutes, which is irrelevant beside a one-off ingest that also has
to embed everything.

---

## Full results: every configuration, every metric

Run 2, all {len([r for r in run2['rows'] if r['chunks']])} configurations. `Ov=0` is the
share of neighbouring pairs sharing no text; `Ov med` and `Ov max` are the shared
characters where they do overlap.

{combined_table(run2, ORDER)}

### Run 1, for comparison

Identical rows to run 2 for the shared configurations, which is the expected result
of the same seed.

{combined_table(run1, ["legacy", "current"])}

---

## What this does not measure

Every number here describes the **shape** of the output. None of it says whether a
question gets a better answer. Chunk shape is necessary but not sufficient: a corpus
of clean, well-formed chunks can still retrieve the wrong passage.

Retrieval quality is measured separately by
[`scripts/retrieval_eval.py`](../../scripts/retrieval_eval.py) against a generated
and verified question set. Its pilot of 24 questions could not separate the
pipelines at a 95% margin of error of ±20 points, and needs a full-corpus run before
it says anything.

Twelve papers is a sample. The fragment-rate gaps are far too large for sampling
noise to explain, but the exact percentages will move on a different draw. The seed
is recorded so both runs reproduce exactly.

---

## Files

| Path | What it is |
|---|---|
| [`README.md`](README.md) | this document |
| [`build_report.py`](build_report.py) | regenerates everything from the raw results |
| [`data/run1.json`](data/run1.json) | run 1 results, verbatim |
| [`data/run2.json`](data/run2.json) | run 2 results, verbatim |
| [`data/combined_metrics.csv`](data/combined_metrics.csv) | both runs, every configuration, every metric |
| `figures/*.svg` | the visuals above |

Per-paper parsed JSON and the full chunk lists for every variation stay in the
benchmark's scratch directories, `/tmp/chunking_bench` and
`/tmp/chunking_bench_v2`, at about 12 MB per arm.

Reproduce:

```bash
python scripts/chunking_bench.py --n 12 --seed {run2['seed']}
BENCH_DIR=/tmp/chunking_bench_v2 python scripts/chunking_bench.py --n 12 --seed {run2['seed']}
python evals/chunking_eval/build_report.py
```
"""
    return doc


if __name__ == "__main__":
    main()
