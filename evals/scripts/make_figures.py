"""Figures for the parser x chunker comparison.

    python evals/scripts/make_figures.py

Reads results.json (chunk-shape run) and, when present, rag_results.json,
questions/dataset.json and the parse cache (retrieval run). Every value comes
from those files; no number is written into this script. Emits SVG to
evals/Reports/assets/.

Colours are fixed hexes rather than CSS variables because a markdown viewer
gives an embedded SVG no theme to inherit; they are chosen to read on a white
and a near-black page alike.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "evals" / "Reports" / "results.json"
ASSETS = REPO / "evals" / "Reports" / "assets"

INK = "#8b949e"
GRID = "#8b949e"
# categorical slots, validated for CVD separation and contrast on both grounds
PARSER_COLOUR = {
    "raw_dump": "#1baf7a",
    "legacy": "#eb6834",
    "oss_docling": "#a87bd6",
    "oss_pymupdf4llm": "#eda100",
    "recrystal": "#2a78d6",
}
# sequential ramp for the heatmap: one hue, light to dark, monotonic lightness
RAMP = ["#eaf2fb", "#c5dcf5", "#94bfec", "#5f9ade", "#3576c4", "#1d4f8f"]
BAD = "#97382a"

PARSERS = ["raw_dump", "legacy", "oss_docling", "oss_pymupdf4llm", "recrystal"]
CHUNKERS = ["fixed_token", "recursive_char", "semantic", "grain_growth"]


def load():
    if not RESULTS.exists():
        raise SystemExit("no results.json yet")
    return json.loads(RESULTS.read_text())


def cell(r, p, c):
    return r["cells"].get(f"{p}|{c}")


def svg_open(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="ui-monospace,Menlo,monospace">',
            f'<title>{title}</title>']


def text(x, y, s, size=11, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def ramp_for(v, lo, hi):
    if v is None:
        return "#d8d8d8"
    t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
    return RAMP[min(len(RAMP) - 1, max(0, int(t * len(RAMP))))]


# --------------------------------------------------------------- fig 1
def fig_parser_tradeoff(r):
    """What each parser costs and what it buys.

    Small multiples rather than a scatter: three measures on wildly different
    scales, and two parsers sit on the identical point in speed-versus-structure
    space, so a scatter collided their labels and hid three zero-VRAM bubbles.
    One panel per measure, each with its own scale, keeps every value readable.
    """
    ps = [p for p in PARSERS if p in r["parsers"]]
    if not ps:
        return None

    panels = [
        ("Parse time", "s/page", lambda d: d.get("sec_per_page") or 0.0, 3),
        ("GPU memory", "MB above baseline", lambda d: d.get("vram_rise_mb") or 0.0, 0),
        ("Structure recovered", "section headings / page",
         lambda d: (d.get("section_barriers") or 0) / max(d.get("pages") or 1, 1), 2),
    ]

    label_w, panel_w, gap = 150, 195, 26
    row_h, top = 30, 76
    W = label_w + len(panels) * (panel_w + gap)
    H = top + len(ps) * row_h + 34

    s = svg_open(W, H, "Parser cost against structure recovered")
    s.append(text(0, 20, "What each parser costs, and what it buys", 13, INK, weight="600"))
    s.append(text(0, 38, "each panel has its own scale; bars are not comparable across panels",
                  10))

    for i, p in enumerate(ps):
        y = top + i * row_h
        s.append(text(label_w - 12, y + 13, p, 10.5, INK, "end"))

    for j, (title, unit, get, nd) in enumerate(panels):
        x0 = label_w + j * (panel_w + gap)
        vals = [get(r["parsers"][p]) for p in ps]
        hi = max(vals) or 1.0
        s.append(text(x0, 58, title, 11, INK, weight="600"))
        s.append(text(x0, 70, unit, 9, INK))
        s.append(f'<line x1="{x0}" y1="{top - 4}" x2="{x0}" y2="{H - 30}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.35"/>')
        for i, p in enumerate(ps):
            v = get(r["parsers"][p])
            y = top + i * row_h
            w = (v / hi) * (panel_w - 58)
            col = PARSER_COLOUR.get(p, INK)
            if w >= 1:
                rad = min(3.0, w / 2)
                s.append(f'<path d="M{x0} {y+3} H {x0+w-rad:.1f} Q {x0+w:.1f} {y+3} '
                         f'{x0+w:.1f} {y+3+rad:.1f} V {y+16-rad:.1f} Q {x0+w:.1f} {y+16} '
                         f'{x0+w-rad:.1f} {y+16} H {x0} Z" fill="{col}" opacity="0.9"/>')
            else:
                # a true zero still needs a mark, or the row reads as missing
                s.append(f'<circle cx="{x0+2}" cy="{y+9.5}" r="2" fill="{col}" opacity="0.55"/>')
            s.append(text(x0 + max(w, 4) + 6, y + 13, f"{v:,.{nd}f}", 9.5))

    s.append(text(0, H - 10,
                  "three of the five parsers use no GPU at all, shown as a dot rather than "
                  "a bar", 9.5))
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------- fig 2
def fig_fragment_heatmap(r):
    """The 5x4 matrix, twice: chunks that start and end mid-sentence."""
    panels = [("mid_start_pct", "Starts mid-sentence"),
              ("mid_end_pct", "Ends mid-sentence")]
    cw, ch = 118, 40
    left, top = 150, 84
    W = left + cw * len(CHUNKERS) + 30
    H = top + ch * len(PARSERS) + 80
    out = []
    for field, label in panels:
        vals = [cell(r, p, c).get(field) for p in PARSERS for c in CHUNKERS
                if cell(r, p, c) and cell(r, p, c).get(field) is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        s = svg_open(W, H, label)
        s.append(text(0, 20, label + " — share of chunks, lower is better", 13, INK, weight="600"))
        s.append(text(0, 38, "darker is worse; DEGENERATE cells are marked", 10))
        for j, c in enumerate(CHUNKERS):
            s.append(text(left + j * cw + cw / 2, top - 10, c, 10, INK, "middle"))
        for i, p in enumerate(PARSERS):
            y = top + i * ch
            s.append(text(left - 10, y + ch / 2 + 4, p, 10, INK, "end"))
            for j, c in enumerate(CHUNKERS):
                x = left + j * cw
                d = cell(r, p, c)
                if not d or d.get(field) is None:
                    s.append(f'<rect x="{x}" y="{y}" width="{cw-3}" height="{ch-3}" '
                             f'fill="none" stroke="{BAD}" stroke-width="1"/>')
                    s.append(text(x + cw / 2 - 1, y + ch / 2 + 4, "NOT RUN", 9, BAD, "middle"))
                    continue
                v = d[field]
                s.append(f'<rect x="{x}" y="{y}" width="{cw-3}" height="{ch-3}" rx="3" '
                         f'fill="{ramp_for(v, lo, hi)}"/>')
                dark = v > lo + 0.6 * (hi - lo)
                s.append(text(x + cw / 2 - 1, y + ch / 2 + 1, f"{100*v:.1f}%", 11,
                              "#ffffff" if dark else "#1d3050", "middle", "600"))
                if d.get("degenerate"):
                    s.append(text(x + cw / 2 - 1, y + ch / 2 + 13, "DEGENERATE", 7,
                                  "#ffffff" if dark else BAD, "middle"))
        s.append("</svg>")
        out.append((field, "\n".join(s)))
    return out


# --------------------------------------------------------------- fig 3
def fig_overlap(r):
    """How much text each combination duplicates between neighbours.

    The percentage sits inside the bar once the bar is long enough to hold it,
    and the median-overlap figure has a fixed column of its own. Trailing both
    labels off the bar end clipped them on every row near 100%.
    """
    rows = [(p, c, cell(r, p, c)) for p in PARSERS for c in CHUNKERS]
    rows = [(p, c, d) for p, c, d in rows if d and d.get("ov_zero_pct") is not None]
    if not rows:
        return None

    W = 880
    bar_h, gap = 14, 5
    top = 74
    H = top + len(rows) * (bar_h + gap) + 44
    x0, x1 = 250, 630
    col_med = 700

    s = svg_open(W, H, "Adjacent chunk pairs sharing no text")
    s.append(text(0, 20, "Adjacent chunk pairs sharing no text", 13, INK, weight="600"))
    s.append(text(0, 38, "longer is better: overlap costs embedding work and returns the same "
                         "passage twice", 10))
    for t in (0, 25, 50, 75, 100):
        gx = x0 + (x1 - x0) * t / 100
        s.append(f'<line x1="{gx:.0f}" y1="{top-8}" x2="{gx:.0f}" y2="{H-34}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.18"/>')
        s.append(text(gx, top - 14, f"{t}%", 9, INK, "middle"))
    s.append(text(col_med, top - 14, "median overlap", 9, INK))

    y = top
    for p, c, d in rows:
        v = d["ov_zero_pct"] * 100
        w = (x1 - x0) * v / 100
        col = PARSER_COLOUR.get(p, INK)
        s.append(text(x0 - 10, y + bar_h - 3, f"{p} · {c}", 9.5, INK, "end"))
        s.append(f'<rect x="{x0}" y="{y}" width="{max(w,1):.1f}" height="{bar_h}" rx="3" '
                 f'fill="{col}" opacity="0.9"/>')
        if w > 52:
            s.append(text(x0 + w - 6, y + bar_h - 3.5, f"{v:.1f}%", 9.5, "#ffffff", "end",
                          "600"))
        else:
            s.append(text(x0 + w + 6, y + bar_h - 3.5, f"{v:.1f}%", 9.5))
        med = d.get("ov_median_chars", 0)
        s.append(text(col_med, y + bar_h - 3.5, f"{med} ch", 9.5))
        y += bar_h + gap

    s.append(text(0, H - 10, "median overlap is over the pairs that do overlap; the strategies "
                             "near 100% barely ever repeat text", 9.5))
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------- fig 4
def fig_size_ranges(r):
    """Min to max chunk size, with the median marked.

    The axis is clamped: one cell reaches five figures and, drawn to true
    scale, squeezes every other row into the left tenth of the panel. Rows that
    run past the clamp are drawn to the edge with a chevron and their real
    maximum printed, so the outlier is visible without hiding the rest.
    """
    rows = [(p, c, cell(r, p, c)) for p in PARSERS for c in CHUNKERS]
    rows = [(p, c, d) for p, c, d in rows if d and d.get("median_chars")]
    if not rows:
        return None

    true_hi = max(d["max_chars"] for _, _, d in rows)
    window = 2600                                  # bge-base holds ~512 tokens
    # keep the readable range: the bulk of the data plus a margin past the window
    clamp = min(true_hi, max(4000, int(1.15 * sorted(d["max_chars"] for _, _, d in rows)[
        int(0.75 * len(rows))])))

    W = 820
    step = 21
    top = 92
    H = top + len(rows) * step + 40
    x0, x1 = 250, 690

    def sx(v):
        return x0 + (x1 - x0) * min(v, clamp) / clamp

    s = svg_open(W, H, "Chunk size ranges")
    s.append(text(0, 20, "Chunk size: smallest to largest, dot at the median", 13, INK,
                  weight="600"))
    s.append(text(0, 38, f"axis clamped at {clamp:,} characters; a chevron means the row runs "
                         f"past it, with its real maximum printed", 10))

    for i in range(6):
        v = clamp * i / 5
        gx = x0 + (x1 - x0) * i / 5
        s.append(f'<line x1="{gx:.0f}" y1="{top-8}" x2="{gx:.0f}" y2="{H-26}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.18"/>')
        s.append(text(gx, top - 14, f"{v:,.0f}", 9, INK, "middle"))

    wx = sx(window)
    s.append(f'<line x1="{wx:.0f}" y1="{top-8}" x2="{wx:.0f}" y2="{H-26}" stroke="{BAD}" '
             f'stroke-width="1.5" stroke-dasharray="4 3" opacity="0.85"/>')
    s.append(text(wx, top - 28, "embedder window", 9, BAD, "middle"))

    y = top + 6
    for p, c, d in rows:
        col = PARSER_COLOUR.get(p, INK)
        s.append(text(x0 - 10, y + 4, f"{p} · {c}", 9.5, INK, "end"))
        ex = sx(d["max_chars"])
        s.append(f'<line x1="{sx(d["min_chars"]):.1f}" y1="{y}" x2="{ex:.1f}" y2="{y}" '
                 f'stroke="{col}" stroke-width="3" opacity="0.35" stroke-linecap="round"/>')
        s.append(f'<circle cx="{sx(d["median_chars"]):.1f}" cy="{y}" r="4.5" fill="{col}"/>')
        if d["max_chars"] > clamp:
            s.append(f'<path d="M{ex-1:.1f} {y-5} l6 5 l-6 5" fill="none" stroke="{col}" '
                     f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
            s.append(text(ex + 10, y + 4, f'{d["max_chars"]:,}', 9, BAD))
        else:
            s.append(text(ex + 7, y + 4, f'{d["max_chars"]:,}', 9))
        y += step

    s.append(text(0, H - 8, "left edge is the smallest chunk the cell produced; anything right "
                            "of the dashed line is truncated at embed time", 9.5))
    s.append("</svg>")
    return "\n".join(s)


# ============================================================ retrieval run
RAG = REPO / "evals" / "Reports" / "rag_results.json"
DATASET = REPO / "evals" / "questions" / "dataset.json"
PARSED = REPO / "evals" / "parsed"
RAG_PARSERS = ["oss_docling", "oss_pymupdf4llm", "recrystal"]
# grain_growth takes the blue that marks "ours" elsewhere; the baseline is grey
CHUNKER_COLOUR = {"fixed_token": "#9aa3ab", "recursive_char": "#1baf7a",
                  "semantic": "#e87ba4", "grain_growth": "#2a78d6"}
SHORT = {"oss_docling": "docling", "oss_pymupdf4llm": "pymupdf4llm", "recrystal": "recrystal"}


def rag_cell(R, p, c):
    v = R["cells"].get(f"{p}|{c}")
    return v if v and v.get("complete") else None


NEAR_MATCH = 0.9   # share of span words that must appear in one span-length window


def span_overlap(span_key, doc_key):
    """Best share of the span's words found in any window of the document of the
    same length, in text_key form. 1.0 for an exact substring.

    The questions were verified against PyMuPDF text, and two of the three
    parsers read words through PyMuPDF while Docling uses its own engine. A strict
    substring test therefore scores Docling down for symbol, spacing and
    line-number differences in text it did extract. Counting words in a sliding
    window tolerates those differences while still rejecting a missing passage.
    """
    from collections import Counter
    if span_key in doc_key:
        return 1.0
    s, d = span_key.split(), doc_key.split()
    n = len(s)
    if not n:
        return 0.0
    # a chunk shorter than the span is compared whole: the window cannot slide
    need, win = Counter(s), Counter(d[:n])
    hit = sum(min(c, win[w]) for w, c in need.items())
    best = hit
    for i in range(n, len(d)):
        out, inn = d[i - n], d[i]
        if out in need and win[out] <= need[out]:
            hit -= 1
        win[out] -= 1
        win[inn] += 1
        if inn in need and win[inn] <= need[inn]:
            hit += 1
        best = max(best, hit)
    return best / n


def answer_coverage(R, threshold=1.0):
    """qids whose answer span is absent from each parser's own output.

    threshold 1.0 is a strict substring match; NEAR_MATCH tolerates text-engine
    differences (see span_overlap)."""
    import sys
    sys.path.insert(0, str(REPO / "evals" / "scripts"))
    from build_questions import text_key
    qs = json.loads(DATASET.read_text())["questions"]
    cache, out = {}, {}
    for p in RAG_PARSERS:
        miss = set()
        for q in qs:
            key = (p, q["paper"])
            if key not in cache:
                f = PARSED / p / f"{q['paper']}.json"
                cache[key] = text_key(" ".join(b.get("text", "") for b in
                                               json.loads(f.read_text())["blocks"])) if f.exists() else ""
            if span_overlap(text_key(q["answer_span"]), cache[key]) < threshold:
                miss.add(q["qid"])
        out[p] = miss
    return out


def fig_rag_heatmap(R, field, title):
    cw, ch, left, top = 132, 46, 130, 92
    W, H = left + cw * len(CHUNKERS) + 20, top + ch * len(RAG_PARSERS) + 30
    vals = [rag_cell(R, p, c)[field] for p in RAG_PARSERS for c in CHUNKERS
            if rag_cell(R, p, c) and rag_cell(R, p, c).get(field) is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    s = svg_open(W, H, title)
    s.append(text(0, 20, title, 13, INK, weight="600"))
    s.append(text(0, 38, "darker is better; outlined cell is the best chunker for that parser", 10))
    for j, c in enumerate(CHUNKERS):
        s.append(text(left + j * cw + cw / 2, top - 12, c, 10, INK, "middle"))
    for i, p in enumerate(RAG_PARSERS):
        y = top + i * ch
        s.append(text(left - 10, y + ch / 2 + 4, SHORT[p], 10.5, INK, "end"))
        row = {c: rag_cell(R, p, c) for c in CHUNKERS}
        best = max((c for c in CHUNKERS if row[c] and row[c].get(field) is not None),
                   key=lambda c: row[c][field], default=None)
        for j, c in enumerate(CHUNKERS):
            x = left + j * cw
            d = row[c]
            if not d or d.get(field) is None:
                s.append(text(x + cw / 2, y + ch / 2 + 4, "NOT RUN", 9, BAD, "middle"))
                continue
            v = d[field]
            s.append(f'<rect x="{x}" y="{y}" width="{cw-4}" height="{ch-4}" rx="3" '
                     f'fill="{ramp_for(v, lo, hi)}"/>')
            if c == best:
                s.append(f'<rect x="{x+1.5}" y="{y+1.5}" width="{cw-7}" height="{ch-7}" rx="3" '
                         f'fill="none" stroke="#eda100" stroke-width="3"/>')
            dark = v > lo + 0.55 * (hi - lo)
            s.append(text(x + cw / 2 - 2, y + ch / 2 + 2, f"{v:.3f}", 12,
                          "#ffffff" if dark else "#1d3050", "middle", "600"))
    s.append("</svg>")
    return "\n".join(s)


def fig_dumbbell(rows, title, sub, a_name, b_name):
    """rows: (label, a, b, colour); a is the filled dot, b the ring."""
    x0, x1, col = 250, 640, 668
    top, step = 88, 24
    W, H = 830, top + len(rows) * step + 24
    s = svg_open(W, H, title)
    s.append(text(0, 20, title, 13, INK, weight="600"))
    s.append(text(0, 38, sub, 10))
    s.append(f'<circle cx="6" cy="55" r="5" fill="{INK}"/>')
    s.append(text(16, 59, a_name, 10))
    bx = 16 + len(a_name) * 6.2 + 26
    s.append(f'<circle cx="{bx:.0f}" cy="55" r="4.5" fill="none" stroke="{INK}" stroke-width="2"/>')
    s.append(text(bx + 10, 59, b_name, 10))
    for t in (0, .25, .5, .75, 1):
        gx = x0 + (x1 - x0) * t
        s.append(f'<line x1="{gx:.0f}" y1="{top-8}" x2="{gx:.0f}" y2="{H-16}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.18"/>')
        s.append(text(gx, top - 12, f"{t:.2f}", 9, INK, "middle"))
    y = top + 4
    for label, a, b, colour in rows:
        ax, bxx = x0 + (x1 - x0) * a, x0 + (x1 - x0) * b
        s.append(text(x0 - 10, y + 4, label, 9.5, INK, "end"))
        s.append(f'<line x1="{ax:.1f}" y1="{y}" x2="{bxx:.1f}" y2="{y}" stroke="{colour}" '
                 f'stroke-width="3" opacity="0.4"/>')
        s.append(f'<circle cx="{ax:.1f}" cy="{y}" r="5" fill="{colour}"/>')
        s.append(f'<circle cx="{bxx:.1f}" cy="{y}" r="4.5" fill="none" stroke="{colour}" stroke-width="2"/>')
        s.append(text(col, y + 4, f"{a:.2f} → {b:.2f}  ({b - a:+.2f})", 9.5))
        y += step
    s.append("</svg>")
    return "\n".join(s)


def fig_k_vs_budget(R):
    rows = [(f"{SHORT[p]} · {c}", rag_cell(R, p, c)["span_hit@5"],
             rag_cell(R, p, c)["span_hit@4000ch"], CHUNKER_COLOUR[c])
            for p in RAG_PARSERS for c in CHUNKERS if rag_cell(R, p, c)]
    return fig_dumbbell(rows, "Fixed k flatters large chunks",
                        "the same answer-retrieval measure at k=5 and at an equal 4,000-character budget",
                        "at k=5", "at 4,000 characters")


def fig_budget_curves(R):
    B = R["config"]["budgets"]
    pw, ph, gap, left, top = 220, 190, 34, 46, 100
    W, H = left + len(RAG_PARSERS) * (pw + gap), top + ph + 44
    lo, hi = 0.2, 0.9
    s = svg_open(W, H, "Answer found within a character budget")
    s.append(text(0, 20, "Answer found within an equal character budget", 13, INK, weight="600"))
    s.append(text(0, 38, "share of questions whose verbatim answer is inside the first N characters retrieved", 10))
    lx = 0
    for c in CHUNKERS:
        s.append(f'<line x1="{lx}" y1="58" x2="{lx+16}" y2="58" stroke="{CHUNKER_COLOUR[c]}" stroke-width="3"/>')
        s.append(text(lx + 22, 62, c, 10))
        lx += 44 + len(c) * 6.3
    for i, p in enumerate(RAG_PARSERS):
        x0 = left + i * (pw + gap)
        sy = lambda v: top + ph - (v - lo) / (hi - lo) * ph
        sx = lambda j: x0 + j / (len(B) - 1) * pw
        s.append(text(x0, top - 12, SHORT[p], 11, INK, weight="600"))
        for t in (0.2, 0.4, 0.6, 0.8):
            s.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x0+pw}" y2="{sy(t):.1f}" '
                     f'stroke="{GRID}" stroke-width="1" opacity="0.18"/>')
            if i == 0:
                s.append(text(x0 - 6, sy(t) + 3, f"{t:.1f}", 9, INK, "end"))
        for j, b in enumerate(B):
            s.append(text(sx(j), top + ph + 16, f"{b // 1000}k ch", 9, INK, "middle"))
        for c in CHUNKERS:
            d = rag_cell(R, p, c)
            if not d:
                continue
            pts = [(sx(j), sy(d[f"span_hit@{b}ch"])) for j, b in enumerate(B)]
            s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                     f'" fill="none" stroke="{CHUNKER_COLOUR[c]}" stroke-width="2.5"/>')
            for x, y in pts:
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{CHUNKER_COLOUR[c]}"/>')
    s.append("</svg>")
    return "\n".join(s)


def fig_home_advantage(R):
    src = {q["qid"]: q["generated_from"] for q in json.loads(DATASET.read_text())["questions"]}
    rows = []
    for p in RAG_PARSERS:
        for c in CHUNKERS:
            d = rag_cell(R, p, c)
            if not d:
                continue
            own = [x["span_hit@5"] for x in d["_rows"] if src.get(x["qid"]) == p and "span_hit@5" in x]
            oth = [x["span_hit@5"] for x in d["_rows"]
                   if src.get(x["qid"]) not in (None, p) and "span_hit@5" in x]
            if own and oth:
                rows.append((f"{SHORT[p]} · {c}", sum(oth) / len(oth), sum(own) / len(own), PARSER_COLOUR[p]))
    return fig_dumbbell(rows, "Home advantage: questions written from a parser's own output",
                        "answer retrieved in the top 5; a long line means the parser does better on its own questions",
                        "other parsers' questions", "its own questions")


def fig_shape_vs_retrieval(r, R):
    """Does a chunk-shape measure predict retrieval? One point per cell."""
    pts = []
    for p in RAG_PARSERS:
        for c in CHUNKERS:
            a, d = cell(r, p, c), rag_cell(R, p, c)
            if a and d and a.get("mid_start_pct") is not None:
                pts.append((p, c, a["mid_start_pct"], d["span_hit@4000ch"]))
    if not pts:
        return None
    W, H = 720, 440
    x0, x1, y0, y1 = 80, 520, 80, 370
    xmax = max(q[2] for q in pts) * 1.1 or 1.0
    ylo, yhi = min(q[3] for q in pts) - 0.04, max(q[3] for q in pts) + 0.04
    sx = lambda v: x0 + v / xmax * (x1 - x0)
    sy = lambda v: y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)
    s = svg_open(W, H, "Chunk fragmentation against retrieval")
    s.append(text(0, 20, "Does chunk fragmentation predict retrieval?", 13, INK, weight="600"))
    s.append(text(0, 38, "one point per parser x chunker cell; the letter marks the parser", 10))
    for i in range(5):
        vx, vy = xmax * i / 4, ylo + (yhi - ylo) * i / 4
        s.append(f'<line x1="{sx(vx):.1f}" y1="{y0}" x2="{sx(vx):.1f}" y2="{y1}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.18"/>')
        s.append(f'<line x1="{x0}" y1="{sy(vy):.1f}" x2="{x1}" y2="{sy(vy):.1f}" '
                 f'stroke="{GRID}" stroke-width="1" opacity="0.18"/>')
        s.append(text(sx(vx), y1 + 16, f"{100 * vx:.0f}%", 9, INK, "middle"))
        s.append(text(x0 - 8, sy(vy) + 3, f"{vy:.2f}", 9, INK, "end"))
    s.append(text((x0 + x1) / 2, y1 + 36, "chunks starting mid-sentence (15-paper shape run)", 10, INK, "middle"))
    mid = (y0 + y1) / 2
    s.append(text(18, mid, "answer within 4,000 ch", 10, INK, "middle")
             .replace("<text ", f'<text transform="rotate(-90 18 {mid:.0f})" '))
    offset = {"oss_docling": (9, 4, "start"), "oss_pymupdf4llm": (-9, 4, "end"), "recrystal": (0, -10, "middle")}
    for p, c, xv, yv in pts:
        cx, cy = sx(xv), sy(yv)
        dx, dy, anchor = offset[p]
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{CHUNKER_COLOUR[c]}" opacity="0.9"/>')
        s.append(text(cx + dx, cy + dy, SHORT[p][0].upper(), 10, INK, anchor, "600"))
    ly = y0 + 6
    for c in CHUNKERS:
        s.append(f'<circle cx="{x1 + 44}" cy="{ly}" r="5" fill="{CHUNKER_COLOUR[c]}"/>')
        s.append(text(x1 + 56, ly + 4, c, 10))
        ly += 20
    ly += 12
    for p in RAG_PARSERS:
        s.append(text(x1 + 40, ly + 4, f"{SHORT[p][0].upper()}  {SHORT[p]}", 10))
        ly += 18
    s.append("</svg>")
    return "\n".join(s)


def fig_coverage(R, exact, near):
    n = len(json.loads(DATASET.read_text())["questions"])
    W, top, step = 720, 92, 46
    H = top + len(RAG_PARSERS) * step + 44
    x0, x1 = 140, 540
    hi = (max(len(v) for v in exact.values()) * 1.25) or 1
    rescued = sum(1 for p in RAG_PARSERS for c in CHUNKERS if rag_cell(R, p, c)
                  for x in rag_cell(R, p, c)["_rows"] if x["qid"] in near[p] and x.get("span_hit@10"))
    s = svg_open(W, H, "Answers lost before retrieval")
    s.append(text(0, 20, "Answers the parser lost before retrieval began", 13, INK, weight="600"))
    s.append(text(0, 38, f"questions whose answer span is missing from the parser's own output, out of {n}", 10))
    s.append(f'<rect x="0" y="52" width="12" height="9" rx="2" fill="{INK}" opacity="0.35"/>')
    s.append(text(18, 60, "exact text match", 10))
    s.append(f'<rect x="130" y="52" width="12" height="9" rx="2" fill="{INK}"/>')
    s.append(text(148, 60, f"near match, {int(NEAR_MATCH * 100)}% of span words: really missing", 10))
    y = top
    for p in RAG_PARSERS:
        s.append(text(x0 - 10, y + 17, SHORT[p], 10.5, INK, "end"))
        for k, h, dy, op in ((len(exact[p]), 14, 0, 0.35), (len(near[p]), 14, 16, 1.0)):
            w = (x1 - x0) * k / hi
            s.append(f'<rect x="{x0}" y="{y + dy}" width="{max(w, 2):.1f}" height="{h}" rx="3" '
                     f'fill="{PARSER_COLOUR[p]}" opacity="{op}"/>')
            s.append(text(x0 + w + 8, y + dy + 11, str(k), 10))
        y += step
    s.append(text(0, H - 12, "none of the really missing answers was retrieved by any chunker"
                  if rescued == 0 else f"{rescued} retrievals recovered an answer missing from the parse", 9.5))
    s.append("</svg>")
    return "\n".join(s)


def main():
    r = load()
    ASSETS.mkdir(parents=True, exist_ok=True)
    written = []

    def put(name, svg):
        if svg:
            (ASSETS / name).write_text(svg, encoding="utf-8")
            written.append(name)

    put("fig1_parser_tradeoff.svg", fig_parser_tradeoff(r))
    put("fig3_overlap.svg", fig_overlap(r))
    put("fig4_size_ranges.svg", fig_size_ranges(r))
    for field, svg in fig_fragment_heatmap(r):
        put(f"fig2_{field}.svg", svg)

    if RAG.exists():
        R = json.loads(RAG.read_text())
        put("fig5_rag_span4k.svg", fig_rag_heatmap(R, "span_hit@4000ch", "Answer retrieved within 4,000 characters"))
        put("fig5_rag_correctness.svg", fig_rag_heatmap(R, "correctness", "Answer correctness, judged against the reference"))
        put("fig6_k_vs_budget.svg", fig_k_vs_budget(R))
        put("fig7_budget_curves.svg", fig_budget_curves(R))
        put("fig8_home_advantage.svg", fig_home_advantage(R))
        put("fig9_shape_vs_retrieval.svg", fig_shape_vs_retrieval(r, R))
        put("fig10_coverage.svg", fig_coverage(R, answer_coverage(R), answer_coverage(R, NEAR_MATCH)))

    for n in sorted(written):
        print(f"  {n}")
    print(f"{len(written)} figures -> {ASSETS}")


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    if "--round1" in sys.argv[1:]:
        # Round 2 took over questions/dataset.json. The round-1 figures must be
        # drawn from round 1's own question set, or the ones that read
        # `generated_from` (fig8, fig10) come out empty.
        DATASET = REPO / "evals" / "questions" / "round1" / "dataset.json"
    main()
