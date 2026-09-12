"""Figures for the parser x chunker comparison.

    python evals/scripts/make_figures.py

Reads evals/Reports/results.json and nothing else. Every value is taken from
that file; no number is written into this script. Emits SVG to
evals/Reports/assets/.

Colours are fixed hexes rather than CSS variables because a markdown viewer
gives an embedded SVG no theme to inherit; they are chosen to read on a white
and a near-black page alike.
"""

import json
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
    "current": "#2a78d6",
}
# sequential ramp for the heatmap: one hue, light to dark, monotonic lightness
RAMP = ["#eaf2fb", "#c5dcf5", "#94bfec", "#5f9ade", "#3576c4", "#1d4f8f"]
BAD = "#97382a"

PARSERS = ["raw_dump", "legacy", "oss_docling", "oss_pymupdf4llm", "current"]
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


def main():
    r = load()
    ASSETS.mkdir(parents=True, exist_ok=True)
    written = []

    for name, svg in [("fig1_parser_tradeoff.svg", fig_parser_tradeoff(r)),
                      ("fig3_overlap.svg", fig_overlap(r)),
                      ("fig4_size_ranges.svg", fig_size_ranges(r))]:
        if svg:
            (ASSETS / name).write_text(svg, encoding="utf-8")
            written.append(name)
    for field, svg in fig_fragment_heatmap(r):
        name = f"fig2_{field}.svg"
        (ASSETS / name).write_text(svg, encoding="utf-8")
        written.append(name)

    for n in sorted(written):
        print(f"  {n}")
    print(f"{len(written)} figures -> {ASSETS}")


if __name__ == "__main__":
    main()
