"""Figures for the round-2 report, from round2_analysis.json.

    python evals/scripts/make_round2_figures.py

Writes evals/Reports/assets/fig_r2_*.svg. Reuses the round-1 drawing helpers so
both rounds look the same in one report.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "evals" / "scripts"))

from make_figures import CHUNKER_COLOUR, INK, PARSER_COLOUR, SHORT, svg_open, text  # noqa: E402

R = REPO / "evals" / "Reports"
ASSETS = R / "assets"
PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "grain_growth"]
PRIMARY = "span_hit_near@4000ch"


def fig_cells(A):
    """Nine cells: the primary retrieval metric, grouped by parser."""
    left, top, bw, gap = 150, 74, 26, 16
    W = 760
    H = top + len(PARSERS) * (len(CHUNKERS) * (bw + 4) + gap) + 30
    s = svg_open(W, H, "Answer retrieved within 4,000 characters")
    s.append(text(0, 20, "Answer found within 4,000 characters retrieved", 13, INK, weight="600"))
    s.append(text(0, 38, "near match; every cell answered the same 400 questions", 10))
    for i, c in enumerate(CHUNKERS):
        s.append(f'<rect x="{300 + i * 150}" y="50" width="11" height="9" rx="2" fill="{CHUNKER_COLOUR[c]}"/>')
        s.append(text(316 + i * 150, 58, c, 9.5))
    x0, x1 = left, W - 90
    y = top
    for p in PARSERS:
        s.append(text(left - 12, y + 34, SHORT[p], 11, INK, "end"))
        for i, c in enumerate(CHUNKERS):
            v = A["cells"][f"{p}|{c}"][PRIMARY]
            w = (x1 - x0) * v
            yy = y + i * (bw + 4)
            s.append(f'<rect x="{x0}" y="{yy}" width="{w:.1f}" height="{bw}" rx="3" fill="{CHUNKER_COLOUR[c]}"/>')
            s.append(text(x0 + w + 8, yy + bw / 2 + 4, f"{v:.3f}", 10.5, INK))
        y += 3 * (bw + 4) + gap
    s.append("</svg>")
    return "\n".join(s)


def fig_parsers(A):
    """Every parser pair under every chunker, with bootstrap intervals."""
    fam = A["parser"]["retrieval"][PRIMARY]
    rows = [(k, v) for k, v in fam.items() if not v.get("pending")]
    W, top, step = 880, 86, 30
    H = top + len(rows) * step + 44
    # zero sits far enough right, and the scale is small enough, that the widest
    # interval still clears the longest label
    zero, scale = 545, 900
    short_chunk = {"fixed_token": "fixed", "recursive_char": "recursive", "grain_growth": "grain"}

    def label(name):
        pair, chunk = name.split(" [")
        a, b = pair.split(" vs ")
        return f"{SHORT.get(a, a)} vs {SHORT.get(b, b)} [{short_chunk.get(chunk[:-1], chunk[:-1])}]"
    s = svg_open(W, H, "Parser differences")
    s.append(text(0, 20, "Parser pairs: difference in answer retrieval", 13, INK, weight="600"))
    s.append(text(0, 38, "positive favours the first parser; bars are 95% intervals over papers; "
                         "filled = significant after Holm", 10))
    s.append(f'<line x1="{zero}" y1="{top - 12}" x2="{zero}" y2="{H - 34}" stroke="{INK}" '
             f'stroke-width="1" opacity="0.35"/>')
    s.append(text(zero, H - 18, "no difference", 9, INK, "middle"))
    y = top
    for name, v in sorted(rows, key=lambda x: x[1]["diff"]):
        lo, hi = v["ci95"]
        colour = PARSER_COLOUR["current"] if v.get("holm") else INK
        s.append(text(zero - 250, y + 4, label(name), 10, INK, "end"))
        s.append(f'<line x1="{zero + lo * scale:.1f}" y1="{y}" x2="{zero + hi * scale:.1f}" y2="{y}" '
                 f'stroke="{colour}" stroke-width="2" opacity="{1 if v.get("holm") else 0.45}"/>')
        s.append(f'<circle cx="{zero + v["diff"] * scale:.1f}" cy="{y}" r="4.5" fill="{colour}" '
                 f'opacity="{1 if v.get("holm") else 0.45}"/>')
        s.append(text(W - 78, y + 4, f"{v['diff']:+.3f}", 10, INK))
        y += step
    s.append("</svg>")
    return "\n".join(s)


def fig_parse_loss(A):
    """Answer spans missing from each parser's own output."""
    pl = A["parse_loss"]["per_parser"]
    W, top, step = 700, 92, 46
    H = top + len(PARSERS) * step + 36
    x0, x1 = 150, 540
    hi = max(v["missing_exact"] for v in pl.values()) * 1.25 or 1
    s = svg_open(W, H, "Answers lost in parsing")
    s.append(text(0, 20, "Answer spans missing from the parser's own output", 13, INK, weight="600"))
    s.append(text(0, 38, f"out of {pl[PARSERS[0]]['of']} questions; no chunker can recover text the parse lost", 10))
    s.append(f'<rect x="0" y="52" width="12" height="9" rx="2" fill="{INK}" opacity="0.35"/>')
    s.append(text(18, 60, "exact text match", 10))
    s.append(f'<rect x="150" y="52" width="12" height="9" rx="2" fill="{INK}"/>')
    s.append(text(168, 60, "near match: really missing", 10))
    y = top
    for p in PARSERS:
        s.append(text(x0 - 10, y + 17, SHORT[p], 10.5, INK, "end"))
        for k, dy, op in (("missing_exact", 0, 0.35), ("missing_near", 16, 1.0)):
            n = pl[p][k]
            w = (x1 - x0) * n / hi
            s.append(f'<rect x="{x0}" y="{y + dy}" width="{max(w, 2):.1f}" height="14" rx="3" '
                     f'fill="{PARSER_COLOUR[p]}" opacity="{op}"/>')
            s.append(text(x0 + w + 8, y + dy + 11, str(n), 10))
        y += step
    s.append("</svg>")
    return "\n".join(s)


def fig_shape(A):
    """Chunk shape against retrieval, both measured on the same 514 papers."""
    shape = json.loads((R / "shape_round2.json").read_text())["cells"]
    W, H, left, bottom, top, right = 820, 380, 92, 300, 80, 640
    xs = [shape[k]["mid_end_pct"] for k in A["cells"]]
    ys = [A["cells"][k][PRIMARY] for k in A["cells"]]
    xlo, xhi, ylo, yhi = 0, min(1.0, max(xs) * 1.1), min(ys) * 0.94, max(ys) * 1.04
    sx = lambda v: left + (right - left) * (v - xlo) / (xhi - xlo)       # noqa: E731
    sy = lambda v: bottom - (bottom - top) * (v - ylo) / (yhi - ylo)     # noqa: E731
    s = svg_open(W, H, "Chunk shape against retrieval")
    s.append(text(0, 20, "Chunks ending mid-sentence against answer retrieval", 13, INK, weight="600"))
    s.append(text(0, 38, "one point per cell, both measured on the same 514 papers; "
                         f"Spearman {A['shape_vs_outcome'].get('mid_end_pct vs ' + PRIMARY, float('nan')):+.2f}", 10))
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx, gy = sx(xlo + frac * (xhi - xlo)), sy(ylo + frac * (yhi - ylo))
        s.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="{INK}" '
                 f'stroke-width="0.6" opacity="0.18"/>')
        s.append(text(left - 8, gy + 4, f"{ylo + frac * (yhi - ylo):.2f}", 9, INK, "end"))
        s.append(text(gx, bottom + 18, f"{100 * (xlo + frac * (xhi - xlo)):.0f}%", 9, INK, "middle"))
    s.append(text((left + right) / 2, bottom + 40, "chunks ending mid-sentence", 10, INK, "middle"))
    for k in A["cells"]:
        p, c = k.split("|")
        cx, cy = sx(shape[k]["mid_end_pct"]), sy(A["cells"][k][PRIMARY])
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{CHUNKER_COLOUR[c]}" opacity="0.9"/>')
        s.append(text(cx + 11, cy + 4, SHORT[p], 9, INK))
    for i, c in enumerate(CHUNKERS):
        s.append(f'<circle cx="{right + 40}" cy="{top + 10 + i * 20}" r="6" fill="{CHUNKER_COLOUR[c]}"/>')
        s.append(text(right + 52, top + 14 + i * 20, c, 9.5))
    s.append("</svg>")
    return "\n".join(s)


def main():
    A = json.loads((R / "round2_analysis.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in (("fig_r2_cells.svg", fig_cells(A)),
                      ("fig_r2_parsers.svg", fig_parsers(A)),
                      ("fig_r2_parse_loss.svg", fig_parse_loss(A)),
                      ("fig_r2_shape.svg", fig_shape(A))):
        (ASSETS / name).write_text(svg, encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
