"""Compare PDF backends (PyMuPDF vs pypdfium2) before switching away from
the AGPL dependency.

    python scripts/pdf_backend_compare.py [PDF ...] [--pages 10]

Measures the three jobs the pipeline actually asks of a PDF library:

1. **Rendering** a page to the RGB array the layout model consumes —
   wall time, and whether the two backends produce the *same pixels*
   (a different rasterizer means the model sees a different image).
2. **Word extraction** with boxes — wall time, word count, and box
   agreement against PyMuPDF as the reference.
3. **Detections** — running the real layout model on each backend's render
   and comparing the regions it finds, which is what actually reaches the
   rest of the pipeline.

pypdfium2 exposes characters, not words, so words are rebuilt here by the
same rule PyMuPDF uses (split on whitespace, union the character boxes).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "parse_manager"))

from common.paths import PDF_DIR  # noqa: E402

RENDER_DPI = 150


# --------------------------------------------------------------- backends
def mupdf_render(path, n):
    import fitz
    doc = fitz.open(path)
    out = []
    for i in range(min(n, len(doc))):
        pix = doc[i].get_pixmap(dpi=RENDER_DPI, colorspace=fitz.csRGB, alpha=False)
        out.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
    doc.close()
    return out


def pdfium_render(path, n):
    """pdfium ceils the pixel size where PyMuPDF rounds (1651 vs 1650 rows for
    a 792pt page at 150 DPI), so the result is trimmed to PyMuPDF's size."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    out = []
    for i in range(min(n, len(doc))):
        page = doc[i]
        w_pts, h_pts = page.get_size()
        want_w, want_h = round(w_pts * RENDER_DPI / 72), round(h_pts * RENDER_DPI / 72)
        arr = np.asarray(page.render(scale=RENDER_DPI / 72).to_pil().convert("RGB"))
        out.append(arr[:want_h, :want_w])
    doc.close()
    return out


def mupdf_words(path, n):
    """[(x0, y0, x1, y1, text)] per page, top-left origin, in points."""
    import fitz
    doc = fitz.open(path)
    pages = []
    for i in range(min(n, len(doc))):
        pages.append([[w[0], w[1], w[2], w[3], w[4]] for w in doc[i].get_text("words")])
    doc.close()
    return pages


def pdfium_words(path, n, loose=True):
    """Same shape, rebuilt from character boxes (pdfium has no word API).

    pdfium reports boxes bottom-left origin, so y is flipped to match
    PyMuPDF. Words break on whitespace, exactly as PyMuPDF's `words` mode.

    `loose=True` asks pdfium for the font-metric box rather than the glyph
    ink box. That matters: the pipeline derives its line-grouping tolerance
    and column-gutter threshold from the *median word height*, and ink boxes
    are both shorter and letter-dependent ("A" 11.3pt vs "Agent" 15.3pt),
    which would silently retune both heuristics.
    """
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    pages = []
    for i in range(min(n, len(doc))):
        page = doc[i]
        height = page.get_size()[1]
        tp = page.get_textpage()
        text = tp.get_text_range()
        words, current, box = [], [], None
        for idx, ch in enumerate(text):
            if ch.isspace():
                if current:
                    words.append(box + ["".join(current)])
                    current, box = [], None
                continue
            try:
                left, bottom, right, top = tp.get_charbox(idx, loose=loose)
            except Exception:
                continue
            if left == right == bottom == top == 0:      # unpositioned glyph
                continue
            x0, y0, x1, y1 = left, height - top, right, height - bottom
            box = [x0, y0, x1, y1] if box is None else [
                min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)]
            current.append(ch)
        if current and box:
            words.append(box + ["".join(current)])
        pages.append(words)
    doc.close()
    return pages


# --------------------------------------------------------------- helpers
def timed(fn, *a, repeat=3):
    fn(*a)                                   # warm
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        result = fn(*a)
        times.append(time.perf_counter() - t)
    return result, min(times)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def compare_words(ref_pages, new_pages):
    """Match words by text and position; report agreement and box error."""
    total = matched = 0
    ious, text_only_ref, text_only_new = [], 0, 0
    for ref, new in zip(ref_pages, new_pages):
        total += len(ref)
        pool = {}
        for w in new:
            pool.setdefault(w[4], []).append(w)
        for w in ref:
            cands = pool.get(w[4])
            if not cands:
                text_only_ref += 1
                continue
            best = max(cands, key=lambda c: iou(w[:4], c[:4]))
            score = iou(w[:4], best[:4])
            if score > 0.5:
                matched += 1
                ious.append(score)
                cands.remove(best)
            else:
                text_only_ref += 1
        text_only_new += sum(len(v) for v in pool.values())
    return {"ref_words": total, "new_words": sum(len(p) for p in new_pages),
            "matched": matched, "mean_iou": statistics.mean(ious) if ious else 0,
            "min_iou": min(ious) if ious else 0,
            "unmatched_ref": text_only_ref, "unmatched_new": text_only_new}


def detect(images):
    from layout_detector import LayoutDetector
    global _DET
    try:
        _DET
    except NameError:
        _DET = LayoutDetector()
    out = []
    for i in range(0, len(images), 4):
        out += _DET._predict(images[i:i + 4])
    return out


def compare_detections(a, b):
    total = matched = 0
    ious = []
    for pa, pb in zip(a, b):
        total += len(pa)
        pool = list(pb)
        for d in pa:
            cands = [c for c in pool if c["label"] == d["label"]]
            if not cands:
                continue
            best = max(cands, key=lambda c: iou(d["bbox"], c["bbox"]))
            if iou(d["bbox"], best["bbox"]) > 0.9:
                matched += 1
                ious.append(iou(d["bbox"], best["bbox"]))
                pool.remove(best)
    return {"ref_regions": total, "new_regions": sum(len(p) for p in b), "matched": matched,
            "mean_iou": statistics.mean(ious) if ious else 0}


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--pages", type=int, default=10)
    args = ap.parse_args()

    names = args.pdfs or [p.name for p in sorted(PDF_DIR.glob("*.pdf"))[:3]]
    agg = {"render_mu": [], "render_pd": [], "words_mu": [], "words_pd": []}

    for name in names:
        pdf = Path(name) if Path(name).exists() else PDF_DIR / Path(name).name
        if not pdf.exists():
            print(f"skip: {name}")
            continue
        print(f"\n=== {pdf.name} (first {args.pages} pages)")

        mu_imgs, t_mu = timed(mupdf_render, str(pdf), args.pages)
        pd_imgs, t_pd = timed(pdfium_render, str(pdf), args.pages)
        agg["render_mu"].append(t_mu); agg["render_pd"].append(t_pd)
        n = len(mu_imgs)
        print(f"  render     PyMuPDF {t_mu:6.3f}s   pypdfium2 {t_pd:6.3f}s   "
              f"({n/t_mu:5.1f} vs {n/t_pd:5.1f} pages/s)")

        same_shape = [a.shape == b.shape for a, b in zip(mu_imgs, pd_imgs)]
        if all(same_shape):
            diffs = [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                     for a, b in zip(mu_imgs, pd_imgs)]
            pct = [float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > 8).mean() * 100)
                   for a, b in zip(mu_imgs, pd_imgs)]
            print(f"             pixels: mean |diff| {statistics.mean(diffs):.2f}/255, "
                  f"{statistics.mean(pct):.2f}% of pixels differ by >8")
        else:
            print(f"             SHAPE MISMATCH {mu_imgs[0].shape} vs {pd_imgs[0].shape}")

        mu_w, tw_mu = timed(mupdf_words, str(pdf), args.pages)
        pd_w, tw_pd = timed(pdfium_words, str(pdf), args.pages)
        agg["words_mu"].append(tw_mu); agg["words_pd"].append(tw_pd)
        print(f"  words      PyMuPDF {tw_mu:6.3f}s   pypdfium2 {tw_pd:6.3f}s   "
              f"({tw_pd/tw_mu:.1f}x)")
        w = compare_words(mu_w, pd_w)
        print(f"             {w['matched']}/{w['ref_words']} words matched "
              f"(pdfium found {w['new_words']}), mean IoU {w['mean_iou']:.4f}, "
              f"min {w['min_iou']:.3f}")
        if w["unmatched_ref"] or w["unmatched_new"]:
            print(f"             unmatched: {w['unmatched_ref']} PyMuPDF-only, "
                  f"{w['unmatched_new']} pdfium-only")

        if all(same_shape):
            d = compare_detections(detect(mu_imgs), detect(pd_imgs))
            print(f"  detections {d['matched']}/{d['ref_regions']} regions identical "
                  f"(pdfium render gave {d['new_regions']}), mean IoU {d['mean_iou']:.4f}")

    print(f"\n{'=' * 70}\nTOTALS across {len(agg['render_mu'])} papers")
    print(f"  render : PyMuPDF {sum(agg['render_mu']):.2f}s | pypdfium2 {sum(agg['render_pd']):.2f}s "
          f"-> {sum(agg['render_pd'])/sum(agg['render_mu']):.2f}x")
    print(f"  words  : PyMuPDF {sum(agg['words_mu']):.2f}s | pypdfium2 {sum(agg['words_pd']):.2f}s "
          f"-> {sum(agg['words_pd'])/sum(agg['words_mu']):.2f}x")


if __name__ == "__main__":
    main()
