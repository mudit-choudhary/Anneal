"""Did each paper actually get parsed in full?

The registry tracks which *stage* a paper reached, never whether that stage
covered the whole document. A parse that died half way still writes its
partial output, still reports `processed`, then `embedded`, and the corpus
silently contains a paper that can answer questions about page one and
nothing else. Two papers were in that state for days before a benchmark
comparing parsed text against the raw PDFs turned them up.

This module is the check that would have caught them at the time. It is
cheap (a page count and a JSON read per paper) so it can run at the end of
every ingest, and it powers the repair action in the UI.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.paths import PARSED_DIR, PDF_DIR, PROCESSED_DIR

# A parse that reached fewer pages than the PDF has is broken, full stop.
# Everything else here is a heuristic, so the thresholds are deliberately
# loose — this must not cry wolf on a genuinely short or figure-heavy paper.
MIN_BLOCKS = 3                 # fewer than this is not a parsed paper
MIN_CHARS_PER_PAGE = 200       # a text page yields thousands; 200 is a floor


def pdf_page_count(stem: str) -> Optional[int]:
    """Pages in the original PDF, or None if it is no longer on disk."""
    pdf = PDF_DIR / f"{stem}.pdf"
    if not pdf.exists():
        return None
    try:
        import fitz
        with fitz.open(pdf) as doc:
            return len(doc)
    except Exception:                                    # noqa: BLE001
        return None


def inspect(stem: str) -> Dict[str, Any]:
    """Coverage facts for one paper, plus any problems found."""
    processed = PROCESSED_DIR / f"{stem}.json"
    row: Dict[str, Any] = {"filename": stem, "problems": []}

    if not processed.exists():
        row["problems"].append("no processed file")
        return row

    try:
        data = json.loads(processed.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        row["problems"].append(f"processed file unreadable: {e}")
        return row

    blocks = data.get("blocks", [])
    chars = sum(len(b.get("text", "")) for b in blocks)
    # Written by the parser from this release on; older artefacts lack it, so
    # fall back to counting the PDF, which also covers a re-check after the
    # parser changed.
    parsed_pages = data.get("num_pages")
    real_pages = data.get("pdf_pages") or pdf_page_count(stem)

    row.update({"parsed_pages": parsed_pages, "pdf_pages": real_pages,
                "blocks": len(blocks), "chars": chars})

    if real_pages and parsed_pages and parsed_pages < real_pages:
        missing = real_pages - parsed_pages
        row["problems"].append(
            f"parsed {parsed_pages} of {real_pages} pages ({missing} never parsed)")
    if len(blocks) < MIN_BLOCKS:
        row["problems"].append(f"only {len(blocks)} block(s) of text")
    if parsed_pages and chars < MIN_CHARS_PER_PAGE * parsed_pages:
        row["problems"].append(
            f"{chars} characters across {parsed_pages} pages is implausibly thin")
    if real_pages is None:
        row["unverified"] = "the PDF is gone, so page coverage cannot be confirmed"

    return row


def audit(stems: Optional[List[str]] = None) -> Dict[str, Any]:
    """Check every processed paper. Returns the bad ones, worst first."""
    if stems is None:
        stems = sorted(p.stem for p in PROCESSED_DIR.glob("*.json"))

    rows = [inspect(s) for s in stems]
    bad = [r for r in rows if r["problems"]]

    def severity(r):
        # biggest proportion of the document missing sorts first
        if r.get("pdf_pages") and r.get("parsed_pages"):
            return r["parsed_pages"] / r["pdf_pages"]
        return 0.0

    bad.sort(key=severity)
    return {
        "checked": len(rows),
        "affected": len(bad),
        "unverifiable": sum(1 for r in rows if r.get("unverified")),
        "papers": bad,
    }


def clear_artifacts(stem: str) -> List[str]:
    """Delete a paper's intermediates so a re-parse cannot skip it.

    Re-embedding already removes a filename's old chunks from the vector
    store, so nothing has to be cleaned there.
    """
    removed = []
    for path in (PARSED_DIR / f"{stem}.json", PARSED_DIR / f"{stem}.txt",
                 PROCESSED_DIR / f"{stem}.json", PROCESSED_DIR / f"{stem}.txt"):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def format_report(result: Dict[str, Any], limit: int = 10) -> str:
    """One-screen summary for a log or a terminal."""
    if not result["affected"]:
        extra = (f" ({result['unverifiable']} unverifiable — PDF pruned)"
                 if result.get("unverifiable") else "")
        return f"coverage: all {result['checked']} paper(s) look complete{extra}"

    lines = [f"coverage: {result['affected']} of {result['checked']} paper(s) "
             f"look incomplete — re-ingest them from the Ingestion tab"]
    for r in result["papers"][:limit]:
        lines.append(f"  {r['filename'][:60]:<62} {'; '.join(r['problems'])}")
    if result["affected"] > limit:
        lines.append(f"  … and {result['affected'] - limit} more")
    return "\n".join(lines)
