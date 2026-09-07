"""Stage 2 of parsing: layout JSON -> assembled, tagged text.

Takes the region-level output of pdf_parser.py and produces the final text
used for chunking/embedding:

- column-aware reading order (full-width bands, then left/right columns)
- paragraph reconstruction across regions, columns, and pages, with
  de-hyphenation
- structure preserved as tags: `#` Title, `##` Section-header, [AUTHORS],
  [CAPTION], [TABLE], [FORMULA], [FOOTNOTE]
- page headers/footers and text inside pictures excluded from the flow
  (kept in the companion JSON as metadata)

Output: data/processed/<name>.txt and <name>.json
"""

import json
from pathlib import Path

import requests

from config import (
    PROCESSED_DIR,
    REGISTRY_URL,
    FULL_WIDTH_FRACTION,
    SINGLE_COLUMN_FRACTION,
)

NOISE_LABELS = {"Page-header", "Page-footer"}
BODY_LABELS = {"Text", "List-item"}
TERMINAL_CHARS = ".!?"


def order_regions(regions, page_width):
    """Return regions in reading order for one page.

    Full-width regions (title, abstract, figures spanning both columns) split
    the page into vertical bands; within each band, the left column is read
    top-to-bottom before the right column. Pages that are mostly full-width
    are treated as single-column and simply read top-to-bottom.
    """
    regions = [r for r in regions if r["label"] not in NOISE_LABELS]
    if not regions:
        return []

    def is_full_width(r):
        return (r["bbox"][2] - r["bbox"][0]) > FULL_WIDTH_FRACTION * page_width

    textual = [r for r in regions if r.get("lines")]
    if textual:
        full_count = sum(1 for r in textual if is_full_width(r))
        if full_count >= SINGLE_COLUMN_FRACTION * len(textual):
            return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    ordered = []
    band = []

    def flush_band():
        mid = page_width / 2
        left = [r for r in band if (r["bbox"][0] + r["bbox"][2]) / 2 < mid]
        right = [r for r in band if (r["bbox"][0] + r["bbox"][2]) / 2 >= mid]
        ordered.extend(sorted(left, key=lambda r: r["bbox"][1]))
        ordered.extend(sorted(right, key=lambda r: r["bbox"][1]))
        band.clear()

    for region in sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0])):
        # Authors boxes sit in the title band, not in the body columns; treat
        # them like full-width separators so they aren't pushed into a column.
        if is_full_width(region) or region["label"] == "Authors":
            flush_band()
            ordered.append(region)
        else:
            band.append(region)
    flush_band()

    return ordered


def join_lines(lines):
    """Join a region's lines into one string, de-hyphenating line wraps."""
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
        elif text.endswith("-"):
            # Line-wrap hyphen: drop it before a lowercase continuation,
            # keep it (no space) before an uppercase one ("non-Euclidean").
            text = text[:-1] + line if line[0].islower() else text + line
        else:
            text += " " + line
    return text


def ends_terminally(text):
    """True if a paragraph looks finished (terminal punctuation, allowing a
    closing quote/bracket after it)."""
    t = text.rstrip()
    if not t:
        return True
    if t[-1] in TERMINAL_CHARS:
        return True
    if t[-1] in "\"'”’)]" and len(t) > 1 and t[-2] in TERMINAL_CHARS:
        return True
    return False


def merge_paragraph(buffer, text):
    """Append a continuation region to an open paragraph buffer."""
    if buffer.endswith("-") and text:
        return buffer[:-1] + text if text[0].islower() else buffer + text
    return buffer + " " + text


class LayoutAssembler:
    """Walks pages in reading order and emits structured blocks.

    A paragraph stays open across intervening captions/footnotes/figures and
    across column/page boundaries until it ends with terminal punctuation or a
    heading forces a break; its block keeps the position where it started.
    """

    def __init__(self):
        self.blocks = []
        self.para_index = None   # index of the open paragraph's block
        self.para_text = ""
        self.list_open = False   # open paragraph block is a list

    def _flush(self):
        if self.para_index is not None:
            self.blocks[self.para_index]["text"] = self.para_text
        self.para_index = None
        self.para_text = ""
        self.list_open = False

    def _open(self, block_type, text, page):
        self.blocks.append({"type": block_type, "page": page, "text": None})
        self.para_index = len(self.blocks) - 1
        self.para_text = text
        self.list_open = block_type == "list"

    def _emit(self, block_type, text, page):
        self.blocks.append({"type": block_type, "page": page, "text": text})

    def add_region(self, region, page):
        label = region["label"]
        text = join_lines(region.get("lines", []))
        if not text and label not in ("Picture",):
            return

        if label in ("Title", "Section-header", "Authors"):
            self._flush()
            kind = {"Title": "title", "Section-header": "section", "Authors": "authors"}[label]
            self._emit(kind, text, page)
        elif label == "Caption":
            self._emit("caption", text, page)
        elif label == "Footnote":
            self._emit("footnote", text, page)
        elif label == "Formula":
            self._emit("formula", text, page)
        elif label == "Table":
            self._emit("table", "\n".join(l.strip() for l in region.get("lines", []) if l.strip()), page)
        elif label == "Picture":
            pass  # anchor only; picture-internal text was already excluded
        elif label == "List-item":
            if self.para_index is not None and self.list_open:
                self.para_text += "\n" + text
            else:
                self._flush()
                self._open("list", text, page)
        else:  # Text (including fallback regions)
            if self.para_index is not None and not self.list_open and not ends_terminally(self.para_text):
                self.para_text = merge_paragraph(self.para_text, text)
            else:
                self._flush()
                self._open("paragraph", text, page)

    def finish(self):
        self._flush()
        return [b for b in self.blocks if b["text"]]


def render_txt(blocks):
    parts = []
    for b in blocks:
        if b["type"] == "title":
            parts.append(f"# {b['text']}")
        elif b["type"] == "section":
            parts.append(f"## {b['text']}")
        elif b["type"] == "authors":
            parts.append(f"[AUTHORS] {b['text']}")
        elif b["type"] == "caption":
            parts.append(f"[CAPTION] {b['text']}")
        elif b["type"] == "footnote":
            parts.append(f"[FOOTNOTE] {b['text']}")
        elif b["type"] == "formula":
            parts.append(f"[FORMULA] {b['text']}")
        elif b["type"] == "table":
            parts.append(f"[TABLE]\n{b['text']}\n[/TABLE]")
        else:
            parts.append(b["text"])
    return "\n\n".join(parts) + "\n"


def resolve_layout_json(arg):
    """Resolve a CLI argument to a layout JSON: a path as given, else a
    filename or bare stem looked up in PARSED_DIR."""
    from config import PARSED_DIR

    p = Path(arg)
    if p.exists():
        return p
    for candidate in (PARSED_DIR / p.name, PARSED_DIR / f"{p.stem}.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"'{arg}' not found (also tried {PARSED_DIR / p.name})")


def process_layout_json(input_json, output_txt=None, output_json=None):
    """Assemble one layout JSON into processed text. Returns the txt path."""
    input_json = Path(input_json)
    with open(input_json, "r", encoding="utf-8") as f:
        layout = json.load(f)

    assembler = LayoutAssembler()
    dropped = {"page_headers": [], "page_footers": [], "picture_text": []}

    for page_entry in layout["pages"]:
        page_no = page_entry["page"]
        for region in page_entry["regions"]:
            if region["label"] == "Page-header":
                dropped["page_headers"].append(join_lines(region.get("lines", [])))
            elif region["label"] == "Page-footer":
                dropped["page_footers"].append(join_lines(region.get("lines", [])))
        for item in page_entry.get("swallowed_text", []):
            if item["label"] == "Picture":
                dropped["picture_text"].append(item["text"])

        for region in order_regions(page_entry["regions"], page_entry["width"]):
            assembler.add_region(region, page_no)

    blocks = assembler.finish()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_json.stem
    output_txt = Path(output_txt) if output_txt else PROCESSED_DIR / f"{stem}.txt"
    output_json = Path(output_json) if output_json else PROCESSED_DIR / f"{stem}.json"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(render_txt(blocks))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "source": layout.get("source_pdf"),
            "num_pages": layout.get("num_pages"),
            "dropped": dropped,
            "blocks": blocks,
        }, f, indent=2)

    _update_registry(stem)
    print(f"[processor] {stem}: {len(blocks)} blocks -> {output_txt.name}")
    return output_txt


def _update_registry(filename):
    try:
        response = requests.post(
            f"{REGISTRY_URL}/update_status",
            json={"filename": filename, "status": "processed"},
            timeout=20,
        )
        if not response.json().get("success"):
            print(f"[processor] registry rejected status update for {filename}")
    except requests.RequestException as e:
        print(f"[processor] registry unreachable ({e}); status not updated")


if __name__ == "__main__":
    import sys
    from config import PARSED_DIR

    targets = sys.argv[1:] or [str(p) for p in sorted(PARSED_DIR.glob("*.json"))[:1]]
    for target in targets:
        process_layout_json(resolve_layout_json(target))
