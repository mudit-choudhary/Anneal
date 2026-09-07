"""Structure-aware chunking of parse_manager's processed JSON.

Consumes the ordered, typed blocks produced by txt_processor.py
(title / section / paragraph / list / caption / table / formula / footnote /
authors) and produces chunks that respect the document's structure:

- headings never form their own chunk; the heading path is prefixed to each
  chunk's embedded text and stored as metadata
- whole paragraphs are packed into a chunk up to `target_chars`; only a
  paragraph that alone exceeds `max_chars` is split, at sentence boundaries,
  with a one-sentence overlap between pieces
- captions and tables are standalone chunks; a caption adjacent to a table
  travels with it
- formulas are packed inline with the surrounding prose (bare equation
  numbers are dropped)
- list blocks split at item boundaries and are packed like paragraphs
- authors and footnotes are excluded from embedding (returned as `skipped`)

This module is deliberately free of config/chromadb imports so it can be
unit-tested and used by scripts/rag_inspect.py without side effects.
"""

import json
import re
from pathlib import Path

DEFAULT_TARGET_CHARS = 1500   # pack paragraphs up to this size
DEFAULT_MAX_CHARS = 2000      # split a single block only beyond this
                              # (bge-base: 512 tokens ~ 2600 chars of paper text)

SKIP_TYPES = {"authors", "footnote"}
STANDALONE_TYPES = {"caption", "table"}
# Formulas extracted from PDF text are fragmentary ("euv = ϕ(hu, hv),");
# they are packed inline with the surrounding prose that explains them, and
# anything this short (a bare equation number like "(3)") is dropped.
MIN_FORMULA_CHARS = 10

_ABBREVIATIONS = re.compile(
    r"\b(e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|Ref|Refs|Alg|Def|Thm|"
    r"Lem|Prop|vs|cf|approx|resp|Dr|Prof|St|Mr|Ms)\.",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“])")


def split_sentences(text):
    """Split prose into sentences, protecting common abbreviations."""
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = _SENTENCE_END.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _split_words(text, limit):
    """Last-resort split of an over-long sentence at word boundaries."""
    out, cur = [], []
    length = 0
    for word in text.split():
        if cur and length + 1 + len(word) > limit:
            out.append(" ".join(cur))
            cur, length = [], 0
        cur.append(word)
        length += len(word) + (1 if length else 0)
    if cur:
        out.append(" ".join(cur))
    return out


def split_long_text(text, target):
    """Split text into pieces of at most ~`target` chars at sentence
    boundaries, carrying one sentence of overlap into the next piece."""
    sentences = []
    for s in split_sentences(text):
        sentences.extend(_split_words(s, target) if len(s) > target else [s])

    pieces, cur, cur_len = [], [], 0
    for s in sentences:
        if cur and cur_len + 1 + len(s) > target:
            pieces.append(" ".join(cur))
            carry = cur[-1] if len(cur[-1]) <= target // 3 else None
            cur = [carry] if carry else []
            cur_len = len(carry) if carry else 0
        cur.append(s)
        cur_len += len(s) + (1 if cur_len else 0)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _split_lines(text, limit):
    """Split multi-line text (tables, lists) into pieces of <= limit chars,
    cutting only at line boundaries."""
    pieces, cur, cur_len = [], [], 0
    for line in text.split("\n"):
        if cur and cur_len + 1 + len(line) > limit:
            pieces.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + (1 if cur_len else 0)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


class _Packer:
    """Accumulates units (paragraphs, list items, sentence pieces) into chunks."""

    def __init__(self, target_chars, on_chunk):
        self.target = target_chars
        self.on_chunk = on_chunk
        self.units = []      # (text, block_type, page)
        self.length = 0

    def add(self, text, block_type, page):
        sep = self._sep(block_type)
        if self.units and self.length + len(sep) + len(text) > self.target:
            self.flush()
            sep = ""
        self.units.append((text, block_type, page))
        self.length += len(sep) + len(text)

    def _sep(self, block_type):
        if not self.units:
            return ""
        # consecutive list items stay on adjacent lines; anything else is a
        # paragraph break
        return "\n" if block_type == "list" and self.units[-1][1] == "list" else "\n\n"

    def flush(self):
        if not self.units:
            return
        body = self.units[0][0]
        for (text, block_type, _), (_, prev_type, _) in zip(self.units[1:], self.units):
            body += ("\n" if block_type == "list" and prev_type == "list" else "\n\n") + text
        self.on_chunk(body, [u[1] for u in self.units], [u[2] for u in self.units])
        self.units, self.length = [], 0


def chunk_document(blocks, filename, target_chars=DEFAULT_TARGET_CHARS,
                   max_chars=DEFAULT_MAX_CHARS):
    """Chunk a processed document.

    `blocks` is the `blocks` list from data/processed/<name>.json.
    Returns (chunks, skipped) where each chunk is
        {"text": <heading path + body, what gets embedded>,
         "body": <body only>,
         "metadata": {filename, title, section, page_start, page_end,
                      block_types, chunk_index, n_chars}}
    and `skipped` lists the authors/footnote blocks left out of embedding.
    """
    title = next((b["text"] for b in blocks if b["type"] == "title"), None) or filename
    state = {"section": ""}
    chunks, skipped = [], []

    def emit(body, types, pages):
        section = state["section"]
        heading = f"{title} › {section}" if section else title
        chunks.append({
            "text": f"{heading}\n\n{body}",
            "body": body,
            "metadata": {
                "filename": filename,
                "title": title,
                "section": section,
                "page_start": min(pages),
                "page_end": max(pages),
                "block_types": ",".join(sorted(set(types))),
                "chunk_index": len(chunks),
                "n_chars": len(body),
            },
        })

    packer = _Packer(target_chars, emit)

    i = 0
    while i < len(blocks):
        b = blocks[i]
        t, text, page = b["type"], b["text"], b["page"]

        if t == "title":
            pass
        elif t == "section":
            packer.flush()
            state["section"] = text
        elif t in SKIP_TYPES:
            skipped.append(b)
        elif t in STANDALONE_TYPES:
            packer.flush()
            group = [b]
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt and {t, nxt["type"]} == {"caption", "table"}:
                group.append(nxt)
                i += 1
            body = "\n\n".join(g["text"] for g in group)
            pages = [g["page"] for g in group]
            types = [g["type"] for g in group]
            if len(body) <= max_chars:
                emit(body, types, pages)
            else:
                for piece in _split_lines(body, max_chars):
                    emit(piece, types, pages)
        elif t == "formula":
            if len(text) >= MIN_FORMULA_CHARS and any(ch.isalpha() for ch in text):
                packer.add(text, "formula", page)
            else:
                skipped.append(b)
        elif t == "list":
            for item in text.split("\n"):
                item = item.strip()
                if not item:
                    continue
                units = [item] if len(item) <= max_chars else split_long_text(item, target_chars)
                for u in units:
                    packer.add(u, "list", page)
        else:  # paragraph
            units = [text] if len(text) <= max_chars else split_long_text(text, target_chars)
            for u in units:
                packer.add(u, "paragraph", page)
        i += 1

    packer.flush()
    return chunks, skipped


def load_blocks(json_path):
    """Read a processed JSON; returns (blocks, source_pdf)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return data["blocks"], data.get("source")


def chunk_file(json_path, target_chars=DEFAULT_TARGET_CHARS, max_chars=DEFAULT_MAX_CHARS):
    json_path = Path(json_path)
    blocks, _ = load_blocks(json_path)
    return chunk_document(blocks, json_path.stem, target_chars, max_chars)
