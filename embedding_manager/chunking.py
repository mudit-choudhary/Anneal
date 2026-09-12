"""Grain-Growth Chunking — next-fit packing under structural barriers.

A chunk grows by accepting the next block whole and keeps growing until either
the size budget is reached or it meets a barrier: a section heading, or a block
that must stand alone such as a table. Barriers are absolute — a chunk never
crosses one, however much budget is left.

The packing is *next-fit* in the bin-packing sense, not first-fit or best-fit:
when a block will not fit the chunk being grown, that chunk is closed for good
and a new one opens, and no earlier chunk is ever reopened to see whether the
block would have fitted there. One pass, in order, which is what keeps reading
order intact.

In the published taxonomy this is document-based / structure-aware chunking;
the closest existing implementation is Unstructured's `by_title` chunker.

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

# Block types wrapped in a labelled fence inside the embedded text, so the
# answering model — and a person reading a citation — can tell content that is
# not prose from content that is. Adding a type here is the whole change
# needed to fence it; nothing below is special-cased per type.
#
# Why a markdown-style fence and not XML tags: for the *prompt* side the two
# are equivalent, but the same string is also what the embedding model sees,
# and a fence costs a couple of tokens where `<table>...</table>` costs more
# and is commonly stripped as markup noise. A fence is also the structure
# these models have seen most often.
# Two behaviours, because the distinction is real rather than cosmetic: a
# formula is fenced but stays packed with the prose that explains it, while a
# table is fenced and stands alone. Adding a type to either set is the whole
# change needed — `list` is deliberately in neither, because a bibliography is
# language and earns its place in the embedding.
FENCED_INLINE = {"formula"}
FENCED_STANDALONE = {"table"}
FENCED_TYPES = FENCED_INLINE | FENCED_STANDALONE
FENCE_CHAR = "~"          # not backticks: papers quote code with those
MIN_FENCE = 3
# Formulas extracted from PDF text are fragmentary ("euv = ϕ(hu, hv),");
# they are packed inline with the surrounding prose that explains them, and
# anything this short (a bare equation number like "(3)") is dropped.
MIN_FORMULA_CHARS = 10

# A formula is usually introduced by the sentence before it and explained by the
# prose after it, so the three belong in one chunk. When that group is too large
# to fit, this many sentences of the introducing paragraph are kept with the
# formula and the rest of the paragraph becomes its own chunk. 0 keeps none.
FORMULA_LEAD_SENTENCES = 2          # valid range 0..10

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


_FENCE_RUN = re.compile(rf"^{re.escape(FENCE_CHAR)}{{{MIN_FENCE},}}", re.M)


def fence(block_type, text, part=None, of=None):
    """Wrap `text` in a labelled fence that is safe for any content.

    The bar is always longer than the longest run of the fence character
    already at the start of a line inside `text`, which is the rule CommonMark
    uses and the reason this cannot be broken by the content it wraps — a
    paper that happens to print `~~~~` in a code listing still fences cleanly.

    `part`/`of` mark a block too large for one chunk, so a model reading piece
    2 of 3 knows it is not holding the whole table.
    """
    longest = max((len(m) for m in _FENCE_RUN.findall(text)), default=0)
    bar = FENCE_CHAR * max(MIN_FENCE, longest + 1)
    label = block_type
    if of and of > 1:
        label += f" part {part} of {of}"
    return f"{bar}{label}\n{text}\n{bar}"


def strip_fences(text):
    """Remove fenced blocks, returning only the prose around them.

    Used by the evaluation: a table that correctly ends in a number is not a
    chunk that "ends mid-sentence", and judging it by prose rules measures the
    wrong thing. Unterminated fences are treated as running to the end, which
    is what a chunk cut through a fenced block actually looks like.
    """
    out, depth, bar = [], 0, None
    for line in text.split("\n"):
        opening = _FENCE_RUN.match(line)
        if opening and depth == 0:
            depth, bar = 1, opening.group(0)
            continue
        if depth and opening and line.strip() == bar:
            depth, bar = 0, None
            continue
        if not depth:
            out.append(line)
    # Removing a block leaves a blank line on each side of the hole; collapse
    # them so the prose reads as it would have without the block.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def fences_balanced(text):
    """True when every fence opened in `text` is also closed."""
    depth, bar = 0, None
    for line in text.split("\n"):
        opening = _FENCE_RUN.match(line)
        if opening and depth == 0:
            depth, bar = 1, opening.group(0)
        elif depth and opening and line.strip() == bar:
            depth, bar = 0, None
    return depth == 0


def has_fence(text):
    return bool(_FENCE_RUN.search(text))


# --- what counts as a clean chunk boundary ---------------------------------
# A fence bar is a structural boundary in its own right: opening one is as
# explicit a start as a capital letter after a full stop, and closing one is as
# explicit an end as the full stop itself. Because `fence()` always labels the
# opening bar and never labels the closing one, the two are told apart by
# whether anything follows the bar on its line — which is also how a chunk that
# begins *inside* a block is detected, and that one is genuinely dirty.
_BARE_BAR = re.compile(rf"^{re.escape(FENCE_CHAR)}{{{MIN_FENCE},}}\s*$")
_STARTS_MID = re.compile(r"^[a-z,;:)\]]")
# Only a full stop, question mark or exclamation mark closes a sentence. A
# colon or semicolon does not: "The loss is formulated as:" is the *opening* of
# a thought, and a chunk that ends there has cut one in half. Closing quotes and
# brackets may follow the mark — the sentence is still closed by it.
_ENDS_CLEAN = re.compile(r"[.!?][)\]\"'”’]*$")


def _first_bar(text):
    """'open' | 'close' | None for the first fence bar in `text`."""
    for line in text.split("\n"):
        if _FENCE_RUN.match(line):
            return "close" if _BARE_BAR.match(line) else "open"
    return None


def starts_inside_fence(text):
    """True when the chunk opens part-way through a fenced block.

    The give-away is a closing bar with no opener before it: the block began in
    the previous chunk. Such a chunk lands the reader in unlabelled tabular
    data, which the prose rule alone would wrongly score as clean whenever the
    first row happens to begin with a capital.
    """
    return _first_bar(text) == "close"


def starts_cleanly(text):
    body = text.strip()
    if not body:
        return True
    if starts_inside_fence(body):
        return False
    if _FENCE_RUN.match(body.split("\n", 1)[0]):
        return True              # opens a labelled block
    return not _STARTS_MID.match(body)


def ends_cleanly(text):
    body = text.strip()
    if not body:
        return True
    last = body.rsplit("\n", 1)[-1]
    if _BARE_BAR.match(last):
        return True              # closes a block
    if _FENCE_RUN.match(last):
        return False             # opened a block and gave it nothing
    return bool(_ENDS_CLEAN.search(body))


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


def _starts_lower(text):
    """True when the first letter of `text` is lower case.

    After a formula this signals an explanation continuing the same sentence —
    "where phi is a pairwise mapping" — rather than a new thought.
    """
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].isalpha() and stripped[0].islower()


def bind_formula_groups(blocks):
    """Merge each formula with the prose that introduces and explains it.

    A formula is bound to the paragraph *before* it when that paragraph does not
    end a sentence, because the formula completes it. It is bound to the
    paragraph *after* it when that paragraph starts lower case, because it is
    explaining the terms. Runs of consecutive formulas are treated as one.

    The result is a block of type "bound" that later stages must not divide,
    carrying the constituent types and pages so metadata stays accurate. For
    tables no such rule applies: a table stands on its own.
    """
    out, i = [], 0
    while i < len(blocks):
        b = blocks[i]
        if b["type"] not in FENCED_INLINE:
            out.append(b)
            i += 1
            continue

        run = [b]
        j = i + 1
        while j < len(blocks) and blocks[j]["type"] in FENCED_INLINE:
            run.append(blocks[j])
            j += 1

        lead = None
        if out and out[-1]["type"] == "paragraph" and not _ENDS_CLEAN.search(out[-1]["text"].strip()):
            lead = out.pop()

        trail = None
        if j < len(blocks) and blocks[j]["type"] == "paragraph" and _starts_lower(blocks[j]["text"]):
            trail = blocks[j]
            j += 1

        if lead is None and trail is None:
            out.extend(run)
            i = j
            continue

        members = ([lead] if lead else []) + run + ([trail] if trail else [])
        out.append({
            "type": "bound",
            "text": "\n\n".join(m["text"] for m in members),
            "page": members[0]["page"],
            "members": members,
        })
        i = j
    return out


class _Packer:
    """Grows a chunk by whole units until a barrier or the size limit.

    Two reasons to keep growing past the target, both capped at `max_chars`:
    a unit marked `whole`, which cannot be divided without harm — a formula
    bound to its explanation; and a chunk whose text does not yet close a
    sentence, because ending on "formulated as:" cuts a thought in half.

    An ordinary paragraph is not `whole`: if it will not fit here it simply
    starts the next chunk, losing nothing.
    """

    def __init__(self, target_chars, on_chunk, max_chars=None):
        self.target = target_chars
        self.max = max_chars or target_chars
        self.on_chunk = on_chunk
        self.units = []      # (text, types, pages)
        self.length = 0

    def _body(self):
        if not self.units:
            return ""
        body = self.units[0][0]
        for (text, types, _), (_, prev_types, _) in zip(self.units[1:], self.units):
            joiner = "\n" if types[-1] == "list" and prev_types[-1] == "list" else "\n\n"
            body += joiner + text
        return body

    def add(self, text, types, pages, whole=False):
        if isinstance(types, str):
            types = [types]
        if isinstance(pages, int):
            pages = [pages]
        sep = "" if not self.units else (
            "\n" if types[-1] == "list" and self.units[-1][1][-1] == "list" else "\n\n")

        if self.units:
            projected = self.length + len(sep) + len(text)
            if projected > self.target:
                unfinished = not _ENDS_CLEAN.search(self._body().strip())
                if not ((whole or unfinished) and projected <= self.max):
                    self.flush()
                    sep = ""
        self.units.append((text, types, pages))
        self.length += len(sep) + len(text)

    def flush(self):
        if not self.units:
            return
        types = [t for u in self.units for t in u[1]]
        pages = [p for u in self.units for p in u[2]]
        self.on_chunk(self._body(), types, pages)
        self.units, self.length = [], 0


def _fit_bound_group(members, max_chars, lead_sentences):
    """Place an over-long formula group without divorcing it from its prose.

    Returns (spill, kept). `spill` is the front of the introducing paragraph
    that has to become its own chunk; `kept` are the members that stay together.
    The last `lead_sentences` sentences of the introduction are kept with the
    formula because they are the ones that introduce it; if that still does not
    fit, fewer are kept, down to none.
    """
    lead = members[0] if members[0]["type"] == "paragraph" else None
    rest = members[1:] if lead else members
    rest_text = "\n\n".join(_render_member(m) for m in rest)

    if lead is None:
        return None, rest

    sentences = split_sentences(lead["text"])
    for n in range(min(lead_sentences, len(sentences)), -1, -1):
        tail = " ".join(sentences[len(sentences) - n:]) if n else ""
        candidate = (tail + "\n\n" + rest_text) if tail else rest_text
        if len(candidate) <= max_chars:
            front = " ".join(sentences[:len(sentences) - n])
            spill = dict(lead, text=front) if front.strip() else None
            kept = ([dict(lead, text=tail)] if tail else []) + rest
            return spill, kept
    return lead, rest


def _render_member(m):
    return fence(m["type"], m["text"]) if m["type"] in FENCED_TYPES else m["text"]


def chunk_document(blocks, filename, target_chars=DEFAULT_TARGET_CHARS,
                   max_chars=DEFAULT_MAX_CHARS,
                   lead_sentences=FORMULA_LEAD_SENTENCES):
    """Chunk a processed document — Grain-Growth Chunking.

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

    # Drop what is never embedded before binding, so a dropped equation number
    # cannot glue a paragraph to a formula that is not there.
    kept = []
    for b in blocks:
        if b["type"] in SKIP_TYPES:
            skipped.append(b)
        elif b["type"] in FENCED_INLINE and not (
                len(b["text"]) >= MIN_FORMULA_CHARS and any(c.isalpha() for c in b["text"])):
            skipped.append(b)
        else:
            kept.append(b)

    packer = _Packer(target_chars, emit, max_chars)
    blocks = bind_formula_groups(kept)

    i = 0
    while i < len(blocks):
        b = blocks[i]
        t, text, page = b["type"], b["text"], b["page"]

        if t == "title":
            pass
        elif t == "section":
            packer.flush()
            state["section"] = text
        elif t == "bound":
            members = b["members"]
            body = "\n\n".join(_render_member(m) for m in members)
            types = [m["type"] for m in members]
            pages = [m["page"] for m in members]
            if len(body) <= max_chars:
                packer.add(body, types, pages, whole=True)
            else:
                spill, keep = _fit_bound_group(members, max_chars, lead_sentences)
                if spill:
                    for u in ([spill["text"]] if len(spill["text"]) <= max_chars
                              else split_long_text(spill["text"], target_chars)):
                        packer.add(u, ["paragraph"], [spill["page"]], whole=True)
                packer.flush()
                kept_body = "\n\n".join(_render_member(m) for m in keep)
                kept_types = [m["type"] for m in keep]
                kept_pages = [m["page"] for m in keep]
                if len(kept_body) <= max_chars:
                    packer.add(kept_body, kept_types, kept_pages, whole=True)
                else:
                    for piece in _split_lines(kept_body, max_chars):
                        packer.add(piece, kept_types, kept_pages, whole=True)
                packer.flush()
        elif t in STANDALONE_TYPES or t in FENCED_STANDALONE:
            packer.flush()
            group = [b]
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt and {t, nxt["type"]} == {"caption", "table"}:
                group.append(nxt)
                i += 1
            pages = [g["page"] for g in group]
            types = [g["type"] for g in group]

            preamble = "\n\n".join(g["text"] for g in group if g["type"] not in FENCED_TYPES)
            payload = "\n\n".join(g["text"] for g in group if g["type"] in FENCED_TYPES)

            if not payload:
                emit(preamble, types, pages)
            else:
                head = f"{preamble}\n\n" if preamble else ""
                budget = max_chars - len(head) - len(FENCE_CHAR) * 2 * MIN_FENCE - 24
                pieces = ([payload] if len(payload) <= budget
                          else _split_lines(payload, budget))
                label = next(g["type"] for g in group if g["type"] in FENCED_TYPES)
                for n, piece in enumerate(pieces, 1):
                    emit(head + fence(label, piece, part=n, of=len(pieces)), types, pages)
        elif t in FENCED_INLINE:
            packer.add(fence(t, text), [t], [page], whole=True)
        elif t == "list":
            for item in text.split("\n"):
                item = item.strip()
                if not item:
                    continue
                units = [item] if len(item) <= max_chars else split_long_text(item, target_chars)
                for u in units:
                    packer.add(u, ["list"], [page])
        else:  # paragraph
            if len(text) <= max_chars:
                # No `whole` here: a paragraph that will not fit the current
                # chunk loses nothing by starting the next one, so growing past
                # the target would buy a larger chunk without saving a split.
                packer.add(text, ["paragraph"], [page])
            else:
                # A paragraph that has to be split starts its own chunk rather
                # than riding along after a complete one.
                packer.flush()
                for u in split_long_text(text, target_chars):
                    packer.add(u, ["paragraph"], [page])
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
