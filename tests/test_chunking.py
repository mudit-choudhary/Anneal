"""Unit tests for structure-aware chunking (embedding_manager/chunking.py).

chunking.py has no config/chromadb dependencies, so it's loaded straight from
its file path — no sys.path juggling with the other managers' config.py.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ch():
    spec = importlib.util.spec_from_file_location(
        "chunking", REPO_ROOT / "embedding_manager" / "chunking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blk(type_, text, page=0):
    return {"type": type_, "page": page, "text": text}


def sentence(i, words=12):
    return f"Sentence number {i} " + " ".join(f"w{j}" for j in range(words)) + "."


class TestSentences:
    def test_basic_split(self, ch):
        assert ch.split_sentences("First one. Second one! Third?") == [
            "First one.", "Second one!", "Third?"]

    def test_abbreviations_protected(self, ch):
        s = ch.split_sentences("See Fig. 2 and Zhang et al. for details. Next sentence here.")
        assert s == ["See Fig. 2 and Zhang et al. for details.", "Next sentence here."]

    def test_citation_then_sentence(self, ch):
        assert ch.split_sentences("Known as over-smoothing [13]. To address this, we act.") == [
            "Known as over-smoothing [13].", "To address this, we act."]


class TestChunkDocument:
    def test_heading_prefix_and_no_heading_chunk(self, ch):
        chunks, _ = ch.chunk_document([
            blk("title", "A Paper"), blk("section", "1. Intro"), blk("paragraph", "Body text.")],
            "paper")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "A Paper › 1. Intro\n\nBody text."
        assert chunks[0]["metadata"]["section"] == "1. Intro"
        assert chunks[0]["metadata"]["title"] == "A Paper"

    def test_short_paragraphs_packed(self, ch):
        chunks, _ = ch.chunk_document(
            [blk("paragraph", f"Paragraph {i} is short.", page=i) for i in range(3)], "p")
        assert len(chunks) == 1
        assert chunks[0]["body"] == "Paragraph 0 is short.\n\nParagraph 1 is short.\n\nParagraph 2 is short."
        assert chunks[0]["metadata"]["page_start"] == 0
        assert chunks[0]["metadata"]["page_end"] == 2

    def test_target_respected_without_splitting_paragraphs(self, ch):
        p = "x" * 890 + " end."
        chunks, _ = ch.chunk_document([blk("paragraph", p), blk("paragraph", p)], "p",
                                      target_chars=1500, max_chars=2000)
        assert len(chunks) == 2
        assert all(c["body"] == p for c in chunks)

    def test_long_paragraph_split_at_sentences_with_overlap(self, ch):
        text = " ".join(sentence(i) for i in range(40))   # ~3000 chars
        chunks, _ = ch.chunk_document([blk("paragraph", text)], "p",
                                      target_chars=1500, max_chars=2000)
        assert len(chunks) >= 2
        for c in chunks:
            assert c["body"].endswith(".")          # never mid-sentence
            assert len(c["body"]) <= 1500 + 100     # near target
        last_sentence = chunks[0]["body"].rsplit("Sentence", 1)[1]
        assert chunks[1]["body"].startswith("Sentence" + last_sentence)   # one-sentence overlap

    def test_section_boundary_flushes(self, ch):
        chunks, _ = ch.chunk_document([
            blk("section", "A"), blk("paragraph", "In A."),
            blk("section", "B"), blk("paragraph", "In B.")], "p")
        assert [c["metadata"]["section"] for c in chunks] == ["A", "B"]

    def test_caption_and_table_travel_together(self, ch):
        chunks, _ = ch.chunk_document([
            blk("paragraph", "Before."), blk("caption", "Table 1: results"),
            blk("table", "a b\n1 2"), blk("paragraph", "After.")], "p")
        assert [c["metadata"]["block_types"] for c in chunks] == [
            "paragraph", "caption,table", "paragraph"]
        assert chunks[1]["body"] == "Table 1: results\n\na b\n1 2"

    def test_formula_inline_and_equation_numbers_dropped(self, ch):
        chunks, skipped = ch.chunk_document([
            blk("paragraph", "The edge representation is computed as"),
            blk("formula", "euv = ϕ(hu, hv),"),
            blk("formula", "(1)"),
            blk("paragraph", "where ϕ is a pairwise mapping.")], "p")
        assert len(chunks) == 1
        assert chunks[0]["body"] == (
            "The edge representation is computed as\n\neuv = ϕ(hu, hv),\n\nwhere ϕ is a pairwise mapping.")
        assert chunks[0]["metadata"]["block_types"] == "formula,paragraph"
        assert [s["text"] for s in skipped] == ["(1)"]

    def test_authors_and_footnotes_skipped(self, ch):
        chunks, skipped = ch.chunk_document([
            blk("authors", "A. Author"), blk("paragraph", "Real content."),
            blk("footnote", "email@x.org")], "p")
        assert len(chunks) == 1
        assert "email" not in chunks[0]["text"] and "Author" not in chunks[0]["text"]
        assert [s["type"] for s in skipped] == ["authors", "footnote"]

    def test_list_items_split_and_joined_by_newline(self, ch):
        chunks, _ = ch.chunk_document([blk("list", "[1] ref one.\n[2] ref two.\n[3] ref three.")], "p")
        assert len(chunks) == 1
        assert chunks[0]["body"] == "[1] ref one.\n[2] ref two.\n[3] ref three."
        assert chunks[0]["metadata"]["block_types"] == "list"

    def test_long_list_packed_at_item_boundaries(self, ch):
        items = "\n".join(f"[{i}] " + "r" * 300 + "." for i in range(10))   # ~3100 chars
        chunks, _ = ch.chunk_document([blk("list", items)], "p", target_chars=1000, max_chars=2000)
        assert len(chunks) >= 3
        for c in chunks:
            assert all(line.startswith("[") for line in c["body"].split("\n"))

    def test_chunk_index_sequential_and_filename(self, ch):
        chunks, _ = ch.chunk_document(
            [blk("paragraph", "x" * 1400 + "."), blk("paragraph", "y" * 1400 + ".")], "mypaper")
        assert [c["metadata"]["chunk_index"] for c in chunks] == [0, 1]
        assert all(c["metadata"]["filename"] == "mypaper" for c in chunks)

    def test_title_falls_back_to_filename(self, ch):
        chunks, _ = ch.chunk_document([blk("paragraph", "Text.")], "Some_Paper")
        assert chunks[0]["text"].startswith("Some_Paper\n\n")
