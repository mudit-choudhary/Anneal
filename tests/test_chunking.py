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
        # the caption stays prose; the table body is fenced so a model can
        # tell data from sentences
        assert chunks[1]["body"] == "Table 1: results\n\n~~~table\na b\n1 2\n~~~"
        assert ch.strip_fences(chunks[1]["body"]) == "Table 1: results"

    def test_formula_inline_and_equation_numbers_dropped(self, ch):
        chunks, skipped = ch.chunk_document([
            blk("paragraph", "The edge representation is computed as"),
            blk("formula", "euv = ϕ(hu, hv),"),
            blk("formula", "(1)"),
            blk("paragraph", "where ϕ is a pairwise mapping.")], "p")
        assert len(chunks) == 1
        # the formula is fenced in place, so the prose either side still reads
        # as one passage once the fence is removed
        assert chunks[0]["body"] == (
            "The edge representation is computed as\n\n"
            "~~~formula\neuv = ϕ(hu, hv),\n~~~\n\n"
            "where ϕ is a pairwise mapping.")
        assert ch.strip_fences(chunks[0]["body"]) == (
            "The edge representation is computed as\n\nwhere ϕ is a pairwise mapping.")
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


class TestTypedFences:
    """Tables and formulas are fenced so the answering model, and a person
    reading a citation, can tell data from prose — and so the evaluation can
    judge each by its own standard."""

    def test_fence_is_labelled_and_balanced(self, ch):
        out = ch.fence("table", "a b c\nd e f")
        assert out.startswith("~~~table\n") and out.endswith("\n~~~")
        assert ch.fences_balanced(out)

    def test_fence_outgrows_content_that_contains_a_fence(self, ch):
        """The whole point of the CommonMark rule: content cannot break out."""
        out = ch.fence("formula", "~~~~ not a real fence\nx = 1")
        assert out.startswith("~~~~~formula")
        assert ch.fences_balanced(out)

    def test_split_blocks_say_which_part_they_are(self, ch):
        assert ch.fence("table", "r", part=2, of=3).startswith("~~~table part 2 of 3")
        assert ch.fence("table", "r", part=1, of=1).startswith("~~~table\n")

    def test_strip_fences_leaves_only_prose(self, ch):
        text = f"Before.\n\n{ch.fence('table', '1 2 3')}\n\nAfter."
        assert ch.strip_fences(text) == "Before.\n\nAfter."

    def test_unterminated_fence_is_detected_and_swallowed(self, ch):
        """A chunker that cuts through a table leaves this shape behind."""
        cut = "Lead in.\n~~~table\n1 2 3\n4 5 6"
        assert not ch.fences_balanced(cut)
        assert ch.strip_fences(cut) == "Lead in."

    def test_table_chunk_carries_its_caption_outside_the_fence(self, ch):
        blocks = [
            {"type": "title", "text": "A Paper", "page": 0},
            {"type": "caption", "text": "Table 1: Results by dataset.", "page": 1},
            {"type": "table", "text": "Cora 91.6\nCiteseer 92.2", "page": 1},
        ]
        chunks, _ = ch.chunk_document(blocks, "A_Paper")
        body = chunks[0]["body"]
        assert body.startswith("Table 1: Results by dataset.")
        assert "~~~table" in body
        # the caption is prose and survives stripping; the numbers do not
        assert "Table 1" in ch.strip_fences(body)
        assert "Cora" not in ch.strip_fences(body)

    def test_a_long_table_splits_into_numbered_parts_each_with_the_caption(self, ch):
        rows = "\n".join(f"row{i} {i} {i}" for i in range(400))
        blocks = [
            {"type": "title", "text": "A Paper", "page": 0},
            {"type": "caption", "text": "Table 2: Long one.", "page": 1},
            {"type": "table", "text": rows, "page": 1},
        ]
        chunks, _ = ch.chunk_document(blocks, "A_Paper")
        assert len(chunks) > 1
        for n, c in enumerate(chunks, 1):
            assert f"part {n} of {len(chunks)}" in c["body"]
            assert c["body"].startswith("Table 2: Long one.")

    def test_a_formula_is_fenced_inside_its_paragraph(self, ch):
        blocks = [
            {"type": "title", "text": "A Paper", "page": 0},
            {"type": "formula", "text": "euv = phi(hu, hv),", "page": 1},
            {"type": "paragraph", "text": "phi denotes a mapping function. " * 3, "page": 1},
        ]
        chunks, _ = ch.chunk_document(blocks, "A_Paper")
        body = chunks[0]["body"]
        assert "~~~formula" in body and "phi denotes" in body

    def test_fencing_generalises_to_any_block_type(self, ch, monkeypatch):
        """Nothing is special-cased per type: adding a type to FENCED_TYPES is
        the entire change needed to fence it. `list` is deliberately not fenced
        in production — a bibliography is language and belongs in the embedding
        — but the mechanism does not care."""
        monkeypatch.setattr(ch, "FENCED_STANDALONE", {"table", "list"})
        monkeypatch.setattr(ch, "FENCED_TYPES", {"table", "formula", "list"})
        chunks, _ = ch.chunk_document([
            {"type": "title", "text": "A Paper", "page": 0},
            {"type": "caption", "text": "Listing 1: steps.", "page": 1},
            {"type": "list", "text": "first item\nsecond item", "page": 1},
        ], "A_Paper")
        body = "\n".join(c["body"] for c in chunks)
        assert "~~~list" in body
        assert ch.fences_balanced(body)

    def test_a_fence_bar_is_a_valid_boundary(self, ch):
        """Opening a labelled block is as clean a start as a capital letter,
        and closing one is as clean an end as a full stop."""
        whole = ch.fence("table", "Cora 1\nCiteseer 2")
        assert ch.starts_cleanly(whole) and ch.ends_cleanly(whole)

    def test_the_head_of_a_cut_table_starts_clean_but_ends_dirty(self, ch):
        """Both halves of a severed table, judged on each edge separately."""
        head = "~~~table\nCora Citeseer\nDegree 91.67"
        assert ch.starts_cleanly(head)        # labelled, nothing severed above
        assert not ch.ends_cleanly(head)      # stops mid-table

    def test_the_tail_of_a_cut_table_starts_dirty_but_ends_clean(self, ch):
        """The prose rule alone would call this clean: the first row happens to
        begin with a capital. The bare closing bar is what gives it away."""
        tail = "Betweenness 91.88\nCloseness 91.24\n~~~"
        assert ch.starts_inside_fence(tail)
        assert not ch.starts_cleanly(tail)
        assert ch.ends_cleanly(tail)

    def test_opening_a_block_and_giving_it_nothing_is_a_dirty_end(self, ch):
        assert not ch.ends_cleanly("Some prose.\n~~~table")


class TestFormulaBinding:
    """A formula belongs with the sentence that introduces it and the prose
    that explains its terms. `~~~` is not a boundary for a formula the way it
    is for a table."""

    def _doc(self, lead, expl, **kw):
        blocks = [{"type": "title", "text": "P", "page": 0},
                  {"type": "paragraph", "text": lead, "page": 1},
                  {"type": "formula", "text": "euv = phi(hu, hv), (7)", "page": 1},
                  {"type": "paragraph", "text": expl, "page": 1}]
        return blocks, kw

    def test_unpunctuated_lead_binds_to_the_formula(self, ch):
        blocks, kw = self._doc("The edge representation is computed as",
                               "where phi is a pairwise mapping.")
        bound = ch.bind_formula_groups(blocks[1:])
        assert [b["type"] for b in bound] == ["bound"]
        assert [m["type"] for m in bound[0]["members"]] == ["paragraph", "formula", "paragraph"]

    def test_a_properly_closed_lead_is_left_alone(self, ch):
        """It ends a sentence, so the formula does not complete it."""
        blocks, kw = self._doc("We now turn to the encoder.",
                               "where phi is a pairwise mapping.")
        bound = ch.bind_formula_groups(blocks[1:])
        assert [b["type"] for b in bound] == ["paragraph", "bound"]
        assert [m["type"] for m in bound[1]["members"]] == ["formula", "paragraph"]

    def test_an_uppercase_follower_is_a_new_thought_not_an_explanation(self, ch):
        blocks, kw = self._doc("We now turn to the encoder.",
                               "Results appear in Table 2.")
        bound = ch.bind_formula_groups(blocks[1:])
        assert [b["type"] for b in bound] == ["paragraph", "formula", "paragraph"]

    def test_consecutive_formulas_bind_as_one_run(self, ch):
        blocks = [{"type": "paragraph", "text": "The updates are given by", "page": 1},
                  {"type": "formula", "text": "a = b + c, (1)", "page": 1},
                  {"type": "formula", "text": "d = e + f, (2)", "page": 1},
                  {"type": "paragraph", "text": "where b and e are learned.", "page": 1}]
        bound = ch.bind_formula_groups(blocks)
        assert len(bound) == 1
        assert [m["type"] for m in bound[0]["members"]] == [
            "paragraph", "formula", "formula", "paragraph"]

    def test_oversize_group_keeps_n_lead_sentences_with_the_formula(self, ch):
        lead = " ".join(f"Sentence number {i} with padding to make it long." for i in range(40))
        lead += " The supervised loss is then formulated as"
        blocks = [{"type": "title", "text": "P", "page": 0},
                  {"type": "paragraph", "text": lead, "page": 1},
                  {"type": "formula", "text": "L = -1/N sum_i y_i log(p_i)", "page": 1},
                  {"type": "paragraph", "text": "where N is the edge count.", "page": 1}]
        for n in (0, 2, 4):
            chunks, _ = ch.chunk_document(blocks, "P", lead_sentences=n)
            last = chunks[-1]["body"]
            # the formula never loses its explanation, whatever n is
            assert "~~~formula" in last and "where N is the edge count." in last
            head = last.split("~~~formula")[0].strip()
            kept = len(ch.split_sentences(head)) if head else 0
            assert kept == n
            assert all(len(c["body"]) <= 2000 for c in chunks)


class TestSentenceClosure:
    def test_only_full_stop_question_and_exclamation_close(self, ch):
        for closed in ("Done.", "Really?", "Stop!", 'He said "go."', "(see Fig. 3.)"):
            assert ch.ends_cleanly(closed), closed
        for open_ in ("formulated as:", "and thus;", "no mark", "a list of("):
            assert not ch.ends_cleanly(open_), open_

    def test_a_chunk_does_not_close_on_a_colon(self, ch):
        """It should reach for the next unit rather than end mid-thought."""
        lead = "x" * 1400 + " The loss is formulated as:"
        chunks, _ = ch.chunk_document(
            [{"type": "title", "text": "P", "page": 0},
             {"type": "paragraph", "text": lead, "page": 1},
             {"type": "paragraph", "text": "It balances the two terms.", "page": 1}],
            "P", target_chars=1500, max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0]["body"].endswith("It balances the two terms.")


class TestParagraphPriority:
    def test_a_paragraph_that_must_be_split_starts_its_own_chunk(self, ch):
        short = "A complete short paragraph."
        long = " ".join(f"Sentence {i} padded out to a reasonable length here." for i in range(60))
        chunks, _ = ch.chunk_document(
            [{"type": "title", "text": "P", "page": 0},
             {"type": "paragraph", "text": short, "page": 1},
             {"type": "paragraph", "text": long, "page": 1}],
            "P", target_chars=1500, max_chars=2000)
        # the complete paragraph is not contaminated by a fragment of the long one
        assert chunks[0]["body"] == short
        assert len(chunks) > 2
