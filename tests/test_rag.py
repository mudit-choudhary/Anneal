"""Unit tests for retrieval context construction (no services required)."""

import importlib
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent / "app"


@pytest.fixture(scope="module")
def rag():
    saved = {k: sys.modules.pop(k) for k in ("config", "rag", "websearch") if k in sys.modules}
    sys.path.insert(0, str(APP_ROOT))
    sys.path.insert(0, str(APP_ROOT / "rag_setup"))
    try:
        module = importlib.import_module("rag")
    finally:
        sys.path.pop(0)
        sys.modules.pop("config", None)
        sys.modules.update(saved)
    return module


def paper(text, **meta):
    return {"text": text, "metadata": meta, "distance": 0.3, "source": "papers"}


def test_build_context_numbers_and_labels(rag):
    ctx, sources = rag.build_context([
        paper("First chunk text.", filename="A_Great_Paper", title="A Great Paper", page_start=3, page_end=3),
        paper("Second chunk text.", filename="Another_One"),
    ])
    assert "[1] (from: A Great Paper, p.4)\nFirst chunk text." in ctx
    assert "[2] (from: Another One)\nSecond chunk text." in ctx
    assert [s["n"] for s in sources] == [1, 2]
    assert sources[0]["kind"] == "paper" and sources[0]["section"] is None


def test_chats_and_web_sources_labeled(rag):
    ctx, sources = rag.build_context(
        [paper("Paper text.", title="P"),
         {"text": "Q: x\nA: y", "metadata": {"chat_id": "c1", "title": "Old chat", "created_at": "2026-09-01T10:00:00"},
          "distance": 0.4, "source": "chats"}],
        web_pages=[{"title": "Site", "url": "https://example.org/a", "text": "Web text."}],
    )
    assert [s["kind"] for s in sources] == ["paper", "chat", "web"]
    assert "[2] (previous conversation: Old chat, 2026-09-01)" in ctx
    assert "[3] (web: Site — https://example.org/a)\nWeb text." in ctx


def test_source_label_fallbacks(rag):
    assert rag.source_label({}) == "unknown"
    assert rag.source_label({"filename": "A_B.txt"}) == "A B"
    assert rag.source_label({"title": "T", "page_start": 0, "page_end": 2}) == "T, pp.1-3"


def test_system_prompt_demands_citations(rag):
    assert "cite" in rag.SYSTEM_PROMPT.lower()
    assert "ONLY" in rag.SYSTEM_PROMPT


def test_thinking_filter_swallows_leaked_block(rag):
    out = "".join(rag._filter_thinking(iter(["<thi", "nk>reasoning…</think>\n\nAnswer ", "here."])))
    assert out == "Answer here."
    assert "".join(rag._filter_thinking(iter(["plain ", "answer"]))) == "plain answer"


def test_answer_stream_dispatches_on_backend(rag, monkeypatch):
    monkeypatch.setattr(rag, "_ollama_stream", lambda c, q, local, history=None: iter(["local:" + local["model"]]))
    monkeypatch.setattr(rag, "_openai_stream", lambda c, q, oa, history=None: iter(["openai:" + oa["model"]]))
    cfg = {"llm": {"backend": "local", "local": {"model": "m1"}, "openai": {"model": "m2"}}}
    assert rag.answer(ctx := "c", "q", cfg) == "local:m1"
    cfg["llm"]["backend"] = "openai"
    assert rag.answer(ctx, "q", cfg) == "openai:m2"


class TestConversationHistory:
    """A follow-up ("the gist of this paper") has no searchable content and no
    subject of its own; both retrieval and generation need the earlier turns."""

    HISTORY = [{"role": "user", "content": "What does the CELP framework do?"},
               {"role": "assistant", "content": "CELP uses community structure."}]

    def test_search_text_prepends_previous_question(self, rag):
        assert rag.search_text("give me the gist", self.HISTORY) == \
            "What does the CELP framework do? give me the gist"

    def test_search_text_unchanged_without_history(self, rag):
        assert rag.search_text("a fresh question") == "a fresh question"
        assert rag.search_text("a fresh question", []) == "a fresh question"

    def test_search_text_ignores_assistant_only_history(self, rag):
        assert rag.search_text("q", [{"role": "assistant", "content": "hi"}]) == "q"

    def test_messages_include_prior_turns_before_the_question(self, rag):
        msgs = rag._messages("CTX", "and its limitations?", self.HISTORY)
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[1]["content"] == self.HISTORY[0]["content"]
        assert msgs[-1]["content"].endswith("Question: and its limitations?")

    def test_history_is_capped(self, rag):
        long_history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        msgs = rag._messages("CTX", "now", long_history)
        assert len(msgs) == 1 + rag.MAX_HISTORY_TURNS + 1        # system + capped + question
        assert msgs[1]["content"] == "q16"                        # kept the most recent

    def test_long_assistant_answers_are_truncated(self, rag):
        history = [{"role": "assistant", "content": "x " * 5000}]
        msgs = rag._messages("CTX", "next", history)
        assert len(msgs[1]["content"]) <= rag.MAX_HISTORY_CHARS + 2
        assert msgs[1]["content"].endswith("…")

    def test_user_turns_are_never_truncated(self, rag):
        question = "y " * 2000
        msgs = rag._messages("CTX", "next", [{"role": "user", "content": question}])
        assert msgs[1]["content"] == question

    def test_answer_stream_passes_history_to_backend(self, rag, monkeypatch):
        seen = {}
        monkeypatch.setattr(rag, "_ollama_stream",
                            lambda c, q, local, history=None: seen.update(history=history) or iter(["ok"]))
        cfg = {"llm": {"backend": "local", "local": {}}}
        rag.answer("ctx", "q", cfg, history=self.HISTORY)
        assert seen["history"] == self.HISTORY


def test_retrieve_degrades_when_web_search_fails(rag, monkeypatch):
    monkeypatch.setattr(rag, "fetch_chunks", lambda *a, **k: [paper("t", title="P")])

    def boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(rag, "search_web", boom)
    cfg = {"retrieval": {"n_results": 6, "n_results_with_web": 4, "use_chats": False,
                         "web_results": 3, "web_chars_per_page": 2000}, "llm": {"backend": "local"}}
    context, sources, warnings = rag.retrieve("q", None, web=True, cfg=cfg)
    assert len(sources) == 1 and warnings and "web search unavailable" in warnings[0]
