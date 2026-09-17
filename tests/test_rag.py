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


CFG = {"retrieval": {"n_results": 6, "n_results_with_web": 4, "use_chats": False,
                     "web_results": 3, "web_chars_per_page": 2000}, "llm": {"backend": "local"}}


class TestRecency:
    """"What's new?" is about arrival dates, which no meaning-based search finds."""

    def test_windows(self, rag):
        assert rag.recency_window("What are the latest updates today?") == 1   # narrowest wins
        assert rag.recency_window("anything new this week?") == 7
        assert rag.recency_window("papers from the last 3 days") == 3
        assert rag.recency_window("What is task-conditioned routing?") is None

    def _registry(self, rag, monkeypatch, papers):
        import common.registry_client as rc
        monkeypatch.setattr(rc, "RegistryClient", lambda *a, **k: type("R", (), {
            "list_papers": lambda self: papers})())

    def test_recent_question_uses_arrival_dates_not_search(self, rag, monkeypatch):
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        self._registry(rag, monkeypatch, [
            {"filename": "New_Paper", "status": "embedded", "embedded_at": today},
            {"filename": "Old_Paper", "status": "embedded", "embedded_at": old},
        ])
        asked = []

        def chunks(query, filenames=None, *a, **k):
            asked.append(filenames)
            return [paper("body", title=filenames[0], filename=filenames[0])]
        monkeypatch.setattr(rag, "fetch_chunks", chunks)
        context, sources, warnings = rag.retrieve("What's new today?", None, web=False, cfg=CFG)
        assert asked == [["New_Paper"]]                      # only the fresh one
        assert "added " in sources[0]["label"] and not warnings
        assert "already been filtered by arrival date" in context   # the model must not re-judge

    def test_empty_window_falls_back_to_newest_with_a_warning(self, rag, monkeypatch):
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        self._registry(rag, monkeypatch, [{"filename": "Old_Paper", "status": "embedded", "embedded_at": old}])
        monkeypatch.setattr(rag, "fetch_chunks",
                            lambda q, f=None, *a, **k: [paper("body", title=f[0], filename=f[0])])
        _, sources, warnings = rag.retrieve("anything new today?", None, web=False, cfg=CFG)
        assert len(sources) == 1 and warnings and "nothing new in the last 1 day" in warnings[0]

    def test_selected_papers_keep_normal_search(self, rag, monkeypatch):
        monkeypatch.setattr(rag, "fetch_chunks", lambda *a, **k: [paper("t", title="P")])
        _, sources, _ = rag.retrieve("latest results?", ["Chosen_Paper"], web=False, cfg=CFG)
        assert sources and "added" not in sources[0]["label"]


class TestArrivalCounts:
    def test_how_many_reads_the_number_asked_for(self, rag):
        assert rag.how_many("What's the 10 new updates since yesterday?") == 10
        assert rag.how_many("top 3 papers this week") == 3
        assert rag.how_many("papers from the last 3 days") is None      # that 3 is the window
        assert rag.how_many("latest 20 papers") == rag.MAX_ARRIVALS     # context has a ceiling

    def test_day_label_says_today_and_yesterday(self, rag):
        from datetime import datetime, timedelta
        now = datetime(2026, 9, 18, 9, 0)
        assert rag.day_label("2026-09-18 03:00:00", now) == "today"
        assert rag.day_label("2026-09-17 22:00:00", now) == "yesterday"
        assert rag.day_label("2026-09-10 10:00:00", now) == "2026-09-10"
        assert rag.day_label("", now) == "an unknown date"

    def test_recency_question_carries_its_own_instruction(self, rag):
        asked = rag.question_for("What's new since yesterday?")
        assert "do not reply that nothing is new" in asked.lower()
        assert rag.question_for("What is task-conditioned routing?") == "What is task-conditioned routing?"

    def test_asking_for_more_than_arrived_warns(self, rag, monkeypatch):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        import common.registry_client as rc
        monkeypatch.setattr(rc, "RegistryClient", lambda *a, **k: type("R", (), {
            "list_papers": lambda self: [{"filename": "One", "status": "embedded", "embedded_at": today}]})())
        monkeypatch.setattr(rag, "fetch_chunks",
                            lambda q, f=None, *a, **k: [paper("body", title=f[0], filename=f[0])])
        _, sources, warnings = rag.retrieve("the 10 new papers today", None, web=False, cfg=CFG)
        assert len(sources) == 1 and any("only 1 paper" in w for w in warnings)


class TestOpeningChunk:
    """A "what is this paper about" excerpt must not be the references page."""

    def chunk(self, rag, section, text, page=0):
        return {"text": text, "metadata": {"section": section, "page_start": page}, "distance": 0.1}

    def test_abstract_beats_index_terms_and_references(self, rag):
        body = "x" * 800
        hits = [self.chunk(rag, "", "Index Terms—Model Context Protocol, TLS"),
                self.chunk(rag, "REFERENCES", body, page=5),
                self.chunk(rag, "1 Introduction", body, page=1),
                self.chunk(rag, "Abstract", body)]
        assert sorted(hits, key=rag.opening_rank)[0]["metadata"]["section"] == "Abstract"

    def test_continued_on_next_page_is_last_resort(self, rag):
        hits = [self.chunk(rag, "E.1 Metadata Strategy Families", "Continued on next page", page=32),
                self.chunk(rag, "V. RESULTS", "y" * 600, page=2)]
        assert sorted(hits, key=rag.opening_rank)[0]["metadata"]["section"] == "V. RESULTS"
