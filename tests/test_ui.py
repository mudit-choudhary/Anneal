"""Smoke tests for the UI service (no Ollama/embedding/registry needed —
everything degrades gracefully or is mocked)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")  # required by fastapi.testclient
from fastapi.testclient import TestClient

APP_ROOT = Path(__file__).resolve().parent.parent / "app"


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    # Isolate the chat DB and settings file from the real data/ directory.
    tmp = tmp_path_factory.mktemp("ui")
    import common.paths as paths
    import common.settings as settings_store
    import common.chatstore as chatstore
    paths.APP_DB = tmp / "app.db"
    chatstore.APP_DB = tmp / "app.db"
    settings_store.SETTINGS_FILE = tmp / "settings.json"

    saved = {k: sys.modules.pop(k) for k in ("config", "rag", "websearch") if k in sys.modules}
    spec = importlib.util.spec_from_file_location("ui_main", APP_ROOT / "UI" / "main.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for k in ("config", "rag", "websearch"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)

    # Refuse to run against the real chat database. These tests create and
    # delete chats in bulk; if the redirection above ever stops working, the
    # damage is silent and permanent. Fail loudly instead.
    live = str((APP_ROOT / "data" / "app.db").resolve())
    assert module.chats.db_path != live, (
        f"the UI service opened the real chat database at {live} — "
        "aborting before the tests destroy it")
    assert str(tmp) in module.chats.db_path
    return module


@pytest.fixture(scope="module")
def client(ui):
    return TestClient(ui.app)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_status_degrades_gracefully(client):
    body = client.get("/v1/status").json()
    assert {"ollama", "embeddings", "model", "backend", "registry", "openai_configured"} <= set(body)


def test_papers_empty_when_service_down(client):
    assert isinstance(client.get("/v1/papers").json()["papers"], list)


def test_settings_roundtrip_masks_secret(client):
    r = client.put("/v1/settings", json={"llm": {"backend": "openai",
                                                 "openai": {"base_url": "http://gw:8080/v1", "api_key": "sk-secret-1234", "model": "m"}}})
    assert r.status_code == 200
    body = r.json()
    assert body["llm"]["backend"] == "openai"
    assert body["llm"]["openai"]["api_key"].startswith("••••") and body["llm"]["openai"]["api_key"].endswith("1234")
    # sending the masked value back must not overwrite the stored secret
    client.put("/v1/settings", json={"llm": {"openai": {"api_key": body["llm"]["openai"]["api_key"], "model": "m2"}}})
    import common.settings as settings_store
    assert settings_store.load()["llm"]["openai"]["api_key"] == "sk-secret-1234"
    client.put("/v1/settings", json={"llm": {"backend": "local"}})


def test_query_reports_unreachable_embedding_service(client, ui, monkeypatch):
    import requests

    def down(*args, **kwargs):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(ui.rag, "retrieve", down)
    r = client.post("/v1/query", json={"query": "what is x?"})
    events = [json.loads(l) for l in r.text.strip().split("\n")]
    assert events[0]["type"] == "chat" and events[0]["chat_id"]
    assert events[-1]["type"] == "error" and "unreachable" in events[-1]["message"]


def test_query_streams_and_persists_chat(client, ui, monkeypatch):
    monkeypatch.setattr(ui.rag, "retrieve", lambda q, f, w, cfg, history=None: (
        "[1] (from: P)\ntext", [{"n": 1, "kind": "paper", "label": "from: P", "text": "text"}], ["web off"]))
    monkeypatch.setattr(ui.rag, "answer_stream",
                        lambda c, q, cfg, history=None: iter(["Hello ", "[1]"]))
    r = client.post("/v1/query", json={"query": "hi?", "web": True})
    events = [json.loads(l) for l in r.text.strip().split("\n")]
    types = [e["type"] for e in events]
    assert types == ["chat", "sources", "warning", "delta", "delta", "done"]
    chat = client.get(f"/v1/chats/{events[0]['chat_id']}").json()
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]
    assert chat["messages"][1]["content"] == "Hello [1]"
    assert chat["messages"][1]["sources"][0]["label"] == "from: P"


def test_chat_crud(client):
    created = client.post("/v1/chats", json={"title": "T"}).json()
    assert client.patch(f"/v1/chats/{created['id']}", json={"title": "Renamed"}).json()["success"]
    assert any(c["title"] == "Renamed" for c in client.get("/v1/chats").json()["chats"])
    assert client.post(f"/v1/chats/{created['id']}/embed").status_code == 422   # no pairs yet
    assert client.delete(f"/v1/chats/{created['id']}").json()["success"]
    assert client.get(f"/v1/chats/{created['id']}").status_code == 404


class TestBulkChatDelete:
    """Deleting chats one at a time does not keep up with the debris a testing
    session leaves behind, so the list and the settings page both delete in bulk."""

    @pytest.fixture(autouse=True)
    def clean(self, ui):
        """Earlier tests in this module leave chats behind, and these assertions
        are about exact counts."""
        ui.chats.delete_many(ui.chats.ids("all"))

    def _make(self, client, ui, title, answered):
        chat = client.post("/v1/chats", json={"title": title}).json()
        ui.chats.add_message(chat["id"], "user", "q")
        if answered:
            ui.chats.add_message(chat["id"], "assistant", "a")
        return chat["id"]

    def test_deletes_a_selection_by_id(self, client, ui):
        keep = self._make(client, ui, "keep", True)
        drop = [self._make(client, ui, f"drop {i}", True) for i in range(3)]
        body = client.request("DELETE", "/v1/chats", json={"ids": drop}).json()
        assert body == {"deleted": 3, "requested": 3}
        ids = {c["id"] for c in client.get("/v1/chats").json()["chats"]}
        assert keep in ids and not (set(drop) & ids)

    def test_ids_that_no_longer_exist_are_not_an_error(self, client, ui):
        real = self._make(client, ui, "real", True)
        body = client.request("DELETE", "/v1/chats", json={"ids": [real, "gone"]}).json()
        assert body["deleted"] == 1 and body["requested"] == 2

    def test_unanswered_scope_spares_answered_chats(self, client, ui):
        answered = self._make(client, ui, "answered", True)
        self._make(client, ui, "asked only", False)
        self._make(client, ui, "asked only 2", False)
        body = client.request("DELETE", "/v1/chats", json={"scope": "unanswered"}).json()
        assert body["deleted"] == 2
        assert [c["id"] for c in client.get("/v1/chats").json()["chats"]] == [answered]

    def test_scope_all_empties_the_list(self, client, ui):
        self._make(client, ui, "a", True)
        self._make(client, ui, "b", False)
        assert client.request("DELETE", "/v1/chats", json={"scope": "all"}).json()["deleted"] >= 2
        assert client.get("/v1/chats").json()["chats"] == []

    def test_ids_and_scope_together_are_rejected(self, client):
        for payload in ({"ids": ["x"], "scope": "all"}, {}):
            r = client.request("DELETE", "/v1/chats", json=payload)
            assert r.status_code == 422

    def test_warns_when_embedded_chats_cannot_leave_the_vector_store(self, client, ui, monkeypatch):
        """A deleted chat that stays in memory would keep surfacing as a source,
        so a silent success here would be a lie."""
        import requests as rq
        chat_id = self._make(client, ui, "embedded", True)
        ui.chats.mark_embedded(chat_id, True)

        def down(*a, **k):
            raise rq.ConnectionError("refused")
        monkeypatch.setattr(ui.requests, "delete", down)
        body = client.request("DELETE", "/v1/chats", json={"ids": [chat_id]}).json()
        assert body["deleted"] == 1
        assert "vector store" in body["note"] and "not reachable" in body["note"]


class TestSchedule:
    """Daily multi-topic downloads. The systemd side is stubbed — these check
    the contract and the validation, not that systemd works."""

    def test_reports_state_without_a_timer_installed(self, client):
        body = client.get("/v1/schedule").json()
        assert {"time", "topics", "installed", "enabled", "linger"} <= set(body)
        assert isinstance(body["topics"], list)

    def test_saves_topics_with_their_own_caps(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_systemctl", lambda *a, **k: (True, "enabled"))
        body = client.put("/v1/schedule", json={"topics": [
            {"topic": "Graph Neural Networks", "max_papers": 3, "enabled": True},
            {"topic": "Quantum ML", "max_papers": 7, "enabled": False},
        ]}).json()
        assert [t["topic"] for t in body["topics"]] == ["Graph Neural Networks", "Quantum ML"]
        assert body["topics"][0]["max_papers"] == 3
        assert body["topics"][1]["enabled"] is False

    def test_blank_topics_are_dropped_and_caps_clamped(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_systemctl", lambda *a, **k: (True, ""))
        body = client.put("/v1/schedule", json={"topics": [
            {"topic": "   ", "max_papers": 5},
            {"topic": "Real", "max_papers": 9999},
        ]}).json()
        assert len(body["topics"]) == 1
        assert body["topics"][0]["max_papers"] == 200

    def test_rejects_a_malformed_time(self, client):
        assert client.put("/v1/schedule", json={"time": "3pm"}).status_code == 422
        assert client.put("/v1/schedule", json={"time": "25:00"}).status_code == 422

    def test_warns_when_the_timer_will_not_survive_logout(self, client, ui, monkeypatch):
        monkeypatch.setattr(ui, "_systemctl", lambda *a, **k: (True, "enabled"))
        monkeypatch.setattr(ui, "get_schedule",
                            lambda: {"time": "03:00", "topics": [], "installed": True,
                                     "enabled": True, "next_run": None, "last_run": None,
                                     "linger": False})
        body = client.put("/v1/schedule", json={"enabled": True}).json()
        assert "catch-up run starts now" in body["note"]
        assert "enable-linger" in body["note"]


class TestCoverage:
    """The check that would have caught two silently truncated papers."""

    def test_flags_a_parse_that_missed_pages(self, tmp_path, monkeypatch):
        import common.coverage as cov
        monkeypatch.setattr(cov, "PROCESSED_DIR", tmp_path)
        (tmp_path / "half.json").write_text(json.dumps({
            "num_pages": 1, "pdf_pages": 11,
            "blocks": [{"text": "x" * 900} for _ in range(6)],
        }))
        row = cov.inspect("half")
        assert row["problems"]
        assert "parsed 1 of 11 pages" in row["problems"][0]

    def test_accepts_a_complete_parse(self, tmp_path, monkeypatch):
        import common.coverage as cov
        monkeypatch.setattr(cov, "PROCESSED_DIR", tmp_path)
        (tmp_path / "whole.json").write_text(json.dumps({
            "num_pages": 10, "pdf_pages": 10,
            "blocks": [{"text": "x" * 900} for _ in range(40)],
        }))
        assert cov.inspect("whole")["problems"] == []

    def test_flags_a_paper_with_almost_no_text(self, tmp_path, monkeypatch):
        import common.coverage as cov
        monkeypatch.setattr(cov, "PROCESSED_DIR", tmp_path)
        (tmp_path / "thin.json").write_text(json.dumps({
            "num_pages": 10, "pdf_pages": 10, "blocks": [{"text": "tiny"}],
        }))
        problems = " ".join(cov.inspect("thin")["problems"])
        assert "block" in problems and "implausibly thin" in problems

    def test_audit_endpoint_answers(self, client):
        body = client.get("/v1/audit").json()
        assert {"checked", "affected", "unverifiable", "papers"} <= set(body)

    def test_repair_needs_filenames(self, client):
        assert client.post("/v1/audit/repair", json={"filenames": []}).status_code == 422

    def test_repair_skips_a_paper_whose_pdf_is_gone(self, client):
        body = client.post("/v1/audit/repair",
                           json={"filenames": ["definitely-not-a-real-paper"]}).json()
        assert body["repaired"] == []
        assert "PDF is gone" in body["skipped"][0]["why"]


class TestPickPapers:
    def test_rejects_nothing_usable(self, client):
        assert client.post("/v1/ingest/pick", json={"urls": []}).status_code == 422
        assert client.post("/v1/ingest/pick", json={"urls": ["notaurl"]}).status_code == 422

    def test_caps_the_batch_size(self, client):
        many = [f"https://arxiv.org/abs/2401.{i:05d}" for i in range(60)]
        assert client.post("/v1/ingest/pick", json={"urls": many}).status_code == 422

    def test_preview_needs_a_topic(self, client):
        assert client.get("/v1/ingest/arxiv/preview?topic=%20%20").status_code == 422


def test_ingestion_without_registry(client):
    body = client.get("/v1/ingestion").json()
    assert "services" in body and "files" in body


def test_logs_stream_rejects_unknown_service(client):
    assert client.get("/v1/logs/stream?service=nope").status_code == 404


class TestServiceControl:
    """The control pane drives scripts/ops.py; these check the guards, not
    the process management itself (which would start real services)."""

    def test_lists_every_service_with_purpose(self, client):
        body = client.get("/v1/services").json()["services"]
        names = {s["name"] for s in body}
        assert {"registry", "parse", "embedding", "prune", "download", "ui"} <= names
        for s in body:
            assert s["title"] and s["purpose"]
            assert isinstance(s["running"], bool)
        assert [s for s in body if s["name"] == "ui"][0]["self"] is True

    def test_ui_cannot_stop_or_restart_itself(self, client):
        for action in ("stop", "restart"):
            r = client.post("/v1/services/ui", json={"action": action})
            assert r.status_code == 409
            assert "cannot stop itself" in r.json()["detail"]

    def test_unknown_service_and_action_rejected(self, client):
        assert client.post("/v1/services/nope", json={"action": "start"}).status_code == 404
        assert client.post("/v1/services/parse", json={"action": "explode"}).status_code == 422

    def test_start_is_a_noop_when_already_running(self, client, ui, monkeypatch):
        ops = ui._ops()
        monkeypatch.setattr(ops, "running", lambda name: True)
        started = []
        monkeypatch.setattr(ops, "start", lambda *a, **k: started.append(a))
        body = client.post("/v1/services/parse", json={"action": "start"}).json()
        assert body["note"] == "already running" and started == []

    def test_embedding_device_is_passed_through(self, client, ui, monkeypatch):
        ops = ui._ops()
        calls = {}
        monkeypatch.setattr(ops, "running", lambda name: False)
        monkeypatch.setattr(ops, "port_owner", lambda port: None)
        monkeypatch.setattr(ops, "start", lambda name, env=None: calls.update(name=name, env=env))
        client.post("/v1/services/embedding", json={"action": "start", "embed_device": "cuda"})
        assert calls["env"] == {"EMBED_DEVICE": "cuda"}

    def test_preset_starts_the_right_set(self, client, ui, monkeypatch):
        ops = ui._ops()
        started = []
        monkeypatch.setattr(ops, "running", lambda name: False)
        monkeypatch.setattr(ops, "port_owner", lambda port: None)
        monkeypatch.setattr(ops, "start", lambda name, env=None: started.append((name, env)))

        client.post("/v1/services/preset/query")
        assert [n for n, _ in started] == ["registry", "embedding"]
        assert dict(started)["embedding"] == {"EMBED_DEVICE": "cpu"}

        started.clear()
        r = client.post("/v1/services/preset/ingest").json()
        assert [n for n, _ in started] == ["registry", "embedding", "parse", "prune"]
        assert dict(started)["embedding"] == {"EMBED_DEVICE": "cuda"}
        assert r["embed_device"] == "cuda"

    def test_unknown_preset_rejected(self, client):
        assert client.post("/v1/services/preset/nope").status_code == 404

    def _stub_embedding_start(self, client, ui, monkeypatch, actual_device):
        """Start the embedder with a health endpoint reporting `actual_device`."""
        ops = ui._ops()
        monkeypatch.setattr(ops, "running", lambda name: False)
        monkeypatch.setattr(ops, "port_owner", lambda port: None)
        monkeypatch.setattr(ops, "start", lambda name, env=None: None)
        monkeypatch.setattr(ui.time, "sleep", lambda s: None)

        class R:
            def json(self):
                return {"status": "ok", "device": actual_device}
        monkeypatch.setattr(ui.requests, "get", lambda *a, **k: R())
        monkeypatch.setattr(ui, "_free_vram_mb", lambda: 24)
        return client.post("/v1/services/embedding",
                           json={"action": "start", "embed_device": "cuda"}).json()

    def test_reports_when_the_embedder_lands_on_the_gpu(self, client, ui, monkeypatch):
        body = self._stub_embedding_start(client, ui, monkeypatch, "cuda")
        assert body["device"] == "cuda"
        assert "running on CUDA" in body["note"]

    def test_warns_when_gpu_was_asked_for_but_cpu_was_used(self, client, ui, monkeypatch):
        """The silent CPU fallback is exactly what cost two papers before; the
        control pane must say so rather than report a bare success."""
        body = self._stub_embedding_start(client, ui, monkeypatch, "cpu")
        assert body["device"] == "cpu"
        assert "not CUDA" in body["note"] and "24 MB free" in body["note"]
