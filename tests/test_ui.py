"""Smoke tests for the UI service (no Ollama/embedding/registry needed —
everything degrades gracefully or is mocked)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")  # required by fastapi.testclient
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    spec = importlib.util.spec_from_file_location("ui_main", REPO_ROOT / "UI" / "main.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for k in ("config", "rag", "websearch"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
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
    monkeypatch.setattr(ui.rag, "retrieve", lambda q, f, w, cfg: (
        "[1] (from: P)\ntext", [{"n": 1, "kind": "paper", "label": "from: P", "text": "text"}], ["web off"]))
    monkeypatch.setattr(ui.rag, "answer_stream", lambda c, q, cfg: iter(["Hello ", "[1]"]))
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


def test_ingestion_without_registry(client):
    body = client.get("/v1/ingestion").json()
    assert "services" in body and "files" in body


def test_logs_stream_rejects_unknown_service(client):
    assert client.get("/v1/logs/stream?service=nope").status_code == 404
