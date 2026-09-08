"""Web UI service (port 4002).

Serves the React build from UI/static/ (or the legacy single-page
UI/index.html if no build exists) and the JSON API it uses:

    POST /v1/query              {query, filenames?, web?, chat_id?}  -> ndjson stream:
                                {type: chat}, {type: sources}, {type: warning}*, {type: delta}*, {type: done|error}
    GET  /v1/status             ollama / embedding / backend health
    GET  /v1/papers             embedded paper stems (sidebar filter)
    GET  /v1/settings           effective settings (secrets masked)
    PUT  /v1/settings           merge + save
    GET  /v1/ingestion          counts, stage timings, ETA, services, disk
    GET  /v1/logs/stream?service=parse   server-sent events tailing run/logs/<service>.log
    GET  /v1/chats  POST /v1/chats  GET/PATCH/DELETE /v1/chats/{id}  POST /v1/chats/{id}/embed
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

UI_DIR = Path(__file__).resolve().parent
REPO_ROOT = UI_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag_setup"))

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import rag
from common import settings as settings_store
from common.chatstore import ChatStore
from common.logsetup import get_logger
from common.paths import EMBEDDING_URL, LOG_DIR, PARSED_DIR, PDF_DIR, PID_DIR, PROCESSED_DIR, REGISTRY_URL
from common.registry_client import RegistryClient, RegistryUnavailable

log = get_logger("ui")
app = FastAPI(title="ui", version="1")
chats = ChatStore()
STATIC_DIR = UI_DIR / "static"
LEGACY_INDEX = UI_DIR / "index.html"
LOG_SERVICES = ("registry", "parse", "embedding", "prune", "ui", "download", "daily-ingest")


# --------------------------------------------------------------- models
class QueryRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None
    web: bool = False
    chat_id: Optional[str] = None


class ChatCreate(BaseModel):
    title: str = "New chat"


class ChatPatch(BaseModel):
    title: str


# --------------------------------------------------------------- status
@app.get("/v1/status")
def status():
    cfg = settings_store.load()
    llm = cfg["llm"]
    ollama_up, model_pulled = False, False
    model = llm["local"].get("model", "")
    try:
        r = requests.get(f"{llm['local'].get('url', 'http://127.0.0.1:11434')}/api/tags", timeout=3)
        ollama_up = r.ok
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model_pulled = model in models or f"{model}:latest" in models
    except requests.RequestException:
        pass
    embeddings_up = False
    try:
        embeddings_up = requests.get(f"{EMBEDDING_URL}/v1/health", timeout=3).ok
    except requests.RequestException:
        pass
    oa = llm.get("openai", {})
    return {
        "backend": llm.get("backend", "local"),
        "ollama": ollama_up, "model": model, "model_pulled": model_pulled,
        "openai_configured": bool(oa.get("base_url") and oa.get("model")),
        "openai_model": oa.get("model", ""),
        "embeddings": embeddings_up,
        "registry": RegistryClient().health(),
    }


@app.get("/v1/papers")
def papers():
    try:
        return {"papers": requests.get(f"{EMBEDDING_URL}/v1/papers", timeout=10).json().get("papers", [])}
    except requests.RequestException:
        return {"papers": []}


# --------------------------------------------------------------- settings
@app.get("/v1/settings")
def get_settings():
    return settings_store.masked(settings_store.load())


@app.put("/v1/settings")
def put_settings(update: Dict[str, Any]):
    current = settings_store.load()
    saved = settings_store.save(settings_store.unmask(update, current))
    log.info("settings updated (backend=%s)", saved["llm"]["backend"])
    return settings_store.masked(saved)


# --------------------------------------------------------------- query
@app.post("/v1/query")
def query(req: QueryRequest):
    question = req.query.strip()
    if not question:
        raise HTTPException(422, "empty query")
    cfg = settings_store.load()

    def event(**kw):
        return json.dumps(kw) + "\n"

    def generate():
        chat_id = req.chat_id
        if chat_id and not chats.get(chat_id):
            chat_id = None
        if not chat_id:
            chat_id = chats.create(question[:80])["id"]
        yield event(type="chat", chat_id=chat_id)
        chats.add_message(chat_id, "user", question)

        try:
            context, sources, warnings = rag.retrieve(question, req.filenames or None, req.web, cfg)
        except requests.RequestException as e:
            yield event(type="error", message=f"Embedding service unreachable: {e}")
            return
        except Exception as e:
            yield event(type="error", message=f"Retrieval failed: {e}")
            return
        yield event(type="sources", sources=sources, backend=cfg["llm"]["backend"])
        for w in warnings:
            yield event(type="warning", message=w)

        text = ""
        try:
            for delta in rag.answer_stream(context, question, cfg):
                text += delta
                yield event(type="delta", text=delta)
        except Exception as e:
            log.exception("generation failed")
            yield event(type="error", message=str(e))
            return
        chats.add_message(chat_id, "assistant", text, sources=[{k: v for k, v in s.items() if k != "text"} for s in sources])
        yield event(type="done")

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# --------------------------------------------------------------- chats
@app.get("/v1/chats")
def list_chats():
    return {"chats": chats.list()}


@app.post("/v1/chats")
def create_chat(body: ChatCreate):
    return chats.create(body.title)


@app.get("/v1/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    return chat


@app.patch("/v1/chats/{chat_id}")
def rename_chat(chat_id: str, body: ChatPatch):
    if not chats.rename(chat_id, body.title):
        raise HTTPException(404, "no such chat")
    return {"success": True}


@app.delete("/v1/chats/{chat_id}")
def delete_chat(chat_id: str):
    if not chats.delete(chat_id):
        raise HTTPException(404, "no such chat")
    try:
        requests.delete(f"{EMBEDDING_URL}/v1/chats/{chat_id}", timeout=10)
    except requests.RequestException:
        pass
    return {"success": True}


@app.post("/v1/chats/{chat_id}/embed")
def embed_chat(chat_id: str):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    pairs = chats.qa_pairs(chat_id)
    if not pairs:
        raise HTTPException(422, "chat has no question/answer pairs yet")
    try:
        r = requests.post(f"{EMBEDDING_URL}/v1/chats",
                          json={"chat_id": chat_id, "title": chat["title"],
                                "created_at": chat["created_at"], "pairs": pairs}, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(503, f"embedding service unreachable: {e}")
    chats.mark_embedded(chat_id, True)
    return {"embedded": r.json().get("embedded", 0)}


# --------------------------------------------------------------- ingestion
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@app.get("/v1/ingestion")
def ingestion():
    services = {}
    if PID_DIR.exists():
        for pidfile in PID_DIR.glob("*.pid"):
            try:
                pid = int(pidfile.read_text().strip())
            except ValueError:
                continue
            services[pidfile.stem] = {"pid": pid, "running": _alive(pid)}
    out: Dict[str, Any] = {"services": services, "registry": None, "eta_seconds": None,
                           "files": {name: (len([p for p in d.glob("*") if p.is_file()]) if d.exists() else 0)
                                     for name, d in (("raw_pdfs", PDF_DIR), ("parsed", PARSED_DIR), ("processed", PROCESSED_DIR))}}
    try:
        stats = RegistryClient().stats()
    except RegistryUnavailable:
        return out
    out["registry"] = stats
    counts = stats["counts"]
    remaining = stats["total"] - counts.get("embedded", 0) - counts.get("error", 0)
    out["remaining"] = remaining
    if remaining:
        # throughput from papers embedded in the last 15 minutes, else stage averages
        now = datetime.now(timezone.utc)
        recent = [datetime.fromisoformat(t) for t in stats.get("recent_embedded_at", [])]
        recent = [t for t in recent if (now - t).total_seconds() < 900]
        if len(recent) >= 2:
            span = (max(recent) - min(recent)).total_seconds() or 1
            out["eta_seconds"] = int(remaining / (len(recent) / span))
        else:
            per_paper = sum(v for v in (stats["stage_seconds"].get(k) for k in ("parse", "process", "embed")) if v) or 30
            out["eta_seconds"] = int(remaining * per_paper)
    return out


@app.get("/v1/logs/stream")
async def logs_stream(service: str = Query("parse"), lines: int = Query(100, ge=0, le=2000)):
    if service not in LOG_SERVICES:
        raise HTTPException(404, f"unknown service; one of {LOG_SERVICES}")
    path = LOG_DIR / f"{service}.log"

    async def events():
        pos = 0
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-lines:] if lines else []
                for line in tail:
                    yield {"data": line.rstrip("\n")}
                f.seek(0, os.SEEK_END)
                pos = f.tell()
        while True:
            await asyncio.sleep(1)
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                for line in f:
                    yield {"data": line.rstrip("\n")}
                pos = f.tell()

    return EventSourceResponse(events())


# --------------------------------------------------------------- static
if (STATIC_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = STATIC_DIR / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(LEGACY_INDEX)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4002, reload=False, log_level="warning")
