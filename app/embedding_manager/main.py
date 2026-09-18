"""Embedding manager service (port 4001).

Background loop: every paper with registry status "processed" is chunked
(structure-aware, see chunking.py), embedded, and stored in ChromaDB;
status becomes "embedded" — or "error" (with the message) if it fails.

    GET    /v1/health
    POST   /v1/search        {query, n_results, filenames?, sources?, n_chat_results?}
    GET    /v1/papers        embedded paper stems
    POST   /v1/chats         {chat_id, title, created_at, pairs:[{question, answer}]}
    DELETE /v1/chats/{id}
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from common.logsetup import get_logger
from common.paths import EMBEDDING_PORT
from common.registry_client import RegistryClient, RegistryUnavailable
from config import MODEL_NAME, PROCESSED_DIR
from embeddings import chunk_and_embed, delete_chat, embed_chat, list_embedded_files, search

log = get_logger("embedding")


def chunks_and_embed_loop(poll_interval=60):
    registry = RegistryClient()
    while True:
        try:
            for paper in registry.list_papers(status="processed"):
                stem = paper["filename"]
                json_path = Path(PROCESSED_DIR) / f"{stem}.json"
                if not json_path.exists():
                    registry.report_error(stem, f"processed JSON missing: {json_path.name}")
                    continue
                try:
                    result = chunk_and_embed(json_path)
                except Exception as e:
                    log.exception("embedding failed for %s", stem)
                    registry.report_error(stem, f"embedding: {e}")
                    continue
                if result["success"]:
                    registry.set_status(stem, "embedded")
                else:
                    registry.report_error(stem, f"embedding: {result.get('error')}")
        except RegistryUnavailable as e:
            log.warning("%s", e)
        except Exception:
            log.exception("embed loop error")
        time.sleep(poll_interval)


class SearchRequest(BaseModel):
    query: str
    n_results: int = 8
    filenames: Optional[List[str]] = None
    sources: List[str] = ["papers"]
    n_chat_results: int = 2

    @field_validator("query")
    @classmethod
    def non_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Missing query.")
        return v


class ChatEmbedRequest(BaseModel):
    chat_id: str
    title: str
    created_at: str
    pairs: List[Dict[str, str]]


embedding = FastAPI(title="embedding_manager", version="1")


@embedding.get("/v1/health")
def health():
    """Includes the device actually in use — it may be CPU even when cuda was
    requested, if the GPU was full when the model loaded."""
    from embeddings import current_device
    return {"status": "ok", "device": current_device(),
            "requested_device": os.environ.get("EMBED_DEVICE", "cuda"),
            "model": MODEL_NAME}


@embedding.post("/v1/search")
def search_endpoint(req: SearchRequest):
    return {"query": req.query,
            "results": search(req.query, req.n_results, req.filenames, req.sources, req.n_chat_results)}


@embedding.get("/v1/papers")
def papers():
    return {"papers": list_embedded_files()}


@embedding.post("/v1/chats")
def embed_chat_endpoint(req: ChatEmbedRequest):
    return {"embedded": embed_chat(req.chat_id, req.title, req.created_at, req.pairs)}


@embedding.delete("/v1/chats/{chat_id}")
def delete_chat_endpoint(chat_id: str):
    delete_chat(chat_id)
    return {"success": True}


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    threading.Thread(target=chunks_and_embed_loop, daemon=True).start()
    uvicorn.run("main:embedding", host="127.0.0.1", port=EMBEDDING_PORT, reload=False, log_level="warning")
