"""Web UI for querying the RAG pipeline (port 4002).

Serves a single-page chat interface and proxies queries through the same
retrieval + answering code the CLI uses (rag_setup/rag.py). Answers from the
local Ollama backend stream token-by-token as newline-delimited JSON.
"""

import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
REPO_ROOT = UI_DIR.parent
# Reuse rag_setup's retrieval/answering code (its config.py comes with it).
sys.path.insert(0, str(REPO_ROOT / "rag_setup"))

import json
from typing import List, Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import rag
from config import EMBEDDING_URL, LLM_BACKEND, N_RESULTS, OLLAMA_MODEL, OLLAMA_URL

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None
    backend: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/status")
def status():
    ollama_up, model_pulled = False, False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_up = r.ok
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model_pulled = OLLAMA_MODEL in models or f"{OLLAMA_MODEL}:latest" in models
    except requests.RequestException:
        pass

    embeddings_up = False
    try:
        embeddings_up = requests.post(f"{EMBEDDING_URL}/healthcheck", timeout=3).ok
    except requests.RequestException:
        pass

    return {
        "ollama": ollama_up,
        "model": OLLAMA_MODEL,
        "model_pulled": model_pulled,
        "embeddings": embeddings_up,
        "default_backend": LLM_BACKEND,
    }


@app.get("/api/papers")
def papers():
    try:
        r = requests.get(f"{EMBEDDING_URL}/list_files", timeout=10)
        return {"files": r.json().get("files", [])}
    except requests.RequestException:
        return {"files": []}


@app.post("/api/query")
def query(request: QueryRequest):
    backend = request.backend or LLM_BACKEND
    filenames = request.filenames or None

    def generate():
        try:
            chunks = rag.fetch_chunks(request.query, filenames)
        except requests.RequestException as e:
            yield json.dumps({"type": "error",
                              "message": f"Embedding service unreachable: {e}"}) + "\n"
            return

        documents = chunks["documents"][:N_RESULTS]
        metadatas = chunks["metadatas"][:N_RESULTS]
        distances = chunks.get("distances", [])[:N_RESULTS]
        context = rag.build_context(documents, metadatas)

        sources = [{
            "n": i + 1,
            "filename": (m or {}).get("filename", "unknown"),
            "text": d,
            "distance": distances[i] if i < len(distances) else None,
        } for i, (d, m) in enumerate(zip(documents, metadatas))]
        yield json.dumps({"type": "sources", "sources": sources, "backend": backend}) + "\n"

        try:
            if backend == "local":
                for delta in rag.stream_local(context, request.query):
                    yield json.dumps({"type": "delta", "text": delta}) + "\n"
            elif backend == "gemini":
                yield json.dumps({"type": "delta",
                                  "text": rag.answer_gemini(context, request.query)}) + "\n"
            else:
                yield json.dumps({"type": "error",
                                  "message": f"Unknown backend: {backend}"}) + "\n"
                return
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            return

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4002, reload=True)
