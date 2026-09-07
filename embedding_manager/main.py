"""Embedding manager service (port 4001).

Background loop: every processed paper (data/processed/<name>.json) whose
registry status is "processed" is chunked (structure-aware, see chunking.py),
embedded, and stored in ChromaDB; status becomes "embedded".
API: GET /get_chunks (retrieval), GET /list_files, POST /healthcheck.
"""

import os
import threading
import time
from typing import List, Optional

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from config import PROCESSED_DIR, REGISTRY_URL
from embeddings import chunk_and_embed, list_embedded_files, query_embeddings


def get_status(filename):
    try:
        r = requests.get(f"{REGISTRY_URL}/get_status", json={"filename": filename}, timeout=20)
        return r.json().get("status") if r.ok else None
    except requests.RequestException as e:
        print(f"[registry] unreachable: {e}")
        return None


def chunks_and_embed_loop(poll_interval=60):
    while True:
        try:
            for file in sorted(os.listdir(PROCESSED_DIR)):
                if not file.endswith(".json"):
                    continue
                filename = os.path.splitext(file)[0]
                if get_status(filename) != "processed":
                    continue

                result = chunk_and_embed(os.path.join(PROCESSED_DIR, file))
                if not result["success"]:
                    print(f"[embed] failed {filename}: {result.get('error')}")
                    continue

                r = requests.post(f"{REGISTRY_URL}/update_status",
                                  json={"filename": filename, "status": "embedded"}, timeout=20)
                if not r.json().get("success"):
                    print(f"[embed] registry rejected status update for {filename}")
        except Exception as e:
            print(f"[embed] loop error: {e}")
        time.sleep(poll_interval)


class QueryRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None
    n_results: int = 8

    @field_validator("query")
    @classmethod
    def non_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Missing query.")
        return v


embedding = FastAPI()


@embedding.get("/")
def read_root():
    return {"Welcome": " to embedding manager!"}


@embedding.post("/healthcheck")
def healthcheck():
    return {"Status": "Okay"}


@embedding.get("/get_chunks")
def fetch_relevant_chunks(request: QueryRequest):
    return query_embeddings(query=request.query, n_results=request.n_results,
                            filename_filter=request.filenames)


@embedding.get("/list_files")
def list_files():
    return {"files": list_embedded_files()}


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    threading.Thread(target=chunks_and_embed_loop, daemon=True).start()
    uvicorn.run("main:embedding", host="127.0.0.1", port=4001, reload=False)
