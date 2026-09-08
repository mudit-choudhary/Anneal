"""Retrieval + answering, shared by the UI, the CLI and rag_inspect.py.

    context, sources, warnings = retrieve(question, filenames, web=False)
    for token in answer_stream(context, question): ...

Backends (settings `llm.backend`, editable in the UI):
    local   — Ollama's native /api/chat (default: qwen3:4b-instruct)
    openai  — any OpenAI-compatible chat API (base_url + api_key + model),
              e.g. a self-hosted gateway; streams if the API allows, else
              returns the whole answer at once.
Retrieval sources: paper chunks (always), saved conversations (optional),
web pages fetched live (optional; given to the model as-is, never embedded).
"""

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from common import settings as settings_store
from common.paths import EMBEDDING_URL
from websearch import search_web

SYSTEM_PROMPT = (
    "You are a research assistant answering questions strictly from the numbered excerpts "
    "provided as context. Excerpts come from research papers; some may be web pages (marked "
    "'web') or the user's previous conversations (marked 'previous conversation'). Rules:\n"
    "- Use ONLY the excerpts; do not add outside knowledge.\n"
    "- Cite the source after each claim using its bracketed number, e.g. [2].\n"
    "- Prefer paper excerpts over web pages and previous conversations when they disagree.\n"
    "- If the excerpts do not contain the answer, say so explicitly.\n"
    "- Be concise and technical."
)


# --------------------------------------------------------------- retrieval
def fetch_chunks(query: str, filenames: Optional[List[str]] = None, n_results: int = 6,
                 sources: Sequence[str] = ("papers",), n_chat_results: int = 2) -> List[dict]:
    r = requests.post(f"{EMBEDDING_URL}/v1/search",
                      json={"query": query, "n_results": n_results, "filenames": filenames,
                            "sources": list(sources), "n_chat_results": n_chat_results},
                      timeout=90)
    r.raise_for_status()
    return r.json()["results"]


def source_label(meta: Optional[dict]) -> str:
    """Paper provenance: title (or prettified filename) + 1-indexed page(s)."""
    meta = meta or {}
    name = meta.get("title")
    if not name:
        name = meta.get("filename", "unknown")
        if name.endswith(".txt"):
            name = name[:-4]
        name = name.replace("_", " ")
    start, end = meta.get("page_start"), meta.get("page_end")
    if start is not None:
        pages = f"p.{start + 1}" if end in (None, start) else f"pp.{start + 1}-{end + 1}"
        name = f"{name}, {pages}"
    return name


def build_context(results: List[dict], web_pages: Sequence[dict] = ()) -> Tuple[str, List[dict]]:
    """Number every excerpt and label it for citation. Returns (context text,
    sources) where each source is {n, kind, label, text, ...fields}."""
    sources: List[dict] = []
    for r in results:
        meta = r.get("metadata") or {}
        if r.get("source") == "chats":
            sources.append({"kind": "chat", "label": f"previous conversation: {meta.get('title', '')}, "
                                                     f"{str(meta.get('created_at', ''))[:10]}",
                            "chat_id": meta.get("chat_id"), "title": meta.get("title")})
        else:
            sources.append({"kind": "paper", "label": f"from: {source_label(meta)}",
                            "filename": meta.get("filename"), "title": meta.get("title"),
                            "section": meta.get("section"), "page_start": meta.get("page_start"),
                            "page_end": meta.get("page_end")})
        sources[-1].update({"text": r.get("text", ""), "distance": r.get("distance")})
    for p in web_pages:
        sources.append({"kind": "web", "label": f"web: {p.get('title', '')} — {p.get('url', '')}",
                        "title": p.get("title"), "url": p.get("url"), "text": p.get("text", ""),
                        "distance": None})
    for i, s in enumerate(sources, start=1):
        s["n"] = i
    context = "\n\n".join(f"[{s['n']}] ({s['label']})\n{s['text']}" for s in sources)
    return context, sources


def retrieve(query: str, filenames: Optional[List[str]] = None, web: bool = False,
             cfg: Optional[dict] = None) -> Tuple[str, List[dict], List[str]]:
    """Everything before generation. Returns (context, sources, warnings)."""
    cfg = cfg or settings_store.load()
    rt = cfg["retrieval"]
    kinds = ["papers"] + (["chats"] if rt.get("use_chats", True) else [])
    n = int(rt.get("n_results_with_web", 4) if web else rt.get("n_results", 6))
    results = fetch_chunks(query, filenames, n, kinds, int(rt.get("n_chat_results", 2)))
    warnings: List[str] = []
    pages: List[dict] = []
    if web:
        try:
            pages = search_web(query, int(rt.get("web_results", 3)), int(rt.get("web_chars_per_page", 2000)))
            if not pages:
                warnings.append("web search returned no usable pages")
        except Exception as e:
            warnings.append(f"web search unavailable: {e}")
    context, sources = build_context(results, pages)
    return context, sources, warnings


# --------------------------------------------------------------- generation
def _messages(context: str, query: str) -> List[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"}]


def strip_thinking(text: str) -> str:
    """Remove a leaked <think>...</think> block."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def _filter_thinking(deltas: Iterator[str]) -> Iterator[str]:
    """Buffer the head of a stream just long enough to swallow a leaked
    <think>...</think> block, then pass everything through."""
    buffer, checking = "", True
    for content in deltas:
        if not checking:
            yield content
            continue
        buffer += content
        head = buffer.lstrip()
        if head.startswith("<think>"):
            if "</think>" in head:
                checking = False
                after = strip_thinking(head)
                if after:
                    yield after
        elif not "<think>".startswith(head[:7]):
            checking = False
            yield buffer
    if checking and buffer:
        yield strip_thinking(buffer.lstrip())


def _ollama_stream(context: str, query: str, local: dict) -> Iterator[str]:
    payload = {
        "model": local.get("model", "qwen3:4b-instruct"),
        "messages": _messages(context, query),
        "stream": True,
        "think": False,
        "keep_alive": local.get("keep_alive", "30m"),
        "options": {"num_ctx": int(local.get("num_ctx", 6144)),
                    "temperature": float(local.get("temperature", 0.2))},
    }
    url = local.get("url", "http://127.0.0.1:11434").rstrip("/")

    def deltas():
        with requests.post(f"{url}/api/chat", json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    break
    yield from _filter_thinking(deltas())


def _openai_stream(context: str, query: str, oa: dict) -> Iterator[str]:
    from openai import OpenAI

    if not oa.get("base_url") or not oa.get("model"):
        raise RuntimeError("OpenAI-compatible backend is not configured: set base_url and model in Settings")
    client = OpenAI(base_url=oa["base_url"], api_key=oa.get("api_key") or "none")
    kwargs = dict(model=oa["model"], messages=_messages(context, query),
                  temperature=float(oa.get("temperature", 0.2)),
                  max_tokens=int(oa.get("max_tokens", 1024)))
    if oa.get("stream", True):
        yielded = False
        try:
            for chunk in client.chat.completions.create(stream=True, **kwargs):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yielded = True
                    yield delta
            return
        except Exception as e:
            if yielded:
                raise
            # The API rejected streaming (or the request failed before any
            # token): fall back to a plain completion and send it whole.
            if "stream" not in str(e).lower() and not _is_client_error(e):
                raise
    resp = client.chat.completions.create(**kwargs)
    yield strip_thinking(resp.choices[0].message.content or "")


def _is_client_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500


def answer_stream(context: str, query: str, cfg: Optional[dict] = None) -> Iterator[str]:
    """Stream the answer for an already-built context using the configured backend."""
    llm = (cfg or settings_store.load())["llm"]
    if llm.get("backend") == "openai":
        yield from _openai_stream(context, query, llm.get("openai", {}))
    else:
        yield from _ollama_stream(context, query, llm.get("local", {}))


def answer(context: str, query: str, cfg: Optional[dict] = None) -> str:
    return "".join(answer_stream(context, query, cfg)).strip()


# --------------------------------------------------------------- CLI
def main(query: str, filenames: Optional[List[str]] = None, web: bool = False):
    cfg = settings_store.load()
    context, sources, warnings = retrieve(query, filenames, web, cfg)
    for w in warnings:
        print(f"! {w}")
    print(f"Retrieved {len(sources)} excerpts; backend: {cfg['llm']['backend']}\n" + "=" * 50)
    for s in sources:
        print(f"  [{s['n']}] {s['label']}")
    print("=" * 50)
    for token in answer_stream(context, query, cfg):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    q = input("Please enter your query: ")
    f = input("Filenames filter, comma-separated (empty for all): ").strip()
    w = input("Include web search? [y/N]: ").strip().lower() == "y"
    main(q, [x.strip() for x in f.split(",")] if f else None, w)
