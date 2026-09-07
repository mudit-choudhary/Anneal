"""RAG query CLI.

Retrieves chunks from the embedding service, then answers with either a
locally hosted SLM via Ollama (default: Qwen3-4B) or Gemini. Select with
LLM_BACKEND=local|gemini (see config.py).
"""

import json
from typing import Iterator, List, Optional

import requests

from config import (
    EMBEDDING_URL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_BACKEND,
    N_RESULTS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_URL,
)

SYSTEM_PROMPT = (
    "You are a research assistant answering questions strictly from excerpts "
    "of research papers provided as context. Rules:\n"
    "- Use ONLY the provided excerpts; do not add outside knowledge.\n"
    "- Cite the source after each claim using its bracketed number, e.g. [2].\n"
    "- If the excerpts do not contain the answer, say so explicitly.\n"
    "- Be concise and technical."
)


def fetch_chunks(query: str, filenames: Optional[List[str]] = None,
                 n_results: int = N_RESULTS):
    response = requests.get(
        f"{EMBEDDING_URL}/get_chunks",
        json={"query": query, "filenames": filenames, "n_results": n_results},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def source_label(meta: Optional[dict]) -> str:
    """Human-readable provenance from chunk metadata: paper title (or
    prettified filename), plus 1-indexed page(s) when known."""
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


def build_context(documents: List[str], metadatas: List[dict]) -> str:
    """Number each chunk and label it with its source for citation. (Chunk
    text already carries its heading path — title › section — as its first
    line, so the label only adds the page.)"""
    parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        parts.append(f"[{i}] (from: {source_label(meta)})\n{doc}")
    return "\n\n".join(parts)


def build_user_prompt(context: str, query: str) -> str:
    return f"Excerpts:\n\n{context}\n\nQuestion: {query}"


def _local_payload(context: str, query: str, stream: bool) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context, query)},
        ],
        "stream": stream,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.2},
    }


def strip_thinking(text: str) -> str:
    """Remove a leaked <think>...</think> block (belt-and-braces: some model
    templates emit reasoning despite think=False)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def answer_local(context: str, query: str) -> str:
    """Query the Ollama daemon (Ollama's native chat API)."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=_local_payload(context, query, stream=False),
        timeout=300,
    )
    response.raise_for_status()
    return strip_thinking(response.json()["message"]["content"])


def stream_local(context: str, query: str) -> Iterator[str]:
    """Yield answer text incrementally from the Ollama daemon (used by the UI).

    Buffers the start of the stream just long enough to detect and swallow a
    leaked <think>...</think> block before yielding real answer text.
    """
    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=_local_payload(context, query, stream=True),
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        buffer, checking = "", True
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content")
            if content:
                if not checking:
                    yield content
                else:
                    buffer += content
                    head = buffer.lstrip()
                    if head.startswith("<think>"):
                        if "</think>" in head:
                            checking = False
                            after = strip_thinking(head)
                            if after:
                                yield after
                    elif not "<think>".startswith(head[:7]):
                        # Definitely not a thinking block; flush and pass through.
                        checking = False
                        yield buffer
            if data.get("done"):
                if checking and buffer:
                    yield strip_thinking(buffer.lstrip())
                break


def answer_gemini(context: str, query: str) -> str:
    from google import genai

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[SYSTEM_PROMPT, build_user_prompt(context, query)],
    )
    return response.text


def main(query: str, filenames: Optional[List[str]] = None, backend: str = LLM_BACKEND):
    chunks = fetch_chunks(query, filenames)
    documents = chunks["documents"][:N_RESULTS]
    metadatas = chunks["metadatas"][:N_RESULTS]
    context = build_context(documents, metadatas)

    print(f"Retrieved {len(documents)} chunks; answering with backend: {backend}\n" + "=" * 50)

    if backend == "local":
        answer = answer_local(context, query)
    elif backend == "gemini":
        answer = answer_gemini(context, query)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")

    print(answer)
    return answer


if __name__ == "__main__":
    query = input("Please enter your query: ")
    filenames = input("Filenames filter, comma-separated (empty for all): ").strip()
    file_list = [f.strip() for f in filenames.split(",")] if filenames else None
    main(query, file_list)
