# Anneal — local-first RAG over research papers.
# Copyright (C) 2026 Mudit Choudhary
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you may redistribute and/or modify it under
# the terms of the GNU Affero General Public License, version 3 or later. It
# is distributed WITHOUT ANY WARRANTY. See the LICENSE file, or
# <https://www.gnu.org/licenses/>.
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
import re
import sys
from datetime import datetime, timedelta
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
    "- Be concise and technical.\n"
    "- In Mermaid diagrams, put every node label in double quotes, e.g. A[\"Query rewriting (optional)\"]."
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
            arrived = meta.get("arrived")
            sources.append({"kind": "paper",
                            "label": f"from: {source_label(meta)}"
                                     + (f", added {arrived}" if arrived else ""),
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


def search_text(query: str, history: Optional[List[dict]] = None) -> str:
    """The text actually embedded for retrieval.

    A follow-up like "give me the gist of this paper" carries no searchable
    content on its own — embedding it alone retrieves something unrelated.
    Prepending the previous user turn restores the subject.
    """
    if not history:
        return query
    previous = [m["content"] for m in history if m.get("role") == "user"]
    return f"{previous[-1]} {query}" if previous else query


# --------------------------------------------------------------- what's new
# "what came in overnight?" is a question about arrival dates, not about the
# text of any paper — a meaning-based search for these words finds nothing.
RECENCY = re.compile(r"\b(latest|newest|recent(?:ly)?|new(?:est)? (?:papers?|research|work)|"
                     r"what'?s new|anything new|today|yesterday|this week|past week|last week|"
                     r"last (\d+) days?|since yesterday|overnight)\b", re.I)
WINDOW_DAYS = {"today": 1, "yesterday": 2, "overnight": 1, "since yesterday": 2,
               "this week": 7, "past week": 7, "last week": 7}
RECENCY_FALLBACK = 5           # papers to show when the window itself is empty
DEFAULT_ARRIVALS = 5           # papers summarised when the question names no number
MAX_ARRIVALS = 12              # a 6144-token window will not hold more
# "the 10 new updates", "top 3 papers" — but never the 3 in "last 3 days".
HOW_MANY = re.compile(r"\b(?:top\s+)?(\d{1,2})\s+(?:new\s+|latest\s+|recent\s+)*"
                      r"(?:papers?|updates?|articles?|results?|additions?|arrivals?)\b", re.I)


def how_many(query: str) -> Optional[int]:
    """The count the question asks for, if any: "the 10 new updates" -> 10."""
    m = HOW_MANY.search(query)
    return min(MAX_ARRIVALS, max(1, int(m.group(1)))) if m else None


def day_label(stamp: str, today: Optional[datetime] = None) -> str:
    """'today' / 'yesterday' / a date. Models are unreliable at working out
    which dates fall inside "since yesterday"; hand them the answer."""
    today = (today or datetime.now()).date()
    try:
        day = datetime.strptime(stamp[:10], "%Y-%m-%d").date()
    except ValueError:
        return "an unknown date"
    delta = (today - day).days
    return {0: "today", 1: "yesterday"}.get(delta, f"{day:%Y-%m-%d}")


def recency_window(query: str) -> Optional[int]:
    """Days of history the question asks for, or None if it is not one.

    The narrowest phrase wins: "the latest updates today" means today, not the
    week that a bare "latest" would imply."""
    windows = []
    for m in RECENCY.finditer(query):
        if m.group(2):                               # "last 3 days"
            windows.append(max(1, int(m.group(2))))
        else:
            windows.append(WINDOW_DAYS.get(m.group(0).lower(), 7))
    return min(windows) if windows else None


def recent_papers(days: int, limit: int = DEFAULT_ARRIVALS) -> Tuple[List[dict], bool]:
    """Embedded papers that arrived within `days`, newest first.

    Returns (papers, within_window). When nothing arrived in the window the
    newest few are returned anyway with within_window False, so the answer can
    say "nothing new today, here is the most recent work" instead of nothing."""
    from common.registry_client import RegistryClient

    papers = [p for p in RegistryClient().list_papers() if p.get("status") == "embedded"]
    stamp = lambda p: (p.get("embedded_at") or p.get("downloaded_at") or "")   # noqa: E731
    papers.sort(key=stamp, reverse=True)
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    fresh = [p for p in papers if stamp(p) >= cutoff]
    return (fresh[:limit], True) if fresh else (papers[:RECENCY_FALLBACK], False)


# What a "what is this paper about" excerpt should be: the abstract, or failing
# that the introduction — never the reference list or a page of index terms.
OPENING_GOOD = re.compile(r"abstract", re.I)
OPENING_OK = re.compile(r"introduction|overview|background", re.I)
OPENING_BAD = re.compile(r"reference|bibliograph|acknowledg|appendix|index terms|"
                         r"table of contents|continued on next page", re.I)


def opening_rank(hit: dict):
    """Sort key picking the most descriptive chunk of a paper (lowest wins)."""
    meta = hit.get("metadata") or {}
    section = str(meta.get("section") or "")
    text = hit.get("text", "")
    if OPENING_GOOD.search(section):
        kind = 0
    elif OPENING_OK.search(section):
        kind = 1
    elif OPENING_BAD.search(section) or OPENING_BAD.search(text[:200]):
        kind = 9
    else:
        kind = 3
    return (kind, 0 if len(text) >= 400 else 1, meta.get("page_start") or 0, -len(text))


def arrivals_context(query: str, days: int, n_per_paper: int = 1) -> Tuple[List[dict], List[str]]:
    """Excerpts from each recently arrived paper, newest first."""
    wanted = how_many(query) or DEFAULT_ARRIVALS
    papers, within = recent_papers(days, wanted)
    warnings: List[str] = []
    if not papers:
        return [], ["no embedded papers yet, so there is nothing recent to report"]
    if not within:
        warnings.append(f"nothing new in the last {days} day(s); showing the {len(papers)} most recent papers")
    elif len(papers) < wanted and how_many(query):
        warnings.append(f"only {len(papers)} paper(s) arrived in that period, not {wanted}")
    results: List[dict] = []
    for p in papers:
        stem = p["filename"]
        # Probe the paper with its own title, then keep its most descriptive
        # chunk: the raw nearest hit is often index terms or the references.
        hits = sorted(fetch_chunks(stem.replace("_", " "), [stem], 6, ("papers",), 0),
                      key=opening_rank)[:n_per_paper]
        for h in hits:
            meta = dict(h.get("metadata") or {})
            meta["arrived"] = day_label(p.get("embedded_at") or p.get("downloaded_at") or "")
            h["metadata"] = meta
        results += hits
    return results, warnings


def question_for(query: str) -> str:
    """What the model is actually asked. A recency question is answered from
    the arrival list itself: left with the user's wording a small model argues
    about which dates count as "since yesterday" and replies "nothing is new"."""
    if recency_window(query) is None:
        return query
    return (f"{query}\n\n(The excerpts above are already the papers that arrived in that period."
            f" List and summarise every one of them in the order given, [1] first (that is"
            f" newest first), with citations. Do not check"
            f" or discuss their dates, and do not reply that nothing is new.)")


def retrieve(query: str, filenames: Optional[List[str]] = None, web: bool = False,
             cfg: Optional[dict] = None,
             history: Optional[List[dict]] = None) -> Tuple[str, List[dict], List[str]]:
    """Everything before generation. Returns (context, sources, warnings).

    `history` is the earlier turns of this conversation, oldest first, as
    {"role": "user"|"assistant", "content": str}.
    """
    cfg = cfg or settings_store.load()
    rt = cfg["retrieval"]
    kinds = ["papers"] + (["chats"] if rt.get("use_chats", True) else [])
    n = int(rt.get("n_results_with_web", 4) if web else rt.get("n_results", 6))
    warnings: List[str] = []
    days = None if filenames else recency_window(query)
    if days:
        results, warnings = arrivals_context(query, days)
    else:
        results = fetch_chunks(search_text(query, history), filenames, n, kinds,
                               int(rt.get("n_chat_results", 2)))
    pages: List[dict] = []
    if web:
        try:
            pages = search_web(query, int(rt.get("web_results", 3)), int(rt.get("web_chars_per_page", 2000)))
            if not pages:
                warnings.append("web search returned no usable pages")
        except Exception as e:
            warnings.append(f"web search unavailable: {e}")
    context, sources = build_context(results, pages)
    if days and results:
        # Without this the model reads "latest updates" as news and refuses.
        # The dates are already decided here: left to itself the model
        # miscounts which ones fall inside "since yesterday" and answers "none".
        since = "today" if days <= 1 else f"in the last {days} days"
        papers_n = len({s.get("filename") for s in sources if s.get("kind") == "paper"})
        context = (f"The excerpts below ARE the {papers_n} paper(s) added to the library"
                   f" {since} — they have already been filtered by arrival date, so treat every"
                   f" one of them as new and do not re-judge the dates. Summarise each in one or"
                   f" two sentences (what it is about, why it matters), keeping the order below"
                   f" ([1] is the newest), citing each."
                   f" Do not say there is nothing new.\n\n{context}")
    return context, sources, warnings


# --------------------------------------------------------------- generation
MAX_HISTORY_TURNS = 4          # messages, not exchanges — keeps num_ctx headroom
MAX_HISTORY_CHARS = 1200       # per remembered assistant answer


def _messages(context: str, query: str, history: Optional[List[dict]] = None) -> List[dict]:
    """System prompt, then the recent conversation, then this turn's excerpts.

    Earlier turns let the model resolve "this paper" / "and its limitations?".
    Old assistant answers are truncated: they are there for reference, and a
    6144-token window is not big enough to carry them whole.
    """
    messages = [{"role": "system",
                 "content": f"{SYSTEM_PROMPT}\nToday is {datetime.now():%A, %-d %B %Y}."}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        content = turn.get("content", "")
        if turn.get("role") == "assistant" and len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS].rsplit(" ", 1)[0] + " …"
        if content:
            messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"})
    return messages


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


def _ollama_stream(context: str, query: str, local: dict, history=None) -> Iterator[str]:
    payload = {
        "model": local.get("model", "qwen3:4b-instruct"),
        "messages": _messages(context, query, history),
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


def _openai_stream(context: str, query: str, oa: dict, history=None) -> Iterator[str]:
    from openai import OpenAI

    if not oa.get("base_url") or not oa.get("model"):
        raise RuntimeError("OpenAI-compatible backend is not configured: set base_url and model in Settings")
    client = OpenAI(base_url=oa["base_url"], api_key=oa.get("api_key") or "none")
    kwargs = dict(model=oa["model"], messages=_messages(context, query, history),
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


def answer_stream(context: str, query: str, cfg: Optional[dict] = None,
                  history: Optional[List[dict]] = None) -> Iterator[str]:
    """Stream the answer for an already-built context using the configured backend."""
    llm = (cfg or settings_store.load())["llm"]
    if llm.get("backend") == "openai":
        yield from _openai_stream(context, query, llm.get("openai", {}), history)
    else:
        yield from _ollama_stream(context, query, llm.get("local", {}), history)


def answer(context: str, query: str, cfg: Optional[dict] = None,
           history: Optional[List[dict]] = None) -> str:
    return "".join(answer_stream(context, query, cfg, history)).strip()


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
