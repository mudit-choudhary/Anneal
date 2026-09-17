"""Optional web search for a question: DuckDuckGo results, each page's main
text extracted with trafilatura, trimmed to a budget, and handed to the
answering model alongside the paper chunks. Nothing is chunked or embedded.
"""

from typing import Dict, List

import requests

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) RAGSetup/1.0"


def search_web(query: str, k: int = 3, chars_per_page: int = 2000, timeout: int = 10) -> List[Dict[str, str]]:
    """Returns [{title, url, text}] — at most k pages, each text <= chars_per_page.
    Pages that can't be fetched fall back to the search snippet."""
    from ddgs import DDGS
    import trafilatura

    try:
        hits = list(DDGS().text(query, max_results=k))
    except Exception as e:  # network / rate limit
        raise RuntimeError(f"web search failed: {e}") from e

    pages = []
    for hit in hits[:k]:
        url = hit.get("href") or hit.get("url") or ""
        title = hit.get("title") or url
        text = ""
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.ok and "html" in r.headers.get("content-type", ""):
                text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        except requests.RequestException:
            pass
        if not text:
            text = hit.get("body") or ""
        text = " ".join(text.split())
        if len(text) > chars_per_page:
            text = text[:chars_per_page].rsplit(" ", 1)[0] + " …"
        if text:
            pages.append({"title": title, "url": url, "text": text})
    return pages
