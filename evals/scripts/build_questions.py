"""Build the question/answer dataset for the retrieval evaluation.

    source evals/keys_export.sh
    python evals/scripts/build_questions.py --shortlist 400

Generates 1-2 question/answer pairs per paper through an OpenAI-compatible
endpoint, verifies them against two text engines, scores them, and shortlists
the best. Writes:

    evals/questions/candidates.json   everything generated, with reject reasons
    evals/questions/dataset.json      the shortlist actually used

Which text does the question writer see?
----------------------------------------
Poppler's `pdftotext` output (evals/text/<stem>.txt). None of the parsers under
test uses poppler: `current` and PyMuPDF4LLM read text through PyMuPDF and
Docling has its own engine. Writing questions from any parser's output gave that
parser a home advantage in round 1; neutral text removes it at the source.

Verification needs two engines to agree: the answer span must appear verbatim
in the pdftotext text, and at least NEAR_MATCH of its words must appear in a
span-length window of the PyMuPDF text. One engine's quirks cannot decide on
their own what counts as a valid answer.

Every question also carries a `short_answer` and atomic `facts`, so answers can
later be scored without an LLM judge (exact match, token F1, fact recall, NLI).
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

CORPUS = REPO / "evals" / "corpus"
TEXT = REPO / "evals" / "text"
QDIR = REPO / "evals" / "questions"
SEED = 20260914
# Tried in order. A model that answers 429 is skipped until its cooldown ends,
# so a throttled provider hands over to the next instead of stalling the run.
MODELS = [m.strip() for m in os.environ.get(
    "QGEN_MODELS", "groq/openai/gpt-oss-120b,nvidia/openai/gpt-oss-20b").split(",") if m.strip()]
PRIMARY_SERVED = "openai/gpt-oss-120b"   # how the gateway reports MODELS[0]
COOLDOWN = 90.0                  # seconds, when the provider gives no Retry-After
_cooled = {}                     # model -> time it may be used again
_cool_lock = threading.Lock()

PROMPT = """You are building an evaluation set for a retrieval system over research papers.

Below is an excerpt from one paper. Write {n} questions that this excerpt answers.

Requirements for each question:
- COMPLEX: it must require a specific fact, number, method name, or comparison -
  not a definition anyone could answer without the paper.
- SELF-CONTAINED: understandable without seeing this excerpt, among hundreds of
  other papers. Name the method, dataset, metric or system involved so the
  question identifies THIS paper. Never write "this paper", "the study",
  "the proposed method" or "described".
- ANSWERABLE from this excerpt alone.
- Each question about a different part of the excerpt.
- Do not ask about figures, page layout, or the reference list.

For each question also give:
- "ideal_answer": a complete answer in one or two sentences, in your own words.
- "short_answer": the core of the answer in at most 8 words (a number, name or term).
- "facts": 1 to 3 short atomic statements, each checkable on its own, that any
  correct answer must contain.
- "answer_span": a span of 8 to 30 words copied VERBATIM from the excerpt,
  character for character, that contains the evidence. Do not paraphrase it,
  do not shorten it with "...", and do not join separate sentences.
- "keywords": 3 to 6 distinctive terms that must appear in any correct answer.

Reply with JSON only:
{{"questions": [{{"question": "...", "ideal_answer": "...", "short_answer": "...",
                  "facts": ["..."], "answer_span": "...", "keywords": ["..."]}}]}}

Excerpt:
{text}
"""


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def write_json(path, obj, **kw):
    """Write to a temp file and swap it in, so a crash mid-write never leaves
    a truncated file that would stop the next resume."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, **kw), encoding="utf-8")
    os.replace(tmp, path)


def text_key(s):
    """Comparison form for matching a span against a document.

    Text engines differ in ligatures, hyphenated line wraps and punctuation.
    This folds those differences away: NFKD so "ﬁ" becomes "fi", wrap hyphens
    removed, punctuation dropped, whitespace collapsed, lower-cased.

    Imported by run_rag_eval so scoring uses exactly the same comparison.
    """
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = s.replace("­", "")                  # soft hyphen
    s = re.sub(r"-\s+", "", s)                   # line-wrap hyphen
    s = re.sub(r"[^0-9A-Za-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


_REFS = re.compile(r"\n\s*(references|bibliography|reference list)\s*\n", re.I)


def excerpt(stem, chars):
    """A seeded random window of the paper's body: past the title and abstract,
    before the reference list, so questions spread across papers' contents."""
    path = TEXT / f"{stem}.txt"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cuts = [m.start() for m in _REFS.finditer(raw) if m.start() > len(raw) * 0.4]
    body = norm(raw[:cuts[-1]] if cuts else raw)
    lo, hi = min(1500, len(body) // 5), max(0, len(body) - chars)
    start = random.Random(f"{SEED}:{stem}").randint(lo, hi) if hi > lo else 0
    return body[start:start + chars]


def pymupdf_text(stem):
    import pymupdf
    with pymupdf.open(CORPUS / f"{stem}.pdf") as doc:
        return " ".join(doc[i].get_text() for i in range(len(doc)))


def _retry_after(e):
    try:
        v = e.response.headers.get("retry-after", "")
        return float(v) if v else COOLDOWN
    except Exception:                                            # noqa: BLE001
        return COOLDOWN


def ask(client, prompt, rounds=6):
    """One generation with failover; returns (reply text, model that served it).

    Each round tries the models that are not cooling down, in order. A 429
    cools that model; any other error moves on to the next. If every model is
    cooling, wait for the earliest to come back.
    """
    from openai import RateLimitError
    last = None
    for _ in range(rounds):
        now = time.time()
        with _cool_lock:
            ready = [m for m in MODELS if _cooled.get(m, 0) <= now]
            wake = min(_cooled.get(m, 0) for m in MODELS)
        if not ready:
            time.sleep(max(1.0, wake - now))
            continue
        for m in ready:
            try:
                r = client.chat.completions.create(
                    model=m, temperature=0.4,
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content or "", r.model
            except RateLimitError as e:
                wait = _retry_after(e)
                with _cool_lock:
                    _cooled[m] = time.time() + wait
                print(f"    {m} rate-limited; cooling {wait:.0f}s", flush=True)
                last = e
            except Exception as e:                               # noqa: BLE001
                print(f"    {m} failed: {type(e).__name__}: {str(e)[:70]}", flush=True)
                last = e
        time.sleep(5)
    raise last or RuntimeError("no model answered")


def parse_reply(reply):
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("questions", [])
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------- scoring
_GENERIC = re.compile(r"^(what is|what are|define|describe|explain)\b", re.I)
# leans on the excerpt instead of naming what it asks about; ambiguous among 514 papers
_DEICTIC = re.compile(r"\b(this|the|that) (paper|study|work|article|excerpt|text)\b|"
                      r"\b(described|mentioned|discussed) (above|here|in)\b|\bin the context of the\b", re.I)


# answer is a reference ("Zhu et al., Phys. Rev. B 105 (2022)"): tests the bibliography,
# not the paper's content, and its year would pass for a numeric answer
_CITATION = re.compile(r"\bet\W*al\b|\(\s*(19|20)\d\d[a-z]?\s*\)|\b(Phys|Rev|Proc|Lett|Conf|Trans|J)\.\W|"
                       r"arXiv\s*:?\s*\d{4}\.\d{4,5}|\bdoi\b", re.I)


def complexity(q):
    """Rank candidates so the shortlist is the most demanding, not the first N.

    Rewards specificity - numbers, named entities, comparisons - and penalises
    questions that read like a glossary lookup.
    """
    text = q["question"]
    score = 0.0
    score += min(len(text.split()) / 12.0, 2.0)                  # longer, more specified
    score += 1.5 if re.search(r"\d", text) else 0.0              # asks for a figure
    score += 1.0 if re.search(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]+[A-Z]\w*", text) else 0.0
    score += 1.0 if re.search(r"\b(compare|versus|vs\.?|differ|outperform|"
                              r"improvement|trade-?off|ablation)\b", text, re.I) else 0.0
    score += min(len(q.get("keywords", [])) / 3.0, 1.5)
    score += min(len(q.get("answer_span", "").split()) / 15.0, 1.0)
    score -= 2.0 if _GENERIC.match(text.strip()) else 0.0        # glossary lookup
    score -= 1.5 if len(text.split()) < 8 else 0.0
    return round(score, 3)


def verify(c, pdftotext_key, pymupdf_key):
    """Reason a candidate is rejected, or None if it is kept."""
    sys.path.insert(0, str(REPO / "evals" / "scripts"))
    from make_figures import NEAR_MATCH, span_overlap
    span = text_key(c["answer_span"])
    if len(span.split()) < 6:
        return "span too short to be evidence"
    if span not in pdftotext_key:
        return "span not verbatim in pdftotext text"
    if span_overlap(span, pymupdf_key) < NEAR_MATCH:
        return "span not found in PyMuPDF text (engines disagree)"
    if len(c["question"].split()) < 8:
        return "question too short"
    if _DEICTIC.search(c["question"]):
        return "question not self-contained"
    if _CITATION.search(c.get("short_answer", "")):
        return "answer is a citation"
    if not c.get("short_answer") or not c.get("facts"):
        return "missing short answer or facts"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, help="default: every paper in the manifest")
    ap.add_argument("--per-paper", type=int, default=2)
    ap.add_argument("--shortlist", type=int, default=400)
    ap.add_argument("--chars", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_BASE_URL"):
        sys.exit("OPENAI_BASE_URL is not set: run `source evals/keys_export.sh` first")
    from openai import OpenAI
    client = OpenAI()

    QDIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    by_id = {p["arxiv_id"]: p for p in manifest["papers"]}
    stems = sorted(by_id)
    random.Random(SEED).shuffle(stems)
    stems = stems[:args.papers] if args.papers else stems

    cand_path = QDIR / "candidates.json"
    candidates = json.loads(cand_path.read_text())["questions"] if cand_path.exists() else []
    done = {c["paper"] for c in candidates}
    todo = [s for s in stems if s not in done]
    print(f"generating from {len(stems)} papers with models={MODELS}, "
          f"{len(done)} already done, {len(todo)} to go", flush=True)

    lock = threading.Lock()

    def one(stem):
        text = excerpt(stem, args.chars)
        if not text or len(text) < 1500:
            return stem, [], None, "no usable pdftotext text"
        reply, served = ask(client, PROMPT.format(n=args.per_paper, text=text))
        return stem, parse_reply(reply), served, None

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, s) for s in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                stem, got, served, why = fut.result()
            except Exception as e:                               # noqa: BLE001
                print(f"  model error: {str(e)[:90]}", flush=True)
                continue
            if why:
                print(f"  {stem}: {why}", flush=True)
                continue
            with lock:
                for q in got:
                    if not (q.get("question") and q.get("ideal_answer") and q.get("answer_span")):
                        continue
                    p = by_id[stem]
                    candidates.append({
                        "question": q["question"].strip(),
                        "ideal_answer": q["ideal_answer"].strip(),
                        "short_answer": str(q.get("short_answer") or "").strip(),
                        "facts": [str(f).strip() for f in q.get("facts", []) if str(f).strip()][:3],
                        "answer_span": q["answer_span"].strip(),
                        "keywords": [str(k).strip() for k in q.get("keywords", []) if str(k).strip()][:6],
                        "paper": stem,
                        "arxiv_id": p.get("arxiv_stamp") or (stem if p.get("source") != "unused_training" else None),
                        "filename": f"{stem}.pdf",
                        "source": p.get("source", "arxiv"),
                        "generated_from": "pdftotext",
                        "served_model": served,
                    })
                write_json(cand_path, {"questions": candidates}, indent=1)
            if n % 25 == 0 or n == len(futures):
                print(f"  [{n}/{len(futures)}] {len(candidates)} candidates", flush=True)

    # ---- verify: two text engines must agree the answer is in the paper ----
    print(f"\nverifying {len(candidates)} candidates against pdftotext and PyMuPDF text", flush=True)
    keys, kept, rejected = {}, [], []
    for c in candidates:
        if c["paper"] not in keys:
            keys[c["paper"]] = (
                text_key((TEXT / f"{c['paper']}.txt").read_text(encoding="utf-8", errors="ignore")),
                text_key(pymupdf_text(c["paper"])))
        why = verify(c, *keys[c["paper"]])
        if why:
            rejected.append({**c, "why": why})
        else:
            kept.append({**c, "complexity": complexity(c)})

    # questions from the primary model first; the fallback only fills the gap to --shortlist
    kept.sort(key=lambda q: (q.get("served_model") == PRIMARY_SERVED, q["complexity"]), reverse=True)

    # one question per paper first; top up with second questions only if needed
    picked, per_paper = [], Counter()
    for q in kept:
        if per_paper[q["paper"]] == 0:
            picked.append(q)
            per_paper[q["paper"]] += 1
        if len(picked) >= args.shortlist:
            break
    for q in kept:
        if len(picked) >= args.shortlist:
            break
        if q not in picked:
            picked.append(q)
            per_paper[q["paper"]] += 1

    for n, q in enumerate(picked):
        q["qid"] = f"q{n:03d}"

    write_json(QDIR / "dataset.json", {
        "seed": SEED,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text_source": "pdftotext (poppler)",
        "models_requested": MODELS,
        "models_served": dict(Counter(q["served_model"] for q in candidates)),
        "papers_sampled": len(stems),
        "candidates": len(candidates),
        "verified": len(kept),
        "rejected": len(rejected),
        "reject_reasons": dict(Counter(r["why"] for r in rejected)),
        "papers_in_shortlist": len(per_paper),
        "source_split": dict(Counter(q["source"] for q in picked)),
        "questions": picked,
    }, indent=1)
    write_json(cand_path, {"questions": candidates, "rejected": rejected}, indent=1)

    print(f"\ncandidates {len(candidates)} -> verified {len(kept)} -> shortlist {len(picked)} "
          f"from {len(per_paper)} papers")
    print("rejects:", dict(Counter(r["why"] for r in rejected)))
    print("served by:", dict(Counter(q["served_model"] for q in candidates)))
    print(f"wrote {QDIR / 'dataset.json'}")


if __name__ == "__main__":
    assert text_key("The ﬁne-\n tuned model, 3.5%") == "the finetuned model 3 5"
    assert parse_reply('```json\n{"questions": [{"question": "x"}]}\n```') == [{"question": "x"}]
    assert _DEICTIC.search("In the context of the convolutional neural network described for comparing")
    assert _DEICTIC.search("What accuracy does this paper report on COCO?")
    assert not _DEICTIC.search("What CPU latency reduction does YOLOv26 achieve with one-to-one assignment?")
    main()
