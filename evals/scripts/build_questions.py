"""Build the question/answer dataset for the retrieval evaluation.

    python evals/scripts/build_questions.py --papers 170 --shortlist 100

Generates 1-2 complex question/answer pairs per paper, verifies them, scores
them, and shortlists the best. Writes:

    evals/questions/candidates.json   everything generated, with reject reasons
    evals/questions/dataset.json      the shortlist actually used

Which parser's text does the question writer see?
------------------------------------------------
Papers are split roughly 33/34/33 across the three parsers under test, and the
source is recorded per question. Generating from a single arm would let that
arm define what is answerable; splitting evenly makes the advantage symmetric,
so it cancels in the comparison between arms rather than favouring one. It also
turns the bias into something measurable: the report can show how each arm
scores on questions written from its own output versus the others'.

Verification is done against raw PyMuPDF text, which is none of the three arms,
so a question survives only if its answer really is in the document.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

CORPUS = REPO / "evals" / "corpus"
PARSED = REPO / "evals" / "parsed"
QDIR = REPO / "evals" / "questions"
PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
SEED = 20260912

PROMPT = """You are building an evaluation set for a retrieval system over research papers.

Below is text from one paper. Write {n} questions that this text answers.

Requirements for each question:
- COMPLEX: it must require a specific fact, number, method name, or comparison —
  not a definition anyone could answer without the paper.
- SELF-CONTAINED: understandable without seeing this text. Name the method,
  dataset, metric or system involved so the question identifies THIS paper.
- ANSWERABLE from this text alone.
- Do not ask about figures, page layout, or the reference list.

For each question also give:
- "ideal_answer": a complete answer in one or two sentences, in your own words.
- "answer_span": a span of 8 to 30 words copied VERBATIM from the text above,
  character for character, that contains the evidence. Do not paraphrase it.
- "keywords": 3 to 6 distinctive terms that must appear in any correct answer.

Reply with JSON only:
{{"questions": [{{"question": "...", "ideal_answer": "...", "answer_span": "...",
                  "keywords": ["...", "..."]}}]}}

Text:
{text}
"""


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def text_key(s):
    """Comparison form for matching a span against a document.

    Questions are written from a parser's output, which has already resolved
    ligatures and joined hyphenated line wraps, while verification reads the
    raw PDF where those artifacts remain. A plain string match therefore
    rejects spans that are genuinely present — 127 of 200 on the first run.
    This folds the difference away: NFKD so "ﬁ" becomes "fi", wrap hyphens
    removed, punctuation dropped, whitespace collapsed, lower-cased.

    Imported by run_rag_eval so scoring uses exactly the same comparison.
    """
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = s.replace("\u00ad", "")                  # soft hyphen
    s = re.sub(r"-\s+", "", s)                   # line-wrap hyphen
    s = re.sub(r"[^0-9A-Za-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def raw_text(stem):
    """Neutral verification text: a plain page dump, not any arm's output."""
    import pymupdf
    pdf = CORPUS / f"{stem}.pdf"
    if not pdf.exists():
        return ""
    with pymupdf.open(pdf) as doc:
        return norm(" ".join(doc[i].get_text() for i in range(len(doc))))


def parser_text(parser, stem, budget):
    path = PARSED / parser / f"{stem}.json"
    if not path.exists():
        return None
    blocks = json.loads(path.read_text(encoding="utf-8"))["blocks"]
    # skip the front matter: title/abstract questions are too easy and every
    # arm answers them, which compresses the differences being measured
    text = "\n\n".join(b["text"] for b in blocks if b.get("text"))
    return text[1500:1500 + budget]


def ask(prompt, cfg, timeout=300):
    import requests
    local = cfg["llm"]["local"]
    r = requests.post(f"{local['url'].rstrip('/')}/api/chat", json={
        "model": local["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, "think": False,
        "options": {"num_ctx": int(local.get("num_ctx", 6144)), "temperature": 0.4},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


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


def complexity(q):
    """Rank candidates so the shortlist is the most demanding, not the first N.

    Rewards specificity — numbers, named entities, comparisons — and penalises
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=170)
    ap.add_argument("--per-paper", type=int, default=2)
    ap.add_argument("--shortlist", type=int, default=100)
    ap.add_argument("--chars", type=int, default=7000)
    args = ap.parse_args()

    from common.settings import load
    cfg = load()

    QDIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    by_id = {p["arxiv_id"]: p for p in manifest["papers"]}

    stems = sorted(by_id)
    rng = random.Random(SEED)
    rng.shuffle(stems)
    stems = stems[:args.papers]

    # 33 / 34 / 33 across the three parsers, deterministic
    assignment = {}
    for i, stem in enumerate(stems):
        assignment[stem] = PARSERS[i % len(PARSERS)]

    cand_path = QDIR / "candidates.json"
    candidates = json.loads(cand_path.read_text())["questions"] if cand_path.exists() else []
    done = {c["paper"] for c in candidates}

    print(f"generating from {len(stems)} papers "
          f"({Counter(assignment.values())}), {len(done)} already done")

    for i, stem in enumerate(stems, 1):
        if stem in done:
            continue
        src = assignment[stem]
        text = parser_text(src, stem, args.chars)
        if not text or len(text) < 800:
            print(f"  [{i}/{len(stems)}] {stem}: no usable {src} text", flush=True)
            continue
        try:
            reply = ask(PROMPT.format(n=args.per_paper, text=text), cfg)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [{i}/{len(stems)}] {stem}: model error {str(e)[:70]}", flush=True)
            continue
        got = parse_reply(reply)
        for q in got:
            if not (q.get("question") and q.get("ideal_answer") and q.get("answer_span")):
                continue
            candidates.append({
                "question": q["question"].strip(),
                "ideal_answer": q["ideal_answer"].strip(),
                "answer_span": q["answer_span"].strip(),
                "keywords": [k.strip() for k in q.get("keywords", []) if k.strip()][:6],
                "paper": stem,
                "arxiv_id": by_id[stem]["arxiv_id"],
                "filename": f"{stem}.pdf",
                "generated_from": src,
            })
        cand_path.write_text(json.dumps({"questions": candidates}, indent=1))
        if i % 10 == 0:
            print(f"  [{i}/{len(stems)}] {len(candidates)} candidates", flush=True)

    # ---- verify against neutral raw text ----
    print(f"\nverifying {len(candidates)} candidates against raw PDF text")
    cache, kept, rejected = {}, [], []
    for c in candidates:
        if c["paper"] not in cache:
            cache[c["paper"]] = text_key(raw_text(c["paper"]))
        raw = cache[c["paper"]]
        span = text_key(c["answer_span"])
        if len(span.split()) < 6:
            rejected.append({**c, "why": "span too short to be evidence"})
        elif span not in raw:
            rejected.append({**c, "why": "span not found verbatim in the PDF"})
        elif len(c["question"].split()) < 8:
            rejected.append({**c, "why": "question too short"})
        else:
            kept.append({**c, "complexity": complexity(c)})

    kept.sort(key=lambda q: q["complexity"], reverse=True)

    # keep the shortlist spread across papers rather than clustered on a few
    picked, per_paper = [], Counter()
    for q in kept:
        if per_paper[q["paper"]] >= 1 and len(picked) < args.shortlist:
            continue
        picked.append(q)
        per_paper[q["paper"]] += 1
        if len(picked) >= args.shortlist:
            break
    if len(picked) < args.shortlist:                 # top up with seconds
        for q in kept:
            if q in picked:
                continue
            picked.append(q)
            if len(picked) >= args.shortlist:
                break

    for n, q in enumerate(picked):
        q["qid"] = f"q{n:03d}"

    (QDIR / "dataset.json").write_text(json.dumps({
        "seed": SEED,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "papers_sampled": len(stems),
        "candidates": len(candidates),
        "verified": len(kept),
        "rejected": len(rejected),
        "reject_reasons": dict(Counter(r["why"] for r in rejected)),
        "source_split": dict(Counter(q["generated_from"] for q in picked)),
        "questions": picked,
    }, indent=1), encoding="utf-8")
    cand_path.write_text(json.dumps({"questions": candidates, "rejected": rejected}, indent=1))

    print(f"\ncandidates {len(candidates)} -> verified {len(kept)} -> shortlist {len(picked)}")
    print("rejects:", dict(Counter(r["why"] for r in rejected)))
    print("source split:", dict(Counter(q["generated_from"] for q in picked)))
    print(f"wrote {QDIR / 'dataset.json'}")


if __name__ == "__main__":
    main()
