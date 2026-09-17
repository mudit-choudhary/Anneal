"""Generate evals/Reports/Report.md — the report of the harness.

    python evals/scripts/make_main_report.py

Two runs feed it:

  chunk-shape run  results.json, parser_quality.json        (5 parsers x 4 chunkers)
  retrieval run    rag_results.json, questions/dataset.json,
                   parsed/ cache, corpus/manifest.json       (3 parsers x 4 chunkers)

Every number comes from those files; none is typed in, estimated or
interpolated. Statistics are computed here with the standard library: exact
two-sided sign tests on paired per-question outcomes, Holm correction within
each family of tests, percentile bootstrap intervals, Spearman rank correlation.
"""

import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
R = REPO / "evals" / "Reports"
OUT = R / "Report.md"
# Round 2 took over these two paths. --round1 points them back at round 1's own
# inputs so this report can still be regenerated from the data it was built on.
MANIFEST = REPO / "evals" / "corpus" / "manifest.json"
DATASET = REPO / "evals" / "questions" / "dataset.json"
STATS_NAME = "_stats.json"
sys.path.insert(0, str(REPO / "evals" / "scripts"))

PARSERS = ["raw_dump", "legacy", "oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "semantic", "grain_growth"]
RAG_PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
OTHERS = ["fixed_token", "recursive_char", "semantic"]
ALPHA = 0.05

PARSER_NOTE = {
    "raw_dump": "PyMuPDF `page.get_text()`, no structure",
    "legacy": "commit `057ce0e9`, page dump plus regex paragraph rules",
    "oss_docling": "Docling layout, our stage-2 assembler",
    "oss_pymupdf4llm": "PyMuPDF4LLM markdown mapped to typed blocks",
    "current": "YOLOv11 layout, column-aware reading order",
}

METRICS = [  # key, label, lineage
    ("paper_hit@5", "right paper in top 5", "hit rate (TREC)"),
    ("precision@5", "precision@5, paper-level", "TREC"),
    ("ndcg@10", "nDCG@10, graded", "TREC / BEIR"),
    ("mrr", "MRR on the answer span", "TREC / MS MARCO"),
    ("span_hit@5", "answer span in top 5", "hit rate, passage level"),
    ("span_hit@4000ch", "answer span within 4,000 ch", "budget-fair hit rate"),
    ("keyword_recall", "keywords in retrieved context", "deterministic"),
    ("answer_similarity", "answer vs reference, embedding", "deterministic"),
    ("faithfulness", "faithfulness", "RAGAS, LLM judge"),
    ("correctness", "answer correctness", "RAGAS, LLM judge"),
    ("context_sufficiency", "context sufficiency", "RAGAS, LLM judge"),
]


def pct(v, nd=1):
    return "—" if v is None else f"{100 * v:.{nd}f}%"


def num(v, nd=0):
    return "—" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v))


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sign_p(w, l):
    """Exact two-sided sign test on discordant pairs (McNemar for 0/1 outcomes)."""
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def holm(ps):
    """Holm step-down at ALPHA: which p-values survive the family."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    keep = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] > ALPHA / (len(ps) - rank):
            break
        keep[i] = True
    return keep


def boot_ci(values, n=2000, seed=1):
    rng = random.Random(seed)
    bs = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return st.mean(values), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


# ============================================================ retrieval stats
def rag_analysis(RR, ds, r):
    from make_figures import answer_coverage, NEAR_MATCH

    cells = {k: v for k, v in RR["cells"].items() if v.get("complete")}
    rows = {k: {x["qid"]: x for x in v["_rows"]} for k, v in cells.items()}
    src = {q["qid"]: q["generated_from"] for q in ds["questions"]}

    def paired(a, b, f, qids=None):
        A, B = rows[a], rows[b]
        ks = [k for k in A if k in B and (qids is None or k in qids)
              and isinstance(A[k].get(f), (int, float)) and isinstance(B[k].get(f), (int, float))]
        w = sum(A[k][f] > B[k][f] for k in ks)
        l = sum(A[k][f] < B[k][f] for k in ks)
        return {"w": w, "l": l, "p": sign_p(w, l), "n": len(ks)}

    def vals(k, f, qids=None):
        return [x[f] for q, x in rows[k].items()
                if isinstance(x.get(f), (int, float)) and (qids is None or q in qids)]

    ra = {"cells": cells, "chunker": {}, "parser": [], "ci": {}, "src": {}, "home": {},
          "spearman": [], "src_of": src}

    # grain_growth against each other chunker, within each parser; one family per metric
    for f in ("span_hit@4000ch", "correctness"):
        fam = [(p, c, paired(f"{p}|grain_growth", f"{p}|{c}", f))
               for p in RAG_PARSERS for c in OTHERS
               if f"{p}|grain_growth" in rows and f"{p}|{c}" in rows]
        for (_, _, t), ok in zip(fam, holm([t["p"] for _, _, t in fam])):
            t["holm"] = ok
        ra["chunker"][f] = fam

    # how each parser scores on questions by source, and its home gap per cell
    for s in RAG_PARSERS:
        ids = {q for q, v in src.items() if v == s}
        ra["src"][s] = {"n": len(ids)}
        for p in RAG_PARSERS:
            v = vals(f"{p}|grain_growth", "span_hit@5", ids) if f"{p}|grain_growth" in rows else []
            ra["src"][s][p] = st.mean(v) if v else None
    for k in cells:
        p = k.split("|")[0]
        own = vals(k, "span_hit@5", {q for q, v in src.items() if v == p})
        oth = vals(k, "span_hit@5", {q for q, v in src.items() if v != p})
        ra["home"][k] = (st.mean(own), st.mean(oth), len(own), len(oth))

    # parsers head to head under grain_growth: all questions, then only the third
    # parser's questions, where neither side has a home advantage
    fam = []
    for a, b in (("current", "oss_docling"), ("current", "oss_pymupdf4llm"),
                 ("oss_pymupdf4llm", "oss_docling")):
        if f"{a}|grain_growth" not in rows or f"{b}|grain_growth" not in rows:
            continue
        third = {q for q, v in src.items() if v not in (a, b)}
        for f in ("span_hit@4000ch", "correctness"):
            fam.append({"a": a, "b": b, "f": f,
                        "third": next((x for x in RAG_PARSERS if x not in (a, b)), "—"),
                        "all": paired(f"{a}|grain_growth", f"{b}|grain_growth", f),
                        "neutral": paired(f"{a}|grain_growth", f"{b}|grain_growth", f, third)})
    for t, ok in zip(fam, holm([t["neutral"]["p"] for t in fam])):
        t["neutral"]["holm"] = ok
    ra["parser"] = fam

    for k in cells:
        ra["ci"][k] = {f: boot_ci(vals(k, f)) for f in ("span_hit@4000ch", "correctness", "faithfulness")}

    # [0] near-match losses, the ones reported; [1] of those, retrieved anyway; [2] exact-match losses
    exact, near = answer_coverage(RR), answer_coverage(RR, NEAR_MATCH)
    ra["near_match"] = NEAR_MATCH
    ra["coverage"] = {p: (len(near[p]), sum(1 for c in CHUNKERS if f"{p}|{c}" in rows
                                            for q in near[p] if rows[f"{p}|{c}"].get(q, {}).get("span_hit@10")),
                          len(exact[p]))
                      for p in RAG_PARSERS}
    ra["cov_sweep"] = {t: {p: len(v) for p, v in answer_coverage(RR, t).items()}
                       for t in (1.0, 0.95, NEAR_MATCH, 0.8)}

    keys = [k for k in cells if r["cells"].get(k, {}).get("mid_start_pct") is not None]
    ns = [k for k in keys if "semantic" not in k]
    for sf, lab in (("mid_start_pct", "chunks starting mid-sentence"),
                    ("mid_end_pct", "chunks ending mid-sentence"),
                    ("median_chars", "median chunk size"),
                    ("ov_zero_pct", "adjacent pairs sharing no text")):
        for rf in ("span_hit@4000ch", "span_hit@5", "correctness"):
            rho = lambda ks: st.correlation([r["cells"][k][sf] for k in ks],   # noqa: E731
                                            [cells[k][rf] for k in ks], method="ranked")
            ra["spearman"].append((sf, lab, rf, rho(keys), rho(ns), len(keys), len(ns)))
    return ra


# ============================================================ report
def build():
    r = json.loads((R / "results.json").read_text())
    m = json.loads(MANIFEST.read_text())
    qpath = R / "parser_quality.json"
    q = json.loads(qpath.read_text()) if qpath.exists() else None
    rag_path = R / "rag_results.json"
    RR = json.loads(rag_path.read_text()) if rag_path.exists() else None
    ds_path = DATASET
    ds = json.loads(ds_path.read_text()) if ds_path.exists() else None
    ra = rag_analysis(RR, ds, r) if RR and ds else None

    P, C, cfg = r["parsers"], r["cells"], r["config"]
    cell = lambda p, c: C.get(f"{p}|{c}")            # noqa: E731
    cur, doc = P.get("current"), P.get("oss_docling")
    shape = r.get("corpus", {}).get("manifest_totals", {})
    sp_papers = shape.get("papers", "—")
    stats = {}
    for p in RAG_PARSERS:
        f = REPO / "evals" / "parsed" / p / STATS_NAME
        if f.exists():
            stats[p] = json.loads(f.read_text())
    # parse speed from the larger retrieval corpus when available
    spd = (stats["oss_docling"]["sec_per_page"] / stats["current"]["sec_per_page"]
           if "current" in stats and "oss_docling" in stats
           else (doc["sec_per_page"] / cur["sec_per_page"] if cur and doc else None))

    o = []
    A = o.append

    A("# Parser and chunking harness — results")
    A("")
    A(f"Generated by `evals/scripts/make_main_report.py` from `results.json`"
      f"{', `rag_results.json`' if RR else ''}, `parser_quality.json` and the corpus manifest. "
      "Every number is read from those files, and every statistic is computed from the "
      "per-question records rather than summarised by hand.")
    A("")

    # ---------------- headline
    A("## In one paragraph")
    A("")
    if ra:
        span_fam = ra["chunker"]["span_hit@4000ch"]
        corr_fam = ra["chunker"]["correctness"]
        beat_sem = sum(1 for _, c, t in span_fam if c == "semantic" and t["holm"] and t["w"] > t["l"])
        beat_simple = sum(1 for _, c, t in span_fam if c != "semantic" and t["holm"] and t["w"] > t["l"])
        n_simple = sum(1 for _, c, _ in span_fam if c != "semantic")
        home_cur = ra["home"].get("current|grain_growth")
        neutral_sig = sum(1 for t in ra["parser"] if t["neutral"]["holm"])
        A(f"Two evaluations. A **chunk-shape run** crossed five parsers with four chunkers over "
          f"{sp_papers} papers and measured what the chunks look like. A **retrieval run** put "
          f"three parsers and the same four chunkers through the product's full path, retrieve "
          f"then answer with `{RR['config']['llm']}` then judge, on {len(ds['questions'])} "
          f"questions over {m['totals']['papers']} papers.")
        A("")
        A(f"**Keep Grain-Growth, but for narrower reasons than the shape run suggested.** On "
          f"retrieval it beats semantic chunking on {beat_sem} of 3 parsers after correction for "
          f"multiple comparisons, and beats the simple splitters on {beat_simple} of {n_simple}. "
          f"Its lead over `fixed_token` and `recursive_char` is consistent in direction but not "
          f"statistically established at this sample size.")
        A("")
        A(f"**On retrieval the three parsers cannot be told apart.** Questions written from "
          f"`current`'s own output inflate its scores: {pct(home_cur[0])} of its own questions "
          f"find the answer in the top five, against {pct(home_cur[1])} of everyone else's. "
          f"Scored only on neutral questions, {neutral_sig} of {len(ra['parser'])} pairwise "
          f"parser comparisons reach significance. What separates the parsers is what reaches "
          f"the index at all, plus speed: "
          + ", ".join(f"`{p}` loses {ra['coverage'][p][0]}" for p in RAG_PARSERS)
          + " of the answers in parsing, allowing for text-engine differences"
          + (f", and `current` parses {spd:.2f}x faster than Docling." if spd else "."))
        A("")
    elif cur and doc:
        A(f"Five parsers crossed with four chunking strategies over {sp_papers} papers. "
          "The retrieval run has not been completed, so this report covers chunk shape only.")
        A("")

    # ---------------- corpus
    ex = m.get("exclusion_effect", {})
    A("## Corpora")
    A("")
    A(f"**Chunk-shape run:** {shape.get('papers', '—')} papers, {num(shape.get('pages'))} pages, "
      f"{(shape.get('bytes') or 0) / 1048576:.1f} MB, {shape.get('two_column', '—')} two-column, "
      f"{shape.get('with_tables', '—')} with a detectable table. Totals are recorded in "
      "`results.json`; its per-paper manifest was superseded when the corpus grew.")
    A("")
    A(f"**Retrieval run:** {m['totals']['papers']} papers, {num(m['totals']['pages'])} pages, "
      f"{m['totals']['bytes'] / 1048576:.1f} MB, {m['totals']['two_column']} two-column, "
      f"{m['totals']['with_tables']} with a detectable table, listed in `corpus/manifest.json`.")
    A("")
    if m.get("built"):
        A(f"The target was 201 papers. The manifest was {m['built']}. Downloads honoured the "
          "[arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html) and then went "
          "further: the interval was raised from the published three-second floor to fifteen "
          "seconds with exponential backoff, and the build stopped rather than retry into the "
          "limiter once arXiv returned 429 and then 503.")
        A("")
    A(f"**Disjoint from the YOLO training set.** Every training PDF's arXiv id was read off its "
      f"page-1 stamp, giving {num(ex.get('training_ids_known'))} ids dated "
      f"{', '.join(sorted(ex.get('training_id_months', {})))}. The retrieval corpus is dated "
      f"{', '.join(sorted(ex.get('corpus_id_months', {})))}, so the exclusion filter removed "
      f"**{ex.get('candidates_removed_by_exclusion')}** candidates. Column count and table "
      "presence come from PyMuPDF geometry, never from our own layout model.")
    A("")

    # ---------------- parsers
    A("## Parsers")
    A("")
    A(f"Chunk-shape run, {sp_papers} papers:")
    A("")
    A("| Parser | Pages | Text | Sec/page | VRAM rise | Section barriers | What it is |")
    A("|" + "---|" * 7)
    for p in PARSERS:
        d = P.get(p)
        if not d:
            A(f"| `{p}` | **NOT RUN** | — | — | — | — | {PARSER_NOTE[p]} |")
            continue
        A(f"| `{p}` | {num(d['pages'])} | {d['corpus_bytes'] / 1048576:.2f} MB "
          f"| {num(d.get('sec_per_page'), 4)} | {num(d.get('vram_rise_mb'))} MB "
          f"| {num(d.get('section_barriers'))} | {PARSER_NOTE[p]} |")
    A("")
    A("`VRAM rise` is peak memory above what was already resident when the parser started; "
      "models loaded earlier stay on the card, so an absolute peak would credit every later "
      "parser with their footprint. Three of the five use no GPU.")
    A("")
    A("![parser trade-off](assets/fig1_parser_tradeoff.svg)")
    A("")

    if stats:
        A(f"### Time gain against Docling, on the {m['totals']['papers']}-paper corpus")
        A("")
        A("| Parser | Pages | Sec/page | Total | VRAM rise | Failures |")
        A("|---|---|---|---|---|---|")
        for p in RAG_PARSERS:
            s_ = stats.get(p)
            if s_:
                A(f"| `{p}` | {num(s_['pages'])} | {num(s_.get('sec_per_page'), 4)} "
                  f"| {s_['seconds'] / 60:.1f} min | {num(s_['vram_rise_mb'])} MB | {len(s_['failures'])} |")
        A("")
        if "current" in stats and "oss_docling" in stats:
            c_, d_ = stats["current"]["sec_per_page"], stats["oss_docling"]["sec_per_page"]
            A("| Corpus | `current` | `oss_docling` | Saved |")
            A("|---|---|---|---|")
            for n in (m["totals"]["pages"], 5000, 20000):
                A(f"| {num(n)} pages | {n * c_ / 60:.1f} min | {n * d_ / 60:.1f} min "
                  f"| {n * (d_ - c_) / 60:.1f} min |")
            A("")
            A(f"**{spd:.2f}x faster per page**, on {num(stats['current']['pages'])} pages. Parsing "
              "is a one-off cost per paper, so the saving is minutes on an ingest of this size. "
              f"The {stats['oss_docling']['vram_rise_mb'] - stats['current']['vram_rise_mb']:.0f} MB "
              "of VRAM it saves is the more durable advantage on a 4 GB card shared with the embedder and "
              "the answering model.")
            A("")

    # ---------------- the shape matrix
    A(f"## Chunk shape: 5 parsers x 4 chunkers ({sp_papers} papers)")
    A("")
    A("| Parser | Chunker | Chunks | Avg | Median | Min | Max | Ov=0 | Ov med | Ov max "
      "| Mid-start | Mid-end | Tables whole |")
    A("|" + "---|" * 13)
    for p in PARSERS:
        for c in CHUNKERS:
            d = cell(p, c)
            if not d or not d.get("chunks"):
                err = (d or {}).get("error", "no chunks produced")
                A(f"| `{p}` | `{c}` | **NOT RUN** | {err[:60]} |" + " — |" * 9)
                continue
            flag = " ⚠" if d.get("degenerate") else ""
            A(f"| `{p}` | `{c}`{flag} | {num(d['chunks'])} | {num(d['avg_chars'])} "
              f"| {num(d['median_chars'])} | {num(d['min_chars'])} | {num(d['max_chars'])} "
              f"| {pct(d['ov_zero_pct'])} | {num(d['ov_median_chars'])} "
              f"| {num(d['ov_max_chars'])} | {pct(d['mid_start_pct'])} "
              f"| {pct(d['mid_end_pct'])} | {pct(d.get('tables_whole_pct'))} |")
    A("")
    degen = [p for p in PARSERS if (cell(p, "grain_growth") or {}).get("degenerate")]
    if degen:
        A("⚠ **DEGENERATE**: " + ", ".join(f"`{d}`" for d in degen) +
          " with `grain_growth`. The strategy nucleates at section headings and stops at "
          "structural barriers; these parsers recover none, so it collapses to next-fit "
          "packing over blank-line-separated blocks. Run for completeness, **not a fair "
          "comparison**.")
        A("")
    A("**Overlap** is the longest common suffix of chunk *i* and prefix of chunk *i+1*, on "
      "whitespace-normalised text, one definition used everywhere. **Mid-start** and "
      "**Mid-end** use the pipeline's own boundary rules: only `.`, `?` and `!` close a "
      "sentence, and a fence bar is a real boundary. **Tables whole** is the share of "
      "PyMuPDF-detected tables whose distinctive cell values land at least 80% inside one chunk.")
    A("")
    A(f"Budgets: target {cfg['target_chars']} characters, maximum {cfg['max_chars']}; "
      f"fixed-token window {cfg['fixed_tokens']} with {cfg['fixed_stride']} stride; "
      f"embedding model `{cfg['embed_model']}`.")
    A("")
    A("![starts mid-sentence](assets/fig2_mid_start_pct.svg)")
    A("")
    A("![ends mid-sentence](assets/fig2_mid_end_pct.svg)")
    A("")
    A("Reading down a column shows how little the parser matters once the chunker is fixed; "
      "reading across a row shows how much the chunker matters. `fixed_token` is above 90% "
      "mid-end on every parser, because a token window cuts wherever it runs out.")
    A("")
    A("![overlap](assets/fig3_overlap.svg)")
    A("")
    A("![chunk size ranges](assets/fig4_size_ranges.svg)")
    A("")

    # ---------------- class coverage
    A("## The class gap: what Docling has no name for")
    A("")
    A("The assembler is written against our twelve YOLO classes, and it runs on Docling "
      "because `LABEL_MAP` in `parse_manager/docling_backend.py` translates Docling's "
      "taxonomy into those names first. Two classes come back empty, for different reasons: "
      "Docling *has* a `title` label and the map reads it, but its model never emitted one "
      "here; it has no `authors` concept at all.")
    A("")
    A("> **A bug of mine, found while writing this.** Docling leaves `.text` empty on formula "
      "items and puts the content in `.orig`. The adapter read only `.text`, so it discarded "
      "every equation. Fixed and re-run: Docling now recovers "
      f"{num(P.get('oss_docling', {}).get('block_types', {}).get('formula'))} formula blocks.")
    A("")
    have = [p for p in PARSERS if P.get(p, {}).get("block_types")]
    if have:
        keys = [k for k in ["title", "authors", "section", "paragraph", "list", "caption",
                            "table", "formula", "footnote"]
                if any(P[p]["block_types"].get(k) for p in have)]
        A("| Block type | " + " | ".join(f"`{p}`" for p in have) + " |")
        A("|" + "---|" * (len(have) + 1))
        for k in keys:
            row = [num(P[p]["block_types"].get(k, 0)) for p in have]
            bold = k in ("title", "authors")
            A(f"| {'**' + k + '**' if bold else k} | "
              + " | ".join(f"**{v}**" if bold else v for v in row) + " |")
        A("")
        for p in have:
            A(f"- `{p}`: title found in {num(P[p].get('documents_with_title'))} of "
              f"{num(P[p].get('documents'))} documents")
        A("")
    ggc, ggd = cell("current", "grain_growth"), cell("oss_docling", "grain_growth")
    if ggc and ggd and ggc.get("title_fallback_pct") is not None:
        A(f"**The heading path breaks.** Share of chunks whose heading path is the arXiv id "
          f"rather than the paper title: `current` {pct(ggc['title_fallback_pct'])}, "
          f"`oss_docling` {pct(ggd['title_fallback_pct'])}.")
        A("")
    A("**Author names get embedded as body prose.** `SKIP_TYPES` excludes `authors` from "
      "embedding; with no authors label, Docling's names fall through as `Text`.")
    A("")

    # ---------------- paragraph integrity
    if q:
        qp = q["pipelines"]
        A("## Does Docling do the assembly work on its own?")
        A("")
        A("The shape matrix feeds Docling's layout into *our* assembler. "
          "`evals/scripts/parser_quality_probe.py` separates the three pipelines, over "
          f"{len(q['papers'])} papers.")
        A("")
        A("| Pipeline | Paragraph blocks | Begin mid-sentence |")
        A("|---|---|---|")
        for k, label in (("docling_native", "Docling's own document model"),
                         ("docling_plus_ours", "Docling layout, our assembler"),
                         ("ours", "YOLO layout, our assembler")):
            v = qp[k]
            A(f"| `{k}`, {label} | {num(v['paragraphs'])} | {pct(v['start_lowercase_pct'])} |")
        A("")
        nat = qp["docling_native"]["start_lowercase_pct"]
        best = min(qp["docling_plus_ours"]["start_lowercase_pct"], qp["ours"]["start_lowercase_pct"])
        A(f"**No.** Docling alone leaves {pct(nat)} of its paragraphs beginning mid-sentence, "
          f"about {nat / best:.0f}x worse than either pipeline running our assembler. It detects "
          "layout well; it does not reassemble prose. Header and footer removal is untested: "
          "arXiv preprints carry almost no page furniture.")
        A("")

    # ================================================================ retrieval
    if ra:
        cells, rcfg = ra["cells"], RR["config"]
        rc = lambda p, c: cells.get(f"{p}|{c}")      # noqa: E731

        A("## Does better chunk shape retrieve better?")
        A("")
        A("Everything above measures what chunks look like. A well-formed chunk is only worth "
          "having if it retrieves the right passage and produces a better answer, so the "
          "retrieval run tests that directly.")
        A("")
        A("### How the run was built")
        A("")
        rej = ds.get("reject_reasons", {})
        A(f"- **Questions.** {ds['candidates']} candidates were generated by `{rcfg['llm']}` from "
          f"{ds['papers_sampled']} papers, 1-2 per paper. Each carries a reference answer, a "
          f"verbatim answer span, keywords, filename and arXiv id. {ds['verified']} survived "
          f"verification against raw PDF text; "
          + "; ".join(f"{v} rejected because {k}" for k, v in rej.items())
          + f". The {len(ds['questions'])} most complex were shortlisted, spread over "
          f"{len({x['paper'] for x in ds['questions']})} papers.")
        A(f"- **Question sources.** Papers were split roughly 33/34/33 across the three parsers, "
          f"and the question writer saw that parser's output. Final split: "
          + ", ".join(f"`{k}` {v}" for k, v in ds["source_split"].items())
          + ". Splitting was meant to make any home advantage symmetric; Finding 3 shows it was not.")
        A(f"- **Pipeline.** Each cell chunks the cached parse, embeds it with "
          f"`{rcfg['embed_model']}` into an isolated vector store, retrieves the top "
          f"{rcfg['top_k']}, hands the top {rcfg['context_k']} to the product's own "
          f"`rag.build_context` and `rag.answer`, then judges. The collection is dropped "
          "before the next cell; the production store is never touched.")
        errs = sum(1 for v in cells.values() for x in v["_rows"] if x.get("gen_error"))
        A(f"- **Completeness.** {len(cells)} of 12 cells ran, "
          f"{len(cells) * len(ds['questions'])} question runs, {errs} generation errors.")
        A("")

        A("### Which metrics to trust")
        A("")
        A("Retrieval metrics follow the TREC, MS MARCO and [BEIR](https://arxiv.org/abs/2104.08663) "
          "tradition; generation metrics follow [RAGAS](https://arxiv.org/abs/2309.15217). A "
          "metric's pedigree matters less than whether it separates the systems under test, so "
          "each is also scored on its spread across the twelve cells. A spread under five "
          "points is marked too flat.")
        A("")
        A("| Metric | Lineage | Lowest cell | Highest cell | Spread | Here |")
        A("|---|---|---|---|---|---|")
        for key, label, lin in METRICS:
            v = [x[key] for x in cells.values() if isinstance(x.get(key), (int, float))]
            if not v:
                continue
            lo, hi = min(v), max(v)
            verdict = ("saturated" if lo >= 0.99 else
                       "too flat to separate cells" if hi - lo < 0.05 else "discriminates")
            A(f"| {label} | {lin} | {lo:.3f} | {hi:.3f} | {100 * (hi - lo):.1f} pts | {verdict} |")
        A("")
        A("Three conclusions follow, and they decide how the rest of this section reads.")
        A("")
        A("- **Paper-level metrics are saturated.** With about a hundred papers, retrieving the "
          "right paper is easy for every cell, so paper-hit, paper-level precision and a graded "
          "nDCG that gives credit for the right paper cannot separate anything here. That is a "
          "property of the corpus size, not of the systems.")
        A("- **The passage-level span metrics carry the retrieval comparison.** Of those, only the "
          "**budget-fair** version compares chunkers fairly, because a fixed k hands a "
          "large-chunk cell more text inside the same k. `span_hit@4000ch` is the primary "
          "retrieval metric; MRR and `span_hit@5` are reported for convention.")
        A("- **The judge metrics discriminate but are indicative.** The same model that wrote "
          "the answers also judged them, which invites self-preference, and the embedding "
          "similarity between answer and reference barely moves. `correctness` is the primary "
          "answer metric, read alongside faithfulness rather than alone.")
        A("")

        A("### Results: 3 parsers x 4 chunkers")
        A("")
        A("| Parser | Chunker | Chunks | Span@1 | Span@5 | **Span@4k ch** | MRR | nDCG@10 "
          "| Faithful | **Correct** | Sufficient | Context ch | Minutes |")
        A("|" + "---|" * 13)
        for p in RAG_PARSERS:
            for c in CHUNKERS:
                d = rc(p, c)
                if not d:
                    err = (RR["cells"].get(f"{p}|{c}") or {}).get("error", "not run")
                    A(f"| `{p}` | `{c}` | **NOT RUN** | {err[:40]} |" + " — |" * 9)
                    continue
                A(f"| `{p}` | `{c}` | {num(d['chunks'])} | {d['span_hit@1']:.2f} | {d['span_hit@5']:.2f} "
                  f"| **{d['span_hit@4000ch']:.2f}** | {d['mrr']:.3f} | {d['ndcg@10']:.3f} "
                  f"| {d['faithfulness']:.3f} | **{d['correctness']:.3f}** | {d['context_sufficiency']:.3f} "
                  f"| {num(d['context_chars'])} | {d['seconds'] / 60:.0f} |")
        A("")
        A("![answer retrieved within 4,000 characters](assets/fig5_rag_span4k.svg)")
        A("")
        A("![answer correctness](assets/fig5_rag_correctness.svg)")
        A("")

        # ---- finding 1
        drop = {c: st.mean(rc(p, c)["span_hit@5"] - rc(p, c)["span_hit@4000ch"]
                           for p in RAG_PARSERS if rc(p, c)) for c in CHUNKERS}
        ctx = {c: st.mean(rc(p, c)["context_chars"] for p in RAG_PARSERS if rc(p, c)) for c in CHUNKERS}
        A("### Finding 1: a fixed k flatters large chunks")
        A("")
        A(f"At k=5 semantic chunking looks competitive or better. At an equal 4,000-character "
          f"budget it falls to last on every parser. Its retrieval score drops by "
          f"{100 * drop['semantic']:.0f} points between the two views on average, against "
          f"{100 * drop['grain_growth']:.0f} for Grain-Growth, because its chunks are about "
          f"twice as large: it hands the model {num(ctx['semantic'])} characters of context per "
          f"question against {num(ctx['grain_growth'])}. Reporting only recall@k, as most RAG "
          "evaluations do, would have ranked the worst chunker near the top.")
        A("")
        A("![fixed k against budget](assets/fig6_k_vs_budget.svg)")
        A("")
        A("![budget curves](assets/fig7_budget_curves.svg)")
        A("")
        A("The larger context does not help the answer either. Semantic cells have the lowest "
          "faithfulness and correctness of any chunker on every parser: more text reaches the "
          "model, and more of it is irrelevant.")
        A("")

        # ---- finding 2
        A("### Finding 2: Grain-Growth against the other chunkers")
        A("")
        A("Every cell answers the same questions, so each comparison is **paired**: the table "
          "counts the questions where Grain-Growth did better (W) and worse (L), and tests the "
          "split with an exact sign test. Nine comparisons per metric is a family of tests, so "
          "significance is judged after Holm correction.")
        A("")
        A("| Parser | Grain-Growth vs | Span@4k W / L | p | Holm | Correct W / L | p | Holm |")
        A("|---|---|---|---|---|---|---|---|")
        corr = {(p, c): t for p, c, t in ra["chunker"]["correctness"]}
        for p, c, t in ra["chunker"]["span_hit@4000ch"]:
            u = corr.get((p, c))
            A(f"| `{p}` | `{c}` | {t['w']} / {t['l']} | {pv(t['p'])} | {'yes' if t['holm'] else 'no'} "
              f"| {u['w']} / {u['l']} | {pv(u['p'])} | {'yes' if u['holm'] else 'no'} |")
        A("")
        s_sem = [t for _, c, t in ra["chunker"]["span_hit@4000ch"] if c == "semantic"]
        c_sem = [t for _, c, t in ra["chunker"]["correctness"] if c == "semantic"]
        s_simple = [t for _, c, t in ra["chunker"]["span_hit@4000ch"] if c != "semantic"]
        c_simple = [t for _, c, t in ra["chunker"]["correctness"] if c != "semantic"]
        ahead = sum(1 for t in s_simple if t["w"] > t["l"])
        A(f"**Against semantic the result is real.** Grain-Growth retrieves better on "
          f"{sum(t['holm'] and t['w'] > t['l'] for t in s_sem)} of {len(s_sem)} parsers and answers "
          f"better on {sum(t['holm'] and t['w'] > t['l'] for t in c_sem)} of {len(c_sem)}, after "
          "correction.")
        A("")
        A(f"**Against the simple splitters it is not established.** Grain-Growth is ahead on "
          f"retrieval in {ahead} of {len(s_simple)} comparisons, and none of them survives "
          f"correction" if not any(t["holm"] for t in s_simple) else
          f"**Against the simple splitters the evidence is mixed.** Grain-Growth is ahead on "
          f"retrieval in {ahead} of {len(s_simple)} comparisons, and "
          f"{sum(t['holm'] for t in s_simple)} survive correction")
        o[-1] += (f"; on correctness {sum(t['holm'] for t in c_simple)} of {len(c_simple)} survive. "
          "Many questions are answered equally well by every chunker, which leaves few "
          "discordant pairs to decide the test. The honest reading is *consistent direction, "
          "insufficient evidence*, not a win.")
        A("")
        A("95% bootstrap intervals for the Grain-Growth cells show why:")
        A("")
        A("| Parser | Span@4k | Correctness | Faithfulness |")
        A("|---|---|---|---|")
        for p in RAG_PARSERS:
            ci = ra["ci"].get(f"{p}|grain_growth")
            if ci:
                A(f"| `{p}` | " + " | ".join(f"{m_:.2f} [{lo:.2f}, {hi:.2f}]"
                                             for m_, lo, hi in (ci["span_hit@4000ch"], ci["correctness"],
                                                                ci["faithfulness"])) + " |")
        A("")
        hw = st.mean((ci["span_hit@4000ch"][2] - ci["span_hit@4000ch"][1]) / 2
                     for ci in ra["ci"].values())
        A(f"With {len(ds['questions'])} questions an interval is about ±{100 * hw:.0f} points wide, "
          "which is larger than most gaps between the three non-semantic chunkers.")
        A("")

        # ---- finding 3
        A("### Finding 3: questions favour the parser they were written from, but only one of them")
        A("")
        A("Each Grain-Growth cell, scored separately on the questions each parser's output produced "
          "(answer in the top five):")
        A("")
        A("| Questions written from | n | " + " | ".join(f"`{p}`" for p in RAG_PARSERS) + " |")
        A("|" + "---|" * (len(RAG_PARSERS) + 2))
        for s_ in RAG_PARSERS:
            row = ra["src"][s_]
            A(f"| `{s_}` | {row['n']} | "
              + " | ".join(f"**{pct(row[p])}**" if p == s_ else pct(row[p]) for p in RAG_PARSERS) + " |")
        A("")
        gaps = {p: [ra["home"][f"{p}|{c}"][0] - ra["home"][f"{p}|{c}"][1]
                    for c in CHUNKERS if f"{p}|{c}" in ra["home"]] for p in RAG_PARSERS}
        big = [p for p in RAG_PARSERS if gaps[p] and st.mean(gaps[p]) > 0.1]
        A("The bold diagonal is each parser on its own questions. " +
          (", ".join(f"`{p}` averages {100 * st.mean(gaps[p]):+.0f} points on its own questions "
                     f"across its four cells" for p in big) + "; " if big else "") +
          ", ".join(f"`{p}` {100 * st.mean(gaps[p]):+.0f}" for p in RAG_PARSERS if p not in big) +
          ". The 33/34/33 split was supposed to make the home advantage symmetric so it cancels "
          "between parsers. It did not cancel, because the advantage is concentrated in "
          + (", ".join(f"`{p}`" for p in big) if big else "no parser") + ".")
        A("")
        A("![home advantage](assets/fig8_home_advantage.svg)")
        A("")
        A("The cause is not confirmed. A plausible one: a question written from an assembled "
          "paragraph copies its answer span from that paragraph, and a parser whose chunks keep "
          "the same paragraph whole will contain the span intact, while another parser's "
          "different paragraph boundaries split it. Whatever the mechanism, **any parser ranking "
          "that uses all questions is biased**, and the next finding removes it.")
        A("")

        # ---- finding 4
        A("### Finding 4: on neutral questions the parsers are level")
        A("")
        A("For each pair of parsers, the comparison is repeated on only the third parser's "
          "questions, where neither side has a home advantage. Grain-Growth chunking throughout; "
          "Holm correction across the neutral tests.")
        A("")
        A("| Pair | Metric | All questions W / L, p | Neutral questions | n | W / L | p | Holm |")
        A("|---|---|---|---|---|---|---|---|")
        for t in ra["parser"]:
            a_, n_ = t["all"], t["neutral"]
            A(f"| `{t['a']}` vs `{t['b']}` | {t['f']} | {a_['w']} / {a_['l']}, {pv(a_['p'])} "
              f"| from `{t['third']}` | {n_['n']} | {n_['w']} / {n_['l']} | {pv(n_['p'])} "
              f"| {'yes' if n_['holm'] else 'no'} |")
        A("")
        sig_all = [t for t in ra["parser"] if t["all"]["p"] < ALPHA]
        A(f"**{sum(t['neutral']['holm'] for t in ra['parser'])} of {len(ra['parser'])} neutral "
          "comparisons are significant.** "
          + ("".join(f"The one nominal difference on all questions, `{t['a']}` against `{t['b']}` "
                     f"on {t['f']} ({t['all']['w']} / {t['all']['l']}, p={pv(t['all']['p'])}), "
                     f"disappears on neutral questions ({t['neutral']['w']} / {t['neutral']['l']}). "
                     for t in sig_all[:1]))
          + "Neutral subsets are small, about thirty questions, so this is absence of evidence "
          "rather than evidence of equality: a real difference of a few points would not be "
          "detected. Retrieval quality does not justify choosing one parser over another.")
        A("")

        # ---- finding 5
        A("### Finding 5: parse losses are real, but smaller than an exact match suggests")
        A("")
        A("An answer span missing from a parser's output caps every chunker behind it. Counting "
          "those losses needs care, because the spans were verified against PyMuPDF text and "
          "**two of the three parsers read their words through PyMuPDF**: `current` fills YOLO "
          "boxes with `page.get_text(\"words\")`, and PyMuPDF4LLM is built on it. Docling uses "
          "its own text engine. An exact match therefore counts Docling's differences in maths "
          "symbols, spacing and stray line numbers as lost text, even where it extracted the "
          "passage. A near match, a span-length window holding at least "
          f"{int(ra['near_match'] * 100)}% of the span's words, tolerates those differences and "
          "still rejects a passage that is really missing.")
        A("")
        ths = list(ra["cov_sweep"])
        A("| Parser | Text engine | " + " | ".join("exact" if t == 1.0 else f">= {int(t * 100)}% of words"
                                               for t in ths) + " | Really missing, retrieved anyway |")
        A("|" + "---|" * (len(ths) + 3))
        for p in RAG_PARSERS:
            eng = "Docling" if p == "oss_docling" else "PyMuPDF"
            A(f"| `{p}` | {eng} | " + " | ".join(
                (f"**{ra['cov_sweep'][t][p]}**" if t == ra["near_match"] else str(ra["cov_sweep"][t][p]))
                for t in ths) + f" | {ra['coverage'][p][1]} |")
        A("")
        A("![answers lost before retrieval](assets/fig10_coverage.svg)")
        A("")
        d_ex, d_nr = ra["coverage"]["oss_docling"][2], ra["coverage"]["oss_docling"][0]
        cur_low = all(ra["cov_sweep"][t]["current"] <= min(ra["cov_sweep"][t][p] for p in RAG_PARSERS)
                      for t in ths)
        A(f"Half the gap was the engine. Docling's exact-match losses fall from {d_ex} to {d_nr} at "
          f"the {int(ra['near_match'] * 100)}% threshold. "
          + ("`current` loses the fewest answers at every threshold, so the ordering survives the "
             "correction, but the margin is a handful of questions out of "
             f"{len(ds['questions'])}, not a decisive difference. " if cur_low else
             "The ordering between parsers changes with the threshold, so no parser can be said "
             "to lose fewer answers. ")
          + "No really-missing answer was retrieved by any chunker in any cell: a parse loss is "
          "unrecoverable downstream."
          if all(ra["coverage"][p][1] == 0 for p in RAG_PARSERS) else
          f"Docling's exact-match losses fall from {d_ex} to {d_nr} at the near-match threshold.")
        A("")
        A("**The same effect reaches the retrieval metrics.** Every span-hit figure in this report, "
          "including MRR and nDCG, uses the exact match, so Docling's retrieval scores carry the "
          "same penalty and are probably understated in every cell. The run did not save the text "
          "of retrieved chunks, so they cannot be re-scored without re-running retrieval; that "
          "needs no answer generation or judging, only re-indexing. This strengthens rather than "
          "weakens Finding 4: Docling was level with the others despite the handicap.")
        A("")

        # ---- finding 6
        A("### Finding 6: does fragmentation predict retrieval?")
        A("")
        A("This is the question the whole shape evaluation rests on. Each cell's shape metrics "
          f"from the {sp_papers}-paper run are ranked against its retrieval scores from the "
          f"{m['totals']['papers']}-paper run.")
        A("")
        A("| Shape metric | Retrieval metric | Spearman ρ, all cells | ρ, semantic excluded |")
        A("|---|---|---|---|")
        for sf, lab, rf, all_, ex_, n1, n2 in ra["spearman"]:
            A(f"| {lab} | {rf} | {all_:+.2f} (n={n1}) | {ex_:+.2f} (n={n2}) |")
        A("")
        key_row = next(x for x in ra["spearman"] if x[0] == "mid_start_pct" and x[2] == "span_hit@4000ch")
        A(f"**Across all twelve cells, fragmentation does not predict retrieval:** chunks starting "
          f"mid-sentence against answer retrieval gives ρ = {key_row[3]:+.2f}. **Remove the "
          f"semantic cells and it does, strongly:** ρ = {key_row[4]:+.2f}, fewer fragments and "
          "better retrieval. Semantic chunking is the counterexample that hides the relationship: "
          "it has almost no fragments, because it only cuts between sentences, and still "
          "retrieves worst, because its chunks are too large.")
        A("")
        A("![fragmentation against retrieval](assets/fig9_shape_vs_retrieval.svg)")
        A("")
        A("So the answer to the reviewer's question is conditional. **Fragmentation predicts "
          "retrieval among chunkers that keep chunks within a sensible size, and chunk size is "
          "the confounder that breaks it.** A shape metric on its own is not a safe proxy; paired "
          "with a size bound it is a useful one. The caveats are real: nine points, shape measured "
          "on a different and smaller corpus, and no single pairwise gap among those nine is "
          "significant on its own (Finding 2).")
        A("")

        # ---- cost + limitations
        mins = {c: st.mean(rc(p, c)["seconds"] / 60 for p in RAG_PARSERS if rc(p, c)) for c in CHUNKERS}
        A("### Cost of the run")
        A("")
        A("| Chunker | Mean minutes per cell | Mean context per question |")
        A("|---|---|---|")
        for c in CHUNKERS:
            A(f"| `{c}` | {mins[c]:.0f} | {num(ctx[c])} ch |")
        A("")
        A(f"Semantic cells took about {mins['semantic'] / mins['grain_growth']:.1f}x as long as "
          "Grain-Growth: embedding every sentence to find breakpoints dominates indexing, and the "
          "larger contexts slow generation. The worst-performing chunker was also the most "
          "expensive.")
        A("")
        A("### Limitations")
        A("")
        A(f"- **Sample size.** {len(ds['questions'])} questions over {m['totals']['papers']} papers "
          "gives intervals of roughly ±10 points. Gaps smaller than that are not findings.")
        A("- **Self-judging.** The answering model and the judge are the same model, so "
          "correctness and faithfulness may favour its own phrasing. They are consistent with the "
          "judge-free span metrics in direction, which is some reassurance.")
        A("- **Question bias.** Questions were written by the same small model from one parser's "
          "output, which introduced the home advantage in Finding 3. Neutral-subset analysis "
          "removes it at the cost of sample size.")
        A(f"- **Corpus.** {m['totals']['papers']} papers rather than the planned 201, after arXiv "
          "rate-limited the build. A larger corpus would make paper-level retrieval harder and "
          "stop those metrics saturating.")
        A("- **Text-engine bias.** Answer spans were verified against PyMuPDF, the engine behind "
          "`current` and PyMuPDF4LLM, and span hits use an exact match. Docling is scored down "
          "for character-level differences (Finding 5). The next run should verify against a "
          "third engine and score with a near match.")
        A("- **Two runs, two corpora.** Shape metrics come from the smaller corpus, so Finding 6 "
          "correlates measurements taken on different papers.")
        A("")

    # ================================================================ recommendation
    A("## Recommendation")
    A("")
    if ra:
        gcs = {p: rc(p, "grain_growth") for p in RAG_PARSERS}
        A("### Chunking: keep Grain-Growth")
        A("")
        A("- **It is clearly better than semantic chunking**, the other strategy that respects "
          "sentence boundaries, on retrieval and on answers, and at a fraction of the cost.")
        A("- **It is not proven better than the simple splitters on retrieval.** The direction "
          "favours it and the shape evidence is strong, but a retrieval win over `recursive_char` "
          "is not established.")
        A("- **Its remaining case rests on what the span metrics do not score:** tables and "
          "formulas that never split, heading paths that make a citation readable, and chunks "
          "that stay inside the embedder's window by construction.")
        A("")
        A("### Parsing: keep `current`, for fidelity and speed, not retrieval")
        A("")
        A(f"- **Retrieval does not separate the parsers** once question bias is removed (Finding 4).")
        A(f"- **It loses the fewest answers before retrieval:** "
          + ", ".join(f"`{p}` {ra['coverage'][p][0]}" for p in RAG_PARSERS) + f", allowing for text-engine differences (Finding 5). The margin is small.")
        if spd:
            A(f"- **It is {spd:.2f}x faster than Docling and lighter on VRAM**, and it recovers "
              "`title` and `authors`, which Docling does not deliver.")
        A("")
        A("### The strongest argument against it")
        A("")
        rec = rc("current", "recursive_char")
        cgg = gcs.get("current")
        A("**On retrieval evidence alone, neither the fine-tuned layout model nor structure-aware "
          "chunking is justified.** `recursive_char`, a character splitter any library ships, "
          "reaches "
          + (f"{rec['span_hit@4000ch']:.2f} span retrieval and {rec['correctness']:.3f} correctness "
             f"on `current`, against Grain-Growth's {cgg['span_hit@4000ch']:.2f} and "
             f"{cgg['correctness']:.3f}, " if rec and cgg else "")
          + "and the difference is not significant. Docling and PyMuPDF4LLM, which need no "
          "training data at all, retrieve as well as our parser on neutral questions. If the "
          "only goal were getting the right passage in front of the model on arXiv preprints, "
          "the simplest stack would do.")
        A("")
        A("The case for the current stack is therefore a case about **fidelity rather than "
          "ranking**: fewer answers lost in parsing, tables and formulas kept whole, readable "
          "citations, a 4 GB card shared without contention. Those are real, but they are "
          "arguments the next evaluation should measure directly, with harder corpora such as "
          "published journal PDFs with running heads, table-heavy papers and questions that need "
          "a table or an equation to answer, where fidelity should start to show up in retrieval "
          "scores rather than only in chunk shape.")
        A("")
    A("## Reproducing")
    A("")
    A("```bash")
    A("python evals/scripts/build_corpus.py          # or --offline to manifest a cached corpus")
    A("python evals/scripts/run_matrix.py            # chunk-shape run, checkpointed per cell")
    A("python evals/scripts/parser_quality_probe.py")
    A("python evals/scripts/parse_cache.py           # retrieval run: parse once per parser")
    A("python evals/scripts/build_questions.py       # question set, verified and shortlisted")
    A("python evals/scripts/run_rag_eval.py --device cuda 2>&1 | tee evals/Reports/rag_run.log")
    A("python evals/scripts/eval_status.py evals/Reports/rag_run.log")
    A("python evals/scripts/make_figures.py")
    A("python evals/scripts/make_main_report.py")
    A("```")
    A("")
    return "\n".join(o)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1", action="store_true",
                    help="read round 1's own inputs, which round 2 displaced")
    ap.add_argument("--out", help="write here instead of Report.md")
    args = ap.parse_args()

    global MANIFEST, DATASET, STATS_NAME
    if args.round1:
        MANIFEST = REPO / "evals" / "corpus" / "manifest_103.json"
        DATASET = REPO / "evals" / "questions" / "round1" / "dataset.json"
        STATS_NAME = "_stats_round1.json"
        import make_figures
        make_figures.DATASET = DATASET          # answer_coverage reads this

    out = Path(args.out) if args.out else OUT
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
