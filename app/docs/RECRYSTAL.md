# Anneal — recrystal: the layout parser, and what happens to it next

> Context for a session working on the parser specifically. For the pipeline as
> a whole, start at [../README.md](../README.md); for how parsing works, read
> [PARSING.md](PARSING.md).

## What `recrystal` is

The name covers **two stages together**:

1. **the detector** — a fine-tuned YOLOv11 model, 12 region classes, run through
   `onnxruntime-gpu` (never `ultralytics` at runtime, which is AGPL);
   `app/parse_manager/pdf_parser.py`, `layout_detector.py`, `onnx_detector.py`
2. **the assembler** — column-aware reading order and paragraph reconstruction;
   `app/parse_manager/txt_processor.py`

In annealing, recrystallisation is the stage that forms new strain-free grains,
and it precedes grain growth — the chunker this parser feeds. The evaluation,
the report and the paper all use `recrystal` for the pair. It replaced the
label `current` on 2026-09-21 (Amendment 6 in
`evals/Reports/round2_amendments.json`), a pure relabel verified by regenerating
both reports byte-identically.

**The assembler is not exclusive to it.** The `oss_docling` arm runs Docling's
layout through the same assembler, which is what makes the parser comparison a
comparison of *detectors*. That assembler now lives in its own library,
`textreflow` (https://github.com/mudit-choudhary/textreflow);
`txt_processor.py` keeps only the file I/O and registry glue around it.

## What it is measured to do

From `evals/Reports/Report.md` (514 papers, 400 questions, pre-registered):

- retrieves **significantly better** than both open-source parsers under every
  chunker: pooled +0.064 against Docling, +0.122 against PyMuPDF4LLM
- loses the fewest answer spans in parsing: **24 of 400**, against Docling's 48
  and PyMuPDF4LLM's 61
- parses **2.88× faster than Docling** per page (0.1375 s/page against 0.3958)
- the assembler, not the detector, is what repairs prose: Docling's own model
  leaves 17.4% of paragraphs starting mid-sentence, 2.9% through this assembler
  (5 papers; a 60-paper re-run may exist as `parser_quality_60.json`)

## Licensing — why this cannot become a permissive library

Two independent AGPL sources, either one sufficient:

- **PyMuPDF** extracts the words the detector's boxes are matched against
  (`pdf_parser.py` imports `fitz`), and PyMuPDF is AGPL
- the **weights** are fine-tuned from Ultralytics YOLO11, which is AGPL, so the
  weights are a derivative

Anneal is therefore AGPL-3.0 (see `LICENSE` at the repository root), and the
detector stays inside it. If the detector is ever to be shared, the honest form
is a **Hugging Face model repository under AGPL-3.0**, carrying the weights plus
an ONNX export, not a PyPI package.

The pieces that link nothing are published permissively instead:
`grain-growth-chunking` (Apache-2.0, on PyPI, DOI 10.5281/zenodo.22862100) and
`textreflow` (Apache-2.0, extracted; pinned in `app/requirements.txt`, not yet on PyPI).

## Open items

- **Reserve the PyPI name `recrystal`** — free as of 2026-09-21. The clean way
  is a *pending publisher* at <https://pypi.org/manage/account/publishing/>,
  which holds the name without uploading anything. A placeholder release also
  works but PyPI discourages holding unused names.
- **Decide whether to publish the weights** on Hugging Face under AGPL, with the
  ONNX export and the 12-class label map.
- **Fine-tuning is documented** in [PARSING.md](PARSING.md); training output
  lives outside the repo, in `$ANNEAL_HOME/runs/`.

## Where things are

| | |
|---|---|
| Detector code | `app/parse_manager/{pdf_parser,layout_detector,onnx_detector}.py` |
| Assembler code | `app/parse_manager/txt_processor.py` |
| Weights | `~/.local/share/anneal/models/` (`.pt` and `.onnx`, 141 MB) |
| Parse cache used by the evaluation | `evals/parsed/recrystal/` |
| Tests | `tests/test_pdf_parser.py`, `tests/test_onnx_detector.py`, `tests/test_txt_processor.py` |
| Venv | `virtual_environments/annealenv` — app requirements exactly; use `probeenv` for anything needing Docling |
