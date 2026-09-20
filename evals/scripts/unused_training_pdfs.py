"""PDFs in the YOLO source folder that never contributed a page to fine-tuning.

    python evals/scripts/unused_training_pdfs.py

A label file is named <pdf stem>_page_<N>.txt (N zero-padded, 2 or 3 digits).
Every stem found in any round's labels/ marks that PDF as used. Anything left in
PDFs/ was never seen by the model, so it can fill the evaluation corpus without
testing the layout model on its own training data.

Writes evals/corpus/unused_training_pdfs.json.
"""

import json
import sys
import re
from pathlib import Path

ROOT = Path("/media/mudit/DarkDwine1/ResearchPapersYOLO_FT")
ROUNDS = ["round_01", "round_02", "round_final"]
PDFS = ROOT / "PDFs"
OUT = Path(__file__).resolve().parent.parent / "corpus" / "unused_training_pdfs.json"

PAGE = re.compile(r"^(.+)_page_\d+\.txt$")


def stem_of(label_name):
    m = PAGE.match(label_name)
    return m.group(1) if m else None


def main():
    used = set()
    for r in ROUNDS:
        d = ROOT / "training_dataset" / r / "labels"
        names = [p.name for p in d.iterdir() if p.suffix == ".txt"]
        stems = {stem_of(n) for n in names} - {None}
        print(f"{r}: {len(names)} label files, {len(stems)} papers")
        used |= stems

    pdfs = sorted(p for p in PDFS.iterdir() if p.suffix.lower() == ".pdf")
    # label stems are capped at 100 characters like the PDF names; the prefix
    # test also covers any label whose name was cut shorter than its PDF's
    unused = [p for p in pdfs
              if p.stem not in used and not any(p.stem.startswith(s) for s in used)]

    print(f"\n{len(used)} unique papers in labels, {len(pdfs)} PDFs, "
          f"{len(pdfs) - len(unused)} used, {len(unused)} never used in training")
    OUT.write_text(json.dumps({
        "label_dirs": [str(ROOT / "training_dataset" / r / "labels") for r in ROUNDS],
        "papers_in_labels": len(used),
        "pdfs_total": len(pdfs),
        "unused": [str(p) for p in unused],
    }, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    assert stem_of("A_B_page_01.txt") == "A_B"
    assert stem_of("X_page_1_page_135.txt") == "X_page_1"
    assert stem_of("notes.txt") is None
    main()
