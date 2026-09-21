"""Stage 2 of parsing: layout JSON -> assembled, tagged text.

Takes the region-level output of pdf_parser.py and produces the final text
used for chunking/embedding:

- column-aware reading order (full-width bands, then left/right columns)
- paragraph reconstruction across regions, columns, and pages, with
  de-hyphenation
- structure preserved as tags: `#` Title, `##` Section-header, [AUTHORS],
  [CAPTION], [TABLE], [FORMULA], [FOOTNOTE]
- page headers/footers and text inside pictures excluded from the flow
  (kept in the companion JSON as metadata)

Output: data/processed/<name>.txt and <name>.json

The assembly is the `textreflow` library; this module adds file I/O and the
registry update around it.
"""

import json
from pathlib import Path

# The assembler itself is the published `textreflow` library (Apache-2.0),
# extracted from this file; these names are re-exported so callers and tests
# keep importing them from here. What stays is Anneal's glue: paths, the
# registry, the CLI.
from textreflow import (  # noqa: F401
    LayoutAssembler,
    assemble,
    collect_hyphenated_vocab,
    ends_terminally,
    join_hyphenated,
    join_lines,
    merge_paragraph,
    order_regions,
    render_txt,
)

from config import (
    PROCESSED_DIR,
    FULL_WIDTH_FRACTION,
    SINGLE_COLUMN_FRACTION,
)


def resolve_layout_json(arg):
    """Resolve a CLI argument to a layout JSON: a path as given, else a
    filename or bare stem looked up in PARSED_DIR."""
    from config import PARSED_DIR

    p = Path(arg)
    if p.exists():
        return p
    for candidate in (PARSED_DIR / p.name, PARSED_DIR / f"{p.stem}.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"'{arg}' not found (also tried {PARSED_DIR / p.name})")


def process_layout_json(input_json, output_txt=None, output_json=None, update_registry=True):
    """Assemble one layout JSON into processed text. Returns the txt path."""
    input_json = Path(input_json)
    with open(input_json, "r", encoding="utf-8") as f:
        layout = json.load(f)

    blocks, dropped = assemble(layout, full_width_fraction=FULL_WIDTH_FRACTION,
                               single_column_fraction=SINGLE_COLUMN_FRACTION)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_json.stem
    output_txt = Path(output_txt) if output_txt else PROCESSED_DIR / f"{stem}.txt"
    output_json = Path(output_json) if output_json else PROCESSED_DIR / f"{stem}.json"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(render_txt(blocks))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "source": layout.get("source_pdf"),
            "num_pages": layout.get("num_pages"),
            "pdf_pages": layout.get("pdf_pages"),
            "dropped": dropped,
            "blocks": blocks,
        }, f, indent=2)

    if update_registry:
        _update_registry(stem)
    print(f"[processor] {stem}: {len(blocks)} blocks -> {output_txt.name}")
    return output_txt


def _update_registry(filename):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.registry_client import RegistryClient, RegistryUnavailable
    try:
        if not RegistryClient().set_status(filename, "processed"):
            print(f"[processor] registry rejected status update for {filename} (not registered?)")
    except RegistryUnavailable as e:
        print(f"[processor] {e}; status not updated")


if __name__ == "__main__":
    import sys
    from config import PARSED_DIR

    targets = sys.argv[1:] or [str(p) for p in sorted(PARSED_DIR.glob("*.json"))[:1]]
    for target in targets:
        process_layout_json(resolve_layout_json(target))
