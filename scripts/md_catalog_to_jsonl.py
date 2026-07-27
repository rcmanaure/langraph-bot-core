"""CLI wrapper around app.services.md_catalog.convert() — for manually
regenerating medical_data/sp-diagnostico-histologico.jsonl from the source
.md (e.g. to inspect the diff before uploading via admin/UI). The admin/UI
upload path (app/services/indexer.py) calls the same convert() function
directly, so both stay in sync with zero duplicated parsing logic.

Usage: python scripts/md_catalog_to_jsonl.py [input.md] [output.jsonl]
Defaults: sp-diagnostico-histologico.md -> medical_data/sp-diagnostico-histologico.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.md_catalog import convert  # noqa: E402


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sp-diagnostico-histologico.md")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("medical_data/sp-diagnostico-histologico.jsonl")

    records = convert(src.read_text(encoding="utf-8"))
    price_items = sum(1 for r in records if r.get("type") in ("biopsy", "cytology", "protocol") and r.get("category") != "políticas")
    print(f"Parsed {len(records)} records ({price_items} priced catalog items) from {src}")

    with dst.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
