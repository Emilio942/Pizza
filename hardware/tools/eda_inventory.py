#!/usr/bin/env python3
"""
EDA Inventory Tool

Scans the hardware project directory for KiCad-related files and outputs a JSON
report listing file paths, sizes, and modification times.

Finds:
- .kicad_pcb, .kicad_sch, .kicad_pro
- Symbol libraries (.kicad_sym), sym-lib-table
- Footprint libraries (.kicad_mod within .pretty dirs), fp-lib-table

Usage:
  python eda_inventory.py [root_dir]
Defaults to the directory containing this script's parent (hardware/).

Writes:
  hardware/docs/eda_inventory.json
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

# File patterns to detect
EXT_TYPES = {
    ".kicad_pcb": "kicad_pcb",
    ".kicad_sch": "kicad_sch",
    ".kicad_pro": "kicad_pro",
    ".kicad_sym": "symbol_lib",
    ".kicad_mod": "footprint_mod",
    ".lib": "legacy_symbol_lib",
    ".dcm": "legacy_symbol_doc",
}
TABLE_FILES = {
    "sym-lib-table": "sym_lib_table",
    "fp-lib-table": "fp_lib_table",
}


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def collect(root: str) -> Dict:
    files: List[Dict] = []
    by_type: Dict[str, int] = {}

    # Normalize root
    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Detect .pretty footprint library directories explicitly
        if dirpath.endswith('.pretty'):
            # Still traverse; .kicad_mod handled below
            pass

        # Table files in any directory
        for table_name, ttype in TABLE_FILES.items():
            table_path = os.path.join(dirpath, table_name)
            if os.path.isfile(table_path):
                st = os.stat(table_path)
                files.append({
                    "type": ttype,
                    "path": table_path,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "mtime_iso": iso(st.st_mtime),
                })
                by_type[ttype] = by_type.get(ttype, 0) + 1

        # Regular files by extension
        for name in filenames:
            full = os.path.join(dirpath, name)
            _, ext = os.path.splitext(name)
            if ext in EXT_TYPES:
                ttype = EXT_TYPES[ext]
                try:
                    st = os.stat(full)
                except FileNotFoundError:
                    continue
                files.append({
                    "type": ttype,
                    "path": full,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "mtime_iso": iso(st.st_mtime),
                })
                by_type[ttype] = by_type.get(ttype, 0) + 1

    report = {
        "scanned_root": root,
        "generated_at": iso(datetime.now(tz=timezone.utc).timestamp()),
        "files": sorted(files, key=lambda x: (x["type"], x["path"])),
        "stats": {
            "total_files": len(files),
            "by_type": by_type,
        },
    }
    return report


def main():
    # Default root: one level up from this script (hardware/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    root = sys.argv[1] if len(sys.argv) > 1 else default_root

    report = collect(root)

    # Output path under hardware/docs
    docs_dir = os.path.join(default_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_path = os.path.join(docs_dir, "eda_inventory.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print short summary to stdout
    print(json.dumps({
        "out_path": out_path,
        "total": report["stats"]["total_files"],
        "by_type": report["stats"]["by_type"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
