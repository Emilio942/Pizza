#!/usr/bin/env python3
"""
Analyze latest JLCPCB upload artifacts.

- Finds most recent Gerber ZIP in manufacturing/JLCPCB_UPLOAD/
- Extracts ZIP into a timestamped folder under .../JLCPCB_UPLOAD/extracted/
- Validates presence of required Gerber layers/files
- Validates BOM and CPL CSV schema and basic stats
- Writes JSON and Markdown summaries in hardware/docs/
"""
from __future__ import annotations
import csv
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "manufacturing" / "JLCPCB_UPLOAD"
DOCS_DIR = ROOT / "docs"

REQUIRED_GERBERS = {
    "GTL": "Top Copper",
    "GBL": "Bottom Copper",
    "GTS": "Top Solder Mask",
    "GBS": "Bottom Solder Mask",
    "GTO": "Top Silkscreen",
    "GBO": "Bottom Silkscreen",
    "GKO": "Board Outline",
}
OPTIONAL_GERBERS = {
    "GTP": "Top Paste",
    "GBP": "Bottom Paste",
}
DRILL_SUFFIXES = {"txt", "drl"}

BOM_REQUIRED_COLUMNS = {"Designator", "Comment", "Footprint", "LCSC", "LCSC Part", "LCSC Part Number"}
CPL_REQUIRED_COLUMNS = {"Designator", "Mid X", "Mid Y", "Layer", "Rotation"}


@dataclass
class FileInfo:
    path: str
    size: int


@dataclass
class GerberCheck:
    present: Dict[str, bool]
    drill_present: bool
    files: List[FileInfo]


@dataclass
class CsvCheck:
    path: Optional[str]
    exists: bool
    columns: List[str]
    has_required: bool
    row_count: int


@dataclass
class UploadReport:
    generated_at: str
    upload_dir: str
    zip_file: Optional[str]
    extracted_to: Optional[str]
    gerber: GerberCheck
    bom: CsvCheck
    cpl: CsvCheck
    warnings: List[str]


def find_latest_zip(upload_dir: Path) -> Optional[Path]:
    zips = sorted(upload_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def extract_zip(zippath: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = UPLOAD_DIR / "extracted" / f"{zippath.stem}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath, 'r') as zf:
        zf.extractall(out_dir)
    return out_dir


def collect_files(folder: Path) -> List[Path]:
    return [p for p in folder.rglob('*') if p.is_file()]


def check_gerbers(files: List[Path]) -> GerberCheck:
    present: Dict[str, bool] = {k: False for k in REQUIRED_GERBERS}
    drill_present = False
    file_infos: List[FileInfo] = []

    for f in files:
        file_infos.append(FileInfo(str(f), f.stat().st_size))
        name = f.name
        # detect required/optional by extension suffix (case-insensitive)
        m = re.search(r"\.([A-Za-z0-9]{2,4})$", name)
        if m:
            ext_raw = m.group(1)
            ext = ext_raw.lower()

            # Map outline variants (.gko, .gm1, sometimes .gml)
            if ext in {"gko", "gm1", "gml"}:
                present["GKO"] = True

            # Standard layer codes
            code_map = {
                "gtl": "GTL",
                "gbl": "GBL",
                "gts": "GTS",
                "gbs": "GBS",
                "gto": "GTO",
                "gbo": "GBO",
                # optional
                "gtp": "GTP",
                "gbp": "GBP",
            }
            if ext in code_map:
                code = code_map[ext]
                if code in present:
                    present[code] = True

            # Drill files
            if ext in DRILL_SUFFIXES:
                drill_present = True

    return GerberCheck(present=present, drill_present=drill_present, files=file_infos)


def read_csv_info(path: Path, required_columns: set[str]) -> CsvCheck:
    if not path.exists():
        return CsvCheck(path=str(path), exists=False, columns=[], has_required=False, row_count=0)
    try:
        with path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = [c.strip() for c in (reader.fieldnames or [])]
            row_count = sum(1 for _ in reader)
        has_required = any(col in set(columns) for col in required_columns)
        return CsvCheck(path=str(path), exists=True, columns=columns, has_required=has_required, row_count=row_count)
    except Exception:
        # Fallback: try simple CSV
        with path.open('r', encoding='utf-8') as f:
            first = f.readline()
            columns = [c.strip() for c in first.split(',')]
        return CsvCheck(path=str(path), exists=True, columns=columns, has_required=any(c in columns for c in required_columns), row_count=0)


def main() -> int:
    warnings: List[str] = []
    zip_path = find_latest_zip(UPLOAD_DIR)

    extracted_dir: Optional[Path] = None
    gerber_check = GerberCheck(present={k: False for k in REQUIRED_GERBERS}, drill_present=False, files=[])

    if zip_path:
        extracted_dir = extract_zip(zip_path)
        files = collect_files(extracted_dir)
        gerber_check = check_gerbers(files)
    else:
        warnings.append("No ZIP found in JLCPCB_UPLOAD")

    # BOM & CPL in upload dir
    bom_path = UPLOAD_DIR / "bom_jlcpcb.csv"
    cpl_path = UPLOAD_DIR / "cpl_jlcpcb.csv"

    bom_info = read_csv_info(bom_path, BOM_REQUIRED_COLUMNS)
    if not bom_info.exists:
        warnings.append("BOM file missing: bom_jlcpcb.csv")
    elif not bom_info.has_required:
        warnings.append("BOM likely missing LCSC-related column (LCSC / LCSC Part / LCSC Part Number)")

    cpl_info = read_csv_info(cpl_path, CPL_REQUIRED_COLUMNS)
    if not cpl_info.exists:
        warnings.append("CPL file missing: cpl_jlcpcb.csv")

    # Gerber required checks
    for code, label in REQUIRED_GERBERS.items():
        if not gerber_check.present.get(code, False):
            warnings.append(f"Gerber missing: {code} ({label})")
    if not gerber_check.drill_present:
        warnings.append("Drill file missing (.TXT or .DRL)")

    report = UploadReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        upload_dir=str(UPLOAD_DIR),
        zip_file=str(zip_path) if zip_path else None,
        extracted_to=str(extracted_dir) if extracted_dir else None,
        gerber=gerber_check,
        bom=bom_info,
        cpl=cpl_info,
        warnings=warnings,
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DOCS_DIR / "mfg_status.json"
    out_md = DOCS_DIR / "mfg_status.md"

    with out_json.open('w', encoding='utf-8') as f:
        json.dump(asdict(report), f, indent=2)

    # Markdown summary
    present = report.gerber.present
    md = [
        f"# JLCPCB Upload Status",
        f"Generated: {report.generated_at}",
        "",
        f"- Upload dir: `{report.upload_dir}`",
        f"- ZIP: `{report.zip_file}`",
        f"- Extracted to: `{report.extracted_to}`",
        "",
        "## Gerber Presence",
    ]
    for code, label in REQUIRED_GERBERS.items():
        md.append(f"- {code} ({label}): {'✅' if present.get(code) else '❌'}")
    md.append(f"- Drill (TXT/DRL): {'✅' if report.gerber.drill_present else '❌'}")

    md += [
        "",
        "## BOM",
        f"- Path: `{report.bom.path}`",
        f"- Exists: {'✅' if report.bom.exists else '❌'}",
        f"- Rows: {report.bom.row_count}",
        f"- Has LCSC column: {'✅' if report.bom.has_required else '❌'}",
        "",
        "## CPL",
        f"- Path: `{report.cpl.path}`",
        f"- Exists: {'✅' if report.cpl.exists else '❌'}",
        f"- Rows: {report.cpl.row_count}",
        "",
        "## Warnings",
    ]
    if report.warnings:
        md += [f"- {w}" for w in report.warnings]
    else:
        md.append("- None")

    with out_md.open('w', encoding='utf-8') as f:
        f.write("\n".join(md))

    # Print short summary for console
    print(json.dumps({
        "json": str(out_json),
        "md": str(out_md),
        "warnings": warnings,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
