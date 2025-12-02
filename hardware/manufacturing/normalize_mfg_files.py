#!/usr/bin/env python3
"""
Normalize existing BOM/CPL CSVs to JLCPCB format for quick Phase 1 validation.

Inputs (existing):
- hardware/manufacturing/bom_jlcpcb.csv
- hardware/manufacturing/cpl_jlcpcb.csv

Outputs (normalized):
- hardware/manufacturing/output/corrected_jlcpcb/bom_jlcpcb.csv
- hardware/manufacturing/output/corrected_jlcpcb/cpl_jlcpcb.csv
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "hardware" / "manufacturing"
OUT_DIR = IN_DIR / "output" / "corrected_jlcpcb"
CORRECTED_BOM = OUT_DIR / "bom" / "bom_jlcpcb.csv"
CORRECTED_CPL = OUT_DIR / "cpl" / "cpl_jlcpcb.csv"

BOM_IN = IN_DIR / "bom_jlcpcb.csv"
CPL_IN = IN_DIR / "cpl_jlcpcb.csv"
BOM_OUT = OUT_DIR / "bom_jlcpcb.csv"
CPL_OUT = OUT_DIR / "cpl_jlcpcb.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_bom() -> dict:
    """Map columns to JLC format: Comment,Designator,Footprint,LCSC Part #,Quantity."""
    issues = {"missing_lcsc": 0, "unknown_pkg": 0}
    # Prefer corrected exporter BOM if available
    if CORRECTED_BOM.exists():
        with CORRECTED_BOM.open(newline="", encoding="utf-8") as f_in, BOM_OUT.open("w", newline="", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = ["Comment", "Designator", "Footprint", "LCSC Part #", "Quantity"]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                comment = row.get("Comment") or row.get("Value") or ""
                designator = row.get("Designator") or row.get("Reference") or ""
                footprint = row.get("Footprint") or row.get("Package") or ""
                lcsc = row.get("LCSC Part #") or row.get("LCSC Part") or row.get("LCSC Part Number") or ""
                qty = row.get("Quantity") or "1"
                if not lcsc or lcsc.strip().upper() in {"N/A", "NA", "NONE", ""}:
                    issues["missing_lcsc"] += 1
                if not footprint or footprint.strip().lower() in {"unknown", "", "smd", "tht"}:
                    issues["unknown_pkg"] += 1
                writer.writerow({
                    "Comment": comment,
                    "Designator": designator,
                    "Footprint": footprint,
                    "LCSC Part #": lcsc,
                    "Quantity": qty,
                })
        return issues

    if not BOM_IN.exists():
        return {"error": f"Missing input BOM: {BOM_IN}"}

    with BOM_IN.open(newline="", encoding="utf-8") as f_in, BOM_OUT.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["Comment", "Designator", "Footprint", "LCSC Part #", "Quantity"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            comment = row.get("Value") or row.get("Comment") or ""
            designator = row.get("Designator") or row.get("Reference") or ""
            footprint = row.get("Package") or row.get("Footprint") or ""
            lcsc = row.get("LCSC Part Number") or row.get("LCSC Part #") or ""
            qty = row.get("Quantity") or row.get("Qty") or "1"

            if not lcsc or lcsc.strip().upper() in {"N/A", "NA", "NONE", ""}:
                issues["missing_lcsc"] += 1
            if not footprint or footprint.strip().lower() in {"unknown", "", "smd", "tht"}:
                issues["unknown_pkg"] += 1

            writer.writerow({
                "Comment": comment,
                "Designator": designator,
                "Footprint": footprint,
                "LCSC Part #": lcsc,
                "Quantity": qty,
            })

    return issues

def _strip_mm(val: str) -> str:
    s = (val or "").strip()
    if s.endswith("mm"):
        s = s[:-2]
    return s.strip()

def normalize_cpl() -> dict:
    """Ensure numeric Mid X/Y in mm (no unit suffix) and standard columns.
    Input columns expected: Designator,Mid X,Mid Y,Layer,Rotation
    Output columns: Designator,Val,Package,Mid X,Mid Y,Rotation,Layer (JLC standard-like)
    """
    issues = {"fixed_units": 0, "zero_rotation": 0}
    # Prefer corrected exporter CPL if available
    if CORRECTED_CPL.exists():
        with CORRECTED_CPL.open(newline="", encoding="utf-8") as f_in, CPL_OUT.open("w", newline="", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)
            # Keep standard 5-column JLC format
            fieldnames = ["Designator", "Mid X", "Mid Y", "Layer", "Rotation"]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                x_raw = row.get("Mid X", "")
                y_raw = row.get("Mid Y", "")
                rot = row.get("Rotation", "0")
                layer = row.get("Layer", "Top")
                # Map shorthand to full layer names
                if layer.upper().startswith("T"):
                    layer = "Top"
                elif layer.upper().startswith("B"):
                    layer = "Bottom"
                x = _strip_mm(x_raw)
                y = _strip_mm(y_raw)
                if x != x_raw or y != y_raw:
                    issues["fixed_units"] += 1
                try:
                    r = float(rot)
                except Exception:
                    r = 0.0
                if abs(r) < 1e-6:
                    issues["zero_rotation"] += 1
                writer.writerow({
                    "Designator": row.get("Designator", ""),
                    "Mid X": x,
                    "Mid Y": y,
                    "Layer": layer,
                    "Rotation": f"{r:.2f}",
                })
        return issues

    if not CPL_IN.exists():
        return {"error": f"Missing input CPL: {CPL_IN}"}

    with CPL_IN.open(newline="", encoding="utf-8") as f_in, CPL_OUT.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["Designator", "Val", "Package", "Mid X", "Mid Y", "Rotation", "Layer"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            x_raw = row.get("Mid X", "")
            y_raw = row.get("Mid Y", "")
            rot = row.get("Rotation", "0")
            layer = row.get("Layer", "Top")

            x = _strip_mm(x_raw)
            y = _strip_mm(y_raw)
            if x != x_raw or y != y_raw:
                issues["fixed_units"] += 1

            try:
                r = float(rot)
            except Exception:
                r = 0.0
            if abs(r) < 1e-6:
                issues["zero_rotation"] += 1

            writer.writerow({
                "Designator": row.get("Designator", ""),
                "Val": row.get("Val", ""),
                "Package": row.get("Package", ""),
                "Mid X": x,
                "Mid Y": y,
                "Rotation": f"{r:.2f}",
                "Layer": layer,
            })

    return issues

def main():
    bom_issues = normalize_bom()
    cpl_issues = normalize_cpl()
    print("BOM normalized to:", BOM_OUT)
    print("CPL normalized to:", CPL_OUT)
    print("Summary:", {"bom": bom_issues, "cpl": cpl_issues})

if __name__ == "__main__":
    main()
