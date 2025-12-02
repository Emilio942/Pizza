#!/usr/bin/env python3
"""
Apply rotation and layer overrides to the normalized JLCPCB CPL.

Inputs:
- output/corrected_jlcpcb/cpl_jlcpcb.csv (normalized)
- cpl_overrides.json (optional), format:
  { "U1": {"rotation": 90, "layer": "Top"}, "J1": {"rotation": 180} }

Output:
- output/corrected_jlcpcb/cpl_jlcpcb.csv (overwritten)
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MFG = ROOT / "hardware" / "manufacturing"
OUT = MFG / "output" / "corrected_jlcpcb"
CPL = OUT / "cpl_jlcpcb.csv"
OVR = MFG / "cpl_overrides.json"
UPLOAD = MFG / "JLCPCB_UPLOAD"

def fmt_mm(val: str) -> str:
    try:
        return f"{float(val):.4f}"
    except Exception:
        return val

def main():
    if not CPL.exists():
        print(f"CPL not found: {CPL}")
        return 1
    overrides = {}
    if OVR.exists():
        try:
            overrides = json.loads(OVR.read_text())
        except Exception as e:
            print(f"Warning: failed to parse overrides {OVR}: {e}")
            overrides = {}

    tmp_out = OUT / "cpl_jlcpcb.tmp.csv"
    with CPL.open(newline='', encoding='utf-8') as f_in, tmp_out.open('w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["Designator", "Mid X", "Mid Y", "Layer", "Rotation"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            ref = (row.get("Designator") or "").strip()
            x = fmt_mm(row.get("Mid X", ""))
            y = fmt_mm(row.get("Mid Y", ""))
            layer = row.get("Layer", "Top").strip()
            rot = row.get("Rotation", "0").strip()

            # Normalize layer names
            up = layer.upper()
            if up.startswith('T'):
                layer = 'Top'
            elif up.startswith('B'):
                layer = 'Bottom'

            # Apply overrides
            o = overrides.get(ref) or {}
            if 'rotation' in o:
                try:
                    rot = f"{float(o['rotation']):.2f}"
                except Exception:
                    pass
            if 'layer' in o:
                lay = str(o['layer']).strip().lower()
                if lay.startswith('t'):
                    layer = 'Top'
                elif lay.startswith('b'):
                    layer = 'Bottom'

            writer.writerow({
                "Designator": ref,
                "Mid X": x,
                "Mid Y": y,
                "Layer": layer,
                "Rotation": rot,
            })

    tmp_out.replace(CPL)

    # Refresh upload copies
    UPLOAD.mkdir(parents=True, exist_ok=True)
    (UPLOAD / 'cpl_jlcpcb.csv').write_text(CPL.read_text())
    print(f"Adjusted CPL written: {CPL}\nUpload copy refreshed: {UPLOAD / 'cpl_jlcpcb.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
