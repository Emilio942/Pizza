#!/usr/bin/env python3
"""
Run KiCad DRC/ERC checks if supported by current KiCad version.

KiCad 7 (kicad-cli 7.x) does not expose pcb.drc or sch.erc; in that case we
report status="skipped" with an actionable hint. On KiCad 8+, we try to run:
- kicad-cli pcb drc --board <.kicad_pcb> --output <report>
- kicad-cli sch erc --schematic <.kicad_sch> --output <report>

Outputs:
- hardware/docs/drc_erc_report.json
- hardware/docs/drc_erc_report.md
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EDA_DIR = ROOT / "eda"

PCB_FILE = EDA_DIR / "PizzaBoard-RP2040.kicad_pcb"
SCH_FILE = EDA_DIR / "PizzaBoard-RP2040.kicad_sch"


@dataclass
class CmdResult:
    ok: bool
    code: int
    stdout: str
    stderr: str


def run(cmd: list[str]) -> CmdResult:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return CmdResult(ok=(p.returncode == 0), code=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except FileNotFoundError:
        return CmdResult(ok=False, code=127, stdout="", stderr="not found")


def get_kicad_version() -> Optional[str]:
    r = run(["kicad-cli", "--version"])
    if not r.ok and r.code == 127:
        return None
    # Expect something like "7.0.11"
    m = re.search(r"(\d+\.\d+\.\d+|\d+\.\d+)", r.stdout)
    return m.group(1) if m else r.stdout.strip() or None


def version_major(v: str) -> int:
    try:
        return int(v.split(".")[0])
    except Exception:
        return -1


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)

    version = get_kicad_version()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kicad_cli_version": version,
        "pcb_file": str(PCB_FILE),
        "sch_file": str(SCH_FILE),
        "drc": {},
        "erc": {},
        "notes": [],
    }

    if not version:
        result["drc"] = {"status": "skipped", "reason": "kicad-cli not found"}
        result["erc"] = {"status": "skipped", "reason": "kicad-cli not found"}
    else:
        major = version_major(version)
        if major < 8:
            # KiCad 7 lacks CLI DRC/ERC
            result["drc"] = {
                "status": "skipped",
                "reason": f"kicad-cli {version} does not support pcb drc; upgrade to KiCad 8+ or run DRC in GUI / pcbnew Python",
            }
            result["erc"] = {
                "status": "skipped",
                "reason": f"kicad-cli {version} does not support sch erc; upgrade to KiCad 8+ or run ERC in GUI",
            }
            result["notes"].append("You can also try using KiCad's Python with pcbnew to trigger DRC headlessly.")
        else:
            # Try running commands (KiCad 8+ expected)
            drc_out = DOCS / "drc_report.rpt"
            erc_out = DOCS / "erc_report.rpt"
            drc_cmd = [
                "kicad-cli", "pcb", "drc",
                "--board", str(PCB_FILE),
                "--output", str(drc_out),
            ]
            erc_cmd = [
                "kicad-cli", "sch", "erc",
                "--schematic", str(SCH_FILE),
                "--output", str(erc_out),
            ]
            drc_res = run(drc_cmd)
            erc_res = run(erc_cmd)
            result["drc"] = {
                "status": "ok" if drc_res.ok else "failed",
                "code": drc_res.code,
                "report_path": str(drc_out),
                "stdout": drc_res.stdout,
                "stderr": drc_res.stderr,
            }
            result["erc"] = {
                "status": "ok" if erc_res.ok else "failed",
                "code": erc_res.code,
                "report_path": str(erc_out),
                "stdout": erc_res.stdout,
                "stderr": erc_res.stderr,
            }

    # Write JSON and Markdown
    out_json = DOCS / "drc_erc_report.json"
    out_md = DOCS / "drc_erc_report.md"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md_lines = [
        "# DRC/ERC Report",
        f"Generated: {result['generated_at']}",
        f"- KiCad CLI: {result['kicad_cli_version']}",
        f"- PCB: `{result['pcb_file']}`",
        f"- SCH: `{result['sch_file']}`",
        "",
        "## DRC",
        f"- Status: {result['drc'].get('status')}",
        f"- Report: {result['drc'].get('report_path', '-')}",
        f"- Note: {result['drc'].get('reason', '-')}",
        "",
        "## ERC",
        f"- Status: {result['erc'].get('status')}",
        f"- Report: {result['erc'].get('report_path', '-')}",
        f"- Note: {result['erc'].get('reason', '-')}",
        "",
    ]
    if result["notes"]:
        md_lines.append("## Notes")
        md_lines += [f"- {n}" for n in result["notes"]]

    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(json.dumps({
        "json": str(out_json),
        "md": str(out_md),
        "drc_status": result["drc"].get("status"),
        "erc_status": result["erc"].get("status"),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
