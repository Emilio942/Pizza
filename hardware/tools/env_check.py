#!/usr/bin/env python3
"""
Environment diagnostics for hardware project.
- Reports KiCad version via `kicad-cli --version` (if available)
- Reports Python version and selected packages (requests, numpy, scipy, matplotlib)
- Optionally probes pcbnew (if KiCad's Python is available in PATH)

Writes JSON to hardware/docs/env_report.json
"""
from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def run_cmd(cmd: list[str]) -> dict:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "stdout": out.strip()}
    except FileNotFoundError:
        return {"ok": False, "error": "not found"}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"exit {e.returncode}", "stdout": e.output.strip()}


def probe_python_packages(pkgs: list[str]) -> dict:
    info = {}
    for p in pkgs:
        try:
            mod = __import__(p)
            ver = getattr(mod, "__version__", None)
            info[p] = {"installed": True, "version": ver}
        except Exception as e:
            info[p] = {"installed": False, "error": str(e)}
    return info


def main() -> int:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split("\n")[0],
            "platform": platform.platform(),
        },
        "kicad": {},
        "packages": {},
    }

    # KiCad via kicad-cli
    kicad_cli = shutil.which("kicad-cli")
    if kicad_cli:
        v = run_cmd([kicad_cli, "--version"])
        report["kicad"]["kicad_cli_path"] = kicad_cli
        report["kicad"]["version_check"] = v
    else:
        report["kicad"]["version_check"] = {"ok": False, "error": "kicad-cli not in PATH"}

    # Try pcbnew import (may require KiCad's Python)
    try:
        import importlib

        pcbnew = importlib.import_module("pcbnew")  # type: ignore
        report["kicad"]["pcbnew"] = {
            "importable": True,
            "version": getattr(pcbnew, "GetBuildVersion", lambda: None)(),
        }
    except Exception as e:
        report["kicad"]["pcbnew"] = {"importable": False, "error": str(e)}

    # Python packages
    report["packages"] = probe_python_packages([
        "requests",
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "gymnasium",
        "stable_baselines3",
    ])

    # Write out
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    docs = os.path.join(root, "docs")
    os.makedirs(docs, exist_ok=True)
    out_path = os.path.join(docs, "env_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "out_path": out_path,
        "kicad_cli": report["kicad"].get("version_check", {}),
        "pcbnew": report["kicad"].get("pcbnew", {}),
        "packages_summary": {k: v.get("version") if v.get("installed") else None for k, v in report["packages"].items()},
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
