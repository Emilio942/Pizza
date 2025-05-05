#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Wrapper für die automatisierte Pizza-Erkennungs-Test-Suite

Ein einfaches Skript zum Ausführen der Pizza-Erkennungs-Tests mit
automatischer Berichterstellung für die kontinuierliche Integration.

Verwendung:
    python scripts/run_pizza_tests.py [--detailed]

Optionen:
    --detailed: Erstellt einen detaillierten HTML-Bericht
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_tests(detailed_report=False):
    """
    Führt die automatisierte Test-Suite aus
    
    Args:
        detailed_report: Ob ein detaillierter HTML-Bericht erstellt werden soll
    
    Returns:
        bool: True, wenn alle Tests bestanden wurden, sonst False
    """
    print("===== Pizza-Erkennungssystem: Automatisierte Test-Suite =====")
    
    # Führe zuerst die PyTest-Klassifikationstests aus
    print("\n[1/2] Führe Unit-Tests für die Klassifikation aus...")
    pytest_cmd = [sys.executable, "-m", "pytest", "-xvs", "tests/test_pizza_classification.py"]
    pytest_result = subprocess.run(pytest_cmd, capture_output=True, text=True)
    
    print("\nTestergebnisse:")
    print(pytest_result.stdout)
    
    if pytest_result.returncode != 0:
        print("\nEinige Tests sind fehlgeschlagen!")
        if pytest_result.stderr:
            print("Fehler:")
            print(pytest_result.stderr)
        return False
    
    # Führe dann die detaillierte Test-Suite mit der Berichterstellung aus
    print("\n[2/2] Führe umfassende Modelltests aus...")
    
    ts_args = ["--generate-images"]
    if detailed_report:
        ts_args.append("--detailed-report")
    
    ts_cmd = [sys.executable, "scripts/automated_test_suite.py"] + ts_args
    ts_result = subprocess.run(ts_cmd)
    
    if ts_result.returncode != 0:
        print("\nDie Modelltests sind fehlgeschlagen!")
        return False
    
    print("\nAlle Tests wurden erfolgreich bestanden!")
    return True


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Test-Wrapper für die Pizza-Erkennungs-Test-Suite")
    parser.add_argument("--detailed", action="store_true", 
                        help="Erstellt einen detaillierten HTML-Bericht")
    
    args = parser.parse_args()
    success = run_tests(detailed_report=args.detailed)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()