#!/usr/bin/env python3
"""
Hilfsskript zur Bereinigung temporärer Dateien und Wartung des Projekts.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_temp_files():
    """Bereinigt temporäre Dateien und Caches"""
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/.DS_Store",
        "**/*.swp",
        "**/*.swo",
        "**/.*.sw*",
        "**/Thumbs.db"
    ]
    
    count = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                count += 1
                print(f"Gelöscht: {path}")
            except Exception as e:
                print(f"Fehler beim Löschen von {path}: {e}")
    
    print(f"\nBereinigung abgeschlossen: {count} Dateien/Verzeichnisse entfernt")

def cleanup_temp_outputs():
    """Bereinigt temporäre Ausgabeverzeichnisse"""
    output_dirs = [
        "output/temp",
        "output/logs"
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
                print(f"Bereinigt und neu erstellt: {dir_path}")
            except Exception as e:
                print(f"Fehler beim Bereinigen von {dir_path}: {e}")

def main():
    print("=== Projekt-Bereinigung ===")
    cleanup_temp_files()
    cleanup_temp_outputs()
    print("\nBereinigung abgeschlossen!")

if __name__ == "__main__":
    main()