#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEICHER-2.5: RAM-Bedarf Schätzung für Int4-Modell
--------------------------------------------------
Dieses Skript schätzt den RAM-Bedarf (insbesondere die Tensor Arena) für das
direkt Int4-quantisierte Modell basierend auf der Modellgröße und -struktur.
"""

import os
import sys
import json
import logging
from pathlib import Path
import time

import numpy as np

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tensor_arena_estimation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def estimate_tensor_arena_size(model_size_kb, model_type="int4", input_size=(48, 48)):
    """
    Schätzt den RAM-Bedarf für die Tensor Arena eines quantisierten Modells.
    
    Args:
        model_size_kb: Modellgröße in KB
        model_type: Art der Quantisierung ("int8" oder "int4")
        input_size: Eingabegröße des Modells (Höhe, Breite)
        
    Returns:
        Geschätzter RAM-Bedarf für die Tensor Arena in KB
    """
    # Parameter für die Tensor-Arena-Schätzung basierend auf Erfahrungswerten
    if model_type == "int4":
        # Für Int4-Modelle
        arena_multiplier = 3.2  # Faktor für die Arena-Größe
        overhead_kb = 12.8      # Fester Overhead
    else:
        # Für Int8-Modelle (Standard)
        arena_multiplier = 2.5
        overhead_kb = 10.0
        
    # Berücksichtige die Eingabegröße
    # Größere Eingaben benötigen mehr Arbeitsspeicher für Zwischenergebnisse
    input_size_factor = (input_size[0] * input_size[1]) / (48 * 48)  # Normalisiert auf 48x48
    
    # Berechne geschätzte Tensor-Arena-Größe
    tensor_arena_kb = model_size_kb * arena_multiplier * np.sqrt(input_size_factor)
    
    # Gesamter RAM-Bedarf ist Tensor Arena plus Overhead
    total_ram_kb = tensor_arena_kb + overhead_kb
    
    return total_ram_kb

def generate_ram_usage_report(int4_evaluation_path, output_path):
    """
    Erstellt einen Bericht über den geschätzten RAM-Bedarf des Int4-Modells.
    
    Args:
        int4_evaluation_path: Pfad zur Int4-Evaluierungsdatei
        output_path: Ausgabepfad für den RAM-Bedarfsbericht
    """
    # Lade Int4-Evaluierungsdaten
    with open(int4_evaluation_path, 'r') as f:
        int4_data = json.load(f)
    
    # Modellgröße
    model_size_kb = int4_data["int4_model"]["model_size_kb"]
    
    # Schätze Tensor-Arena und RAM-Bedarf
    tensor_arena_kb = estimate_tensor_arena_size(model_size_kb, "int4")
    
    # Schätze Flash-Bedarf (Modellgröße plus Overhead)
    flash_usage_kb = model_size_kb + 5.0  # 5KB Overhead für Code und Konstanten
    
    # Erstelle RAM-Bedarfsbericht
    ram_report = {
        "model_type": "int4_direct",
        "model_path": int4_data["int4_model"]["model_path"],
        "model_size_kb": model_size_kb,
        "tensor_arena_estimate_kb": tensor_arena_kb,
        "flash_usage_estimate_kb": flash_usage_kb,
        "total_ram_estimate_kb": tensor_arena_kb + 10.0,  # Plus 10KB für Stack, Heap, etc.
        "fits_rp2040_constraint": str((tensor_arena_kb + 10.0) < 204.0),  # RP2040 RAM constraint - convert bool to string
        "notes": (
            "Diese Schätzung basiert auf Erfahrungswerten für Int4-quantisierte Modelle. "
            "Die tatsächliche RAM-Nutzung kann je nach Modellarchitektur und "
            "Implementierungsdetails variieren."
        )
    }
    
    # Speichere RAM-Bedarfsbericht
    with open(output_path, 'w') as f:
        json.dump(ram_report, f, indent=2)
    
    # Erstelle auch eine Markdown-Version
    markdown_path = output_path.replace('.json', '.md')
    with open(markdown_path, 'w') as f:
        f.write(f"""# RAM-Bedarfsschätzung für Int4-Modell

## Modell-Information
- **Modelltyp:** Direkte Int4-Quantisierung
- **Modellpfad:** `{ram_report["model_path"]}`
- **Modellgröße:** {ram_report["model_size_kb"]:.2f} KB

## RAM-Bedarfsschätzung
- **Geschätzte Tensor-Arena-Größe:** {ram_report["tensor_arena_estimate_kb"]:.2f} KB
- **Geschätzter Flash-Bedarf:** {ram_report["flash_usage_estimate_kb"]:.2f} KB
- **Geschätzter Gesamt-RAM-Bedarf:** {ram_report["total_ram_estimate_kb"]:.2f} KB

## RP2040-Kompatibilität
- **Passt in RP2040-RAM-Beschränkung (204 KB):** {'Ja' if ram_report["total_ram_estimate_kb"] < 204.0 else 'Nein'}

## Hinweise
{ram_report["notes"]}
""")
    
    return ram_report

def main():
    """Hauptfunktion"""
    int4_eval_path = "output/evaluation/int4_model_evaluation.json"
    output_path = "output/evaluation/int4_ram_estimate.json"
    
    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generiere RAM-Bedarfsbericht
    ram_report = generate_ram_usage_report(int4_eval_path, output_path)
    
    # Zeige Zusammenfassung
    print("\n====== Int4-Modell RAM-Bedarfsschätzung ======")
    print(f"Modellgröße: {ram_report['model_size_kb']:.2f} KB")
    print(f"Geschätzte Tensor-Arena: {ram_report['tensor_arena_estimate_kb']:.2f} KB")
    print(f"Geschätzter Gesamt-RAM: {ram_report['total_ram_estimate_kb']:.2f} KB")
    print(f"Passt in RP2040 (< 204 KB): {'Ja' if ram_report['fits_rp2040_constraint'] else 'Nein'}")
    print("==============================================\n")
    
    logger.info(f"RAM-Bedarfsschätzung abgeschlossen und gespeichert unter: {output_path}")

if __name__ == "__main__":
    main()
