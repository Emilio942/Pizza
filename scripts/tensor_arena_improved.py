#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verbesserte Tensor-Arena-Größen-Berechnung

Dieses Skript implementiert die verbesserte Berechnung der Tensor-Arena-Größe
für den RP2040-Emulator und korrigiert das EMU-02-Problem.

Das Skript kann als eigenständiges Tool verwendet oder in den Emulator
integriert werden.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from pathlib import Path
import logging
import sys

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.pizza_detector import load_model
except ImportError:
    print("Fehler beim Importieren der Module. Stellen Sie sicher, dass Sie im richtigen Verzeichnis sind.")
    sys.exit(1)

def calculate_tensor_arena_size(model, input_size=(3, 48, 48), quantized=True):
    """
    Berechnet eine genauere Schätzung der Tensor-Arena-Größe basierend auf der Modellarchitektur.
    
    Args:
        model: Das PyTorch-Modell
        input_size: Die Eingabegröße als (Kanäle, Höhe, Breite)
        quantized: Ob das Modell quantisiert ist (int8)
        
    Returns:
        int: Geschätzte Tensor-Arena-Größe in Bytes
    """
    # Bestimme Byte pro Wert basierend auf Quantisierung
    bytes_per_value = 1 if quantized else 4
    
    # Finde die maximale Anzahl von Feature-Maps in einem Layer
    max_feature_maps = 0
    for name, layer in model.named_modules():
        if hasattr(layer, 'out_features'):  # Linear Layer
            max_feature_maps = max(max_feature_maps, layer.out_features)
        elif hasattr(layer, 'out_channels'):  # Conv Layer
            max_feature_maps = max(max_feature_maps, layer.out_channels)
    
    # Schätze die Größe der größten Aktivierungsebene
    batch_size = 1  # Typischerweise 1 für Inferenz
    # Aktivierungen werden meist auf halber Auflösung des Eingabebildes gespeichert
    # (aufgrund der Pooling-Schichten)
    activation_size = batch_size * max_feature_maps * (input_size[1]//2) * (input_size[2]//2) * bytes_per_value
    
    # TFLite-Interpreter hat einen Overhead für Verwaltungsstrukturen
    overhead_factor = 1.2  # 20% Overhead
    tensor_arena_size = int(activation_size * overhead_factor)
    
    return tensor_arena_size

def simplified_tensor_arena_estimate(model_size_bytes, is_quantized, input_size=(3, 48, 48)):
    """
    Einfachere, aber immer noch verbesserte Schätzung der Tensor-Arena-Größe.
    
    Diese Funktion kann verwendet werden, wenn kein Zugriff auf das tatsächliche Modell
    möglich ist oder eine schnellere, approximative Schätzung benötigt wird.
    
    Args:
        model_size_bytes: Größe des Modells in Bytes
        is_quantized: Ob das Modell quantisiert ist
        input_size: Die Eingabegröße als (Kanäle, Höhe, Breite)
        
    Returns:
        int: Geschätzte Tensor-Arena-Größe in Bytes
    """
    # Schätze die maximale Anzahl von Feature-Maps basierend auf der Modellgröße
    # Typischerweise haben kleine Modelle (~2-5KB) ca. 8-16 Feature-Maps
    # Mittlere Modelle (~5-20KB) ca. 16-32 Feature-Maps
    # Große Modelle (>20KB) ca. 32-64 Feature-Maps
    
    if model_size_bytes < 5 * 1024:  # <5KB
        max_feature_maps = 16
    elif model_size_bytes < 20 * 1024:  # <20KB
        max_feature_maps = 32
    else:
        max_feature_maps = 64
    
    bytes_per_value = 1 if is_quantized else 4
    activation_size = max_feature_maps * (input_size[1]//2) * (input_size[2]//2) * bytes_per_value
    
    # Overhead für den TFLite-Interpreter
    overhead_factor = 1.2
    tensor_arena_size = int(activation_size * overhead_factor)
    
    return tensor_arena_size

def compare_methods(model_path, verbose=False):
    """
    Vergleicht die verschiedenen Methoden zur Tensor-Arena-Größen-Schätzung.
    
    Args:
        model_path: Pfad zum PyTorch-Modell
        verbose: Ob detaillierte Informationen ausgegeben werden sollen
        
    Returns:
        dict: Vergleich der verschiedenen Schätzmethoden
    """
    model_size_bytes = os.path.getsize(model_path)
    is_quantized = "int8" in model_path.lower()
    
    # Lade das Modell, falls verfügbar
    try:
        model = load_model(model_path)
        model_available = True
    except Exception as e:
        print(f"Warnung: Konnte Modell nicht laden: {e}")
        model = None
        model_available = False
    
    # Aktuelle EMU-02 Schätzung
    if is_quantized:
        emu02_estimate = int(model_size_bytes * 0.2)
    else:
        emu02_estimate = int(model_size_bytes * 0.5)
    
    # Verbesserte Schätzungen
    if model_available:
        architecture_estimate = calculate_tensor_arena_size(model, quantized=is_quantized)
    else:
        architecture_estimate = None
    
    simplified_estimate = simplified_tensor_arena_estimate(model_size_bytes, is_quantized)
    
    # Erstelle Vergleichsbericht
    comparison = {
        "model_info": {
            "path": model_path,
            "size_bytes": model_size_bytes,
            "size_kb": model_size_bytes / 1024,
            "is_quantized": is_quantized
        },
        "emu02_estimate": {
            "description": "Aktuelle EMU-02 Schätzung (20%/50% der Modellgröße)",
            "tensor_arena_bytes": emu02_estimate,
            "tensor_arena_kb": emu02_estimate / 1024
        },
        "simplified_estimate": {
            "description": "Vereinfachte verbesserte Schätzung (basierend auf Modellgröße)",
            "tensor_arena_bytes": simplified_estimate,
            "tensor_arena_kb": simplified_estimate / 1024,
            "increase_percentage": (simplified_estimate - emu02_estimate) / emu02_estimate * 100 if emu02_estimate > 0 else float('inf')
        }
    }
    
    if architecture_estimate is not None:
        comparison["architecture_estimate"] = {
            "description": "Architekturbasierte Schätzung (genaueste Methode)",
            "tensor_arena_bytes": architecture_estimate,
            "tensor_arena_kb": architecture_estimate / 1024,
            "increase_percentage": (architecture_estimate - emu02_estimate) / emu02_estimate * 100 if emu02_estimate > 0 else float('inf')
        }
    
    # Analyse und Empfehlung
    simplified_diff = abs((simplified_estimate - emu02_estimate) / emu02_estimate * 100) if emu02_estimate > 0 else float('inf')
    architecture_diff = abs((architecture_estimate - emu02_estimate) / emu02_estimate * 100) if architecture_estimate is not None and emu02_estimate > 0 else float('inf')
    
    threshold = 5  # 5% Schwellwert für Unterschiedstoleranz
    
    comparison["analysis"] = {
        "simplified_diff_percentage": simplified_diff,
        "architecture_diff_percentage": architecture_diff if architecture_estimate is not None else None,
        "exceeds_threshold": simplified_diff > threshold or architecture_diff > threshold,
        "recommended_method": "architecture" if architecture_estimate is not None else "simplified"
    }
    
    if comparison["analysis"]["exceeds_threshold"]:
        comparison["recommendation"] = {
            "message": "Die EMU-02 Schätzung sollte korrigiert werden. Die aktuelle Methode unterschätzt die Tensor-Arena-Größe erheblich.",
            "suggested_implementation": "Siehe tensor_arena_improved.py für die empfohlene Implementierung."
        }
    else:
        comparison["recommendation"] = {
            "message": "Die EMU-02 Schätzung ist akzeptabel. Die Abweichung liegt innerhalb der Toleranz von 5%."
        }
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Verbesserte Tensor-Arena-Größen-Berechnung")
    parser.add_argument("--model", type=str, default="models/pizza_model_int8.pth",
                        help="Pfad zum PyTorch-Modell")
    parser.add_argument("--output", type=str, default="output/ram_analysis/improved_tensor_arena.json",
                        help="Pfad zur Ausgabedatei")
    parser.add_argument("--verbose", action="store_true",
                        help="Gibt detaillierte Informationen aus")
    
    args = parser.parse_args()
    
    # Vergleiche die verschiedenen Methoden
    comparison = compare_methods(args.model, args.verbose)
    
    # Ausgabe
    print("\n" + "=" * 80)
    print("TENSOR-ARENA-VERBESSERUNGSANALYSE")
    print("=" * 80)
    print(f"Modell: {comparison['model_info']['path']}")
    print(f"Größe: {comparison['model_info']['size_kb']:.2f} KB")
    print(f"Quantisiert: {'Ja' if comparison['model_info']['is_quantized'] else 'Nein'}")
    print("\nSchätzungen:")
    print(f"  EMU-02 (aktuell): {comparison['emu02_estimate']['tensor_arena_kb']:.2f} KB")
    print(f"  Vereinfacht (verbessert): {comparison['simplified_estimate']['tensor_arena_kb']:.2f} KB (+{comparison['simplified_estimate']['increase_percentage']:.2f}%)")
    
    if "architecture_estimate" in comparison:
        print(f"  Architekturbasiert (genau): {comparison['architecture_estimate']['tensor_arena_kb']:.2f} KB (+{comparison['architecture_estimate']['increase_percentage']:.2f}%)")
    
    print("\nAnalyse:")
    if comparison["analysis"]["exceeds_threshold"]:
        print("  WARNUNG: Die EMU-02 Schätzung unterschätzt die Tensor-Arena-Größe erheblich!")
        print(f"  Empfohlene Methode: {comparison['analysis']['recommended_method']}")
        print(f"  Empfohlene Korrektur: {comparison['recommendation']['message']}")
    else:
        print("  Die EMU-02 Schätzung ist akzeptabel.")
    
    # Speichere Ergebnisse
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nErgebnisse wurden in {args.output} gespeichert.")

if __name__ == "__main__":
    main()
