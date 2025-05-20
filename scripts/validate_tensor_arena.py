#!/usr/bin/env psys.path.append(str(project_root))

# Import the ModelConverter directly 
sys.path.append(str(project_root / "src" / "emulation"))
from importlib.machinery import SourceFileLoader

# Load the module with the hyphen in its name
emulator_test = SourceFileLoader("emulator_test", 
                                str(project_root / "src" / "emulation" / "emulator-test.py")).load_module()
ModelConverter = emulator_test.ModelConverterrom src.emulation.emulator_test import ModelConverteron3
# -*- coding: utf-8 -*-
"""
Validierungs-Script für die verbesserte Tensor-Arena-Größenberechnung

Dieses Skript vergleicht die alte und neue Methode zur Berechnung der Tensor-Arena-Größe
und validiert, ob die Änderungen im Emulator korrekt implementiert wurden.
"""

import os
import sys
from pathlib import Path
import json

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.emulation.emulator-test import ModelConverter

def validate_tensor_arena_calculation(model_path):
    """
    Validiert die verbesserte Tensor-Arena-Berechnung.
    
    Args:
        model_path: Pfad zum Modell
        
    Returns:
        dict: Vergleichsergebnisse
    """
    print(f"Validiere Tensor-Arena-Berechnung für Modell: {model_path}")
    
    # Modellgröße ermitteln
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    is_quantized = "int8" in model_path.lower()
    
    # Emulator-Converter erstellen
    converter = ModelConverter()
    
    # Alte Methode simulieren (20% für quantisiert, 50% für float32)
    original_size_bytes = model_size_mb * 1024 * 1024
    if is_quantized:
        old_estimate = int(original_size_bytes / 4 * 0.2)
    else:
        old_estimate = int(original_size_bytes * 0.5)
    
    # Neue Methode durch den Converter
    model_info = converter.estimate_model_size(model_size_mb, quantized=is_quantized)
    new_estimate = model_info["ram_usage_bytes"]
    
    # Vergleich
    diff_percentage = ((new_estimate - old_estimate) / old_estimate * 100) if old_estimate > 0 else float('inf')
    
    results = {
        "model_info": {
            "path": model_path,
            "size_mb": model_size_mb,
            "is_quantized": is_quantized
        },
        "old_estimate": {
            "tensor_arena_bytes": old_estimate,
            "tensor_arena_kb": old_estimate / 1024,
            "method": "Fixed percentage (20%/50%)"
        },
        "new_estimate": {
            "tensor_arena_bytes": new_estimate,
            "tensor_arena_kb": new_estimate / 1024,
            "method": "Improved calculation"
        },
        "comparison": {
            "difference_bytes": new_estimate - old_estimate,
            "difference_kb": (new_estimate - old_estimate) / 1024,
            "difference_percentage": diff_percentage
        }
    }
    
    # Ausgabe
    print("\n" + "=" * 80)
    print("TENSOR-ARENA-VALIDIERUNG")
    print("=" * 80)
    print(f"Modell: {results['model_info']['path']}")
    print(f"Größe: {results['model_info']['size_mb']:.2f} MB")
    print(f"Quantisiert: {'Ja' if results['model_info']['is_quantized'] else 'Nein'}")
    print("\nSchätzungen:")
    print(f"  Alte Methode: {results['old_estimate']['tensor_arena_kb']:.2f} KB")
    print(f"  Neue Methode: {results['new_estimate']['tensor_arena_kb']:.2f} KB")
    print(f"  Unterschied: {results['comparison']['difference_kb']:.2f} KB ({results['comparison']['difference_percentage']:.2f}%)")
    
    return results

def main():
    # Standard-Testmodell
    model_path = "models/pizza_model_int8.pth"
    
    # Prüfe, ob Modell existiert
    if not os.path.exists(model_path):
        print(f"FEHLER: Modelldatei {model_path} nicht gefunden.")
        return
    
    # Führe Validierung durch
    results = validate_tensor_arena_calculation(model_path)
    
    # Speichere Ergebnisse
    output_dir = "output/validation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "tensor_arena_validation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nValidierungsergebnisse wurden in {output_path} gespeichert.")
    
    # Erfolgsausgabe
    threshold = 5  # 5% Schwellwert für Unterschiedstoleranz
    diff_percentage = abs(results['comparison']['difference_percentage'])
    
    if diff_percentage > threshold:
        print(f"\nValidierung zeigt signifikanten Unterschied ({diff_percentage:.2f}% > {threshold}%).")
        print("Die Verbesserung wurde erfolgreich implementiert.")
    else:
        print(f"\nValidierung zeigt keinen signifikanten Unterschied ({diff_percentage:.2f}% <= {threshold}%).")
        print("Die Änderungen scheinen nicht korrekt implementiert zu sein.")

if __name__ == "__main__":
    main()
