#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die verbesserte Tensor-Arena-Größenberechnung mit verschiedenen Modellgrößen

Dieses Skript testet die verbesserte Berechnung mit verschiedenen Modellgrößen,
um die Skalierung zu validieren.
"""

import sys
import os
from pathlib import Path

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Führe die verbesserte Berechnung direkt aus
def test_tensor_arena_scaling():
    print("\n=== TENSOR ARENA SCALING TEST ===")
    
    # Test-Modellgrößen in Bytes
    model_sizes = [
        2 * 1024,      # 2KB (Sehr kleines Modell)
        5 * 1024,      # 5KB (Kleines Modell)
        10 * 1024,     # 10KB (Mittleres Modell)
        30 * 1024,     # 30KB (Großes Modell)
        100 * 1024     # 100KB (Sehr großes Modell)
    ]
    
    # Definiere die input_size
    input_size = (3, 96, 96)  # (Kanäle, Höhe, Breite)
    
    # Teste für quantisiert und nicht-quantisiert
    for is_quantized in [True, False]:
        print(f"\nModell Typ: {'Quantisiert (INT8)' if is_quantized else 'Nicht-quantisiert (FLOAT32)'}")
        print("-" * 60)
        print(f"{'Modellgröße':<15} | {'Alte Methode':<15} | {'Neue Methode':<15} | {'Unterschied':<15}")
        print("-" * 60)
        
        for size_bytes in model_sizes:
            # Alte Methode (feste Prozentsätze)
            if is_quantized:
                old_estimate = int(size_bytes * 0.2)
            else:
                old_estimate = int(size_bytes * 0.5)
            
            # Neue Methode (verbessert)
            if size_bytes < 5 * 1024:  # <5KB
                max_feature_maps = 16
            elif size_bytes < 20 * 1024:  # <20KB
                max_feature_maps = 32
            else:
                max_feature_maps = 64
            
            bytes_per_value = 1 if is_quantized else 4
            activation_size = max_feature_maps * (input_size[1]//2) * (input_size[2]//2) * bytes_per_value
            overhead_factor = 1.2
            new_estimate = int(activation_size * overhead_factor)
            
            # Unterschied berechnen
            diff_percentage = ((new_estimate - old_estimate) / old_estimate * 100) if old_estimate > 0 else float('inf')
            
            # Ausgabe
            print(f"{size_bytes/1024:.1f}KB{' ':>10} | "
                  f"{old_estimate/1024:.1f}KB{' ':>9} | "
                  f"{new_estimate/1024:.1f}KB{' ':>9} | "
                  f"{diff_percentage:.1f}%{' ':>8}")
    
    print("\n=== TEST ABGESCHLOSSEN ===")

if __name__ == "__main__":
    test_tensor_arena_scaling()
