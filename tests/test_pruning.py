#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vereinfachtes Skript zum Testen des strukturbasierten Prunings

Dieses Skript implementiert das Pruning direkt, ohne die 
komplexe Evaluierungsinfrastruktur.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Importiere benötigte Module
from src.pizza_detector import MicroPizzaNetV2
from scripts.pruning_tool import get_filter_importance, create_pruned_model, quantize_model

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pruning_test(sparsity=0.3):
    """
    Führt einen einfachen Pruning-Test mit der angegebenen Sparsity durch
    
    Args:
        sparsity (float): Sparsity-Rate (0.0-1.0)
    """
    logger.info(f"Starte Pruning-Test mit Sparsity {sparsity:.2f}")
    
    # Erstelle Modell
    model = MicroPizzaNetV2(num_classes=4)
    logger.info(f"Modell erstellt. Parameter: {model.count_parameters():,}")
    
    # Berechne Filter-Wichtigkeit
    importance_dict = get_filter_importance(model)
    
    # Erstelle gepruntes Modell
    pruned_model = create_pruned_model(model, importance_dict, sparsity)
    logger.info(f"Gepruntes Modell erstellt. Parameter: {pruned_model.count_parameters():,}")
    
    # Teste das geprunte Modell mit einem Dummy-Input
    dummy_input = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        output = pruned_model(dummy_input)
    
    logger.info(f"Gepruntes Modell erfolgreich ausgeführt. Ausgabegröße: {output.shape}")
    
    # Speichere das geprunte Modell
    os.makedirs("models_pruned", exist_ok=True)
    model_path = f"models_pruned/micropizzanetv2_pruned_s{int(sparsity*100)}.pth"
    torch.save(pruned_model.state_dict(), model_path)
    logger.info(f"Gepruntes Modell gespeichert: {model_path}")
    
    return {
        'original_model': model,
        'pruned_model': pruned_model,
        'model_path': model_path,
        'sparsity': sparsity
    }

if __name__ == "__main__":
    # Führe Tests mit verschiedenen Sparsity-Raten durch
    sparsities = [0.1, 0.2, 0.3]
    results = {}
    
    for sparsity in sparsities:
        results[sparsity] = run_pruning_test(sparsity)
        
    # Ausgabe der Ergebnisse
    logger.info("=== Pruning-Tests abgeschlossen ===")
    for sparsity, result in results.items():
        model = result['original_model']
        pruned_model = result['pruned_model']
        reduction = 100 * (1 - pruned_model.count_parameters() / model.count_parameters())
        logger.info(f"Sparsity {sparsity:.2f}: Parameter reduziert von {model.count_parameters():,} auf {pruned_model.count_parameters():,} ({reduction:.2f}%)")
