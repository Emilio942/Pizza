#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vereinfachter Test für das strukturbasierte Pruning
"""

import torch
import sys
from pathlib import Path

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Importiere MicroPizzaNetV2
from src.pizza_detector import MicroPizzaNetV2

# Erstelle ein Modell
model = MicroPizzaNetV2(num_classes=4)
print(f"Original model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Berechne die Wichtigkeit der Filter basierend auf L1-Norm
importance_dict = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and module.groups == 1:
        weight = module.weight.data.clone()
        importance = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
        importance_dict[name] = importance
        print(f"Layer {name}: {importance.shape} importance values")

# Erstelle ein neues Modell mit 30% Pruning
sparsity = 0.3
pruned_model = MicroPizzaNetV2(num_classes=4)

# Identifiziere zu entfernende Filter
prune_targets = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and name in importance_dict:
        filter_importance = importance_dict[name]
        n_filters = len(filter_importance)
        n_prune = int(n_filters * sparsity)
        
        _, indices = torch.sort(filter_importance)
        prune_indices = indices[:n_prune].tolist()
        keep_indices = indices[n_prune:].tolist()
        
        prune_targets[name] = {
            'prune_indices': prune_indices,
            'keep_indices': keep_indices
        }
        print(f"Layer {name}: keeping {len(keep_indices)} filters, pruning {len(prune_indices)} filters")

# Teste Output der Modelle
dummy_input = torch.randn(1, 3, 48, 48)
with torch.no_grad():
    output_original = model(dummy_input)
    print(f"Original model output shape: {output_original.shape}")
    
    # Versuch, das geprunte Modell auszuführen (sollte fehlschlagen, da wir die Gewichte noch nicht angepasst haben)
    try:
        output_pruned = pruned_model(dummy_input)
        print(f"Pruned model output shape: {output_pruned.shape}")
    except Exception as e:
        print(f"Error running pruned model (expected at this point): {e}")
