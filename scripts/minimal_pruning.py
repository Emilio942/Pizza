#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Structured Pruning Implementation

This script provides a minimal implementation of structured pruning for MicroPizzaNetV2 
without external dependencies.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import MicroPizzaNetV2
from src.pizza_detector import MicroPizzaNetV2

print("Creating MicroPizzaNetV2 model...")
model = MicroPizzaNetV2(num_classes=4)
print(f"Model created: {model.count_parameters():,} parameters")

# Calculate filter importance
print("\nCalculating filter importance...")
importance_dict = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and module.groups == 1:
        weight = module.weight.data.clone()
        importance = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
        importance_dict[name] = importance
        print(f"Layer {name}: {len(importance)} filters")

# Process different sparsity rates
sparsities = [0.1, 0.2, 0.3]
results = {}

for sparsity in sparsities:
    print(f"\nProcessing sparsity {sparsity:.2f}...")
    
    # Create pruned model
    pruned_model = MicroPizzaNetV2(num_classes=4)
    
    # Identify filters to keep
    prune_targets = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in importance_dict:
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
            print(f"Layer {name}: keeping {len(keep_indices)}/{n_filters} filters, pruning {len(prune_indices)} filters")
    
    # Apply pruning (just to block1.0 for simplicity)
    print("Applying pruning...")
    with torch.no_grad():
        if 'block1.0' in prune_targets:
            keep_indices = prune_targets['block1.0']['keep_indices']
            
            # Update weights
            pruned_model.block1[0].weight.data = model.block1[0].weight.data[keep_indices].clone()
            
            # Update BatchNorm
            pruned_model.block1[1].weight.data = model.block1[1].weight.data[keep_indices].clone()
            pruned_model.block1[1].bias.data = model.block1[1].bias.data[keep_indices].clone()
            pruned_model.block1[1].running_mean.data = model.block1[1].running_mean.data[keep_indices].clone()
            pruned_model.block1[1].running_var.data = model.block1[1].running_var.data[keep_indices].clone()
    
    # Test model before and after pruning
    dummy_input = torch.randn(1, 3, 48, 48)
    
    # Original model
    with torch.no_grad():
        try:
            orig_output = model(dummy_input)
            print(f"Original model output shape: {orig_output.shape}")
        except Exception as e:
            print(f"Error with original model: {e}")
    
    # Pruned model
    with torch.no_grad():
        try:
            pruned_output = pruned_model(dummy_input)
            print(f"Pruned model output shape: {pruned_output.shape}")
        except Exception as e:
            print(f"Error with pruned model: {e}")
    
    # Save pruned model
    os.makedirs("models_pruned", exist_ok=True)
    pruned_path = f"models_pruned/micropizzanetv2_pruned_s{int(sparsity*100)}.pth"
    torch.save(pruned_model.state_dict(), pruned_path)
    print(f"Saved pruned model to {pruned_path}")
    
    # Store results
    results[f"pruned_s{int(sparsity*100)}"] = {
        "sparsity": sparsity,
        "parameters_original": model.count_parameters(),
        "parameters_pruned": pruned_model.count_parameters(),
        "reduction_percent": 100 * (1 - pruned_model.count_parameters() / model.count_parameters()),
        "model_path": pruned_path
    }

# Print summary
print("\n===== PRUNING RESULTS =====")
print(f"Original model: {model.count_parameters():,} parameters")

for sparsity in sparsities:
    key = f"pruned_s{int(sparsity*100)}"
    result = results[key]
    print(f"Sparsity {sparsity:.2f}: {result['parameters_pruned']:,} parameters ({result['reduction_percent']:.2f}% reduction)")

# Save results
os.makedirs("output/model_optimization", exist_ok=True)
results_path = "output/model_optimization/minimal_pruning_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
