#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to test the pruning tool and evaluate pruned models
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import modules
from src.pizza_detector import MicroPizzaNetV2
from scripts.pruning_tool import create_pruned_model, quantize_model, save_pruned_model

def test_pruning(sparsity=0.2):
    """
    Test pruning functionality and verify the output model
    """
    print(f"Testing pruning with sparsity {sparsity}")
    
    # Create model
    model = MicroPizzaNetV2(num_classes=4)
    print(f"Original parameters: {model.count_parameters():,}")
    
    # Prune model
    pruned_model = create_pruned_model(model, sparsity)
    print(f"Pruned parameters: {pruned_model.count_parameters():,}")
    print(f"Reduction: {100 * (1 - pruned_model.count_parameters() / model.count_parameters()):.2f}%")
    
    # Test model with dummy input
    try:
        dummy_input = torch.randn(1, 3, 48, 48)
        with torch.no_grad():
            output = pruned_model(dummy_input)
        
        print(f"Model works! Output shape: {output.shape}")
        
        # Try to quantize
        print("Testing quantization...")
        quantized_model = quantize_model(pruned_model)
        
        # Test quantized model
        with torch.no_grad():
            quant_output = quantized_model(dummy_input)
        
        print(f"Quantized model works! Output shape: {quant_output.shape}")
        
        # Save models for testing
        save_pruned_model(pruned_model, sparsity, "test_models", False)
        save_pruned_model(quantized_model, sparsity, "test_models", True)
        
        print("Models saved to test_models directory")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pruning tool")
    parser.add_argument("--sparsity", type=float, default=0.2, help="Sparsity rate (default: 0.2)")
    
    args = parser.parse_args()
    test_pruning(args.sparsity)
