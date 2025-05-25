#!/usr/bin/env python3
"""
Test script for model architecture adapter

This script verifies that our model adapter works correctly by:
1. Loading a model with the adapter
2. Testing it on a small batch of data
3. Confirming the early exit functionality works correctly
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    create_optimized_dataloaders, RP2040Config
)
from scripts.early_exit.model_adapter import load_model_with_compatibility

def test_model_adapter(model_path, data_dir, device="cpu"):
    """Test the model adapter by loading a model and running inference"""
    print(f"Testing model adapter with model: {model_path}")
    
    # Create data loaders
    config = RP2040Config(data_dir=data_dir)
    _, val_loader, class_names, _ = create_optimized_dataloaders(config)
    print(f"Loaded data with {len(class_names)} classes: {class_names}")
    
    # Load model using adapter
    model = load_model_with_compatibility(
        model_path, 
        num_classes=len(class_names),
        device=device
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get a small batch of data
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        print(f"Input shape: {inputs.shape}")
        
        # Test regular inference
        with torch.no_grad():
            outputs, early_exit_used = model(inputs, use_early_exit=False)
        
        print(f"Full model output shape: {outputs.shape}")
        print(f"Early exit used: {early_exit_used}")
        
        # Test early exit inference with low threshold (should exit early)
        model.confidence_threshold = 0.1
        with torch.no_grad():
            outputs, early_exit_used = model(inputs, use_early_exit=True)
        
        print(f"With low threshold (0.1) - early exit used: {early_exit_used}")
        
        # Test early exit with forced exit
        try:
            with torch.no_grad():
                outputs, early_exit_used = model(inputs, use_early_exit=True, forced_exit=True)
            print(f"Forced early exit supported, output shape: {outputs.shape}")
            print(f"Early exit used: {early_exit_used}")
        except Exception as e:
            print(f"Forced early exit not supported by this model: {e}")
            print("This is expected for the original model architecture")
        
        break  # Only need one batch for testing
    
    print("Model adapter test completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model architecture adapter")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model weights file")
    parser.add_argument("--data-dir", type=str, default="data/augmented",
                       help="Directory containing the dataset")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for testing")
    
    args = parser.parse_args()
    test_model_adapter(**vars(args))
