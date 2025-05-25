#!/usr/bin/env python3
"""
Model Adapter for Early Exit Models

This script provides utilities to handle compatibility between different 
versions of early exit models (original and improved versions).
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.early_exit.improved_early_exit import ImprovedMicroPizzaNetWithEarlyExit
from scripts.early_exit.micropizzanet_early_exit import MicroPizzaNetWithEarlyExit

def load_model_with_compatibility(model_path, num_classes=6, device="cpu"):
    """
    Load a model with compatibility handling between original and improved versions.
    
    This function attempts to load the model directly first. If that fails due to
    architecture mismatch, it tries to adapt the model architecture accordingly.
    
    Args:
        model_path: Path to the model weights file
        num_classes: Number of output classes
        device: Device to load the model on ("cpu" or "cuda")
        
    Returns:
        The loaded model
    """
    # First try: Load the improved model directly
    try:
        model = ImprovedMicroPizzaNetWithEarlyExit(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded improved early exit model")
        return model
    except Exception as e:
        print(f"Could not load as improved model: {e}")
    
    # Second try: Load as original model
    try:
        model = MicroPizzaNetWithEarlyExit(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded original early exit model")
        return model
    except Exception as e:
        print(f"Could not load as original model either: {e}")
    
    # If both fail, try to adapt the model structure
    print("Attempting to adapt model architecture...")
    if "improved" in model_path:
        # This is an improved model but we need to use original architecture
        return adapt_improved_to_original(model_path, num_classes, device)
    else:
        # This is an original model but we need to use improved architecture
        return adapt_original_to_improved(model_path, num_classes, device)

def adapt_improved_to_original(model_path, num_classes=6, device="cpu"):
    """
    Adapt an improved model to work with the original architecture
    
    This creates a new model with the original architecture and copies
    the compatible weights from the improved model.
    
    Args:
        model_path: Path to the improved model weights
        num_classes: Number of output classes
        device: Device to load the model on
        
    Returns:
        Adapted model with original architecture
    """
    # Load the state dict of the improved model
    improved_state_dict = torch.load(model_path, map_location=device)
    
    # Create a new model with the original architecture
    original_model = MicroPizzaNetWithEarlyExit(num_classes=num_classes)
    
    # Copy compatible weights
    original_state_dict = original_model.state_dict()
    
    # For each parameter in the original model, try to find a match in the improved model
    for name, param in original_state_dict.items():
        if name in improved_state_dict and improved_state_dict[name].size() == param.size():
            original_state_dict[name] = improved_state_dict[name]
    
    # Special handling for the early exit classifier, which has different sizes
    # We can only copy the output layer, leaving the additional hidden layer behind
    if 'early_exit_classifier.0.weight' in improved_state_dict and 'early_exit_classifier.0.weight' in original_state_dict:
        original_state_dict['early_exit_classifier.0.weight'] = improved_state_dict['early_exit_classifier.0.weight']
    
    if 'early_exit_classifier.0.bias' in improved_state_dict and 'early_exit_classifier.0.bias' in original_state_dict:
        original_state_dict['early_exit_classifier.0.bias'] = improved_state_dict['early_exit_classifier.0.bias']
    
    # Load the modified state dict
    original_model.load_state_dict(original_state_dict)
    print("Adapted improved model to original architecture")
    
    return original_model

def adapt_original_to_improved(model_path, num_classes=6, device="cpu", dropout_rate=0.3, confidence_threshold=0.5):
    """
    Adapt an original model to work with the improved architecture
    
    This creates a new model with the improved architecture and copies
    the compatible weights from the original model.
    
    Args:
        model_path: Path to the original model weights
        num_classes: Number of output classes
        device: Device to load the model on
        dropout_rate: Dropout rate for the improved model
        confidence_threshold: Confidence threshold for early exit
        
    Returns:
        Adapted model with improved architecture
    """
    # Load the state dict of the original model
    original_state_dict = torch.load(model_path, map_location=device)
    
    # Create a new model with the improved architecture
    improved_model = ImprovedMicroPizzaNetWithEarlyExit(
        num_classes=num_classes, 
        dropout_rate=dropout_rate, 
        confidence_threshold=confidence_threshold
    )
    
    # Copy compatible weights
    improved_state_dict = improved_model.state_dict()
    
    # For each parameter in the improved model, try to find a match in the original model
    for name, param in improved_state_dict.items():
        # Skip the new hidden layer in the early exit classifier
        if 'early_exit_classifier.2' in name or 'early_exit_classifier.4' in name:
            continue
            
        if name in original_state_dict and original_state_dict[name].size() == param.size():
            improved_state_dict[name] = original_state_dict[name]
    
    # Load the modified state dict
    improved_model.load_state_dict(improved_state_dict, strict=False)
    print("Adapted original model to improved architecture")
    
    return improved_model

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapt early exit models between different architectures")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the adapted model")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of output classes")
    parser.add_argument("--target", type=str, choices=["original", "improved"], 
                        help="Target architecture (if not specified, will be auto-detected)")
    
    args = parser.parse_args()
    
    # Auto-detect target architecture if not specified
    if args.target is None:
        if "improved" in args.model_path:
            args.target = "original"
        else:
            args.target = "improved"
    
    # Load and adapt the model
    if args.target == "original":
        model = adapt_improved_to_original(args.model_path, args.num_classes)
    else:
        model = adapt_original_to_improved(args.model_path, args.num_classes)
    
    # Save the adapted model
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved adapted model to {args.output_path}")
