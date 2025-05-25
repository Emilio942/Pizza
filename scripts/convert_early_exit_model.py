#!/usr/bin/env python3
"""
Model Converter Script

This script converts saved models between the original and improved early exit 
architectures, fixing compatibility issues.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.early_exit.model_adapter import (
    adapt_improved_to_original, adapt_original_to_improved
)

def main(args):
    """Main converter function"""
    print(f"Converting model from {args.input_path} to {args.output_path}")
    print(f"Target architecture: {args.target_arch}")
    
    if args.target_arch == "original":
        model = adapt_improved_to_original(
            args.input_path, 
            num_classes=args.num_classes,
            device=args.device
        )
    else:
        model = adapt_original_to_improved(
            args.input_path, 
            num_classes=args.num_classes,
            device=args.device,
            dropout_rate=args.dropout_rate,
            confidence_threshold=args.confidence_threshold
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save the converted model
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved converted model to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert early exit models between architectures")
    
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to the input model file (.pth)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the converted model file (.pth)")
    parser.add_argument("--target-arch", type=str, choices=["original", "improved"], required=True,
                        help="Target architecture to convert to")
    parser.add_argument("--num-classes", type=int, default=6,
                        help="Number of output classes")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for conversion (cuda or cpu)")
    parser.add_argument("--dropout-rate", type=float, default=0.3,
                        help="Dropout rate (for improved model)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold (for improved model)")
    
    args = parser.parse_args()
    main(args)
