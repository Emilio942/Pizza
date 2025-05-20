#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sample verification data for the formal verification framework.

This script creates a verification dataset (.npz file) containing sample images
and their classes, which can be used for formal verification.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.constants import CLASS_NAMES, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate verification dataset for formal verification'
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing class subdirectories with images')
    
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the verification dataset (.npz)')
    
    parser.add_argument('--samples-per-class', type=int, default=3,
                        help='Number of samples to include per class')
    
    parser.add_argument('--include-preprocess', action='store_true',
                        help='Preprocess images (resize, normalize)')
    
    parser.add_argument('--critical-pairs', type=str, default='0,2',
                        help='Comma-separated list of critical class pairs (e.g., "0,2,1,3")')
    
    return parser.parse_args()

def load_images(data_dir, samples_per_class, preprocess=False):
    """
    Load sample images from each class for verification.
    
    Args:
        data_dir: Directory containing class subdirectories
        samples_per_class: Number of samples to include per class
        preprocess: Whether to preprocess the images
        
    Returns:
        Tuple of (images, classes)
    """
    images = []
    classes = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Find the class directory (might be lowercase or uppercase)
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            class_dir = os.path.join(data_dir, class_name.lower())
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found for {class_name}")
            continue
            
        # Get all image files in the directory
        image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
        image_files.extend(glob.glob(os.path.join(class_dir, '*.jpeg')))
        image_files.extend(glob.glob(os.path.join(class_dir, '*.png')))
        
        if not image_files:
            print(f"Warning: No images found for class {class_name}")
            continue
            
        # Limit the number of samples
        image_files = image_files[:samples_per_class]
        
        # Load and preprocess images
        for img_file in image_files:
            try:
                img = Image.open(img_file)
                
                if preprocess:
                    # Resize to model input size
                    img = img.resize((INPUT_SIZE, INPUT_SIZE))
                    
                    # Convert to numpy array
                    img_array = np.array(img) / 255.0
                    
                    # Normalize if needed
                    if IMAGE_MEAN is not None and IMAGE_STD is not None:
                        img_array = (img_array - IMAGE_MEAN) / IMAGE_STD
                else:
                    # Just convert to numpy array
                    img_array = np.array(img)
                
                images.append(img_array)
                classes.append(class_idx)
                print(f"Loaded {img_file} for class {class_idx} ({class_name})")
                
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    return np.array(images), np.array(classes)

def parse_critical_pairs(pairs_str):
    """Parse critical class pairs from string."""
    try:
        pairs_list = [int(x) for x in pairs_str.split(',')]
        
        # Check if we have an even number of items
        if len(pairs_list) % 2 != 0:
            print("Warning: Odd number of elements in critical pairs. Ignoring the last one.")
            pairs_list = pairs_list[:-1]
        
        # Create pairs
        pairs = [(pairs_list[i], pairs_list[i+1]) for i in range(0, len(pairs_list), 2)]
        return pairs
    except Exception as e:
        print(f"Error parsing critical pairs: {e}")
        return [(0, 2)]  # Default pair: raw vs well-done

def main():
    """Main function to generate verification data."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load images
    print(f"Loading images from {args.data_dir}")
    images, classes = load_images(
        args.data_dir,
        args.samples_per_class,
        args.include_preprocess
    )
    
    if len(images) == 0:
        print("Error: No images loaded")
        return 1
        
    print(f"Loaded {len(images)} images for verification")
    
    # Parse critical pairs
    critical_pairs = parse_critical_pairs(args.critical_pairs)
    print(f"Critical class pairs: {critical_pairs}")
    
    # Save verification data
    np.savez(
        args.output_file,
        images=images,
        classes=classes,
        critical_pairs=critical_pairs
    )
    
    print(f"Verification data saved to {args.output_file}")
    print(f"Images shape: {images.shape}")
    print(f"Classes shape: {classes.shape}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
