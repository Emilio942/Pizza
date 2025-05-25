#!/usr/bin/env python3
"""
Analyze class distribution for early exit model improvements
specifically targeting the data/augmented directory.
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging
import json
import torch
from torch import nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def count_images_in_dir(directory):
    """Count image files in a directory"""
    if not os.path.isdir(directory):
        return 0
        
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            count += 1
    return count

def analyze_dataset(data_dir):
    """Analyze class distribution in the dataset"""
    logger.info(f"Analyzing dataset in {data_dir}...")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d)) 
                 and not d.startswith('.')]
    
    # Count images in each class
    class_counts = {}
    total_images = 0
    
    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        count = count_images_in_dir(class_dir)
        
        class_counts[class_name] = count
        total_images += count
        
        logger.info(f"Class {class_name}: {count} images")
    
    logger.info(f"Total images: {total_images}")
    
    # Calculate class weights for balancing
    if total_images > 0:
        num_classes = len(class_dirs)
        class_weights = {}
        
        for class_name, count in class_counts.items():
            if count > 0:
                # Formula: N / (K * n_c) where N=total, K=num_classes, n_c=count in class c
                weight = total_images / (num_classes * count)
            else:
                weight = 0.0  # Handle empty classes
                
            class_weights[class_name] = weight
            logger.info(f"Class {class_name} weight: {weight:.4f}")
        
        # Convert class weights to tensor format for CrossEntropyLoss
        class_idx_weights = []
        class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_dirs))}
        
        for cls_name in sorted(class_dirs):
            idx = class_to_idx[cls_name]
            if cls_name in class_weights:
                class_idx_weights.append(class_weights[cls_name])
            else:
                class_idx_weights.append(1.0)  # Default weight
                
        logger.info(f"Class weights tensor: {class_idx_weights}")
    
    # Visualize the distribution
    visualize_class_distribution(class_counts, class_weights if total_images > 0 else {})
    
    return class_counts, class_weights if total_images > 0 else {}

def visualize_class_distribution(class_counts, class_weights):
    """Create visualization of class distribution and weights"""
    if not class_counts:
        logger.warning("No data to visualize")
        return
        
    # Sort by class name for consistent visualization
    sorted_classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in sorted_classes]
    weights = [class_weights.get(cls, 0) for cls in sorted_classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot image counts
    bars = ax1.bar(sorted_classes, counts)
    ax1.set_title('Number of Images per Class')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # Plot class weights
    bars = ax2.bar(sorted_classes, weights)
    ax2.set_title('Class Weights for Balanced Training')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Weight')
    
    # Add weight labels on top of bars
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{weight:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution_early_exit.png')
    logger.info("Class distribution visualization saved to class_distribution_early_exit.png")
    plt.close()
    
    # Save class weights to JSON for later use
    with open('class_weights.json', 'w') as f:
        json.dump({
            'class_counts': class_counts,
            'class_weights': class_weights,
            'class_weights_list': [class_weights.get(cls, 0) for cls in sorted_classes]
        }, f, indent=4)
    logger.info("Class weights saved to class_weights.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class distribution in pizza dataset")
    parser.add_argument("--data-dir", type=str, default="data/augmented",
                        help="Path to the dataset directory")
    
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir)
