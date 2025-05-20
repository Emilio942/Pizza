#!/usr/bin/env python3
"""
Analyze Class Distribution - Pizza Dataset

This script analyzes the distribution of pizza images across different classes
in the training dataset and outputs the results to a JSON file.
"""

import os
import json
from pathlib import Path
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def count_images_in_directory(dir_path, extension='jpg'):
    """Count the number of images with specified extension in a directory (including subdirectories)."""
    # Count files directly in this directory
    pattern = os.path.join(dir_path, f"*.{extension}")
    direct_count = len(glob.glob(pattern))
    
    # Count files in subdirectories
    subdirs_pattern = os.path.join(dir_path, "**", f"*.{extension}")
    subdir_count = len(glob.glob(subdirs_pattern, recursive=True))
    
    return direct_count + subdir_count

def analyze_classes():
    """Analyze the distribution of images across classes in the training dataset."""
    # Define paths
    project_root = Path(__file__).parent
    output_dir = project_root / "output" / "data_analysis"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class definitions from JSON
    class_file = project_root / "data" / "class_definitions.json"
    with open(class_file, 'r') as f:
        class_definitions = json.load(f)
    
    # Define all datasets to analyze
    datasets = {
        "augmented_pizza": project_root / "augmented_pizza",
        "augmented_pizza_legacy": project_root / "augmented_pizza_legacy"
    }
    
    # Count images in each class directory
    results = {}
    
    # First dataset: augmented_pizza
    dataset_path = datasets["augmented_pizza"]
    class_counts = {}
    
    for class_name in class_definitions.keys():
        class_dir = dataset_path / class_name
        if class_dir.exists():
            # Count jpg and JPG files
            jpg_count = count_images_in_directory(class_dir, "jpg")
            jpg_upper_count = count_images_in_directory(class_dir, "JPG")
            # Count png and PNG files
            png_count = count_images_in_directory(class_dir, "png")
            png_upper_count = count_images_in_directory(class_dir, "PNG")
            
            class_counts[class_name] = jpg_count + jpg_upper_count + png_count + png_upper_count
    
    results["augmented_pizza"] = class_counts
    
    # Second dataset: augmented_pizza_legacy
    # This folder doesn't have subdirectories by class, but has filenames in format "pizza_X_class_Y_Z.jpg"
    dataset_path = datasets["augmented_pizza_legacy"]
    
    if dataset_path.exists():
        # Parse filenames to extract class information
        legacy_class_counts = Counter()
        
        for file_name in os.listdir(dataset_path):
            if file_name.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                parts = file_name.split('_')
                if len(parts) >= 3:
                    class_name = parts[2]  # e.g., "basic" or "burnt" from "pizza_0_basic_0_0.jpg"
                    legacy_class_counts[class_name] += 1
        
        results["augmented_pizza_legacy"] = dict(legacy_class_counts)
    
    # Combine results and calculate total
    total_counts = Counter()
    for dataset_counts in results.values():
        for class_name, count in dataset_counts.items():
            total_counts[class_name] += count
    
    results["total"] = dict(total_counts)
    
    # Calculate percentages
    total_images = sum(total_counts.values())
    percentages = {}
    for class_name, count in total_counts.items():
        percentages[class_name] = round((count / total_images) * 100, 2) if total_images > 0 else 0
    
    results["percentages"] = percentages
    
    # Write results to JSON file
    output_file = output_dir / "class_distribution_train.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    
    # Create visualization
    create_distribution_chart(results, output_dir)
    
    return results

def create_distribution_chart(results, output_dir):
    """Create a bar chart visualization of the class distribution."""
    total_counts = results["total"]
    class_names = list(total_counts.keys())
    counts = list(total_counts.values())
    
    plt.figure(figsize=(12, 8))
    
    # Create bars
    bars = plt.bar(class_names, counts)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    # Add percentage labels inside bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = results["percentages"][class_names[i]]
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{percentage}%', ha='center', va='center', 
                 color='white', fontweight='bold')
    
    plt.title('Pizza Dataset - Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_dir / "class_distribution_train.png")
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Pizza Dataset - Class Distribution (%)')
    plt.tight_layout()
    
    # Save pie chart
    plt.savefig(output_dir / "class_distribution_pie.png")

if __name__ == "__main__":
    analyze_classes()
