#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the results of the input size evaluation.

This script generates plots comparing the accuracy and RAM usage
for different input image sizes.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Evaluation directory
EVAL_DIR = project_root / "output" / "evaluation"
OUTPUT_DIR = project_root / "output" / "visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_evaluation_data():
    """
    Load evaluation data from all size reports
    
    Returns:
        Dictionary with size as key and results as value
    """
    results = {}
    
    # Try to load the summary file first
    summary_path = EVAL_DIR / "input_size_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            if 'summary' in summary:
                return summary['summary']
    
    # If summary doesn't exist or doesn't have the data we need, 
    # load individual evaluation files
    for eval_file in EVAL_DIR.glob("eval_size_*.json"):
        with open(eval_file, 'r') as f:
            data = json.load(f)
            size = data.get("input_size")
            if size:
                size_key = f"{size}x{size}"
                results[size_key] = {
                    "accuracy": data.get("accuracy"),
                    "total_ram_kb": data.get("ram_usage", {}).get("total_ram_kb")
                }
    
    return results

def plot_results(data):
    """
    Generate plots for the evaluation results
    
    Args:
        data: Dictionary with evaluation results by size
    """
    # Extract sizes and metrics
    sizes = []
    accuracies = []
    ram_usages = []
    
    for size_str, metrics in data.items():
        size = int(size_str.split('x')[0])
        sizes.append(size)
        # Handle None values in accuracy
        accuracy = metrics.get("accuracy")
        accuracies.append(0 if accuracy is None else accuracy * 100)  # Convert to percentage
        ram_usages.append(metrics.get("total_ram_kb", 0))
    
    # Sort by size
    sorted_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    ram_usages = [ram_usages[i] for i in sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot accuracy vs input size
    ax1.plot(sizes, accuracies, 'bo-', linewidth=2)
    ax1.set_xlabel('Input Image Size (pixels)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy vs. Input Size', fontsize=14)
    ax1.grid(True)
    
    # Add exact values as labels
    for i, (size, acc) in enumerate(zip(sizes, accuracies)):
        ax1.annotate(f"{acc:.1f}%", 
                     (size, acc),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    
    # Plot RAM usage vs input size
    ax2.plot(sizes, ram_usages, 'ro-', linewidth=2)
    ax2.set_xlabel('Input Image Size (pixels)', fontsize=12)
    ax2.set_ylabel('RAM Usage (KB)', fontsize=12)
    ax2.set_title('Total RAM Usage vs. Input Size', fontsize=14)
    ax2.grid(True)
    
    # Add exact values as labels
    for i, (size, ram) in enumerate(zip(sizes, ram_usages)):
        ax2.annotate(f"{ram:.1f} KB", 
                     (size, ram),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    
    # Add a horizontal line at 204 KB to indicate the RAM limit
    ax2.axhline(y=204, color='r', linestyle='--', 
                label='RAM Limit (204 KB)')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = OUTPUT_DIR / "input_size_evaluation.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Create a summary figure that shows the trade-off
    plt.figure(figsize=(10, 8))
    
    # Scale RAM values to the same range as accuracy for visualization
    ram_scaled = [r / max(ram_usages) * max(accuracies) for r in ram_usages]
    
    # Plot both metrics on the same axes
    plt.plot(sizes, accuracies, 'bo-', linewidth=2, label='Accuracy (%)')
    plt.plot(sizes, ram_scaled, 'ro-', linewidth=2, label='Relative RAM Usage')
    
    # Create a second y-axis for RAM usage
    ax3 = plt.gca().twinx()
    ax3.plot(sizes, ram_usages, 'ro-', alpha=0)  # Invisible plot to set the scale
    ax3.set_ylabel('RAM Usage (KB)', color='r', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='r')
    
    # Add horizontal line at 204 KB RAM limit
    ax3.axhline(y=204, color='r', linestyle='--', alpha=0.5, 
                label='RAM Limit (204 KB)')
    
    # Set labels and title
    plt.xlabel('Input Image Size (pixels)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs. RAM Usage Trade-off', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    # Save the trade-off figure
    tradeoff_path = OUTPUT_DIR / "input_size_tradeoff.png"
    plt.savefig(tradeoff_path, dpi=300)
    print(f"Trade-off plot saved to {tradeoff_path}")

def main():
    """Main function"""
    print("Generating visualizations of input size evaluation results...")
    
    # Load data
    eval_data = load_evaluation_data()
    
    if not eval_data:
        print("No evaluation data found. Please run the evaluation first.")
        return
    
    # Generate plots
    plot_results(eval_data)
    print("Visualization complete.")

if __name__ == "__main__":
    main()
