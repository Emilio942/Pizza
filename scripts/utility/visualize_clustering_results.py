#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of Weight Clustering Results
------------------------------------------
This script visualizes the results of weight clustering with different
cluster counts (16, 32, 64) and INT4 quantization.

The script generates plots for:
1. Model size comparison
2. Memory usage comparison
3. Accuracy comparison
4. Inference time comparison
5. Compression ratio vs. accuracy tradeoff

Usage:
    python scripts/visualize_clustering_results.py --results_json output/clustering_evaluation/all_results.json --output_dir output/clustering_evaluation/plots
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_bar_chart(data, labels, title, ylabel, output_path, baseline_label="baseline"):
    """Creates a bar chart comparing different model variants."""
    plt.figure(figsize=(10, 6))
    
    # Set colors
    colors = ['gray']  # Baseline in gray
    variant_colors = ['lightblue', 'blue', 'darkblue',  # Clustered
                      'lightgreen', 'green', 'darkgreen']  # INT4 + Clustered
    colors.extend(variant_colors)
    
    # Get baseline value for normalization
    baseline_value = data[labels.index(baseline_label)]
    
    # Create bars
    bars = plt.bar(labels, data, color=colors)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        value = data[i]
        if labels[i] != baseline_label:
            # Calculate change percentage from baseline
            change_pct = (value - baseline_value) / baseline_value * 100
            sign = "+" if change_pct > 0 else ""
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(data) * 0.02,
                    f"{value:.2f}\n({sign}{change_pct:.1f}%)",
                    ha='center', va='bottom', fontsize=8)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(data) * 0.02,
                    f"{value:.2f}",
                    ha='center', va='bottom', fontsize=8)
    
    # Add baseline horizontal line
    plt.axhline(y=baseline_value, color='r', linestyle='--', alpha=0.5)
    
    # Set chart title and labels
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_scatter_plot(data, output_path):
    """Creates a scatter plot showing the tradeoff between compression and accuracy."""
    plt.figure(figsize=(10, 6))
    
    model_variants = list(data.keys())
    colors = {'clustered_16': 'lightblue', 'clustered_32': 'blue', 'clustered_64': 'darkblue',
              'int4_baseline': 'orange',
              'int4_clustered_16': 'lightgreen', 'int4_clustered_32': 'green', 'int4_clustered_64': 'darkgreen'}
    
    baseline_size = data['baseline']['model_size_kb']
    baseline_accuracy = data['baseline']['accuracy'] * 100
    
    # Add reference point for baseline
    plt.scatter(0, baseline_accuracy, color='red', s=100, label='baseline', marker='*')
    
    # Add other points
    for variant, metrics in data.items():
        if variant == 'baseline' or 'clustering_stats' in variant:
            continue
            
        size_reduction = (1 - metrics['model_size_kb'] / baseline_size) * 100
        accuracy = metrics['accuracy'] * 100
        
        # Adjust label for display
        display_label = variant.replace('_', ' ')
        
        plt.scatter(size_reduction, accuracy, color=colors.get(variant, 'gray'), s=80, label=display_label)
    
    # Add annotations
    for variant, metrics in data.items():
        if variant == 'baseline' or 'clustering_stats' in variant:
            continue
            
        size_reduction = (1 - metrics['model_size_kb'] / baseline_size) * 100
        accuracy = metrics['accuracy'] * 100
        
        plt.annotate(variant, 
                    (size_reduction, accuracy),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Set chart properties
    plt.title('Compression vs. Accuracy Trade-off')
    plt.xlabel('Model Size Reduction (%)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    with open(args.results_json, 'r') as f:
        all_results = json.load(f)
    
    # Extract model variants and metrics
    variants = []
    model_sizes = []
    memory_usages = []
    accuracies = []
    inference_times = []
    
    # First add baseline
    variants.append('baseline')
    model_sizes.append(all_results['baseline']['model_size_kb'])
    memory_usages.append(all_results['baseline']['tensor_arena_kb'])
    accuracies.append(all_results['baseline']['accuracy'] * 100)  # Convert to percentage
    inference_times.append(all_results['baseline']['inference_time_ms'])
    
    # Add INT4 baseline (without clustering)
    variants.append('int4_baseline')
    model_sizes.append(all_results['int4_baseline']['model_size_kb'])
    memory_usages.append(all_results['int4_baseline']['tensor_arena_kb'])
    accuracies.append(all_results['int4_baseline']['accuracy'] * 100)
    inference_times.append(all_results['int4_baseline']['inference_time_ms'])
    
    # Add clustered variants
    for cluster_size in [16, 32, 64]:
        # Regular clustering
        variant = f'clustered_{cluster_size}'
        if variant in all_results:
            variants.append(variant)
            model_sizes.append(all_results[variant]['model_size_kb'])
            memory_usages.append(all_results[variant]['tensor_arena_kb'])
            accuracies.append(all_results[variant]['accuracy'] * 100)
            inference_times.append(all_results[variant]['inference_time_ms'])
        
        # INT4 + clustering
        variant = f'int4_clustered_{cluster_size}'
        if variant in all_results:
            variants.append(variant)
            model_sizes.append(all_results[variant]['model_size_kb'])
            memory_usages.append(all_results[variant]['tensor_arena_kb'])
            accuracies.append(all_results[variant]['accuracy'] * 100)
            inference_times.append(all_results[variant]['inference_time_ms'])
    
    # Create visualizations
    create_bar_chart(
        model_sizes, variants,
        'Model Size Comparison', 'Size (KB)',
        os.path.join(args.output_dir, 'model_size_comparison.png')
    )
    
    create_bar_chart(
        memory_usages, variants,
        'Memory Usage Comparison', 'Memory (KB)',
        os.path.join(args.output_dir, 'memory_usage_comparison.png')
    )
    
    # For accuracy, higher is better, so we need to invert the comparison
    create_bar_chart(
        accuracies, variants,
        'Accuracy Comparison', 'Accuracy (%)',
        os.path.join(args.output_dir, 'accuracy_comparison.png')
    )
    
    create_bar_chart(
        inference_times, variants,
        'Inference Time Comparison', 'Time (ms)',
        os.path.join(args.output_dir, 'inference_time_comparison.png')
    )
    
    # Create tradeoff scatter plot
    create_scatter_plot(
        all_results,
        os.path.join(args.output_dir, 'compression_accuracy_tradeoff.png')
    )
    
    # Create clustering details chart if available
    if 'clustering_stats' in all_results:
        cluster_sizes = []
        unique_before = []
        unique_after = []
        compression_ratios = []
        
        for size, stats in all_results['clustering_stats'].items():
            cluster_sizes.append(f"{size} clusters")
            unique_before.append(stats['unique_values_before'])
            unique_after.append(stats['unique_values_after'])
            compression_ratios.append(stats['compression_ratio'] * 100)
        
        # Create grouped bar chart for unique values
        plt.figure(figsize=(10, 6))
        x = np.arange(len(cluster_sizes))
        width = 0.35
        
        plt.bar(x - width/2, unique_before, width, label='Before Clustering', color='skyblue')
        plt.bar(x + width/2, unique_after, width, label='After Clustering', color='navy')
        
        plt.title('Unique Weight Values Before and After Clustering')
        plt.xlabel('Cluster Size')
        plt.ylabel('Number of Unique Values')
        plt.xticks(x, cluster_sizes)
        plt.legend()
        
        # Add compression ratio labels
        for i, ratio in enumerate(compression_ratios):
            plt.text(i, max(unique_before) * 0.5, f"{ratio:.1f}% reduction", 
                    ha='center', rotation=90, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'clustering_details.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize weight clustering results")
    parser.add_argument("--results_json", type=str, required=True, help="Path to the JSON file with clustering results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualization plots")
    
    args = parser.parse_args()
    main(args)
