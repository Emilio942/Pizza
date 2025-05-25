#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruning Evaluation Report Visualization

This script creates visualization plots for the pruning evaluation results.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_report(report_path):
    """Load the pruning evaluation report"""
    with open(report_path, 'r') as f:
        return json.load(f)

def create_visualizations(results, output_dir):
    """Create visualization plots for the pruning results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    base_accuracy = results["base_model"]["accuracy"]
    base_size = results["base_model"]["size_kb"]
    base_ram = results["base_model"]["ram_usage_kb"]
    base_time = results["base_model"]["inference_time_ms"]
    
    sparsities = []
    pruned_accuracy = []
    pruned_size = []
    pruned_ram = []
    pruned_time = []
    
    quantized_accuracy = []
    quantized_size = []
    quantized_ram = []
    quantized_time = []
    quantized_arena = []
    
    for key, model_info in results["pruned_models"].items():
        sparsity = model_info["sparsity"]
        sparsities.append(sparsity)
        
        # Pruned model metrics
        pruned = model_info["pruned"]
        pruned_accuracy.append(pruned["accuracy"])
        pruned_size.append(pruned["size_kb"])
        pruned_ram.append(pruned["ram_usage_kb"])
        pruned_time.append(pruned["inference_time_ms"])
        
        # Quantized model metrics
        quantized = model_info["quantized"]
        quantized_accuracy.append(quantized["accuracy"])
        quantized_size.append(quantized["size_kb"])
        quantized_ram.append(quantized["ram_usage_kb"])
        quantized_time.append(quantized["inference_time_ms"])
        quantized_arena.append(quantized["tensor_arena_kb"])
    
    # Sort data by sparsity
    indices = np.argsort(sparsities)
    sparsities = [sparsities[i] for i in indices]
    pruned_accuracy = [pruned_accuracy[i] for i in indices]
    pruned_size = [pruned_size[i] for i in indices]
    pruned_ram = [pruned_ram[i] for i in indices]
    pruned_time = [pruned_time[i] for i in indices]
    
    quantized_accuracy = [quantized_accuracy[i] for i in indices]
    quantized_size = [quantized_size[i] for i in indices]
    quantized_ram = [quantized_ram[i] for i in indices]
    quantized_time = [quantized_time[i] for i in indices]
    quantized_arena = [quantized_arena[i] for i in indices]
    
    # Add base model as sparsity 0
    sparsities = [0] + sparsities
    pruned_accuracy = [base_accuracy] + pruned_accuracy
    pruned_size = [base_size] + pruned_size
    pruned_ram = [base_ram] + pruned_ram
    pruned_time = [base_time] + pruned_time
    
    quantized_accuracy = [base_accuracy] + quantized_accuracy
    quantized_size = [base_size] + quantized_size
    quantized_ram = [base_ram] + quantized_ram
    quantized_time = [base_time] + quantized_time
    quantized_arena = [base_ram] + quantized_arena  # Use base RAM as proxy for base tensor arena
    
    # Create plots
    
    # 1. Accuracy vs. Sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, pruned_accuracy, 'o-', label='Pruned Model')
    plt.plot(sparsities, quantized_accuracy, 's-', label='Pruned + Quantized Model')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy vs. Pruning Sparsity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_sparsity.png'))
    
    # 2. Model Size vs. Sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, pruned_size, 'o-', label='Pruned Model')
    plt.plot(sparsities, quantized_size, 's-', label='Pruned + Quantized Model')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('Model Size (KB)')
    plt.title('Model Size vs. Pruning Sparsity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'size_vs_sparsity.png'))
    
    # 3. RAM Usage vs. Sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, pruned_ram, 'o-', label='Pruned Model RAM')
    plt.plot(sparsities, quantized_ram, 's-', label='Pruned + Quantized Model RAM')
    plt.plot(sparsities, quantized_arena, '^-', label='Tensor Arena Size')
    plt.axhline(y=204, color='r', linestyle='--', label='204 KB RAM Limit')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('RAM Usage (KB)')
    plt.title('RAM Usage vs. Pruning Sparsity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ram_vs_sparsity.png'))
    
    # 4. Inference Time vs. Sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, pruned_time, 'o-', label='Pruned Model')
    plt.plot(sparsities, quantized_time, 's-', label='Pruned + Quantized Model')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time vs. Pruning Sparsity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'inference_time_vs_sparsity.png'))
    
    # 5. Tradeoff: Accuracy vs. Size
    plt.figure(figsize=(10, 6))
    plt.plot(pruned_size, pruned_accuracy, 'o-', label='Pruned Model')
    plt.plot(quantized_size, quantized_accuracy, 's-', label='Pruned + Quantized Model')
    
    # Add annotations for sparsity rates
    for i, s in enumerate(sparsities):
        plt.annotate(f"s={s:.1f}", (pruned_size[i], pruned_accuracy[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"s={s:.1f}", (quantized_size[i], quantized_accuracy[i]), 
                     textcoords="offset points", xytext=(0,-15), ha='center')
    
    plt.xlabel('Model Size (KB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Model Size Tradeoff')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_size.png'))
    
    print(f"Created visualization plots in {output_dir}")

def main():
    """Main function"""
    report_path = os.path.join(project_root, "output", "model_optimization", "pruning_evaluation.json")
    output_dir = os.path.join(project_root, "output", "model_optimization", "visualizations")
    
    try:
        results = load_report(report_path)
        create_visualizations(results, output_dir)
        print("Visualization completed successfully")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
