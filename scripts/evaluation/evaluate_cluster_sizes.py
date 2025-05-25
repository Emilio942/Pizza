#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight Clustering Evaluation with Different Cluster Sizes
--------------------------------------------------------
This script implements and evaluates weight clustering with different cluster counts
(16, 32, 64) and combines it with INT4 quantization to optimize the MicroPizzaNet model.

The evaluation includes metrics on model size, accuracy, RAM usage, and inference time
for each configuration.

Usage:
    python scripts/evaluate_cluster_sizes.py --model_path models/micro_pizza_model.pth --output_dir output/clustering_evaluation
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)

# Import modules
from scripts.model_optimization.weight_pruning import WeightClusterer, evaluate_model
from scripts.model_optimization.int4_quantization import INT4Quantizer
from src.pizza_detector import (
    MicroPizzaNet, create_optimized_dataloaders, MemoryEstimator
)
from config.config import DefaultConfig as Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cluster_sizes_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def measure_inference_time(model, val_loader, device, num_runs=50):
    """
    Measures the average inference time of the model.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        device: Device for computations
        num_runs: Number of runs
        
    Returns:
        Average inference time in milliseconds
    """
    logger.info(f"Measuring inference time over {num_runs} runs...")
    
    model.eval()
    
    # Get a single batch for repeated inference
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        break
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
    
    # Measure time
    inference_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(inputs)
            inference_time = (time.time() - start_time) * 1000  # in ms
            inference_times.append(inference_time)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    return avg_inference_time

def estimate_model_metrics(model, val_loader, class_names, device, bits=32):
    """
    Estimates model metrics including size, memory usage, accuracy, and inference time.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        class_names: List of class names
        device: Device for computations
        bits: Bit precision of the model
        
    Returns:
        Dictionary with model metrics
    """
    # Estimate model size
    model_size_kb = MemoryEstimator.estimate_model_size(model, bits=bits)
    
    # Estimate RAM usage (tensor arena)
    tensor_arena_kb = MemoryEstimator.estimate_tensor_arena(model)
    
    # Evaluate accuracy
    accuracy_results = evaluate_model(model, val_loader, class_names, device)
    
    # Measure inference time
    inference_time_ms = measure_inference_time(model, val_loader, device)
    
    return {
        "model_size_kb": model_size_kb,
        "tensor_arena_kb": tensor_arena_kb,
        "total_ram_kb": model_size_kb + tensor_arena_kb,
        "accuracy": accuracy_results["accuracy"],
        "inference_time_ms": inference_time_ms,
        "class_accuracies": accuracy_results["class_accuracies"]
    }

def apply_weight_clustering(model, num_clusters):
    """
    Applies weight clustering to the model with the specified number of clusters.
    
    Args:
        model: PyTorch model
        num_clusters: Number of clusters
        
    Returns:
        Clustered model and clustering statistics
    """
    logger.info(f"Starting weight clustering with {num_clusters} clusters...")
    
    # Initialize clusterer
    clusterer = WeightClusterer(
        model=model,
        num_clusters=num_clusters
    )
    
    # Perform clustering
    clusterer.cluster_weights()
    
    # Get statistics
    clustering_stats = clusterer.get_clustering_stats()
    
    return model, clustering_stats

def apply_int4_quantization(model, val_loader, device):
    """
    Applies INT4 quantization to the model.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        device: Device for computations
        
    Returns:
        Quantized model and quantization statistics
    """
    logger.info("Applying INT4 quantization to the model...")
    
    # Copy model to avoid modifying the original
    quantized_model = MicroPizzaNet(num_classes=model.num_classes)
    quantized_model.load_state_dict(model.state_dict())
    quantized_model = quantized_model.to(device)
    
    # Create INT4 quantizer
    quantizer = INT4Quantizer(
        model=quantized_model,
        calibration_loader=val_loader
    )
    
    # Apply quantization
    quantized_model, quant_stats = quantizer.quantize()
    
    return quantized_model, quant_stats

def generate_comparison_table(all_results):
    """
    Generates a markdown table comparing all model variants.
    
    Args:
        all_results: Dictionary containing results for all model variants
        
    Returns:
        Markdown string with comparison tables
    """
    # Header for the report
    markdown = "# Weight Clustering Evaluation with Different Cluster Sizes\n\n"
    markdown += "## Executive Summary\n\n"
    
    # Calculate best configuration based on combined metrics
    best_config = None
    best_score = 0
    
    for variant, data in all_results.items():
        if variant == "baseline":
            continue
            
        # Simple scoring: normalize each metric and sum
        # Higher accuracy is better, lower size/memory/time is better
        acc_score = data["accuracy"] / all_results["baseline"]["accuracy"]
        size_score = all_results["baseline"]["model_size_kb"] / data["model_size_kb"]
        ram_score = all_results["baseline"]["tensor_arena_kb"] / data["tensor_arena_kb"]
        time_score = all_results["baseline"]["inference_time_ms"] / data["inference_time_ms"]
        
        # Weight the scores (can be adjusted based on priorities)
        total_score = acc_score * 0.4 + size_score * 0.2 + ram_score * 0.2 + time_score * 0.2
        
        if total_score > best_score:
            best_score = total_score
            best_config = variant
    
    # Add executive summary
    markdown += f"This report evaluates weight clustering with different cluster sizes (16, 32, 64) and INT4 quantization on the MicroPizzaNet model. "
    markdown += f"The evaluation shows that the **{best_config}** configuration provides the best balance of model size, accuracy, RAM usage, and inference time.\n\n"
    
    # Model Size Comparison table
    markdown += "## Model Size Comparison\n\n"
    markdown += "| Model Variant | Size (KB) | Reduction |\n"
    markdown += "|---------------|-----------|----------|\n"
    
    baseline_size = all_results["baseline"]["model_size_kb"]
    for variant, data in all_results.items():
        size = data["model_size_kb"]
        reduction = (1 - size / baseline_size) * 100
        markdown += f"| {variant} | {size:.2f} | {reduction:.2f}% |\n"
    
    # Memory Usage Comparison table
    markdown += "\n## Memory Usage Comparison\n\n"
    markdown += "| Model Variant | Memory (KB) | Reduction |\n"
    markdown += "|---------------|-------------|----------|\n"
    
    baseline_ram = all_results["baseline"]["tensor_arena_kb"]
    for variant, data in all_results.items():
        ram = data["tensor_arena_kb"]
        reduction = (1 - ram / baseline_ram) * 100
        markdown += f"| {variant} | {ram:.2f} | {reduction:.2f}% |\n"
    
    # Performance Metrics table
    markdown += "\n## Performance Metrics\n\n"
    markdown += "| Model Variant | Accuracy (%) | Inference Time (ms) |\n"
    markdown += "|---------------|--------------|---------------------|\n"
    
    for variant, data in all_results.items():
        accuracy = data["accuracy"] * 100
        inf_time = data["inference_time_ms"]
        markdown += f"| {variant} | {accuracy:.2f} | {inf_time:.2f} |\n"
    
    # Clustering Details
    if "clustering_stats" in all_results:
        markdown += "\n## Clustering Details\n\n"
        markdown += "| Cluster Size | Unique Values Before | Unique Values After | Reduction |\n"
        markdown += "|--------------|----------------------|---------------------|----------|\n"
        
        for size, stats in all_results["clustering_stats"].items():
            before = stats["unique_values_before"]
            after = stats["unique_values_after"]
            reduction = stats["compression_ratio"] * 100
            markdown += f"| {size} | {before} | {after} | {reduction:.2f}% |\n"
    
    return markdown

def main(args):
    """Main function to evaluate weight clustering with different cluster sizes."""
    # Load configuration
    config = Config()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    
    # Load pre-trained model
    logger.info(f"Loading pre-trained model from: {args.model_path}")
    model = MicroPizzaNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate baseline model
    logger.info("Evaluating baseline model")
    baseline_metrics = estimate_model_metrics(model, val_loader, class_names, device)
    
    # Store all results
    all_results = {
        "baseline": baseline_metrics
    }
    
    # Store clustering statistics
    all_results["clustering_stats"] = {}
    
    # Evaluate different cluster sizes
    cluster_sizes = [16, 32, 64]
    
    for num_clusters in cluster_sizes:
        logger.info(f"\n{'='*50}\nEvaluating clustering with {num_clusters} clusters\n{'='*50}")
        
        # Create a copy of the model for this configuration
        current_model = MicroPizzaNet(num_classes=len(class_names))
        current_model.load_state_dict(model.state_dict())
        current_model = current_model.to(device)
        
        # Apply weight clustering
        clustered_model, clustering_stats = apply_weight_clustering(current_model, num_clusters)
        all_results["clustering_stats"][num_clusters] = clustering_stats
        
        # Evaluate clustered model
        logger.info(f"Evaluating clustered model with {num_clusters} clusters")
        clustered_metrics = estimate_model_metrics(clustered_model, val_loader, class_names, device)
        all_results[f"clustered_{num_clusters}"] = clustered_metrics
        
        # Save clustered model
        clustered_model_path = os.path.join(args.output_dir, f"clustered_{num_clusters}.pth")
        torch.save(clustered_model.state_dict(), clustered_model_path)
        
        # Apply INT4 quantization
        int4_model, quant_stats = apply_int4_quantization(clustered_model, val_loader, device)
        
        # Evaluate INT4 quantized clustered model
        logger.info(f"Evaluating INT4 quantized model with {num_clusters} clusters")
        int4_metrics = estimate_model_metrics(int4_model, val_loader, class_names, device, bits=4)
        all_results[f"int4_clustered_{num_clusters}"] = int4_metrics
        
        # Save INT4 quantized model
        int4_model_path = os.path.join(args.output_dir, f"int4_clustered_{num_clusters}.pth")
        torch.save(int4_model.state_dict(), int4_model_path)
    
    # Also evaluate INT4 quantization without clustering
    base_model = MicroPizzaNet(num_classes=len(class_names))
    base_model.load_state_dict(model.state_dict())
    base_model = base_model.to(device)
    
    int4_base_model, base_quant_stats = apply_int4_quantization(base_model, val_loader, device)
    
    logger.info("Evaluating INT4 quantized baseline model (without clustering)")
    int4_base_metrics = estimate_model_metrics(int4_base_model, val_loader, class_names, device, bits=4)
    all_results["int4_baseline"] = int4_base_metrics
    
    # Generate comparison report
    report_markdown = generate_comparison_table(all_results)
    
    # Save the report
    report_path = os.path.join(args.output_dir, "clustered_sizes_evaluation.md")
    with open(report_path, 'w') as f:
        f.write(report_markdown)
    
    # Save all results as JSON for further analysis
    results_path = os.path.join(args.output_dir, "all_results.json")
    
    # Convert non-serializable objects to serializable format
    serializable_results = {}
    for key, value in all_results.items():
        if key == "clustering_stats":
            serializable_stats = {}
            for size, stats in value.items():
                serializable_stats[size] = {
                    "unique_values_before": stats["unique_values_before"],
                    "unique_values_after": stats["unique_values_after"],
                    "compression_ratio": stats["compression_ratio"]
                }
            serializable_results[key] = serializable_stats
        else:
            serializable_results[key] = {
                k: v for k, v in value.items() 
                if not isinstance(v, (torch.Tensor, np.ndarray))
            }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation completed. Report saved to {report_path}")
    logger.info(f"Detailed results saved to {results_path}")
    
    return report_path, results_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate weight clustering with different cluster sizes")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation (cuda or cpu)")
    
    args = parser.parse_args()
    main(args)
