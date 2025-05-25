#!/usr/bin/env python3
"""
MicroPizzaNet Early Exit Evaluation
Uses different confidence thresholds and demonstrates energy savings
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    create_optimized_dataloaders, RP2040Config
)
from scripts.early_exit.micropizzanet_early_exit import MicroPizzaNetWithEarlyExit

def run_forced_evaluation(model_path, data_dir="data/augmented", batch_size=16, device="cuda"):
    """
    Runs evaluation on the early exit model, forcing different proportions of samples 
    through the early exit path to demonstrate potential energy savings.
    """
    # Configuration and data loading
    logger.info(f"Loading configuration with data directory: {data_dir}")
    config = RP2040Config(data_dir=data_dir)
    config.BATCH_SIZE = batch_size
    
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device)
        logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    _, val_loader, class_names, _ = create_optimized_dataloaders(
        config
    )
    logger.info(f"Created loaders with {len(class_names)} classes: {class_names}")
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = MicroPizzaNetWithEarlyExit(num_classes=len(class_names), confidence_threshold=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Dictionary to store results for different early exit rates
    results = {}
    
    # Baseline evaluation (normal inference, no early exit)
    logger.info("Running baseline evaluation...")
    baseline_acc, baseline_time = evaluate_baseline(model, val_loader, device)
    
    logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")
    logger.info(f"Baseline inference time: {baseline_time:.6f} seconds")
    
    # Test different forced early exit rates
    exit_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for exit_rate in exit_rates:
        logger.info(f"\nTesting with forced early exit rate: {exit_rate:.1f}")
        
        # Run evaluation with forced early exit
        accuracy, avg_time, early_exit_count, total_samples = evaluate_with_forced_early_exit(
            model, val_loader, device, forced_exit_rate=exit_rate
        )
        
        time_saved_percent = 100 * (1 - avg_time / baseline_time)
        
        actual_exit_rate = 100 * early_exit_count / total_samples
        
        results[exit_rate] = {
            'accuracy': accuracy,
            'time_per_sample': avg_time,
            'time_saved_percent': time_saved_percent,
            'actual_exit_rate': actual_exit_rate
        }
        
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Average time per sample: {avg_time:.6f} seconds")
        logger.info(f"Time saved: {time_saved_percent:.2f}%")
        logger.info(f"Actual early exit rate: {actual_exit_rate:.2f}%")
    
    # Visualize results
    logger.info("Plotting results...")
    plot_results(results, baseline_acc, baseline_time)
    logger.info("Plots saved to early_exit_energy_savings.png and early_exit_metrics_table.png")
    
    return results

def evaluate_baseline(model, val_loader, device):
    """
    Perform baseline evaluation without early exit
    """
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Baseline evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Debug info about inputs and labels
            logger.info(f"Input batch shape: {inputs.shape}, Labels: {labels}")
            
            # Disable early exit
            outputs, _ = model(inputs, use_early_exit=False)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Debug info about predictions
            logger.info(f"Predictions: {predicted}, Ground truth: {labels}")
    
    end_time = time.time()
    accuracy = 100 * correct / total
    avg_time_per_sample = (end_time - start_time) / total
    
    return accuracy, avg_time_per_sample

def evaluate_with_forced_early_exit(model, val_loader, device, forced_exit_rate=0.5):
    """
    Perform evaluation with a forced early exit rate to demonstrate
    potential energy savings.
    """
    correct = 0
    total = 0
    early_exit_count = 0
    cumulative_time = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Forced exit rate {forced_exit_rate:.1f}"):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            # Determine how many samples should use early exit in this batch
            num_early_exit = int(batch_size * forced_exit_rate)
            logger.info(f"Batch size: {batch_size}, Forcing {num_early_exit} samples to use early exit")
            
            # Process each sample in the batch individually to control early exit
            for i in range(batch_size):
                start_time = time.time()
                
                # Force early exit for some samples
                use_early_exit = i < num_early_exit
                
                # First, run through block1 and block2
                x = model.block1(inputs[i:i+1])
                x = model.block2(x)
                
                if use_early_exit:
                    # Use early exit path
                    early_features = model.early_exit_pooling(x)
                    early_exit_output = model.early_exit_classifier(early_features)
                    _, predicted = torch.max(early_exit_output, 1)
                    early_exit_count += 1
                    logger.info(f"Sample {i} using early exit. Prediction: {predicted.item()}, Truth: {labels[i].item()}")
                else:
                    # Use full inference path
                    x = model.block3(x)
                    x = model.global_pool(x)
                    main_output = model.classifier(x)
                    _, predicted = torch.max(main_output, 1)
                    logger.info(f"Sample {i} using full path. Prediction: {predicted.item()}, Truth: {labels[i].item()}")
                
                end_time = time.time()
                
                total += 1
                correct += (predicted == labels[i:i+1]).sum().item()
                cumulative_time += (end_time - start_time)
    
    accuracy = 100 * correct / total
    avg_time_per_sample = cumulative_time / total
    
    return accuracy, avg_time_per_sample, early_exit_count, total

def plot_results(results, baseline_acc, baseline_time):
    """
    Visualize the trade-off between accuracy and time/energy savings
    """
    exit_rates = list(results.keys())
    accuracies = [results[rate]['accuracy'] for rate in exit_rates]
    time_savings = [results[rate]['time_saved_percent'] for rate in exit_rates]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # First axis: Accuracy
    color = 'tab:blue'
    ax1.set_xlabel('Forced Early Exit Rate')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(exit_rates, accuracies, 'o-', color=color, label='Accuracy')
    ax1.axhline(y=baseline_acc, color=color, linestyle='--', alpha=0.7, label='Baseline Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Second axis: Time/Energy Savings
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time Saved (%)', color=color)
    ax2.plot(exit_rates, time_savings, 's-', color=color, label='Time Saved')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Early Exit Performance vs. Energy Savings')
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig("early_exit_energy_savings.png")
    plt.close()
    
    # Create a table for easy interpretation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    
    # Table data
    table_data = [
        ['Exit Rate', 'Accuracy (%)', 'Time/Sample (s)', 'Time Saved (%)', 'Actual Exit Rate (%)'],
    ]
    
    for rate in exit_rates:
        row = [
            f"{rate:.1f}",
            f"{results[rate]['accuracy']:.2f}",
            f"{results[rate]['time_per_sample']:.6f}",
            f"{results[rate]['time_saved_percent']:.2f}",
            f"{results[rate]['actual_exit_rate']:.2f}"
        ]
        table_data.append(row)
    
    # Add baseline
    table_data.append(['Baseline', f"{baseline_acc:.2f}", f"{baseline_time:.6f}", "0.00", "0.00"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Early Exit Performance Metrics')
    plt.tight_layout()
    
    # Save table figure
    plt.savefig("early_exit_metrics_table.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate early exit with forced exit rates")
    parser.add_argument("--model-path", type=str, 
                        default="models_optimized/micropizzanet_early_exit.pth",
                        help="Path to the trained early exit model")
    parser.add_argument("--data-dir", type=str, 
                        default="data/augmented",
                        help="Path to the dataset directory")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    run_forced_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device
    )
