#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Impact of Smaller Input Image Sizes

This script tests different input image sizes (32x32, 40x40, etc.) and evaluates:
1. Model accuracy with the new input size
2. RAM requirements (Framebuffer + Tensor Arena)

For each size, it:
- Updates src/constants.py to use the new input size
- Trains a new model with the smaller image size
- Evaluates model accuracy
- Measures/estimates RAM usage
- Generates a summary report
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import time
import shutil

# Add parent directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import INPUT_SIZE  # Original input size before modification
from src.pizza_detector import RP2040Config, MemoryEstimator
from src.emulation.frame_buffer import FrameBuffer, PixelFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("input_size_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = project_root / "output" / "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def update_input_size_constant(size):
    """
    Update the INPUT_SIZE constant in src/constants.py
    
    Args:
        size: New size value to set
    """
    constants_file = project_root / "src" / "constants.py"
    with open(constants_file, 'r') as f:
        content = f.read()
    
    # Replace the INPUT_SIZE line
    updated_content = content.replace(f"INPUT_SIZE = {INPUT_SIZE}", f"INPUT_SIZE = {size}")
    
    with open(constants_file, 'w') as f:
        f.write(updated_content)
    
    logger.info(f"Updated INPUT_SIZE to {size} in constants.py")

def train_model_with_size(size):
    """
    Train a model with the specified input size
    
    Args:
        size: Input image size
        
    Returns:
        Path to the trained model file
    """
    logger.info(f"Training model with input size {size}x{size}")
    
    # Create output model directory
    model_dir = project_root / "models" / f"size_{size}x{size}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Run the training script
    train_cmd = [
        sys.executable, 
        str(project_root / "scripts" / "train_pizza_model.py"),
        "--input_size", str(size),
        "--epochs", "30",  # Use fewer epochs for quicker testing
        "--output", str(model_dir / f"pizza_model_size_{size}.pth"),
        "--no_early_exit"  # Disable early exit for consistent comparison
    ]
    
    process = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Training failed with size {size}x{size}")
        logger.error(process.stderr)
        raise RuntimeError(f"Training failed for size {size}x{size}")
    
    # Extract training results
    model_path = model_dir / f"pizza_model_size_{size}.pth"
    
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Model trained successfully: {model_path}")
    return model_path

def evaluate_model_accuracy(model_path, size):
    """
    Evaluate model accuracy on the test dataset
    
    Args:
        model_path: Path to the trained model
        size: Input image size
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model accuracy for size {size}x{size}")
    
    # Run evaluation script
    eval_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pizza_tests.py"),
        "--model", str(model_path),
        "--quiet"  # Less verbose output
    ]
    
    process = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Evaluation failed for size {size}x{size}")
        logger.error(process.stderr)
        raise RuntimeError(f"Evaluation failed for size {size}x{size}")
    
    # Parse the evaluation output to extract metrics
    output = process.stdout
    metrics = {}
    
    # Extract accuracy from output
    for line in output.splitlines():
        if "Accuracy:" in line:
            metrics["accuracy"] = float(line.split(":")[1].strip().replace("%", "")) / 100
        elif "F1 Score:" in line:
            metrics["f1_score"] = float(line.split(":")[1].strip())
    
    logger.info(f"Model evaluated with accuracy: {metrics.get('accuracy', 'N/A')}")
    return metrics

def estimate_ram_usage(size):
    """
    Estimate RAM usage for the given input size
    
    Args:
        size: Input image size
        
    Returns:
        Dictionary with RAM usage estimates
    """
    logger.info(f"Estimating RAM usage for size {size}x{size}")
    
    # Calculate framebuffer size
    framebuffer = FrameBuffer(width=size, height=size, pixel_format=PixelFormat.RGB888)
    framebuffer_kb = framebuffer.total_size_bytes / 1024
    
    # Run tensor arena estimation
    model_path = project_root / "models" / f"size_{size}x{size}" / f"pizza_model_size_{size}.pth"
    
    tensor_arena_cmd = [
        sys.executable,
        str(project_root / "scripts" / "measure_tensor_arena.py"),
        "--model", str(model_path),
        "--input_size", str(size)
    ]
    
    process = subprocess.run(tensor_arena_cmd, capture_output=True, text=True)
    tensor_arena_kb = None
    
    # Extract tensor arena size from output
    for line in process.stdout.splitlines():
        if "Tensor Arena Size:" in line:
            tensor_arena_kb = float(line.split(":")[1].strip().replace("KB", "").strip())
    
    if tensor_arena_kb is None:
        # Fallback estimation
        config = RP2040Config()
        memory_estimator = MemoryEstimator(config)
        model_size_kb = os.path.getsize(model_path) / 1024
        tensor_arena_kb = memory_estimator.estimate_tensor_arena_size(model_size_kb, is_quantized=False)
    
    # Total RAM usage
    total_ram_kb = framebuffer_kb + tensor_arena_kb
    
    ram_usage = {
        "framebuffer_kb": round(framebuffer_kb, 2),
        "tensor_arena_kb": round(tensor_arena_kb, 2),
        "total_ram_kb": round(total_ram_kb, 2),
        "percentage_of_rp2040_ram": round((total_ram_kb / 264) * 100, 2)  # 264KB is total RP2040 RAM
    }
    
    logger.info(f"RAM usage for {size}x{size}: Framebuffer {ram_usage['framebuffer_kb']}KB, "
                f"Tensor Arena {ram_usage['tensor_arena_kb']}KB, "
                f"Total {ram_usage['total_ram_kb']}KB "
                f"({ram_usage['percentage_of_rp2040_ram']}% of available RAM)")
    
    return ram_usage

def generate_report(size, accuracy_metrics, ram_usage):
    """
    Generate a JSON report for the evaluation
    
    Args:
        size: Input image size
        accuracy_metrics: Dictionary with accuracy metrics
        ram_usage: Dictionary with RAM usage metrics
        
    Returns:
        Path to the generated report
    """
    report = {
        "input_size": size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy_metrics": accuracy_metrics,
        "ram_usage": ram_usage
    }
    
    report_path = OUTPUT_DIR / f"eval_size_{size}x{size}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {report_path}")
    return report_path

def main():
    """Main function to evaluate different input sizes"""
    parser = argparse.ArgumentParser(description="Evaluate impact of different input image sizes")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 40, 48],
                        help="List of image sizes to evaluate")
    args = parser.parse_args()
    
    # Save original constants.py as backup
    constants_file = project_root / "src" / "constants.py"
    constants_backup = constants_file.with_suffix(".py.bak")
    shutil.copy(constants_file, constants_backup)
    logger.info(f"Backed up constants.py to {constants_backup}")
    
    # Results summary
    summary = {}
    
    try:
        # Process each size
        for size in args.sizes:
            logger.info(f"=============== Processing size {size}x{size} ===============")
            
            # Update INPUT_SIZE constant
            update_input_size_constant(size)
            
            # Train model
            model_path = train_model_with_size(size)
            
            # Evaluate accuracy
            accuracy_metrics = evaluate_model_accuracy(model_path, size)
            
            # Estimate RAM usage
            ram_usage = estimate_ram_usage(size)
            
            # Generate report
            report_path = generate_report(size, accuracy_metrics, ram_usage)
            
            # Add to summary
            summary[f"{size}x{size}"] = {
                "accuracy": accuracy_metrics.get("accuracy", "N/A"),
                "total_ram_kb": ram_usage["total_ram_kb"],
                "report_path": str(report_path)
            }
        
        # Generate summary report
        summary_path = OUTPUT_DIR / "input_size_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
    finally:
        # Restore original constants.py
        shutil.copy(constants_backup, constants_file)
        logger.info(f"Restored original constants.py from backup")

if __name__ == "__main__":
    main()
