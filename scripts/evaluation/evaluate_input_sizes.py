#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified script to evaluate impact of different input image sizes on
accuracy and RAM usage for the pizza classification model.

This script tests each specified image size and generates reports of accuracy and RAM usage.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import INPUT_SIZE
from src.emulation.frame_buffer import FrameBuffer, PixelFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = project_root / "output" / "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_framebuffer_size(size):
    """Calculate framebuffer size for a specific image size"""
    framebuffer = FrameBuffer(width=size, height=size, pixel_format=PixelFormat.RGB888)
    return framebuffer.total_size_bytes / 1024  # Return size in KB

def train_and_evaluate_size(size):
    """
    Train and evaluate a model with the given input size
    
    Args:
        size: Input image size to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating input size: {size}x{size}")
    
    # Step 1: Train the model
    model_dir = project_root / "models" / f"size_{size}x{size}"
    model_path = model_dir / f"pizza_model_size_{size}.pth"
    os.makedirs(model_dir, exist_ok=True)
    
    train_cmd = [
        sys.executable,
        str(project_root / "scripts" / "train_pizza_model.py"),
        "--input-size", str(size),
        "--epochs", "30",  # Reduce epochs for faster evaluation
        "--output", str(model_path)
    ]
    
    logger.info(f"Training model with size {size}x{size}...")
    train_process = subprocess.run(train_cmd, capture_output=True, text=True)
    training_output = train_process.stdout
    
    if not model_path.exists():
        logger.error(f"Model training failed for size {size}x{size}")
        return None
    
    # Extract best validation accuracy from model training output
    accuracy = None
    validation_text = "Beste Validation Accuracy:"
    best_accuracy_line = None
    
    for line in training_output.splitlines():
        if validation_text in line:
            best_accuracy_line = line
            break
    
    if best_accuracy_line:
        # Parse the accuracy value from a line like "Beste Validation Accuracy: 35.00% in Epoch 1"
        parts = best_accuracy_line.split(validation_text)
        if len(parts) > 1:
            acc_part = parts[1].strip()
            if "%" in acc_part:
                try:
                    accuracy = float(acc_part.split("%")[0].strip()) / 100
                except ValueError:
                    logger.warning(f"Could not parse accuracy from {acc_part}")
    
    # Step 2: Evaluate model accuracy using the test set
    eval_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pizza_tests.py"),
        "--model", str(model_path)
    ]
    
    logger.info(f"Evaluating accuracy for size {size}x{size}...")
    eval_process = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    # Extract accuracy from evaluation output if available
    test_accuracy = None
    for line in eval_process.stdout.splitlines():
        if "Accuracy:" in line:
            try:
                test_accuracy = float(line.split(":")[1].strip().replace("%", "")) / 100
                # If we found a test accuracy, use it instead of validation accuracy
                if test_accuracy is not None:
                    accuracy = test_accuracy
                break
            except ValueError:
                pass
    
    # Step 3: Measure tensor arena size
    logger.info(f"Measuring tensor arena size for size {size}x{size}...")
    tensor_cmd = [
        sys.executable,
        str(project_root / "scripts" / "measure_tensor_arena.py"),
        "--model", str(model_path),
        "--input_size", str(size)
    ]
    
    tensor_process = subprocess.run(tensor_cmd, capture_output=True, text=True)
    
    # Extract tensor arena size from output
    tensor_arena_kb = None
    for line in tensor_process.stdout.splitlines():
        if "Tensor-Arena-Größe:" in line:
            tensor_arena_kb = float(line.split(":")[1].strip().replace("KB", "").strip())
            break
    
    # Fallback if automatic extraction fails
    if tensor_arena_kb is None:
        tensor_arena_kb = os.path.getsize(model_path) / 1024 * 0.2  # Simple estimate: 20% of model size
    
    # Step 4: Calculate framebuffer size
    framebuffer_kb = calculate_framebuffer_size(size)
    
    # Total RAM usage
    total_ram_kb = framebuffer_kb + tensor_arena_kb
    
    # Create result report
    results = {
        "input_size": size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy,
        "ram_usage": {
            "framebuffer_kb": round(framebuffer_kb, 2),
            "tensor_arena_kb": round(tensor_arena_kb, 2),
            "total_ram_kb": round(total_ram_kb, 2),
            "percentage_of_rp2040_ram": round((total_ram_kb / 264) * 100, 2)  # 264KB is total RP2040 RAM
        }
    }
    
    # Save report
    report_path = OUTPUT_DIR / f"eval_size_{size}x{size}.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation for size {size}x{size} completed and saved to {report_path}")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate impact of different input image sizes")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 40, 48],
                        help="List of image sizes to evaluate")
    args = parser.parse_args()
    
    # Summary of all results
    summary = {}
    
    # Process each size
    for size in args.sizes:
        results = train_and_evaluate_size(size)
        if results:
            summary[f"{size}x{size}"] = {
                "accuracy": results["accuracy"],
                "total_ram_kb": results["ram_usage"]["total_ram_kb"]
            }
    
    # Create summary report
    summary_path = OUTPUT_DIR / "input_size_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Evaluation of different input sizes for RAM usage vs. accuracy trade-off"
        }, f, indent=2)
    
    # Print summary
    print("\n===== Input Size Evaluation Summary =====")
    print(f"{'Size':<10} {'Accuracy':<15} {'RAM Usage':<15}")
    print("-" * 40)
    for size_name, data in summary.items():
        accuracy = data.get('accuracy')
        ram_usage = data.get('total_ram_kb')
        accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        ram_str = f"{ram_usage:.2f}KB" if ram_usage is not None else "N/A"
        print(f"{size_name:<10} {accuracy_str:<15} {ram_str:<15}")
    
    print(f"\nFull summary saved to {summary_path}")

if __name__ == "__main__":
    main()
