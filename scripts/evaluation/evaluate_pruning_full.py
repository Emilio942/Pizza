#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Pruning Evaluation for MicroPizzaNetV2

This script implements comprehensive structured pruning for the MicroPizzaNetV2 model.
It evaluates models at various sparsity rates (10%, 20%, 30%), measures performance,
quantizes the models, and compares results to find the optimal trade-off.

Requirements:
- TensorFlow or tflite_runtime for tensor arena size measurement
- PyTorch for model pruning and quantization
- Pre-trained MicroPizzaNetV2 model

Usage:
    python evaluate_pruning_full.py [--sparsities 0.1 0.2 0.3] [--base_model PATH] [--output_dir DIR]
"""

import os
import sys
import json
import time
import torch
import numpy as np
import logging
import argparse
from datetime import datetime
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from src.pizza_detector import MicroPizzaNetV2, create_optimized_dataloaders
from scripts.pruning_tool import create_pruned_model, quantize_model, save_pruned_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'pruning_evaluation_full.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_model_size(model):
    """Calculate model size in KB (parameters only)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / 1024

def evaluate_model(model, data_loader, device="cpu"):
    """Evaluate model accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(4)}  # Assuming 4 classes
    class_total = {i: 0 for i in range(4)}
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Total accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Calculate accuracies
    accuracy = 100. * correct / total
    class_accuracies = {i: 100. * class_correct[i] / max(1, class_total[i]) for i in range(4)}
    
    logger.info(f'Overall accuracy: {accuracy:.2f}%')
    for i in range(4):
        logger.info(f'Class {i} accuracy: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'correct': correct,
        'total': total
    }

def finetune_model(model, train_loader, val_loader, epochs=5, lr=0.0005, device="cpu"):
    """Finetune a pruned model"""
    logger.info(f"Finetuning model for {epochs} epochs")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    model.to(device)
    best_accuracy = 0
    best_model = None
    history = {'train_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        
        # Validation phase
        result = evaluate_model(model, val_loader, device)
        accuracy = result['accuracy']
        history['val_accuracy'].append(accuracy)
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            logger.info(f"New best model with accuracy: {best_accuracy:.2f}%")
    
    # Restore best model
    model.load_state_dict(best_model)
    logger.info(f"Finetuning completed. Best accuracy: {best_accuracy:.2f}%")
    
    return model, history

def measure_inference_time(model, input_size=(3, 48, 48), device="cpu", num_runs=100):
    """Measure average inference time"""
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    logger.info(f"Average inference time: {avg_time_ms:.2f} ms")
    
    # Extra info: standard deviation
    times = []
    with torch.no_grad():
        for _ in range(min(20, num_runs)):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)
    
    std_dev = np.std(times)
    logger.info(f"Standard deviation: {std_dev:.2f} ms")
    
    return {
        'avg_time_ms': avg_time_ms,
        'std_dev_ms': std_dev,
        'num_runs': num_runs
    }

def measure_tensor_arena_size(model_path):
    """Measure tensor arena size using the measure_tensor_arena script"""
    logger.info(f"Measuring tensor arena size for model: {model_path}")
    
    try:
        script_path = project_root / "scripts" / "measure_tensor_arena.py"
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return None
        
        # Try to import the function directly
        try:
            sys.path.append(str(script_path.parent))
            from measure_tensor_arena import calculate_tensor_memory_usage
            
            report = calculate_tensor_memory_usage(model_path, verbose=False)
            return report
        except ImportError:
            logger.warning("Could not import calculate_tensor_memory_usage directly")
            
            # Fall back to subprocess call
            import subprocess
            cmd = [sys.executable, str(script_path), "--model", model_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to measure tensor arena size: {result.stderr}")
                return None
            
            # Parse output to get tensor arena size
            for line in result.stdout.split('\n'):
                if "Tensor-Arena-Größe:" in line:
                    size_kb = float(line.split(':')[1].split('KB')[0].strip())
                    return {'tensor_arena_size_kb': size_kb}
            
            logger.error("Could not parse tensor arena size from output")
            return None
    except Exception as e:
        logger.error(f"Error measuring tensor arena size: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(results, output_dir):
    """Create visualizations of pruning results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plots
    sparsities = [results[k]['sparsity'] for k in results.keys() if k != 'original']
    sparsities = [0.0] + sorted(sparsities)  # Add original (0.0) and sort
    
    data = {
        'sparsity': sparsities,
        'accuracy': [],
        'model_size_kb': [],
        'inference_time_ms': [],
        'tensor_arena_kb': []
    }
    
    # Original model
    data['accuracy'].append(results['original']['accuracy'])
    data['model_size_kb'].append(results['original']['model_size_kb'])
    data['inference_time_ms'].append(results['original']['inference_time_ms'])
    data['tensor_arena_kb'].append(results['original'].get('tensor_arena_kb', 0))
    
    # Pruned models
    for sparsity in sparsities[1:]:
        key = f"pruned_s{int(sparsity*100)}"
        data['accuracy'].append(results[key]['accuracy_after_finetuning'])
        data['model_size_kb'].append(results[key]['model_size_kb'])
        data['inference_time_ms'].append(results[key]['inference_time_after_ms'])
        data['tensor_arena_kb'].append(results[key].get('tensor_arena_kb', 0))
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # Accuracy vs Sparsity
    plt.subplot(2, 2, 1)
    plt.plot(data['sparsity'], data['accuracy'], 'o-', label='Accuracy')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Sparsity Rate')
    plt.grid(True)
    
    # Model Size vs Sparsity
    plt.subplot(2, 2, 2)
    plt.plot(data['sparsity'], data['model_size_kb'], 's-', label='Model Size')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('Model Size (KB)')
    plt.title('Model Size vs Sparsity Rate')
    plt.grid(True)
    
    # Inference Time vs Sparsity
    plt.subplot(2, 2, 3)
    plt.plot(data['sparsity'], data['inference_time_ms'], 'v-', label='Inference Time')
    plt.xlabel('Sparsity Rate')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time vs Sparsity Rate')
    plt.grid(True)
    
    # Tensor Arena Size vs Sparsity
    if any(data['tensor_arena_kb']):
        plt.subplot(2, 2, 4)
        plt.plot(data['sparsity'], data['tensor_arena_kb'], '^-', label='Tensor Arena Size')
        plt.xlabel('Sparsity Rate')
        plt.ylabel('Tensor Arena Size (KB)')
        plt.title('RAM Usage vs Sparsity Rate')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pruning_results.png')
    logger.info(f"Created visualization: {output_dir / 'pruning_results.png'}")
    
    # Create trade-off plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['model_size_kb'], data['accuracy'], 'o-')
    for i, s in enumerate(data['sparsity']):
        plt.annotate(f"{int(s*100)}%", 
                    (data['model_size_kb'][i], data['accuracy'][i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.xlabel('Model Size (KB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Model Size Trade-off')
    plt.grid(True)
    plt.savefig(output_dir / 'accuracy_size_tradeoff.png')
    logger.info(f"Created trade-off plot: {output_dir / 'accuracy_size_tradeoff.png'}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate structured pruning for MicroPizzaNetV2")
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.1, 0.2, 0.3],
                        help="Sparsity rates to evaluate (default: 0.1 0.2 0.3)")
    parser.add_argument("--base_model", type=str, default="models/micropizzanetv2_base.pth",
                        help="Path to base model (default: models/micropizzanetv2_base.pth)")
    parser.add_argument("--output_dir", type=str, default="output/model_optimization",
                        help="Output directory (default: output/model_optimization)")
    parser.add_argument("--finetune_epochs", type=int, default=5,
                        help="Number of epochs for finetuning (default: 5)")
    parser.add_argument("--finetune_lr", type=float, default=0.0005,
                        help="Learning rate for finetuning (default: 0.0005)")
    parser.add_argument("--skip_quantization", action="store_true",
                        help="Skip quantization step")
    parser.add_argument("--skip_tensor_arena", action="store_true",
                        help="Skip tensor arena measurement (requires TensorFlow)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save arguments for reproducibility
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create/load model
    logger.info("Creating MicroPizzaNetV2 model")
    model = MicroPizzaNetV2(num_classes=4)
    
    # Load model if it exists
    if os.path.exists(args.base_model):
        logger.info(f"Loading model from {args.base_model}")
        model.load_state_dict(torch.load(args.base_model))
    else:
        logger.error(f"Model not found: {args.base_model}")
        logger.error("Please train a base model first")
        return
    
    # Create data loaders
    logger.info("Creating data loaders")
    try:
        # Try to use create_optimized_dataloaders
        train_loader, val_loader, class_names, _ = create_optimized_dataloaders()
        logger.info(f"Created data loaders with classes: {class_names}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        # Create simple dataloaders as fallback
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Try to find a dataset directory
        for dataset_dir in ["augmented_pizza", "augmented_pizza_legacy", "data/pizza"]:
            if os.path.exists(dataset_dir):
                break
        else:
            logger.error("No dataset directory found")
            return
        
        try:
            dataset = ImageFolder(root=dataset_dir, transform=transform)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            logger.info(f"Created fallback data loaders from {dataset_dir}")
        except Exception as e:
            logger.error(f"Failed to create fallback data loaders: {e}")
            return
    
    # Try to use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Evaluate original model
    logger.info("Evaluating original model")
    orig_result = evaluate_model(model, val_loader, device)
    orig_size = get_model_size(model)
    orig_time = measure_inference_time(model, device=device)
    
    # Save original model if not already saved
    orig_path = output_dir / "micropizzanetv2_base.pth"
    if not orig_path.exists():
        torch.save(model.state_dict(), orig_path)
        logger.info(f"Saved original model to {orig_path}")
    
    # Create results dictionary
    results = {
        "original": {
            "sparsity": 0.0,
            "parameters": model.count_parameters(),
            "model_size_kb": orig_size,
            "accuracy": orig_result['accuracy'],
            "inference_time_ms": orig_time['avg_time_ms'],
            "model_path": str(orig_path)
        }
    }
    
    # Measure tensor arena size for original model
    if not args.skip_tensor_arena:
        tensor_arena_result = measure_tensor_arena_size(str(orig_path))
        if tensor_arena_result and 'tensor_arena_size_bytes' in tensor_arena_result:
            results["original"]["tensor_arena_kb"] = tensor_arena_result['tensor_arena_size_bytes'] / 1024
    
    # For each sparsity rate
    for sparsity in args.sparsities:
        logger.info(f"\n{'='*80}\nProcessing sparsity {sparsity:.2f}\n{'='*80}")
        
        # Calculate layer importance scores and create pruned model
        logger.info("Creating pruned model")
        pruned_model = create_pruned_model(model, sparsity)
        
        # Validate that the pruned model works
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 48, 48)
                _ = pruned_model(dummy_input)
            logger.info("Pruned model forward pass succeeded")
        except Exception as e:
            logger.error(f"Pruned model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Save model before finetuning
        pruned_dir = output_dir / "pruned"
        pruned_dir.mkdir(exist_ok=True, parents=True)
        pruned_path = pruned_dir / f"micropizzanetv2_pruned_s{int(sparsity*100)}.pth"
        torch.save(pruned_model.state_dict(), pruned_path)
        
        # Evaluate model before finetuning
        logger.info("Evaluating pruned model before finetuning")
        pruned_result_before = evaluate_model(pruned_model, val_loader, device)
        pruned_size = get_model_size(pruned_model)
        pruned_time_before = measure_inference_time(pruned_model, device=device)
        
        # Finetune model
        logger.info("Finetuning pruned model")
        finetuned_model, history = finetune_model(
            pruned_model, train_loader, val_loader, 
            epochs=args.finetune_epochs, 
            lr=args.finetune_lr,
            device=device
        )
        
        # Save finetuned model
        finetuned_path = pruned_dir / f"micropizzanetv2_pruned_finetuned_s{int(sparsity*100)}.pth"
        torch.save(finetuned_model.state_dict(), finetuned_path)
        
        # Evaluate model after finetuning
        logger.info("Evaluating pruned model after finetuning")
        pruned_result_after = evaluate_model(finetuned_model, val_loader, device)
        pruned_time_after = measure_inference_time(finetuned_model, device=device)
        
        # Store results
        results[f"pruned_s{int(sparsity*100)}"] = {
            "sparsity": sparsity,
            "parameters": pruned_model.count_parameters(),
            "model_size_kb": pruned_size,
            "accuracy_before_finetuning": pruned_result_before['accuracy'],
            "inference_time_before_ms": pruned_time_before['avg_time_ms'],
            "accuracy_after_finetuning": pruned_result_after['accuracy'],
            "inference_time_after_ms": pruned_time_after['avg_time_ms'],
            "model_path": str(finetuned_path)
        }
        
        # Measure tensor arena size
        if not args.skip_tensor_arena:
            tensor_arena_result = measure_tensor_arena_size(str(finetuned_path))
            if tensor_arena_result and 'tensor_arena_size_bytes' in tensor_arena_result:
                results[f"pruned_s{int(sparsity*100)}"]["tensor_arena_kb"] = tensor_arena_result['tensor_arena_size_bytes'] / 1024
        
        # Quantize model (if not skipped)
        if not args.skip_quantization:
            logger.info("Quantizing pruned model")
            try:
                quantized_model = quantize_model(finetuned_model, train_loader)
                
                # Save quantized model
                quantized_dir = output_dir / "quantized"
                quantized_dir.mkdir(exist_ok=True, parents=True)
                quantized_path = quantized_dir / f"micropizzanetv2_quantized_s{int(sparsity*100)}.pth"
                torch.save(quantized_model.state_dict(), quantized_path)
                
                # Evaluate quantized model
                logger.info("Evaluating quantized model")
                quantized_result = evaluate_model(quantized_model, val_loader, device)
                quantized_size = get_model_size(quantized_model)
                quantized_time = measure_inference_time(quantized_model, device=device)
                
                # Store quantized results
                results[f"quantized_s{int(sparsity*100)}"] = {
                    "sparsity": sparsity,
                    "parameters": quantized_model.count_parameters(),
                    "model_size_kb": quantized_size,
                    "accuracy": quantized_result['accuracy'],
                    "inference_time_ms": quantized_time['avg_time_ms'],
                    "model_path": str(quantized_path)
                }
                
                # Measure tensor arena size for quantized model
                if not args.skip_tensor_arena:
                    tensor_arena_result = measure_tensor_arena_size(str(quantized_path))
                    if tensor_arena_result and 'tensor_arena_size_bytes' in tensor_arena_result:
                        results[f"quantized_s{int(sparsity*100)}"]["tensor_arena_kb"] = tensor_arena_result['tensor_arena_size_bytes'] / 1024
            
            except Exception as e:
                logger.error(f"Error during quantization: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    results_path = output_dir / "pruning_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Create visualization
    create_visualization(results, output_dir)
    
    # Print summary as table
    logger.info("\n===== PRUNING EVALUATION SUMMARY =====")
    
    # Create a pandas DataFrame for nice table display
    summary_data = []
    
    # Add original row
    summary_data.append({
        "Model": "Original",
        "Sparsity": f"0%",
        "Parameters": f"{results['original']['parameters']:,}",
        "Size (KB)": f"{results['original']['model_size_kb']:.2f}",
        "Accuracy (%)": f"{results['original']['accuracy']:.2f}",
        "Inf. Time (ms)": f"{results['original']['inference_time_ms']:.2f}",
        "RAM (KB)": f"{results['original'].get('tensor_arena_kb', 'N/A')}"
    })
    
    # Add pruned models
    for sparsity in args.sparsities:
        key = f"pruned_s{int(sparsity*100)}"
        if key in results:
            summary_data.append({
                "Model": f"Pruned {int(sparsity*100)}%",
                "Sparsity": f"{int(sparsity*100)}%",
                "Parameters": f"{results[key]['parameters']:,}",
                "Size (KB)": f"{results[key]['model_size_kb']:.2f}",
                "Accuracy (%)": f"{results[key]['accuracy_after_finetuning']:.2f}",
                "Inf. Time (ms)": f"{results[key]['inference_time_after_ms']:.2f}",
                "RAM (KB)": f"{results[key].get('tensor_arena_kb', 'N/A')}"
            })
    
    # Add quantized models
    if not args.skip_quantization:
        for sparsity in args.sparsities:
            key = f"quantized_s{int(sparsity*100)}"
            if key in results:
                summary_data.append({
                    "Model": f"Quantized {int(sparsity*100)}%",
                    "Sparsity": f"{int(sparsity*100)}%",
                    "Parameters": f"{results[key]['parameters']:,}",
                    "Size (KB)": f"{results[key]['model_size_kb']:.2f}",
                    "Accuracy (%)": f"{results[key]['accuracy']:.2f}",
                    "Inf. Time (ms)": f"{results[key]['inference_time_ms']:.2f}",
                    "RAM (KB)": f"{results[key].get('tensor_arena_kb', 'N/A')}"
                })
    
    # Create and print the summary table
    summary_df = pd.DataFrame(summary_data)
    table = tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False)
    logger.info(f"\n{table}")
    
    # Save summary table to file
    with open(output_dir / "pruning_summary.txt", "w") as f:
        f.write(table)
    
    logger.info("Pruning evaluation completed!")
    
    # Recommend best model based on accuracy and size
    best_model = None
    best_score = -float('inf')
    
    for key, data in results.items():
        if key == 'original':
            continue
        
        if 'accuracy_after_finetuning' in data:  # Pruned model
            accuracy = data['accuracy_after_finetuning']
        else:  # Quantized model
            accuracy = data['accuracy']
        
        # Score based on accuracy and model size (you can adjust the weights)
        # Higher accuracy and lower size are better
        accuracy_weight = 0.7
        size_weight = 0.3
        
        orig_accuracy = results['original']['accuracy']
        orig_size = results['original']['model_size_kb']
        
        # Normalized values (percentage of original)
        norm_accuracy = accuracy / orig_accuracy
        norm_size = 1 - (data['model_size_kb'] / orig_size)  # Lower is better, so invert
        
        score = (accuracy_weight * norm_accuracy) + (size_weight * norm_size)
        
        if score > best_score:
            best_score = score
            best_model = key
    
    if best_model:
        logger.info(f"\nRECOMMENDED MODEL: {best_model}")
        if 'quantized' in best_model:
            logger.info(f"Accuracy: {results[best_model]['accuracy']:.2f}%")
        else:
            logger.info(f"Accuracy: {results[best_model]['accuracy_after_finetuning']:.2f}%")
        logger.info(f"Model Size: {results[best_model]['model_size_kb']:.2f} KB")
        logger.info(f"Inference Time: {results[best_model]['inference_time_after_ms' if 'pruned' in best_model else 'inference_time_ms']:.2f} ms")
        logger.info(f"Model Path: {results[best_model]['model_path']}")

if __name__ == "__main__":
    main()
