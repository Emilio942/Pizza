#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced batch verification script for comparing formal properties
across different MicroPizzaNet model architectures and versions.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import time
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Try to import the formal verification framework
# If auto_LiRPA is not available, use the mock implementation
try:
    from models.formal_verification.formal_verification import (
        ModelVerifier, 
        VerificationProperty,
        VerificationResult,
        load_model_for_verification,
        CLASS_NAMES,
        VERIFICATION_DEPENDENCIES_INSTALLED
    )
    
    # Import model
    from src.pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE
    
    if not VERIFICATION_DEPENDENCIES_INSTALLED:
        raise ImportError("auto_LiRPA is not installed")
        
except ImportError:
    print("Using mock verification framework since auto_LiRPA is not installed")
    # Use absolute import path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mock_verification import (
        ModelVerifier, 
        VerificationProperty,
        VerificationResult,
        load_model_for_verification,
        CLASS_NAMES
    )

def load_test_set(data_dir, num_samples_per_class=3, img_size=48):
    """
    Load a balanced test set for verification.
    
    Args:
        data_dir: Directory containing image data
        num_samples_per_class: Number of samples to load per class
        img_size: Size to resize images to
        
    Returns:
        Dictionary with class names as keys and lists of images as values,
        along with a list of all images and their true classes
    """
    class_images = {class_name: [] for class_name in CLASS_NAMES}
    all_images = []
    all_labels = []
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Load a balanced set of images from each class
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping.")
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith('.jpg') or f.endswith('.png')]
        
        # Select a subset of images or use all if fewer than requested
        selected_files = image_files[:num_samples_per_class] if num_samples_per_class > 0 else image_files
        
        for img_file in selected_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).numpy()
                
                class_images[class_name].append(img_tensor)
                all_images.append(img_tensor)
                all_labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(all_images)} images for verification")
    for class_name, images in class_images.items():
        print(f"  Class {class_name}: {len(images)} images")
    
    return class_images, all_images, all_labels

def verify_model(model_info, test_images, test_labels, config):
    """
    Verify a single model against the test set.
    
    Args:
        model_info: Dictionary with model information
        test_images: List of test images
        test_labels: List of test labels
        config: Configuration dictionary
        
    Returns:
        Verification results and timing information
    """
    model_path = model_info['path']
    model_type = model_info['type']
    model_name = model_info['name']
    
    print(f"\nVerifying model: {model_name}")
    
    # Load model
    try:
        model = load_model_for_verification(
            model_path=model_path,
            model_type=model_type,
            num_classes=len(CLASS_NAMES),
            device=config['device']
        )
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Creating a new model for demonstration purposes")
        if model_type == 'MicroPizzaNet':
            model = MicroPizzaNet(num_classes=len(CLASS_NAMES))
        elif model_type == 'MicroPizzaNetV2':
            model = MicroPizzaNetV2(num_classes=len(CLASS_NAMES))
        elif model_type == 'MicroPizzaNetWithSE':
            model = MicroPizzaNetWithSE(num_classes=len(CLASS_NAMES))
    
    # Create verifier
    verifier = ModelVerifier(
        model=model,
        input_size=(config['img_size'], config['img_size']),
        device=config['device'],
        epsilon=config['epsilon'],
        norm_type=config['norm_type'],
        verify_backend=config['backend']
    )
    
    # Select a subset of images for verification
    if config['max_images'] > 0 and len(test_images) > config['max_images']:
        selected_indices = np.random.choice(
            len(test_images), config['max_images'], replace=False)
        verification_images = [test_images[i] for i in selected_indices]
        verification_labels = [test_labels[i] for i in selected_indices]
    else:
        verification_images = test_images
        verification_labels = test_labels
    
    # Time the verification process
    start_time = time.time()
    
    # Verify all properties
    all_results = verifier.verify_all_properties(
        input_images=verification_images,
        true_classes=verification_labels,
        critical_class_pairs=config['critical_class_pairs'],
        robustness_eps=config['epsilon'],
        brightness_range=config['brightness_range']
    )
    
    verification_time = time.time() - start_time
    
    # Generate verification report
    report_filename = f"verification_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(config['output_dir'], report_filename)
    report = verifier.generate_verification_report(all_results, output_path=report_path)
    
    print(f"Verification completed for {model_name} in {verification_time:.2f} seconds")
    print(f"Report saved to: {report_path}")
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'report': report,
        'verification_time': verification_time,
        'report_path': report_path
    }

def compare_models(verification_results):
    """
    Compare verification results across different models.
    
    Args:
        verification_results: List of verification results
        
    Returns:
        Comparison data and visualizations
    """
    # Extract data for comparison
    comparison_data = []
    
    for result in verification_results:
        model_name = result['model_name']
        report = result['report']
        
        # Extract overall metrics
        overall_rate = report['summary']['overall_verification_rate']
        total_properties = report['summary']['total_properties_checked']
        verified_properties = report['summary']['total_verified']
        verification_time = result['verification_time']
        
        # Extract property-specific metrics
        property_data = {}
        for prop_name, prop_info in report['properties'].items():
            property_data[f"{prop_name}_rate"] = prop_info['verification_rate']
            property_data[f"{prop_name}_time"] = prop_info['avg_time']
        
        # Combine data
        model_data = {
            'model_name': model_name,
            'model_type': result['model_type'],
            'overall_rate': overall_rate,
            'total_properties': total_properties,
            'verified_properties': verified_properties,
            'verification_time': verification_time,
            **property_data
        }
        
        comparison_data.append(model_data)
    
    # Convert to DataFrame for easier analysis
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df

def visualize_results(comparison_df, output_dir):
    """
    Create visualizations of verification results.
    
    Args:
        comparison_df: DataFrame with comparison data
        output_dir: Directory to save visualizations
        
    Returns:
        Paths to visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall verification rate by model
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_df['model_name'], comparison_df['overall_rate'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Model')
    plt.ylabel('Verification Rate')
    plt.title('Overall Verification Rate by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    overall_rate_path = os.path.join(output_dir, 'overall_verification_rate.png')
    plt.savefig(overall_rate_path)
    
    # 2. Property-specific verification rates
    property_columns = [col for col in comparison_df.columns if col.endswith('_rate') and col != 'overall_rate']
    if property_columns:
        plt.figure(figsize=(12, 8))
        
        # Set width of bars
        barWidth = 0.2
        
        # Set positions of bars on X axis
        r = np.arange(len(comparison_df))
        
        # Make the plot
        for i, property_col in enumerate(property_columns):
            property_name = property_col.split('_rate')[0]
            plt.bar(r + i*barWidth, comparison_df[property_col], 
                    width=barWidth, label=property_name)
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('Verification Rate')
        plt.title('Verification Rate by Property and Model')
        plt.xticks(r + barWidth*(len(property_columns)-1)/2, comparison_df['model_name'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        property_rate_path = os.path.join(output_dir, 'property_verification_rates.png')
        plt.savefig(property_rate_path)
    
    # 3. Verification time by model
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_df['model_name'], comparison_df['verification_time'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Model')
    plt.ylabel('Verification Time (seconds)')
    plt.title('Total Verification Time by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    time_path = os.path.join(output_dir, 'verification_time.png')
    plt.savefig(time_path)
    
    return {
        'overall_rate': overall_rate_path,
        'property_rates': property_rate_path if property_columns else None,
        'verification_time': time_path
    }

def export_report(comparison_df, visualization_paths, output_dir):
    """
    Generate a comprehensive report of the verification results.
    
    Args:
        comparison_df: DataFrame with comparison data
        visualization_paths: Paths to visualization files
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    report_path = os.path.join(output_dir, 'model_verification_comparison.html')
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pizza Model Formal Verification Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .highlight {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>MicroPizzaNet Formal Verification Comparison</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <div class="highlight">
            <p>This report compares the formal verification properties of different MicroPizzaNet models.</p>
            <p>Higher verification rates indicate better formal guarantees for the properties tested.</p>
        </div>
        
        <h2>Model Comparison</h2>
        {comparison_df.to_html(index=False, float_format=lambda x: f'{x:.2%}' if 'rate' in str(x) else f'{x:.2f}' if isinstance(x, float) else x)}
        
        <h2>Visualizations</h2>
        
        <h3>Overall Verification Rate</h3>
        <img src="{os.path.basename(visualization_paths['overall_rate'])}" alt="Overall Verification Rate">
        
    """
    
    if visualization_paths.get('property_rates'):
        html_content += f"""
        <h3>Property-Specific Verification Rates</h3>
        <img src="{os.path.basename(visualization_paths['property_rates'])}" alt="Property Verification Rates">
        """
    
    html_content += f"""
        <h3>Verification Time</h3>
        <img src="{os.path.basename(visualization_paths['verification_time'])}" alt="Verification Time">
        
        <h2>Conclusions</h2>
        <div class="highlight">
            <p>Best performing model by verification rate: {comparison_df.loc[comparison_df['overall_rate'].idxmax()]['model_name']} ({comparison_df['overall_rate'].max():.2%})</p>
            <p>Fastest model to verify: {comparison_df.loc[comparison_df['verification_time'].idxmin()]['model_name']} ({comparison_df['verification_time'].min():.2f} seconds)</p>
        </div>
        
        <h2>Notes on Formal Verification</h2>
        <p>Formal verification provides mathematical guarantees that neural networks satisfy certain properties, unlike
        traditional testing which only checks specific inputs. The verification properties tested include:</p>
        <ul>
            <li><strong>Robustness</strong>: The model maintains the same prediction when inputs are perturbed slightly.</li>
            <li><strong>Brightness Invariance</strong>: The model gives consistent predictions across different brightness levels.</li>
            <li><strong>Class Separation</strong>: Critical classes (e.g., raw vs. burnt) are never confused.</li>
        </ul>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Copy visualization files to the output directory
    for viz_path in visualization_paths.values():
        if viz_path:
            viz_name = os.path.basename(viz_path)
            # Files should already be in the output directory, but just in case
            if os.path.dirname(viz_path) != output_dir:
                import shutil
                shutil.copy(viz_path, os.path.join(output_dir, viz_name))
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Batch verification of pizza detection models')
    parser.add_argument('--data-dir', default=os.path.join(project_root, 'augmented_pizza'),
                        help='Directory containing test images')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'models', 'formal_verification', 'reports'),
                        help='Directory to save verification reports')
    parser.add_argument('--max-images', type=int, default=5,
                        help='Maximum number of images to verify per model')
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help='Perturbation size for robustness verification')
    parser.add_argument('--backend', default='crown',
                        help='Verification backend (crown or beta-crown)')
    parser.add_argument('--device', default='cpu',
                        help='Device to run verification on (cpu or cuda)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define configuration
    config = {
        'img_size': 48,
        'epsilon': args.epsilon,
        'norm_type': 'L_inf',
        'brightness_range': (0.7, 1.3),
        'critical_class_pairs': [(0, 1), (2, 5)],  # Example: raw vs. burnt, combined vs. segment
        'device': args.device,
        'max_images': args.max_images,
        'backend': args.backend,
        'output_dir': args.output_dir
    }
    
    # Define models to verify
    models = [
        {
            'name': 'MicroPizzaNet_Base',
            'type': 'MicroPizzaNet',
            'path': os.path.join(project_root, 'models', 'pizza_model_float32.pth')
        },
        {
            'name': 'MicroPizzaNetV2_InvertedRes',
            'type': 'MicroPizzaNetV2',
            'path': os.path.join(project_root, 'models', 'pizza_model_v2.pth')
        },
        {
            'name': 'MicroPizzaNet_WithSE',
            'type': 'MicroPizzaNetWithSE',
            'path': os.path.join(project_root, 'models', 'pizza_model_with_se.pth')
        }
    ]
    
    # Load test images
    _, test_images, test_labels = load_test_set(
        args.data_dir, 
        num_samples_per_class=max(2, args.max_images // len(CLASS_NAMES))
    )
    
    # Verify each model
    verification_results = []
    for model_info in models:
        try:
            result = verify_model(model_info, test_images, test_labels, config)
            verification_results.append(result)
        except Exception as e:
            print(f"Error verifying model {model_info['name']}: {e}")
    
    # Compare results
    if len(verification_results) > 1:
        print("\nComparing verification results across models...")
        comparison_df = compare_models(verification_results)
        
        # Visualize comparison
        visualization_paths = visualize_results(comparison_df, args.output_dir)
        
        # Generate comparison report
        report_path = export_report(comparison_df, visualization_paths, args.output_dir)
        print(f"\nComparison report saved to: {report_path}")
    else:
        print("\nNot enough models verified for comparison")

if __name__ == "__main__":
    main()
