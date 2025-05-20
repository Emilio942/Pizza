#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the formal verification framework
to verify properties of a trained MicroPizzaNet model.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
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

def load_sample_images(data_dir, num_samples=5, img_size=48):
    """
    Load a few sample images for verification.
    
    Args:
        data_dir: Directory containing image data
        num_samples: Number of samples to load per class
        img_size: Size to resize images to
        
    Returns:
        List of images and their true classes
    """
    images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Load a few images from each class
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping.")
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith('.jpg') or f.endswith('.png')][:num_samples]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).numpy()
                images.append(img_tensor)
                labels.append(class_idx)
                print(f"Loaded image {img_path}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, labels

def main():
    # Configuration
    model_path = os.path.join(project_root, "models", "pizza_model_float32.pth")
    data_dir = os.path.join(project_root, "augmented_pizza")
    
    # Parameters for verification
    epsilon = 0.03  # Perturbation size for robustness
    brightness_range = (0.7, 1.3)  # Range for brightness invariance
    
    # Critical class pairs for verification
    # For example, we never want to confuse "raw" (0) with "burnt" (1)
    critical_class_pairs = [(0, 1), (2, 5)]
    
    # 1. Load the model
    print("Loading model for verification...")
    try:
        model = load_model_for_verification(
            model_path=model_path,
            model_type='MicroPizzaNet',
            num_classes=len(CLASS_NAMES),
            device='cpu'
        )
        print(f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If model not found, create a simple one for demonstration
        print("Creating a new model for demonstration purposes")
        model = MicroPizzaNet(num_classes=len(CLASS_NAMES))
    
    # 2. Create the verifier
    verifier = ModelVerifier(
        model=model,
        input_size=(48, 48),
        device='cpu',
        epsilon=epsilon,
        norm_type='L_inf'
    )
    
    # 3. Load sample images for verification
    print("\nLoading sample images for verification...")
    sample_images, sample_labels = load_sample_images(data_dir)
    
    if not sample_images:
        print("No images loaded. Using random synthetic images for demonstration.")
        # Create synthetic images if no real ones are available
        sample_images = [np.random.rand(3, 48, 48).astype(np.float32) for _ in range(10)]
        sample_labels = [np.random.randint(0, len(CLASS_NAMES)) for _ in range(10)]
    
    # 4. Verify individual properties for demonstration
    
    # 4.1 Verify robustness for the first image
    print("\nVerifying robustness for a single image...")
    robustness_result = verifier.verify_robustness(
        input_image=sample_images[0],
        true_class=sample_labels[0]
    )
    print(robustness_result)
    
    # 4.2 Verify brightness invariance for the first image
    print("\nVerifying brightness invariance for a single image...")
    brightness_result = verifier.verify_brightness_invariance(
        input_image=sample_images[0],
        true_class=sample_labels[0],
        brightness_range=brightness_range
    )
    print(brightness_result)
    
    # 5. Verify all properties for all images
    print("\nVerifying all properties for all sample images...")
    print("This may take some time depending on the number of images and verification complexity.")
    
    # Limit to first 5 images if there are many
    verification_images = sample_images[:5]
    verification_labels = sample_labels[:5]
    
    # Verify all properties
    all_results = verifier.verify_all_properties(
        input_images=verification_images,
        true_classes=verification_labels,
        critical_class_pairs=critical_class_pairs,
        robustness_eps=epsilon,
        brightness_range=brightness_range
    )
    
    # 6. Generate and save a verification report
    print("\nGenerating verification report...")
    report_path = os.path.join(project_root, "models", "formal_verification", "verification_report.json")
    report = verifier.generate_verification_report(all_results, output_path=report_path)
    
    print(f"\nVerification report saved to: {report_path}")
    print("\nVerification summary:")
    
    # Print overall summary
    overall_verified = report['summary']['total_verified']
    overall_total = report['summary']['total_properties_checked']
    overall_rate = report['summary']['overall_verification_rate']
    
    print(f"- Properties verified: {overall_verified}/{overall_total} ({overall_rate*100:.1f}%)")
    print(f"- Total verification time: {report['summary']['total_time_seconds']:.2f} seconds")
    
    for prop_name, prop_data in report['properties'].items():
        print(f"\n{prop_name.upper()}:")
        print(f"- Verified: {prop_data['verified']}/{prop_data['total']} ({prop_data['verification_rate']*100:.1f}%)")
        print(f"- Average verification time: {prop_data['avg_time']:.2f} seconds")

if __name__ == "__main__":
    main()
