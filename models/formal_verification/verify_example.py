#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for using the formal verification framework on the pizza detection model.
This demonstrates how to verify various properties of the model.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
import json
import logging
from PIL import Image

# Add project directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.formal_verification.formal_verification import (
    ModelVerifier, VerificationProperty, load_model_for_verification
)
from src.constants import CLASS_NAMES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_example")

def load_test_image(path, target_size=(48, 48)):
    """Load and preprocess a test image."""
    img = Image.open(path)
    img = img.resize(target_size)
    return np.array(img)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Verify properties of a pizza detection model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model file (.pth)')
    
    parser.add_argument('--model-type', type=str, default='MicroPizzaNet',
                        choices=['MicroPizzaNet', 'MicroPizzaNetV2', 'MicroPizzaNetWithSE'],
                        help='Type of model architecture')
    
    parser.add_argument('--input-size', type=int, default=48,
                        help='Input image size (square)')
    
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Epsilon value for robustness verification')
    
    parser.add_argument('--test-image', type=str,
                        help='Path to a test image for single verification')
    
    parser.add_argument('--test-class', type=int,
                        help='True class of the test image (0-5)')
    
    parser.add_argument('--test-dir', type=str,
                        help='Directory with test images organized by class folders')
    
    parser.add_argument('--property', type=str, 
                        choices=['robustness', 'brightness', 'class_separation', 'all'],
                        default='all',
                        help='Property to verify')
    
    parser.add_argument('--brightness-range', type=str, default='0.8,1.2',
                        help='Brightness range for invariance verification (min,max)')
    
    parser.add_argument('--output', type=str, default='verification_report.json',
                        help='Path to save the verification report')
    
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA for verification (if available)')
    
    return parser.parse_args()

def verify_single_image(verifier, img, true_class, property_type, args):
    """Run verification on a single image."""
    results = {}
    
    if property_type in ['robustness', 'all']:
        logger.info("Verifying robustness...")
        result = verifier.verify_robustness(img, true_class, epsilon=args.epsilon)
        results[VerificationProperty.ROBUSTNESS.value] = [result]
        logger.info(f"Robustness verified: {result.verified}")
    
    if property_type in ['brightness', 'all']:
        logger.info("Verifying brightness invariance...")
        brightness_min, brightness_max = map(float, args.brightness_range.split(','))
        result = verifier.verify_brightness_invariance(
            img, true_class, brightness_range=(brightness_min, brightness_max)
        )
        results[VerificationProperty.BRIGHTNESS_INVARIANCE.value] = [result]
        logger.info(f"Brightness invariance verified: {result.verified}")
    
    return results

def verify_multiple_images(verifier, test_dir, property_type, args):
    """Run verification on multiple images from the test directory."""
    # Load test images
    input_images = []
    true_classes = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(test_dir, class_name.lower())
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit to 3 images per class for the example
        for img_file in image_files[:3]:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = load_test_image(img_path, (args.input_size, args.input_size))
                input_images.append(img)
                true_classes.append(class_idx)
                logger.info(f"Loaded image {img_path} with class {class_idx} ({class_name})")
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
    
    if not input_images:
        logger.error("No test images found")
        return {}
        
    # Define critical class pairs (for class separation)
    # Example: raw (0) should never be confused with well-done (2)
    critical_pairs = [(0, 2)]
    
    # Parse brightness range
    brightness_min, brightness_max = map(float, args.brightness_range.split(','))
    
    # Run comprehensive verification
    logger.info(f"Starting verification of {len(input_images)} images...")
    
    if property_type == 'all':
        results = verifier.verify_all_properties(
            input_images=input_images,
            true_classes=true_classes,
            critical_class_pairs=critical_pairs,
            robustness_eps=args.epsilon,
            brightness_range=(brightness_min, brightness_max)
        )
    else:
        # Initialize results dictionary with empty lists
        results = {
            VerificationProperty.ROBUSTNESS.value: [],
            VerificationProperty.BRIGHTNESS_INVARIANCE.value: [],
            VerificationProperty.CLASS_SEPARATION.value: []
        }
        
        # Run specific property verification
        if property_type == 'robustness':
            for i, (img, cls) in enumerate(zip(input_images, true_classes)):
                logger.info(f"Verifying robustness for image {i+1}/{len(input_images)}")
                result = verifier.verify_robustness(img, cls, epsilon=args.epsilon)
                results[VerificationProperty.ROBUSTNESS.value].append(result)
                
        elif property_type == 'brightness':
            for i, (img, cls) in enumerate(zip(input_images, true_classes)):
                logger.info(f"Verifying brightness invariance for image {i+1}/{len(input_images)}")
                result = verifier.verify_brightness_invariance(
                    img, cls, brightness_range=(brightness_min, brightness_max)
                )
                results[VerificationProperty.BRIGHTNESS_INVARIANCE.value].append(result)
                
        elif property_type == 'class_separation':
            for class1, class2 in critical_pairs:
                examples_class1 = [img for img, cls in zip(input_images, true_classes) if cls == class1]
                if examples_class1:
                    logger.info(f"Verifying class separation for {CLASS_NAMES[class1]} and {CLASS_NAMES[class2]}")
                    result = verifier.verify_class_separation(
                        class1=class1, class2=class2, examples=examples_class1, robustness_eps=args.epsilon
                    )
                    results[VerificationProperty.CLASS_SEPARATION.value].append(result)
    
    return results

def main():
    """Main function to run verification."""
    args = parse_args()
    
    # Set device
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info("Using CUDA for verification")
    else:
        logger.info("Using CPU for verification")
    
    # Load the model
    logger.info(f"Loading model from {args.model_path}")
    try:
        model = load_model_for_verification(
            model_path=args.model_path,
            model_type=args.model_type,
            num_classes=len(CLASS_NAMES),
            device=device
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return 1
        
    # Create the verifier
    verifier = ModelVerifier(
        model=model,
        input_size=(args.input_size, args.input_size),
        device=device,
        epsilon=args.epsilon
    )
    
    # Run verification
    start_time = time.time()
    
    try:
        if args.test_image and args.test_class is not None:
            # Single image verification
            logger.info(f"Verifying single image: {args.test_image}")
            img = load_test_image(args.test_image, (args.input_size, args.input_size))
            results = verify_single_image(verifier, img, args.test_class, args.property, args)
        elif args.test_dir:
            # Multiple images verification
            logger.info(f"Verifying images from directory: {args.test_dir}")
            results = verify_multiple_images(verifier, args.test_dir, args.property, args)
        else:
            logger.error("Either --test-image and --test-class or --test-dir must be provided")
            return 1
            
        # Check if we have any results
        if not any(results.values()):
            logger.error("No verification results produced")
            return 1
            
        # Generate verification report
        report = verifier.generate_verification_report(
            results=results,
            output_path=args.output
        )
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    # Print summary
    total_time = time.time() - start_time
    logger.info(f"Verification completed in {total_time:.2f} seconds")
    
    # Get overall verification rate
    if report and 'summary' in report:
        verified = report['summary']['total_verified']
        total = report['summary']['total_properties_checked']
        rate = report['summary']['overall_verification_rate']
        
        logger.info(f"Overall verification results: {verified}/{total} properties verified ({rate:.2%})")
        logger.info(f"Report saved to {args.output}")
        
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
