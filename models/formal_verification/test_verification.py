#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the formal verification framework.
This script tests both the mock implementation and the real implementation 
(if auto_LiRPA is available).
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("verification_test")

# Try to import the real implementation first
try:
    from models.formal_verification.formal_verification import (
        ModelVerifier as RealVerifier,
        VerificationProperty,
        VerificationResult,
        load_model_for_verification,
        CLASS_NAMES,
        VERIFICATION_DEPENDENCIES_INSTALLED
    )
    
    logger.info("Using real verification framework.")
    
    if not VERIFICATION_DEPENDENCIES_INSTALLED:
        logger.warning("auto_LiRPA is not installed. Will use mock implementation.")
        raise ImportError("auto_LiRPA is not installed")
        
except ImportError:
    logger.warning("Falling back to mock verification framework.")
    # Use absolute import path for mock implementation
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mock_verification import (
        ModelVerifier as MockVerifier,
        VerificationProperty,
        VerificationResult,
        load_model_for_verification,
        CLASS_NAMES
    )
    
    VERIFICATION_DEPENDENCIES_INSTALLED = False


def load_test_images(
    data_dir: str = None,
    max_images: int = 3
) -> Tuple[List[np.ndarray], List[int]]:
    """Load test images for verification testing."""
    from PIL import Image
    import random
    import torchvision.transforms as transforms
    
    if data_dir is None:
        data_dir = os.path.join(project_root, "augmented_pizza")
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return [], []
    
    # Transform for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    
    images = []
    labels = []
    
    # Load images from each class directory
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        # List image files in the directory
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select random images, up to max_per_class
        max_per_class = max(1, max_images // len(CLASS_NAMES))
        selected_files = random.sample(image_files, 
                                      min(max_per_class, len(image_files)))
        
        for img_file in selected_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                img_np = img_tensor.numpy()  # Shape: (C, H, W)
                
                images.append(img_np)
                labels.append(i)
                
                logger.info(f"Loaded image {img_path}")
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {str(e)}")
    
    return images, labels


def verify_with_implementation(
    implementation_name: str,
    model_path: str,
    model_type: str,
    images: List[np.ndarray],
    labels: List[int],
    epsilon: float = 0.03,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    output_dir: str = "reports",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Run verification with the specified implementation."""
    start_time = time.time()
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = load_model_for_verification(
        model_path=model_path,
        model_type=model_type,
        device=device
    )
    
    # Initialize verifier
    if implementation_name == "real" and VERIFICATION_DEPENDENCIES_INSTALLED:
        verifier = RealVerifier(
            model=model,
            input_size=(48, 48),
            device=device,
            epsilon=epsilon
        )
    else:
        verifier = MockVerifier(
            model=model,
            input_size=(48, 48),
            device=device,
            epsilon=epsilon
        )
    
    # Define critical class pairs for verification
    critical_pairs = [
        (0, 1),  # basic vs burnt
        (2, 5),  # combined vs segment
    ]
    
    # Verify all properties
    logger.info(f"Verifying all properties with {implementation_name} implementation")
    results = verifier.verify_all_properties(
        input_images=images,
        true_classes=labels,
        critical_class_pairs=critical_pairs,
        robustness_eps=epsilon,
        brightness_range=brightness_range
    )
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        output_dir, 
        f"verification_report_{implementation_name}_{model_type}_{timestamp}.json"
    )
    
    report = verifier.generate_verification_report(
        results=results,
        output_path=report_path
    )
    
    # Calculate timing
    total_time = time.time() - start_time
    logger.info(f"Verification completed in {total_time:.2f} seconds")
    
    # Print summary
    print_verification_summary(report)
    
    return report


def print_verification_summary(report: Dict[str, Any]) -> None:
    """Print a summary of the verification results."""
    print("\nVerification Summary:")
    print("-" * 40)
    
    summary = report["summary"]
    print(f"Model: {report['model_name']}")
    print(f"Total properties checked: {summary['total_properties_checked']}")
    print(f"Properties verified: {summary['total_verified']}/{summary['total_properties_checked']} "
          f"({summary['overall_verification_rate']*100:.1f}%)")
    print(f"Total verification time: {summary['total_time_seconds']:.2f} seconds")
    
    print("\nResults by property type:")
    for prop_name, prop_data in report["properties"].items():
        if prop_data["total"] > 0:
            print(f"  {prop_name.upper()}:")
            print(f"  - Verified: {prop_data['verified']}/{prop_data['total']} "
                  f"({prop_data['verification_rate']*100:.1f}%)")
            print(f"  - Average verification time: {prop_data['avg_time']:.2f} seconds")
    
    print("-" * 40)


def compare_implementations(
    real_report: Optional[Dict[str, Any]],
    mock_report: Dict[str, Any],
    output_dir: str = "reports"
) -> None:
    """Compare results between real and mock implementations."""
    if real_report is None:
        logger.warning("No real implementation results available for comparison")
        return
    
    # Create comparison report
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "real_model_name": real_report["model_name"],
        "mock_model_name": mock_report["model_name"],
        "property_comparison": {},
        "timing_comparison": {
            "real_total_time": real_report["summary"]["total_time_seconds"],
            "mock_total_time": mock_report["summary"]["total_time_seconds"],
            "time_ratio": (real_report["summary"]["total_time_seconds"] / 
                          max(0.001, mock_report["summary"]["total_time_seconds"]))
        }
    }
    
    # Compare property results
    for prop_name in real_report["properties"]:
        if prop_name in mock_report["properties"]:
            real_prop = real_report["properties"][prop_name]
            mock_prop = mock_report["properties"][prop_name]
            
            comparison["property_comparison"][prop_name] = {
                "real_verification_rate": real_prop["verification_rate"],
                "mock_verification_rate": mock_prop["verification_rate"],
                "agreement_rate": calculate_agreement_rate(
                    real_prop["details"], mock_prop["details"]
                ),
                "real_avg_time": real_prop["avg_time"],
                "mock_avg_time": mock_prop["avg_time"]
            }
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, 
        f"implementation_comparison_{timestamp}.json"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Implementation comparison saved to: {output_path}")
    
    # Print comparison summary
    print("\nImplementation Comparison:")
    print("-" * 40)
    print(f"Real implementation total time: {comparison['timing_comparison']['real_total_time']:.2f}s")
    print(f"Mock implementation total time: {comparison['timing_comparison']['mock_total_time']:.2f}s")
    print(f"Time ratio (real/mock): {comparison['timing_comparison']['time_ratio']:.2f}x")
    
    print("\nVerification rate comparison:")
    for prop_name, prop_comp in comparison["property_comparison"].items():
        print(f"  {prop_name.upper()}:")
        print(f"  - Real: {prop_comp['real_verification_rate']*100:.1f}%")
        print(f"  - Mock: {prop_comp['mock_verification_rate']*100:.1f}%")
        print(f"  - Agreement: {prop_comp['agreement_rate']*100:.1f}%")
    print("-" * 40)


def calculate_agreement_rate(real_details, mock_details) -> float:
    """Calculate the agreement rate between real and mock verification results."""
    if not real_details or not mock_details:
        return 0.0
    
    # Match details by index if possible
    if len(real_details) == len(mock_details):
        agreements = sum(
            1 for r, m in zip(real_details, mock_details)
            if r.get("verified") == m.get("verified")
        )
        return agreements / len(real_details)
    
    # If sizes don't match, return a default value
    return 0.5  # Cannot reliably calculate agreement


def main():
    parser = argparse.ArgumentParser(description="Test verification framework")
    parser.add_argument("--model-path", type=str, 
                       default=os.path.join(project_root, "models", "pizza_model_float32.pth"),
                       help="Path to the model file")
    parser.add_argument("--model-type", type=str, 
                       default="MicroPizzaNet",
                       choices=["MicroPizzaNet", "MicroPizzaNetV2", "MicroPizzaNetWithSE"],
                       help="Type of model to verify")
    parser.add_argument("--data-dir", type=str, 
                       default=os.path.join(project_root, "augmented_pizza"),
                       help="Directory containing test images")
    parser.add_argument("--max-images", type=int, default=6,
                       help="Maximum number of images to test")
    parser.add_argument("--epsilon", type=float, default=0.03,
                       help="Epsilon value for robustness verification")
    parser.add_argument("--output-dir", type=str, default="reports",
                       help="Directory to save reports")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"], help="Device to run verification on")
    parser.add_argument("--skip-real", action="store_true",
                       help="Skip real implementation even if available")
    parser.add_argument("--only-mock", action="store_true",
                       help="Use only mock implementation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test images
    images, labels = load_test_images(
        data_dir=args.data_dir,
        max_images=args.max_images
    )
    
    if not images:
        logger.error("No test images loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(images)} test images")
    
    # Run verification with available implementations
    real_report = None
    mock_report = None
    
    # Real implementation
    if VERIFICATION_DEPENDENCIES_INSTALLED and not args.skip_real and not args.only_mock:
        try:
            logger.info("Testing with real verification implementation")
            real_report = verify_with_implementation(
                implementation_name="real",
                model_path=args.model_path,
                model_type=args.model_type,
                images=images,
                labels=labels,
                epsilon=args.epsilon,
                output_dir=args.output_dir,
                device=args.device
            )
        except Exception as e:
            logger.error(f"Error with real implementation: {str(e)}")
    
    # Mock implementation
    try:
        logger.info("Testing with mock verification implementation")
        mock_report = verify_with_implementation(
            implementation_name="mock",
            model_path=args.model_path,
            model_type=args.model_type,
            images=images,
            labels=labels,
            epsilon=args.epsilon,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Error with mock implementation: {str(e)}")
    
    # Compare implementations if both reports are available
    if real_report and mock_report:
        compare_implementations(
            real_report=real_report,
            mock_report=mock_report,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
