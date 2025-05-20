#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI/CD Integration for Formal Verification

This script integrates formal verification into the CI/CD pipeline,
verifying key model properties before deployment.
"""

import os
import sys
import argparse
import numpy as np
import json
import logging
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import verification dependencies
try:
    import torch
    from auto_LiRPA import BoundedModule
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

# Import verification module
if VERIFICATION_AVAILABLE:
    from models.formal_verification.formal_verification import (
        ModelVerifier, VerificationProperty, load_model_for_verification
    )
from src.constants import CLASS_NAMES, PROJECT_ROOT, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ci_verify")

def parse_args():
    """Parse command line arguments for CI verification."""
    parser = argparse.ArgumentParser(
        description='Run formal verification as part of CI/CD pipeline'
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the model file (.pth)')
    
    parser.add_argument('--model-type', type=str, default='MicroPizzaNet',
                      choices=['MicroPizzaNet', 'MicroPizzaNetV2', 'MicroPizzaNetWithSE'],
                      help='Type of model to verify')
    
    parser.add_argument('--verification-data', type=str, required=True,
                      help='Path to verification dataset (stored as .npz)')
    
    parser.add_argument('--report-path', type=str, 
                      default=os.path.join(OUTPUT_DIR, 'verification_report.json'),
                      help='Path to save verification report')
    
    parser.add_argument('--verification-threshold', type=float, default=0.7,
                      help='Minimum acceptable verification rate (0.0-1.0)')
    
    parser.add_argument('--epsilon', type=float, default=0.01,
                      help='Epsilon for robustness verification')
    
    parser.add_argument('--brightness-range', type=str, default='0.8,1.2',
                      help='Brightness range for verification (min,max)')
    
    parser.add_argument('--ci-mode', action='store_true',
                      help='Run in CI mode (fail on verification threshold)')
    
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU for verification if available')
    
    return parser.parse_args()

def prepare_verification_data(data_path):
    """Load and prepare verification data from numpy file."""
    try:
        data = np.load(data_path, allow_pickle=True)
        
        # Check if it's a .npz file with multiple arrays
        if isinstance(data, np.lib.npyio.NpzFile):
            images = data['images']
            classes = data['classes']
            pairs = data.get('critical_pairs', [(0, 2)])  # Default pairs if not provided
        else:
            # If it's a single .npy file, it should be a dictionary
            data_dict = data.item()
            images = data_dict.get('images')
            classes = data_dict.get('classes')
            pairs = data_dict.get('critical_pairs', [(0, 2)])
        
        if len(images) != len(classes):
            raise ValueError("Number of images and classes must match")
            
        return {
            'images': images,
            'classes': classes,
            'critical_pairs': pairs
        }
    except Exception as e:
        logger.error(f"Error loading verification data: {e}")
        raise

def run_verification(args):
    """Run formal verification as part of CI/CD pipeline."""
    if not VERIFICATION_AVAILABLE:
        logger.error(
            "Formal verification dependencies not available. "
            "Install with: pip install auto_LiRPA"
        )
        return False, None
        
    start_time = time.time()
    
    # Determine device for verification
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info("Using GPU for verification")
    else:
        logger.info("Using CPU for verification")
        
    try:
        # Load model for verification
        logger.info(f"Loading model from {args.model_path}")
        model = load_model_for_verification(
            model_path=args.model_path,
            model_type=args.model_type,
            num_classes=len(CLASS_NAMES),
            device=device
        )
        
        # Create verifier
        logger.info(f"Creating verifier with epsilon={args.epsilon}")
        verifier = ModelVerifier(
            model=model,
            input_size=(48, 48),  # Standard size for the project
            device=device,
            epsilon=args.epsilon
        )
        
        # Load verification data
        logger.info(f"Loading verification data from {args.verification_data}")
        ver_data = prepare_verification_data(args.verification_data)
        
        images = ver_data['images']
        classes = ver_data['classes']
        critical_pairs = ver_data['critical_pairs']
        
        logger.info(f"Loaded {len(images)} images for verification")
        logger.info(f"Critical class pairs: {critical_pairs}")
        
        # Parse brightness range
        brightness_min, brightness_max = map(float, args.brightness_range.split(','))
        
        # Run verification on all properties
        logger.info("Starting comprehensive verification...")
        results = verifier.verify_all_properties(
            input_images=images,
            true_classes=classes,
            critical_class_pairs=critical_pairs,
            robustness_eps=args.epsilon,
            brightness_range=(brightness_min, brightness_max)
        )
        
        # Generate verification report
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        report = verifier.generate_verification_report(
            results=results,
            output_path=args.report_path
        )
        
        # Calculate verification rate
        verification_rate = report['summary']['overall_verification_rate']
        verified_count = report['summary']['total_verified']
        total_count = report['summary']['total_properties_checked']
        
        logger.info(f"Verification completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Verification rate: {verification_rate:.2%} ({verified_count}/{total_count})")
        logger.info(f"Verification threshold: {args.verification_threshold:.2%}")
        logger.info(f"Verification report saved to {args.report_path}")
        
        # Check if verification passed threshold
        passed = verification_rate >= args.verification_threshold
        
        if passed:
            logger.info("Verification PASSED ✓")
        else:
            logger.warning("Verification FAILED ✗")
            
        return passed, report
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

def main():
    """Main function for CI verification."""
    args = parse_args()
    
    logger.info("Starting formal verification in CI/CD pipeline")
    passed, report = run_verification(args)
    
    if args.ci_mode and not passed:
        logger.error("Verification failed to meet threshold in CI mode. Exiting with error.")
        sys.exit(1)
        
    if report:
        # Print detailed results for each property
        for prop_name, prop_data in report['properties'].items():
            verified = prop_data['verified']
            total = prop_data['total']
            rate = prop_data['verification_rate']
            logger.info(f"Property {prop_name}: {verified}/{total} verified ({rate:.2%})")
    
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()
