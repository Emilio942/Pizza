#!/usr/bin/env python3
"""
Train an improved MicroPizzaNet with Early Exit
Implements dynamic inference for RP2040 to save power

This implementation adds an early exit branch after the second block
of the MicroPizzaNet architecture that allows early classification 
when the confidence threshold is met, saving computation time.

Improvements:
- Class weighting for imbalanced dataset
- Increased regularization
- Improved early exit path design
- Better loss balancing with higher lambda_ee
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import logging
import json
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    RP2040Config, create_optimized_dataloaders
)
from scripts.early_exit.improved_early_exit import (
    ImprovedMicroPizzaNetWithEarlyExit, train_improved_early_exit_model
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_early_exit_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_class_weights(weights_path):
    """Load class weights from JSON file"""
    if not os.path.exists(weights_path):
        logger.warning(f"Class weights file {weights_path} not found.")
        return None
        
    try:
        with open(weights_path, 'r') as f:
            data = json.load(f)
            
        if 'class_weights_list' in data:
            weights = data['class_weights_list']
            return torch.FloatTensor(weights)
        else:
            logger.warning("No class_weights_list found in weights file.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading class weights: {e}")
        return None

def main(args):
    """Main function to train and evaluate the improved early exit model"""
    logger.info("Starting improved early exit model training")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(project_root, "output", "model_optimization", "improved_early_exit")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load class weights if available
    class_weights = None
    if args.use_class_weights:
        if args.weights_path:
            class_weights = load_class_weights(args.weights_path)
        else:
            # Generate weights file using analyze_dataset_early_exit.py
            logger.info("No weights file provided. Running dataset analysis...")
            from analyze_dataset_early_exit import analyze_dataset
            analyze_dataset(args.data_dir)
            class_weights = load_class_weights('class_weights.json')
        
        if class_weights is not None:
            logger.info(f"Using class weights: {class_weights}")
    
    # Create configuration
    config = RP2040Config(data_dir=args.data_dir)
    
    # Create data loaders
    logger.info(f"Creating data loaders with data directory: {args.data_dir}")
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    logger.info(f"Data loaders created with {len(class_names)} classes: {class_names}")
    
    # Initialize model
    model = ImprovedMicroPizzaNetWithEarlyExit(
        num_classes=len(class_names), 
        dropout_rate=args.dropout_rate, 
        confidence_threshold=args.confidence_threshold
    )
    
    logger.info(f"Model initialized with {model.count_parameters()} parameters")
    logger.info(f"Confidence threshold: {args.confidence_threshold}")
    
    # Train the model
    if args.mode in ['train', 'both']:
        logger.info("Starting model training...")
        history, model = train_improved_early_exit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            class_names=class_names,
            epochs=args.epochs,
            early_stopping_patience=args.patience,
            lambda_ee=args.lambda_ee,
            model_name=args.model_name,
            class_weights=class_weights,
            output_dir=output_dir
        )
        
        logger.info("Training completed")
    
    # Evaluate the model
    if args.mode in ['evaluate', 'both']:
        logger.info("Starting model evaluation...")
        
        # If in evaluate-only mode, load the pre-trained model
        if args.mode == 'evaluate':
            model_path = os.path.join(project_root, "models_optimized", f"{args.model_name}.pth")
            if os.path.exists(model_path):
                try:
                    # First attempt: direct loading
                    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
                    logger.info(f"Loaded pre-trained model from {model_path}")
                except Exception as e:
                    logger.warning(f"Error loading model directly: {e}")
                    logger.info("Attempting to use model adapter...")
                    
                    # Use the adapter to load the model
                    from scripts.early_exit.model_adapter import load_model_with_compatibility
                    model = load_model_with_compatibility(
                        model_path, 
                        num_classes=len(class_names), 
                        device=config.DEVICE
                    )
                    logger.info("Model loaded using adapter")
            else:
                logger.error(f"Model file {model_path} not found.")
                return
        
        # Run script to evaluate the model with forced early exit rates
        logger.info("Running forced early exit evaluation...")
        from scripts.early_exit.unified_evaluate_forced_early_exit import run_forced_evaluation
        
        model_path = os.path.join(project_root, "models_optimized", f"{args.model_name}.pth")
        run_forced_evaluation(
            model_path=model_path,
            data_dir=args.data_dir,
            batch_size=config.BATCH_SIZE,
            device=str(config.DEVICE),
            output_dir=os.path.join(output_dir, "evaluation")
        )
        
        logger.info("Evaluation completed. See logs and output files for results.")
    
    logger.info("Process completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate improved MicroPizzaNet with Early Exit")
    
    parser.add_argument("--mode", type=str, choices=['train', 'evaluate', 'both'], default='both',
                       help="Operation mode: train, evaluate, or both")
    parser.add_argument("--data-dir", type=str, default="data/augmented",
                       help="Directory containing the dataset")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--lambda-ee", type=float, default=0.5,
                       help="Weight of early exit loss (0-1)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for early exit")
    parser.add_argument("--dropout-rate", type=float, default=0.3,
                       help="Dropout rate for regularization")
    parser.add_argument("--use-class-weights", action="store_true",
                       help="Use class weights for handling imbalanced data")
    parser.add_argument("--weights-path", type=str, default="class_weights.json",
                       help="Path to class weights JSON file")
    parser.add_argument("--model-name", type=str, default="improved_early_exit",
                       help="Name for the saved model")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    main(args)
