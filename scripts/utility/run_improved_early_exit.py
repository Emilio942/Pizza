#!/usr/bin/env python3
"""
Run the complete improved early exit model workflow:
1. Analyze dataset and get class weights
2. Train improved early exit model with class weighting
3. Evaluate model with forced early exit rates
4. Generate visualizations and metrics

This is a complete pipeline to implement dynamic inference for RP2040
to save power during inference.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_early_exit_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def run_command(cmd):
    """Run a shell command and log the output"""
    logger.info(f"Running command: {cmd}")
    return_code = os.system(cmd)
    if return_code != 0:
        logger.warning(f"Command exited with code {return_code}")
    return return_code

def main(args):
    """Run the complete improved early exit workflow"""
    logger.info("Starting improved early exit workflow")
    logger.info(f"Arguments: {args}")
    
    # Step 1: Analyze dataset and get class weights
    logger.info("Step 1: Analyzing dataset and generating class weights")
    analyze_cmd = (
        f"python analyze_dataset_early_exit.py --data-dir {args.data_dir}"
    )
    if run_command(analyze_cmd) != 0:
        logger.error("Dataset analysis failed. Aborting workflow.")
        return
    
    # Check if analysis succeeded
    if not os.path.exists('class_weights.json'):
        logger.error("Class weights file not generated. Aborting workflow.")
        return
    
    # Step 2: Train improved early exit model
    logger.info("Step 2: Training improved early exit model")
    train_cmd = (
        f"python scripts/train_improved_early_exit_model.py "
        f"--mode both "
        f"--data-dir {args.data_dir} "
        f"--epochs {args.epochs} "
        f"--patience {args.patience} "
        f"--lambda-ee {args.lambda_ee} "
        f"--confidence-threshold {args.confidence_threshold} "
        f"--dropout-rate {args.dropout_rate} "
        f"--use-class-weights "
        f"--weights-path class_weights.json "
        f"--model-name {args.model_name} "
        f"--output-dir {args.output_dir}"
    )
    if run_command(train_cmd) != 0:
        logger.error("Training failed. Aborting workflow.")
        return
    
    # Step 3: Generate additional evaluation metrics using the unified evaluation script
    logger.info("Step 3: Generating detailed evaluation metrics")
    eval_cmd = (
        f"python scripts/early_exit/unified_evaluate_forced_early_exit.py "
        f"--model-path models_optimized/{args.model_name}.pth "
        f"--data-dir {args.data_dir} "
        f"--batch-size 12 "
        f"--output-dir {args.output_dir}/evaluation"
    )
    if run_command(eval_cmd) != 0:
        logger.warning("Detailed evaluation failed, but workflow will continue.")
    
    # Compare against baseline model if specified
    if args.compare_baseline and os.path.exists('models_optimized/micropizzanet_early_exit.pth'):
        logger.info("Step 4: Comparing against baseline early exit model")
        baseline_eval_cmd = (
            f"python scripts/early_exit/unified_evaluate_forced_early_exit.py "
            f"--model-path models_optimized/micropizzanet_early_exit.pth "
            f"--data-dir {args.data_dir} "
            f"--batch-size 12 "
            f"--output-dir {args.output_dir}/evaluation/baseline"
        )
        if run_command(baseline_eval_cmd) != 0:
            logger.warning("Baseline evaluation failed, but workflow will continue.")
    
    logger.info("Improved early exit workflow completed successfully!")
    logger.info(f"Check the following files for results:")
    logger.info(f"1. class_distribution_early_exit.png - Dataset analysis")
    logger.info(f"2. {args.output_dir}/{args.model_name}_training.png - Training curves")
    logger.info(f"3. output/model_optimization/improved_early_exit/evaluation/ - Evaluation metrics")
    logger.info(f"4. improved_early_exit_workflow.log - Complete workflow logs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete improved early exit workflow")
    
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
    parser.add_argument("--model-name", type=str, default="improved_early_exit",
                       help="Name for the saved model")
    parser.add_argument("--output-dir", type=str, 
                       default="output/model_optimization/improved_early_exit",
                       help="Output directory for results")
    parser.add_argument("--compare-baseline", action="store_true",
                       help="Compare against baseline early exit model")
    
    args = parser.parse_args()
    main(args)
