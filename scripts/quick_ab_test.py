#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified DIFFUSION-4.2: A/B-Tests with faster execution for demonstration
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.integration.diffusion_training_integration import PizzaDiffusionTrainer, DEFAULT_PARAMS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_quick_ab_test():
    """Run a quick A/B test with reduced parameters for faster completion."""
    
    base_dir = Path("/home/emilio/Documents/ai/pizza")
    synthetic_dir = base_dir / "data" / "synthetic_filtered"
    real_dir = base_dir / "data" / "augmented"
    output_dir = base_dir / "output" / "diffusion_evaluation_quick"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup quick training parameters for demonstration
    params = DEFAULT_PARAMS.copy()
    params.update({
        "epochs": 3,  # Very reduced for quick demonstration
        "batch_size": 16,
        "learning_rate": 0.001,
        "early_stopping": 2,
        "synthetic_ratio": 0.5,
    })
    
    logger.info("Starting quick A/B test demonstration...")
    logger.info(f"Parameters: {params}")
    
    # Create trainer
    trainer = PizzaDiffusionTrainer(
        real_data_dir=str(real_dir),
        synthetic_data_dir=str(synthetic_dir),
        output_dir=str(output_dir),
        params=params
    )
    
    # Load datasets
    trainer.load_datasets()
    
    # Results storage
    results = {}
    
    try:
        # 1. Train with real data only
        logger.info("=== EXPERIMENT 1: Real data only ===")
        real_result = trainer.train_with_real_data_only()
        results["real_only"] = real_result
        logger.info(f"Real only result: {real_result['best_val_accuracy']:.4f}")
        
        # 2. Train with mixed data
        logger.info("=== EXPERIMENT 2: Mixed data (50% synthetic) ===")
        mixed_result = trainer.train_with_mixed_data(synthetic_ratio=0.5)
        results["mixed_50"] = mixed_result
        logger.info(f"Mixed data result: {mixed_result['best_val_accuracy']:.4f}")
        
        # 3. Calculate impact
        real_acc = real_result["best_val_accuracy"]
        mixed_acc = mixed_result["best_val_accuracy"]
        improvement = ((mixed_acc - real_acc) / real_acc * 100) if real_acc > 0 else 0
        
        # Create summary report
        summary = {
            "experiment_type": "quick_demonstration",
            "training_epochs": params["epochs"],
            "results": {
                "real_only_accuracy": real_acc,
                "mixed_data_accuracy": mixed_acc,
                "improvement_percent": improvement,
                "synthetic_data_beneficial": improvement > 0
            },
            "note": "This is a quick demonstration with reduced epochs. Full training would take longer."
        }
        
        # Save results
        with open(output_dir / "quick_ab_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=== QUICK A/B TEST RESULTS ===")
        logger.info(f"Real only accuracy: {real_acc:.4f}")
        logger.info(f"Mixed data accuracy: {mixed_acc:.4f}")
        logger.info(f"Improvement: {improvement:.2f}%")
        logger.info(f"Synthetic data beneficial: {'Yes' if improvement > 0 else 'No'}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Quick A/B test failed: {e}")
        raise

if __name__ == "__main__":
    run_quick_ab_test()
