#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIFFUSION-4.2: A/B-Tests (Synthetisch vs. Real) im Training durchführen

This script implements the complete A/B testing workflow for evaluating
the impact of diffusion-generated synthetic data on pizza recognition
model performance.

Author: GitHub Copilot (2025-05-24)
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.integration.diffusion_training_integration import PizzaDiffusionTrainer, DEFAULT_PARAMS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiffusionABTestRunner:
    """
    Comprehensive A/B testing runner for DIFFUSION-4.2 implementation.
    """
    
    def __init__(self, base_dir: str = "/home/emilio/Documents/ai/pizza"):
        """
        Initialize the A/B test runner.
        
        Args:
            base_dir: Base project directory
        """
        self.base_dir = Path(base_dir)
        
        # Set up directories
        self.synthetic_dir = self.base_dir / "data" / "synthetic"
        self.real_dir = self.base_dir / "data" / "augmented"
        self.test_dir = self.base_dir / "data" / "test"
        self.output_dir = self.base_dir / "output" / "diffusion_evaluation"
        self.filtered_synthetic_dir = self.base_dir / "data" / "synthetic_filtered"
        
        # Evaluation results file
        self.evaluation_file = self.base_dir / "output" / "diffusion_analysis" / "full_synthetic_evaluation_20250524_025813.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DIFFUSION-4.2 A/B Test Runner")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Synthetic data: {self.synthetic_dir}")
        logger.info(f"  Real data: {self.real_dir}")
        logger.info(f"  Test data: {self.test_dir}")
        logger.info(f"  Output: {self.output_dir}")
    
    def load_evaluation_results(self) -> Dict[str, Any]:
        """
        Load the synthetic dataset evaluation results.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Loading synthetic dataset evaluation results...")
        
        if not self.evaluation_file.exists():
            raise FileNotFoundError(f"Evaluation file not found: {self.evaluation_file}")
        
        with open(self.evaluation_file, 'r') as f:
            evaluation_data = json.load(f)
        
        logger.info(f"Loaded evaluation for {evaluation_data['total_images']} images")
        logger.info(f"Quality distribution: {evaluation_data['summary']['quality_distribution']}")
        
        return evaluation_data
    
    def filter_synthetic_dataset(self, quality_threshold: float = 0.4) -> int:
        """
        Filter synthetic dataset to remove very poor quality images.
        
        Args:
            quality_threshold: Minimum quality score to keep (default: 0.4)
            
        Returns:
            Number of images kept after filtering
        """
        logger.info("Filtering synthetic dataset based on quality evaluation...")
        
        # Load evaluation results
        evaluation_data = self.load_evaluation_results()
        
        # Create filtered directory
        if self.filtered_synthetic_dir.exists():
            shutil.rmtree(self.filtered_synthetic_dir)
        self.filtered_synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter images based on quality score and category
        kept_images = 0
        removed_images = 0
        
        for image_data in evaluation_data['detailed_metrics']:
            filename = image_data['filename']
            quality_score = image_data['quality_score']
            quality_category = image_data['quality_category']
            
            # Keep images that are not "very_poor" and have quality score >= threshold
            if quality_category != "very_poor" and quality_score >= quality_threshold:
                # Copy image to filtered directory
                src_path = self.synthetic_dir / filename
                dst_path = self.filtered_synthetic_dir / filename
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    kept_images += 1
                else:
                    logger.warning(f"Source image not found: {src_path}")
            else:
                removed_images += 1
                logger.debug(f"Removed {filename} (quality: {quality_score:.4f}, category: {quality_category})")
        
        # Save filtering report
        filtering_report = {
            "timestamp": evaluation_data["timestamp"],
            "original_count": evaluation_data["total_images"],
            "kept_count": kept_images,
            "removed_count": removed_images,
            "quality_threshold": quality_threshold,
            "filtering_criteria": {
                "exclude_very_poor": True,
                "min_quality_score": quality_threshold
            },
            "improvement": {
                "removal_percentage": (removed_images / evaluation_data["total_images"]) * 100,
                "kept_percentage": (kept_images / evaluation_data["total_images"]) * 100
            }
        }
        
        report_file = self.output_dir / "synthetic_filtering_report.json"
        with open(report_file, 'w') as f:
            json.dump(filtering_report, f, indent=2)
        
        logger.info(f"Synthetic dataset filtering completed:")
        logger.info(f"  Original images: {evaluation_data['total_images']}")
        logger.info(f"  Kept images: {kept_images} ({kept_images/evaluation_data['total_images']*100:.1f}%)")
        logger.info(f"  Removed images: {removed_images} ({removed_images/evaluation_data['total_images']*100:.1f}%)")
        logger.info(f"  Filtering report saved: {report_file}")
        
        return kept_images
    
    def verify_data_availability(self) -> Dict[str, int]:
        """
        Verify that all required datasets are available.
        
        Returns:
            Dictionary with dataset sizes
        """
        logger.info("Verifying data availability...")
        
        data_info = {}
        
        # Check real training data
        if self.real_dir.exists():
            real_images = list(self.real_dir.glob("**/*.jpg")) + list(self.real_dir.glob("**/*.png"))
            data_info['real_training'] = len(real_images)
            logger.info(f"Real training data: {data_info['real_training']} images")
        else:
            logger.error(f"Real training data directory not found: {self.real_dir}")
            data_info['real_training'] = 0
        
        # Check filtered synthetic data
        if self.filtered_synthetic_dir.exists():
            synthetic_images = list(self.filtered_synthetic_dir.glob("*.jpg")) + list(self.filtered_synthetic_dir.glob("*.png"))
            data_info['synthetic'] = len(synthetic_images)
            logger.info(f"Filtered synthetic data: {data_info['synthetic']} images")
        else:
            logger.warning(f"Filtered synthetic data directory not found: {self.filtered_synthetic_dir}")
            data_info['synthetic'] = 0
        
        # Check test data
        if self.test_dir.exists():
            test_images = list(self.test_dir.glob("**/*.jpg")) + list(self.test_dir.glob("**/*.png"))
            data_info['test'] = len(test_images)
            logger.info(f"Test data: {data_info['test']} images")
        else:
            logger.warning(f"Test data directory not found: {self.test_dir}")
            data_info['test'] = 0
        
        return data_info
    
    def run_ab_tests(self, synthetic_ratios: List[float] = None) -> Dict[str, Any]:
        """
        Run the complete A/B testing experiment.
        
        Args:
            synthetic_ratios: List of synthetic data ratios to test
            
        Returns:
            Dictionary containing all experiment results
        """
        logger.info("Starting A/B testing experiments...")
        
        if synthetic_ratios is None:
            synthetic_ratios = [0.3, 0.5]  # Test a few key ratios
        
        # Verify data availability
        data_info = self.verify_data_availability()
        
        if data_info['real_training'] == 0:
            raise RuntimeError("No real training data available")
        
        if data_info['synthetic'] == 0:
            logger.warning("No filtered synthetic data available, will only test real data")
            synthetic_ratios = []
        
        # Setup training parameters optimized for A/B testing
        params = DEFAULT_PARAMS.copy()
        params.update({
            "epochs": 15,  # Reduced for faster comparison
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": 5,
            "synthetic_ratio": 0.5,  # Default, will be overridden
        })
        
        # Create trainer
        trainer = PizzaDiffusionTrainer(
            real_data_dir=str(self.real_dir),
            synthetic_data_dir=str(self.filtered_synthetic_dir),
            output_dir=str(self.output_dir),
            params=params
        )
        
        # Load datasets
        trainer.load_datasets()
        
        # Load evaluation data for timestamp
        try:
            evaluation_data = self.load_evaluation_results()
            timestamp = evaluation_data.get("timestamp", "unknown")
        except:
            timestamp = "unknown"
        
        # Results storage
        experiment_results = {
            "experiment_info": {
                "timestamp": timestamp,
                "real_training_images": data_info['real_training'],
                "filtered_synthetic_images": data_info['synthetic'],
                "test_images": data_info['test'],
                "synthetic_ratios_tested": synthetic_ratios,
                "training_params": params
            },
            "results": {}
        }
        
        # 1. Train with real data only
        logger.info("=== EXPERIMENT 1: Training with REAL DATA ONLY ===")
        real_only_result = trainer.train_with_real_data_only()
        experiment_results["results"]["real_only"] = real_only_result
        
        # 2. Train with mixed data for each ratio
        for ratio in synthetic_ratios:
            logger.info(f"=== EXPERIMENT 2.{synthetic_ratios.index(ratio)+1}: Training with MIXED DATA (ratio: {ratio:.1f}) ===")
            mixed_result = trainer.train_with_mixed_data(synthetic_ratio=ratio)
            experiment_results["results"][f"mixed_ratio_{ratio:.1f}"] = mixed_result
        
        # 3. Generate comparison report
        logger.info("=== GENERATING COMPARISON REPORT ===")
        trainer.compare_and_report()
        
        # 4. Calculate impact metrics
        real_acc = real_only_result.get("best_val_accuracy", 0)
        
        # Find best mixed result
        mixed_results = {k: v for k, v in experiment_results["results"].items() if k.startswith("mixed")}
        
        if mixed_results:
            best_mixed_key = max(mixed_results.items(), key=lambda x: x[1].get("best_val_accuracy", 0))[0]
            best_mixed_result = mixed_results[best_mixed_key]
            best_mixed_acc = best_mixed_result.get("best_val_accuracy", 0)
            
            improvement = ((best_mixed_acc - real_acc) / real_acc * 100) if real_acc > 0 else 0
            
            experiment_results["impact_analysis"] = {
                "real_only_accuracy": real_acc,
                "best_mixed_accuracy": best_mixed_acc,
                "best_mixed_configuration": best_mixed_key,
                "absolute_improvement": best_mixed_acc - real_acc,
                "relative_improvement_percent": improvement,
                "synthetic_data_beneficial": improvement > 0,
                "improvement_significance": "significant" if improvement > 5 else "marginal" if improvement > 0 else "negative"
            }
            
            logger.info("=== A/B TESTING RESULTS ===")
            logger.info(f"Real data only accuracy: {real_acc:.4f}")
            logger.info(f"Best mixed data accuracy: {best_mixed_acc:.4f} ({best_mixed_key})")
            logger.info(f"Improvement: {improvement:.2f}%")
            logger.info(f"Synthetic data is {'beneficial' if improvement > 0 else 'not beneficial'}")
        else:
            experiment_results["impact_analysis"] = {
                "real_only_accuracy": real_acc,
                "synthetic_data_beneficial": False,
                "note": "No mixed data experiments conducted"
            }
        
        # Save complete results
        results_file = self.output_dir / "synthetic_data_impact.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        logger.info(f"Complete A/B testing results saved: {results_file}")
        
        return experiment_results
    
    def update_task_status(self, results: Dict[str, Any]):
        """
        Update the task file to mark DIFFUSION-4.2 as completed.
        
        Args:
            results: Experiment results dictionary
        """
        logger.info("Updating task status...")
        
        task_file = self.base_dir / "aufgaben.txt"
        
        if not task_file.exists():
            logger.warning(f"Task file not found: {task_file}")
            return
        
        # Read current content
        with open(task_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add completion note
        impact_analysis = results.get("impact_analysis", {})
        improvement = impact_analysis.get("relative_improvement_percent", 0)
        
        # Get timestamp
        timestamp = results.get("experiment_info", {}).get("timestamp", "current")
        
        completion_note = f"""
DIFFUSION-4.2 COMPLETED ({timestamp}):
- A/B tests durchgeführt: Real vs. Synthetic training data
- Real-only accuracy: {impact_analysis.get('real_only_accuracy', 'N/A'):.4f}
- Best mixed accuracy: {impact_analysis.get('best_mixed_accuracy', 'N/A'):.4f}
- Improvement: {improvement:.2f}%
- Synthetic data impact: {impact_analysis.get('improvement_significance', 'unknown')}
- Results: output/diffusion_evaluation/synthetic_data_impact.json
"""
        
        # Update content
        if "DIFFUSION-4.2:" in content:
            # Replace existing entry
            lines = content.split('\n')
            new_lines = []
            skip_next = False
            
            for i, line in enumerate(lines):
                if "DIFFUSION-4.2:" in line and "TODO" in line:
                    new_lines.append(line.replace("TODO", "DONE"))
                    new_lines.append(completion_note)
                    skip_next = True
                elif skip_next and line.strip() == "":
                    skip_next = False
                    new_lines.append(line)
                elif not skip_next:
                    new_lines.append(line)
            
            updated_content = '\n'.join(new_lines)
        else:
            updated_content = content + completion_note
        
        # Write back
        with open(task_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logger.info(f"Task status updated in {task_file}")


def main():
    """Main function to run the complete DIFFUSION-4.2 A/B testing workflow."""
    logger.info("Starting DIFFUSION-4.2: A/B-Tests (Synthetisch vs. Real) im Training durchführen")
    
    try:
        # Initialize runner
        runner = DiffusionABTestRunner()
        
        # Step 1: Filter synthetic dataset based on quality evaluation
        logger.info("Step 1: Filtering synthetic dataset...")
        kept_images = runner.filter_synthetic_dataset(quality_threshold=0.4)
        
        if kept_images == 0:
            logger.error("No synthetic images passed quality filtering. Cannot proceed with A/B testing.")
            return
        
        # Step 2: Run A/B tests
        logger.info("Step 2: Running A/B tests...")
        results = runner.run_ab_tests(synthetic_ratios=[0.3, 0.5])
        
        # Step 3: Update task status
        logger.info("Step 3: Updating task status...")
        runner.update_task_status(results)
        
        # Summary
        impact = results.get("impact_analysis", {})
        logger.info("=== DIFFUSION-4.2 COMPLETED SUCCESSFULLY ===")
        logger.info(f"Synthetic data impact: {impact.get('improvement_significance', 'unknown')}")
        logger.info(f"Performance improvement: {impact.get('relative_improvement_percent', 0):.2f}%")
        logger.info(f"Results saved to: output/diffusion_evaluation/synthetic_data_impact.json")
        
    except Exception as e:
        logger.error(f"A/B testing failed: {e}")
        raise


if __name__ == "__main__":
    main()
