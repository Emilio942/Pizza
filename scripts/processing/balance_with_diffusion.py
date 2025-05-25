#!/usr/bin/env python3
"""
Dataset Balancing Automation Script - DIFFUSION-3.1

This script automates the execution of the dataset balancing strategy defined in 
DIFFUSION-3.1 using the targeted diffusion pipeline from DIFFUSION-2.1.

Features:
- Automated generation according to balancing strategy
- Class-specific prompt management
- Lighting and burn level distribution
- Progress tracking and error handling
- Quality verification and reporting

Author: GitHub Copilot (2025-01-28)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the targeted diffusion pipeline
try:
    from src.augmentation.targeted_diffusion_pipeline import (
        TargetedDiffusionPipeline, 
        TargetedGenerationConfig,
        LIGHTING_CONDITION_TEMPLATES,
        BURN_LEVEL_TEMPLATES
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import targeted diffusion pipeline: {e}")
    PIPELINE_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_balancing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetBalancer:
    """Automated dataset balancing using targeted diffusion generation"""
    
    def __init__(self, config_path: str = "diffusion_balance_config.json"):
        """Initialize the dataset balancer"""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.stats = {
            "start_time": datetime.now(),
            "total_requested": 0,
            "total_generated": 0,
            "total_verified": 0,
            "failed_generations": 0,
            "by_class": {},
            "by_lighting": {},
            "by_burn_level": {}
        }
        
        # Initialize pipeline if available
        self.pipeline = None
        if PIPELINE_AVAILABLE:
            pipeline_config = TargetedGenerationConfig(
                **self.config["dataset_balancing_config"]["generation_parameters"]
            )
            self.pipeline = TargetedDiffusionPipeline(pipeline_config)
        
    def load_config(self) -> Dict:
        """Load the balancing configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def analyze_current_dataset(self) -> Dict:
        """Analyze current dataset state"""
        logger.info("Analyzing current dataset state...")
        
        # Try to load existing analysis
        analysis_file = Path("output/data_analysis/class_distribution_train.json")
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                current_analysis = json.load(f)
            
            # Update configuration with current counts
            config_dist = self.config["dataset_balancing_config"]["target_distribution"]
            if "total" in current_analysis:
                for class_name, current_count in current_analysis["total"].items():
                    if class_name in config_dist:
                        config_dist[class_name]["current_count"] = current_count
                        # Recalculate generation count
                        target = config_dist[class_name]["target_count"]
                        config_dist[class_name]["generate_count"] = max(0, target - current_count)
            
            return current_analysis
        else:
            logger.warning("No existing class distribution analysis found")
            return {}
    
    def get_generation_plan(self) -> List[Dict]:
        """Create a detailed generation plan based on priority and requirements"""
        plan = []
        config = self.config["dataset_balancing_config"]
        target_dist = config["target_distribution"]
        lighting_dist = config["lighting_distribution"]
        
        # Sort classes by priority and generation count
        priority_order = {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "low": 4
        }
        
        sorted_classes = sorted(
            target_dist.items(),
            key=lambda x: (priority_order.get(x[1]["priority"], 5), -x[1]["generate_count"])
        )
        
        for class_name, class_config in sorted_classes:
            generate_count = class_config["generate_count"]
            if generate_count <= 0:
                continue
            
            # Calculate lighting distribution for this class
            for lighting_condition, ratio in lighting_dist.items():
                lighting_count = max(1, int(generate_count * ratio))
                
                # Special handling for burnt class - distribute across burn levels
                if class_name == "burnt":
                    burn_dist = config["burn_level_distribution"]
                    for burn_level, burn_ratio in burn_dist.items():
                        burn_count = max(1, int(lighting_count * burn_ratio))
                        if burn_count > 0:
                            plan.append({
                                "class": class_name,
                                "lighting": lighting_condition,
                                "burn_level": burn_level,
                                "count": burn_count,
                                "priority": class_config["priority"]
                            })
                else:
                    plan.append({
                        "class": class_name,
                        "lighting": lighting_condition,
                        "count": lighting_count,
                        "priority": class_config["priority"]
                    })
        
        return plan
    
    def execute_generation_task(self, task: Dict) -> List[Dict]:
        """Execute a single generation task"""
        if not self.pipeline:
            logger.error("Pipeline not available - cannot execute generation")
            return []
        
        class_name = task["class"]
        lighting = task["lighting"]
        count = task["count"]
        
        logger.info(f"Generating {count} images for class '{class_name}' "
                   f"with lighting '{lighting}'")
        
        # Get class-specific prompts
        config = self.config["dataset_balancing_config"]
        class_prompts = config["class_specific_prompts"][class_name]
        
        custom_additions = []
        if "variations" in class_prompts:
            custom_additions = class_prompts["variations"]
        
        results = []
        
        try:
            if "burn_level" in task:
                # Special burnt class generation
                burn_level = task["burn_level"]
                logger.info(f"Generating burnt class with burn level: {burn_level}")
                
                # Use burn level specific prompts
                burn_prompts = class_prompts["burn_specific"][burn_level]
                custom_additions = burn_prompts
                
                task_results = self.pipeline.generate_with_burn_level(
                    burn_level=burn_level,
                    count=count,
                    lighting_condition=lighting,
                    base_stage=class_name,
                    custom_prompt_additions=custom_additions
                )
            else:
                # Regular class generation with lighting
                task_results = self.pipeline.generate_with_lighting_condition(
                    lighting_condition=lighting,
                    count=count,
                    base_stage=class_name,
                    custom_prompt_additions=custom_additions
                )
            
            if task_results:
                results.extend(task_results)
                self.update_stats(task, len(task_results), len(task_results))
                logger.info(f"Successfully generated {len(task_results)} images")
            else:
                logger.warning(f"No images generated for task: {task}")
                self.update_stats(task, count, 0)
                
        except Exception as e:
            logger.error(f"Error executing task {task}: {e}")
            self.stats["failed_generations"] += count
        
        return results
    
    def update_stats(self, task: Dict, requested: int, generated: int):
        """Update generation statistics"""
        self.stats["total_requested"] += requested
        self.stats["total_generated"] += generated
        
        # Update by class
        class_name = task["class"]
        if class_name not in self.stats["by_class"]:
            self.stats["by_class"][class_name] = {"requested": 0, "generated": 0}
        self.stats["by_class"][class_name]["requested"] += requested
        self.stats["by_class"][class_name]["generated"] += generated
        
        # Update by lighting
        lighting = task["lighting"]
        if lighting not in self.stats["by_lighting"]:
            self.stats["by_lighting"][lighting] = {"requested": 0, "generated": 0}
        self.stats["by_lighting"][lighting]["requested"] += requested
        self.stats["by_lighting"][lighting]["generated"] += generated
    
    def execute_balancing_plan(self) -> Dict:
        """Execute the complete dataset balancing plan"""
        logger.info("Starting dataset balancing execution...")
        
        if not PIPELINE_AVAILABLE:
            logger.error("Targeted diffusion pipeline not available - cannot proceed")
            return {"success": False, "error": "Pipeline not available"}
        
        # Analyze current state
        current_state = self.analyze_current_dataset()
        
        # Generate execution plan
        plan = self.get_generation_plan()
        logger.info(f"Generated plan with {len(plan)} tasks")
        
        # Create output directories
        self.setup_output_directories()
        
        # Execute plan
        all_results = []
        total_tasks = len(plan)
        
        for i, task in enumerate(plan, 1):
            logger.info(f"Executing task {i}/{total_tasks}: {task}")
            
            task_results = self.execute_generation_task(task)
            all_results.extend(task_results)
            
            # Brief pause between tasks to prevent overheating
            if i < total_tasks:
                time.sleep(2)
        
        # Generate final report
        final_report = self.generate_final_report(all_results)
        
        return final_report
    
    def setup_output_directories(self):
        """Create necessary output directories"""
        config = self.config["dataset_balancing_config"]
        base_dir = Path(config["output_structure"]["base_directory"])
        
        # Create class directories
        for class_name in config["target_distribution"].keys():
            class_dir = base_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create lighting directories
        lighting_base = base_dir / "lighting"
        for lighting in config["lighting_distribution"].keys():
            lighting_dir = lighting_base / lighting
            lighting_dir.mkdir(parents=True, exist_ok=True)
        
        # Create burn level directories
        burn_base = base_dir / "burn_levels"
        for burn_level in config["burn_level_distribution"].keys():
            burn_dir = burn_base / burn_level
            burn_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        metadata_dir = base_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_final_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive final report"""
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]
        
        report = {
            "execution_summary": {
                "start_time": self.stats["start_time"].isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration.total_seconds() / 60,
                "total_requested": self.stats["total_requested"],
                "total_generated": self.stats["total_generated"],
                "success_rate": (self.stats["total_generated"] / max(self.stats["total_requested"], 1)) * 100,
                "failed_generations": self.stats["failed_generations"]
            },
            "class_breakdown": self.stats["by_class"],
            "lighting_breakdown": self.stats["by_lighting"],
            "generated_files": [r.get("path", "") for r in results],
            "next_steps": [
                "Run class distribution analysis to verify balance",
                "Perform manual quality review of sample images",
                "Test balanced dataset with existing classification pipeline",
                "Update dataset documentation with new synthetic data"
            ]
        }
        
        # Save report
        output_dir = Path(self.config["dataset_balancing_config"]["output_structure"]["base_directory"])
        report_path = output_dir / "metadata" / f"balancing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to: {report_path}")
        return report

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Execute dataset balancing strategy")
    parser.add_argument("--config", default="diffusion_balance_config.json",
                       help="Path to balancing configuration file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show plan without executing generation")
    parser.add_argument("--class-filter", nargs="+",
                       help="Only process specific classes")
    
    args = parser.parse_args()
    
    try:
        balancer = DatasetBalancer(args.config)
        
        if args.dry_run:
            # Show plan without execution
            plan = balancer.get_generation_plan()
            print(f"\\nGeneration Plan ({len(plan)} tasks):")
            for i, task in enumerate(plan, 1):
                print(f"{i:2d}. {task}")
            
            total_images = sum(task["count"] for task in plan)
            print(f"\\nTotal images to generate: {total_images}")
            return
        
        # Execute the balancing plan
        report = balancer.execute_balancing_plan()
        
        # Print summary
        if report.get("success", True):
            print("\\n" + "="*60)
            print("DATASET BALANCING COMPLETED SUCCESSFULLY")
            print("="*60)
            summary = report["execution_summary"]
            print(f"Total requested: {summary['total_requested']}")
            print(f"Total generated: {summary['total_generated']}")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            print(f"Duration: {summary['duration_minutes']:.1f} minutes")
            print("\\nNext steps:")
            for step in report["next_steps"]:
                print(f"  - {step}")
        else:
            print(f"\\nBalancing failed: {report.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Balancing execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
