"""
Memory-Optimized Dataset Balancing Automation Script - DIFFUSION-3.1

This script automates the execution of the dataset balancing strategy defined in 
DIFFUSION-3.1 using memory-optimized diffusion models instead of SDXL.

Features:
- Memory-optimized model selection (sd-food, kandinsky)
- Smaller image sizes to reduce VRAM usage
- CPU offloading for memory management
- All other features from the original DIFFUSION-3.1 script

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_balancing_memory_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryOptimizedDatasetBalancer:
    """Memory-optimized implementation of the dataset balancing strategy"""
    
    def __init__(self, config_path: str = "diffusion_balance_config.json"):
        """Initialize the memory-optimized dataset balancer"""
        self.config_path = config_path
        self.config = self.load_config()
        self.pipeline = None
        self.generation_stats = {
            "total_generated": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "start_time": None,
            "end_time": None
        }
        
    def load_config(self) -> Dict:
        """Load the configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default memory-optimized configuration"""
        return {
            "model_config": {
                "model_type": "sd-food",  # Use smaller model instead of SDXL
                "image_size": 512,        # Smaller image size
                "batch_size": 1,          # Minimum batch size
                "use_cpu_offload": True,  # Enable CPU offloading
                "expand_segments": True   # Memory optimization
            },
            "dataset_balancing_config": {
                "target_distribution": {
                    "combined": {
                        "target_count": 120,
                        "current_count": 0,
                        "generate_count": 120,
                        "priority": "critical"
                    },
                    "slightly_burnt": {
                        "target_count": 100,
                        "current_count": 0,
                        "generate_count": 100,
                        "priority": "high"
                    },
                    "burnt": {
                        "target_count": 80,
                        "current_count": 0,
                        "generate_count": 80,
                        "priority": "high"
                    },
                    "raw": {
                        "target_count": 60,
                        "current_count": 0,
                        "generate_count": 60,
                        "priority": "medium"
                    },
                    "dough": {
                        "target_count": 40,
                        "current_count": 0,
                        "generate_count": 40,
                        "priority": "medium"
                    },
                    "cooked": {
                        "target_count": 80,
                        "current_count": 0,
                        "generate_count": 80,
                        "priority": "medium"
                    }
                },
                "lighting_distribution": {
                    "overhead_harsh": 0.3,
                    "side_dramatic": 0.3,
                    "dim_ambient": 0.25,
                    "natural_diffuse": 0.15
                },
                "burn_level_distribution": {
                    "lightly_burnt": 0.4,
                    "moderately_burnt": 0.35,
                    "heavily_burnt": 0.25
                }
            }
        }
    
    def initialize_pipeline(self):
        """Initialize the memory-optimized diffusion pipeline"""
        if not PIPELINE_AVAILABLE:
            logger.error("Targeted diffusion pipeline not available")
            return False
        
        try:
            # Use memory-optimized configuration
            model_config = self.config["model_config"]
            
            # Create memory-optimized generation config
            generation_config = TargetedGenerationConfig(
                model_type=model_config["model_type"],
                image_size=model_config["image_size"],
                batch_size=model_config["batch_size"],
                use_cpu_offload=model_config["use_cpu_offload"],
                expand_segments=model_config["expand_segments"]
            )
            
            self.pipeline = TargetedDiffusionPipeline(generation_config)
            logger.info(f"Initialized memory-optimized pipeline with {model_config['model_type']} model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def analyze_current_dataset(self) -> Dict:
        """Analyze the current dataset distribution"""
        logger.info("Analyzing current dataset state...")
        
        # Try to load existing analysis
        analysis_file = "data/analysis/class_distribution_analysis.json"
        if os.path.exists(analysis_file):
            try:
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
            except Exception as e:
                logger.warning(f"Error loading existing analysis: {e}")
        
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
    
    def execute_generation_task(self, task: Dict) -> List[str]:
        """Execute a single generation task with memory optimization"""
        class_name = task["class"]
        lighting = task["lighting"]
        count = task["count"]
        
        # Get class-specific prompts
        class_prompts = self.config.get("class_specific_prompts", {})
        
        try:
            if "burn_level" in task:
                # Handle burnt class with burn levels
                burn_level = task["burn_level"]
                logger.info(f"Generating {count} images for class '{class_name}' with lighting '{lighting}' and burn level '{burn_level}'")
                
                # Get burn-specific prompts
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
                # Handle regular classes
                logger.info(f"Generating {count} images for class '{class_name}' with lighting '{lighting}'")
                
                # Get lighting-specific prompts
                lighting_prompts = class_prompts["lighting_specific"][lighting]
                stage_prompts = class_prompts["stage_specific"][class_name]
                custom_additions = lighting_prompts + stage_prompts
                
                task_results = self.pipeline.generate_with_lighting_condition(
                    lighting_condition=lighting,
                    count=count,
                    base_stage=class_name,
                    custom_prompt_additions=custom_additions
                )
            
            if task_results and len(task_results) > 0:
                logger.info(f"Successfully generated {len(task_results)} images for task")
                self.generation_stats["total_generated"] += len(task_results)
                self.generation_stats["successful_tasks"] += 1
                return task_results
            else:
                logger.warning(f"No images generated for task: {task}")
                self.generation_stats["failed_tasks"] += 1
                return []
                
        except Exception as e:
            logger.error(f"Error executing generation task: {e}")
            self.generation_stats["failed_tasks"] += 1
            return []
    
    def execute_balancing_strategy(self):
        """Execute the complete dataset balancing strategy with memory optimization"""
        logger.info("Starting memory-optimized dataset balancing execution...")
        self.generation_stats["start_time"] = datetime.now()
        
        # Step 1: Analyze current dataset
        current_analysis = self.analyze_current_dataset()
        
        # Step 2: Generate execution plan
        generation_plan = self.get_generation_plan()
        logger.info(f"Generated plan with {len(generation_plan)} tasks")
        
        # Step 3: Execute each task with memory optimization
        for i, task in enumerate(generation_plan, 1):
            logger.info(f"Executing task {i}/{len(generation_plan)}: {task}")
            
            results = self.execute_generation_task(task)
            
            if not results:
                logger.warning(f"No images generated for task: {task}")
            
            # Add delay between tasks to allow memory cleanup
            time.sleep(2)
        
        self.generation_stats["end_time"] = datetime.now()
        self.log_final_statistics()
    
    def log_final_statistics(self):
        """Log final generation statistics"""
        stats = self.generation_stats
        duration = stats["end_time"] - stats["start_time"]
        
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total images generated: {stats['total_generated']}")
        logger.info(f"Successful tasks: {stats['successful_tasks']}")
        logger.info(f"Failed tasks: {stats['failed_tasks']}")
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Success rate: {stats['successful_tasks'] / (stats['successful_tasks'] + stats['failed_tasks']) * 100:.1f}%")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Memory-Optimized Dataset Balancer - DIFFUSION-3.1")
    parser.add_argument("--config", default="diffusion_balance_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--model", default="sd-food", choices=["sd-food", "kandinsky", "sdxl-turbo"],
                       help="Model to use for generation")
    parser.add_argument("--image-size", type=int, default=512,
                       help="Image size for generation")
    
    args = parser.parse_args()
    
    # Create balancer instance
    balancer = MemoryOptimizedDatasetBalancer(args.config)
    
    # Override model configuration with command line args
    balancer.config["model_config"]["model_type"] = args.model
    balancer.config["model_config"]["image_size"] = args.image_size
    
    # Initialize pipeline
    if not balancer.initialize_pipeline():
        logger.error("Failed to initialize pipeline, exiting")
        return
    
    # Execute balancing strategy
    balancer.execute_balancing_strategy()

if __name__ == "__main__":
    main()
