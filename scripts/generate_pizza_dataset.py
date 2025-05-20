#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Pizza Cooking Dataset with Diffusion Models

This script is the main entry point for generating a high-quality synthetic
dataset of pizza cooking images using state-of-the-art diffusion models.
It handles dataset configuration, model selection, and runs the generation
process with advanced quality control.

Author: GitHub Copilot (2025-05-10)
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our diffusion modules
from src.augmentation.diffusion_pizza_generator import PizzaDiffusionGenerator, PIZZA_STAGES
from src.augmentation.advanced_pizza_diffusion_control import (
    AdvancedPizzaDiffusionControl, 
    COOKING_REGION_TEMPLATES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pizza_generator.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration presets for different generation modes
GENERATION_PRESETS = {
    "small_diverse": {
        "description": "A small but diverse dataset with all pizza types",
        "distribution": {
            "basic": 20,
            "burnt": 20, 
            "mixed": 20,
            "progression": 20,
            "segment": 20,
            "combined": 20
        },
        "template_weights": {
            "edge_burn": 0.3,
            "center_burn": 0.2,
            "half_burn": 0.15,
            "quarter_burn": 0.15,
            "random_spots": 0.2
        }
    },
    "training_focused": {
        "description": "Large dataset optimized for model training",
        "distribution": {
            "basic": 100,
            "burnt": 100,
            "mixed": 150,
            "progression": 100,
            "segment": 50,
            "combined": 150
        },
        "template_weights": {
            "edge_burn": 0.25,
            "center_burn": 0.2,
            "half_burn": 0.2,
            "quarter_burn": 0.15,
            "random_spots": 0.2
        }
    },
    "progression_heavy": {
        "description": "Dataset focused on cooking progression stages",
        "distribution": {
            "basic": 50,
            "burnt": 50,
            "mixed": 100,
            "progression": 300,
            "segment": 30,
            "combined": 70
        },
        "template_weights": {
            "edge_burn": 0.2,
            "center_burn": 0.2,
            "half_burn": 0.2,
            "quarter_burn": 0.15,
            "random_spots": 0.25
        }
    },
    "burn_patterns": {
        "description": "Dataset with various burn patterns",
        "distribution": {
            "basic": 30,
            "burnt": 200,
            "mixed": 100,
            "progression": 50,
            "segment": 20,
            "combined": 150
        },
        "template_weights": {
            "edge_burn": 0.3,
            "center_burn": 0.25,
            "half_burn": 0.2,
            "quarter_burn": 0.15,
            "random_spots": 0.1
        }
    }
}

# Model presets for different diffusion models to use
MODEL_PRESETS = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "sdxl",
        "description": "Stable Diffusion XL 1.0 - High quality general purpose model"
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "model_type": "sdxl",
        "description": "Stable Diffusion XL Turbo - Faster generation with good quality"
    },
    "sd-food": {
        "model_id": "prompthero/openjourney-v4",  # This would be a food-specific fine-tuned model
        "model_type": "custom",
        "description": "Food-specific fine-tuned model optimized for food photography"
    },
    "kandinsky": {
        "model_id": "kandinsky-community/kandinsky-2-2-decoder",
        "model_type": "kandinsky",
        "description": "Kandinsky 2.2 - Alternative diffusion model with different characteristics"
    }
}

def list_presets():
    """List all available generation and model presets"""
    print("\nGeneration Presets:")
    print("-----------------")
    for name, preset in GENERATION_PRESETS.items():
        print(f"  {name}: {preset['description']}")
        print(f"    Total images: {sum(preset['distribution'].values())}")
        print(f"    Class distribution: {', '.join([f'{k}: {v}' for k, v in preset['distribution'].items()])}")
        print()
    
    print("\nModel Presets:")
    print("-------------")
    for name, preset in MODEL_PRESETS.items():
        print(f"  {name}: {preset['description']}")
        print(f"    Model ID: {preset['model_id']}")
        print()


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import diffusers
        import transformers
        import accelerate
        
        logger.info("All required dependencies are installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
            cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using CUDA device: {cuda_device} with {cuda_memory:.1f}GB memory")
        else:
            logger.warning("CUDA not available, using CPU. Generation will be very slow!")
            
        return True
    
    except ImportError as e:
        missing_package = str(e).split("'")[1]
        logger.error(f"Missing dependency: {missing_package}")
        
        # Print installation instructions
        print(f"\nPlease install the missing dependency with:")
        print(f"pip install {missing_package}")
        print("\nOr install all dependencies with:")
        print("pip install diffusers transformers accelerate torch")
        
        return False


def generate_dataset(args):
    """Generate a synthetic dataset based on arguments"""
    start_time = time.time()
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"pizza_dataset_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating dataset in {output_dir}")
    
    # Save configuration
    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["start_time"] = datetime.now().isoformat()
    
    # Get generation preset
    if args.preset and args.preset in GENERATION_PRESETS:
        preset = GENERATION_PRESETS[args.preset]
        distribution = preset["distribution"]
        template_weights = preset["template_weights"]
        logger.info(f"Using preset: {args.preset} - {preset['description']}")
    else:
        # Use custom distribution if specified
        distribution = {}
        for stage in PIZZA_STAGES.keys():
            count = getattr(args, stage, 0)
            if count > 0:
                distribution[stage] = count
        
        template_weights = {template: 1.0 / len(COOKING_REGION_TEMPLATES) 
                           for template in COOKING_REGION_TEMPLATES.keys()}
    
    # Get model preset
    if args.model and args.model in MODEL_PRESETS:
        model_preset = MODEL_PRESETS[args.model]
        model_id = model_preset["model_id"]
        model_type = model_preset["model_type"]
        logger.info(f"Using model preset: {args.model} - {model_preset['description']}")
    else:
        # Use default model
        model_id = MODEL_PRESETS["sdxl"]["model_id"]
        model_type = MODEL_PRESETS["sdxl"]["model_type"]
        logger.info(f"Using default model: {model_id}")
    
    # Save configuration to output directory
    config["distribution"] = distribution
    config["template_weights"] = template_weights
    config["model_id"] = model_id
    config["model_type"] = model_type
    
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create controller
    controller = AdvancedPizzaDiffusionControl(
        image_size=args.image_size,
        image_size=args.image_size,
        device=torch.device("cpu") if args.cpu else None,
        output_dir=str(output_dir),
        model_id=model_id,
        model_type=model_type,
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold,
        config={"offload_to_cpu": args.offload_to_cpu},
        config={"offload_to_cpu": args.offload_to_cpu}
    )
    
    try:
        # Generate balanced dataset
        results = controller.generate_balanced_dataset(
            total_count=sum(distribution.values()),
            template_weights=template_weights,
            stage_weights={stage: count / sum(distribution.values()) 
                          for stage, count in distribution.items()},
            seed=args.seed
        )
        
        # Evaluate the dataset
        metrics = controller.evaluate_dataset_quality(output_dir)
        
        # Add metrics to config
        config["metrics"] = metrics
        config["end_time"] = datetime.now().isoformat()
        config["total_time_seconds"] = time.time() - start_time
        
        # Update configuration file with final stats
        with open(output_dir / "generation_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        total_generated = sum(len(stage_results) for template_results in results.values() 
                              for stage_results in template_results.values())
        
        logger.info(f"Dataset generation complete!")
        logger.info(f"Generated {total_generated} images in {time.time() - start_time:.1f} seconds")
        logger.info(f"Average quality score: {metrics.get('average_quality', 'N/A')}")
        logger.info(f"Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating dataset: {str(e)}")
        raise
    
    finally:
        # Clean up resources
        controller.cleanup()
    
    return output_dir


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate Pizza Cooking Dataset with Diffusion Models")
    
    # Main configuration
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                        help="Directory to save generated dataset")
    parser.add_argument("--preset", type=str, choices=list(GENERATION_PRESETS.keys()),
                        help="Generation preset to use")
    parser.add_argument("--model", type=str, choices=list(MODEL_PRESETS.keys()), default="sdxl",
                        help="Diffusion model preset to use")
    parser.add_argument("--list_presets", action="store_true",
                        help="List all available presets and exit")
    
    # Custom distribution (used if no preset is specified)
    parser.add_argument("--basic", type=int, default=0,
                        help="Number of basic (raw) pizza images to generate")
    parser.add_argument("--burnt", type=int, default=0,
                        help="Number of burnt pizza images to generate")
    parser.add_argument("--mixed", type=int, default=0,
                        help="Number of mixed pizza images to generate")
    parser.add_argument("--progression", type=int, default=0,
                        help="Number of progression pizza images to generate")
    parser.add_argument("--segment", type=int, default=0,
                        help="Number of segment pizza images to generate")
    parser.add_argument("--combined", type=int, default=0,
                        help="Number of combined pizza images to generate")
    
    # Advanced options
    parser.add_argument("--offload_to_cpu", action="store_true",
                        help="Offload model components to CPU when not in use")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU (slow)")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Size of generated images (width and height)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (reduce if VRAM issues)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--quality_threshold", type=float, default=0.65,
                        help="Minimum quality threshold (0.0-1.0)")
    parser.add_argument("--check_dependencies", action="store_true",
                        help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        list_presets()
        return
    
    # Check dependencies if requested
    if args.check_dependencies:
        check_dependencies()
        return
    
    # Make sure we have all dependencies
    if not check_dependencies():
        logger.error("Missing dependencies, exiting")
        return
    
    # Validate arguments
    if not args.preset and all(getattr(args, stage, 0) == 0 for stage in PIZZA_STAGES.keys()):
        logger.error("Error: Must specify either a preset or custom distribution")
        parser.print_help()
        return
    
    # Generate the dataset
    try:
        output_dir = generate_dataset(args)
        print(f"\nDataset generation successful!")
        print(f"Output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"\nDataset generation failed: {str(e)}")
        print("See log file for details")


if __name__ == "__main__":
    main()
