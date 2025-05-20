#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Optimized Pizza Dataset Generator

This script is a wrapper around the main generate_pizza_dataset.py script
that adds memory optimization options for GPUs with limited VRAM.
It specifically targets the NVIDIA GeForce RTX 3060 with 12GB RAM.

Usage:
    python memory_optimized_generator.py --preset small_diverse --model sd-food --image_size 512
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory_optimized_generator.log')
    ]
)
logger = logging.getLogger(__name__)

def get_arg_parser():
    """Creates the argument parser with memory optimization options"""
    parser = argparse.ArgumentParser(description="Memory-Optimized Pizza Dataset Generator")
    
    # Main options (passed to generate_pizza_dataset.py)
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                      help="Directory to save generated dataset")
    parser.add_argument("--preset", type=str, 
                      choices=["small_diverse", "training_focused", "progression_heavy", "burn_patterns"],
                      help="Generation preset to use")
    parser.add_argument("--model", type=str, 
                      choices=["sdxl", "sdxl-turbo", "sd-food", "kandinsky"],
                      default="sd-food",
                      help="Diffusion model preset to use (default: sd-food for memory efficiency)")
    
    # Custom memory optimization options
    parser.add_argument("--image_size", type=int, default=512,
                      help="Size of generated images (default: 512, SDXL default is 1024)")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for generation (default: 1 for memory efficiency)")
    parser.add_argument("--use_cpu", action="store_true",
                      help="Use CPU instead of GPU (very slow but guaranteed to work)")
    parser.add_argument("--half_precision", action="store_true",
                      help="Use half precision (fp16) to save memory")
    parser.add_argument("--use_sequential", action="store_true",
                      help="Load models sequentially (one at a time)")
    parser.add_argument("--offload_to_cpu", action="store_true",
                      help="Offload model components to CPU when not in use")
    parser.add_argument("--expand_segments", action="store_true",
                      help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    return parser

def patch_diffusion_generator():
    """
    Creates a patched version of diffusion_pizza_generator.py with memory optimizations
    """
    # Path to original file
    original_file = Path("src/augmentation/diffusion_pizza_generator.py")
    
    # Check if the file exists
    if not original_file.exists():
        logger.error(f"Could not find {original_file}")
        sys.exit(1)
    
    # Read the original file
    with open(original_file, "r") as f:
        content = f.read()
    
    # Create backup if it doesn't exist
    backup_file = original_file.with_suffix(".py.bak")
    if not backup_file.exists():
        with open(backup_file, "w") as f:
            f.write(content)
        logger.info(f"Created backup: {backup_file}")
    
    # Apply patches
    
    # Patch 1: Add image_size parameter to __init__
    if "image_size: int = 1024," not in content:
        content = content.replace(
            "def __init__(",
            "def __init__(\n        image_size: int = 1024,",
        )
        
        # Also store the image_size parameter
        content = content.replace(
            "self.config = config or {}",
            "self.config = config or {}\n        self.image_size = image_size"
        )
    
    # Patch 2: Add model offloading support
    if "enable_model_cpu_offload" not in content:
        content = content.replace(
            "self.text2img_pipe.enable_attention_slicing()",
            "self.text2img_pipe.enable_attention_slicing()\n                # Enable memory optimizations\n                if self.config.get('offload_to_cpu', False):\n                    self.text2img_pipe.enable_model_cpu_offload()"
        )
    
    # Patch 3: Add image size parameter to pipeline calls
    if "height=self.image_size, width=self.image_size" not in content:
        content = content.replace(
            "images = self.text2img_pipe(",
            "images = self.text2img_pipe(\n                height=self.image_size, width=self.image_size,"
        )
    
    # Write the modified file
    with open(original_file, "w") as f:
        f.write(content)
    
    logger.info(f"Applied memory optimization patches to {original_file}")

def patch_generate_script():
    """
    Creates a patched version of generate_pizza_dataset.py with memory optimizations
    """
    # Path to original file
    original_file = Path("scripts/generate_pizza_dataset.py")
    
    # Check if the file exists
    if not original_file.exists():
        logger.error(f"Could not find {original_file}")
        sys.exit(1)
    
    # Read the original file
    with open(original_file, "r") as f:
        content = f.read()
    
    # Create backup if it doesn't exist
    backup_file = original_file.with_suffix(".py.bak")
    if not backup_file.exists():
        with open(backup_file, "w") as f:
            f.write(content)
        logger.info(f"Created backup: {backup_file}")
    
    # Apply patches
    
    # Patch 1: Add image_size argument
    if "--image_size" not in content:
        content = content.replace(
            "# Advanced options",
            "# Advanced options\n    parser.add_argument(\"--image_size\", type=int, default=1024,\n                        help=\"Size of generated images (width and height)\")"
        )
    
    # Patch 2: Add CPU option
    if "--cpu" not in content:
        content = content.replace(
            "# Advanced options",
            "# Advanced options\n    parser.add_argument(\"--cpu\", action=\"store_true\",\n                        help=\"Use CPU instead of GPU (slow)\")"
        )
    
    # Patch 3: Add offload option
    if "--offload_to_cpu" not in content:
        content = content.replace(
            "# Advanced options",
            "# Advanced options\n    parser.add_argument(\"--offload_to_cpu\", action=\"store_true\",\n                        help=\"Offload model components to CPU when not in use\")"
        )
    
    # Patch 4: Pass image_size to controller
    content = content.replace(
        "controller = AdvancedPizzaDiffusionControl(",
        "controller = AdvancedPizzaDiffusionControl(\n        image_size=args.image_size,"
    )
    
    # Patch 5: Pass device to controller
    if "device=torch.device(\"cpu\") if args.cpu else None," not in content:
        content = content.replace(
            "controller = AdvancedPizzaDiffusionControl(",
            "controller = AdvancedPizzaDiffusionControl(\n        device=torch.device(\"cpu\") if args.cpu else None,"
        )
    
    # Patch 6: Pass offload_to_cpu to config
    content = content.replace(
        "quality_threshold=args.quality_threshold",
        "quality_threshold=args.quality_threshold,\n        config={\"offload_to_cpu\": args.offload_to_cpu}"
    )
    
    # Write the modified file
    with open(original_file, "w") as f:
        f.write(content)
    
    logger.info(f"Applied memory optimization patches to {original_file}")

def patch_advanced_controller():
    """
    Creates a patched version of advanced_pizza_diffusion_control.py with memory optimizations
    """
    # Path to original file
    original_file = Path("src/augmentation/advanced_pizza_diffusion_control.py")
    
    # Check if the file exists
    if not original_file.exists():
        logger.error(f"Could not find {original_file}")
        sys.exit(1)
    
    # Read the original file
    with open(original_file, "r") as f:
        content = f.read()
    
    # Create backup if it doesn't exist
    backup_file = original_file.with_suffix(".py.bak")
    if not backup_file.exists():
        with open(backup_file, "w") as f:
            f.write(content)
        logger.info(f"Created backup: {backup_file}")
    
    # Apply patches
    
    # Patch 1: Add image_size parameter to __init__
    if "image_size: int = 1024," not in content:
        content = content.replace(
            "def __init__(",
            "def __init__(\n        image_size: int = 1024,"
        )
    
    # Patch 2: Pass image_size to generator
    content = content.replace(
        "self.generator = PizzaDiffusionGenerator(",
        "self.generator = PizzaDiffusionGenerator(\n            image_size=image_size,"
    )
    
    # Patch 3: Update mask generator size
    content = content.replace(
        "self.mask_generator = CookingControlMask()",
        "self.mask_generator = CookingControlMask(size=(image_size, image_size))"
    )
    
    # Write the modified file
    with open(original_file, "w") as f:
        f.write(content)
    
    logger.info(f"Applied memory optimization patches to {original_file}")

def run_generator(args):
    """
    Run the patched generator with the specified arguments
    """
    # Set environment variables for memory optimization
    env = os.environ.copy()
    
    if args.expand_segments:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # Disable Hugging Face telemetry to save some memory
    env["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/generate_pizza_dataset.py",
        f"--output_dir={args.output_dir}",
        f"--batch_size={args.batch_size}",
        f"--image_size={args.image_size}"
    ]
    
    # Add preset if specified
    if args.preset:
        cmd.append(f"--preset={args.preset}")
    
    # Add model if specified
    if args.model:
        cmd.append(f"--model={args.model}")
    
    # Add CPU option if specified
    if args.use_cpu:
        cmd.append("--cpu")
    
    # Add offload option if specified
    if args.offload_to_cpu:
        cmd.append("--offload_to_cpu")
    
    # Log the command
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, env=env)
    
    return result.returncode

def main():
    """
    Main entry point
    """
    parser = get_arg_parser()
    args = parser.parse_args()
    
    print("Memory-Optimized Pizza Dataset Generator")
    print("=======================================")
    print(f"Model: {args.model}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using CPU: {'Yes' if args.use_cpu else 'No'}")
    print(f"Offload to CPU: {'Yes' if args.offload_to_cpu else 'No'}")
    print(f"Expand segments: {'Yes' if args.expand_segments else 'No'}")
    print()
    
    # Apply patches to the source code
    patch_diffusion_generator()
    patch_generate_script()
    patch_advanced_controller()
    
    # Run the generator
    return_code = run_generator(args)
    
    if return_code == 0:
        print("\nGeneration completed successfully!")
    else:
        print(f"\nGeneration failed with code: {return_code}")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())
