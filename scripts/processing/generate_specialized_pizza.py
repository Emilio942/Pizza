#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialized Pizza Image Generator

This script extends the diffusion integration to generate images with specific properties
that are difficult to capture in real photos, such as specific burn patterns, lighting
conditions, and transitional cooking states.

Usage:
    python scripts/generate_specialized_pizza.py --lighting
    python scripts/generate_specialized_pizza.py --burn-patterns
    python scripts/generate_specialized_pizza.py --transitions
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('specialized_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
SPECIALIZED_PROMPTS = {
    "lighting": [
        "pizza with harsh overhead lighting creating sharp shadows, professionally photographed",
        "pizza in dim lighting conditions, low exposure, detail visible in shadows",
        "pizza with side lighting casting long shadows across the surface, studio lighting setup",
        "pizza photographed in high contrast lighting, strong shadows and highlights",
        "pizza with backlighting causing rim lighting effect on the crust, professional food photography",
        "pizza in blue-tinted cold lighting, commercial refrigerator aesthetic",
        "pizza under warm yellow lighting, restaurant ambiance",
        "pizza with multiple light sources creating complex shadow patterns",
        "pizza in natural daylight through a window, soft shadows",
        "pizza with ring light illumination, even circular highlight reflections"
    ],
    "burn_patterns": [
        "pizza with precisely burnt outer ring and perfect center, professional food photography",
        "pizza with irregular burnt patches scattered randomly across the surface, detailed texture",
        "pizza with one half completely burnt black and other half perfectly cooked, sharp division",
        "pizza with burnt edges and spots in shape of leopard pattern, wood-fired appearance",
        "pizza with burnt bubbles in the crust where dough has risen, artisanal style",
        "pizza with burnt toppings but undercooked dough, contrast in cooking levels",
        "pizza with radial burn pattern from center outward, gradient effect",
        "pizza with perfect outer ring and burnt center, inverse of traditional cooking",
        "pizza with crosshatch burn pattern, griddle marks visible on bottom",
        "pizza with small concentrated burnt spots resembling pepperoni, misleading visual"
    ],
    "transitions": [
        "time-lapse sequence showing pizza dough transitioning from raw to perfectly cooked",
        "pizza showing gradient of cooking from raw on left to burnt on right, clear progression",
        "pizza captured midway through cooking process, half melted cheese, partially cooked dough",
        "pizza showing early stage of cooking with cheese just beginning to melt, dough still pale",
        "pizza at critical moment between cooked and burnt, golden brown transitioning to dark spots",
        "pizza showing specific temperature zones with different cooking levels, thermal gradient visible", 
        "pizza with uneven cooking due to oven hot spots, multiple states in one image",
        "sequence of pizza cooking progression focusing on color change of crust and cheese",
        "multiple stages of pizza cooking arranged in circular timeline format",
        "pizza with distinct cooking zones: raw dough, melting cheese, bubbling sauce, and caramelization"
    ]
}

def generate_specialized_images(category, count=5, model="sd-food", image_size=512):
    """
    Generate images with specialized prompts for the given category
    """
    if category not in SPECIALIZED_PROMPTS:
        logger.error(f"Unknown category: {category}")
        return False
    
    prompts = SPECIALIZED_PROMPTS[category]
    logger.info(f"Generating {count} images for category: {category}")
    
    # Create temporary prompt file
    prompt_file = Path(f"temp_specialized_prompts_{category}.txt")
    with open(prompt_file, "w") as f:
        for prompt in prompts[:count]:
            f.write(f"{prompt}\n")
    
    # Build command for the generator
    cmd = [
        "./memory_optimized_generator.py",
        f"--model={model}",
        f"--image_size={image_size}",
        "--batch_size=1",
        "--offload_to_cpu",
        "--expand_segments",
        f"--output_dir=data/synthetic/{category}_specialized"
    ]
    
    # Add prompts option if the generator supports it
    # Normally this would be implemented in the generator
    cmd.append(f"--prompt_file={prompt_file}")
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        logger.info(f"Generation completed with return code: {result.returncode}")
        
        # Remove temporary file
        prompt_file.unlink()
        
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Generation failed: {str(e)}")
        return False
    finally:
        # Clean up temp file if still exists
        if prompt_file.exists():
            prompt_file.unlink()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate specialized pizza images with diffusion models")
    
    # Category options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lighting", action="store_true", help="Generate images with various lighting conditions")
    group.add_argument("--burn-patterns", action="store_true", help="Generate images with specific burn patterns")
    group.add_argument("--transitions", action="store_true", help="Generate images showing cooking transitions")
    
    # Generation parameters
    parser.add_argument("--count", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--model", type=str, default="sd-food", 
                       choices=["sd-food", "sdxl-turbo", "sdxl", "kandinsky"],
                       help="Diffusion model to use")
    parser.add_argument("--image-size", type=int, default=512, help="Size of generated images")
    
    args = parser.parse_args()
    
    # Determine category
    if args.lighting:
        category = "lighting"
    elif args.burn_patterns:
        category = "burn_patterns"
    elif args.transitions:
        category = "transitions"
    
    # Generate images
    success = generate_specialized_images(
        category=category,
        count=args.count,
        model=args.model,
        image_size=args.image_size
    )
    
    if success:
        print(f"\nSuccessfully generated specialized {category} images.")
        print(f"Images saved to data/synthetic/{category}_specialized")
        print("\nTo integrate these images into the dataset:")
        print("1. Review them for quality")
        print("2. Run: python scripts/integrate_diffusion_images.py --organize-images")
        print("3. Run: python scripts/integrate_diffusion_images.py --integrate")
    else:
        print(f"\nFailed to generate specialized {category} images.")
        print("Check the log file for details.")

if __name__ == "__main__":
    main()
