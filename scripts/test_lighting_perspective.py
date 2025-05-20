#!/usr/bin/env python3
"""
Test script for lighting and perspective augmentations.
This script generates example images to demonstrate the new augmentations.
"""

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

try:
    import torch
    from torchvision.transforms import functional as TVF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using PIL-only mode.")

# Import augmentation functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.augment_functions import (
    apply_lighting_augmentation,
    apply_perspective_augmentation,
    apply_combined_light_perspective_augmentation
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test lighting and perspective augmentations")
    parser.add_argument("--input-dir", type=str, default="data/classified",
                        help="Directory containing pizza images to augment")
    parser.add_argument("--output-dir", type=str, default="output/augmentation_demo",
                        help="Directory to save demonstration images")
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Number of example images to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use-gpu", action="store_true", default=False,
                        help="Use GPU for augmentation if available")
    return parser.parse_args()

def show_augmentation_grid(original_img, augmented_imgs, titles, output_path=None, 
                           fig_title="Augmentation Examples"):
    """Display a grid of images showing original and augmented versions"""
    n_images = len(augmented_imgs) + 1  # +1 for original
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(cols * 4, rows * 4))
    plt.suptitle(fig_title, fontsize=16)
    
    # Show original first
    plt.subplot(rows, cols, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis('off')
    
    # Show augmented images
    for i, (img, title) in enumerate(zip(augmented_imgs, titles)):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved example to: {output_path}")
    
    plt.close()

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    input_dir = Path(args.input_dir)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(input_dir.glob(f"**/*{ext}")))
    
    # If no images found, try augmented_pizza directory
    if not image_files:
        fallback_dir = Path("augmented_pizza/raw")
        if fallback_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(list(fallback_dir.glob(f"**/*{ext}")))
    
    if not image_files:
        print(f"No images found in {input_dir}. Please specify a valid directory.")
        return 1
    
    print(f"Found {len(image_files)} images.")
    
    # Select random images for demonstration
    if len(image_files) > args.num_examples:
        demo_files = random.sample(image_files, args.num_examples)
    else:
        demo_files = image_files
    
    # Setup device
    device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() and args.use_gpu else "cpu")
    
    # Process each image
    for idx, img_path in enumerate(demo_files):
        print(f"Processing image {idx+1}/{len(demo_files)}: {img_path.name}")
        
        # Load image
        original_img = Image.open(img_path).convert('RGB')
        
        # Convert to tensor if PyTorch available
        if TORCH_AVAILABLE:
            img_tensor = TVF.to_tensor(original_img).to(device)
            
            # Apply lighting augmentations
            lighting_augmented = []
            lighting_titles = []
            for i in range(3):
                aug_img = apply_lighting_augmentation(img_tensor, device)
                # Convert back to PIL for display
                pil_img = TVF.to_pil_image(aug_img.cpu())
                lighting_augmented.append(pil_img)
                lighting_titles.append(f"Lighting {i+1}")
            
            # Apply perspective augmentations
            perspective_augmented = []
            perspective_titles = []
            for i in range(3):
                aug_img = apply_perspective_augmentation(img_tensor, device)
                # Convert back to PIL for display
                pil_img = TVF.to_pil_image(aug_img.cpu())
                perspective_augmented.append(pil_img)
                perspective_titles.append(f"Perspective {i+1}")
            
            # Apply combined augmentations
            combined_augmented = []
            combined_titles = []
            for i in range(3):
                aug_img = apply_combined_light_perspective_augmentation(img_tensor, device)
                # Convert back to PIL for display
                pil_img = TVF.to_pil_image(aug_img.cpu())
                combined_augmented.append(pil_img)
                combined_titles.append(f"Combined {i+1}")
        else:
            # PIL fallback mode
            # Apply lighting augmentations
            lighting_augmented = []
            lighting_titles = []
            for i in range(3):
                aug_img = apply_lighting_augmentation(original_img)
                lighting_augmented.append(aug_img)
                lighting_titles.append(f"Lighting {i+1}")
            
            # Apply perspective augmentations
            perspective_augmented = []
            perspective_titles = []
            for i in range(3):
                aug_img = apply_perspective_augmentation(original_img)
                perspective_augmented.append(aug_img)
                perspective_titles.append(f"Perspective {i+1}")
            
            # Apply combined augmentations
            combined_augmented = []
            combined_titles = []
            for i in range(3):
                aug_img = apply_combined_light_perspective_augmentation(original_img)
                combined_augmented.append(aug_img)
                combined_titles.append(f"Combined {i+1}")
        
        # Generate and save lighting examples
        show_augmentation_grid(
            original_img, 
            lighting_augmented,
            lighting_titles,
            output_path=output_dir / f"{img_path.stem}_lighting_examples.png",
            fig_title="Lighting Augmentation Examples"
        )
        
        # Generate and save perspective examples
        show_augmentation_grid(
            original_img, 
            perspective_augmented,
            perspective_titles,
            output_path=output_dir / f"{img_path.stem}_perspective_examples.png",
            fig_title="Perspective Augmentation Examples"
        )
        
        # Generate and save combined examples
        show_augmentation_grid(
            original_img, 
            combined_augmented,
            combined_titles,
            output_path=output_dir / f"{img_path.stem}_combined_examples.png",
            fig_title="Combined Light & Perspective Augmentation Examples"
        )
    
    print(f"Augmentation examples saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
