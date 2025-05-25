#!/usr/bin/env python3
"""
Small test script for lighting and perspective augmentations
"""

import os
import sys
import argparse
from PIL import Image
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as TVF

# Try importing our augmentation modules
try:
    from scripts.augment_classes import (
        DirectionalLightEffect, CLAHEEffect, ExposureVariationEffect, 
        PerspectiveTransformEffect, create_shadow_mask
    )
    from scripts.augment_functions import (
        apply_lighting_augmentation, 
        apply_perspective_augmentation,
        apply_combined_light_perspective_augmentation
    )
    print("Successfully imported augmentation modules!")
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing augmentation modules: {e}")
    MODULES_AVAILABLE = False
    sys.exit(1)

def show_images(images, titles=None, cols=3, figsize=(15, 10), save_path=None):
    """Display multiple images in a grid with optional saving"""
    if not images:
        print("No images to display")
        return
    
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        plt.subplot(rows, cols, i + 1)
        
        if torch.is_tensor(img):
            # Convert tensor to NumPy for display
            img = img.cpu().detach()
            img = img.permute(1, 2, 0).numpy()
            # Normalize to [0,1] if needed
            img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        if titles is not None and i < len(titles):
            plt.title(titles[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()

def test_augmentations(image_path, output_dir):
    """Test our lighting and perspective augmentations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize results
    results = [img]
    titles = ["Original"]
    
    # Test lighting augmentations
    print("Testing lighting augmentations...")
    lighting_environments = ['bright_daylight', 'indoor_restaurant', 'evening_mood', 'direct_sunlight']
    for env in lighting_environments:
        # Call our lighting augmentation function
        img_tensor = TVF.to_tensor(img).unsqueeze(0)
        lit_img = apply_lighting_augmentation(img_tensor, device=device, environment=env)
        pil_img = TVF.to_pil_image(lit_img.squeeze(0))
        
        results.append(pil_img)
        titles.append(f"Lighting: {env}")
    
    # Test perspective augmentations
    print("Testing perspective augmentations...")
    view_angles = ['overhead', 'table_level', 'closeup', 'angled']
    for angle in view_angles:
        # Call our perspective augmentation function
        img_tensor = TVF.to_tensor(img).unsqueeze(0)
        perspective_img = apply_perspective_augmentation(img_tensor, device=device, view_angle=angle)
        pil_img = TVF.to_pil_image(perspective_img.squeeze(0))
        
        results.append(pil_img)
        titles.append(f"Perspective: {angle}")
    
    # Test combined augmentations
    print("Testing combined augmentations...")
    scenarios = ['restaurant_table', 'outdoor_picnic', 'kitchen_countertop', 'delivery_photo']
    for scenario in scenarios:
        # Call our combined augmentation function
        img_tensor = TVF.to_tensor(img).unsqueeze(0)
        combined_img = apply_combined_light_perspective_augmentation(img_tensor, device=device, scenario=scenario)
        pil_img = TVF.to_pil_image(combined_img.squeeze(0))
        
        results.append(pil_img)
        titles.append(f"Combined: {scenario}")
    
    # Display and save results
    print("Saving results...")
    show_images(results, titles, cols=4, figsize=(16, 12),
               save_path=os.path.join(output_dir, "augmentation_test_results.png"))
    
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test lighting and perspective augmentations")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="augmentation_test_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run test
    test_augmentations(args.image, args.output_dir)

if __name__ == "__main__":
    main()
