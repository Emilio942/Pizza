#!/usr/bin/env python3
"""
Test script for lighting and perspective augmentations
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TVF
import random

# Try importing from augment_classes and augment_functions
try:
    # Import the actual augmentation classes and functions
    from scripts.augment_classes import (
        DirectionalLightEffect, CLAHEEffect, ExposureVariationEffect, 
        PerspectiveTransformEffect, create_shadow_mask
    )
    from scripts.augment_functions import (
        apply_lighting_augmentation, 
        apply_perspective_augmentation,
        apply_combined_light_perspective_augmentation
    )
    AUGMENTATIONS_AVAILABLE = True
    print("Successfully imported augmentation modules!")
except ImportError as e:
    print(f"Error importing augmentation modules: {e}")
    AUGMENTATIONS_AVAILABLE = False

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

def test_augmentations(image_path, output_dir, use_gpu=False):
    """Test lighting and perspective augmentations on a single image"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select device
    device = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_tensor = TVF.to_tensor(img).to(device)
    
    if not AUGMENTATIONS_AVAILABLE:
        print("Augmentation modules not available. Cannot test.")
        return
    
    # Create lighting effects
    directional_light = DirectionalLightEffect(
        light_intensity_min=0.6,
        light_intensity_max=1.5,
        shadow_intensity_min=0.5,
        shadow_intensity_max=0.9,
        light_position='random',
        specular_highlight_prob=0.5
    )
    
    clahe_effect = CLAHEEffect(
        clip_limit=random.uniform(2.0, 6.0),
        tile_grid_size=(random.randint(4, 10), random.randint(4, 10)),
        contrast_limit=(0.7, 1.5),
        detail_enhancement=0.4
    )
    
    exposure_variation = ExposureVariationEffect(
        underexposure_prob=0.4,
        overexposure_prob=0.4,
        exposure_range=(0.4, 1.7),
        vignette_prob=0.5,
        color_temp_variation=True,
        noise_prob=0.3
    )
    
    # Create perspective effect
    perspective_effect = PerspectiveTransformEffect(
        rotation_range=(-25, 25),
        shear_range=(-12, 12),
        perspective_strength=(0.05, 0.25),
        border_handling='reflect',
        view_angle='random',
        zoom_range=(0.9, 1.1)
    )
    
    # Test individual effects
    results = []
    titles = []
    
    # Original image
    results.append(img)
    titles.append("Original")
    
    # Test directional light
    dl_result = directional_light(img_tensor)
    results.append(TVF.to_pil_image(dl_result.cpu()))
    titles.append("Directional Light")
    
    # Test CLAHE
    clahe_result = clahe_effect(img_tensor)
    results.append(TVF.to_pil_image(clahe_result.cpu()))
    titles.append("CLAHE")
    
    # Test exposure variation
    exp_result = exposure_variation(img_tensor)
    results.append(TVF.to_pil_image(exp_result.cpu()))
    titles.append("Exposure Variation")
    
    # Test perspective
    persp_result = perspective_effect(img_tensor)
    results.append(TVF.to_pil_image(persp_result.cpu()))
    titles.append("Perspective")
    
    # Test shadows
    pil_img = TVF.to_pil_image(img_tensor.cpu())
    shadow_mask = create_shadow_mask(
        pil_img.size, 
        num_shadows=2,
        shadow_dimension=0.5,
        blur_radius=20,
        shadow_type='directional',
        direction='random'
    )
    shadow_result = Image.composite(
        pil_img, 
        Image.new('RGB', pil_img.size, (0, 0, 0)),
        shadow_mask
    )
    results.append(shadow_result)
    titles.append("Shadows")
    
    # Test combined augmentations
    # 1. Full lighting augmentation
    lighting_result = apply_lighting_augmentation(img_tensor, device)
    results.append(TVF.to_pil_image(lighting_result.cpu()))
    titles.append("Lighting Augmentation")
    
    # 2. Full perspective augmentation
    perspective_result = apply_perspective_augmentation(img_tensor, device)
    results.append(TVF.to_pil_image(perspective_result.cpu()))
    titles.append("Perspective Augmentation")
    
    # 3. Combined lighting and perspective
    combined_result = apply_combined_light_perspective_augmentation(img_tensor, device)
    results.append(TVF.to_pil_image(combined_result.cpu()))
    titles.append("Combined Augmentation")
    
    # Show results
    show_images(results, titles, cols=3, figsize=(15, 15),
                save_path=os.path.join(output_dir, "augmentation_results.png"))
    
    # Generate additional combined samples
    combined_samples = []
    combined_titles = []
    
    for i in range(9):
        combined_result = apply_combined_light_perspective_augmentation(img_tensor, device)
        combined_samples.append(TVF.to_pil_image(combined_result.cpu()))
        combined_titles.append(f"Combined #{i+1}")
    
    show_images(combined_samples, combined_titles, cols=3, figsize=(15, 15),
                save_path=os.path.join(output_dir, "combined_samples.png"))
    
    print(f"Results saved to {output_dir}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test lighting and perspective augmentations")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="augmentation_test_results", help="Output directory")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Use GPU for processing")
    
    args = parser.parse_args()
    
    # Run test
    test_augmentations(args.image, args.output_dir, args.use_gpu)

if __name__ == "__main__":
    main()
