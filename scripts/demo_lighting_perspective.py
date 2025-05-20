#!/usr/bin/env python3
"""
Simple test for enhanced lighting and perspective augmentations
"""

import os
import sys
import argparse
import random
import time
import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TVF
import torchvision.transforms as transforms

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

def create_shadow_mask(img_size, num_shadows=3, shadow_dimension=0.5, blur_radius=20):
    """
    Create a shadow mask with random polygons
    """
    width, height = img_size
    mask = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(mask)
    
    max_dim = max(width, height)
    
    for _ in range(num_shadows):
        # Random shadow intensity (0=black, 255=white)
        shadow_val = random.randint(100, 200)
        
        # Random number of points for polygon (3-5)
        num_points = random.randint(3, 6)
        
        # Generate random points for the shadow polygon
        # Anchor shadows to image edges for more realism
        points = []
        for _ in range(num_points):
            # Randomly decide whether to anchor this point to an edge
            if random.random() < 0.5:
                # Anchor to edge
                edge = random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    points.append((random.randint(0, width), 0))
                elif edge == 'bottom':
                    points.append((random.randint(0, width), height))
                elif edge == 'left':
                    points.append((0, random.randint(0, height)))
                else:  # right
                    points.append((width, random.randint(0, height)))
            else:
                # Random point within image
                shadow_range = int(max_dim * shadow_dimension)
                points.append((
                    random.randint(width // 4, width * 3 // 4),
                    random.randint(height // 4, height * 3 // 4)
                ))
        
        # Draw the shadow polygon
        draw.polygon(points, fill=shadow_val)
    
    # Apply blur to soften shadow edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask

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

def apply_lighting_variation(img, light_position='random', exposure_factor=None, underexposed=None):
    """Apply lighting variations"""
    if isinstance(img, Image.Image):
        img_tensor = TVF.to_tensor(img)
    else:
        img_tensor = img.clone()
    
    # Create dimensions
    c, h, w = img_tensor.shape
    
    # 1. Apply directional lighting
    # Create coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing='ij'
    )
    
    # Determine light direction
    if light_position == 'random':
        light_pos = random.choice(['overhead', 'side', 'corner'])
    else:
        light_pos = light_position
    
    # Create light direction vector based on position
    if light_pos == 'overhead':
        # Light coming from above
        light_dir_x = random.uniform(-0.2, 0.2)
        light_dir_y = random.uniform(-0.2, 0.2)
        light_dir_z = -1  # Down
    elif light_pos == 'side':
        # Side lighting
        side = random.choice(['left', 'right'])
        light_dir_x = -1 if side == 'left' else 1
        light_dir_y = random.uniform(-0.2, 0.2)
        light_dir_z = -0.5
    else:  # corner
        # Corner lighting
        corner_x = random.choice([-1, 1])
        corner_y = random.choice([-1, 1])
        light_dir_x = corner_x
        light_dir_y = corner_y
        light_dir_z = -0.5
    
    # Normalize
    norm = (light_dir_x**2 + light_dir_y**2 + light_dir_z**2)**0.5
    light_dir_x /= norm
    light_dir_y /= norm
    light_dir_z /= norm
    
    # Create a simple lighting mask
    # Approximate normal as pointing up
    normal_x = 0
    normal_y = 0
    normal_z = 1
    
    # Add slight variation to normal based on position (simulate surface)
    center_dist = torch.sqrt(x_coords**2 + y_coords**2)
    mask = center_dist < 0.8  # Pizza region
    
    # Dot product between light and normal
    light_dot_normal = -light_dir_x * normal_x - light_dir_y * normal_y + light_dir_z * normal_z
    
    # Scale to [0, 1]
    light_mask = torch.ones_like(center_dist)
    light_mask[mask] = 0.5 + 0.5 * light_dot_normal  # Scale to [0, 1]
    
    # Apply lighting mask
    light_intensity = random.uniform(0.8, 1.5)
    for i in range(c):
        img_tensor[i] = img_tensor[i] * (light_mask * (light_intensity - 1) + 1)
    
    # 2. Apply exposure variation
    if exposure_factor is None:
        exposure_factor = random.uniform(0.7, 1.3)
    
    img_tensor = img_tensor * exposure_factor
    
    # 3. Apply color temperature variation
    if underexposed is None:
        underexposed = exposure_factor < 1.0
    
    if underexposed:
        # Cool/blue tint for low light
        if random.random() < 0.7:
            img_tensor[2] = torch.clamp(img_tensor[2] * random.uniform(1.0, 1.2), 0, 1)  # Boost blue
            img_tensor[0] = img_tensor[0] * random.uniform(0.8, 1.0)  # Reduce red
    else:
        # Warm tint for bright light
        if random.random() < 0.7:
            img_tensor[0] = torch.clamp(img_tensor[0] * random.uniform(1.0, 1.15), 0, 1)  # Boost red
            img_tensor[2] = img_tensor[2] * random.uniform(0.8, 1.0)  # Reduce blue
    
    # Ensure values are in valid range
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    if isinstance(img, Image.Image):
        return TVF.to_pil_image(img_tensor)
    return img_tensor

def apply_perspective_variation(img, view_angle='random'):
    """Apply perspective variations"""
    if isinstance(img, Image.Image):
        img_pil = img.copy()
    else:
        img_pil = TVF.to_pil_image(img)
    
    width, height = img_pil.size
    
    # Select viewing angle setup
    if view_angle == 'random':
        view = random.choice(['overhead', 'table', 'angled'])
    else:
        view = view_angle
    
    # Configure parameters based on view
    if view == 'overhead':
        rotation_range = (-10, 10)
        perspective_strength = random.uniform(0.02, 0.1)
    elif view == 'table':
        rotation_range = (-15, 15)
        perspective_strength = random.uniform(0.1, 0.2)
    else:  # angled
        rotation_range = (-25, 25)
        perspective_strength = random.uniform(0.15, 0.25)
    
    # Apply rotation
    rotation_angle = random.uniform(rotation_range[0], rotation_range[1])
    rotated = img_pil.rotate(rotation_angle, resample=Image.BICUBIC)
    
    # Apply perspective transform
    # Define the 4 corner points
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    
    # Modify corners based on perspective strength
    warped_corners = corners.copy()
    
    # Choose which corner(s) to modify
    if view == 'overhead':
        # Minimal distortion for overhead
        idx = random.randint(0, 3)
        offset = int(min(width, height) * perspective_strength)
        if idx == 0:  # Top-left
            warped_corners[0] = (offset, offset)
        elif idx == 1:  # Top-right
            warped_corners[1] = (width - offset, offset)
        elif idx == 2:  # Bottom-right
            warped_corners[2] = (width - offset, height - offset)
        else:  # Bottom-left
            warped_corners[3] = (offset, height - offset)
    elif view == 'table':
        # Table view - perspective from bottom
        offset_top = int(min(width, height) * perspective_strength)
        offset_bottom = int(offset_top / 2)
        warped_corners[0] = (offset_top, offset_top)
        warped_corners[1] = (width - offset_top, offset_top)
        warped_corners[2] = (width - offset_bottom, height - offset_bottom)
        warped_corners[3] = (offset_bottom, height - offset_bottom)
    else:  # angled
        # Stronger perspective from one side
        side = random.choice(['left', 'right', 'top', 'bottom'])
        offset = int(min(width, height) * perspective_strength)
        
        if side == 'left':
            warped_corners[0] = (offset, offset)
            warped_corners[3] = (offset, height - offset)
        elif side == 'right':
            warped_corners[1] = (width - offset, offset)
            warped_corners[2] = (width - offset, height - offset)
        elif side == 'top':
            warped_corners[0] = (offset, offset)
            warped_corners[1] = (width - offset, offset)
        else:  # bottom
            warped_corners[2] = (width - offset, height - offset)
            warped_corners[3] = (offset, height - offset)
    
    # Calculate perspective transform coefficients
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)
    
    coeffs = find_coeffs(corners, warped_corners)
    
    # Apply transform
    transformed = rotated.transform((width, height), Image.PERSPECTIVE, coeffs, 
                                    Image.BICUBIC)
    
    if isinstance(img, Image.Image):
        return transformed
    else:
        return TVF.to_tensor(transformed)

def test_augmentations(image_path, output_dir):
    """Test lighting and perspective augmentations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Test augmentations
    results = []
    titles = []
    
    # Original image
    results.append(img)
    titles.append("Original")
    
    # Lighting variations
    lighting_positions = ['overhead', 'side', 'corner']
    for pos in lighting_positions:
        light_img = apply_lighting_variation(img, light_position=pos)
        results.append(light_img)
        titles.append(f"Light: {pos}")
    
    # Exposure variations
    exposures = [0.6, 1.0, 1.4]
    for exposure in exposures:
        exp_img = apply_lighting_variation(img, exposure_factor=exposure)
        results.append(exp_img)
        titles.append(f"Exposure: {exposure:.1f}")
    
    # Perspective variations
    view_angles = ['overhead', 'table', 'angled']
    for angle in view_angles:
        persp_img = apply_perspective_variation(img, view_angle=angle)
        results.append(persp_img)
        titles.append(f"View: {angle}")
    
    # Shadow effect
    shadow_mask = create_shadow_mask(img.size, num_shadows=2, blur_radius=20)
    shadow_img = Image.composite(
        img, 
        ImageOps.colorize(shadow_mask, (0, 0, 0), (255, 255, 255)),
        shadow_mask
    )
    results.append(shadow_img)
    titles.append("Shadows")
    
    # Combined effects
    for i in range(5):
        # Apply perspective first
        combined = apply_perspective_variation(img, view_angle='random')
        # Then lighting
        combined = apply_lighting_variation(combined, light_position='random')
        # Add shadows 
        if random.random() < 0.5:
            shadow_mask = create_shadow_mask(combined.size, num_shadows=random.randint(1, 3), blur_radius=20)
            combined = Image.composite(
                combined, 
                ImageOps.colorize(shadow_mask, (0, 0, 0), (255, 255, 255)),
                shadow_mask
            )
        results.append(combined)
        titles.append(f"Combined #{i+1}")
    
    # Show and save results
    show_images(results, titles, cols=4, figsize=(16, 12),
                save_path=os.path.join(output_dir, "lighting_perspective_test.png"))
    
    print(f"Results saved to {output_dir}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test lighting and perspective augmentations")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="lighting_perspective_test", help="Output directory")
    
    args = parser.parse_args()
    
    # Run test
    test_augmentations(args.image, args.output_dir)

if __name__ == "__main__":
    main()
