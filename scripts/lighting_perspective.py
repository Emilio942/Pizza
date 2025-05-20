#!/usr/bin/env python3
"""
Lighting and Perspective Augmentation Implementations for Pizza Dataset

This module implements specialized augmentation classes and functions for:
1. Realistic light conditions:
   - Directional lighting
   - Shadows
   - Over/underexposure
   - CLAHE-like contrast enhancement

2. Perspective changes:
   - Advanced rotations
   - Shearing and distortions

These augmentations are specifically designed to improve model robustness
for pizza recognition under various lighting conditions and viewing angles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF
import random
import numpy as np
import math
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

class DirectionalLightEffect(nn.Module):
    """
    Simulates directional lighting effects on pizza images
    by adding highlights and shadows based on a light direction vector
    """
    
    def __init__(self, 
                 light_intensity_min=0.5,
                 light_intensity_max=1.5,
                 shadow_intensity_min=0.4, 
                 shadow_intensity_max=0.8):
        super().__init__()
        self.light_intensity_min = light_intensity_min
        self.light_intensity_max = light_intensity_max
        self.shadow_intensity_min = shadow_intensity_min
        self.shadow_intensity_max = shadow_intensity_max
    
    def forward(self, img):
        """Apply directional lighting effect to pizza image"""
        if not torch.is_tensor(img):
            # Convert PIL to tensor
            img_tensor = TVF.to_tensor(img)
        else:
            img_tensor = img.clone()
        
        # Get image dimensions
        c, h, w = img_tensor.shape
        
        # Generate random light direction (normalized vector)
        light_angle = random.uniform(0, 2 * math.pi)
        light_dir_x = math.cos(light_angle)
        light_dir_y = math.sin(light_angle)
        
        # Create coordinates grid
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        
        # Calculate the dot product between position and light direction
        # This creates the directional gradient
        dot_product = light_dir_x * x_coords + light_dir_y * y_coords
        
        # Normalize to [0, 1] range
        light_mask = (dot_product + 1) / 2
        
        # Add some noise for realism
        noise = torch.randn(h, w) * 0.1
        light_mask = torch.clamp(light_mask + noise, 0, 1)
        
        # Apply light intensity adjustment
        light_intensity = random.uniform(self.light_intensity_min, self.light_intensity_max)
        shadow_intensity = random.uniform(self.shadow_intensity_min, self.shadow_intensity_max)
        
        # Create highlight and shadow masks
        highlight_mask = light_mask
        shadow_mask = 1 - light_mask
        
        # Apply the effect channel-wise
        for i in range(c):
            # Apply highlights to bright areas
            img_tensor[i] = img_tensor[i] * (1 + (light_intensity - 1) * highlight_mask)
            # Apply shadows to dark areas
            img_tensor[i] = img_tensor[i] * (1 - (1 - shadow_intensity) * shadow_mask)
        
        # Ensure values are in valid range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        if not torch.is_tensor(img):
            # Convert back to PIL
            return TVF.to_pil_image(img_tensor)
        else:
            return img_tensor


class CLAHEEffect(nn.Module):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) implementation
    for improving local contrast in pizza images, especially in areas with
    poor exposure.
    """
    
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def _apply_clahe_pil(self, img):
        """Apply CLAHE using PIL operations"""
        # Convert to LAB color space (using YCbCr as approximation available in PIL)
        img_ycbcr = img.convert('YCbCr')
        y, cb, cr = img_ycbcr.split()
        
        # Apply contrast enhancement to Y channel
        y_enhanced = ImageOps.equalize(y)
        
        # Merge back
        img_enhanced = Image.merge('YCbCr', (y_enhanced, cb, cr))
        return img_enhanced.convert('RGB')
    
    def _apply_clahe_torch(self, img_tensor):
        """Apply CLAHE-like effect using torch operations"""
        # Clone to avoid modifying the original
        result = img_tensor.clone()
        
        # Get dimensions
        c, h, w = img_tensor.shape
        
        # Only apply to luminance/intensity channel approximated as average of RGB
        lum = torch.mean(img_tensor, dim=0, keepdim=True)
        
        # Create grid of tiles
        tile_h, tile_w = self.tile_grid_size
        h_step, w_step = h // tile_h, w // tile_w
        
        # Process each tile
        for i in range(tile_h):
            for j in range(tile_w):
                # Get current tile coordinates
                h_start, h_end = i * h_step, min((i + 1) * h_step, h)
                w_start, w_end = j * w_step, min((j + 1) * w_step, w)
                
                # Skip if tile is empty
                if h_end <= h_start or w_end <= w_start:
                    continue
                
                # Get current tile
                tile = lum[0, h_start:h_end, w_start:w_end]
                
                # Calculate histogram
                hist = torch.histc(tile, bins=256, min=0, max=1)
                
                # Apply clipping
                if self.clip_limit > 0:
                    clip_limit = max(1, int(self.clip_limit * (h_end - h_start) * (w_end - w_start) / 256))
                    excess = torch.sum(torch.clamp(hist - clip_limit, min=0))
                    # Redistribute excess
                    clipped_hist = torch.clamp(hist, max=clip_limit)
                    redistrib_per_bin = excess / 256
                    hist = clipped_hist + redistrib_per_bin
                
                # Calculate cumulative distribution function
                cdf = torch.cumsum(hist, dim=0)
                cdf = cdf / cdf[-1]  # Normalize to [0, 1]
                
                # Map using CDF
                tile_min, tile_max = torch.min(tile), torch.max(tile)
                if tile_min < tile_max:  # Avoid division by zero
                    norm_tile = (tile - tile_min) / (tile_max - tile_min)
                    # Get indices for lookup
                    indices = (norm_tile * 255).long()
                    # Apply transformation with bilinear interpolation
                    enhanced_tile = cdf[indices]
                    
                    # Adjust contrast gain to avoid excessive enhancement
                    alpha = random.uniform(0.5, 1.0)
                    enhanced_tile = alpha * enhanced_tile + (1 - alpha) * tile
                    
                    # Apply to all channels with color preservation
                    for ch in range(c):
                        color_ratio = torch.ones_like(tile)
                        nonzero_mask = tile > 1e-5
                        if nonzero_mask.any():
                            color_ratio[nonzero_mask] = img_tensor[ch, h_start:h_end, w_start:w_end][nonzero_mask] / tile[nonzero_mask]
                        result[ch, h_start:h_end, w_start:w_end] = enhanced_tile * color_ratio
        
        # Ensure values are in valid range
        result = torch.clamp(result, 0, 1)
        return result
    
    def forward(self, img):
        """Apply CLAHE effect to pizza image"""
        if torch.is_tensor(img):
            return self._apply_clahe_torch(img)
        else:
            return self._apply_clahe_pil(img)


class ExposureVariationEffect(nn.Module):
    """
    Simulate over-exposure and under-exposure conditions in pizza images,
    focusing on realistic lighting variations encountered in real-world settings.
    """
    
    def __init__(self, 
                 underexposure_prob=0.3,
                 overexposure_prob=0.3,
                 exposure_range=(0.3, 1.8),
                 vignette_prob=0.4):
        super().__init__()
        self.underexposure_prob = underexposure_prob
        self.overexposure_prob = overexposure_prob
        self.exposure_range = exposure_range
        self.vignette_prob = vignette_prob
    
    def _apply_vignette(self, img_tensor):
        """Apply vignette effect (darkened corners)"""
        c, h, w = img_tensor.shape
        
        # Create center coordinates
        center_y, center_x = h / 2, w / 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        # Calculate squared distance from center
        dist_squared = (x - center_x)**2 + (y - center_y)**2
        max_dist_squared = max(center_x**2, center_y**2) * 2
        
        # Create normalized vignette mask
        vignette_mask = 1 - torch.sqrt(dist_squared / max_dist_squared)
        
        # Adjust vignette intensity and smoothness
        intensity = random.uniform(0.6, 0.9)
        vignette_mask = torch.clamp(vignette_mask, 0, 1) ** random.uniform(0.5, 3.0)
        vignette_mask = intensity + (1 - intensity) * vignette_mask
        
        # Apply vignette mask to all channels
        for i in range(c):
            img_tensor[i] = img_tensor[i] * vignette_mask
        
        return img_tensor
    
    def forward(self, img):
        """Apply exposure variation effect to pizza image"""
        if torch.is_tensor(img):
            img_tensor = img.clone()
        else:
            img_tensor = TVF.to_tensor(img)
        
        # Decide which effect to apply
        r = random.random()
        if r < self.underexposure_prob:
            # Under-exposure effect
            exposure_factor = random.uniform(
                self.exposure_range[0], 1.0)
            img_tensor = img_tensor * exposure_factor
            
            # Add blue tint to simulate evening/night lighting
            if random.random() < 0.5:
                # Slightly increase blue channel
                blue_factor = random.uniform(1.0, 1.2)
                img_tensor[2] = torch.clamp(img_tensor[2] * blue_factor, 0, 1)
                
        elif r < self.underexposure_prob + self.overexposure_prob:
            # Over-exposure effect
            exposure_factor = random.uniform(
                1.0, self.exposure_range[1])
            img_tensor = torch.clamp(img_tensor * exposure_factor, 0, 1)
            
            # Add warm tint to simulate bright/sunny lighting
            if random.random() < 0.5:
                # Slightly increase red channel
                red_factor = random.uniform(1.0, 1.15)
                img_tensor[0] = torch.clamp(img_tensor[0] * red_factor, 0, 1)
        
        # Apply vignette with probability
        if random.random() < self.vignette_prob:
            img_tensor = self._apply_vignette(img_tensor)
        
        # Ensure values are in valid range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        if torch.is_tensor(img):
            return img_tensor
        else:
            return TVF.to_pil_image(img_tensor)


class PerspectiveTransformEffect(nn.Module):
    """
    Advanced perspective transformations for pizza images,
    simulating various viewing angles and camera positions.
    Combines rotation, shearing, and perspective changes.
    """
    
    def __init__(self, 
                 rotation_range=(-30, 30),
                 shear_range=(-15, 15),
                 perspective_strength=(0.05, 0.3),
                 border_handling='reflect'):
        super().__init__()
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.perspective_strength = perspective_strength
        self.border_handling = border_handling  # 'reflect', 'edge', 'black'
    
    def forward(self, img):
        """Apply perspective transformation to pizza image"""
        if torch.is_tensor(img):
            # Convert to PIL for easier perspective transformation
            pil_img = TVF.to_pil_image(img.cpu() if img.device.type != 'cpu' else img)
        else:
            pil_img = img.copy()
        
        # Get image dimensions
        width, height = pil_img.size
        
        # Random rotation angle
        rotation_angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        
        # Random shear angles
        shear_x = random.uniform(self.shear_range[0], self.shear_range[1])
        shear_y = random.uniform(self.shear_range[0], self.shear_range[1])
        
        # Random perspective strength
        perspective_factor = random.uniform(
            self.perspective_strength[0], self.perspective_strength[1])
        
        # Compute perspective transform matrix
        # First apply rotation
        angle_rad = math.radians(rotation_angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply shear
        shear_x_rad = math.radians(shear_x)
        shear_y_rad = math.radians(shear_y)
        shear_matrix = np.array([
            [1, math.tan(shear_x_rad), 0],
            [math.tan(shear_y_rad), 1, 0],
            [0, 0, 1]
        ])
        
        # Apply perspective distortion
        # Randomly warp one of the four corners
        corner_idx = random.randint(0, 3)
        perspective_matrix = np.eye(3)
        
        # Scale perspective effect with image size
        effect_scale = min(width, height) * perspective_factor
        
        if corner_idx == 0:  # Top-left
            perspective_matrix[0, 2] = random.uniform(0, effect_scale)
            perspective_matrix[1, 2] = random.uniform(0, effect_scale)
        elif corner_idx == 1:  # Top-right
            perspective_matrix[0, 2] = random.uniform(-effect_scale, 0)
            perspective_matrix[1, 2] = random.uniform(0, effect_scale)
        elif corner_idx == 2:  # Bottom-right
            perspective_matrix[0, 2] = random.uniform(-effect_scale, 0)
            perspective_matrix[1, 2] = random.uniform(-effect_scale, 0)
        else:  # Bottom-left
            perspective_matrix[0, 2] = random.uniform(0, effect_scale)
            perspective_matrix[1, 2] = random.uniform(-effect_scale, 0)
        
        # Compute combined transformation
        combined_matrix = rotation_matrix @ shear_matrix @ perspective_matrix
        
        # Convert to PIL's perspective transform format
        # (8 coefficients for projective transform)
        coeffs = (
            combined_matrix[0, 0], combined_matrix[0, 1], combined_matrix[0, 2],
            combined_matrix[1, 0], combined_matrix[1, 1], combined_matrix[1, 2],
            combined_matrix[2, 0], combined_matrix[2, 1]
        )
        
        # Apply transform
        transformed_img = pil_img.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0) if self.border_handling == 'black' else None
        )
        
        # Handle borders if necessary
        if self.border_handling == 'reflect':
            # Apply reflection padding
            transformed_img = ImageOps.expand(transformed_img, border=10, fill=(0, 0, 0))
            transformed_img = transformed_img.crop((10, 10, width + 10, height + 10))
        elif self.border_handling == 'edge':
            # Use edge replication
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([0, 0, width, height], fill=255)
            mask = mask.transform(
                (width, height),
                Image.PERSPECTIVE,
                coeffs,
                resample=Image.BICUBIC,
                fillcolor=0
            )
            
            blurred = transformed_img.filter(ImageFilter.GaussianBlur(radius=3))
            transformed_img = Image.composite(transformed_img, blurred, mask)
        
        if torch.is_tensor(img):
            # Convert back to tensor
            result = TVF.to_tensor(transformed_img)
            if img.device.type != 'cpu':
                result = result.to(img.device)
            return result
        else:
            return transformed_img


# Utility functions for lighting and perspective augmentations
def create_shadow_mask(img_size, num_shadows=3, shadow_dimension=0.5, blur_radius=20):
    """
    Create a shadow mask with random polygons
    
    Args:
        img_size: Tuple of (width, height)
        num_shadows: Number of shadow polygons to generate
        shadow_dimension: Maximum size of shadow as fraction of image dimension
        blur_radius: Blur radius for shadow edges
    
    Returns:
        PIL Image mask of shadows (grayscale)
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
        for i in range(num_points):
            # Decide if this point is on an edge
            if random.random() < 0.7:  # 70% chance to be on edge
                if random.random() < 0.5:  # Horizontal edges
                    x = random.randint(0, width)
                    y = 0 if random.random() < 0.5 else height
                else:  # Vertical edges
                    x = 0 if random.random() < 0.5 else width
                    y = random.randint(0, height)
            else:  # Random point within image
                x = random.randint(0, width)
                y = random.randint(0, height)
            points.append((x, y))
        
        # Draw the polygon
        draw.polygon(points, fill=shadow_val)
    
    # Apply blur for soft shadows
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask
