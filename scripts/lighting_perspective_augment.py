#!/usr/bin/env python3
"""
Implementation of specialized lighting and perspective augmentation functions
for the pizza dataset. These functions integrate the custom augmentation
classes to provide realistic variations in lighting conditions and perspective.
"""

import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from scripts.lighting_perspective import (
    DirectionalLightEffect,
    CLAHEEffect,
    ExposureVariationEffect,
    PerspectiveTransformEffect,
    create_shadow_mask
)

def apply_lighting_augmentation(img, device=None):
    """
    Apply realistic lighting condition augmentations to pizza images.
    
    This function applies a combination of:
    - Directional lighting (spotlights, side lighting)
    - Over/underexposure variations
    - Shadow effects
    - CLAHE-like local contrast enhancement
    
    Args:
        img: PIL Image or torch.Tensor
        device: torch device (for tensor operations)
        
    Returns:
        Augmented image (same type as input)
    """
    # Create augmentation objects
    directional_light = DirectionalLightEffect(
        light_intensity_min=0.6,
        light_intensity_max=1.5,
        shadow_intensity_min=0.5,
        shadow_intensity_max=0.9
    )
    
    clahe_effect = CLAHEEffect(
        clip_limit=random.uniform(2.0, 6.0),
        tile_grid_size=(random.randint(4, 10), random.randint(4, 10))
    )
    
    exposure_variation = ExposureVariationEffect(
        underexposure_prob=0.4,
        overexposure_prob=0.4,
        exposure_range=(0.4, 1.7),
        vignette_prob=0.5
    )
    
    # Convert torch tensor to PIL if needed for processing
    is_tensor = torch.is_tensor(img)
    if is_tensor:
        if device and img.device != device:
            img = img.to(device)
        img_pil = None
    else:
        img_pil = img.copy()
    
    # Randomly select which effects to apply and in what order
    effects = []
    
    # Decide which effects to use
    use_directional = random.random() < 0.6
    use_clahe = random.random() < 0.4
    use_exposure = random.random() < 0.7
    use_shadow = random.random() < 0.5
    
    # Add selected effects to list
    if use_directional:
        effects.append(('directional', directional_light))
    if use_clahe:
        effects.append(('clahe', clahe_effect))
    if use_exposure:
        effects.append(('exposure', exposure_variation))
    
    # Shuffle the effects to apply them in random order
    random.shuffle(effects)
    
    # Apply the effects sequentially
    if is_tensor:
        result = img.clone()
        for effect_name, effect in effects:
            result = effect(result)
            
        # Apply shadow mask if selected (requires PIL conversion)
        if use_shadow:
            # Convert to PIL temporarily
            pil_img = Image.fromarray(
                (result.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            
            # Create and apply shadow mask
            shadow_mask = create_shadow_mask(
                pil_img.size, 
                num_shadows=random.randint(1, 3),
                shadow_dimension=random.uniform(0.3, 0.7),
                blur_radius=random.randint(10, 30)
            )
            
            # Apply mask to image
            pil_img = Image.composite(
                pil_img, 
                ImageOps.colorize(shadow_mask, (0, 0, 0), (255, 255, 255)),
                shadow_mask
            )
            
            # Convert back to tensor
            result = torch.from_numpy(
                np.array(pil_img).astype(np.float32) / 255.0
            ).permute(2, 0, 1)
            
            if device:
                result = result.to(device)
    else:
        result = img_pil
        for effect_name, effect in effects:
            result = effect(result)
            
        # Apply shadow mask if selected
        if use_shadow:
            # Create and apply shadow mask
            shadow_mask = create_shadow_mask(
                result.size, 
                num_shadows=random.randint(1, 3),
                shadow_dimension=random.uniform(0.3, 0.7),
                blur_radius=random.randint(10, 30)
            )
            
            # Apply mask to image
            result = Image.composite(
                result, 
                ImageOps.colorize(shadow_mask, (0, 0, 0), (255, 255, 255)),
                shadow_mask
            )
    
    return result

def apply_perspective_augmentation(img, device=None):
    """
    Apply advanced perspective transformations to pizza images.
    
    This function applies a combination of:
    - Realistic viewing angle variations
    - Rotation and shearing
    - Perspective distortions
    
    Args:
        img: PIL Image or torch.Tensor
        device: torch device (for tensor operations)
        
    Returns:
        Augmented image (same type as input)
    """
    # Create perspective transform with varying parameters
    perspective_effect = PerspectiveTransformEffect(
        rotation_range=(-25, 25),
        shear_range=(-12, 12),
        perspective_strength=(0.05, 0.25),
        border_handling=random.choice(['reflect', 'edge', 'black'])
    )
    
    # Apply transformation
    if torch.is_tensor(img):
        if device and img.device != device:
            img = img.to(device)
        result = perspective_effect(img)
    else:
        result = perspective_effect(img.copy())
    
    return result

def apply_combined_light_perspective_augmentation(img, device=None):
    """
    Apply combined lighting and perspective augmentations to pizza images.
    
    This function applies both lighting and perspective effects with
    appropriate probabilities to create realistic variations of pizza images.
    
    Args:
        img: PIL Image or torch.Tensor
        device: torch device (for tensor operations)
        
    Returns:
        Augmented image (same type as input)
    """
    is_tensor = torch.is_tensor(img)
    
    # Decide which effects to apply
    apply_lighting = random.random() < 0.8
    apply_perspective = random.random() < 0.7
    
    # If neither effect is selected, choose one randomly
    if not apply_lighting and not apply_perspective:
        if random.random() < 0.5:
            apply_lighting = True
        else:
            apply_perspective = True
    
    # Apply effects in random order
    effects_order = []
    if apply_lighting:
        effects_order.append('lighting')
    if apply_perspective:
        effects_order.append('perspective')
    
    random.shuffle(effects_order)
    
    # Process image
    if is_tensor:
        result = img.clone()
    else:
        result = img.copy()
    
    for effect in effects_order:
        if effect == 'lighting':
            result = apply_lighting_augmentation(result, device)
        elif effect == 'perspective':
            result = apply_perspective_augmentation(result, device)
    
    return result
