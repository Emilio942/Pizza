import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import gc

from scripts.augment_classes import PizzaBurningEffect, OvenEffect, PizzaSegmentEffect

def apply_basic_augmentation(img, device=None):
    """Apply basic augmentation to a single image"""
    if torch.is_tensor(img) or hasattr(transforms, 'Compose'):
        # Use torchvision transforms for more efficient processing
        basic_transforms = [
            # Various rotations
            transforms.RandomApply([
                transforms.RandomRotation(180)
            ], p=0.9),
            
            # Various crops
            transforms.RandomApply([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1))
            ], p=0.8),
            
            # Flips
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1)
            ], p=0.6),
            
            # Color variations
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, 
                    saturation=0.4, hue=0.2
                )
            ], p=0.8),
            
            # Perspective transformations
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.2)
            ], p=0.3),
            
            # Various filters
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
            ], p=0.4),
            
            # Sharpening
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2)
            ], p=0.3),
        ]
        
        # Apply a random subset of transformations
        transform_list = []
        for transform in basic_transforms:
            if random.random() < 0.5:  # 50% chance for each transform
                transform_list.append(transform)
        
        # Ensure at least one transformation is used
        if not transform_list:
            transform_list.append(random.choice(basic_transforms))
        
        # Create composition and apply
        composed = transforms.Compose(transform_list)
        
        # Apply to PIL image or Tensor depending on input
        if isinstance(img, torch.Tensor):
            if device:
                img = img.to(device)
            pil_img = TVF.to_pil_image(img.cpu() if img.device.type != 'cpu' else img)
            augmented = composed(pil_img)
            augmented = TVF.to_tensor(augmented)
            if device:
                augmented = augmented.to(device)
            return augmented
        else:
            # PIL image input
            augmented = composed(img)
            return augmented
    else:
        # Fallback to traditional PIL-based augmentation (similar to original)
        augmented_images = []
        
        # Original image
        augmented_images.append(img.copy())
        
        # Horizontal flip
        if random.random() > 0.5:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(flipped)
        
        # Rotation (slight)
        for angle in [random.uniform(-20, 20) for _ in range(2)]:
            rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False)
            augmented_images.append(rotated)
        
        # Brightness variation
        brightness_factor = random.uniform(0.8, 1.2)
        brightness = ImageEnhance.Brightness(img).enhance(brightness_factor)
        augmented_images.append(brightness)
        
        # Contrast variation
        contrast_factor = random.uniform(0.8, 1.2)
        contrast = ImageEnhance.Contrast(img).enhance(contrast_factor)
        augmented_images.append(contrast)
        
        # Saturation variation
        saturation_factor = random.uniform(0.8, 1.2)
        saturation = ImageEnhance.Color(img).enhance(saturation_factor)
        augmented_images.append(saturation)
        
        # Sharpness variation
        sharpness_factor = random.uniform(0.8, 1.5)
        sharpness = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
        augmented_images.append(sharpness)
        
        # Gaussian blur
        sigma = random.uniform(0.5, 1.0)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        augmented_images.append(blurred)
        
        # Random crop and resize
        width, height = img.size
        crop_size = int(min(width, height) * random.uniform(0.8, 0.95))
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        cropped = img.crop((left, top, left + crop_size, top + crop_size))
        resized = cropped.resize((width, height), Image.BICUBIC)
        augmented_images.append(resized)
        
        # Return a random augmentation
        return random.choice(augmented_images)

def apply_burning_augmentation(img, device=None):
    """Apply burning effects to an image"""
    if not torch.is_tensor(img) and not hasattr(torch, 'Tensor'):
        raise ValueError("PyTorch is required for burning augmentation")
    
    # Create burn effect with random settings
    burn_intensity_min = random.uniform(0.1, 0.3)
    burn_intensity_max = random.uniform(0.4, 0.8)
    burn_pattern = random.choice(['random', 'edge', 'spot', 'streak'])
    
    burning_effect = PizzaBurningEffect(
        burn_intensity_min=burn_intensity_min,
        burn_intensity_max=burn_intensity_max,
        burn_pattern=burn_pattern
    )
    if device:
        burning_effect = burning_effect.to(device)
    
    # Apply burning effect
    if isinstance(img, torch.Tensor):
        if device:
            img = img.to(device)
    else:
        img = TVF.to_tensor(img)
        if device:
            img = img.to(device)
    
    result = burning_effect(img)
    
    # Randomly apply oven effect too
    if random.random() < 0.5:
        oven_effect = OvenEffect(
            effect_strength=random.uniform(0.5, 1.0),
            scipy_available=hasattr(np, 'ndimage')
        )
        if device:
            oven_effect = oven_effect.to(device)
        result = oven_effect(result)
    
    return result

def apply_mixup(img1, img2, device=None, alpha=0.3):
    """Apply MixUp augmentation to two images"""
    if not torch.is_tensor(img1) and not hasattr(torch, 'Tensor'):
        raise ValueError("PyTorch is required for MixUp augmentation")
    
    # Ensure both images are tensors and on the right device
    if not isinstance(img1, torch.Tensor):
        img1 = TVF.to_tensor(img1)
        if device:
            img1 = img1.to(device)
    elif device:
        img1 = img1.to(device)
    
    if not isinstance(img2, torch.Tensor):
        img2 = TVF.to_tensor(img2)
        if device:
            img2 = img2.to(device)
    elif device:
        img2 = img2.to(device)
    
    # Ensure both images have the same size
    if img1.shape != img2.shape:
        img2 = F.interpolate(img2.unsqueeze(0), size=(img1.shape[1], img1.shape[2]), 
                            mode='bilinear', align_corners=False).squeeze(0)
    
    # Generate mix parameter
    lam = np.random.beta(alpha, alpha)
    
    # Mix the images
    mixed = lam * img1 + (1 - lam) * img2
    
    return mixed

def apply_cutmix(img1, img2, device=None):
    """Apply CutMix augmentation to two images"""
    if not torch.is_tensor(img1) and not hasattr(torch, 'Tensor'):
        raise ValueError("PyTorch is required for CutMix augmentation")
    
    # Ensure both images are tensors and on the right device
    if not isinstance(img1, torch.Tensor):
        img1 = TVF.to_tensor(img1)
        if device:
            img1 = img1.to(device)
    elif device:
        img1 = img1.to(device)
    
    if not isinstance(img2, torch.Tensor):
        img2 = TVF.to_tensor(img2)
        if device:
            img2 = img2.to(device)
    elif device:
        img2 = img2.to(device)
    
    # Ensure both images have the same size
    if img1.shape != img2.shape:
        img2 = F.interpolate(img2.unsqueeze(0), size=(img1.shape[1], img1.shape[2]), 
                            mode='bilinear', align_corners=False).squeeze(0)
    
    # Copy the first image
    result = img1.clone()
    
    h, w = img1.shape[1], img1.shape[2]
    center_x, center_y = w // 2, h // 2
    
    # Generate coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=img1.device),
        torch.arange(w, device=img1.device),
        indexing='ij'
    )
    
    # Decide between wedge or circular segment
    if random.random() < 0.5:
        # Circular segment
        radius = random.uniform(0.3, 0.7) * min(h, w) / 2
        dist = torch.sqrt(((x_coords - center_x)**2 + (y_coords - center_y)**2).float())
        mask = (dist <= radius)
    else:
        # Wedge-shaped segment
        start_angle = random.uniform(0, 2 * np.pi)
        angle_width = random.uniform(np.pi/6, np.pi/2)  # 30 to 90 degrees
        
        # Calculate angle for all pixels simultaneously
        delta_x = (x_coords - center_x).float()
        delta_y = (y_coords - center_y).float()
        angles = torch.atan2(delta_y, delta_x)
        
        # Normalize angles to [0, 2π]
        angles = torch.where(angles < 0, angles + 2 * np.pi, angles)
        
        # Mask for angle range
        if start_angle + angle_width <= 2 * np.pi:
            mask = (angles >= start_angle) & (angles <= start_angle + angle_width)
        else:
            # Handle overflow
            mask = (angles >= start_angle) | (angles <= (start_angle + angle_width) % (2 * np.pi))
    
    # Expand mask for all channels
    mask = mask.unsqueeze(0).expand_as(result)
    
    # Apply the mask
    result = torch.where(mask, img2, result)
    
    return result

def apply_burning_progression(img, num_steps=5, device=None):
    """Generate a series of images with increasing burning level"""
    if not torch.is_tensor(img) and not hasattr(torch, 'Tensor'):
        raise ValueError("PyTorch is required for burning progression augmentation")
    
    # Ensure image is a tensor and on the right device
    if not isinstance(img, torch.Tensor):
        img = TVF.to_tensor(img)
        if device:
            img = img.to(device)
    elif device:
        img = img.to(device)
    
    results = [img.clone()]  # Original image as first
    
    # Create burning effects with different intensities directly from original
    # (Not cumulative, to avoid unrealistic burning)
    for i in range(num_steps):
        # Increasing burning level
        intensity_min = 0.1 + (i * 0.15)
        intensity_max = 0.2 + (i * 0.15)
        
        burn_effect = PizzaBurningEffect(
            burn_intensity_min=intensity_min,
            burn_intensity_max=intensity_max
        )
        if device:
            burn_effect = burn_effect.to(device)
        
        # Apply effect directly to the original, not the previous stage
        burnt_img = burn_effect(img.clone())
        results.append(burnt_img)
    
    return results

def apply_segment_augmentation(img, device=None):
    """Apply segment-based effects to an image"""
    if not torch.is_tensor(img) and not hasattr(torch, 'Tensor'):
        raise ValueError("PyTorch is required for segment-based augmentation")
    
    if device is None and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device is None:
        device = torch.device('cpu')
    
    # Create segment effect
    segment_effect = PizzaSegmentEffect(
        device=device,
        burning_min=random.uniform(0, 0.2),
        burning_max=random.uniform(0.5, 0.9)
    )
    
    # Apply effect
    if isinstance(img, torch.Tensor):
        img = img.to(device)
    else:
        img = TVF.to_tensor(img).to(device)
    
    result = segment_effect(img)
    
    return result


# ================ LIGHTING AND PERSPECTIVE AUGMENTATION FUNCTIONS ================

def apply_lighting_augmentation(img, device=None):
    """
    Apply realistic lighting condition augmentations to pizza images.
    
    This function applies a combination of:
    - Directional lighting (spotlights, side lighting, overhead, etc.)
    - Over/underexposure variations
    - Shadow effects with various patterns and directions
    - CLAHE-like local contrast enhancement to recover details
    - Color temperature variations simulating different environments
    
    Args:
        img: PIL Image or torch.Tensor
        device: torch device (for tensor operations)
        
    Returns:
        Augmented image (same type as input)
    """
    from scripts.augment_classes import DirectionalLightEffect, CLAHEEffect, ExposureVariationEffect, create_shadow_mask

    # Define lighting environments that will determine parameter combinations
    lighting_environments = {
        'bright_daylight': {
            'light_position': 'overhead',
            'light_intensity_range': (1.0, 1.5),
            'shadow_intensity_range': (0.6, 0.8),
            'underexposure_prob': 0.1,
            'overexposure_prob': 0.6,
            'light_color': (1.0, 0.98, 0.9),  # Slightly warm
            'clahe_strength': (2.0, 4.0),
            'shadow_type': 'directional',
            'shadow_direction': 'random'
        },
        'indoor_restaurant': {
            'light_position': 'side',
            'light_intensity_range': (0.8, 1.2),
            'shadow_intensity_range': (0.5, 0.7),
            'underexposure_prob': 0.3,
            'overexposure_prob': 0.2,
            'light_color': (1.0, 0.95, 0.8),  # Warm indoor lighting
            'clahe_strength': (3.0, 5.0),
            'shadow_type': 'object',
            'shadow_direction': None
        },
        'evening_mood': {
            'light_position': 'side',
            'light_intensity_range': (0.6, 0.9),
            'shadow_intensity_range': (0.4, 0.6),
            'underexposure_prob': 0.6,
            'overexposure_prob': 0.1,
            'light_color': (0.9, 0.9, 1.0),  # Slightly cool
            'clahe_strength': (4.0, 6.0),
            'shadow_type': 'edge',
            'shadow_direction': None
        },
        'dim_home_kitchen': {
            'light_position': 'corner',
            'light_intensity_range': (0.5, 0.8),
            'shadow_intensity_range': (0.4, 0.5),
            'underexposure_prob': 0.7,
            'overexposure_prob': 0.0,
            'light_color': (1.0, 0.9, 0.8),  # Warm but dim
            'clahe_strength': (3.0, 6.0),
            'shadow_type': 'directional',
            'shadow_direction': 'top'
        }
    }
    
    # Randomly select an environment
    environment = random.choice(list(lighting_environments.values()))
    
    # Create augmentation objects with environment-specific parameters
    directional_light = DirectionalLightEffect(
        light_intensity_min=environment['light_intensity_range'][0],
        light_intensity_max=environment['light_intensity_range'][1],
        shadow_intensity_min=environment['shadow_intensity_range'][0],
        shadow_intensity_max=environment['shadow_intensity_range'][1],
        light_position=environment['light_position'],
        light_color=environment['light_color'],
        specular_highlight_prob=0.3
    )
    
    clahe_effect = CLAHEEffect(
        clip_limit=random.uniform(environment['clahe_strength'][0], environment['clahe_strength'][1]),
        tile_grid_size=(random.randint(4, 10), random.randint(4, 10)),
        contrast_limit=(0.7, 1.5),
        detail_enhancement=0.4  # Higher detail enhancement for pizza textures
    )
    
    exposure_variation = ExposureVariationEffect(
        underexposure_prob=environment['underexposure_prob'],
        overexposure_prob=environment['overexposure_prob'],
        exposure_range=(0.4, 1.7),
        vignette_prob=0.4,
        color_temp_variation=True,
        noise_prob=0.4  # Add noise in low-light conditions
    )
    
    # Convert torch tensor to PIL if needed for processing
    is_tensor = torch.is_tensor(img)
    if is_tensor:
        if device and img.device != device:
            img = img.to(device)
        img_pil = None
    else:
        img_pil = img.copy()
    
    # Decide which effects to apply and in what order
    effects = []
    
    # Determine which effects to use based on probabilities
    use_directional = random.random() < 0.7
    use_clahe = random.random() < 0.5
    use_exposure = random.random() < 0.8
    use_shadow = random.random() < 0.6
    
    # Add selected effects to list
    if use_directional:
        effects.append(('directional', directional_light))
    if use_clahe:
        effects.append(('clahe', clahe_effect))
    if use_exposure:
        effects.append(('exposure', exposure_variation))
    
    # Ensure at least one effect is applied
    if not effects:
        if random.random() < 0.5:
            effects.append(('directional', directional_light))
        else:
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
            
            # Create and apply shadow mask with environment-specific parameters
            shadow_mask = create_shadow_mask(
                pil_img.size, 
                num_shadows=random.randint(1, 3),
                shadow_dimension=random.uniform(0.3, 0.7),
                blur_radius=random.randint(10, 30),
                shadow_type=environment['shadow_type'],
                direction=environment['shadow_direction']
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
            # Create and apply shadow mask with environment-specific parameters
            shadow_mask = create_shadow_mask(
                result.size, 
                num_shadows=random.randint(1, 3),
                shadow_dimension=random.uniform(0.3, 0.7),
                blur_radius=random.randint(10, 30),
                shadow_type=environment['shadow_type'],
                direction=environment['shadow_direction']
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
    - Realistic viewing angle variations (overhead, table-level, closeup, angled)
    - Natural rotation and shearing adjustments
    - Perspective distortions mimicking real camera positions
    - Optional zoom effects for different composition styles
    
    Args:
        img: PIL Image or torch.Tensor
        device: torch device (for tensor operations)
        
    Returns:
        Augmented image (same type as input)
    """
    from scripts.augment_classes import PerspectiveTransformEffect
    
    # Define realistic viewing scenarios
    viewing_scenarios = {
        'overhead_photo': {
            'view_angle': 'overhead',
            'rotation_range': (-10, 10),
            'shear_range': (-5, 5),
            'perspective_strength': (0.01, 0.1),
            'zoom_range': (0.95, 1.05),
            'border_handling': 'reflect'
        },
        'restaurant_table': {
            'view_angle': 'table',
            'rotation_range': (-15, 15),
            'shear_range': (-10, 10),
            'perspective_strength': (0.1, 0.2),
            'zoom_range': (0.9, 1.0),
            'border_handling': 'edge'
        },
        'closeup_detail': {
            'view_angle': 'closeup',
            'rotation_range': (-5, 5),
            'shear_range': (-5, 5),
            'perspective_strength': (0.05, 0.15),
            'zoom_range': (1.05, 1.2),
            'border_handling': 'reflect'
        },
        'dramatic_angle': {
            'view_angle': 'angle',
            'rotation_range': (-25, 25),
            'shear_range': (-15, 15),
            'perspective_strength': (0.15, 0.25),
            'zoom_range': (0.85, 1.0),
            'border_handling': random.choice(['reflect', 'edge', 'black'])
        }
    }
    
    # Randomly select a viewing scenario
    scenario = random.choice(list(viewing_scenarios.values()))
    
    # Create perspective transform with scenario-specific parameters
    perspective_effect = PerspectiveTransformEffect(
        rotation_range=scenario['rotation_range'],
        shear_range=scenario['shear_range'],
        perspective_strength=scenario['perspective_strength'],
        border_handling=scenario['border_handling'],
        view_angle=scenario['view_angle'],
        zoom_range=scenario['zoom_range']
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
    
    This function intelligently combines both lighting and perspective effects
    in a way that produces realistic and coherent results. It ensures that
    the lighting direction and shadows are consistent with the perspective view.
    
    Args:
        img: PIL Image or torch.Tensor
        device: torch device (for tensor operations)
        
    Returns:
        Augmented image (same type as input)
    """
    is_tensor = torch.is_tensor(img)
    
    # Define specific combinations that work well together
    combined_scenarios = {
        'overhead_bright': {
            'perspective': 'overhead_photo',
            'lighting': 'bright_daylight',
            'perspective_first': True
        },
        'table_restaurant': {
            'perspective': 'restaurant_table',
            'lighting': 'indoor_restaurant',
            'perspective_first': False
        },
        'closeup_evening': {
            'perspective': 'closeup_detail',
            'lighting': 'evening_mood',
            'perspective_first': True
        },
        'angle_dim': {
            'perspective': 'dramatic_angle',
            'lighting': 'dim_home_kitchen',
            'perspective_first': False
        },
        'table_daylight': {
            'perspective': 'restaurant_table',
            'lighting': 'bright_daylight',
            'perspective_first': True
        },
        'overhead_dim': {
            'perspective': 'overhead_photo',
            'lighting': 'dim_home_kitchen',
            'perspective_first': False
        }
    }
    
    # Select a combined scenario
    scenario = random.choice(list(combined_scenarios.values()))
    
    # Define perspective scenarios (duplicate from apply_perspective_augmentation)
    perspective_scenarios = {
        'overhead_photo': {
            'view_angle': 'overhead',
            'rotation_range': (-10, 10),
            'shear_range': (-5, 5),
            'perspective_strength': (0.01, 0.1),
            'zoom_range': (0.95, 1.05),
            'border_handling': 'reflect'
        },
        'restaurant_table': {
            'view_angle': 'table',
            'rotation_range': (-15, 15),
            'shear_range': (-10, 10),
            'perspective_strength': (0.1, 0.2),
            'zoom_range': (0.9, 1.0),
            'border_handling': 'edge'
        },
        'closeup_detail': {
            'view_angle': 'closeup',
            'rotation_range': (-5, 5),
            'shear_range': (-5, 5),
            'perspective_strength': (0.05, 0.15),
            'zoom_range': (1.05, 1.2),
            'border_handling': 'reflect'
        },
        'dramatic_angle': {
            'view_angle': 'angle',
            'rotation_range': (-25, 25),
            'shear_range': (-15, 15),
            'perspective_strength': (0.15, 0.25),
            'zoom_range': (0.85, 1.0),
            'border_handling': random.choice(['reflect', 'edge', 'black'])
        }
    }
    
    # Define lighting environments (duplicate from apply_lighting_augmentation)
    lighting_environments = {
        'bright_daylight': {
            'light_position': 'overhead',
            'light_intensity_range': (1.0, 1.5),
            'shadow_intensity_range': (0.6, 0.8),
            'underexposure_prob': 0.1,
            'overexposure_prob': 0.6,
            'light_color': (1.0, 0.98, 0.9),  # Slightly warm
            'clahe_strength': (2.0, 4.0),
            'shadow_type': 'directional',
            'shadow_direction': 'random'
        },
        'indoor_restaurant': {
            'light_position': 'side',
            'light_intensity_range': (0.8, 1.2),
            'shadow_intensity_range': (0.5, 0.7),
            'underexposure_prob': 0.3,
            'overexposure_prob': 0.2,
            'light_color': (1.0, 0.95, 0.8),  # Warm indoor lighting
            'clahe_strength': (3.0, 5.0),
            'shadow_type': 'object',
            'shadow_direction': None
        },
        'evening_mood': {
            'light_position': 'side',
            'light_intensity_range': (0.6, 0.9),
            'shadow_intensity_range': (0.4, 0.6),
            'underexposure_prob': 0.6,
            'overexposure_prob': 0.1,
            'light_color': (0.9, 0.9, 1.0),  # Slightly cool
            'clahe_strength': (4.0, 6.0),
            'shadow_type': 'edge',
            'shadow_direction': None
        },
        'dim_home_kitchen': {
            'light_position': 'corner',
            'light_intensity_range': (0.5, 0.8),
            'shadow_intensity_range': (0.4, 0.5),
            'underexposure_prob': 0.7,
            'overexposure_prob': 0.0,
            'light_color': (1.0, 0.9, 0.8),  # Warm but dim
            'clahe_strength': (3.0, 6.0),
            'shadow_type': 'directional',
            'shadow_direction': 'top'
        }
    }
    
    # Create perspective effect
    perspective_config = perspective_scenarios[scenario['perspective']]
    from scripts.augment_classes import PerspectiveTransformEffect
    
    perspective_effect = PerspectiveTransformEffect(
        rotation_range=perspective_config['rotation_range'],
        shear_range=perspective_config['shear_range'],
        perspective_strength=perspective_config['perspective_strength'],
        border_handling=perspective_config['border_handling'],
        view_angle=perspective_config['view_angle'],
        zoom_range=perspective_config['zoom_range']
    )
    
    # Create lighting effects
    lighting_config = lighting_environments[scenario['lighting']]
    from scripts.augment_classes import DirectionalLightEffect, CLAHEEffect, ExposureVariationEffect, create_shadow_mask
    
    # Adjust lighting based on perspective view
    # For example, if we're using a top-down perspective, light should come from above
    perspective_view = perspective_config['view_angle']
    
    # Make lighting consistent with perspective view
    if perspective_view == 'overhead':
        lighting_config['light_position'] = 'overhead'
        lighting_config['shadow_direction'] = 'random'
    elif perspective_view == 'table':
        lighting_config['light_position'] = random.choice(['side', 'front'])
        lighting_config['shadow_direction'] = random.choice(['left', 'right'])
    elif perspective_view == 'angle':
        lighting_config['light_position'] = 'corner'
        lighting_config['shadow_direction'] = random.choice(['left', 'right'])
    
    directional_light = DirectionalLightEffect(
        light_intensity_min=lighting_config['light_intensity_range'][0],
        light_intensity_max=lighting_config['light_intensity_range'][1],
        shadow_intensity_min=lighting_config['shadow_intensity_range'][0],
        shadow_intensity_max=lighting_config['shadow_intensity_range'][1],
        light_position=lighting_config['light_position'],
        light_color=lighting_config['light_color'],
        specular_highlight_prob=0.3
    )
    
    clahe_effect = CLAHEEffect(
        clip_limit=random.uniform(lighting_config['clahe_strength'][0], lighting_config['clahe_strength'][1]),
        tile_grid_size=(random.randint(4, 10), random.randint(4, 10)),
        contrast_limit=(0.7, 1.5),
        detail_enhancement=0.4
    )
    
    exposure_variation = ExposureVariationEffect(
        underexposure_prob=lighting_config['underexposure_prob'],
        overexposure_prob=lighting_config['overexposure_prob'],
        exposure_range=(0.4, 1.7),
        vignette_prob=0.4,
        color_temp_variation=True,
        noise_prob=0.4
    )
    
    # Decide which lighting effects to apply
    lighting_effects = []
    
    # Add lighting effects with appropriate probabilities
    if random.random() < 0.7:
        lighting_effects.append(('directional', directional_light))
    if random.random() < 0.5:
        lighting_effects.append(('clahe', clahe_effect))
    if random.random() < 0.8:
        lighting_effects.append(('exposure', exposure_variation))
    
    # Ensure at least one lighting effect is applied
    if not lighting_effects:
        if random.random() < 0.5:
            lighting_effects.append(('directional', directional_light))
        else:
            lighting_effects.append(('exposure', exposure_variation))
    
    # Shuffle lighting effects
    random.shuffle(lighting_effects)
    
    # Process image based on the order specified in the scenario
    if scenario['perspective_first']:
        # Apply perspective first, then lighting
        if is_tensor:
            if device and img.device != device:
                img = img.to(device)
            # Apply perspective
            result = perspective_effect(img)
            
            # Apply lighting effects sequentially
            for effect_name, effect in lighting_effects:
                result = effect(result)
                
            # Optionally apply shadow mask
            if random.random() < 0.6:
                # Convert to PIL temporarily
                pil_img = Image.fromarray(
                    (result.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                
                # Create and apply shadow mask
                shadow_mask = create_shadow_mask(
                    pil_img.size, 
                    num_shadows=random.randint(1, 3),
                    shadow_dimension=random.uniform(0.3, 0.7),
                    blur_radius=random.randint(10, 30),
                    shadow_type=lighting_config['shadow_type'],
                    direction=lighting_config['shadow_direction']
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
            # Apply perspective
            result = perspective_effect(img.copy())
            
            # Apply lighting effects sequentially
            for effect_name, effect in lighting_effects:
                result = effect(result)
                
            # Optionally apply shadow mask
            if random.random() < 0.6:
                # Create and apply shadow mask
                shadow_mask = create_shadow_mask(
                    result.size, 
                    num_shadows=random.randint(1, 3),
                    shadow_dimension=random.uniform(0.3, 0.7),
                    blur_radius=random.randint(10, 30),
                    shadow_type=lighting_config['shadow_type'],
                    direction=lighting_config['shadow_direction']
                )
                
                # Apply mask to image
                result = Image.composite(
                    result, 
                    ImageOps.colorize(shadow_mask, (0, 0, 0), (255, 255, 255)),
                    shadow_mask
                )
    else:
        # Apply lighting first, then perspective
        if is_tensor:
            if device and img.device != device:
                img = img.to(device)
            
            # Apply lighting effects sequentially
            result = img.clone()
            for effect_name, effect in lighting_effects:
                result = effect(result)
                
            # Optionally apply shadow mask
            if random.random() < 0.6:
                # Convert to PIL temporarily
                pil_img = Image.fromarray(
                    (result.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                
                # Create and apply shadow mask
                shadow_mask = create_shadow_mask(
                    pil_img.size, 
                    num_shadows=random.randint(1, 3),
                    shadow_dimension=random.uniform(0.3, 0.7),
                    blur_radius=random.randint(10, 30),
                    shadow_type=lighting_config['shadow_type'],
                    direction=lighting_config['shadow_direction']
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
            
            # Apply perspective
            result = perspective_effect(result)
        else:
            # Apply lighting effects sequentially
            result = img.copy()
            for effect_name, effect in lighting_effects:
                result = effect(result)
                
            # Optionally apply shadow mask
            if random.random() < 0.6:
                # Create and apply shadow mask
                shadow_mask = create_shadow_mask(
                    result.size, 
                    num_shadows=random.randint(1, 3),
                    shadow_dimension=random.uniform(0.3, 0.7),
                    blur_radius=random.randint(10, 30),
                    shadow_type=lighting_config['shadow_type'],
                    direction=lighting_config['shadow_direction']
                )
                
                # Apply mask to image
                result = Image.composite(
                    result, 
                    ImageOps.colorize(shadow_mask, (0, 0, 0), (255, 255, 255)),
                    shadow_mask
                )
            
            # Apply perspective
            result = perspective_effect(result)
    
    return result
