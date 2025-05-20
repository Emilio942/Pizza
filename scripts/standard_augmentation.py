#!/usr/bin/env python3
"""
Standard Augmentation Pipeline for Pizza Dataset

This module defines a standard pipeline of augmentation techniques
for training the pizza classification model. It provides configurable
parameters and probabilities for each augmentation technique.

Usage:
    - Import the StandardAugmentationPipeline class
    - Create an instance with desired parameters
    - Use the transform() method to apply augmentations

Example:
    from scripts.standard_augmentation import StandardAugmentationPipeline
    
    # Create augmentation pipeline
    aug_pipeline = StandardAugmentationPipeline()
    
    # Apply to an image
    augmented_img = aug_pipeline.transform(original_img)
    
    # Use in training dataloader
    transform = aug_pipeline.get_transforms()
    train_dataset = YourDataset(data_dir, transform=transform)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TVF
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# Try to import custom augmentation classes if available
try:
    from scripts.augment_classes import PizzaBurningEffect, OvenEffect, PizzaSegmentEffect
    from scripts.augment_functions import (
        apply_burning_augmentation, 
        apply_mixup, apply_cutmix, 
        apply_burning_progression, apply_segment_augmentation
    )
    ADVANCED_AUGMENTATIONS_AVAILABLE = True
except ImportError:
    ADVANCED_AUGMENTATIONS_AVAILABLE = False


class StandardAugmentationPipeline:
    """
    Implements a standard augmentation pipeline for pizza images
    with configurable parameters and probabilities.
    
    The pipeline includes:
    - Geometric transformations (rotation, flipping, cropping)
    - Color adjustments (brightness, contrast, saturation)
    - Noise addition (Gaussian, salt & pepper)
    - Blurring and sharpening
    - Pizza-specific augmentations (burning effects)
    
    Each transformation has an associated probability and parameter range.
    """
    
    def __init__(
        self,
        image_size=224,
        apply_geometric=True,
        apply_color=True,
        apply_noise=True,
        apply_blur_sharpen=True,
        apply_pizza_specific=True,
        geometric_prob=0.8,
        color_prob=0.8,
        noise_prob=0.5,
        blur_sharpen_prob=0.5,
        pizza_specific_prob=0.3,
        rotation_range=(-25, 25),
        scale_range=(0.8, 1.0),
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_range=(-0.05, 0.05),
        blur_radius_range=(0.1, 2.0),
        noise_intensity=(0.01, 0.05),
        sharpness_range=(0.8, 1.5),
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.1,
        random_crop_prob=0.8,
        random_perspective_prob=0.3,
        use_advanced_augmentations=True,
    ):
        """
        Initialize the augmentation pipeline with configurable parameters.
        
        Args:
            image_size (int): Target image size
            apply_geometric (bool): Whether to apply geometric transformations
            apply_color (bool): Whether to apply color adjustments
            apply_noise (bool): Whether to apply noise
            apply_blur_sharpen (bool): Whether to apply blurring/sharpening
            apply_pizza_specific (bool): Whether to apply pizza-specific transforms
            geometric_prob (float): Probability of applying any geometric transform
            color_prob (float): Probability of applying any color adjustment
            noise_prob (float): Probability of applying noise
            blur_sharpen_prob (float): Probability of applying blur/sharpen
            pizza_specific_prob (float): Probability of applying pizza-specific transforms
            rotation_range (tuple): Range of rotation angles in degrees
            scale_range (tuple): Range of scaling factors
            brightness_range (tuple): Range of brightness adjustment factors
            contrast_range (tuple): Range of contrast adjustment factors
            saturation_range (tuple): Range of saturation adjustment factors
            hue_range (tuple): Range of hue adjustment factors
            blur_radius_range (tuple): Range of Gaussian blur radii
            noise_intensity (tuple): Range of noise intensity values
            sharpness_range (tuple): Range of sharpness adjustment factors
            horizontal_flip_prob (float): Probability of horizontal flip
            vertical_flip_prob (float): Probability of vertical flip
            random_crop_prob (float): Probability of random crop
            random_perspective_prob (float): Probability of perspective transform
            use_advanced_augmentations (bool): Whether to use advanced augmentations
        """
        self.image_size = image_size
        self.apply_geometric = apply_geometric
        self.apply_color = apply_color
        self.apply_noise = apply_noise
        self.apply_blur_sharpen = apply_blur_sharpen
        self.apply_pizza_specific = apply_pizza_specific and ADVANCED_AUGMENTATIONS_AVAILABLE
        
        self.geometric_prob = geometric_prob
        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.blur_sharpen_prob = blur_sharpen_prob
        self.pizza_specific_prob = pizza_specific_prob
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.blur_radius_range = blur_radius_range
        self.noise_intensity = noise_intensity
        self.sharpness_range = sharpness_range
        
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.random_crop_prob = random_crop_prob
        self.random_perspective_prob = random_perspective_prob
        
        self.use_advanced_augmentations = use_advanced_augmentations and ADVANCED_AUGMENTATIONS_AVAILABLE
        
        # Check if we can use torch or need PIL fallback
        self.use_torch = hasattr(transforms, 'Compose')
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        if self.use_torch:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
    
    def get_transforms(self):
        """
        Get a torchvision transforms Compose object that applies the augmentation pipeline.
        
        Returns:
            transforms.Compose: A composition of transform operations
        """
        if not self.use_torch:
            return self.transform  # Return function reference for PIL-based transformation
        
        transform_list = []
        
        # Standard resize to ensure consistent input size
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Geometric transformations
        if self.apply_geometric:
            geom_transforms = []
            
            # Rotation
            if self.rotation_range[1] > self.rotation_range[0]:
                geom_transforms.append(
                    transforms.RandomRotation(
                        self.rotation_range, 
                        interpolation=transforms.InterpolationMode.BICUBIC
                    )
                )
            
            # Random crop and resize
            if self.random_crop_prob > 0:
                geom_transforms.append(
                    transforms.RandomResizedCrop(
                        self.image_size,
                        scale=self.scale_range,
                        ratio=(0.9, 1.1),
                        interpolation=transforms.InterpolationMode.BICUBIC
                    )
                )
            
            # Horizontal flip
            if self.horizontal_flip_prob > 0:
                geom_transforms.append(transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob))
            
            # Vertical flip
            if self.vertical_flip_prob > 0:
                geom_transforms.append(transforms.RandomVerticalFlip(p=self.vertical_flip_prob))
            
            # Perspective
            if self.random_perspective_prob > 0:
                geom_transforms.append(
                    transforms.RandomPerspective(
                        distortion_scale=0.15,
                        p=self.random_perspective_prob
                    )
                )
            
            if geom_transforms:
                transform_list.append(
                    transforms.RandomApply(geom_transforms, p=self.geometric_prob)
                )
        
        # Color adjustments
        if self.apply_color:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=self.brightness_range[1] - self.brightness_range[0],
                        contrast=self.contrast_range[1] - self.contrast_range[0],
                        saturation=self.saturation_range[1] - self.saturation_range[0],
                        hue=self.hue_range[1] - self.hue_range[0]
                    )],
                    p=self.color_prob
                )
            )
        
        # Blur and sharpen
        if self.apply_blur_sharpen:
            blur_sharpen_transforms = []
            
            # Gaussian blur
            blur_sharpen_transforms.append(
                transforms.GaussianBlur(
                    kernel_size=5,
                    sigma=self.blur_radius_range
                )
            )
            
            # Sharpness adjustment
            blur_sharpen_transforms.append(
                transforms.RandomAdjustSharpness(
                    sharpness_factor=self.sharpness_range[1],
                    p=0.5
                )
            )
            
            if blur_sharpen_transforms:
                transform_list.append(
                    transforms.RandomApply(
                        blur_sharpen_transforms,
                        p=self.blur_sharpen_prob
                    )
                )
        
        # Add custom pizza-specific transform using Lambda
        if self.apply_pizza_specific and self.use_advanced_augmentations:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.Lambda(self._apply_pizza_specific_transform)],
                    p=self.pizza_specific_prob
                )
            )
        
        # Convert to tensor if the transforms don't already do that
        if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
            transform_list.append(transforms.ToTensor())
        
        # Normalize with ImageNet mean and std (common practice)
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
        return transforms.Compose(transform_list)
    
    def _apply_pizza_specific_transform(self, img):
        """
        Apply pizza-specific transformations (burning effect, etc.)
        
        Args:
            img (PIL.Image or torch.Tensor): Input image
            
        Returns:
            torch.Tensor: Transformed image
        """
        if not self.use_advanced_augmentations or not ADVANCED_AUGMENTATIONS_AVAILABLE:
            return img
        
        # Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        
        # Apply a random pizza-specific transformation
        transform_type = random.choice(['burning', 'oven', 'segment'])
        
        if transform_type == 'burning':
            # Apply burning effect
            return apply_burning_augmentation(img, device=self.device)
        elif transform_type == 'oven':
            # Apply oven effect
            oven_effect = OvenEffect(
                effect_strength=random.uniform(0.3, 0.7)
            )
            return oven_effect(img)
        elif transform_type == 'segment':
            # Apply segment effect
            return apply_segment_augmentation(img, device=self.device)
        
        return img
    
    def _add_noise(self, img):
        """
        Add random noise to the image
        
        Args:
            img (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Image with added noise
        """
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        
        # Get noise intensity
        intensity = random.uniform(self.noise_intensity[0], self.noise_intensity[1])
        
        # Determine noise type
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            # Gaussian noise
            noise = torch.randn_like(img) * intensity
            noisy_img = img + noise
            return torch.clamp(noisy_img, 0, 1)
        
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            noise = torch.rand_like(img)
            # Salt (white pixels)
            salt = noise < intensity / 2
            # Pepper (black pixels)
            pepper = noise > 1 - intensity / 2
            
            noisy_img = img.clone()
            noisy_img[salt] = 1.0
            noisy_img[pepper] = 0.0
            return noisy_img
        
        elif noise_type == 'speckle':
            # Speckle noise (multiplicative)
            noise = torch.randn_like(img) * intensity
            noisy_img = img * (1 + noise)
            return torch.clamp(noisy_img, 0, 1)
        
        return img
    
    def transform(self, img):
        """
        Apply the full augmentation pipeline to an image
        
        Args:
            img (PIL.Image or torch.Tensor): Input image
            
        Returns:
            PIL.Image or torch.Tensor: Augmented image
        """
        if self.use_torch:
            # Use torchvision transforms
            transform = self.get_transforms()
            
            # Handle PIL or tensor input
            if isinstance(img, Image.Image):
                return transform(img)
            elif isinstance(img, torch.Tensor):
                # Ensure the tensor is in the right format
                if img.dim() == 3 and img.shape[0] == 3:  # Already CHW format
                    return transform(img)
                elif img.dim() == 3 and img.shape[2] == 3:  # HWC format
                    img = img.permute(2, 0, 1)  # Convert to CHW
                    return transform(img)
        
        # Fallback to PIL-based transformations
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL
            if img.dim() == 3 and img.shape[0] == 3:  # CHW format
                pil_img = TVF.to_pil_image(img)
            elif img.dim() == 3 and img.shape[2] == 3:  # HWC format
                img = img.permute(2, 0, 1)  # Convert to CHW
                pil_img = TVF.to_pil_image(img)
            else:
                raise ValueError(f"Unsupported tensor shape: {img.shape}")
        else:
            pil_img = img
        
        # Ensure the image has the right size
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Apply geometric transformations
        if self.apply_geometric and random.random() < self.geometric_prob:
            # Rotation
            if random.random() < 0.7:
                angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
                pil_img = pil_img.rotate(angle, resample=Image.BICUBIC)
            
            # Random crop and resize
            if random.random() < self.random_crop_prob:
                width, height = pil_img.size
                scale = random.uniform(self.scale_range[0], self.scale_range[1])
                crop_size = int(min(width, height) * scale)
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                pil_img = pil_img.crop((left, top, left + crop_size, top + crop_size))
                pil_img = pil_img.resize((self.image_size, self.image_size), Image.BICUBIC)
            
            # Flips
            if random.random() < self.horizontal_flip_prob:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            if random.random() < self.vertical_flip_prob:
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Apply color adjustments
        if self.apply_color and random.random() < self.color_prob:
            # Brightness
            if random.random() < 0.7:
                factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
                pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
            
            # Contrast
            if random.random() < 0.7:
                factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
                pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)
            
            # Saturation
            if random.random() < 0.7:
                factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
                pil_img = ImageEnhance.Color(pil_img).enhance(factor)
        
        # Apply blur or sharpen
        if self.apply_blur_sharpen and random.random() < self.blur_sharpen_prob:
            if random.random() < 0.5:
                # Blur
                radius = random.uniform(self.blur_radius_range[0], self.blur_radius_range[1])
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            else:
                # Sharpen
                factor = random.uniform(self.sharpness_range[0], self.sharpness_range[1])
                pil_img = ImageEnhance.Sharpness(pil_img).enhance(factor)
        
        # Convert back to tensor for noise and pizza-specific transforms
        tensor_img = TVF.to_tensor(pil_img)
        
        # Apply noise
        if self.apply_noise and random.random() < self.noise_prob:
            tensor_img = self._add_noise(tensor_img)
        
        # Apply pizza-specific transformations
        if (self.apply_pizza_specific and self.use_advanced_augmentations and 
            random.random() < self.pizza_specific_prob and ADVANCED_AUGMENTATIONS_AVAILABLE):
            tensor_img = self._apply_pizza_specific_transform(tensor_img)
        
        # Normalize tensor
        tensor_img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(tensor_img)
        
        return tensor_img


def get_standard_augmentation_pipeline(image_size=224, intensity='medium'):
    """
    Factory function to create a standard augmentation pipeline with predefined settings.
    
    Args:
        image_size (int): Target image size
        intensity (str): Augmentation intensity level ('low', 'medium', 'high')
        
    Returns:
        StandardAugmentationPipeline: Configured augmentation pipeline
    """
    if intensity == 'low':
        return StandardAugmentationPipeline(
            image_size=image_size,
            geometric_prob=0.7,
            color_prob=0.7,
            noise_prob=0.3,
            blur_sharpen_prob=0.3,
            pizza_specific_prob=0.2,
            rotation_range=(-15, 15),
            scale_range=(0.9, 1.0),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            saturation_range=(0.9, 1.1),
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.0,
            random_perspective_prob=0.1,
        )
    elif intensity == 'high':
        return StandardAugmentationPipeline(
            image_size=image_size,
            geometric_prob=0.9,
            color_prob=0.9,
            noise_prob=0.7,
            blur_sharpen_prob=0.7,
            pizza_specific_prob=0.6,
            rotation_range=(-30, 30),
            scale_range=(0.7, 1.0),
            brightness_range=(0.7, 1.3),
            contrast_range=(0.7, 1.3),
            saturation_range=(0.7, 1.3),
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.2,
            random_perspective_prob=0.4,
        )
    else:  # medium (default)
        return StandardAugmentationPipeline(
            image_size=image_size,
        )


if __name__ == "__main__":
    # Example usage
    import sys
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load an example image if provided
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("Please provide an image path as argument")
        sys.exit(1)
    
    # Load image
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Create augmentation pipelines with different intensities
    aug_low = get_standard_augmentation_pipeline(intensity='low')
    aug_medium = get_standard_augmentation_pipeline(intensity='medium')
    aug_high = get_standard_augmentation_pipeline(intensity='high')
    
    # Apply augmentations
    img_low = aug_low.transform(img)
    img_medium = aug_medium.transform(img)
    img_high = aug_high.transform(img)
    
    # Convert tensors back to PIL images for display
    def tensor_to_pil(tensor):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        return TVF.to_pil_image(tensor)
    
    img_low_pil = tensor_to_pil(img_low)
    img_medium_pil = tensor_to_pil(img_medium)
    img_high_pil = tensor_to_pil(img_high)
    
    # Display images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_low_pil)
    plt.title('Low Intensity Augmentation')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_medium_pil)
    plt.title('Medium Intensity Augmentation')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(img_high_pil)
    plt.title('High Intensity Augmentation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()
