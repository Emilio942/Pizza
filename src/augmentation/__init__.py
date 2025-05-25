"""
Image augmentation functions for pizza classification.
"""

import numpy as np
import cv2
from typing import Union
import random
from PIL import Image


def _convert_to_numpy(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert PIL Image to numpy array if needed.
    
    Args:
        image: Input image (PIL Image or numpy array)
    
    Returns:
        Image as numpy array
    """
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


def _convert_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Image as PIL Image
    """
    return Image.fromarray(image)


def add_noise(image: Union[np.ndarray, Image.Image], noise_factor: float = 0.1) -> Image.Image:
    """
    Add random noise to an image.
    
    Args:
        image: Input image as numpy array or PIL Image
        noise_factor: Factor controlling noise intensity (0-1)
    
    Returns:
        Image with added noise as PIL Image
    """
    img_array = _convert_to_numpy(image)
    
    if len(img_array.shape) == 3:
        noise = np.random.randint(0, int(255 * noise_factor), img_array.shape, dtype=np.uint8)
    else:
        noise = np.random.randint(0, int(255 * noise_factor), img_array.shape, dtype=np.uint8)
    
    # Add noise and clip to valid range
    noisy_image = np.clip(img_array.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return _convert_to_pil(noisy_image)


def adjust_brightness(image: Union[np.ndarray, Image.Image], brightness_factor: float = 1.2) -> Image.Image:
    """
    Adjust brightness of an image.
    
    Args:
        image: Input image as numpy array or PIL Image
        brightness_factor: Brightness adjustment factor (>1 brighter, <1 darker)
    
    Returns:
        Brightness-adjusted image as PIL Image
    """
    img_array = _convert_to_numpy(image)
    adjusted = img_array.astype(np.float32) * brightness_factor
    result = np.clip(adjusted, 0, 255).astype(np.uint8)
    return _convert_to_pil(result)


def adjust_contrast(image: Union[np.ndarray, Image.Image], contrast_factor: float = 1.2) -> Image.Image:
    """
    Adjust contrast of an image.
    
    Args:
        image: Input image as numpy array or PIL Image
        contrast_factor: Contrast adjustment factor (>1 more contrast, <1 less contrast)
    
    Returns:
        Contrast-adjusted image as PIL Image
    """
    img_array = _convert_to_numpy(image)
    # Convert to float and adjust contrast around mean
    mean = np.mean(img_array)
    adjusted = (img_array.astype(np.float32) - mean) * contrast_factor + mean
    result = np.clip(adjusted, 0, 255).astype(np.uint8)
    return _convert_to_pil(result)


def rotate_image(image: Union[np.ndarray, Image.Image], angle: float = 15.0) -> Image.Image:
    """
    Rotate an image by a given angle.
    
    Args:
        image: Input image as numpy array or PIL Image
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image as PIL Image
    """
    img_array = _convert_to_numpy(image)
    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(img_array, rotation_matrix, (width, height))
    return _convert_to_pil(rotated)


def apply_blur(image: Union[np.ndarray, Image.Image], blur_intensity: int = 5) -> Image.Image:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image as numpy array or PIL Image
        blur_intensity: Blur kernel size (must be odd)
    
    Returns:
        Blurred image as PIL Image
    """
    img_array = _convert_to_numpy(image)
    # Ensure kernel size is odd
    if blur_intensity % 2 == 0:
        blur_intensity += 1
    
    blurred = cv2.GaussianBlur(img_array, (blur_intensity, blur_intensity), 0)
    return _convert_to_pil(blurred)


def apply_shadow(image: Union[np.ndarray, Image.Image], shadow_intensity: float = 0.3) -> Image.Image:
    """
    Apply shadow effect to an image.
    
    Args:
        image: Input image as numpy array or PIL Image
        shadow_intensity: Shadow intensity (0-1, higher means darker shadow)
    
    Returns:
        Image with shadow effect as PIL Image
    """
    img_array = _convert_to_numpy(image)
    height, width = img_array.shape[:2]
    
    # Create a random shadow mask
    shadow_mask = np.ones((height, width), dtype=np.float32)
    
    # Create shadow in random region
    x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
    x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
    
    # Apply gradient shadow
    shadow_mask[y1:y2, x1:x2] *= (1 - shadow_intensity)
    
    # Apply shadow to image
    if len(img_array.shape) == 3:
        shadow_mask = np.stack([shadow_mask] * img_array.shape[2], axis=2)
    
    shadowed = img_array.astype(np.float32) * shadow_mask
    result = np.clip(shadowed, 0, 255).astype(np.uint8)
    return _convert_to_pil(result)


def apply_highlight(image: Union[np.ndarray, Image.Image], highlight_intensity: float = 0.3) -> Image.Image:
    """
    Apply highlight effect to an image.
    
    Args:
        image: Input image as numpy array or PIL Image
        highlight_intensity: Highlight intensity (0-1, higher means brighter highlight)
    
    Returns:
        Image with highlight effect as PIL Image
    """
    img_array = _convert_to_numpy(image)
    height, width = img_array.shape[:2]
    
    # Create a random highlight mask
    highlight_mask = np.ones((height, width), dtype=np.float32)
    
    # Create highlight in random region
    x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
    x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
    
    # Apply gradient highlight
    highlight_mask[y1:y2, x1:x2] *= (1 + highlight_intensity)
    
    # Apply highlight to image
    if len(img_array.shape) == 3:
        highlight_mask = np.stack([highlight_mask] * img_array.shape[2], axis=2)
    
    highlighted = img_array.astype(np.float32) * highlight_mask
    result = np.clip(highlighted, 0, 255).astype(np.uint8)
    return _convert_to_pil(result)


# Export all functions
__all__ = [
    'add_noise',
    'adjust_brightness', 
    'adjust_contrast',
    'rotate_image',
    'apply_blur',
    'apply_shadow',
    'apply_highlight'
]
