#!/usr/bin/env python3
"""
Pizza Preprocessor for image preprocessing
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import torch
import torchvision.transforms as transforms

from ..constants import INPUT_SIZE, IMAGE_MEAN, IMAGE_STD


class PizzaPreprocessor:
    """Image preprocessor for pizza detection"""
    
    def __init__(self, img_size: int = INPUT_SIZE):
        """
        Initialize the preprocessor
        
        Args:
            img_size: Target image size for preprocessing
        """
        self.img_size = img_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            return tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def preprocess_batch(self, image_paths: list) -> torch.Tensor:
        """
        Preprocess a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch of preprocessed image tensors
        """
        batch = []
        
        for img_path in image_paths:
            tensor = self.preprocess_image(img_path)
            batch.append(tensor)
        
        return torch.stack(batch)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image using OpenCV
        
        Args:
            image: Input image array
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to [0, 1]
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image: np.ndarray, 
                     rotation: Optional[float] = None,
                     brightness: Optional[float] = None,
                     contrast: Optional[float] = None) -> np.ndarray:
        """
        Apply simple augmentations to image
        
        Args:
            image: Input image
            rotation: Rotation angle in degrees
            brightness: Brightness adjustment factor
            contrast: Contrast adjustment factor
            
        Returns:
            Augmented image
        """
        result = image.copy()
        
        # Apply rotation
        if rotation is not None:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            result = cv2.warpAffine(result, matrix, (image.shape[1], image.shape[0]))
        
        # Apply brightness
        if brightness is not None:
            result = cv2.convertScaleAbs(result, alpha=1.0, beta=brightness)
        
        # Apply contrast
        if contrast is not None:
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
        
        return result


# Backward compatibility
def preprocess_image(image_path: Union[str, Path], img_size: int = INPUT_SIZE) -> torch.Tensor:
    """Standalone preprocessing function for backward compatibility"""
    preprocessor = PizzaPreprocessor(img_size)
    return preprocessor.preprocess_image(image_path)


__all__ = ['PizzaPreprocessor', 'preprocess_image']
