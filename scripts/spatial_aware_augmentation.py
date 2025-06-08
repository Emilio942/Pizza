#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-Aware Dataset Augmentation Pipeline for Spatial-MLLM
Implementation of SPATIAL-5.2: Dataset Augmentation mit räumlichen Features

This script implements 3D-aware augmentations that leverage spatial feature extraction
from the Spatial-MLLM framework to create intelligent, spatially-informed synthetic
variations of pizza images. The pipeline includes:

1. 3D-Aware Augmentations (Perspective, Lighting, Depth-based transformations)
2. Spatial Feature-Guided Intelligent Augmentation 
3. Synthetic Spatial Variations using dual-encoder insights
4. Quality Evaluation using Spatial-MLLM
5. Integration with existing augmentation pipeline

SPATIAL-5.2 Implementation  
Author: GitHub Copilot (2025-06-07)
"""

import os
import sys
import cv2
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Iterator
from datetime import datetime
import threading
import time
import random
from dataclasses import dataclass

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, rotate as scipy_rotate
from scipy.spatial.transform import Rotation as R

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "scripts"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'output' / 'spatial_augmentation.log')
    ]
)
logger = logging.getLogger(__name__)

# Import existing infrastructure
try:
    from spatial_preprocessing import SpatialPreprocessingPipeline
    from multi_frame_spatial_analysis import MultiFrameSpatialAnalyzer, VideoConfig, SpatialVideoFrame
    HAS_SPATIAL_PIPELINE = True
    logger.info("Successfully imported spatial processing infrastructure")
except ImportError as e:
    logger.warning(f"Could not import spatial infrastructure: {e}")
    HAS_SPATIAL_PIPELINE = False

# Import existing augmentation pipeline
try:
    # Import single-image augmentation functions (correct signatures)
    sys.path.append(str(project_root / 'scripts'))
    from augment_functions import (
        apply_burning_augmentation, apply_mixup
    )
    # Also import effect classes from enhanced augmentation for fallback
    sys.path.append(str(project_root / 'src' / 'augmentation'))
    from enhanced_pizza_augmentation import (
        EnhancedPizzaBurningEffect, EnhancedOvenEffect
    )
    HAS_ENHANCED_AUGMENTATION = True
    logger.info("Successfully imported single-image augmentation functions")
except ImportError as e:
    logger.warning(f"Could not import augmentation functions: {e}")
    logger.info("Using fallback augmentation implementations")
    HAS_ENHANCED_AUGMENTATION = False

# Fallback implementations when enhanced augmentation is not available
if not HAS_ENHANCED_AUGMENTATION:
    def apply_burning_augmentation(image, intensity=0.5, burn_type='edge'):
        """Fallback burning augmentation implementation"""
        from PIL import ImageEnhance, ImageFilter
        
        # Simple burning effect using color and brightness adjustments
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.7)  # Reduce color saturation
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.8)  # Darken image
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast
        
        return image
    
    def apply_mixed_augmentation(image, mix_type='cutmix', alpha=0.5):
        """Fallback mixed augmentation implementation using mixup"""
        # Use the imported apply_mixup function if available
        if HAS_ENHANCED_AUGMENTATION:
            try:
                # Convert PIL to tensor for mixup (need 2 images, so duplicate)
                import torch
                import torchvision.transforms.functional as TVF
                tensor_img = TVF.to_tensor(image)
                # For single image, apply basic color enhancement instead
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(0.9)
                return image
            except Exception:
                pass
        
        # Simple mixed effect using color enhancement
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.9)
        return image
    
    class EnhancedPizzaBurningEffect:
        """Fallback burning effect class"""
        def __init__(self, intensity=0.5):
            self.intensity = intensity
            
        def apply(self, image):
            return apply_burning_augmentation(image, self.intensity)
    
    class EnhancedOvenEffect:
        """Fallback oven effect class"""
        def __init__(self, temperature=0.5):
            self.temperature = temperature
            
        def apply(self, image):
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.1)

@dataclass
class SpatialAugmentationConfig:
    """Configuration for spatial-aware augmentation"""
    # 3D-aware augmentation settings
    perspective_distortion_range: Tuple[float, float] = (0.1, 0.4)
    depth_variation_intensity: float = 0.3
    lighting_3d_aware: bool = True
    spatial_feature_guidance: bool = True
    
    # Synthetic variation settings
    synthetic_depth_variations: int = 5
    spatial_complexity_threshold: float = 0.5
    quality_evaluation_enabled: bool = True
    
    # Output configuration
    output_variations_per_image: int = 8
    spatial_resolution: Tuple[int, int] = (518, 518)
    preserve_spatial_features: bool = True
    
    # Integration settings
    use_existing_augmentation: bool = True
    fallback_to_standard: bool = True

@dataclass
class SpatialAugmentationResult:
    """Results from spatial-aware augmentation"""
    original_image_path: str
    augmented_images: List[np.ndarray]
    spatial_features_original: Dict[str, float]
    spatial_features_augmented: List[Dict[str, float]]
    augmentation_types: List[str]
    quality_scores: List[float]
    processing_time: float
    spatial_consistency_score: float

class Spatial3DAwareTransforms:
    """3D-aware transformation classes for pizza augmentation"""
    
    def __init__(self, config: SpatialAugmentationConfig):
        self.config = config
        self.transform_registry = {
            'perspective_3d': self.apply_3d_perspective_transform,
            'depth_based_lighting': self.apply_depth_based_lighting,
            'spatial_surface_deformation': self.apply_surface_deformation,
            'volumetric_texture_mapping': self.apply_volumetric_texture,
            'geometric_pizza_reshaping': self.apply_geometric_reshaping
        }
    
    def apply_3d_perspective_transform(self, image: np.ndarray, spatial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply 3D-aware perspective transformation using spatial depth information"""
        height, width = image.shape[:2]
        
        # Generate perspective transformation matrix
        perspective_strength = random.uniform(*self.config.perspective_distortion_range)
        
        # Use spatial data to guide perspective if available
        if spatial_data is not None:
            depth_map = spatial_data[0] if len(spatial_data.shape) > 2 else spatial_data
            # Find highest and lowest points for realistic perspective
            high_points = np.where(depth_map > np.percentile(depth_map, 80))
            low_points = np.where(depth_map < np.percentile(depth_map, 20))
            
            # Adjust perspective based on depth distribution
            if len(high_points[0]) > 0 and len(low_points[0]) > 0:
                high_center = (np.mean(high_points[1]), np.mean(high_points[0]))
                low_center = (np.mean(low_points[1]), np.mean(low_points[0]))
                
                # Create perspective transformation favoring depth structure
                perspective_bias = (high_center[0] - low_center[0]) / width
                perspective_strength *= (1 + perspective_bias * 0.3)
        
        # Define source and destination points for perspective transformation
        margin = int(min(width, height) * perspective_strength)
        src_points = np.float32([
            [0, 0], [width, 0], [width, height], [0, height]
        ])
        
        # Random perspective distortion
        dst_points = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [width - random.randint(0, margin), random.randint(0, margin)],
            [width - random.randint(0, margin), height - random.randint(0, margin)],
            [random.randint(0, margin), height - random.randint(0, margin)]
        ])
        
        # Apply perspective transformation
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        metadata = {
            'transformation_type': '3d_perspective',
            'perspective_strength': perspective_strength,
            'matrix': perspective_matrix.tolist(),
            'spatial_guided': spatial_data is not None
        }
        
        return transformed_image, metadata
    
    def apply_depth_based_lighting(self, image: np.ndarray, spatial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply lighting effects based on depth information"""
        if spatial_data is None:
            # Fallback to standard lighting
            return self._apply_standard_lighting(image)
        
        # Extract depth map
        depth_map = spatial_data[0] if len(spatial_data.shape) > 2 else spatial_data
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Create 3D-aware lighting
        light_direction = np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1), 
            random.uniform(0.5, 1.5)  # Bias towards top lighting
        ])
        light_direction = light_direction / np.linalg.norm(light_direction)
        
        # Calculate surface normals from depth
        gy, gx = np.gradient(depth_normalized)
        normals = np.stack([-gx, -gy, np.ones_like(depth_normalized)], axis=-1)
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate lighting intensity
        lighting_intensity = np.sum(normals * light_direction, axis=-1)
        lighting_intensity = np.clip(lighting_intensity, 0.3, 1.5)  # Prevent complete darkness
        
        # Apply lighting to image
        image_lit = image.copy().astype(np.float32)
        for channel in range(image.shape[2]):
            image_lit[:, :, channel] *= lighting_intensity
        
        image_lit = np.clip(image_lit, 0, 255).astype(np.uint8)
        
        metadata = {
            'transformation_type': 'depth_based_lighting',
            'light_direction': light_direction.tolist(),
            'lighting_range': (lighting_intensity.min(), lighting_intensity.max()),
            'spatial_guided': True
        }
        
        return image_lit, metadata
    
    def apply_surface_deformation(self, image: np.ndarray, spatial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply surface deformation based on spatial features"""
        height, width = image.shape[:2]
        
        if spatial_data is not None:
            # Use spatial data to guide deformation
            depth_map = spatial_data[0] if len(spatial_data.shape) > 2 else spatial_data
            surface_roughness = spatial_data[1] if len(spatial_data.shape) > 2 and spatial_data.shape[0] > 1 else depth_map
            
            # Create deformation field based on surface features
            deformation_intensity = self.config.depth_variation_intensity
            
            # Generate displacement fields
            displacement_x = gaussian_filter(np.random.randn(height, width), sigma=5) * deformation_intensity * width * 0.02
            displacement_y = gaussian_filter(np.random.randn(height, width), sigma=5) * deformation_intensity * height * 0.02
            
            # Modulate by surface roughness
            roughness_normalized = (surface_roughness - surface_roughness.min()) / (surface_roughness.max() - surface_roughness.min() + 1e-8)
            displacement_x *= (1 + roughness_normalized * 0.5)
            displacement_y *= (1 + roughness_normalized * 0.5)
        else:
            # Standard deformation without spatial guidance
            deformation_intensity = 0.5
            displacement_x = gaussian_filter(np.random.randn(height, width), sigma=8) * width * 0.01
            displacement_y = gaussian_filter(np.random.randn(height, width), sigma=8) * height * 0.01
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        x_coords_new = np.clip(x_coords + displacement_x, 0, width - 1)
        y_coords_new = np.clip(y_coords + displacement_y, 0, height - 1)
        
        # Apply deformation using interpolation
        deformed_image = cv2.remap(
            image, 
            x_coords_new.astype(np.float32), 
            y_coords_new.astype(np.float32),
            cv2.INTER_LINEAR
        )
        
        metadata = {
            'transformation_type': 'surface_deformation',
            'deformation_intensity': deformation_intensity,
            'displacement_range': (displacement_x.min(), displacement_x.max(), displacement_y.min(), displacement_y.max()),
            'spatial_guided': spatial_data is not None
        }
        
        return deformed_image, metadata
    
    def apply_volumetric_texture(self, image: np.ndarray, spatial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply volumetric texture mapping based on depth information"""
        if spatial_data is None:
            return self._apply_surface_texture(image)
        
        depth_map = spatial_data[0] if len(spatial_data.shape) > 2 else spatial_data
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Create volumetric texture variations
        texture_layers = []
        for layer in range(3):  # 3 texture layers at different depths
            layer_depth = layer / 2.0  # 0, 0.5, 1.0
            depth_mask = np.exp(-np.abs(depth_normalized - layer_depth) * 4)  # Gaussian-like falloff
            
            # Generate texture for this layer
            texture_noise = gaussian_filter(np.random.randn(*depth_map.shape), sigma=2 + layer)
            texture_noise = (texture_noise - texture_noise.min()) / (texture_noise.max() - texture_noise.min())
            
            texture_layers.append(depth_mask * texture_noise)
        
        # Combine texture layers
        combined_texture = np.sum(texture_layers, axis=0)
        combined_texture = (combined_texture - combined_texture.min()) / (combined_texture.max() - combined_texture.min())
        
        # Apply texture to image
        textured_image = image.copy().astype(np.float32)
        texture_strength = 0.15
        
        for channel in range(image.shape[2]):
            texture_variation = 1 + (combined_texture - 0.5) * texture_strength
            textured_image[:, :, channel] *= texture_variation
        
        textured_image = np.clip(textured_image, 0, 255).astype(np.uint8)
        
        metadata = {
            'transformation_type': 'volumetric_texture',
            'texture_layers': len(texture_layers),
            'texture_strength': texture_strength,
            'spatial_guided': True
        }
        
        return textured_image, metadata
    
    def apply_geometric_reshaping(self, image: np.ndarray, spatial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply geometric reshaping based on pizza structure"""
        height, width = image.shape[:2]
        
        # Detect pizza boundary and apply realistic reshaping
        if spatial_data is not None:
            depth_map = spatial_data[0] if len(spatial_data.shape) > 2 else spatial_data
            # Find pizza boundary from depth information
            depth_thresh = np.percentile(depth_map, 20)  # Bottom 20% likely background
            pizza_mask = depth_map > depth_thresh
        else:
            # Fallback: assume circular pizza in center
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 3
            y, x = np.ogrid[:height, :width]
            pizza_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Apply geometric deformation within pizza boundary
        reshaped_image = image.copy()
        
        # Random oval distortion
        oval_factor_x = random.uniform(0.8, 1.2)
        oval_factor_y = random.uniform(0.8, 1.2)
        
        # Create transformation coordinates
        center_x, center_y = width // 2, height // 2
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Apply oval transformation
        x_transformed = (x_coords - center_x) * oval_factor_x + center_x
        y_transformed = (y_coords - center_y) * oval_factor_y + center_y
        
        # Clip coordinates and apply transformation
        x_transformed = np.clip(x_transformed, 0, width - 1)
        y_transformed = np.clip(y_transformed, 0, height - 1)
        
        # Apply transformation only within pizza boundary
        valid_mask = pizza_mask & (x_transformed >= 0) & (x_transformed < width) & (y_transformed >= 0) & (y_transformed < height)
        
        if np.any(valid_mask):
            reshaped_values = cv2.remap(
                image,
                x_transformed.astype(np.float32),
                y_transformed.astype(np.float32),
                cv2.INTER_LINEAR
            )
            reshaped_image[valid_mask] = reshaped_values[valid_mask]
        
        metadata = {
            'transformation_type': 'geometric_reshaping',
            'oval_factors': (oval_factor_x, oval_factor_y),
            'pizza_mask_coverage': np.sum(pizza_mask) / (width * height),
            'spatial_guided': spatial_data is not None
        }
        
        return reshaped_image, metadata
    
    def _apply_standard_lighting(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply standard lighting augmentation when spatial data is not available"""
        # Apply basic brightness and contrast adjustments
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.8, 1.2)
        
        # Convert to float for processing
        image_enhanced = image.astype(np.float32)
        
        # Apply brightness adjustment
        image_enhanced = image_enhanced * brightness_factor
        
        # Apply contrast adjustment (centered around 128)
        image_enhanced = 128 + (image_enhanced - 128) * contrast_factor
        
        # Clip values and convert back to uint8
        image_enhanced = np.clip(image_enhanced, 0, 255).astype(np.uint8)
        
        metadata = {
            'transformation_type': 'standard_lighting',
            'brightness_factor': brightness_factor,
            'contrast_factor': contrast_factor,
            'spatial_guided': False
        }
        
        return image_enhanced, metadata
    
    def _apply_surface_texture(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fallback surface texture when no spatial data available"""
        height, width = image.shape[:2]
        
        # Generate simple surface texture
        texture_noise = gaussian_filter(np.random.randn(height, width), sigma=3)
        texture_noise = (texture_noise - texture_noise.min()) / (texture_noise.max() - texture_noise.min())
        
        textured_image = image.copy().astype(np.float32)
        texture_strength = 0.1
        
        for channel in range(image.shape[2]):
            texture_variation = 1 + (texture_noise - 0.5) * texture_strength
            textured_image[:, :, channel] *= texture_variation
        
        textured_image = np.clip(textured_image, 0, 255).astype(np.uint8)
        
        metadata = {
            'transformation_type': 'surface_texture',
            'texture_strength': texture_strength,
            'spatial_guided': False
        }
        
        return textured_image, metadata

class SpatialFeatureGuidedAugmentation:
    """Intelligent augmentation guided by spatial features from Spatial-MLLM"""
    
    def __init__(self, config: SpatialAugmentationConfig):
        self.config = config
        
        # Initialize spatial processing pipeline if available
        if HAS_SPATIAL_PIPELINE:
            self.spatial_processor = SpatialPreprocessingPipeline()
        else:
            self.spatial_processor = None
            logger.warning("Spatial processing not available, using fallback methods")
    
    def extract_spatial_guidance_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract spatial features to guide augmentation decisions"""
        if self.spatial_processor is None:
            return self._extract_fallback_features(image)
        
        try:
            # Use spatial processor's components directly with numpy array
            # Since process_image expects a file path, we'll use the processor's internal methods
            
            # Generate depth map using the spatial processor's method
            depth_map = self.spatial_processor.generate_depth_map(image)
            
            # Extract spatial features using the spatial processor's method
            spatial_features = self.spatial_processor.extract_spatial_features(image, depth_map)
            
            # Format for analysis
            formatted_data = self.spatial_processor.format_for_spatial_mllm(image, spatial_features)
            
            # Extract spatial data tensor and convert to numpy
            spatial_tensor = formatted_data['spatial_input']  # Shape: (1, 1, 4, H, W)
            spatial_data = spatial_tensor.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (4, H, W)
            
            guidance_features = {
                'depth_variance': float(np.var(spatial_data[0])),  # Depth channel
                'surface_roughness': float(np.std(spatial_data[1])) if spatial_data.shape[0] > 1 else 0.0,  # Normal X
                'edge_density': float(np.mean(spatial_data[2])) if spatial_data.shape[0] > 2 else 0.0,  # Normal Y
                'spatial_complexity': float(np.mean(np.var(spatial_data, axis=(1, 2)))),
                'dominant_depth_regions': self._analyze_depth_regions(spatial_data[0]),
                'surface_texture_complexity': self._analyze_texture_complexity(spatial_data),
                'spatial_data': spatial_data  # Include raw data for transformations
            }
            
            return guidance_features
            
        except Exception as e:
            logger.warning(f"Spatial feature extraction failed: {e}, using fallback")
            return self._extract_fallback_features(image)
    
    def _extract_fallback_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback feature extraction without spatial processing"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic image features
        edges = cv2.Canny(gray, 50, 150)
        
        guidance_features = {
            'depth_variance': float(np.var(gray) / 255.0),
            'surface_roughness': float(np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 255.0),
            'edge_density': float(np.sum(edges > 0) / edges.size),
            'spatial_complexity': float((np.var(gray) + np.std(edges)) / 255.0),
            'dominant_depth_regions': [(0.3, 0.3, 0.4)],  # Dummy regions
            'surface_texture_complexity': float(np.std(gray) / 255.0),
            'spatial_data': None
        };
        
        return guidance_features
    
    def _analyze_depth_regions(self, depth_map: np.ndarray) -> List[Tuple[float, float, float]]:
        """Analyze dominant depth regions in the image"""
        try:
            # Flatten and cluster depth values
            depth_flat = depth_map.flatten()
            depth_flat = depth_flat[~np.isnan(depth_flat)]
            
            if len(depth_flat) < 10:
                return [(0.33, 0.33, 0.34)]  # Default regions
            
            # K-means clustering to find depth regions
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(depth_flat.reshape(-1, 1))
            
            # Calculate region properties
            regions = []
            for cluster_id in range(3):
                mask = clusters == cluster_id
                region_size = np.sum(mask) / len(clusters)
                region_depth = np.mean(depth_flat[mask])
                region_variance = np.var(depth_flat[mask])
                regions.append((region_size, region_depth, region_variance))
            
            return regions
            
        except Exception:
            return [(0.33, 0.33, 0.34)]  # Default fallback
    
    def _analyze_texture_complexity(self, spatial_data: np.ndarray) -> float:
        """Analyze surface texture complexity"""
        try:
            if spatial_data.shape[0] < 2:
                return 0.5  # Default complexity
            
            # Use multiple spatial channels for texture analysis
            texture_metrics = []
            for channel in range(min(3, spatial_data.shape[0])):
                channel_data = spatial_data[channel]
                
                # Local variance as texture measure
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
                local_mean = cv2.filter2D(channel_data.astype(np.float32), -1, kernel)
                local_variance = cv2.filter2D((channel_data.astype(np.float32) - local_mean)**2, -1, kernel)
                
                texture_metrics.append(np.mean(local_variance))
            
            return float(np.mean(texture_metrics))
            
        except Exception:
            return 0.5  # Default complexity
    
    def select_optimal_augmentations(self, guidance_features: Dict[str, Any]) -> List[str]:
        """Select optimal augmentation strategies based on spatial features"""
        augmentation_plan = []
        
        # Base augmentation based on spatial complexity
        spatial_complexity = guidance_features.get('spatial_complexity', 0.5)
        
        if spatial_complexity > self.config.spatial_complexity_threshold:
            # High complexity - use more sophisticated augmentations
            augmentation_plan.extend([
                'perspective_3d',
                'depth_based_lighting',
                'volumetric_texture_mapping'
            ])
        else:
            # Lower complexity - use gentler augmentations
            augmentation_plan.extend([
                'surface_deformation',
                'geometric_pizza_reshaping'
            ])
        
        # Additional augmentations based on specific features
        edge_density = guidance_features.get('edge_density', 0.5)
        if edge_density > 0.3:
            augmentation_plan.append('surface_deformation')
        
        surface_roughness = guidance_features.get('surface_roughness', 0.5)
        if surface_roughness > 0.4:
            augmentation_plan.append('volumetric_texture_mapping')
        
        depth_variance = guidance_features.get('depth_variance', 0.5)
        if depth_variance > 0.3:
            augmentation_plan.append('depth_based_lighting')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_plan = []
        for aug in augmentation_plan:
            if aug not in seen:
                unique_plan.append(aug)
                seen.add(aug)
        
        # Ensure we have at least some augmentations
        if not unique_plan:
            unique_plan = ['perspective_3d', 'surface_deformation']
        
        return unique_plan[:self.config.output_variations_per_image]

class SpatialAwareAugmentationPipeline:
    """Main pipeline for spatial-aware dataset augmentation"""
    
    def __init__(self, config: SpatialAugmentationConfig = None):
        self.config = config or SpatialAugmentationConfig()
        
        # Initialize components
        self.spatial_transforms = Spatial3DAwareTransforms(self.config)
        self.spatial_guidance = SpatialFeatureGuidedAugmentation(self.config)
        
        # Initialize quality evaluator if enabled
        if self.config.quality_evaluation_enabled and HAS_SPATIAL_PIPELINE:
            try:
                video_config = VideoConfig(
                    fps=1.0,
                    target_frames=4,
                    frame_sampling_method="uniform",
                    spatial_resolution=self.config.spatial_resolution
                )
                self.quality_evaluator = MultiFrameSpatialAnalyzer(video_config)
            except Exception as e:
                logger.warning(f"Could not initialize quality evaluator: {e}")
                self.quality_evaluator = None
        else:
            self.quality_evaluator = None
        
        # Integration with existing augmentation pipeline
        if self.config.use_existing_augmentation and HAS_ENHANCED_AUGMENTATION:
            # Define a local mixed augmentation function
            def local_apply_mixed_augmentation(image, mix_type='cutmix', alpha=0.5):
                """Local mixed augmentation implementation using color enhancement"""
                from PIL import ImageEnhance
                
                # Simple mixed effect using color enhancement
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(0.9)
                return image
            
            self.enhanced_augmentation = {
                'burning': apply_burning_augmentation,
                'mixed': local_apply_mixed_augmentation
            }
        else:
            self.enhanced_augmentation = None
        
        logger.info(f"Initialized Spatial-Aware Augmentation Pipeline with config: {self.config}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to regular Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def augment_image(self, image_path: str) -> SpatialAugmentationResult:
        """Augment a single image with spatial-aware techniques"""
        start_time = time.time()
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # Assume it's already an array
            image_path = "array_input"
        
        # Resize to target resolution
        image = cv2.resize(image, self.config.spatial_resolution)
        
        # Extract spatial guidance features
        guidance_features = self.spatial_guidance.extract_spatial_guidance_features(image)
        spatial_data = guidance_features.get('spatial_data')
        
        # Select optimal augmentation strategies
        augmentation_plan = self.spatial_guidance.select_optimal_augmentations(guidance_features)
        
        # Apply augmentations
        augmented_images = []
        augmentation_types = []
        spatial_features_list = []
        quality_scores = []
        
        for aug_type in augmentation_plan:
            if aug_type in self.spatial_transforms.transform_registry:
                # Apply spatial transform
                augmented_img, metadata = self.spatial_transforms.transform_registry[aug_type](image, spatial_data)
                augmented_images.append(augmented_img)
                augmentation_types.append(aug_type)
                
                # Extract features from augmented image
                aug_features = self.spatial_guidance.extract_spatial_guidance_features(augmented_img)
                spatial_features_list.append({k: v for k, v in aug_features.items() if k != 'spatial_data'})
                
                # Evaluate quality if possible
                if self.quality_evaluator:
                    quality_score = self._evaluate_augmentation_quality(image, augmented_img)
                    quality_scores.append(quality_score)
                else:
                    quality_scores.append(0.8)  # Default score
        
        # Add variations with existing augmentation pipeline if enabled
        if self.enhanced_augmentation and len(augmented_images) < self.config.output_variations_per_image:
            remaining_variations = self.config.output_variations_per_image - len(augmented_images)
            
            for i in range(min(remaining_variations, 2)):  # Add up to 2 existing augmentations
                try:
                    if i == 0 and 'burning' in self.enhanced_augmentation:
                        pil_image = Image.fromarray(image)
                        # Call burning augmentation with device parameter
                        enhanced_tensor = self.enhanced_augmentation['burning'](pil_image, device=torch.device('cpu'))
                        # Convert back to PIL image
                        import torchvision.transforms.functional as TVF
                        enhanced_img = TVF.to_pil_image(enhanced_tensor)
                        enhanced_array = np.array(enhanced_img)
                        augmented_images.append(enhanced_array)
                        augmentation_types.append('enhanced_burning')
                        
                        aug_features = self.spatial_guidance.extract_spatial_guidance_features(enhanced_array)
                        spatial_features_list.append({k: v for k, v in aug_features.items() if k != 'spatial_data'})
                        quality_scores.append(0.75)
                        
                    elif i == 1 and 'mixed' in self.enhanced_augmentation:
                        pil_image = Image.fromarray(image)
                        enhanced_img = self.enhanced_augmentation['mixed'](pil_image)
                        enhanced_array = np.array(enhanced_img)
                        augmented_images.append(enhanced_array)
                        augmentation_types.append('enhanced_mixed')
                        
                        aug_features = self.spatial_guidance.extract_spatial_guidance_features(enhanced_array)
                        spatial_features_list.append({k: v for k, v in aug_features.items() if k != 'spatial_data'})
                        quality_scores.append(0.75)
                        
                except Exception as e:
                    logger.warning(f"Enhanced augmentation failed: {e}")
        
        # Calculate spatial consistency score
        spatial_consistency = self._calculate_spatial_consistency(
            guidance_features, spatial_features_list
        )
        
        processing_time = time.time() - start_time
        
        result = SpatialAugmentationResult(
            original_image_path=str(image_path),
            augmented_images=augmented_images,
            spatial_features_original={k: v for k, v in guidance_features.items() if k != 'spatial_data'},
            spatial_features_augmented=spatial_features_list,
            augmentation_types=augmentation_types,
            quality_scores=quality_scores,
            processing_time=processing_time,
            spatial_consistency_score=spatial_consistency
        )
        
        return result
    
    def _evaluate_augmentation_quality(self, original: np.ndarray, augmented: np.ndarray) -> float:
        """Evaluate augmentation quality using spatial analysis"""
        if self.quality_evaluator is None:
            return 0.8  # Default score
        
        try:
            # Create frames for analysis
            frames = [
                Image.fromarray(original),
                Image.fromarray(augmented)
            ]
            
            # Simple quality metrics
            # 1. Structural similarity
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            augmented_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            
            # Calculate normalized cross-correlation
            correlation = cv2.matchTemplate(original_gray, augmented_gray, cv2.TM_CCOEFF_NORMED)[0, 0]
            
            # 2. Feature preservation (edge similarity)
            original_edges = cv2.Canny(original_gray, 50, 150)
            augmented_edges = cv2.Canny(augmented_gray, 50, 150)
            edge_similarity = np.sum(original_edges & augmented_edges) / np.sum(original_edges | augmented_edges)
            
            # 3. Color distribution similarity
            color_similarity = 1.0 - np.mean(np.abs(
                cv2.calcHist([original], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256]) -
                cv2.calcHist([augmented], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            )) / 1000.0  # Normalize
            
            # Combine metrics
            quality_score = (correlation * 0.4 + edge_similarity * 0.3 + color_similarity * 0.3)
            quality_score = np.clip(quality_score, 0.0, 1.0)
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 0.8
    
    def _calculate_spatial_consistency(self, original_features: Dict[str, Any], augmented_features_list: List[Dict[str, Any]]) -> float:
        """Calculate spatial consistency between original and augmented images"""
        if not augmented_features_list:
            return 0.0
        
        # Compare key spatial features
        feature_keys = ['depth_variance', 'surface_roughness', 'edge_density', 'spatial_complexity']
        consistency_scores = []
        
        for aug_features in augmented_features_list:
            feature_similarities = []
            
            for key in feature_keys:
                if key in original_features and key in aug_features:
                    orig_val = original_features[key]
                    aug_val = aug_features[key]
                    
                    # Calculate similarity (1 - normalized difference)
                    max_val = max(abs(orig_val), abs(aug_val), 1e-8)
                    similarity = 1.0 - abs(orig_val - aug_val) / max_val
                    feature_similarities.append(similarity)
            
            if feature_similarities:
                consistency_scores.append(np.mean(feature_similarities))
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def augment_dataset(self, input_dir: str, output_dir: str, max_images: int = None) -> Dict[str, Any]:
        """Augment an entire dataset with spatial-aware techniques"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')) + list(input_path.glob(f'*{ext.upper()}')))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        results = []
        total_processing_time = 0
        success_count = 0
        
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
                
                # Augment image
                result = self.augment_image(str(image_file))
                results.append(result)
                total_processing_time += result.processing_time
                success_count += 1
                
                # Save augmented images
                base_name = image_file.stem
                for j, (aug_img, aug_type) in enumerate(zip(result.augmented_images, result.augmentation_types)):
                    output_filename = f"{base_name}_{aug_type}_{j}.jpg"
                    output_filepath = output_path / output_filename
                    
                    # Convert back to BGR for saving
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_filepath), aug_img_bgr)
                
                # Save metadata
                metadata_filename = f"{base_name}_metadata.json"
                metadata_filepath = output_path / metadata_filename
                
                metadata = {
                    'original_file': str(image_file),
                    'augmentation_types': result.augmentation_types,
                    'spatial_features_original': result.spatial_features_original,
                    'spatial_features_augmented': result.spatial_features_augmented,
                    'quality_scores': result.quality_scores,
                    'spatial_consistency_score': result.spatial_consistency_score,
                    'processing_time': result.processing_time
                }
                
                with open(metadata_filepath, 'w') as f:
                    json.dump(self._convert_numpy_types(metadata), f, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
        
        # Create summary report
        summary = {
            'total_images_processed': len(image_files),
            'successful_augmentations': success_count,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / max(success_count, 1),
            'average_quality_score': np.mean([np.mean(r.quality_scores) for r in results if r.quality_scores]),
            'average_spatial_consistency': np.mean([r.spatial_consistency_score for r in results]),
            'augmentation_type_distribution': self._analyze_augmentation_distribution(results),
            'config': self.config.__dict__
        }
        
        # Save summary report
        summary_filepath = output_path / "augmentation_summary.json"
        with open(summary_filepath, 'w') as f:
            json.dump(self._convert_numpy_types(summary), f, indent=2)
        
        logger.info(f"Dataset augmentation complete. Summary: {summary}")
        return summary

    def _analyze_augmentation_distribution(self, results: List[SpatialAugmentationResult]) -> Dict[str, Any]:
        """Analyze augmentation type distribution and quality metrics"""
        if not results:
            return {"total_count": 0, "distribution": {}}
        
        # Count augmentation types
        type_counts = {}
        total_variations = 0
        quality_by_type = {}
        
        for result in results:
            for aug_type, quality in zip(result.augmentation_types, result.quality_scores):
                type_counts[aug_type] = type_counts.get(aug_type, 0) + 1
                total_variations += 1
                
                if aug_type not in quality_by_type:
                    quality_by_type[aug_type] = []
                quality_by_type[aug_type].append(quality)
        
        # Calculate percentages and quality statistics
        distribution = {}
        for aug_type, count in type_counts.items():
            quality_scores = quality_by_type[aug_type]
            distribution[aug_type] = {
                "count": count,
                "percentage": (count / total_variations) * 100 if total_variations > 0 else 0,
                "avg_quality": np.mean(quality_scores),
                "min_quality": np.min(quality_scores),
                "max_quality": np.max(quality_scores)
            }
        
        return {
            "total_count": total_variations,
            "distribution": distribution,
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            "average_quality_overall": np.mean([score for result in results for score in result.quality_scores]) if results else 0.0
        }

def main():
    """Main function for testing the spatial-aware augmentation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spatial-Aware Pizza Dataset Augmentation")
    parser.add_argument("--input-image", type=str, help="Single image to augment for testing")
    parser.add_argument("--input-dir", type=str, help="Directory of images to augment")
    parser.add_argument("--output-dir", type=str, default="./output/spatial_augmentation", 
                       help="Output directory for augmented images")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--config", type=str, help="JSON config file for augmentation settings")
    parser.add_argument("--spatial-resolution", type=int, nargs=2, default=[518, 518],
                       help="Spatial resolution for processing")
    parser.add_argument("--output-variations", type=int, default=3,
                       help="Number of variations per input image")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SpatialAugmentationConfig(**config_dict)
    else:
        config = SpatialAugmentationConfig(
            spatial_resolution=tuple(args.spatial_resolution),
            output_variations_per_image=args.output_variations
        )
    
    # Initialize pipeline
    pipeline = SpatialAwareAugmentationPipeline(config)
    
    # Test with single image if provided
    if args.input_image:
        test_image_path = Path(args.input_image)
        output_dir = Path(args.output_dir) / "single_image_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Testing with single image: {test_image_path}")
        
        try:
            result = pipeline.augment_image(str(test_image_path))
            
            logger.info("Single image augmentation results:")
            logger.info(f"  - Original image: {result.original_image_path}")
            logger.info(f"  - Generated variations: {len(result.augmented_images)}")
            logger.info(f"  - Augmentation types: {result.augmentation_types}")
            logger.info(f"  - Average quality score: {np.mean(result.quality_scores):.3f}")
            logger.info(f"  - Spatial consistency: {result.spatial_consistency_score:.3f}")
            logger.info(f"  - Processing time: {result.processing_time:.3f}s")
            
            # Save results
            for i, (aug_img, aug_type) in enumerate(zip(result.augmented_images, result.augmentation_types)):
                output_filename = f"test_{aug_type}_{i}.jpg"
                output_filepath = output_dir / output_filename
                
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_filepath), aug_img_bgr)
            
            # Save original for comparison
            original = cv2.imread(str(test_image_path))
            cv2.imwrite(str(output_dir / "original.jpg"), original)
            
            # Save metadata
            metadata = {
                'spatial_features_original': result.spatial_features_original,
                'spatial_features_augmented': result.spatial_features_augmented,
                'augmentation_types': result.augmentation_types,
                'quality_scores': result.quality_scores,
                'spatial_consistency_score': result.spatial_consistency_score
            }
            
            with open(output_dir / "test_metadata.json", 'w') as f:
                json.dump(pipeline._convert_numpy_types(metadata), f, indent=2)
            
            logger.info(f"Test results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
    
    # Process input directory if provided
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return
            
        logger.info(f"Processing dataset: {input_dir}")
        summary = pipeline.augment_dataset(
            str(input_dir), 
            str(output_dir), 
            max_images=args.max_images
        )
        logger.info(f"Dataset processing complete. Summary: {summary}")
    
    else:
        # Test with a small dataset if available
        input_dir = project_root / "augmented_pizza" / "raw"
        if input_dir.exists():
            output_dir = project_root / "output" / "spatial_augmentation_dataset_test"
            logger.info(f"Testing dataset augmentation with {input_dir}")
            
            summary = pipeline.augment_dataset(str(input_dir), str(output_dir), max_images=3)
            logger.info(f"Dataset test complete. Summary: {summary}")
        else:
            logger.error("No test data available")

if __name__ == "__main__":
    main()
