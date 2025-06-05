#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Preprocessing Pipeline for Spatial-MLLM Pizza Classification

This script implements a comprehensive preprocessing pipeline to prepare pizza images
for the Spatial-MLLM dual-encoder architecture. It handles:

1. 2D Visual Preprocessing: Standard image preprocessing for Qwen2.5-VL encoder
2. 3D Spatial Data Generation: Synthetic depth maps and 3D features for VGGT encoder  
3. Dual-Encoder Data Formatting: Proper tensor formatting for both encoders
4. Quality Validation: Ensures preprocessing quality and consistency

SPATIAL-2.2 Implementation
Author: GitHub Copilot (2025-06-02)
"""

import os
import sys
import cv2
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'output' / 'spatial_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

class SpatialPreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for Spatial-MLLM dual-encoder architecture.
    Generates both 2D visual features and 3D spatial data from pizza images.
    """
    
    def __init__(self, 
                 output_size: Tuple[int, int] = (518, 518),
                 depth_estimation_method: str = "monocular",
                 enable_quality_validation: bool = True):
        """
        Initialize the spatial preprocessing pipeline.
        
        Args:
            output_size: Target image size for Spatial-MLLM (default: 518x518)
            depth_estimation_method: Method for 3D data generation ('monocular', 'shape_from_shading', 'edge_based')
            enable_quality_validation: Whether to validate preprocessing quality
        """
        self.output_size = output_size
        self.depth_method = depth_estimation_method
        self.enable_validation = enable_quality_validation
        
        # Initialize preprocessing transforms
        self._setup_transforms()
        
        # Initialize depth estimation
        self._setup_depth_estimation()
        
        # Statistics tracking
        self.stats = {
            'processed_images': 0,
            'failed_images': 0,
            'depth_quality_scores': [],
            'processing_times': []
        }
        
        logger.info(f"Initialized SpatialPreprocessingPipeline with output_size={output_size}, depth_method={depth_estimation_method}")
    
    def _setup_transforms(self):
        """Setup image transformation pipelines for dual-encoder input."""
        
        # 2D Visual Encoder Transform (Qwen2.5-VL compatible)
        self.visual_transform = transforms.Compose([
            transforms.Resize(self.output_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # Qwen2.5-VL normalization parameters
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # Basic preprocessing for spatial analysis
        self.spatial_preprocess = transforms.Compose([
            transforms.Resize(self.output_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        
        logger.info("Initialized transform pipelines for dual-encoder preprocessing")
    
    def _setup_depth_estimation(self):
        """Setup depth estimation methods for 3D spatial data generation."""
        
        if self.depth_method == "monocular":
            self._setup_monocular_depth()
        elif self.depth_method == "shape_from_shading":
            self._setup_shape_from_shading()
        elif self.depth_method == "edge_based":
            self._setup_edge_based_depth()
        else:
            logger.warning(f"Unknown depth method {self.depth_method}, falling back to edge_based")
            self.depth_method = "edge_based"
            self._setup_edge_based_depth()
    
    def _setup_monocular_depth(self):
        """Setup monocular depth estimation (requires external model)."""
        try:
            # Try to import MiDaS for depth estimation
            import torch
            torch.hub._validate_not_a_forked_repo = lambda a,b,c: True  # Disable validation
            self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            self.depth_transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
            self.depth_model.eval()
            logger.info("MiDaS depth estimation model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load MiDaS model: {e}")
            logger.info("Falling back to edge-based depth estimation")
            self.depth_method = "edge_based"
            self._setup_edge_based_depth()
    
    def _setup_shape_from_shading(self):
        """Setup shape-from-shading depth estimation."""
        # Simple shape-from-shading based on luminance gradients
        self.shading_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.shading_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        logger.info("Shape-from-shading depth estimation initialized")
    
    def _setup_edge_based_depth(self):
        """Setup edge-based depth estimation using gradient analysis."""
        # Kernels for edge detection and gradient analysis
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        self.laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        logger.info("Edge-based depth estimation initialized")
    
    def generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map from RGB image using selected method.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            depth_map: Depth values as numpy array (H, W)
        """
        if self.depth_method == "monocular" and hasattr(self, 'depth_model'):
            return self._generate_monocular_depth(image)
        elif self.depth_method == "shape_from_shading":
            return self._generate_shape_from_shading_depth(image)
        else:  # edge_based or fallback
            return self._generate_edge_based_depth(image)
    
    def _generate_monocular_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate depth using MiDaS monocular depth estimation."""
        try:
            # Convert to PIL for MiDaS transform
            pil_image = Image.fromarray(image)
            input_tensor = self.depth_transform.small_transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                depth = self.depth_model(input_tensor)
                depth = F.interpolate(depth.unsqueeze(1), size=self.output_size, mode='bilinear', align_corners=False)
                depth = depth.squeeze().cpu().numpy()
            
            # Normalize depth to [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"MiDaS depth estimation failed: {e}, falling back to edge-based")
            return self._generate_edge_based_depth(image)
    
    def _generate_shape_from_shading_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate depth using shape-from-shading technique."""
        # Robust input validation and conversion
        if image.dtype == np.float64:
            image = image.astype(np.float32)
        
        # Ensure image is in proper uint8 format for OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to grayscale with explicit uint8 input
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Calculate gradients with explicit CV_32F output
        grad_x = cv2.filter2D(gray, cv2.CV_32F, self.shading_kernel_x).astype(np.float32)
        grad_y = cv2.filter2D(gray, cv2.CV_32F, self.shading_kernel_y).astype(np.float32)
        
        # Estimate surface normals from gradients
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
        
        # Simple depth estimation from gradient magnitude
        # Areas with high gradients are assumed to be edges/height changes
        depth = (1.0 - grad_magnitude).astype(np.float32)
        depth = cv2.GaussianBlur(depth, (5, 5), 1.0).astype(np.float32)
        
        # Resize to target size
        depth = cv2.resize(depth, self.output_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth = ((depth - depth_min) / (depth_max - depth_min + 1e-8)).astype(np.float32)
        return depth
    
    def _generate_edge_based_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate depth using edge-based analysis and pizza-specific heuristics."""
        # Robust input validation and conversion
        if image.dtype == np.float64:
            image = image.astype(np.float32)
        
        # Ensure image is in proper uint8 format for OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to grayscale with explicit uint8 input
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Calculate edge information with explicit CV_32F output
        edges_x = cv2.filter2D(gray, cv2.CV_32F, self.sobel_x).astype(np.float32)
        edges_y = cv2.filter2D(gray, cv2.CV_32F, self.sobel_y).astype(np.float32)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2).astype(np.float32)
        
        # Calculate second derivatives (surface curvature indicators)
        laplacian = cv2.filter2D(gray, cv2.CV_32F, self.laplacian).astype(np.float32)
        
        # Pizza-specific depth heuristics
        depth = self._apply_pizza_depth_heuristics(gray, edge_magnitude, laplacian)
        
        # Resize to target size
        depth = cv2.resize(depth, self.output_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth = ((depth - depth_min) / (depth_max - depth_min + 1e-8)).astype(np.float32)
        return depth
    
    def _apply_pizza_depth_heuristics(self, gray: np.ndarray, edges: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
        """Apply pizza-specific heuristics for depth estimation."""
        # Initialize depth map
        depth = np.ones_like(gray, dtype=np.float32) * 0.5  # Base depth
        
        # Crust detection: typically brighter and at edges
        # Find bright regions (potential crust)
        bright_mask = gray > np.percentile(gray, 75)
        
        # Find image border regions (where crust typically is)
        h, w = gray.shape
        border_mask = np.zeros_like(gray, dtype=bool)
        border_width = min(h, w) // 8
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True
        
        # Crust regions (bright + border) get higher depth
        crust_mask = bright_mask & border_mask
        depth[crust_mask] += 0.3
        
        # Cheese/topping regions: middle brightness, potentially bubbly
        # Detect bubbles/texture using Laplacian response
        bubble_mask = np.abs(laplacian) > np.percentile(np.abs(laplacian), 80)
        medium_brightness = (gray > np.percentile(gray, 30)) & (gray < np.percentile(gray, 75))
        cheese_mask = medium_brightness & bubble_mask
        depth[cheese_mask] += 0.15
        
        # Dark regions (burnt areas or shadows) get lower depth
        dark_mask = gray < np.percentile(gray, 25)
        depth[dark_mask] -= 0.2
        
        # Smooth the depth map to remove noise
        depth = cv2.GaussianBlur(depth, (7, 7), 2.0).astype(np.float32)
        
        # Ensure depth is in valid range
        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)
        
        return depth
    
    def extract_spatial_features(self, image: np.ndarray, depth_map: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract 3D spatial features for VGGT encoder."""
        
        # Surface normals from depth gradients
        normals = self._compute_surface_normals(depth_map)
        
        # Curvature analysis
        curvature = self._compute_surface_curvature(depth_map)
        
        # Height statistics
        height_stats = self._compute_height_statistics(depth_map)
        
        # Texture-height correlation
        texture_height = self._analyze_texture_height_correlation(image, depth_map)
        
        spatial_features = {
            'depth_map': depth_map,
            'surface_normals': normals,
            'curvature': curvature,
            'height_stats': height_stats,
            'texture_height_correlation': texture_height
        }
        
        return spatial_features
    
    def _compute_surface_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute surface normals from depth gradients."""
        # Calculate depth gradients
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute normal vectors
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        normals = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
        normals[:, :, 0] = -grad_x
        normals[:, :, 1] = -grad_y
        normals[:, :, 2] = 1.0
        
        # Normalize vectors
        norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True)) + 1e-8
        normals = normals / norm
        
        return normals
    
    def _compute_surface_curvature(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute surface curvature from depth map."""
        # Ensure depth_map is float32
        depth_map = depth_map.astype(np.float32)
        
        # Calculate second derivatives
        grad_xx = cv2.Sobel(cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 1, 0, ksize=3).astype(np.float32)
        grad_yy = cv2.Sobel(cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3), cv2.CV_32F, 0, 1, ksize=3).astype(np.float32)
        grad_xy = cv2.Sobel(cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 0, 1, ksize=3).astype(np.float32)
        
        # Mean curvature approximation
        curvature = ((grad_xx + grad_yy) / 2.0).astype(np.float32)
        
        return curvature
    
    def _compute_height_statistics(self, depth_map: np.ndarray) -> Dict[str, float]:
        """Compute statistical measures of height distribution."""
        flat_depth = depth_map.flatten()
        
        stats = {
            'mean_height': float(np.mean(flat_depth)),
            'std_height': float(np.std(flat_depth)),
            'min_height': float(np.min(flat_depth)),
            'max_height': float(np.max(flat_depth)),
            'height_range': float(np.max(flat_depth) - np.min(flat_depth)),
            'skewness': float(self._compute_skewness(flat_depth)),
            'height_percentiles': {
                'p25': float(np.percentile(flat_depth, 25)),
                'p50': float(np.percentile(flat_depth, 50)),
                'p75': float(np.percentile(flat_depth, 75)),
                'p90': float(np.percentile(flat_depth, 90))
            }
        }
        
        return stats
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of height distribution."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _analyze_texture_height_correlation(self, image: np.ndarray, depth_map: np.ndarray) -> Dict[str, float]:
        """Analyze correlation between texture and height."""
        # Ensure image is uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_resized = cv2.resize(gray, depth_map.shape[::-1]).astype(np.float32)
        
        # Ensure depth_map is float32
        depth_map = depth_map.astype(np.float32)
        
        # Calculate texture measures
        # Variance in local windows
        laplacian_result = cv2.Laplacian(gray_resized, cv2.CV_32F).astype(np.float32)
        texture_variance = float(laplacian_result.var())
        
        # Correlation between brightness and height
        correlation = np.corrcoef(gray_resized.flatten(), depth_map.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Local contrast analysis
        sobel_x = cv2.Sobel(gray_resized, cv2.CV_32F, 1, 0, ksize=3).astype(np.float32)
        sobel_y = cv2.Sobel(gray_resized, cv2.CV_32F, 0, 1, ksize=3).astype(np.float32)
        local_contrast = (sobel_x**2 + sobel_y**2).astype(np.float32)
        contrast_height_corr = np.corrcoef(local_contrast.flatten(), depth_map.flatten())[0, 1]
        if np.isnan(contrast_height_corr):
            contrast_height_corr = 0.0
        
        return {
            'texture_variance': float(texture_variance),
            'brightness_height_correlation': float(correlation),
            'contrast_height_correlation': float(contrast_height_corr)
        }
    
    def format_for_spatial_mllm(self, image: np.ndarray, spatial_features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Format preprocessed data for Spatial-MLLM dual-encoder input.
        
        Args:
            image: Original RGB image
            spatial_features: Extracted spatial features
            
        Returns:
            formatted_data: Dictionary with tensors for both encoders
        """
        # Convert image to PIL for transforms
        pil_image = Image.fromarray(image)
        
        # 2D Visual Encoder input (Qwen2.5-VL format)
        visual_tensor = self.visual_transform(pil_image)  # Shape: (3, H, W)
        
        # 3D Spatial data for VGGT encoder
        # Combine depth map with surface normals as multi-channel spatial data
        depth_map = spatial_features['depth_map']
        surface_normals = spatial_features['surface_normals']
        
        # Create spatial tensor: depth + normals = 4 channels
        spatial_data = np.concatenate([
            depth_map[..., np.newaxis],  # Add channel dimension to depth
            surface_normals  # Already has 3 channels (x, y, z normals)
        ], axis=2)  # Shape: (H, W, 4)
        
        # Convert to tensor and rearrange to (C, H, W)
        spatial_tensor = torch.from_numpy(spatial_data).permute(2, 0, 1).float()
        
        # Format for Spatial-MLLM input
        # Both tensors need batch and frame dimensions: (B, F, C, H, W)
        visual_input = visual_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
        spatial_input = spatial_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, H, W)
        
        formatted_data = {
            'visual_input': visual_input,  # For Qwen2.5-VL encoder
            'spatial_input': spatial_input,  # For VGGT encoder  
            'spatial_features': spatial_features,  # Additional features for analysis
            'metadata': {
                'original_shape': image.shape,
                'output_shape': self.output_size,
                'depth_method': self.depth_method,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return formatted_data
    
    def validate_preprocessing_quality(self, original_image: np.ndarray, 
                                     processed_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validate the quality of preprocessing results."""
        if not self.enable_validation:
            return {'validation_skipped': True}
        
        metrics = {}
        
        # Extract data for validation
        visual_tensor = processed_data['visual_input'].squeeze()  # Remove batch/frame dims
        spatial_features = processed_data['spatial_features']
        depth_map = spatial_features['depth_map']
        
        # 1. Visual preprocessing quality
        # Check if image maintains important visual information
        visual_np = visual_tensor.permute(1, 2, 0).numpy()  # Convert back to HWC
        visual_np = (visual_np * np.array([0.26862954, 0.26130258, 0.27577711]) + 
                    np.array([0.48145466, 0.4578275, 0.40821073]))  # Denormalize
        visual_np = np.clip(visual_np, 0, 1)
        
        # Structural similarity with original (resized)
        original_resized = cv2.resize(original_image, self.output_size) / 255.0
        ssim_score = self._compute_ssim(original_resized, visual_np)
        metrics['visual_ssim'] = ssim_score
        
        # 2. Depth map quality
        # Check depth map consistency and realism
        depth_variance = np.var(depth_map)
        depth_gradient_consistency = self._check_depth_gradient_consistency(depth_map)
        
        metrics['depth_variance'] = float(depth_variance)
        metrics['depth_gradient_consistency'] = depth_gradient_consistency
        
        # 3. Spatial feature quality
        height_stats = spatial_features['height_stats']
        metrics['height_range'] = height_stats['height_range']
        metrics['height_distribution_quality'] = self._assess_height_distribution(height_stats)
        
        # 4. Overall quality score
        quality_components = [
            ssim_score * 0.4,  # Visual preservation
            min(depth_variance * 10, 1.0) * 0.3,  # Depth variability
            depth_gradient_consistency * 0.2,  # Depth consistency
            metrics['height_distribution_quality'] * 0.1  # Height distribution
        ]
        
        metrics['overall_quality'] = sum(quality_components)
        
        # Quality classification
        if metrics['overall_quality'] > 0.8:
            metrics['quality_level'] = 'excellent'
        elif metrics['overall_quality'] > 0.6:
            metrics['quality_level'] = 'good'
        elif metrics['overall_quality'] > 0.4:
            metrics['quality_level'] = 'acceptable'
        else:
            metrics['quality_level'] = 'poor'
        
        return metrics
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute simple SSIM-like metric."""
        # Ensure images are uint8 format for OpenCV operations
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
            
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Simple correlation-based similarity
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def _check_depth_gradient_consistency(self, depth_map: np.ndarray) -> float:
        """Check if depth gradients are smooth and consistent."""
        # Calculate gradients
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate second derivatives (should be relatively smooth)
        grad_xx = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3)
        
        # Measure smoothness (lower second derivative variance = smoother)
        smoothness = 1.0 / (1.0 + np.var(grad_xx) + np.var(grad_yy))
        
        return float(smoothness)
    
    def _assess_height_distribution(self, height_stats: Dict[str, Any]) -> float:
        """Assess quality of height distribution."""
        # Good height distribution should have:
        # 1. Reasonable range (not too flat, not too extreme)
        # 2. Realistic skewness for pizza surfaces
        
        range_score = min(height_stats['height_range'] * 2, 1.0)  # Prefer some height variation
        
        # Prefer slight positive skewness (pizza typically has some raised areas)
        skewness = height_stats['skewness']
        skewness_score = max(0, 1.0 - abs(skewness - 0.3) / 2.0)  # Optimal around 0.3
        
        distribution_quality = (range_score + skewness_score) / 2.0
        return distribution_quality
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single image through the complete spatial preprocessing pipeline.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            processed_data: Complete preprocessing results
        """
        start_time = datetime.now()
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            logger.info(f"Processing image: {image_path}")
            
            # Generate depth map
            depth_map = self.generate_depth_map(image_np)
            
            # Extract spatial features
            spatial_features = self.extract_spatial_features(image_np, depth_map)
            
            # Format for Spatial-MLLM
            formatted_data = self.format_for_spatial_mllm(image_np, spatial_features)
            
            # Validate quality
            quality_metrics = self.validate_preprocessing_quality(image_np, formatted_data)
            
            # Combine all results
            processed_data = {
                **formatted_data,
                'quality_metrics': quality_metrics,
                'source_path': str(image_path)
            }
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processed_images'] += 1
            self.stats['processing_times'].append(processing_time)
            if 'overall_quality' in quality_metrics:
                self.stats['depth_quality_scores'].append(quality_metrics['overall_quality'])
            
            logger.info(f"Successfully processed {image_path} in {processing_time:.2f}s "
                       f"(quality: {quality_metrics.get('quality_level', 'unknown')})")
            
            return processed_data
            
        except Exception as e:
            self.stats['failed_images'] += 1
            logger.error(f"Failed to process {image_path}: {e}")
            return {'error': str(e), 'source_path': str(image_path)}
    
    def process_dataset(self, 
                       input_dir: Union[str, Path], 
                       output_dir: Union[str, Path],
                       max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a complete dataset through spatial preprocessing.
        
        Args:
            input_dir: Directory containing pizza images
            output_dir: Directory to save processed data
            max_images: Maximum number of images to process (None for all)
            
        Returns:
            processing_summary: Summary of processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))
            image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process in {input_dir}")
        
        # Process each image
        processed_results = []
        
        for i, image_path in enumerate(image_files):
            logger.info(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
            
            # Process image
            result = self.process_image(image_path)
            
            if 'error' not in result:
                # Save processed data
                relative_path = image_path.relative_to(input_dir)
                output_file = output_dir / f"{relative_path.stem}_spatial.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save tensors and metadata
                torch.save({
                    'visual_input': result['visual_input'],
                    'spatial_input': result['spatial_input'],
                    'spatial_features': result['spatial_features'],
                    'quality_metrics': result['quality_metrics'],
                    'metadata': result['metadata']
                }, output_file)
                
                logger.info(f"Saved processed data to {output_file}")
            
            processed_results.append(result)
        
        # Generate processing summary
        summary = self._generate_processing_summary(processed_results, output_dir)
        
        # Save summary
        summary_file = output_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Processing complete. Summary saved to {summary_file}")
        
        return summary
    
    def _generate_processing_summary(self, results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive processing summary."""
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        summary = {
            'processing_statistics': {
                'total_images': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0
            },
            'performance_metrics': {
                'average_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'total_processing_time': sum(self.stats['processing_times']),
                'min_processing_time': min(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'max_processing_time': max(self.stats['processing_times']) if self.stats['processing_times'] else 0
            },
            'quality_analysis': self._analyze_quality_metrics(successful_results),
            'configuration': {
                'output_size': self.output_size,
                'depth_estimation_method': self.depth_method,
                'enable_validation': self.enable_validation
            },
            'output_directory': str(output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        if failed_results:
            summary['failed_images'] = [r['source_path'] for r in failed_results]
            summary['error_analysis'] = self._analyze_errors(failed_results)
        
        return summary
    
    def _analyze_quality_metrics(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality metrics across all processed images."""
        if not successful_results:
            return {'no_data': True}
        
        # Extract quality metrics
        quality_scores = []
        quality_levels = []
        depth_variances = []
        visual_ssims = []
        
        for result in successful_results:
            metrics = result.get('quality_metrics', {})
            if 'overall_quality' in metrics:
                quality_scores.append(metrics['overall_quality'])
            if 'quality_level' in metrics:
                quality_levels.append(metrics['quality_level'])
            if 'depth_variance' in metrics:
                depth_variances.append(metrics['depth_variance'])
            if 'visual_ssim' in metrics:
                visual_ssims.append(metrics['visual_ssim'])
        
        analysis = {
            'overall_quality': {
                'mean': np.mean(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0,
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0
            },
            'quality_distribution': {
                level: quality_levels.count(level) for level in set(quality_levels)
            } if quality_levels else {},
            'depth_variance_stats': {
                'mean': np.mean(depth_variances) if depth_variances else 0,
                'std': np.std(depth_variances) if depth_variances else 0
            },
            'visual_similarity_stats': {
                'mean': np.mean(visual_ssims) if visual_ssims else 0,
                'std': np.std(visual_ssims) if visual_ssims else 0
            }
        }
        
        return analysis
    
    def _analyze_errors(self, failed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns in failed processing."""
        error_types = {}
        
        for result in failed_results:
            error = result.get('error', 'unknown')
            # Categorize errors
            if 'memory' in error.lower() or 'allocation' in error.lower():
                category = 'memory_error'
            elif 'format' in error.lower() or 'decode' in error.lower():
                category = 'format_error'
            elif 'size' in error.lower() or 'dimension' in error.lower():
                category = 'size_error'
            else:
                category = 'other_error'
            
            error_types[category] = error_types.get(category, 0) + 1
        
        return {
            'error_categories': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spatial preprocessing pipeline for Spatial-MLLM')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing pizza images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--depth_method', type=str, default='edge_based',
                       choices=['monocular', 'shape_from_shading', 'edge_based'],
                       help='Depth estimation method')
    parser.add_argument('--output_size', type=int, nargs=2, default=[518, 518],
                       help='Output image size (height width)')
    parser.add_argument('--disable_validation', action='store_true',
                       help='Disable quality validation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpatialPreprocessingPipeline(
        output_size=tuple(args.output_size),
        depth_estimation_method=args.depth_method,
        enable_quality_validation=not args.disable_validation
    )
    
    # Process dataset
    summary = pipeline.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_images=args.max_images
    )
    
    # Print summary
    print("\n" + "="*50)
    print("SPATIAL PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Total images: {summary['processing_statistics']['total_images']}")
    print(f"Successful: {summary['processing_statistics']['successful']}")
    print(f"Failed: {summary['processing_statistics']['failed']}")
    print(f"Success rate: {summary['processing_statistics']['success_rate']:.2%}")
    print(f"Average processing time: {summary['performance_metrics']['average_processing_time']:.2f}s")
    
    if 'quality_analysis' in summary and 'overall_quality' in summary['quality_analysis']:
        quality = summary['quality_analysis']['overall_quality']
        print(f"Average quality score: {quality['mean']:.3f} ± {quality['std']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
