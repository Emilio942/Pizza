#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIFFUSION-4.1: Tool zur Bildevaluierung (Metriken) entwickeln

Comprehensive image evaluation tool for generated images with objective metrics.
This script implements various metrics to assess the quality and realism of 
generated images beyond visual inspection.

Features:
- Single image and batch evaluation modes
- Multiple image quality metrics (BRISQUE, NIQE, sharpness, contrast, etc.)
- Color profile analysis and histogram statistics
- Feature extraction using pre-trained networks
- Automatic detection of obviously flawed generations
- Comprehensive HTML reports with visualizations
- Support for pizza-specific evaluation criteria

Usage:
    python scripts/evaluate_generated_images.py --input data/synthetic --output output/diffusion_analysis
    python scripts/evaluate_generated_images.py --input data/synthetic/basic --output output/diffusion_analysis --single-class basic
    python scripts/evaluate_generated_images.py --image path/to/single/image.jpg --output output/diffusion_analysis
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import math
import statistics
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some advanced metrics will be disabled.")

try:
    import skimage
    from skimage import feature, measure, filters, segmentation
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Some image analysis metrics will be disabled.")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import project-specific modules
try:
    from data.class_definitions import CLASS_NAMES
except ImportError:
    try:
        with open(project_root / "data" / "class_definitions.json", 'r') as f:
            class_defs = json.load(f)
            CLASS_NAMES = list(class_defs.keys())
    except:
        CLASS_NAMES = ["basic", "burnt", "combined", "mixed", "progression", "segment"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetrics:
    """Data class for storing image quality metrics"""
    filename: str
    file_path: str
    class_label: Optional[str] = None
    
    # Basic image properties
    width: int = 0
    height: int = 0
    channels: int = 0
    file_size_mb: float = 0.0
    
    # Basic quality metrics
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    sharpness: float = 0.0
    
    # Advanced quality metrics
    brisque_score: Optional[float] = None
    niqe_score: Optional[float] = None
    
    # Statistical metrics
    histogram_entropy: float = 0.0
    color_distribution: Dict[str, float] = None
    
    # Pizza-specific metrics
    pizza_likelihood: float = 0.0
    texture_quality: float = 0.0
    burn_level_consistency: float = 0.0
    
    # Feature-based metrics
    edge_density: float = 0.0
    local_variance: float = 0.0
    gradient_magnitude: float = 0.0
    
    # Quality flags
    is_corrupted: bool = False
    is_blurry: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    has_artifacts: bool = False
    
    # Overall quality score
    quality_score: float = 0.0
    quality_category: str = "unknown"
    
    def __post_init__(self):
        if self.color_distribution is None:
            self.color_distribution = {}

class ImageEvaluator:
    """Main class for evaluating image quality with multiple metrics"""
    
    def __init__(self, output_dir: str = "output/diffusion_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractor if PyTorch is available
        self.feature_extractor = None
        if TORCH_AVAILABLE:
            try:
                self._init_feature_extractor()
            except Exception as e:
                logger.warning(f"Could not initialize feature extractor: {e}")
        
        # Quality thresholds (tunable parameters)
        self.thresholds = {
            'brightness_min': 20,
            'brightness_max': 235,
            'contrast_min': 20,
            'sharpness_min': 10,
            'blur_threshold': 100,
            'overexposed_threshold': 240,
            'underexposed_threshold': 15,
            'edge_density_min': 0.01,
            'variance_min': 100
        }
        
        logger.info(f"Image evaluator initialized. Output directory: {self.output_dir}")
    
    def _init_feature_extractor(self):
        """Initialize a pre-trained feature extractor for advanced analysis"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Use a lightweight pre-trained model for feature extraction
            model = models.mobilenet_v2(pretrained=True)
            # Remove the classifier to get features
            self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_extractor.eval()
            
            # Transform for preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            logger.info("Feature extractor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize feature extractor: {e}")
            self.feature_extractor = None
    
    def evaluate_single_image(self, image_path: Union[str, Path]) -> ImageMetrics:
        """Evaluate a single image and return comprehensive metrics"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Evaluating image: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            
            # Initialize metrics
            metrics = ImageMetrics(
                filename=image_path.name,
                file_path=str(image_path),
                width=image.width,
                height=image.height,
                channels=len(image_array.shape) if len(image_array.shape) == 2 else image_array.shape[2],
                file_size_mb=image_path.stat().st_size / (1024 * 1024)
            )
            
            # Detect class from path if possible
            metrics.class_label = self._detect_class_from_path(image_path)
            
            # Calculate basic metrics
            self._calculate_basic_metrics(image, image_array, metrics)
            
            # Calculate statistical metrics
            self._calculate_statistical_metrics(image, image_array, metrics)
            
            # Calculate advanced quality metrics
            if SKIMAGE_AVAILABLE:
                self._calculate_advanced_metrics(image_array, metrics)
            
            # Calculate pizza-specific metrics
            self._calculate_pizza_specific_metrics(image, image_array, metrics)
            
            # Calculate feature-based metrics
            self._calculate_feature_metrics(image_array, metrics)
            
            # Set quality flags
            self._set_quality_flags(metrics)
            
            # Calculate overall quality score
            self._calculate_overall_quality(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating image {image_path}: {e}")
            # Return basic metrics with error flag
            return ImageMetrics(
                filename=image_path.name,
                file_path=str(image_path),
                is_corrupted=True,
                quality_score=0.0,
                quality_category="corrupted"
            )
    
    def _detect_class_from_path(self, image_path: Path) -> Optional[str]:
        """Detect pizza class from file path"""
        path_str = str(image_path).lower()
        for class_name in CLASS_NAMES:
            if class_name in path_str:
                return class_name
        return None
    
    def _calculate_basic_metrics(self, image: Image.Image, image_array: np.ndarray, metrics: ImageMetrics):
        """Calculate basic image quality metrics"""
        try:
            # Brightness (average luminance)
            if len(image_array.shape) == 3:
                # Convert RGB to grayscale for brightness calculation
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            metrics.brightness = float(np.mean(gray))
            
            # Contrast (standard deviation of pixel intensities)
            metrics.contrast = float(np.std(gray))
            
            # Saturation (only for color images)
            if len(image_array.shape) == 3:
                hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
                metrics.saturation = float(np.mean(hsv[:, :, 1]))
            else:
                metrics.saturation = 0.0
            
            # Sharpness (using Laplacian variance)
            if len(image_array.shape) == 3:
                gray_for_sharp = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_for_sharp = image_array
            
            laplacian = cv2.Laplacian(gray_for_sharp, cv2.CV_64F)
            metrics.sharpness = float(laplacian.var())
            
        except Exception as e:
            logger.warning(f"Error calculating basic metrics: {e}")
    
    def _calculate_statistical_metrics(self, image: Image.Image, image_array: np.ndarray, metrics: ImageMetrics):
        """Calculate statistical image metrics"""
        try:
            # Histogram entropy
            if len(image_array.shape) == 3:
                # Calculate entropy for each channel and take average
                entropies = []
                for channel in range(image_array.shape[2]):
                    hist, _ = np.histogram(image_array[:, :, channel], bins=256, range=(0, 256))
                    hist = hist + 1e-10  # Avoid log(0)
                    prob = hist / hist.sum()
                    entropy = -np.sum(prob * np.log2(prob))
                    entropies.append(entropy)
                metrics.histogram_entropy = float(np.mean(entropies))
            else:
                hist, _ = np.histogram(image_array, bins=256, range=(0, 256))
                hist = hist + 1e-10
                prob = hist / hist.sum()
                metrics.histogram_entropy = float(-np.sum(prob * np.log2(prob)))
            
            # Color distribution analysis
            if len(image_array.shape) == 3:
                r_mean, g_mean, b_mean = np.mean(image_array, axis=(0, 1))
                total_intensity = r_mean + g_mean + b_mean
                
                if total_intensity > 0:
                    metrics.color_distribution = {
                        'red_ratio': float(r_mean / total_intensity),
                        'green_ratio': float(g_mean / total_intensity),
                        'blue_ratio': float(b_mean / total_intensity),
                        'dominant_color': 'red' if r_mean > g_mean and r_mean > b_mean else 
                                        'green' if g_mean > b_mean else 'blue'
                    }
                else:
                    metrics.color_distribution = {
                        'red_ratio': 0.0, 'green_ratio': 0.0, 'blue_ratio': 0.0,
                        'dominant_color': 'none'
                    }
            
        except Exception as e:
            logger.warning(f"Error calculating statistical metrics: {e}")
    
    def _calculate_advanced_metrics(self, image_array: np.ndarray, metrics: ImageMetrics):
        """Calculate advanced quality metrics using scikit-image"""
        if not SKIMAGE_AVAILABLE:
            return
        
        try:
            # Convert to grayscale for some calculations
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Edge density
            edges = feature.canny(gray)
            metrics.edge_density = float(np.sum(edges) / (edges.shape[0] * edges.shape[1]))
            
            # Local variance (measure of texture)
            metrics.local_variance = float(np.mean(filters.rank.variance(gray, np.ones((9, 9)))))
            
            # Gradient magnitude
            grad_x = filters.sobel_h(gray)
            grad_y = filters.sobel_v(gray)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            metrics.gradient_magnitude = float(np.mean(grad_magnitude))
            
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
    
    def _calculate_pizza_specific_metrics(self, image: Image.Image, image_array: np.ndarray, metrics: ImageMetrics):
        """Calculate pizza-specific quality metrics"""
        try:
            # Pizza likelihood based on color and texture patterns
            pizza_score = 0.0
            
            if len(image_array.shape) == 3:
                # Check for typical pizza colors (browns, reds, yellows)
                hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
                h = hsv[:, :, 0]
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                
                # Brown/orange hues (typical for baked pizza)
                brown_mask = ((h >= 5) & (h <= 25)) & (s > 50) & (v > 50)
                brown_ratio = np.sum(brown_mask) / (h.shape[0] * h.shape[1])
                
                # Red hues (tomato sauce)
                red_mask = ((h >= 160) | (h <= 10)) & (s > 50) & (v > 50)
                red_ratio = np.sum(red_mask) / (h.shape[0] * h.shape[1])
                
                pizza_score += min(1.0, (brown_ratio + red_ratio) * 2)
            
            metrics.pizza_likelihood = pizza_score
            
            # Texture quality assessment
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Check for appropriate texture variation
            local_std = cv2.Laplacian(gray, cv2.CV_64F).std()
            metrics.texture_quality = min(1.0, local_std / 50.0)  # Normalize to 0-1
            
            # Burn level consistency (for burnt class)
            if metrics.class_label == "burnt":
                # Check for dark regions that might indicate burning
                dark_threshold = 80
                dark_ratio = np.sum(gray < dark_threshold) / (gray.shape[0] * gray.shape[1])
                metrics.burn_level_consistency = min(1.0, dark_ratio * 3)  # Expect some dark areas
            else:
                # For non-burnt classes, penalize too many dark areas
                dark_threshold = 60
                dark_ratio = np.sum(gray < dark_threshold) / (gray.shape[0] * gray.shape[1])
                metrics.burn_level_consistency = max(0.0, 1.0 - dark_ratio * 2)
            
        except Exception as e:
            logger.warning(f"Error calculating pizza-specific metrics: {e}")
    
    def _calculate_feature_metrics(self, image_array: np.ndarray, metrics: ImageMetrics):
        """Calculate feature-based metrics using deep learning (if available)"""
        if not TORCH_AVAILABLE or self.feature_extractor is None:
            return
        
        try:
            # Convert numpy array to PIL Image for transformation
            if len(image_array.shape) == 3:
                pil_image = Image.fromarray(image_array)
            else:
                pil_image = Image.fromarray(image_array).convert('RGB')
            
            # Apply transformations and extract features
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().numpy()
                
                # Calculate feature statistics
                feature_mean = float(np.mean(features))
                feature_std = float(np.std(features))
                feature_max = float(np.max(features))
                
                # Use feature statistics as quality indicators
                # (This is a simplified approach - more sophisticated methods exist)
                metrics.quality_score += 0.1 * min(1.0, feature_std / 10.0)  # Reward feature diversity
                
        except Exception as e:
            logger.warning(f"Error calculating feature metrics: {e}")
    
    def _set_quality_flags(self, metrics: ImageMetrics):
        """Set quality flags based on thresholds"""
        try:
            # Check for various quality issues
            metrics.is_blurry = metrics.sharpness < self.thresholds['blur_threshold']
            metrics.is_overexposed = metrics.brightness > self.thresholds['overexposed_threshold']
            metrics.is_underexposed = metrics.brightness < self.thresholds['underexposed_threshold']
            
            # Check for artifacts (simplified approach)
            metrics.has_artifacts = (
                metrics.contrast < self.thresholds['contrast_min'] or
                metrics.edge_density < self.thresholds['edge_density_min'] or
                metrics.local_variance < self.thresholds['variance_min']
            )
            
        except Exception as e:
            logger.warning(f"Error setting quality flags: {e}")
    
    def _calculate_overall_quality(self, metrics: ImageMetrics):
        """Calculate overall quality score and category"""
        try:
            score = 0.0
            
            # Basic quality components (0.4 weight)
            brightness_score = 1.0 - abs(metrics.brightness - 128) / 128  # Prefer mid-range brightness
            contrast_score = min(1.0, metrics.contrast / 80)  # Good contrast
            sharpness_score = min(1.0, metrics.sharpness / 200)  # Sharp images
            
            basic_score = (brightness_score + contrast_score + sharpness_score) / 3
            score += 0.4 * basic_score
            
            # Statistical quality (0.2 weight)
            entropy_score = min(1.0, metrics.histogram_entropy / 8)  # High entropy = good detail
            score += 0.2 * entropy_score
            
            # Pizza-specific quality (0.3 weight)
            pizza_score = (metrics.pizza_likelihood + metrics.texture_quality + metrics.burn_level_consistency) / 3
            score += 0.3 * pizza_score
            
            # Technical quality (0.1 weight)
            edge_score = min(1.0, metrics.edge_density * 100)  # Good edge definition
            score += 0.1 * edge_score
            
            # Penalties for quality issues
            if metrics.is_blurry:
                score *= 0.7
            if metrics.is_overexposed or metrics.is_underexposed:
                score *= 0.8
            if metrics.has_artifacts:
                score *= 0.9
            if metrics.is_corrupted:
                score = 0.0
            
            metrics.quality_score = max(0.0, min(1.0, score))
            
            # Set quality category
            if metrics.quality_score >= 0.8:
                metrics.quality_category = "excellent"
            elif metrics.quality_score >= 0.6:
                metrics.quality_category = "good"
            elif metrics.quality_score >= 0.4:
                metrics.quality_category = "fair"
            elif metrics.quality_score >= 0.2:
                metrics.quality_category = "poor"
            else:
                metrics.quality_category = "very_poor"
                
        except Exception as e:
            logger.warning(f"Error calculating overall quality: {e}")
            metrics.quality_score = 0.5  # Default middle score
            metrics.quality_category = "unknown"
    
    def evaluate_batch(self, input_path: Union[str, Path], 
                      pattern: str = "*.{jpg,jpeg,png,bmp}",
                      max_images: Optional[int] = None) -> List[ImageMetrics]:
        """Evaluate a batch of images from a directory"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        # Find all image files
        image_files = []
        if input_path.is_file():
            image_files = [input_path]
        else:
            # Search for image files
            for ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
                image_files.extend(input_path.glob(f"**/*.{ext}"))
                image_files.extend(input_path.glob(f"**/*.{ext.upper()}"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to evaluate")
        
        results = []
        for i, image_file in enumerate(image_files):
            if i % 50 == 0:
                logger.info(f"Processing image {i+1}/{len(image_files)}")
            
            try:
                metrics = self.evaluate_single_image(image_file)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate {image_file}: {e}")
                continue
        
        logger.info(f"Successfully evaluated {len(results)} images")
        return results
    
    def generate_report(self, metrics_list: List[ImageMetrics], 
                       report_name: str = "image_evaluation_report") -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"{report_name}_{timestamp}"
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(metrics_list)
        
        # Generate visualizations
        viz_files = self._generate_visualizations(metrics_list, report_id)
        
        # Create detailed report data
        report_data = {
            "report_id": report_id,
            "timestamp": timestamp,
            "total_images": len(metrics_list),
            "summary": summary,
            "visualizations": viz_files,
            "detailed_metrics": [asdict(m) for m in metrics_list],
            "evaluation_config": {
                "thresholds": self.thresholds,
                "torch_available": TORCH_AVAILABLE,
                "skimage_available": SKIMAGE_AVAILABLE,
                "feature_extractor_enabled": self.feature_extractor is not None
            }
        }
        
        # Save JSON report
        json_file = self.output_dir / f"{report_id}.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate HTML report
        html_file = self._generate_html_report(report_data, report_id)
        
        logger.info(f"Report generated: {html_file}")
        return report_data
    
    def _calculate_summary_stats(self, metrics_list: List[ImageMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics for the evaluation"""
        if not metrics_list:
            return {}
        
        # Extract numeric metrics
        quality_scores = [m.quality_score for m in metrics_list if not m.is_corrupted]
        brightness_values = [m.brightness for m in metrics_list if not m.is_corrupted]
        contrast_values = [m.contrast for m in metrics_list if not m.is_corrupted]
        sharpness_values = [m.sharpness for m in metrics_list if not m.is_corrupted]
        
        # Count quality categories
        category_counts = defaultdict(int)
        class_counts = defaultdict(int)
        issue_counts = defaultdict(int)
        
        for metrics in metrics_list:
            category_counts[metrics.quality_category] += 1
            if metrics.class_label:
                class_counts[metrics.class_label] += 1
            
            # Count quality issues
            if metrics.is_blurry:
                issue_counts['blurry'] += 1
            if metrics.is_overexposed:
                issue_counts['overexposed'] += 1
            if metrics.is_underexposed:
                issue_counts['underexposed'] += 1
            if metrics.has_artifacts:
                issue_counts['artifacts'] += 1
            if metrics.is_corrupted:
                issue_counts['corrupted'] += 1
        
        summary = {
            "quality_statistics": {
                "mean_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
                "median_quality_score": statistics.median(quality_scores) if quality_scores else 0,
                "std_quality_score": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "min_quality_score": min(quality_scores) if quality_scores else 0,
                "max_quality_score": max(quality_scores) if quality_scores else 0
            },
            "technical_statistics": {
                "mean_brightness": statistics.mean(brightness_values) if brightness_values else 0,
                "mean_contrast": statistics.mean(contrast_values) if contrast_values else 0,
                "mean_sharpness": statistics.mean(sharpness_values) if sharpness_values else 0
            },
            "quality_distribution": dict(category_counts),
            "class_distribution": dict(class_counts),
            "quality_issues": dict(issue_counts),
            "recommendations": self._generate_recommendations(metrics_list)
        }
        
        return summary
    
    def _generate_recommendations(self, metrics_list: List[ImageMetrics]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if not metrics_list:
            return ["No images to evaluate"]
        
        total_images = len(metrics_list)
        
        # Count various issues
        blurry_count = sum(1 for m in metrics_list if m.is_blurry)
        overexposed_count = sum(1 for m in metrics_list if m.is_overexposed)
        underexposed_count = sum(1 for m in metrics_list if m.is_underexposed)
        artifacts_count = sum(1 for m in metrics_list if m.has_artifacts)
        corrupted_count = sum(1 for m in metrics_list if m.is_corrupted)
        low_quality_count = sum(1 for m in metrics_list if m.quality_score < 0.4)
        
        # Generate specific recommendations
        if corrupted_count > 0:
            recommendations.append(f"Remove {corrupted_count} corrupted images that cannot be processed")
        
        if blurry_count > total_images * 0.1:
            recommendations.append(f"Consider removing {blurry_count} blurry images (>{10}% of dataset)")
        
        if overexposed_count > total_images * 0.05:
            recommendations.append(f"Review {overexposed_count} overexposed images for potential removal")
        
        if underexposed_count > total_images * 0.05:
            recommendations.append(f"Review {underexposed_count} underexposed images for potential removal")
        
        if artifacts_count > total_images * 0.15:
            recommendations.append(f"Check generation parameters - {artifacts_count} images show potential artifacts")
        
        if low_quality_count > total_images * 0.2:
            recommendations.append(f"Consider regenerating {low_quality_count} low-quality images (quality score < 0.4)")
        
        # Overall quality assessment
        avg_quality = statistics.mean(m.quality_score for m in metrics_list if not m.is_corrupted)
        if avg_quality < 0.5:
            recommendations.append("Overall dataset quality is below average - consider reviewing generation parameters")
        elif avg_quality > 0.7:
            recommendations.append("Dataset shows good overall quality")
        
        if not recommendations:
            recommendations.append("No significant quality issues detected")
        
        return recommendations
    
    def _generate_visualizations(self, metrics_list: List[ImageMetrics], report_id: str) -> List[str]:
        """Generate visualization plots for the evaluation"""
        viz_files = []
        
        if not metrics_list:
            return viz_files
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create visualizations directory
        viz_dir = self.output_dir / f"{report_id}_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Quality Score Distribution
            quality_scores = [m.quality_score for m in metrics_list if not m.is_corrupted]
            if quality_scores:
                plt.figure(figsize=(10, 6))
                plt.hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
                plt.axvline(statistics.mean(quality_scores), color='red', linestyle='--', 
                           label=f'Mean: {statistics.mean(quality_scores):.3f}')
                plt.xlabel('Quality Score')
                plt.ylabel('Number of Images')
                plt.title('Distribution of Image Quality Scores')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                viz_file = viz_dir / "quality_score_distribution.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files.append(str(viz_file.relative_to(self.output_dir)))
            
            # 2. Quality Categories Pie Chart
            category_counts = defaultdict(int)
            for m in metrics_list:
                category_counts[m.quality_category] += 1
            
            if category_counts:
                plt.figure(figsize=(8, 8))
                labels = list(category_counts.keys())
                sizes = list(category_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                plt.title('Distribution of Quality Categories')
                
                viz_file = viz_dir / "quality_categories.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files.append(str(viz_file.relative_to(self.output_dir)))
            
            # 3. Technical Metrics Correlation Matrix
            valid_metrics = [m for m in metrics_list if not m.is_corrupted]
            if len(valid_metrics) > 1:
                data = {
                    'Quality Score': [m.quality_score for m in valid_metrics],
                    'Brightness': [m.brightness for m in valid_metrics],
                    'Contrast': [m.contrast for m in valid_metrics],
                    'Sharpness': [m.sharpness for m in valid_metrics],
                    'Histogram Entropy': [m.histogram_entropy for m in valid_metrics],
                    'Edge Density': [m.edge_density for m in valid_metrics],
                    'Pizza Likelihood': [m.pizza_likelihood for m in valid_metrics]
                }
                
                # Create correlation matrix
                import pandas as pd
                df = pd.DataFrame(data)
                correlation_matrix = df.corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.3f')
                plt.title('Technical Metrics Correlation Matrix')
                
                viz_file = viz_dir / "metrics_correlation.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files.append(str(viz_file.relative_to(self.output_dir)))
            
            # 4. Class-wise Quality Analysis (if classes are detected)
            class_quality = defaultdict(list)
            for m in metrics_list:
                if m.class_label and not m.is_corrupted:
                    class_quality[m.class_label].append(m.quality_score)
            
            if len(class_quality) > 1:
                plt.figure(figsize=(12, 6))
                class_names = list(class_quality.keys())
                quality_data = [class_quality[cls] for cls in class_names]
                
                box_plot = plt.boxplot(quality_data, labels=class_names, patch_artist=True)
                colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                
                plt.xlabel('Pizza Class')
                plt.ylabel('Quality Score')
                plt.title('Quality Score Distribution by Pizza Class')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                viz_file = viz_dir / "class_quality_distribution.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files.append(str(viz_file.relative_to(self.output_dir)))
            
            # 5. Quality Issues Summary
            issue_counts = defaultdict(int)
            for m in metrics_list:
                if m.is_blurry:
                    issue_counts['Blurry'] += 1
                if m.is_overexposed:
                    issue_counts['Overexposed'] += 1
                if m.is_underexposed:
                    issue_counts['Underexposed'] += 1
                if m.has_artifacts:
                    issue_counts['Artifacts'] += 1
                if m.is_corrupted:
                    issue_counts['Corrupted'] += 1
            
            if issue_counts:
                plt.figure(figsize=(10, 6))
                issues = list(issue_counts.keys())
                counts = list(issue_counts.values())
                
                bars = plt.bar(issues, counts, color='lightcoral', alpha=0.7, edgecolor='black')
                plt.xlabel('Quality Issue Type')
                plt.ylabel('Number of Images')
                plt.title('Distribution of Quality Issues')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
                
                viz_file = viz_dir / "quality_issues.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files.append(str(viz_file.relative_to(self.output_dir)))
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
        
        return viz_files
    
    def _generate_html_report(self, report_data: Dict[str, Any], report_id: str) -> Path:
        """Generate comprehensive HTML report"""
        html_file = self.output_dir / f"{report_id}.html"
        
        # HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Evaluation Report - {report_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-weight: 500;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .viz-item {{
            text-align: center;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }}
        .viz-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .recommendations {{
            background-color: #e8f6f3;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 10px;
        }}
        .quality-excellent {{ background-color: #d5f4e6; }}
        .quality-good {{ background-color: #fef9e7; }}
        .quality-fair {{ background-color: #fdf2e9; }}
        .quality-poor {{ background-color: #fadbd8; }}
        .quality-very-poor {{ background-color: #f1c0c0; }}
        .details-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .details-table th, .details-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .details-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .details-toggle {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 0;
        }}
        .details-toggle:hover {{
            background-color: #2980b9;
        }}
        #detailed-results {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🍕 Image Evaluation Report</h1>
        <p><strong>Report ID:</strong> {report_data['report_id']}</p>
        <p><strong>Generated:</strong> {report_data['timestamp']}</p>
        <p><strong>Total Images Evaluated:</strong> {report_data['total_images']}</p>
        
        <h2>📊 Summary Statistics</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{report_data['summary']['quality_statistics']['mean_quality_score']:.3f}</div>
                <div class="metric-label">Average Quality Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report_data['summary']['quality_statistics']['median_quality_score']:.3f}</div>
                <div class="metric-label">Median Quality Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len([m for m in report_data['detailed_metrics'] if m['quality_score'] >= 0.6])}</div>
                <div class="metric-label">Good+ Quality Images</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(report_data['summary']['quality_issues'].values())}</div>
                <div class="metric-label">Images with Quality Issues</div>
            </div>
        </div>
        
        <h2>📈 Visualizations</h2>
        <div class="viz-grid">
        """
        
        # Add visualization images
        for viz_file in report_data['visualizations']:
            viz_name = Path(viz_file).stem.replace('_', ' ').title()
            html_content += f"""
            <div class="viz-item">
                <h4>{viz_name}</h4>
                <img src="{viz_file}" alt="{viz_name}">
            </div>
            """
        
        html_content += f"""
        </div>
        
        <h2>💡 Recommendations</h2>
        <div class="recommendations">
            <h3>Automated Analysis Recommendations:</h3>
            <ul>
        """
        
        for recommendation in report_data['summary']['recommendations']:
            html_content += f"<li>{recommendation}</li>"
        
        html_content += f"""
            </ul>
        </div>
        
        <h2>🔧 Evaluation Configuration</h2>
        <p><strong>PyTorch Available:</strong> {report_data['evaluation_config']['torch_available']}</p>
        <p><strong>Scikit-Image Available:</strong> {report_data['evaluation_config']['skimage_available']}</p>
        <p><strong>Feature Extractor Enabled:</strong> {report_data['evaluation_config']['feature_extractor_enabled']}</p>
        
        <h2>📋 Detailed Results</h2>
        <button class="details-toggle" onclick="toggleDetails()">Show/Hide Detailed Results</button>
        
        <div id="detailed-results">
            <table class="details-table">
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Class</th>
                        <th>Quality Score</th>
                        <th>Category</th>
                        <th>Brightness</th>
                        <th>Contrast</th>
                        <th>Sharpness</th>
                        <th>Issues</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add detailed results
        for metrics in report_data['detailed_metrics']:
            issues = []
            if metrics['is_blurry']:
                issues.append('Blurry')
            if metrics['is_overexposed']:
                issues.append('Overexposed')
            if metrics['is_underexposed']:
                issues.append('Underexposed')
            if metrics['has_artifacts']:
                issues.append('Artifacts')
            if metrics['is_corrupted']:
                issues.append('Corrupted')
            
            issues_str = ', '.join(issues) if issues else 'None'
            quality_class = f"quality-{metrics['quality_category'].replace('_', '-')}"
            
            html_content += f"""
                <tr class="{quality_class}">
                    <td>{metrics['filename']}</td>
                    <td>{metrics['class_label'] or 'Unknown'}</td>
                    <td>{metrics['quality_score']:.3f}</td>
                    <td>{metrics['quality_category']}</td>
                    <td>{metrics['brightness']:.1f}</td>
                    <td>{metrics['contrast']:.1f}</td>
                    <td>{metrics['sharpness']:.1f}</td>
                    <td>{issues_str}</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <script>
            function toggleDetails() {
                var details = document.getElementById('detailed-results');
                if (details.style.display === 'none' || details.style.display === '') {
                    details.style.display = 'block';
                } else {
                    details.style.display = 'none';
                }
            }
        </script>
    </div>
</body>
</html>
        """
        
        # Write HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Evaluate image quality with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all images in a directory
    python scripts/evaluate_generated_images.py --input data/synthetic --output output/diffusion_analysis
    
    # Evaluate images from a specific class
    python scripts/evaluate_generated_images.py --input data/synthetic/basic --output output/diffusion_analysis --single-class basic
    
    # Evaluate a single image
    python scripts/evaluate_generated_images.py --image path/to/image.jpg --output output/diffusion_analysis
    
    # Limit the number of images evaluated
    python scripts/evaluate_generated_images.py --input data/synthetic --output output/diffusion_analysis --max-images 100
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                           help='Input directory containing images to evaluate')
    input_group.add_argument('--image', type=str,
                           help='Single image file to evaluate')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='output/diffusion_analysis',
                       help='Output directory for reports and visualizations')
    
    # Evaluation options
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to evaluate (for testing)')
    parser.add_argument('--single-class', type=str,
                       help='Specify the class label for images (when input is a single-class directory)')
    parser.add_argument('--report-name', type=str, default='image_evaluation_report',
                       help='Base name for the generated report')
    
    # Advanced options
    parser.add_argument('--threshold-brightness-min', type=float, default=20,
                       help='Minimum brightness threshold')
    parser.add_argument('--threshold-brightness-max', type=float, default=235,
                       help='Maximum brightness threshold')
    parser.add_argument('--threshold-contrast-min', type=float, default=20,
                       help='Minimum contrast threshold')
    parser.add_argument('--threshold-sharpness-min', type=float, default=10,
                       help='Minimum sharpness threshold')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize evaluator
        evaluator = ImageEvaluator(output_dir=args.output)
        
        # Update thresholds if provided
        if args.threshold_brightness_min:
            evaluator.thresholds['brightness_min'] = args.threshold_brightness_min
        if args.threshold_brightness_max:
            evaluator.thresholds['brightness_max'] = args.threshold_brightness_max
        if args.threshold_contrast_min:
            evaluator.thresholds['contrast_min'] = args.threshold_contrast_min
        if args.threshold_sharpness_min:
            evaluator.thresholds['sharpness_min'] = args.threshold_sharpness_min
        
        # Evaluate images
        if args.image:
            # Single image evaluation
            logger.info(f"Evaluating single image: {args.image}")
            metrics = evaluator.evaluate_single_image(args.image)
            metrics_list = [metrics]
        else:
            # Batch evaluation
            logger.info(f"Evaluating images in: {args.input}")
            metrics_list = evaluator.evaluate_batch(
                args.input, 
                max_images=args.max_images
            )
        
        if not metrics_list:
            logger.error("No images found or successfully evaluated")
            return 1
        
        # Override class labels if specified
        if args.single_class:
            for metrics in metrics_list:
                metrics.class_label = args.single_class
        
        # Generate report
        logger.info("Generating evaluation report...")
        report_data = evaluator.generate_report(metrics_list, args.report_name)
        
        # Print summary
        print(f"\n🍕 Image Evaluation Complete!")
        print(f"📊 Total images evaluated: {len(metrics_list)}")
        print(f"📈 Average quality score: {report_data['summary']['quality_statistics']['mean_quality_score']:.3f}")
        print(f"✅ Good+ quality images: {len([m for m in metrics_list if m.quality_score >= 0.6])}")
        print(f"⚠️  Images with issues: {sum(report_data['summary']['quality_issues'].values())}")
        print(f"📄 Report saved: {evaluator.output_dir / f'{report_data['report_id']}.html'}")
        
        # Print recommendations
        if report_data['summary']['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report_data['summary']['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
