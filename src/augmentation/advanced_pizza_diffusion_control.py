#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Pizza Diffusion Control Pipeline

This script enhances the pizza diffusion integration with controlled generation,
advanced prompts, and pipeline optimization for the RP2040 pizza recognition system.

It provides specialized features for generating realistic pizza images with precise 
control over cooking states and appearance. This implements the "control" layer between
the diffusion model and the pizza recognition pipeline.

Author: GitHub Copilot (2025-05-10)
"""

import os
import sys
import time
import argparse
import logging
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageStat, ImageOps, ImageFilter, ImageDraw, ImageChops

# Import the diffusion generator
from src.augmentation.diffusion_pizza_generator import PizzaDiffusionGenerator, PIZZA_STAGES

# Setup project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.utils.validation import validate_image_quality, validate_dataset
    from src.integration.diffusion_data_agent import AgentConfig, DiffusionDataAgent
    from src.utils.utils import setup_logging
    PROJECT_UTILS_AVAILABLE = True
except ImportError:
    PROJECT_UTILS_AVAILABLE = False
    print("Warning: Project utilities not found, using standalone mode")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diffusion_control.log')
    ]
)
logger = logging.getLogger(__name__)

# Temperature and cooking region control settings
COOKING_REGION_TEMPLATES = {
    "edge_burn": {
        "description": "Pizza burned around the edges",
        "regions": [
            (0.0, 0.0, 1.0, 0.15, "burnt"),  # Top edge
            (0.0, 0.0, 0.15, 1.0, "burnt"),  # Left edge
            (0.85, 0.0, 1.0, 1.0, "burnt"),  # Right edge
            (0.0, 0.85, 1.0, 1.0, "burnt"),  # Bottom edge
            (0.15, 0.15, 0.85, 0.85, "basic")  # Center
        ],
        "weight": 0.3
    },
    "center_burn": {
        "description": "Pizza burned in the center",
        "regions": [
            (0.3, 0.3, 0.7, 0.7, "burnt"),   # Center burn
            (0.0, 0.0, 1.0, 0.3, "basic"),   # Top region
            (0.0, 0.3, 0.3, 0.7, "basic"),   # Left region
            (0.7, 0.3, 1.0, 0.7, "basic"),   # Right region
            (0.0, 0.7, 1.0, 1.0, "basic")    # Bottom region
        ],
        "weight": 0.2
    },
    "half_burn": {
        "description": "Pizza half burned",
        "regions": [
            (0.0, 0.0, 0.5, 1.0, "burnt"),   # Left half
            (0.5, 0.0, 1.0, 1.0, "basic")    # Right half
        ],
        "weight": 0.15
    },
    "quarter_burn": {
        "description": "Pizza with one quarter burned",
        "regions": [
            (0.0, 0.0, 0.5, 0.5, "burnt"),   # Top-left quarter
            (0.5, 0.0, 1.0, 0.5, "basic"),   # Top-right quarter
            (0.0, 0.5, 0.5, 1.0, "basic"),   # Bottom-left quarter
            (0.5, 0.5, 1.0, 1.0, "basic")    # Bottom-right quarter
        ],
        "weight": 0.15
    },
    "random_spots": {
        "description": "Pizza with random burned spots",
        "dynamic": True,  # This indicates we'll generate spots procedurally
        "weight": 0.2
    }
}

class CookingControlMask:
    """
    Generate control masks for different pizza cooking patterns.
    These masks can be used to guide the diffusion process or for post-processing.
    """
    
    def __init__(
        self, 
        image_size: int = 1024,
        size: Tuple[int, int] = (512, 512)):
        """
        Initialize the cooking control mask generator.
        
        Args:
            size: Size of the masks to generate (width, height)
        """
        self.size = size
    
    def generate_template_mask(self, template_name: str) -> Image.Image:
        """
        Generate a mask based on a predefined template.
        
        Args:
            template_name: Name of the template to use
            
        Returns:
            PIL Image with the mask
        """
        if template_name not in COOKING_REGION_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = COOKING_REGION_TEMPLATES[template_name]
        
        # Create a blank mask
        mask = Image.new("RGB", self.size, (128, 128, 128))  # Neutral gray
        draw = ImageDraw.Draw(mask)
        
        # If it's a dynamic template, handle it specially
        if template.get("dynamic", False):
            if template_name == "random_spots":
                return self._generate_random_spots()
            else:
                # Default handler for unknown dynamic templates
                return mask
        
        # Draw regions based on template definition
        for region in template["regions"]:
            x1, y1, x2, y2, region_type = region
            
            # Convert relative coordinates to absolute
            x1, y1 = int(x1 * self.size[0]), int(y1 * self.size[1])
            x2, y2 = int(x2 * self.size[0]), int(y2 * self.size[1])
            
            # Set color based on region type
            if region_type == "burnt":
                color = (50, 50, 50)  # Dark for burnt
            elif region_type == "basic":
                color = (200, 200, 200)  # Light for basic/raw
            else:
                color = (128, 128, 128)  # Neutral for other
            
            # Draw the region
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        return mask
    
    def _generate_random_spots(self) -> Image.Image:
        """Generate a mask with random burnt spots"""
        # Create a blank mask
        mask = Image.new("RGB", self.size, (200, 200, 200))  # Light for basic
        draw = ImageDraw.Draw(mask)
        
        # Number of spots
        num_spots = random.randint(3, 8)
        
        # Draw random spots
        for _ in range(num_spots):
            # Random position
            x = random.randint(0, self.size[0])
            y = random.randint(0, self.size[1])
            
            # Random radius
            radius = random.randint(20, 80)
            
            # Draw the spot
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(50, 50, 50))
        
        return mask
    
    def combine_masks(self, masks: List[Image.Image], weights: Optional[List[float]] = None) -> Image.Image:
        """
        Combine multiple masks with optional weights.
        
        Args:
            masks: List of masks to combine
            weights: Optional list of weights for each mask
            
        Returns:
            Combined mask as a PIL Image
        """
        if not masks:
            return Image.new("RGB", self.size, (128, 128, 128))
        
        if weights is None:
            weights = [1.0 / len(masks)] * len(masks)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Convert masks to numpy arrays
        mask_arrays = [np.array(mask).astype(np.float32) for mask in masks]
        
        # Weighted average
        combined = np.zeros_like(mask_arrays[0])
        for mask, weight in zip(mask_arrays, weights):
            combined += mask * weight
        
        # Convert back to PIL
        return Image.fromarray(combined.astype(np.uint8))
    
    def random_mask(self) -> Tuple[Image.Image, str]:
        """
        Generate a random mask based on templates.
        
        Returns:
            Tuple of (mask, template_name)
        """
        # Choose a template based on weights
        templates = list(COOKING_REGION_TEMPLATES.keys())
        weights = [COOKING_REGION_TEMPLATES[t]["weight"] for t in templates]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Choose template
        template_name = random.choices(templates, weights=weights, k=1)[0]
        
        # Generate mask
        mask = self.generate_template_mask(template_name)
        
        return mask, template_name


class AdvancedPizzaDiffusionControl:
    """
    Advanced control system for pizza diffusion generation.
    This class extends the basic diffusion generator with specialized
    pizza-specific controls and pipeline optimization.
    """
    
    def __init__(
        self,
        image_size: int = 1024,
        output_dir: str = "data/synthetic",
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        model_type: str = "sdxl",
        use_control_net: bool = False,
        batch_size: int = 4,
        quality_threshold: float = 0.7,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the advanced pizza diffusion control system.
        
        Args:
            output_dir: Directory to save generated images
            model_id: Diffusion model ID or path
            model_type: Type of diffusion model to use
            use_control_net: Whether to use ControlNet for guided generation
            batch_size: Batch size for batch processing
            quality_threshold: Minimum quality threshold for images
            device: Device to run inference on
            config: Additional configuration parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the base diffusion generator
        self.generator = PizzaDiffusionGenerator(
            image_size=image_size,
            model_id=model_id,
            model_type=model_type,
            device=device,
            output_dir=str(output_dir),
            batch_size=batch_size,
            quality_threshold=quality_threshold,
            config=config
        )
        
        # Create the mask generator
        self.mask_generator = CookingControlMask(size=(image_size, image_size))
        
        # Initialize statistics
        self.stats = {
            "generation_count": 0,
            "templates_used": {},
            "generation_time": 0,
            "accepted_count": 0,
            "rejected_count": 0
        }
    
    def generate_with_template(
        self,
        template_name: str,
        count: int = 10,
        stage: str = "combined",
        custom_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images using a specific template.
        
        Args:
            template_name: Name of the template to use
            count: Number of images to generate
            stage: Pizza cooking stage
            custom_prompt: Optional custom prompt
            seed: Optional random seed
            **kwargs: Additional parameters for the generator
            
        Returns:
            List of dictionaries with generated image metadata
        """
        logger.info(f"Generating {count} images with template: {template_name}")
        start_time = time.time()
        
        # Generate the mask
        mask, _ = self.mask_generator.random_mask() if template_name == "random" else (
            self.mask_generator.generate_template_mask(template_name), template_name)
        
        # Create directory for this template if it doesn't exist
        template_dir = self.output_dir / "templates" / template_name
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the mask
        mask_path = template_dir / f"{template_name}_mask.png"
        mask.save(mask_path)
        
        # Create custom prompt if not provided
        if custom_prompt is None:
            template_desc = COOKING_REGION_TEMPLATES.get(template_name, {}).get("description", "")
            custom_prompt = f"{template_desc}, professional food photography, high quality, detailed"
        
        # Generate images
        results = self.generator.generate_images(
            stage=stage,
            num_images=count,
            custom_prompts=[custom_prompt] * count,
            seed=seed,
            prefix=f"template_{template_name}",
            **kwargs
        )
        
        # Apply post-processing if needed
        processed_results = []
        for metadata in results:
            try:
                # Load the generated image
                image_path = metadata["path"]
                image = Image.open(image_path).convert("RGB")
                
                # Apply template-specific post-processing
                processed_image = self._apply_template_processing(image, template_name, mask)
                
                # Save the processed image
                processed_path = Path(image_path).with_stem(f"{Path(image_path).stem}_processed")
                processed_image.save(processed_path)
                
                # Update metadata
                processed_metadata = metadata.copy()
                processed_metadata["original_path"] = image_path
                processed_metadata["path"] = str(processed_path)
                processed_metadata["template"] = template_name
                processed_metadata["post_processed"] = True
                
                # Save updated metadata
                metadata_path = processed_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(processed_metadata, f, indent=2)
                
                processed_results.append(processed_metadata)
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
        
        # Update statistics
        self.stats["generation_count"] += count
        self.stats["templates_used"][template_name] = self.stats["templates_used"].get(template_name, 0) + len(processed_results)
        self.stats["generation_time"] += time.time() - start_time
        self.stats["accepted_count"] += len(processed_results)
        self.stats["rejected_count"] += count - len(processed_results)
        
        logger.info(f"Generated {len(processed_results)} images with template {template_name} in {time.time() - start_time:.2f}s")
        
        return processed_results
    
    def _apply_template_processing(self, image: Image.Image, template_name: str, mask: Image.Image) -> Image.Image:
        """
        Apply template-specific post-processing to an image.
        
        Args:
            image: PIL Image to process
            template_name: Name of the template
            mask: Template mask
            
        Returns:
            Processed PIL Image
        """
        # Resize mask to match image if needed
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        # We can perform various manipulations based on the mask
        # For simplicity, we'll do a basic blend
        processed = image.copy()
        
        # For the "random_spots" template, we can add some burning effects
        if template_name == "random_spots":
            # Create a burning effect overlay
            burn_overlay = Image.new("RGB", image.size, (50, 20, 10))
            # Use the mask as an alpha channel for the overlay (invert it)
            mask_alpha = ImageOps.invert(ImageOps.grayscale(mask))
            processed = Image.composite(burn_overlay, processed, mask_alpha)
            
        # For edge burn, darken the edges
        elif template_name == "edge_burn":
            # Create a burning effect overlay
            burn_overlay = Image.new("RGB", image.size, (30, 10, 0))
            # Use mask as alpha
            mask_alpha = ImageOps.grayscale(mask)
            # Threshold to make edges black
            mask_alpha = mask_alpha.point(lambda p: 255 if p < 100 else 0)
            processed = Image.composite(burn_overlay, processed, mask_alpha)
            
        # For half burn, add a gradient
        elif template_name == "half_burn":
            # Convert mask to grayscale
            mask_gray = ImageOps.grayscale(mask)
            # Create a burning effect overlay
            burn_overlay = Image.new("RGB", image.size, (20, 5, 0))
            # Use threshold on mask
            mask_alpha = mask_gray.point(lambda p: 255 if p < 100 else 0)
            processed = Image.composite(burn_overlay, processed, mask_alpha)
            
        # Default processing for other templates
        else:
            # Convert mask to grayscale
            mask_gray = ImageOps.grayscale(mask)
            # Adjust brightness based on mask
            brightness_factor = 0.7
            processed = ImageChops.multiply(processed, Image.blend(
                Image.new("RGB", image.size, (255, 255, 255)),
                Image.new("RGB", image.size, (int(255*brightness_factor), int(255*brightness_factor), int(255*brightness_factor))),
                mask_gray.point(lambda p: (255-p)/255.0)
            ))
        
        return processed
    
    def generate_balanced_dataset(
        self,
        total_count: int = 100,
        template_weights: Optional[Dict[str, float]] = None,
        stage_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a balanced dataset with different templates and stages.
        
        Args:
            total_count: Total number of images to generate
            template_weights: Optional custom weights for templates
            stage_weights: Optional custom weights for stages
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary with results grouped by template and stage
        """
        logger.info(f"Generating balanced dataset with {total_count} images")
        
        # Set up template weights
        if template_weights is None:
            template_weights = {name: data["weight"] for name, data in COOKING_REGION_TEMPLATES.items()}
        
        templates = list(template_weights.keys())
        template_values = list(template_weights.values())
        template_total = sum(template_values)
        template_probs = [v / template_total for v in template_values]
        
        # Set up stage weights
        if stage_weights is None:
            # Default to combined and mixed stages for templates
            stage_weights = {
                "combined": 0.5,
                "mixed": 0.3,
                "burnt": 0.1,
                "basic": 0.1
            }
        
        stages = list(stage_weights.keys())
        stage_values = list(stage_weights.values())
        stage_total = sum(stage_values)
        stage_probs = [v / stage_total for v in stage_values]
        
        # Calculate counts for each template and stage
        template_counts = {}
        for template, prob in zip(templates, template_probs):
            template_counts[template] = max(1, int(total_count * prob))
        
        # Adjust counts to match total
        total_calculated = sum(template_counts.values())
        if total_calculated != total_count:
            diff = total_count - total_calculated
            # Add/subtract from the templates with the highest weights
            sorted_templates = sorted(templates, key=lambda t: template_weights[t], reverse=True)
            for i in range(abs(diff)):
                template = sorted_templates[i % len(sorted_templates)]
                template_counts[template] += 1 if diff > 0 else -1
        
        # Generate images
        all_results = {}
        
        for template, count in template_counts.items():
            logger.info(f"Generating {count} images with template: {template}")
            
            # Randomly assign stages based on weights
            template_stages = random.choices(stages, weights=stage_probs, k=count)
            stage_counts = {stage: template_stages.count(stage) for stage in stages if template_stages.count(stage) > 0}
            
            template_results = {}
            for stage, stage_count in stage_counts.items():
                if stage_count > 0:
                    results = self.generate_with_template(
                        template_name=template,
                        count=stage_count,
                        stage=stage,
                        **kwargs
                    )
                    template_results[stage] = results
            
            all_results[template] = template_results
        
        # Generate summary report
        summary = {
            "total_generated": sum(len(stage_results) for template_results in all_results.values() 
                                   for stage_results in template_results.values()),
            "by_template": {template: sum(len(stage_results) for stage_results in template_results.values()) 
                             for template, template_results in all_results.items()},
            "by_stage": {stage: sum(len(template_results.get(stage, [])) for template_results in all_results.values())
                          for stage in stages}
        }
        
        logger.info(f"Dataset generation complete. Summary: {summary}")
        
        # Save summary report
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return all_results
    
    def augment_existing_dataset(
        self,
        input_dir: str,
        multiplier: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Augment an existing dataset using diffusion models.
        
        Args:
            input_dir: Directory with existing images
            multiplier: Number of variations to generate per image
            **kwargs: Additional parameters for generation
            
        Returns:
            List of dictionaries with generated image metadata
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        logger.info(f"Augmenting existing dataset in {input_dir}")
        
        # Find images
        image_files = []
        for ext in [".png", ".jpg", ".jpeg"]:
            image_files.extend(input_dir.glob(f"**/*{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} existing images to augment")
        
        all_results = []
        
        for img_path in image_files:
            try:
                # Try to determine the stage from the path
                path_str = str(img_path)
                stage = None
                for s in PIZZA_STAGES.keys():
                    if s in path_str:
                        stage = s
                        break
                
                if stage is None:
                    stage = "combined"  # Default if we can't determine
                
                # Use the generator's refine_images function
                results = self.generator.refine_images(
                    input_dir=str(img_path.parent),
                    stage=stage,
                    strength=0.4,  # Medium strength to keep some characteristics
                    **kwargs
                )
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error augmenting image {img_path}: {e}")
        
        logger.info(f"Augmentation complete. Generated {len(all_results)} new images")
        return all_results
    
    def evaluate_dataset_quality(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated dataset.
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            Dictionary with quality metrics
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        logger.info(f"Evaluating dataset quality in {dataset_dir}")
        
        # Find images
        image_files = []
        for ext in [".png", ".jpg", ".jpeg"]:
            image_files.extend(dataset_dir.glob(f"**/*{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in {dataset_dir}")
            return {"error": "No images found"}
        
        # Initialize metrics
        metrics = {
            "total_images": len(image_files),
            "stages": {},
            "average_quality": 0.0,
            "size_distribution": {},
            "histogram_stats": {}
        }
        
        # Analyze images
        quality_scores = []
        histograms = []
        
        for img_path in image_files:
            try:
                # Load image
                img = Image.open(img_path).convert("RGB")
                
                # Get stage from path
                path_str = str(img_path)
                stage = "unknown"
                for s in PIZZA_STAGES.keys():
                    if s in path_str:
                        stage = s
                        break
                
                if stage not in metrics["stages"]:
                    metrics["stages"][stage] = 0
                metrics["stages"][stage] += 1
                
                # Assess quality
                quality = self.generator._evaluate_image_quality(img)
                quality_scores.append(quality)
                
                # Calculate histogram
                hist = img.histogram()
                histograms.append(hist)
                
                # Record image size
                size_key = f"{img.width}x{img.height}"
                if size_key not in metrics["size_distribution"]:
                    metrics["size_distribution"][size_key] = 0
                metrics["size_distribution"][size_key] += 1
                
            except Exception as e:
                logger.error(f"Error analyzing image {img_path}: {e}")
        
        # Calculate average quality
        if quality_scores:
            metrics["average_quality"] = sum(quality_scores) / len(quality_scores)
            metrics["min_quality"] = min(quality_scores)
            metrics["max_quality"] = max(quality_scores)
        
        # Calculate histogram statistics
        if histograms:
            avg_hist = np.mean(histograms, axis=0)
            std_hist = np.std(histograms, axis=0)
            
            # Simplify by using aggregate stats
            metrics["histogram_stats"] = {
                "red_mean": float(np.mean(avg_hist[:256])),
                "green_mean": float(np.mean(avg_hist[256:512])),
                "blue_mean": float(np.mean(avg_hist[512:])),
                "red_std": float(np.std(avg_hist[:256])),
                "green_std": float(np.std(avg_hist[256:512])),
                "blue_std": float(np.std(avg_hist[512:]))
            }
        
        # Save metrics
        metrics_path = dataset_dir / "quality_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Dataset quality evaluation complete. Metrics saved to {metrics_path}")
        return metrics
    
    def cleanup(self):
        """Clean up resources"""
        self.generator.cleanup()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Advanced Pizza Diffusion Control")
    parser.add_argument("--output_dir", type=str, default="data/synthetic/controlled",
                        help="Directory to save generated images")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Diffusion model ID or path")
    parser.add_argument("--model_type", type=str, default="sdxl", choices=["sdxl", "kandinsky", "custom"],
                        help="Type of diffusion model to use")
    parser.add_argument("--template", type=str, default=None, choices=list(COOKING_REGION_TEMPLATES.keys()) + ["all", "random"],
                        help="Template to use for generation")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of images to generate")
    parser.add_argument("--stage", type=str, default="combined", choices=list(PIZZA_STAGES.keys()),
                        help="Pizza cooking stage to generate")
    parser.add_argument("--augment", type=str, default=None,
                        help="Augment existing dataset in the specified directory")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Evaluate dataset quality in the specified directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--balanced", action="store_true",
                        help="Generate a balanced dataset with all templates")
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Create controller
    controller = AdvancedPizzaDiffusionControl(
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_type=args.model_type,
        batch_size=args.batch_size
    )
    
    try:
        if args.evaluate:
            # Evaluate dataset quality
            metrics = controller.evaluate_dataset_quality(args.evaluate)
            print(f"Dataset quality metrics: {metrics}")
            
        elif args.augment:
            # Augment existing dataset
            results = controller.augment_existing_dataset(args.augment)
            print(f"Augmented dataset with {len(results)} new images")
            
        elif args.balanced:
            # Generate balanced dataset
            results = controller.generate_balanced_dataset(
                total_count=args.count,
                seed=args.seed
            )
            total_generated = sum(len(stage_results) for template_results in results.values() 
                                   for stage_results in template_results.values())
            print(f"Generated balanced dataset with {total_generated} images")
            
        elif args.template == "all":
            # Generate with all templates
            all_results = []
            for template in COOKING_REGION_TEMPLATES.keys():
                count = max(1, int(args.count * COOKING_REGION_TEMPLATES[template]["weight"]))
                results = controller.generate_with_template(
                    template_name=template,
                    count=count,
                    stage=args.stage,
                    seed=args.seed
                )
                all_results.extend(results)
            print(f"Generated {len(all_results)} images with all templates")
            
        else:
            # Generate with specified template
            template = args.template or "random"
            results = controller.generate_with_template(
                template_name=template,
                count=args.count,
                stage=args.stage,
                seed=args.seed
            )
            print(f"Generated {len(results)} images with template {template}")
    
    finally:
        # Clean up
        controller.cleanup()


if __name__ == "__main__":
    main()
