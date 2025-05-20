#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pizza Diffusion Generator

Advanced synthetic data generation for pizza recognition using state-of-the-art
diffusion models. This module provides specialized pipelines for generating
high-quality synthetic pizza images at different cooking stages and can be used 
to augment the training dataset with realistic variations.

This implementation supports:
1. Multiple diffusion models (Stable Diffusion XL, Kandinsky, etc.)
2. Specialized prompting strategies for pizza generation
3. Controlled attributes (cooking stages, toppings, angles)
4. Quality filtering and metadata tracking
5. Resource-efficient batch processing
6. Evaluation metrics and comparison to real data

Author: GitHub Copilot (2025-05-10)
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageStat, ImageOps, ImageFilter

# Import diffusers components
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    DiffusionPipeline,
    KandinskyV22Pipeline,
    KandinskyV22ControlnetPipeline,
    AutoencoderKL,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    ControlNetModel
)
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# Import project utilities if available
try:
    from src.utils.validation import validate_image_quality
    from src.integration.diffusion_data_agent import AgentConfig, DiffusionDataAgent
    PROJECT_UTILS_AVAILABLE = True
except ImportError:
    PROJECT_UTILS_AVAILABLE = False
    print("Warning: Project utilities not found, using standalone mode")

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# Constants and defaults for pizza generation
PIZZA_STAGES = {
    "basic": {
        "prompts": [
            "a raw pizza dough with tomato sauce, unbaked, high quality food photography",
            "unbaked pizza with fresh ingredients, raw dough, top view, studio lighting",
            "raw pizza before baking, with cheese and toppings, professional food photo",
            "fresh prepared pizza ready for the oven, raw dough, detailed texture"
        ],
        "negative_prompts": [
            "burnt, overcooked, black, charred, low quality, blurry"
        ],
        "guidance_scale": 7.5,
        "seed_range": (1000, 5000)
    },
    "burnt": {
        "prompts": [
            "severely burnt pizza with black charred crust, overcooked, high quality food photography",
            "pizza with burnt edges and black bottom, overcooked in oven, detailed texture",
            "charred pizza with black burnt spots, smoke damage, detailed top view",
            "completely burnt pizza with black crust and toppings, ruined food, detailed"
        ],
        "negative_prompts": [
            "raw, unbaked, undercooked, low quality, blurry"
        ],
        "guidance_scale": 7.5,
        "seed_range": (5001, 10000)
    },
    "mixed": {
        "prompts": [
            "partially cooked pizza with both raw and cooked areas, detailed food photography",
            "pizza with unevenly cooked surface, some parts golden, some parts still raw, detailed view",
            "imperfectly baked pizza with varied cooking levels across the surface, detailed texture",
            "pizza with mixed cooking results, some toppings cooked, others still raw, high quality photo"
        ],
        "negative_prompts": [
            "completely burnt, entirely raw, low quality, blurry"
        ],
        "guidance_scale": 8.0,
        "seed_range": (10001, 15000)
    },
    "progression": {
        "prompts": [
            "sequence of pizza cooking from raw to cooked, time-lapse effect, detailed food photography",
            "pizza at different stages of baking, progression from raw to cooked, studio lighting",
            "pizza baking process, showing transitional cooking states, professional detail",
            "multiple images showing pizza cooking progression in oven, detailed textures"
        ],
        "negative_prompts": [
            "completely burnt, low quality, blurry, animated"
        ],
        "guidance_scale": 8.0,
        "seed_range": (15001, 20000)
    },
    "segment": {
        "prompts": [
            "a single slice of pizza, detailed view of the cut piece, professional food photography",
            "one portion of pizza cut from whole, showing crust and toppings, high quality",
            "detailed pizza slice on a plate, showing texture and layers, studio lighting",
            "close-up of an individual pizza segment, showing interior and toppings, gourmet photography"
        ],
        "negative_prompts": [
            "whole pizza, multiple slices, low quality, blurry"
        ],
        "guidance_scale": 7.0,
        "seed_range": (20001, 25000)
    },
    "combined": {
        "prompts": [
            "pizza with multiple cooking levels in different regions, some burnt, some raw, detailed food photo",
            "pizza with combined states - partly burnt, partly undercooked, high quality detailed view",
            "half-burnt, half-raw pizza showing clear contrast between regions, detailed texture",
            "pizza with irregular cooking pattern, some regions charred, others still raw, professional photo"
        ],
        "negative_prompts": [
            "uniform cooking, consistent appearance, low quality, blurry"
        ],
        "guidance_scale": 8.5,
        "seed_range": (25001, 30000)
    }
}

class PizzaDiffusionGenerator:
    """
    Synthetic pizza image generator using state-of-the-art diffusion models.
    Supports various pizza stages and controlled generation with quality filters.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        model_type: str = "sdxl",
        image_size: int = 1024,
        device: Optional[torch.device] = None,
        output_dir: str = "data/synthetic",
        batch_size: int = 4,
        quality_threshold: float = 0.7,
        save_metadata: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pizza diffusion generator with model and configuration.
        
        Args:
            model_id: Diffusion model ID or path to use
            model_type: Type of model to use ('sdxl', 'kandinsky', etc.)
            device: Device to run inference on (defaults to CUDA if available)
            output_dir: Directory to save generated images
            batch_size: Batch size for generation
            quality_threshold: Minimum quality score to accept generated images
            save_metadata: Whether to save image metadata
            config: Additional configuration parameters
        """
        self.model_id = model_id
        self.model_type = model_type.lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        self.save_metadata = save_metadata
        self.config = config or {}
        self.image_size = image_size
        
        # Create output directory and subdirectories for each pizza stage
        self._create_output_directories()
        
        # Initialize statistics tracking
        self.stats = {
            "total_generated": 0,
            "accepted": 0,
            "rejected": 0,
            "generation_time": 0,
            "per_class": {stage: 0 for stage in PIZZA_STAGES.keys()}
        }
        
        # Load model(s) based on configuration
        self._load_models()
        
        logger.info(f"Initialized PizzaDiffusionGenerator using {self.model_type} on {self.device}")
        
    def _create_output_directories(self):
        """Create output directories for each pizza stage"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_dirs = {}
        
        for stage in PIZZA_STAGES.keys():
            stage_dir = self.output_dir / stage
            stage_dir.mkdir(exist_ok=True)
            self.class_dirs[stage] = stage_dir
            
        logger.info(f"Created output directories in {self.output_dir}")
        
    def _load_models(self):
        """Load the appropriate diffusion models based on configuration"""
        try:
            logger.info(f"Loading {self.model_type} model: {self.model_id}")
            
            if self.model_type == "sdxl":
                # Load Stable Diffusion XL pipeline
                self.text2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device.type == "cuda" else None,
                )
                self.text2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.text2img_pipe.scheduler.config)
                self.text2img_pipe.to(self.device)
                
                # If we have enough VRAM, also load the img2img pipeline (optional)
                if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10_000_000_000:  # >10GB VRAM
                    refiner_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
                    self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        refiner_id,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                    )
                    self.img2img_pipe.to(self.device)
                    logger.info(f"Also loaded img2img pipeline: {refiner_id}")
                else:
                    self.img2img_pipe = None
                    logger.info("Skipped loading img2img pipeline (insufficient memory)")
                    
            elif self.model_type == "kandinsky":
                # Load Kandinsky 2.2 pipeline
                self.text2img_pipe = KandinskyV22Pipeline.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-decoder",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
                self.text2img_pipe.to(self.device)
                self.img2img_pipe = None  # Not using img2img for Kandinsky in this implementation
                
            elif self.model_type == "custom":
                # Load a custom pipeline from a local path
                self.text2img_pipe = DiffusionPipeline.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
                self.text2img_pipe.to(self.device)
                self.img2img_pipe = None
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            # Enable memory optimization if on CUDA
            if self.device.type == "cuda":
                self.text2img_pipe.enable_attention_slicing()
                # Enable memory optimizations
                if self.config.get('offload_to_cpu', False):
                    self.text2img_pipe.enable_model_cpu_offload()
                if self.img2img_pipe:
                    self.img2img_pipe.enable_attention_slicing()
                    
            logger.info(f"Successfully loaded {self.model_type} model")
            
        except Exception as e:
            logger.error(f"Error loading diffusion model: {e}")
            raise
        
    def generate_images(self, 
                       stage: str, 
                       num_images: int = 10, 
                       custom_prompts: Optional[List[str]] = None,
                       custom_negative_prompts: Optional[List[str]] = None,
                       guidance_scale: Optional[float] = None,
                       seed: Optional[int] = None,
                       prefix: str = "",
                       **kwargs) -> List[Dict[str, Any]]:
        """
        Generate synthetic pizza images for a given stage.
        
        Args:
            stage: Pizza cooking stage ('basic', 'burnt', etc.)
            num_images: Number of images to generate
            custom_prompts: Optional list of custom prompts to use
            custom_negative_prompts: Optional list of custom negative prompts
            guidance_scale: Optional custom guidance scale
            seed: Optional random seed for reproducibility
            prefix: Optional prefix for saved file names
            **kwargs: Additional parameters for the diffusion pipeline
            
        Returns:
            List of dictionaries with generated image metadata
        """
        if stage not in PIZZA_STAGES:
            raise ValueError(f"Unsupported pizza stage: {stage}. Valid stages: {list(PIZZA_STAGES.keys())}")
        
        stage_config = PIZZA_STAGES[stage]
        prompts = custom_prompts or stage_config["prompts"]
        negative_prompts = custom_negative_prompts or stage_config["negative_prompts"]
        guidance = guidance_scale or stage_config["guidance_scale"]
        
        # Setup random seeds for reproducibility if requested
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Keep track of results and metadata
        results = []
        start_time = time.time()
        
        # Progress tracking
        logger.info(f"Generating {num_images} images for stage: {stage}")
        remaining = num_images
        
        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            
            # Select prompts and seeds for this batch
            batch_prompts = [random.choice(prompts) for _ in range(batch_size)]
            batch_negative_prompts = [random.choice(negative_prompts) for _ in range(batch_size)]
            
            # Generate seeds within the configured range for this stage
            seed_min, seed_max = stage_config["seed_range"]
            batch_seeds = [random.randint(seed_min, seed_max) for _ in range(batch_size)]
            
            try:
                # Generate images
                logger.info(f"Generating batch of {batch_size} images")
                generator = [torch.Generator(device=self.device).manual_seed(s) for s in batch_seeds]
                
                # Run the diffusion pipeline
                if self.model_type == "sdxl":
                    outputs = self.text2img_pipe(
                        prompt=batch_prompts,
                        negative_prompt=batch_negative_prompts,
                        guidance_scale=guidance,
                        num_inference_steps=30,
                        generator=generator,
                        width=512,
                        height=512,
                        **kwargs
                    )
                elif self.model_type == "kandinsky":
                    outputs = self.text2img_pipe(
                        prompt=batch_prompts,
                        negative_prompt=batch_negative_prompts,
                        guidance_scale=guidance,
                        num_inference_steps=30,
                        generator=generator,
                        width=512,
                        height=512,
                        **kwargs
                    )
                else:
                    # Generic handling for other model types
                    outputs = self.text2img_pipe(
                        prompt=batch_prompts,
                        negative_prompt=batch_negative_prompts,
                        guidance_scale=guidance,
                        num_inference_steps=30,
                        generator=generator,
                        **kwargs
                    )
                
                # Process and save each generated image
                for i, image in enumerate(outputs.images):
                    # Evaluate image quality
                    quality_score = self._evaluate_image_quality(image)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_id = f"{prefix}_{stage}_{timestamp}_{i}_{batch_seeds[i]}"
                    
                    # Skip images that don't meet quality threshold
                    if quality_score < self.quality_threshold:
                        logger.warning(f"Image {image_id} rejected: quality score {quality_score:.2f} < threshold {self.quality_threshold}")
                        self.stats["rejected"] += 1
                        continue
                    
                    # Save the image
                    output_path = self.class_dirs[stage] / f"{image_id}.png"
                    image.save(output_path)
                    
                    # Create metadata
                    metadata = {
                        "id": image_id,
                        "path": str(output_path),
                        "stage": stage,
                        "prompt": batch_prompts[i],
                        "negative_prompt": batch_negative_prompts[i],
                        "seed": batch_seeds[i],
                        "guidance_scale": guidance,
                        "quality_score": quality_score,
                        "generation_time": time.time() - start_time,
                        "model": self.model_id,
                        "model_type": self.model_type,
                        "timestamp": timestamp
                    }
                    
                    # Save metadata if requested
                    if self.save_metadata:
                        metadata_path = self.class_dirs[stage] / f"{image_id}.json"
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
                    results.append(metadata)
                    self.stats["accepted"] += 1
                    self.stats["per_class"][stage] += 1
                    
                # Update statistics
                self.stats["total_generated"] += batch_size
                
                # Free up memory
                del outputs
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                remaining -= batch_size
                
            except Exception as e:
                logger.error(f"Error generating batch: {str(e)}")
                # Reduce batch size and try again if possible
                if self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.info(f"Reduced batch size to {self.batch_size}")
                else:
                    # If we can't reduce batch size further, skip this batch
                    logger.error("Cannot reduce batch size further, skipping remaining images")
                    break
        
        # Update final stats
        self.stats["generation_time"] = time.time() - start_time
        logger.info(f"Generated {len(results)}/{num_images} images for stage {stage} " +
                    f"in {self.stats['generation_time']:.1f}s " +
                    f"(accepted: {self.stats['accepted']}, rejected: {self.stats['rejected']})")
        
        return results
    
    def _evaluate_image_quality(self, image) -> float:
        """
        Evaluate the quality of a generated image.
        Returns a normalized quality score between 0.0 and 1.0.
        
        Args:
            image: PIL.Image to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if PROJECT_UTILS_AVAILABLE:
            try:
                # Use the project's built-in image validation if available
                return validate_image_quality(image)
            except:
                pass
        
        # Fallback quality assessment implementation
        try:
            # Convert to numpy if needed
            if isinstance(image, torch.Tensor):
                if image.ndim == 4:
                    image = image[0]  # Take first image if it's a batch
                image = TF.to_pil_image(image)
            
            # Basic image statistics
            stat = ImageStat.Stat(image)
            
            # Check for too dark or too bright images
            mean_brightness = sum(stat.mean) / len(stat.mean)
            if mean_brightness < 20 or mean_brightness > 235:
                return 0.3  # Too dark or too bright
            
            # Check contrast
            contrast = sum(stat.stddev) / len(stat.stddev)
            if contrast < 10:
                return 0.4  # Low contrast
            
            # Check for too much uniformity (potentially failed generation)
            if max(stat.stddev) < 20:
                return 0.5  # Too uniform
            
            # Calculate sharpness score
            sharp_image = image.filter(ImageFilter.FIND_EDGES)
            sharp_stat = ImageStat.Stat(sharp_image)
            sharpness = sum(sharp_stat.mean) / len(sharp_stat.mean)
            
            # Normalize sharpness to 0-1 range
            sharpness_score = min(1.0, max(0.0, sharpness / 30.0))
            
            # Combined score with weights
            quality_score = 0.4 + (0.2 * (contrast / 80)) + (0.4 * sharpness_score)
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error in quality evaluation: {e}")
            return 0.5  # Default middle score on error
    
    def generate_dataset(self, distribution: Dict[str, int], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a complete dataset with the specified distribution of pizza stages.
        
        Args:
            distribution: Dictionary mapping stage names to number of images
            **kwargs: Additional parameters for generate_images
            
        Returns:
            Dictionary mapping stage names to lists of image metadata
        """
        all_results = {}
        total_images = sum(distribution.values())
        
        for stage, count in distribution.items():
            logger.info(f"Generating {count} images for stage {stage} ({count/total_images*100:.1f}% of dataset)")
            results = self.generate_images(stage, count, **kwargs)
            all_results[stage] = results
        
        # Save distribution report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_generated": self.stats["total_generated"],
            "total_accepted": self.stats["accepted"],
            "total_rejected": self.stats["rejected"],
            "rejection_rate": self.stats["rejected"] / max(1, self.stats["total_generated"]),
            "generation_time": self.stats["generation_time"],
            "distribution": {stage: len(results) for stage, results in all_results.items()},
            "per_class": self.stats["per_class"]
        }
        
        report_path = self.output_dir / "generation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Dataset generation complete. Report saved to {report_path}")
        return all_results

    def refine_images(self, input_dir: str, stage: str, strength: float = 0.3, **kwargs) -> List[Dict[str, Any]]:
        """
        Refine existing images using img2img pipeline if available.
        
        Args:
            input_dir: Directory containing images to refine
            stage: Pizza stage to use for prompts
            strength: Strength parameter for img2img (0.0-1.0)
            **kwargs: Additional parameters for img2img pipeline
            
        Returns:
            List of dictionaries with refined image metadata
        """
        if self.img2img_pipe is None:
            logger.error("img2img pipeline not available - refinement not possible")
            return []
        
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all images in the directory
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []
        
        logger.info(f"Refining {len(image_files)} images for stage: {stage}")
        stage_config = PIZZA_STAGES[stage]
        results = []
        
        for img_path in image_files:
            try:
                # Load the image
                init_image = Image.open(img_path).convert("RGB")
                init_image = init_image.resize((512, 512))
                
                # Select prompt and seed
                prompt = random.choice(stage_config["prompts"])
                negative_prompt = random.choice(stage_config["negative_prompts"])
                seed = random.randint(*stage_config["seed_range"])
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                # Run img2img pipeline
                output = self.img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=stage_config["guidance_scale"],
                    generator=generator,
                    num_inference_steps=30,
                    **kwargs
                )
                
                refined_image = output.images[0]
                quality_score = self._evaluate_image_quality(refined_image)
                
                # Skip if below threshold
                if quality_score < self.quality_threshold:
                    continue
                
                # Save refined image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_id = f"refined_{stage}_{timestamp}_{seed}"
                output_path = self.class_dirs[stage] / f"{image_id}.png"
                refined_image.save(output_path)
                
                # Create metadata
                metadata = {
                    "id": image_id,
                    "path": str(output_path),
                    "stage": stage,
                    "source_image": str(img_path),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "strength": strength,
                    "quality_score": quality_score,
                    "timestamp": timestamp
                }
                
                if self.save_metadata:
                    metadata_path = self.class_dirs[stage] / f"{image_id}.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                results.append(metadata)
                
                # Free memory
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error refining image {img_path}: {str(e)}")
        
        logger.info(f"Refined {len(results)} images for stage {stage}")
        return results

    def extract_features(self, image_dir: str, output_file: str = "features.npz"):
        """
        Extract features from images using the diffusion model's encoder.
        Useful for comparing synthetic vs. real data distributions.
        
        Args:
            image_dir: Directory containing images to extract features from
            output_file: File to save extracted features to
            
        Returns:
            Path to saved features file
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Find all images recursively
        image_files = []
        for ext in [".png", ".jpg", ".jpeg"]:
            image_files.extend(image_dir.glob(f"**/*{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in {image_dir}")
            return None
        
        logger.info(f"Extracting features from {len(image_files)} images")
        
        # We'll use a simpler approach if the UNet model is available
        features = []
        labels = []
        
        # Basic preprocessing transform
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        for img_path in image_files:
            try:
                # Extract stage from path
                path_parts = str(img_path).split("/")
                stage = None
                for part in path_parts:
                    if part in PIZZA_STAGES:
                        stage = part
                        break
                
                if stage is None:
                    stage = "unknown"
                
                # Load and preprocess the image
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Extract simple features (using mean and std across spatial dimensions)
                with torch.no_grad():
                    # Simple statistical features if we can't access model internals
                    mean_features = tensor.mean(dim=[2, 3]).squeeze().cpu().numpy()
                    std_features = tensor.std(dim=[2, 3]).squeeze().cpu().numpy()
                    combined_features = np.concatenate([mean_features, std_features])
                    
                features.append(combined_features)
                labels.append(stage)
                
            except Exception as e:
                logger.error(f"Error extracting features from {img_path}: {str(e)}")
        
        # Save features and labels
        if features:
            output_path = Path(output_file)
            np.savez(
                output_path, 
                features=np.array(features), 
                labels=np.array(labels),
                paths=[str(p) for p in image_files]
            )
            logger.info(f"Saved features for {len(features)} images to {output_path}")
            return output_path
        
        return None

    def compare_distributions(self, real_features_file: str, synthetic_features_file: str):
        """
        Compare feature distributions between real and synthetic datasets.
        
        Args:
            real_features_file: Path to features file for real images
            synthetic_features_file: Path to features file for synthetic images
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            real_data = np.load(real_features_file)
            synth_data = np.load(synthetic_features_file)
            
            real_features = real_data["features"]
            real_labels = real_data["labels"]
            
            synth_features = synth_data["features"]
            synth_labels = synth_data["labels"]
            
            # Basic comparison metrics
            metrics = {}
            
            # Overall feature mean and std difference
            metrics["mean_diff"] = np.linalg.norm(
                np.mean(real_features, axis=0) - np.mean(synth_features, axis=0)
            )
            
            metrics["std_diff"] = np.linalg.norm(
                np.std(real_features, axis=0) - np.std(synth_features, axis=0)
            )
            
            # Per-class statistics if we have class labels
            unique_labels = set(np.concatenate([real_labels, synth_labels]))
            metrics["per_class"] = {}
            
            for label in unique_labels:
                real_class = real_features[real_labels == label]
                synth_class = synth_features[synth_labels == label]
                
                # Skip if we don't have enough samples
                if len(real_class) < 5 or len(synth_class) < 5:
                    continue
                
                metrics["per_class"][label] = {
                    "mean_diff": np.linalg.norm(
                        np.mean(real_class, axis=0) - np.mean(synth_class, axis=0)
                    ),
                    "std_diff": np.linalg.norm(
                        np.std(real_class, axis=0) - np.std(synth_class, axis=0)
                    ),
                    "real_count": len(real_class),
                    "synth_count": len(synth_class)
                }
            
            # Print summary
            logger.info(f"Distribution comparison metrics:")
            logger.info(f"  Overall mean difference: {metrics['mean_diff']:.4f}")
            logger.info(f"  Overall std difference: {metrics['std_diff']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error comparing distributions: {str(e)}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources and free memory"""
        try:
            # Clear pipeline from GPU memory
            del self.text2img_pipe
            if hasattr(self, 'img2img_pipe') and self.img2img_pipe is not None:
                del self.img2img_pipe
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Successfully cleaned up resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Pizza Diffusion Generator")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", 
                        help="Directory to save generated images")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Diffusion model ID or path")
    parser.add_argument("--model_type", type=str, default="sdxl", choices=["sdxl", "kandinsky", "custom"],
                        help="Type of diffusion model to use")
    parser.add_argument("--basic", type=int, default=50,
                        help="Number of 'basic' pizza images to generate")
    parser.add_argument("--burnt", type=int, default=50,
                        help="Number of 'burnt' pizza images to generate")
    parser.add_argument("--mixed", type=int, default=50,
                        help="Number of 'mixed' pizza images to generate")
    parser.add_argument("--progression", type=int, default=50,
                        help="Number of 'progression' pizza images to generate")
    parser.add_argument("--segment", type=int, default=50,
                        help="Number of 'segment' pizza images to generate")
    parser.add_argument("--combined", type=int, default=50,
                        help="Number of 'combined' pizza images to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--quality_threshold", type=float, default=0.65,
                        help="Minimum quality threshold (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if CUDA is available")
    parser.add_argument("--extract_features", action="store_true",
                        help="Extract features after generation for analysis")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create generator
    generator = PizzaDiffusionGenerator(
        model_id=args.model_id,
        model_type=args.model_type,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold
    )
    
    # Generate dataset with distribution
    distribution = {
        "basic": args.basic,
        "burnt": args.burnt,
        "mixed": args.mixed, 
        "progression": args.progression,
        "segment": args.segment,
        "combined": args.combined
    }
    
    # Filter out classes with zero count
    distribution = {k: v for k, v in distribution.items() if v > 0}
    
    try:
        # Generate the dataset
        results = generator.generate_dataset(distribution, seed=args.seed)
        
        # Extract features if requested
        if args.extract_features:
            features_file = generator.extract_features(args.output_dir)
            logger.info(f"Features extracted to {features_file}")
            
            # If we have a 'real' dataset, compare distributions
            real_dir = Path("data/processed")
            if real_dir.exists():
                real_features = generator.extract_features(real_dir, "real_features.npz")
                if real_features:
                    metrics = generator.compare_distributions(real_features, features_file)
                    
                    # Save metrics to file
                    with open(Path(args.output_dir) / "distribution_metrics.json", 'w') as f:
                        json.dump(metrics, f, indent=2)
    finally:
        # Clean up resources
        generator.cleanup()
    
    logger.info("Pizza diffusion generation complete!")


if __name__ == "__main__":
    main()
