#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pizza Diffusion Integration Script

This script integrates the DiffusionDataAgent with the existing pizza augmentation
system to improve reliability and quality of synthetic pizza data generation.
"""

import os
import sys
import torch
import random
import argparse
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageStat
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Import the agent
from diffusion_data_agent import DiffusionDataAgent, AgentConfig

# Try to import pizza augmentation modules
try:
    # Import from enhanced augmentation (first priority)
    from enhanced_pizza_augmentation import (
        EnhancedPizzaBurningEffect, 
        EnhancedOvenEffect,
        apply_burning_augmentation,
        apply_mixed_augmentation,
        apply_progression_augmentation
    )
    USING_ENHANCED = True
except ImportError:
    try:
        # Fall back to optimized augmentation - use proper module name
        from optimized_pizza_augmentation import (
            PizzaBurningEffect,
            SimpleOvenEffect,
            pizza_burning_augmentation,
            pizza_basic_augmentation
        )
        USING_ENHANCED = False
    except ImportError:
        # Final fallback to test module
        from test import (
            PizzaBurningEffect,
            SimpleOvenEffect,
            pizza_burning_augmentation_generator,
            pizza_basic_augmentation_generator
        )
        USING_ENHANCED = False

# Default agent configuration specifically for pizza data
PIZZA_AGENT_CONFIG = {
    "batch_size": 16,
    "min_disk_space_gb": 5,
    "quality_threshold": 0.65,  # Slightly more lenient for pizza generation
    "image_size": (256, 256),
    "save_format": "jpg",
    "save_quality": 90,
    "max_gpu_memory_usage": 0.9,
    "min_brightness": 15,  # Allow darker images for burned pizzas
    "max_brightness": 240,
    "min_contrast": 0.08,  # Allow lower contrast
    "min_file_size_kb": 5,
    "blur_threshold": 50,
    "log_level": "INFO",
    "metadata_schema": {
        "burn_effect_type": "",     # edge, spot, streak, random
        "burn_intensity": 0.0,      # 0.0-1.0 
        "oven_effects_applied": [], # list of effects used
        "augmentation_type": "",    # basic, burning, mixed, progression
        "progression_step": 0,      # if using progression, which step
        "source_image": "",         # original file used for augmentation
        "generation_timestamp": "",
        "quality_score": 0.0,
        "batch_id": ""
    }
}

class PizzaDiffusionIntegration:
    """
    Integration between the Diffusion Data Agent and pizza augmentation system
    """
    
    def __init__(self, agent_config=None):
        """Initialize the integration with a configuration"""
        # Initialize the agent
        config = agent_config or AgentConfig(PIZZA_AGENT_CONFIG)
        self.agent = DiffusionDataAgent(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize effect modules
        self._init_effect_modules()
    
    def _init_effect_modules(self):
        """Initialize pizza effect modules based on available implementations"""
        if USING_ENHANCED:
            self.burning_effects = [
                # Light burning
                EnhancedPizzaBurningEffect(
                    burn_intensity_min=0.1, 
                    burn_intensity_max=0.3,
                    burn_pattern='random'
                ).to(self.device),
                
                # Medium edge burning
                EnhancedPizzaBurningEffect(
                    burn_intensity_min=0.3, 
                    burn_intensity_max=0.6,
                    burn_pattern='edge'
                ).to(self.device),
                
                # Heavy spot burning
                EnhancedPizzaBurningEffect(
                    burn_intensity_min=0.5, 
                    burn_intensity_max=0.9,
                    burn_pattern='spot'
                ).to(self.device),
                
                # Streak burning
                EnhancedPizzaBurningEffect(
                    burn_intensity_min=0.3, 
                    burn_intensity_max=0.7,
                    burn_pattern='streak'
                ).to(self.device)
            ]
            
            self.oven_effect = EnhancedOvenEffect().to(self.device)
        else:
            # Fallback to simpler effects
            self.burning_effects = [
                PizzaBurningEffect(
                    burn_intensity_min=0.2,
                    burn_intensity_max=0.8
                ).to(self.device)
            ]
            
            self.oven_effect = SimpleOvenEffect().to(self.device)
    
    def generate_pizza_batch(self, batch_size=16, augmentation_type="burning", **kwargs):
        """
        Generate a batch of pizza images with the specified augmentation type
        
        Args:
            batch_size: Number of images to generate
            augmentation_type: Type of augmentation (basic, burning, mixed, progression)
            **kwargs: Additional parameters for specific generators
            
        Returns:
            List of image tensors and list of metadata
        """
        self.agent.logger.info(f"Generating {batch_size} images with {augmentation_type} augmentation")
        
        # Initialize results
        images = []
        metadata = []
        
        try:
            # Generate based on type
            if augmentation_type == "basic":
                images, metadata = self._generate_basic_batch(batch_size, **kwargs)
            elif augmentation_type == "burning":
                images, metadata = self._generate_burning_batch(batch_size, **kwargs)
            elif augmentation_type == "mixed":
                images, metadata = self._generate_mixed_batch(batch_size, **kwargs)
            elif augmentation_type == "progression":
                images, metadata = self._generate_progression_batch(batch_size, **kwargs)
            else:
                self.agent.logger.error(f"Unknown augmentation type: {augmentation_type}")
                return [], []
            
            return images, metadata
            
        except Exception as e:
            self.agent.logger.error(f"Error generating {augmentation_type} batch: {str(e)}")
            return [], []
    
    def _generate_basic_batch(self, batch_size, **kwargs):
        """Generate basic-augmented pizza images"""
        images = []
        metadata = []
        
        with torch.no_grad():
            # Load source images
            source_paths = kwargs.get('image_paths', [])
            if not source_paths:
                self.agent.logger.error("No source images provided")
                return [], []
            
            # Basic transformations
            transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(180)], p=0.7),
                transforms.RandomApply([transforms.RandomResizedCrop(
                    self.agent.config.image_size, 
                    scale=(0.8, 1.0), 
                    ratio=(0.95, 1.05),
                    antialias=True
                )], p=0.8),
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.1
                )], p=0.7),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
            
            # Use either dataset approach or direct loading
            if hasattr(self, 'PizzaAugmentationDataset'):
                # Use dataset if available
                dataset = self.PizzaAugmentationDataset(source_paths, transform=transform)
                indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
                
                for idx in indices:
                    img = dataset[idx]
                    if isinstance(img, tuple):
                        img = img[0]  # Handle case where label is returned
                    
                    images.append(img)
                    
                    # Create metadata
                    meta = self.agent.config.metadata_schema.copy()
                    meta["augmentation_type"] = "basic"
                    meta["source_image"] = str(source_paths[idx % len(source_paths)])
                    meta["generation_timestamp"] = datetime.now().isoformat()
                    metadata.append(meta)
            else:
                # Direct loading fallback
                selected_paths = random.sample(source_paths, min(batch_size, len(source_paths)))
                for path in selected_paths:
                    try:
                        with Image.open(path) as img:
                            img = transform(img)
                            images.append(img)
                            
                            # Create metadata
                            meta = self.agent.config.metadata_schema.copy()
                            meta["augmentation_type"] = "basic"
                            meta["source_image"] = str(path)
                            meta["generation_timestamp"] = datetime.now().isoformat()
                            metadata.append(meta)
                    except Exception as e:
                        self.agent.logger.error(f"Error processing {path}: {str(e)}")
        
        return images, metadata
    
    def _generate_burning_batch(self, batch_size, **kwargs):
        """Generate burning-augmented pizza images"""
        images = []
        metadata = []
        
        with torch.no_grad():
            # Load source images
            source_paths = kwargs.get('image_paths', [])
            if not source_paths:
                self.agent.logger.error("No source images provided")
                return [], []
            
            # Base transformations (before burning effects)
            transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
                transforms.RandomApply([transforms.RandomResizedCrop(
                    self.agent.config.image_size, 
                    scale=(0.8, 1.0), 
                    ratio=(0.95, 1.05),
                    antialias=True
                )], p=0.5),
                transforms.ToTensor(),
            ])
            
            # Direct loading approach
            selected_indices = random.sample(range(len(source_paths)), 
                                       min(batch_size, len(source_paths)))
            
            for idx in selected_indices:
                try:
                    path = source_paths[idx]
                    
                    # Load and transform image
                    with Image.open(path) as img:
                        img_tensor = transform(img).to(self.device)
                    
                    # Select random burning effect
                    burn_effect = random.choice(self.burning_effects)
                    
                    # Apply burning effect
                    burned_img = burn_effect(img_tensor)
                    
                    # Optionally apply oven effect
                    apply_oven = random.random() < 0.6
                    if apply_oven:
                        burned_img = self.oven_effect(burned_img)
                    
                    # Create metadata
                    meta = self.agent.config.metadata_schema.copy()
                    meta["augmentation_type"] = "burning"
                    meta["source_image"] = str(path)
                    meta["generation_timestamp"] = datetime.now().isoformat()
                    
                    # Add effect-specific metadata
                    if hasattr(burn_effect, 'burn_pattern'):
                        meta["burn_effect_type"] = burn_effect.burn_pattern
                    if hasattr(burn_effect, 'burn_intensity_min') and hasattr(burn_effect, 'burn_intensity_max'):
                        meta["burn_intensity"] = random.uniform(
                            burn_effect.burn_intensity_min, 
                            burn_effect.burn_intensity_max
                        )
                    
                    meta["oven_effects_applied"] = ["basic"] if apply_oven else []
                    
                    # Add to results
                    images.append(burned_img.cpu())
                    metadata.append(meta)
                    
                except Exception as e:
                    self.agent.logger.error(f"Error generating burned image from {path}: {str(e)}")
        
        return images, metadata
    
    def _generate_mixed_batch(self, batch_size, **kwargs):
        """Generate mixed (MixUp/CutMix) pizza images"""
        images = []
        metadata = []
        alpha = kwargs.get('alpha', 0.4)  # MixUp alpha parameter
        
        with torch.no_grad():
            # Load source images
            source_paths = kwargs.get('image_paths', [])
            if not source_paths or len(source_paths) < 2:
                self.agent.logger.error("Insufficient source images for mixing")
                return [], []
            
            # Base transformations before mixing
            transform = transforms.Compose([
                transforms.Resize(self.agent.config.image_size, antialias=True),
                transforms.ToTensor(),
            ])
            
            # Generate pairs for mixing
            num_pairs = batch_size
            
            for _ in range(num_pairs):
                try:
                    # Select two random images
                    path1, path2 = random.sample(source_paths, 2)
                    
                    # Load and transform images
                    with Image.open(path1) as img1, Image.open(path2) as img2:
                        img1_tensor = transform(img1).to(self.device)
                        img2_tensor = transform(img2).to(self.device)
                    
                    # Apply either MixUp or CutMix
                    if random.random() < 0.5:
                        # MixUp
                        lam = random.betavariate(alpha, alpha)
                        mixed_img = lam * img1_tensor + (1 - lam) * img2_tensor
                        mix_type = "mixup"
                    else:
                        # CutMix
                        mixed_img = img1_tensor.clone()
                        
                        # Generate random box
                        h, w = mixed_img.shape[1:]
                        cut_ratio = random.random() * 0.5 + 0.1  # Between 0.1 and 0.6
                        
                        # Calculate box dimensions
                        cut_h = int(h * cut_ratio)
                        cut_w = int(w * cut_ratio)
                        
                        # Random box position
                        cx = random.randint(0, w - cut_w)
                        cy = random.randint(0, h - cut_h)
                        
                        # Apply cutmix by replacing the box
                        mixed_img[:, cy:cy+cut_h, cx:cx+cut_w] = img2_tensor[:, cy:cy+cut_h, cx:cx+cut_w]
                        mix_type = "cutmix"
                    
                    # Create metadata
                    meta = self.agent.config.metadata_schema.copy()
                    meta["augmentation_type"] = "mixed"
                    meta["mix_type"] = mix_type
                    meta["source_image"] = f"{path1}+{path2}"
                    meta["generation_timestamp"] = datetime.now().isoformat()
                    
                    # Add to results
                    images.append(mixed_img.cpu())
                    metadata.append(meta)
                    
                except Exception as e:
                    self.agent.logger.error(f"Error generating mixed image: {str(e)}")
        
        return images, metadata
    
    def _generate_progression_batch(self, batch_size, **kwargs):
        """Generate progressive burning images (multiple burn levels per image)"""
        images = []
        metadata = []
        num_steps = kwargs.get('num_steps', 5)
        
        with torch.no_grad():
            # Load source images
            source_paths = kwargs.get('image_paths', [])
            if not source_paths:
                self.agent.logger.error("No source images provided")
                return [], []
            
            # Base transformations before progression
            transform = transforms.Compose([
                transforms.Resize(self.agent.config.image_size, antialias=True),
                transforms.ToTensor(),
            ])
            
            # Number of original images to process
            num_originals = batch_size // (num_steps + 1)  # +1 for original
            if num_originals < 1:
                num_originals = 1
                
            # Select images to process
            selected_paths = random.sample(source_paths, min(num_originals, len(source_paths)))
            
            for path in selected_paths:
                try:
                    # Load and transform image
                    with Image.open(path) as img:
                        original_tensor = transform(img).to(self.device)
                    
                    # Add original image to results
                    images.append(original_tensor.cpu())
                    meta_orig = self.agent.config.metadata_schema.copy()
                    meta_orig["augmentation_type"] = "progression"
                    meta_orig["progression_step"] = 0
                    meta_orig["source_image"] = str(path)
                    meta_orig["generation_timestamp"] = datetime.now().isoformat()
                    metadata.append(meta_orig)
                    
                    # Select an effect
                    burn_effect = random.choice(self.burning_effects)
                    if hasattr(burn_effect, 'burn_pattern'):
                        effect_type = burn_effect.burn_pattern
                    else:
                        effect_type = "default" 
                    
                    # Generate progression steps
                    for step in range(1, num_steps + 1):
                        # Calculate progressive intensity
                        intensity = step / num_steps
                        
                        # Configure burn effect intensity
                        if hasattr(burn_effect, 'burn_intensity_min') and hasattr(burn_effect, 'burn_intensity_max'):
                            # Save original values
                            orig_min = burn_effect.burn_intensity_min
                            orig_max = burn_effect.burn_intensity_max
                            
                            # Set to progressive values
                            burn_effect.burn_intensity_min = intensity * 0.7
                            burn_effect.burn_intensity_max = intensity * 0.9
                            
                            # Apply effect
                            burned_img = burn_effect(original_tensor)
                            
                            # Restore original values
                            burn_effect.burn_intensity_min = orig_min
                            burn_effect.burn_intensity_max = orig_max
                        else:
                            # Simpler fallback if custom intensity settings aren't available
                            burned_img = burn_effect(original_tensor)
                        
                        # Apply oven effect for later steps
                        if step > num_steps // 2 and random.random() < 0.7:
                            burned_img = self.oven_effect(burned_img)
                            oven_applied = True
                        else:
                            oven_applied = False
                        
                        # Create metadata
                        meta = self.agent.config.metadata_schema.copy()
                        meta["augmentation_type"] = "progression"
                        meta["progression_step"] = step
                        meta["burn_effect_type"] = effect_type
                        meta["burn_intensity"] = intensity
                        meta["source_image"] = str(path)
                        meta["generation_timestamp"] = datetime.now().isoformat()
                        meta["oven_effects_applied"] = ["basic"] if oven_applied else []
                        
                        # Add to results
                        images.append(burned_img.cpu())
                        metadata.append(meta)
                    
                except Exception as e:
                    self.agent.logger.error(f"Error generating progression from {path}: {str(e)}")
        
        return images, metadata
    
    def run_full_generation(self, 
                          input_dir, 
                          output_dir, 
                          total_count=1000, 
                          distribution=None,
                          batch_size=16):
        """
        Run a full generation job with multiple augmentation types
        
        Args:
            input_dir: Directory containing source images
            output_dir: Directory to save generated images
            total_count: Total number of images to generate
            distribution: Dictionary with percentages for each type
                          e.g. {"basic": 0.3, "burning": 0.4, "mixed": 0.15, "progression": 0.15}
            batch_size: Size of each generation batch
        """
        # Set default distribution if not provided
        if distribution is None:
            distribution = {
                "basic": 0.3,       # 30% basic augmentations
                "burning": 0.4,      # 40% burning effects
                "mixed": 0.15,       # 15% mixed images
                "progression": 0.15  # 15% progression steps
            }
        
        # Calculate image counts for each type
        counts = {}
        remaining = total_count
        for aug_type, percentage in distribution.items():
            count = int(total_count * percentage)
            counts[aug_type] = count
            remaining -= count
        
        # Assign any remaining images to burning (rounding errors)
        if remaining > 0:
            counts["burning"] += remaining
        
        # Get source image paths
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(list(Path(input_dir).glob(ext)))
        
        if not image_paths:
            self.agent.logger.error(f"No images found in {input_dir}")
            return False
        
        self.agent.logger.info(f"Found {len(image_paths)} source images in {input_dir}")
        self.agent.logger.info(f"Distribution: {counts}")
        
        # Create generation output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save generation config
        config_path = os.path.join(output_dir, "generation_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "total_count": total_count,
                "distribution": distribution,
                "source_dir": str(input_dir),
                "image_count": len(image_paths),
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size
            }, f, indent=2)
        
        # Run generation for each type
        for aug_type, count in counts.items():
            if count <= 0:
                continue
                
            aug_output_dir = os.path.join(output_dir, aug_type)
            os.makedirs(aug_output_dir, exist_ok=True)
            
            self.agent.logger.info(f"Starting {aug_type} generation: {count} images")
            
            # Define generator function that will be called by the agent
            def generator_func(batch_size, **kwargs):
                # Call our batch generator
                return self.generate_pizza_batch(
                    batch_size=batch_size,
                    augmentation_type=aug_type,
                    image_paths=image_paths,
                    **kwargs
                )[0]  # Return only images, metadata handled separately
            
            # Run batch job via agent
            self.agent.run_batch_job(
                generator_function=generator_func,
                output_dir=aug_output_dir,
                total_count=count,
                batch_size=batch_size,
                batch_params={"augmentation_type": aug_type}
            )
        
        # Generate final report
        report_file = os.path.join(output_dir, "full_generation_report.json")
        report = self.agent.generate_report(report_file)
        
        self.agent.logger.info(f"Full generation complete. Report saved to {report_file}")
        return True


# CLI interface
def main():
    parser = argparse.ArgumentParser(description='Pizza Diffusion Integration')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--output-dir', type=str, default='./pizza_diffusion_output', help='Output directory')
    parser.add_argument('--total', type=int, default=1000, help='Total number of images to generate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--config', type=str, help='Path to agent configuration file')
    parser.add_argument('--distribution', type=str, help='JSON string with distribution percentages')
    
    args = parser.parse_args()
    
    # Load agent configuration
    config = None
    if args.config:
        config = AgentConfig.from_file(args.config)
    else:
        config = AgentConfig(PIZZA_AGENT_CONFIG)
    
    # Override batch size if provided
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Parse distribution if provided
    distribution = None
    if args.distribution:
        try:
            distribution = json.loads(args.distribution)
        except json.JSONDecodeError:
            print(f"Error parsing distribution JSON: {args.distribution}")
            return
    
    # Create and run integration
    integration = PizzaDiffusionIntegration(config)
    integration.run_full_generation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        total_count=args.total,
        distribution=distribution,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()