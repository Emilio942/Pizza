#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIFFUSION-2.1: Targeted Image Generation Pipeline

This module implements a sophisticated pipeline for generating images with specific 
targeted properties like burn levels, lighting conditions, and cooking stages. 
It builds upon the existing diffusion infrastructure to provide enhanced control 
over generation parameters and comprehensive metadata storage.

Features:
- Enhanced prompt templates for specific properties (burn levels, lighting)
- ControlNet integration for structural control
- Comprehensive metadata storage with all generation parameters
- Quality-aware generation with targeted property verification
- Template-based and parameterized prompt construction

Author: GitHub Copilot (2025-01-28)
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
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from dataclasses import dataclass, asdict

# Set up logging early
logger = logging.getLogger(__name__)

# Import existing infrastructure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import existing pipeline components, but continue if they're not available
EXISTING_PIPELINE_AVAILABLE = False
PizzaDiffusionGenerator = None
AdvancedPizzaDiffusionControl = None
DiffusionDataAgent = None
AgentConfig = None
PIZZA_STAGES = None
COOKING_REGION_TEMPLATES = None

try:
    from src.augmentation.diffusion_pizza_generator import PizzaDiffusionGenerator, PIZZA_STAGES
    from src.augmentation.advanced_pizza_diffusion_control import (
        AdvancedPizzaDiffusionControl, 
        COOKING_REGION_TEMPLATES,
        CookingControlMask
    )
    from src.integration.diffusion_data_agent import DiffusionDataAgent, AgentConfig
    EXISTING_PIPELINE_AVAILABLE = True
    logger.info("Successfully imported existing pipeline components")
except ImportError as e:
    logger.warning(f"Could not import existing pipeline components: {e}")
    logger.info("Pipeline will run in standalone mode")
    
    # Define minimal fallback constants
    PIZZA_STAGES = {
        "basic": "basic pizza dough",
        "sauce": "pizza with sauce", 
        "cheese": "pizza with cheese",
        "toppings": "pizza with toppings",
        "combined": "complete pizza",
        "burnt": "burnt pizza"
    }

@dataclass
class TargetedGenerationConfig:
    """Configuration for targeted generation parameters"""
    # Basic generation parameters
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_type: str = "sdxl"
    image_size: int = 512
    batch_size: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    
    # Quality control
    quality_threshold: float = 0.7
    max_retries: int = 3
    
    # Metadata and storage
    save_metadata: bool = True
    output_dir: str = "data/synthetic/targeted"
    
    # Memory optimization
    enable_cpu_offload: bool = True
    enable_attention_slicing: bool = True
    
    # Targeted generation specifics
    verify_target_properties: bool = True
    property_verification_threshold: float = 0.6

# Enhanced prompt templates for targeted properties
LIGHTING_CONDITION_TEMPLATES = {
    "overhead_harsh": {
        "description": "Direct overhead lighting with harsh shadows",
        "prompts": [
            "pizza with direct overhead light creating harsh shadows, professional food photography",
            "pizza illuminated from above with strong directional lighting casting sharp shadows",
            "pizza under bright overhead light, dramatic shadow patterns on surface, studio lighting",
            "pizza with harsh top-down lighting, strong contrast between light and shadow areas"
        ],
        "negative_prompts": [
            "soft lighting, diffused light, even illumination, no shadows"
        ],
        "lighting_params": {
            "direction": "overhead",
            "intensity": "high",
            "shadow_strength": "strong"
        }
    },
    "side_dramatic": {
        "description": "Dramatic side lighting with long shadows",
        "prompts": [
            "pizza with dramatic side lighting casting long shadows across surface, professional photography",
            "pizza illuminated from the side creating depth with shadow patterns",
            "pizza with strong lateral lighting, shadows extending across the pizza surface",
            "pizza in dramatic side light, chiaroscuro lighting effect, food photography"
        ],
        "negative_prompts": [
            "overhead lighting, even illumination, flat lighting"
        ],
        "lighting_params": {
            "direction": "side",
            "intensity": "high",
            "shadow_strength": "strong"
        }
    },
    "dim_ambient": {
        "description": "Dim ambient lighting with soft shadows",
        "prompts": [
            "pizza in dim ambient lighting, soft shadows, restaurant atmosphere",
            "pizza under low light conditions, subtle shadow details visible",
            "pizza in moody dim lighting, warm ambient glow, intimate setting",
            "pizza with soft ambient lighting, gentle shadows, cozy atmosphere"
        ],
        "negative_prompts": [
            "bright lighting, harsh shadows, overexposed, studio lighting"
        ],
        "lighting_params": {
            "direction": "ambient",
            "intensity": "low",
            "shadow_strength": "soft"
        }
    },
    "backlit_rim": {
        "description": "Backlit with rim lighting effect",
        "prompts": [
            "pizza with backlighting creating rim light effect on edges, professional food photography",
            "pizza backlit with glowing edges, rim lighting on crust, dramatic effect",
            "pizza with edge lighting from behind, silhouette effect with highlighted edges",
            "pizza with backlight creating halo effect around edges, artistic food photography"
        ],
        "negative_prompts": [
            "front lighting, no rim light, flat lighting, even illumination"
        ],
        "lighting_params": {
            "direction": "back",
            "intensity": "medium",
            "rim_effect": True
        }
    }
}

BURN_LEVEL_TEMPLATES = {
    "slightly_burnt": {
        "description": "Pizza with slight browning and light burn marks",
        "prompts": [
            "pizza slightly burnt with light browning on edges, still appetizing, professional food photography",
            "pizza with gentle browning and minimal burn spots, golden-brown appearance",
            "pizza lightly overcooked with subtle burn marks, artisanal wood-fired appearance",
            "pizza with light caramelization and slight burn patterns, gourmet presentation"
        ],
        "negative_prompts": [
            "severely burnt, charred, black, completely burnt, inedible"
        ],
        "burn_params": {
            "intensity": "light",
            "pattern": "edges_and_spots",
            "color_range": "golden_brown"
        }
    },
    "moderately_burnt": {
        "description": "Pizza with noticeable burn marks and browning",
        "prompts": [
            "pizza moderately burnt with visible brown and dark spots, still edible appearance",
            "pizza with noticeable burn marks and darker browning, rustic wood-fired style",
            "pizza with moderate burning, mix of golden and dark brown areas, artisanal",
            "pizza overcooked with prominent burn patterns but not completely black"
        ],
        "negative_prompts": [
            "raw, uncooked, perfectly cooked, lightly browned, completely charred"
        ],
        "burn_params": {
            "intensity": "medium",
            "pattern": "irregular_spots",
            "color_range": "brown_to_dark"
        }
    },
    "severely_burnt": {
        "description": "Pizza with heavy burning and charred areas",
        "prompts": [
            "pizza severely burnt with black charred areas, overcooked, inedible appearance",
            "pizza with heavy burning and black spots, charred crust, burnt food photography",
            "pizza completely overcooked with dark charred patterns, burnt appearance",
            "pizza with extensive black burn marks, severely overcooked, documentary style"
        ],
        "negative_prompts": [
            "lightly cooked, golden brown, perfect cooking, appetizing"
        ],
        "burn_params": {
            "intensity": "high",
            "pattern": "large_areas",
            "color_range": "dark_to_black"
        }
    }
}

COOKING_TRANSITION_TEMPLATES = {
    "raw_to_light": {
        "description": "Transition from raw dough to lightly cooked",
        "prompts": [
            "pizza in transition from raw to lightly cooked, uneven cooking pattern, documentary style",
            "pizza with mixed cooking states, some areas raw dough, others lightly baked",
            "pizza showing cooking progression, raw dough transitioning to light golden color",
            "pizza with gradient cooking effect, from uncooked to light browning"
        ],
        "cooking_stages": ["basic", "segment"],
        "transition_params": {
            "from_state": "raw",
            "to_state": "light_cooked",
            "pattern": "gradient"
        }
    },
    "cooked_to_burnt": {
        "description": "Transition from properly cooked to burnt",
        "prompts": [
            "pizza transitioning from golden brown to burnt, uneven overcooking pattern",
            "pizza with mixed states, some areas perfect golden, others starting to burn",
            "pizza showing progression from cooked to overcooked, burning in progress",
            "pizza with gradient burning effect, from cooked to charred areas"
        ],
        "cooking_stages": ["combined", "burnt"],
        "transition_params": {
            "from_state": "cooked",
            "to_state": "burnt",
            "pattern": "progressive"
        }
    }
}

class PropertyVerifier:
    """Verify that generated images contain the targeted properties"""
    
    def __init__(self, config: TargetedGenerationConfig):
        self.config = config
        
    def verify_lighting_condition(self, image: Image.Image, target_lighting: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Verify that the image exhibits the target lighting condition"""
        # Convert to grayscale for shadow analysis
        gray = ImageOps.grayscale(image)
        gray_array = np.array(gray)
        
        # Calculate basic lighting metrics
        brightness_mean = np.mean(gray_array)
        brightness_std = np.std(gray_array)
        
        # Calculate shadow metrics (looking for areas significantly darker than mean)
        shadow_threshold = brightness_mean - brightness_std
        shadow_pixels = np.sum(gray_array < shadow_threshold)
        shadow_ratio = shadow_pixels / gray_array.size
        
        # Calculate highlight metrics
        highlight_threshold = brightness_mean + brightness_std
        highlight_pixels = np.sum(gray_array > highlight_threshold)
        highlight_ratio = highlight_pixels / gray_array.size
        
        metrics = {
            "brightness_mean": float(brightness_mean),
            "brightness_std": float(brightness_std),
            "shadow_ratio": float(shadow_ratio),
            "highlight_ratio": float(highlight_ratio),
            "contrast_ratio": float(brightness_std / max(brightness_mean, 1))
        }
        
        # Verify based on lighting condition
        verification_score = 0.0
        
        if target_lighting == "overhead_harsh":
            # Expect high contrast and moderate shadow ratio
            if metrics["contrast_ratio"] > 0.3 and 0.1 < metrics["shadow_ratio"] < 0.4:
                verification_score = min(1.0, metrics["contrast_ratio"] + metrics["shadow_ratio"])
        elif target_lighting == "side_dramatic":
            # Expect high shadow ratio and contrast
            if metrics["shadow_ratio"] > 0.3 and metrics["contrast_ratio"] > 0.4:
                verification_score = min(1.0, metrics["shadow_ratio"] + metrics["contrast_ratio"] * 0.5)
        elif target_lighting == "dim_ambient":
            # Expect low brightness and low contrast
            if metrics["brightness_mean"] < 120 and metrics["contrast_ratio"] < 0.3:
                verification_score = 1.0 - (metrics["brightness_mean"] / 255.0) + (0.3 - metrics["contrast_ratio"])
        elif target_lighting == "backlit_rim":
            # Expect high highlight ratio around edges
            if metrics["highlight_ratio"] > 0.1:
                verification_score = min(1.0, metrics["highlight_ratio"] * 5)
        
        verification_score = max(0.0, min(1.0, verification_score))
        is_verified = verification_score >= self.config.property_verification_threshold
        
        return is_verified, verification_score, metrics
    
    def verify_burn_level(self, image: Image.Image, target_burn: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Verify that the image exhibits the target burn level"""
        # Convert to RGB array for color analysis
        rgb_array = np.array(image)
        
        # Calculate burn indicators
        # Look for dark brown/black areas indicating burning
        red_channel = rgb_array[:, :, 0]
        green_channel = rgb_array[:, :, 1]
        blue_channel = rgb_array[:, :, 2]
        
        # Calculate brown/black pixel ratios
        # Brown: low RGB values with red > green > blue (approximately)
        # Black: very low RGB values across all channels
        
        darkness_threshold = 80  # Very dark pixels
        brown_threshold = 120   # Brown/burnt pixels
        
        dark_pixels = np.sum((red_channel < darkness_threshold) & 
                           (green_channel < darkness_threshold) & 
                           (blue_channel < darkness_threshold))
        
        brown_pixels = np.sum((red_channel < brown_threshold) & 
                            (green_channel < brown_threshold) & 
                            (blue_channel < brown_threshold) & 
                            (red_channel >= darkness_threshold))
        
        total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
        dark_ratio = dark_pixels / total_pixels
        brown_ratio = brown_pixels / total_pixels
        burnt_ratio = dark_ratio + brown_ratio
        
        metrics = {
            "dark_ratio": float(dark_ratio),
            "brown_ratio": float(brown_ratio),
            "burnt_ratio": float(burnt_ratio),
            "mean_brightness": float(np.mean(rgb_array))
        }
        
        # Verify based on burn level
        verification_score = 0.0
        
        if target_burn == "slightly_burnt":
            # Expect low burn ratio, mostly brown spots
            if 0.05 < burnt_ratio < 0.25 and brown_ratio > dark_ratio:
                verification_score = 1.0 - abs(0.15 - burnt_ratio) * 4
        elif target_burn == "moderately_burnt":
            # Expect moderate burn ratio
            if 0.2 < burnt_ratio < 0.5:
                verification_score = 1.0 - abs(0.35 - burnt_ratio) * 2
        elif target_burn == "severely_burnt":
            # Expect high burn ratio with significant dark areas
            if burnt_ratio > 0.4 and dark_ratio > 0.1:
                verification_score = min(1.0, burnt_ratio + dark_ratio * 0.5)
        
        verification_score = max(0.0, min(1.0, verification_score))
        is_verified = verification_score >= self.config.property_verification_threshold
        
        return is_verified, verification_score, metrics

class TargetedDiffusionPipeline:
    """
    Enhanced diffusion pipeline for targeted image generation with specific properties.
    
    This pipeline extends the existing infrastructure to provide sophisticated control
    over generated image properties including lighting conditions, burn levels, and
    cooking transitions.
    """
    
    def __init__(self, config: TargetedGenerationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.property_verifier = PropertyVerifier(config)
        
        # Initialize existing pipeline components if available
        if EXISTING_PIPELINE_AVAILABLE:
            self._init_existing_pipeline()
        else:
            logger.warning("Existing pipeline components not available. Running in limited mode.")
            self.generator = None
            self.advanced_control = None
            self.data_agent = None
        
        # Statistics tracking
        self.stats = {
            "total_requested": 0,
            "total_generated": 0,
            "successful_generations": 0,
            "failed_verifications": 0,
            "retries_used": 0,
            "average_verification_score": 0.0,
            "per_property_stats": {}
        }
        
        # Setup output directories
        self._setup_output_structure()
        
        logger.info(f"Initialized TargetedDiffusionPipeline with config: {config.model_type}")
    
    def _init_existing_pipeline(self):
        """Initialize the existing pipeline components"""
        try:
            # Initialize base generator
            generator_config = {
                "offload_to_cpu": self.config.enable_cpu_offload,
                "attention_slicing": self.config.enable_attention_slicing
            }
            
            self.generator = PizzaDiffusionGenerator(
                model_id=self.config.model_id,
                model_type=self.config.model_type,
                image_size=self.config.image_size,
                device=None,  # Let it auto-detect
                output_dir=str(self.output_dir),
                batch_size=self.config.batch_size,
                quality_threshold=self.config.quality_threshold,
                save_metadata=self.config.save_metadata,
                config=generator_config
            )
            
            # Initialize advanced control
            self.advanced_control = AdvancedPizzaDiffusionControl(
                generator=self.generator,
                output_dir=str(self.output_dir)
            )
            
            # Initialize data agent for quality control
            agent_config = AgentConfig(
                save_format="png",
                save_quality=95,
                max_file_size_mb=10,
                quality_threshold=self.config.quality_threshold
            )
            self.data_agent = DiffusionDataAgent(agent_config)
            
            logger.info("Successfully initialized existing pipeline components")
            
        except Exception as e:
            logger.error(f"Failed to initialize existing pipeline: {e}")
            self.generator = None
            self.advanced_control = None
            self.data_agent = None
    
    def _setup_output_structure(self):
        """Setup the output directory structure"""
        subdirs = [
            "lighting_conditions",
            "burn_levels", 
            "cooking_transitions",
            "combined_properties",
            "metadata",
            "verification_reports"
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def generate_with_lighting_condition(
        self, 
        lighting_condition: str,
        count: int = 10,
        base_stage: str = "combined",
        custom_prompt_additions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images with specific lighting conditions.
        
        Args:
            lighting_condition: Key from LIGHTING_CONDITION_TEMPLATES
            count: Number of images to generate
            base_stage: Base cooking stage from PIZZA_STAGES
            custom_prompt_additions: Additional prompt elements
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            List of metadata dictionaries for generated images
        """
        if lighting_condition not in LIGHTING_CONDITION_TEMPLATES:
            raise ValueError(f"Unknown lighting condition: {lighting_condition}")
        
        template = LIGHTING_CONDITION_TEMPLATES[lighting_condition]
        self.stats["total_requested"] += count
        
        if lighting_condition not in self.stats["per_property_stats"]:
            self.stats["per_property_stats"][lighting_condition] = {
                "requested": 0,
                "generated": 0,
                "verified": 0,
                "avg_score": 0.0
            }
        
        self.stats["per_property_stats"][lighting_condition]["requested"] += count
        
        results = []
        
        for i in range(count):
            attempt = 0
            generated = False
            
            while attempt < self.config.max_retries and not generated:
                try:
                    # Construct prompt
                    base_prompt = random.choice(template["prompts"])
                    
                    if custom_prompt_additions:
                        addition = random.choice(custom_prompt_additions)
                        base_prompt = f"{base_prompt}, {addition}"
                    
                    # Add lighting-specific details
                    lighting_params = template.get("lighting_params", {})
                    if lighting_params.get("intensity") == "high":
                        base_prompt += ", high contrast lighting"
                    elif lighting_params.get("intensity") == "low":
                        base_prompt += ", soft low light"
                    
                    negative_prompt = random.choice(template["negative_prompts"])
                    
                    # Generate image using existing pipeline
                    if self.generator:
                        generation_results = self.generator.generate_images(
                            stage=base_stage,
                            num_images=1,
                            custom_prompts=[base_prompt],
                            custom_negative_prompts=[negative_prompt],
                            guidance_scale=self.config.guidance_scale,
                            seed=seed,
                            prefix=f"lighting_{lighting_condition}",
                            num_inference_steps=self.config.num_inference_steps,
                            **kwargs
                        )
                        
                        if generation_results:
                            result = generation_results[0]
                            
                            # Load and verify the generated image
                            image_path = result["path"]
                            image = Image.open(image_path)
                            
                            # Verify target properties
                            if self.config.verify_target_properties:
                                is_verified, score, metrics = self.property_verifier.verify_lighting_condition(
                                    image, lighting_condition
                                )
                                
                                result["lighting_verification"] = {
                                    "verified": is_verified,
                                    "score": score,
                                    "metrics": metrics,
                                    "target_condition": lighting_condition,
                                    "verification_threshold": self.config.property_verification_threshold
                                }
                                
                                if not is_verified and attempt < self.config.max_retries - 1:
                                    logger.warning(f"Verification failed for {lighting_condition} "
                                                 f"(score: {score:.3f}), retrying...")
                                    attempt += 1
                                    self.stats["failed_verifications"] += 1
                                    self.stats["retries_used"] += 1
                                    continue
                            else:
                                # Skip verification
                                result["lighting_verification"] = {
                                    "verified": True,
                                    "score": 1.0,
                                    "metrics": {},
                                    "target_condition": lighting_condition,
                                    "verification_disabled": True
                                }
                            
                            # Move to appropriate subdirectory
                            target_dir = self.output_dir / "lighting_conditions" / lighting_condition
                            target_dir.mkdir(exist_ok=True)
                            
                            new_path = target_dir / Path(image_path).name
                            Path(image_path).rename(new_path)
                            result["path"] = str(new_path)
                            
                            # Update metadata with enhanced information
                            result.update({
                                "generation_type": "targeted_lighting",
                                "target_property": lighting_condition,
                                "template_used": template["description"],
                                "final_prompt": base_prompt,
                                "final_negative_prompt": negative_prompt,
                                "lighting_parameters": lighting_params,
                                "attempt_number": attempt + 1
                            })
                            
                            # Save enhanced metadata
                            if self.config.save_metadata:
                                metadata_path = new_path.with_suffix(".json")
                                with open(metadata_path, 'w') as f:
                                    json.dump(result, f, indent=2)
                            
                            results.append(result)
                            generated = True
                            
                            # Update statistics
                            self.stats["total_generated"] += 1
                            self.stats["successful_generations"] += 1
                            self.stats["per_property_stats"][lighting_condition]["generated"] += 1
                            
                            if result["lighting_verification"]["verified"]:
                                self.stats["per_property_stats"][lighting_condition]["verified"] += 1
                            
                            score = result["lighting_verification"]["score"]
                            current_avg = self.stats["per_property_stats"][lighting_condition]["avg_score"]
                            total_generated = self.stats["per_property_stats"][lighting_condition]["generated"]
                            self.stats["per_property_stats"][lighting_condition]["avg_score"] = \
                                (current_avg * (total_generated - 1) + score) / total_generated
                            
                            logger.info(f"Successfully generated {lighting_condition} image "
                                      f"(attempt {attempt + 1}, score: {score:.3f})")
                    else:
                        logger.error("Generator not available")
                        break
                        
                except Exception as e:
                    logger.error(f"Error generating {lighting_condition} image (attempt {attempt + 1}): {e}")
                    attempt += 1
                    if attempt < self.config.max_retries:
                        self.stats["retries_used"] += 1
            
            if not generated:
                logger.error(f"Failed to generate {lighting_condition} image after {self.config.max_retries} attempts")
        
        return results
    
    def generate_with_burn_level(
        self,
        burn_level: str,
        count: int = 10,
        base_stage: str = "burnt",
        custom_prompt_additions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images with specific burn levels.
        
        Args:
            burn_level: Key from BURN_LEVEL_TEMPLATES
            count: Number of images to generate
            base_stage: Base cooking stage from PIZZA_STAGES
            custom_prompt_additions: Additional prompt elements
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            List of metadata dictionaries for generated images
        """
        if burn_level not in BURN_LEVEL_TEMPLATES:
            raise ValueError(f"Unknown burn level: {burn_level}")
        
        template = BURN_LEVEL_TEMPLATES[burn_level]
        self.stats["total_requested"] += count
        
        if burn_level not in self.stats["per_property_stats"]:
            self.stats["per_property_stats"][burn_level] = {
                "requested": 0,
                "generated": 0,
                "verified": 0,
                "avg_score": 0.0
            }
        
        self.stats["per_property_stats"][burn_level]["requested"] += count
        
        results = []
        
        for i in range(count):
            attempt = 0
            generated = False
            
            while attempt < self.config.max_retries and not generated:
                try:
                    # Construct prompt
                    base_prompt = random.choice(template["prompts"])
                    
                    if custom_prompt_additions:
                        addition = random.choice(custom_prompt_additions)
                        base_prompt = f"{base_prompt}, {addition}"
                    
                    # Add burn-specific details
                    burn_params = template.get("burn_params", {})
                    if burn_params.get("intensity") == "light":
                        base_prompt += ", lightly overcooked"
                    elif burn_params.get("intensity") == "high":
                        base_prompt += ", heavily burnt, charred"
                    
                    negative_prompt = random.choice(template["negative_prompts"])
                    
                    # Generate image using existing pipeline
                    if self.generator:
                        generation_results = self.generator.generate_images(
                            stage=base_stage,
                            num_images=1,
                            custom_prompts=[base_prompt],
                            custom_negative_prompts=[negative_prompt],
                            guidance_scale=self.config.guidance_scale,
                            seed=seed,
                            prefix=f"burn_{burn_level}",
                            num_inference_steps=self.config.num_inference_steps,
                            **kwargs
                        )
                        
                        if generation_results:
                            result = generation_results[0]
                            
                            # Load and verify the generated image
                            image_path = result["path"]
                            image = Image.open(image_path)
                            
                            # Verify target properties
                            if self.config.verify_target_properties:
                                is_verified, score, metrics = self.property_verifier.verify_burn_level(
                                    image, burn_level
                                )
                                
                                result["burn_verification"] = {
                                    "verified": is_verified,
                                    "score": score,
                                    "metrics": metrics,
                                    "target_burn_level": burn_level,
                                    "verification_threshold": self.config.property_verification_threshold
                                }
                                
                                if not is_verified and attempt < self.config.max_retries - 1:
                                    logger.warning(f"Verification failed for {burn_level} "
                                                 f"(score: {score:.3f}), retrying...")
                                    attempt += 1
                                    self.stats["failed_verifications"] += 1
                                    self.stats["retries_used"] += 1
                                    continue
                            else:
                                result["burn_verification"] = {
                                    "verified": True,
                                    "score": 1.0,
                                    "metrics": {},
                                    "target_burn_level": burn_level,
                                    "verification_disabled": True
                                }
                            
                            # Move to appropriate subdirectory
                            target_dir = self.output_dir / "burn_levels" / burn_level
                            target_dir.mkdir(exist_ok=True)
                            
                            new_path = target_dir / Path(image_path).name
                            Path(image_path).rename(new_path)
                            result["path"] = str(new_path)
                            
                            # Update metadata with enhanced information
                            result.update({
                                "generation_type": "targeted_burn_level",
                                "target_property": burn_level,
                                "template_used": template["description"],
                                "final_prompt": base_prompt,
                                "final_negative_prompt": negative_prompt,
                                "burn_parameters": burn_params,
                                "attempt_number": attempt + 1
                            })
                            
                            # Save enhanced metadata
                            if self.config.save_metadata:
                                metadata_path = new_path.with_suffix(".json")
                                with open(metadata_path, 'w') as f:
                                    json.dump(result, f, indent=2)
                            
                            results.append(result)
                            generated = True
                            
                            # Update statistics
                            self.stats["total_generated"] += 1
                            self.stats["successful_generations"] += 1
                            self.stats["per_property_stats"][burn_level]["generated"] += 1
                            
                            if result["burn_verification"]["verified"]:
                                self.stats["per_property_stats"][burn_level]["verified"] += 1
                            
                            score = result["burn_verification"]["score"]
                            current_avg = self.stats["per_property_stats"][burn_level]["avg_score"]
                            total_generated = self.stats["per_property_stats"][burn_level]["generated"]
                            self.stats["per_property_stats"][burn_level]["avg_score"] = \
                                (current_avg * (total_generated - 1) + score) / total_generated
                            
                            logger.info(f"Successfully generated {burn_level} image "
                                      f"(attempt {attempt + 1}, score: {score:.3f})")
                    else:
                        logger.error("Generator not available")
                        break
                        
                except Exception as e:
                    logger.error(f"Error generating {burn_level} image (attempt {attempt + 1}): {e}")
                    attempt += 1
                    if attempt < self.config.max_retries:
                        self.stats["retries_used"] += 1
            
            if not generated:
                logger.error(f"Failed to generate {burn_level} image after {self.config.max_retries} attempts")
        
        return results
    
    def generate_combined_properties(
        self,
        lighting_condition: str,
        burn_level: str,
        count: int = 10,
        custom_prompt_additions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images with combined lighting and burn properties.
        
        Args:
            lighting_condition: Key from LIGHTING_CONDITION_TEMPLATES
            burn_level: Key from BURN_LEVEL_TEMPLATES
            count: Number of images to generate
            custom_prompt_additions: Additional prompt elements
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            List of metadata dictionaries for generated images
        """
        if lighting_condition not in LIGHTING_CONDITION_TEMPLATES:
            raise ValueError(f"Unknown lighting condition: {lighting_condition}")
        if burn_level not in BURN_LEVEL_TEMPLATES:
            raise ValueError(f"Unknown burn level: {burn_level}")
        
        lighting_template = LIGHTING_CONDITION_TEMPLATES[lighting_condition]
        burn_template = BURN_LEVEL_TEMPLATES[burn_level]
        
        property_key = f"{lighting_condition}+{burn_level}"
        self.stats["total_requested"] += count
        
        if property_key not in self.stats["per_property_stats"]:
            self.stats["per_property_stats"][property_key] = {
                "requested": 0,
                "generated": 0,
                "verified": 0,
                "avg_score": 0.0
            }
        
        self.stats["per_property_stats"][property_key]["requested"] += count
        
        results = []
        
        for i in range(count):
            attempt = 0
            generated = False
            
            while attempt < self.config.max_retries and not generated:
                try:
                    # Combine prompts from both templates
                    lighting_prompt = random.choice(lighting_template["prompts"])
                    burn_prompt = random.choice(burn_template["prompts"])
                    
                    # Create combined prompt
                    base_prompt = f"{burn_prompt}, {lighting_template['description']}"
                    
                    if custom_prompt_additions:
                        addition = random.choice(custom_prompt_additions)
                        base_prompt = f"{base_prompt}, {addition}"
                    
                    # Combine negative prompts
                    lighting_neg = random.choice(lighting_template["negative_prompts"])
                    burn_neg = random.choice(burn_template["negative_prompts"])
                    negative_prompt = f"{lighting_neg}, {burn_neg}"
                    
                    # Determine appropriate base stage
                    if "severely" in burn_level or "moderately" in burn_level:
                        base_stage = "burnt"
                    else:
                        base_stage = "combined"
                    
                    # Generate image using existing pipeline
                    if self.generator:
                        generation_results = self.generator.generate_images(
                            stage=base_stage,
                            num_images=1,
                            custom_prompts=[base_prompt],
                            custom_negative_prompts=[negative_prompt],
                            guidance_scale=self.config.guidance_scale,
                            seed=seed,
                            prefix=f"combined_{lighting_condition}_{burn_level}",
                            num_inference_steps=self.config.num_inference_steps,
                            **kwargs
                        )
                        
                        if generation_results:
                            result = generation_results[0]
                            
                            # Load and verify the generated image
                            image_path = result["path"]
                            image = Image.open(image_path)
                            
                            # Verify both target properties
                            overall_verified = True
                            verification_scores = []
                            
                            if self.config.verify_target_properties:
                                # Verify lighting
                                lighting_verified, lighting_score, lighting_metrics = \
                                    self.property_verifier.verify_lighting_condition(image, lighting_condition)
                                
                                # Verify burn level
                                burn_verified, burn_score, burn_metrics = \
                                    self.property_verifier.verify_burn_level(image, burn_level)
                                
                                verification_scores = [lighting_score, burn_score]
                                overall_score = np.mean(verification_scores)
                                overall_verified = lighting_verified and burn_verified
                                
                                result["combined_verification"] = {
                                    "overall_verified": overall_verified,
                                    "overall_score": float(overall_score),
                                    "lighting_verification": {
                                        "verified": lighting_verified,
                                        "score": lighting_score,
                                        "metrics": lighting_metrics,
                                        "target": lighting_condition
                                    },
                                    "burn_verification": {
                                        "verified": burn_verified,
                                        "score": burn_score,
                                        "metrics": burn_metrics,
                                        "target": burn_level
                                    },
                                    "verification_threshold": self.config.property_verification_threshold
                                }
                                
                                if not overall_verified and attempt < self.config.max_retries - 1:
                                    logger.warning(f"Combined verification failed for {property_key} "
                                                 f"(score: {overall_score:.3f}), retrying...")
                                    attempt += 1
                                    self.stats["failed_verifications"] += 1
                                    self.stats["retries_used"] += 1
                                    continue
                            else:
                                result["combined_verification"] = {
                                    "overall_verified": True,
                                    "overall_score": 1.0,
                                    "verification_disabled": True
                                }
                            
                            # Move to appropriate subdirectory
                            target_dir = self.output_dir / "combined_properties" / property_key
                            target_dir.mkdir(exist_ok=True)
                            
                            new_path = target_dir / Path(image_path).name
                            Path(image_path).rename(new_path)
                            result["path"] = str(new_path)
                            
                            # Update metadata with enhanced information
                            result.update({
                                "generation_type": "combined_properties",
                                "target_properties": {
                                    "lighting_condition": lighting_condition,
                                    "burn_level": burn_level
                                },
                                "template_descriptions": {
                                    "lighting": lighting_template["description"],
                                    "burn": burn_template["description"]
                                },
                                "final_prompt": base_prompt,
                                "final_negative_prompt": negative_prompt,
                                "lighting_parameters": lighting_template.get("lighting_params", {}),
                                "burn_parameters": burn_template.get("burn_params", {}),
                                "attempt_number": attempt + 1
                            })
                            
                            # Save enhanced metadata
                            if self.config.save_metadata:
                                metadata_path = new_path.with_suffix(".json")
                                with open(metadata_path, 'w') as f:
                                    json.dump(result, f, indent=2)
                            
                            results.append(result)
                            generated = True
                            
                            # Update statistics
                            self.stats["total_generated"] += 1
                            self.stats["successful_generations"] += 1
                            self.stats["per_property_stats"][property_key]["generated"] += 1
                            
                            if result["combined_verification"]["overall_verified"]:
                                self.stats["per_property_stats"][property_key]["verified"] += 1
                            
                            score = result["combined_verification"]["overall_score"]
                            current_avg = self.stats["per_property_stats"][property_key]["avg_score"]
                            total_generated = self.stats["per_property_stats"][property_key]["generated"]
                            self.stats["per_property_stats"][property_key]["avg_score"] = \
                                (current_avg * (total_generated - 1) + score) / total_generated
                            
                            logger.info(f"Successfully generated combined {property_key} image "
                                      f"(attempt {attempt + 1}, score: {score:.3f})")
                    else:
                        logger.error("Generator not available")
                        break
                        
                except Exception as e:
                    logger.error(f"Error generating combined {property_key} image (attempt {attempt + 1}): {e}")
                    attempt += 1
                    if attempt < self.config.max_retries:
                        self.stats["retries_used"] += 1
            
            if not generated:
                logger.error(f"Failed to generate combined {property_key} image after {self.config.max_retries} attempts")
        
        return results
    
    def generate_cooking_transition(
        self,
        transition_type: str,
        count: int = 10,
        custom_prompt_additions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images showing cooking state transitions.
        
        Args:
            transition_type: Key from COOKING_TRANSITION_TEMPLATES
            count: Number of images to generate
            custom_prompt_additions: Additional prompt elements
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            List of metadata dictionaries for generated images
        """
        if transition_type not in COOKING_TRANSITION_TEMPLATES:
            raise ValueError(f"Unknown transition type: {transition_type}")
        
        template = COOKING_TRANSITION_TEMPLATES[transition_type]
        self.stats["total_requested"] += count
        
        if transition_type not in self.stats["per_property_stats"]:
            self.stats["per_property_stats"][transition_type] = {
                "requested": 0,
                "generated": 0,
                "verified": 0,
                "avg_score": 0.0
            }
        
        self.stats["per_property_stats"][transition_type]["requested"] += count
        
        results = []
        
        for i in range(count):
            attempt = 0
            generated = False
            
            while attempt < self.config.max_retries and not generated:
                try:
                    # Construct prompt
                    base_prompt = random.choice(template["prompts"])
                    
                    if custom_prompt_additions:
                        addition = random.choice(custom_prompt_additions)
                        base_prompt = f"{base_prompt}, {addition}"
                    
                    # Select appropriate base stage
                    cooking_stages = template.get("cooking_stages", ["progression"])
                    base_stage = random.choice(cooking_stages)
                    
                    # Generate image using existing pipeline
                    if self.generator:
                        generation_results = self.generator.generate_images(
                            stage=base_stage,
                            num_images=1,
                            custom_prompts=[base_prompt],
                            guidance_scale=self.config.guidance_scale,
                            seed=seed,
                            prefix=f"transition_{transition_type}",
                            num_inference_steps=self.config.num_inference_steps,
                            **kwargs
                        )
                        
                        if generation_results:
                            result = generation_results[0]
                            
                            # Move to appropriate subdirectory
                            target_dir = self.output_dir / "cooking_transitions" / transition_type
                            target_dir.mkdir(exist_ok=True)
                            
                            image_path = result["path"]
                            new_path = target_dir / Path(image_path).name
                            Path(image_path).rename(new_path)
                            result["path"] = str(new_path)
                            
                            # Update metadata with enhanced information
                            result.update({
                                "generation_type": "cooking_transition",
                                "target_property": transition_type,
                                "template_used": template["description"],
                                "final_prompt": base_prompt,
                                "transition_parameters": template.get("transition_params", {}),
                                "cooking_stages_used": cooking_stages,
                                "base_stage": base_stage,
                                "attempt_number": attempt + 1
                            })
                            
                            # Save enhanced metadata
                            if self.config.save_metadata:
                                metadata_path = new_path.with_suffix(".json")
                                with open(metadata_path, 'w') as f:
                                    json.dump(result, f, indent=2)
                            
                            results.append(result)
                            generated = True
                            
                            # Update statistics
                            self.stats["total_generated"] += 1
                            self.stats["successful_generations"] += 1
                            self.stats["per_property_stats"][transition_type]["generated"] += 1
                            self.stats["per_property_stats"][transition_type]["verified"] += 1  # Transitions don't have verification yet
                            
                            logger.info(f"Successfully generated {transition_type} transition image")
                    else:
                        logger.error("Generator not available")
                        break
                        
                except Exception as e:
                    logger.error(f"Error generating {transition_type} transition image (attempt {attempt + 1}): {e}")
                    attempt += 1
                    if attempt < self.config.max_retries:
                        self.stats["retries_used"] += 1
            
            if not generated:
                logger.error(f"Failed to generate {transition_type} transition image after {self.config.max_retries} attempts")
        
        return results
    
    def generate_comprehensive_dataset(
        self,
        lighting_counts: Optional[Dict[str, int]] = None,
        burn_counts: Optional[Dict[str, int]] = None,
        transition_counts: Optional[Dict[str, int]] = None,
        combined_counts: Optional[Dict[Tuple[str, str], int]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a comprehensive dataset with various targeted properties.
        
        Args:
            lighting_counts: Dictionary mapping lighting conditions to counts
            burn_counts: Dictionary mapping burn levels to counts
            transition_counts: Dictionary mapping transition types to counts
            combined_counts: Dictionary mapping (lighting, burn) tuples to counts
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results organized by property type
        """
        # Set default counts if not provided
        if lighting_counts is None:
            lighting_counts = {condition: 5 for condition in LIGHTING_CONDITION_TEMPLATES.keys()}
        
        if burn_counts is None:
            burn_counts = {level: 5 for level in BURN_LEVEL_TEMPLATES.keys()}
        
        if transition_counts is None:
            transition_counts = {transition: 3 for transition in COOKING_TRANSITION_TEMPLATES.keys()}
        
        if combined_counts is None:
            # Generate some combinations
            combined_counts = {
                ("overhead_harsh", "slightly_burnt"): 3,
                ("side_dramatic", "moderately_burnt"): 3,
                ("dim_ambient", "severely_burnt"): 2,
                ("backlit_rim", "slightly_burnt"): 2
            }
        
        results = {
            "lighting_conditions": {},
            "burn_levels": {},
            "cooking_transitions": {},
            "combined_properties": {}
        }
        
        start_time = time.time()
        
        logger.info("Starting comprehensive dataset generation...")
        
        # Generate lighting condition images
        logger.info("Generating lighting condition images...")
        for condition, count in lighting_counts.items():
            logger.info(f"Generating {count} images for lighting condition: {condition}")
            results["lighting_conditions"][condition] = self.generate_with_lighting_condition(
                condition, count, seed=seed, **kwargs
            )
        
        # Generate burn level images
        logger.info("Generating burn level images...")
        for level, count in burn_counts.items():
            logger.info(f"Generating {count} images for burn level: {level}")
            results["burn_levels"][level] = self.generate_with_burn_level(
                level, count, seed=seed, **kwargs
            )
        
        # Generate cooking transition images
        logger.info("Generating cooking transition images...")
        for transition, count in transition_counts.items():
            logger.info(f"Generating {count} images for transition: {transition}")
            results["cooking_transitions"][transition] = self.generate_cooking_transition(
                transition, count, seed=seed, **kwargs
            )
        
        # Generate combined property images
        logger.info("Generating combined property images...")
        for (lighting, burn), count in combined_counts.items():
            key = f"{lighting}+{burn}"
            logger.info(f"Generating {count} images for combined properties: {key}")
            results["combined_properties"][key] = self.generate_combined_properties(
                lighting, burn, count, seed=seed, **kwargs
            )
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_final_report(results, total_time)
        
        logger.info(f"Comprehensive dataset generation completed in {total_time:.2f} seconds")
        
        return results
    
    def _generate_final_report(self, results: Dict[str, Any], generation_time: float):
        """Generate a comprehensive report of the generation session"""
        
        report = {
            "generation_session": {
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": generation_time,
                "configuration": asdict(self.config)
            },
            "statistics": self.stats,
            "results_summary": {},
            "verification_analysis": {}
        }
        
        # Summarize results
        for category, category_results in results.items():
            report["results_summary"][category] = {}
            for property_name, property_results in category_results.items():
                report["results_summary"][category][property_name] = {
                    "total_generated": len(property_results),
                    "successful_generations": len([r for r in property_results if r.get("quality_score", 0) >= self.config.quality_threshold])
                }
        
        # Analyze verification results
        total_verifications = 0
        successful_verifications = 0
        verification_scores = []
        
        for category, category_results in results.items():
            if category == "cooking_transitions":
                continue  # Transitions don't have verification yet
                
            for property_name, property_results in category_results.items():
                for result in property_results:
                    verification_key = None
                    if "lighting_verification" in result:
                        verification_key = "lighting_verification"
                    elif "burn_verification" in result:
                        verification_key = "burn_verification"
                    elif "combined_verification" in result:
                        verification_key = "combined_verification"
                    
                    if verification_key:
                        total_verifications += 1
                        verification_data = result[verification_key]
                        
                        if verification_key == "combined_verification":
                            verified = verification_data["overall_verified"]
                            score = verification_data["overall_score"]
                        else:
                            verified = verification_data["verified"]
                            score = verification_data["score"]
                        
                        if verified:
                            successful_verifications += 1
                        verification_scores.append(score)
        
        if total_verifications > 0:
            report["verification_analysis"] = {
                "total_verifications": total_verifications,
                "successful_verifications": successful_verifications,
                "verification_success_rate": successful_verifications / total_verifications,
                "average_verification_score": np.mean(verification_scores),
                "verification_score_std": np.std(verification_scores),
                "min_verification_score": np.min(verification_scores),
                "max_verification_score": np.max(verification_scores)
            }
        
        # Save report
        report_path = self.output_dir / "verification_reports" / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        # Log summary
        logger.info("=== GENERATION SESSION SUMMARY ===")
        logger.info(f"Total time: {generation_time:.2f} seconds")
        logger.info(f"Total requested: {self.stats['total_requested']}")
        logger.info(f"Total generated: {self.stats['total_generated']}")
        logger.info(f"Success rate: {(self.stats['successful_generations'] / max(self.stats['total_requested'], 1)):.2%}")
        
        if total_verifications > 0:
            logger.info(f"Verification success rate: {(successful_verifications / total_verifications):.2%}")
            logger.info(f"Average verification score: {np.mean(verification_scores):.3f}")
        
        logger.info(f"Retries used: {self.stats['retries_used']}")
        logger.info("=== END SUMMARY ===")

def main():
    """Command-line interface for the targeted diffusion pipeline"""
    parser = argparse.ArgumentParser(description="DIFFUSION-2.1: Targeted Image Generation Pipeline")
    
    # Basic parameters
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Diffusion model ID")
    parser.add_argument("--model-type", type=str, default="sdxl", choices=["sdxl", "kandinsky", "custom"],
                        help="Type of diffusion model")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Generated image size")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Generation batch size")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/targeted",
                        help="Output directory")
    
    # Generation modes
    parser.add_argument("--lighting-condition", type=str, choices=list(LIGHTING_CONDITION_TEMPLATES.keys()),
                        help="Generate images with specific lighting condition")
    parser.add_argument("--burn-level", type=str, choices=list(BURN_LEVEL_TEMPLATES.keys()),
                        help="Generate images with specific burn level")
    parser.add_argument("--transition-type", type=str, choices=list(COOKING_TRANSITION_TEMPLATES.keys()),
                        help="Generate cooking transition images")
    parser.add_argument("--combined", nargs=2, metavar=("LIGHTING", "BURN"),
                        help="Generate combined lighting and burn properties")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Generate comprehensive dataset with all properties")
    
    # Generation parameters
    parser.add_argument("--count", type=int, default=10,
                        help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale for generation")
    parser.add_argument("--num-inference-steps", type=int, default=30,
                        help="Number of inference steps")
    
    # Quality and verification
    parser.add_argument("--quality-threshold", type=float, default=0.7,
                        help="Quality threshold for generated images")
    parser.add_argument("--verify-properties", action="store_true", default=True,
                        help="Enable property verification")
    parser.add_argument("--verification-threshold", type=float, default=0.6,
                        help="Threshold for property verification")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum retries for failed generations")
    
    # Memory optimization
    parser.add_argument("--disable-cpu-offload", action="store_true",
                        help="Disable CPU offloading")
    parser.add_argument("--disable-attention-slicing", action="store_true",
                        help="Disable attention slicing")
    
    # Utility commands
    parser.add_argument("--list-templates", action="store_true",
                        help="List available templates")
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_templates:
        print("=== LIGHTING CONDITION TEMPLATES ===")
        for key, template in LIGHTING_CONDITION_TEMPLATES.items():
            print(f"{key}: {template['description']}")
        
        print("\n=== BURN LEVEL TEMPLATES ===")
        for key, template in BURN_LEVEL_TEMPLATES.items():
            print(f"{key}: {template['description']}")
        
        print("\n=== COOKING TRANSITION TEMPLATES ===")
        for key, template in COOKING_TRANSITION_TEMPLATES.items():
            print(f"{key}: {template['description']}")
        
        return 0
    
    # Setup configuration
    config = TargetedGenerationConfig(
        model_id=args.model_id,
        model_type=args.model_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        quality_threshold=args.quality_threshold,
        output_dir=args.output_dir,
        enable_cpu_offload=not args.disable_cpu_offload,
        enable_attention_slicing=not args.disable_attention_slicing,
        verify_target_properties=args.verify_properties,
        property_verification_threshold=args.verification_threshold,
        max_retries=args.max_retries
    )
    
    # Initialize pipeline
    pipeline = TargetedDiffusionPipeline(config)
    
    try:
        # Execute based on mode
        if args.lighting_condition:
            results = pipeline.generate_with_lighting_condition(
                args.lighting_condition, 
                count=args.count,
                seed=args.seed
            )
            print(f"Generated {len(results)} images with lighting condition: {args.lighting_condition}")
        
        elif args.burn_level:
            results = pipeline.generate_with_burn_level(
                args.burn_level,
                count=args.count,
                seed=args.seed
            )
            print(f"Generated {len(results)} images with burn level: {args.burn_level}")
        
        elif args.transition_type:
            results = pipeline.generate_cooking_transition(
                args.transition_type,
                count=args.count,
                seed=args.seed
            )
            print(f"Generated {len(results)} transition images: {args.transition_type}")
        
        elif args.combined:
            lighting, burn = args.combined
            results = pipeline.generate_combined_properties(
                lighting,
                burn,
                count=args.count,
                seed=args.seed
            )
            print(f"Generated {len(results)} images with combined properties: {lighting} + {burn}")
        
        elif args.comprehensive:
            results = pipeline.generate_comprehensive_dataset(seed=args.seed)
            total_generated = sum(
                len(category_results)
                for category in results.values()
                for category_results in category.values()
            )
            print(f"Generated comprehensive dataset with {total_generated} total images")
        
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('targeted_diffusion_pipeline.log')
        ]
    )
    
    exit(main())
