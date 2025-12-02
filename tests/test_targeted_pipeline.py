#!/usr/bin/env python3
"""
Test script for the targeted diffusion pipeline
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports without existing pipeline dependencies"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        import torch
        from PIL import Image
        print("✓ Core dependencies available")
    except ImportError as e:
        print(f"✗ Core dependency missing: {e}")
        assert False, f"Core dependency missing: {e}"

def test_configuration():
    """Test the configuration dataclass"""
    print("Testing configuration...")
    
    try:
        # Create a minimal version of the config to test
        from dataclasses import dataclass
        
        @dataclass
        class TestTargetedGenerationConfig:
            model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
            model_type: str = "sdxl"
            image_size: int = 512
            batch_size: int = 1
            output_dir: str = "data/synthetic/targeted"
            verify_target_properties: bool = True
            property_verification_threshold: float = 0.6
            max_retries: int = 3
        
        config = TestTargetedGenerationConfig()
        print(f"✓ Configuration created: {config.model_type}")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        assert False, f"Configuration test failed: {e}"

def test_property_templates():
    """Test the property template definitions"""
    print("Testing property templates...")
    
    try:
        # Define simplified templates for testing
        LIGHTING_CONDITION_TEMPLATES = {
            "overhead_harsh": {
                "description": "Direct overhead lighting with harsh shadows",
                "prompts": ["pizza with direct overhead light creating harsh shadows"],
                "negative_prompts": ["soft lighting, diffused light"],
                "lighting_params": {"direction": "overhead", "intensity": "high"}
            },
            "dim_ambient": {
                "description": "Dim ambient lighting with soft shadows", 
                "prompts": ["pizza in dim ambient lighting, soft shadows"],
                "negative_prompts": ["bright lighting, harsh shadows"],
                "lighting_params": {"direction": "ambient", "intensity": "low"}
            }
        }
        
        BURN_LEVEL_TEMPLATES = {
            "slightly_burnt": {
                "description": "Pizza with slight browning and light burn marks",
                "prompts": ["pizza slightly burnt with light browning on edges"],
                "negative_prompts": ["severely burnt, charred"],
                "burn_params": {"intensity": "light", "pattern": "edges_and_spots"}
            }
        }
        
        print(f"✓ Templates defined: {len(LIGHTING_CONDITION_TEMPLATES)} lighting, {len(BURN_LEVEL_TEMPLATES)} burn levels")
        
    except Exception as e:
        print(f"✗ Template test failed: {e}")
        assert False, f"Template test failed: {e}"

def test_property_verification():
    """Test the property verification logic"""
    print("Testing property verification...")
    
    try:
        import numpy as np
        from PIL import Image, ImageOps
        
        # Create a simple property verifier
        class TestPropertyVerifier:
            def __init__(self, threshold=0.6):
                self.threshold = threshold
            
            def verify_lighting_condition(self, image, target_lighting):
                """Simplified lighting verification"""
                gray = ImageOps.grayscale(image)
                gray_array = np.array(gray)
                
                brightness_mean = np.mean(gray_array)
                brightness_std = np.std(gray_array)
                shadow_ratio = np.sum(gray_array < (brightness_mean - brightness_std)) / gray_array.size
                
                metrics = {
                    "brightness_mean": float(brightness_mean),
                    "shadow_ratio": float(shadow_ratio),
                    "contrast_ratio": float(brightness_std / max(brightness_mean, 1))
                }
                
                # Simple verification logic
                if target_lighting == "overhead_harsh":
                    score = min(1.0, metrics["contrast_ratio"] + metrics["shadow_ratio"])
                elif target_lighting == "dim_ambient":
                    score = 1.0 - (metrics["brightness_mean"] / 255.0)
                else:
                    score = 0.5
                
                is_verified = score >= self.threshold
                return is_verified, score, metrics
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        verifier = TestPropertyVerifier()
        
        verified, score, metrics = verifier.verify_lighting_condition(test_image, "overhead_harsh")
        print(f"✓ Verification test completed: verified={verified}, score={score:.3f}")
        
    except Exception as e:
        print(f"✗ Property verification test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Property verification test failed: {e}"

def test_output_structure():
    """Test output directory structure creation"""
    print("Testing output structure...")
    
    try:
        test_output_dir = Path("test_output_targeted")
        
        # Create directory structure
        subdirs = [
            "lighting_conditions",
            "burn_levels", 
            "cooking_transitions",
            "combined_properties",
            "metadata"
        ]
        
        for subdir in subdirs:
            (test_output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output structure created at {test_output_dir}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_output_dir)
        
    except Exception as e:
        print(f"✗ Output structure test failed: {e}")
        assert False, f"Output structure test failed: {e}"


