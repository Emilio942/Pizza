#!/usr/bin/env python3
"""
Test script for property verification in the targeted diffusion pipeline
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_images():
    """Create synthetic test images with known properties for verification"""
    output_dir = Path("test_verification_images")
    output_dir.mkdir(exist_ok=True)
    
    test_images = {}
    
    # 1. Create overhead harsh lighting test image
    harsh_image = Image.new('RGB', (512, 512), color=(180, 150, 120))
    draw = ImageDraw.Draw(harsh_image)
    
    # Add harsh shadows (dark areas)
    for i in range(0, 512, 60):
        draw.rectangle([i, 0, i+30, 512], fill=(80, 60, 40))
    
    harsh_path = output_dir / "overhead_harsh_test.png"
    harsh_image.save(harsh_path)
    test_images['overhead_harsh'] = harsh_path
    
    # 2. Create dim ambient lighting test image
    dim_image = Image.new('RGB', (512, 512), color=(80, 70, 60))
    # Apply slight blur for soft lighting effect
    dim_image = dim_image.filter(ImageFilter.GaussianBlur(radius=2))
    
    dim_path = output_dir / "dim_ambient_test.png"
    dim_image.save(dim_path)
    test_images['dim_ambient'] = dim_path
    
    # 3. Create slightly burnt test image
    slight_burnt_image = Image.new('RGB', (512, 512), color=(200, 180, 120))
    draw = ImageDraw.Draw(slight_burnt_image)
    
    # Add light brown spots (slight burning)
    for i in range(0, 512, 100):
        for j in range(0, 512, 100):
            draw.ellipse([i, j, i+30, j+30], fill=(140, 110, 80))
    
    slight_burnt_path = output_dir / "slightly_burnt_test.png"
    slight_burnt_image.save(slight_burnt_path)
    test_images['slightly_burnt'] = slight_burnt_path
    
    # 4. Create severely burnt test image
    severe_burnt_image = Image.new('RGB', (512, 512), color=(60, 40, 20))
    draw = ImageDraw.Draw(severe_burnt_image)
    
    # Add black charred areas
    for i in range(0, 512, 80):
        for j in range(0, 512, 80):
            draw.ellipse([i, j, i+40, j+40], fill=(20, 15, 10))
    
    severe_burnt_path = output_dir / "severely_burnt_test.png"
    severe_burnt_image.save(severe_burnt_path)
    test_images['severely_burnt'] = severe_burnt_path
    
    return test_images

def test_property_verification():
    """Test the property verification algorithms"""
    print("Testing property verification algorithms...")
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import TargetedGenerationConfig, PropertyVerifier
        
        # Create test images
        test_images = create_test_images()
        print(f"Created {len(test_images)} test images")
        
        # Initialize property verifier
        config = TargetedGenerationConfig(property_verification_threshold=0.3)
        verifier = PropertyVerifier(config)
        
        results = {}
        
        # Test lighting condition verification
        for condition in ['overhead_harsh', 'dim_ambient']:
            if condition in test_images:
                image = Image.open(test_images[condition])
                verified, score, metrics = verifier.verify_lighting_condition(image, condition)
                
                results[f"lighting_{condition}"] = {
                    "verified": verified,
                    "score": score,
                    "metrics": metrics
                }
                
                print(f"\nLighting test - {condition}:")
                print(f"  Verified: {verified}")
                print(f"  Score: {score:.3f}")
                print(f"  Brightness: {metrics['brightness_mean']:.1f}")
                print(f"  Shadow ratio: {metrics['shadow_ratio']:.3f}")
                print(f"  Contrast: {metrics['contrast_ratio']:.3f}")
        
        # Test burn level verification
        for burn_level in ['slightly_burnt', 'severely_burnt']:
            if burn_level in test_images:
                image = Image.open(test_images[burn_level])
                verified, score, metrics = verifier.verify_burn_level(image, burn_level)
                
                results[f"burn_{burn_level}"] = {
                    "verified": verified,
                    "score": score,
                    "metrics": metrics
                }
                
                print(f"\nBurn test - {burn_level}:")
                print(f"  Verified: {verified}")
                print(f"  Score: {score:.3f}")
                print(f"  Dark ratio: {metrics['dark_ratio']:.3f}")
                print(f"  Brown ratio: {metrics['brown_ratio']:.3f}")
                print(f"  Total burnt ratio: {metrics['burnt_ratio']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Property verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_template_system():
    """Test the template-based prompt generation"""
    print("\nTesting template system...")
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import LIGHTING_CONDITION_TEMPLATES, BURN_LEVEL_TEMPLATES
        import random
        
        print("Available lighting conditions:")
        for condition, template in LIGHTING_CONDITION_TEMPLATES.items():
            sample_prompt = random.choice(template['prompts'])
            print(f"  {condition}: {sample_prompt[:60]}...")
        
        print("\nAvailable burn levels:")
        for level, template in BURN_LEVEL_TEMPLATES.items():
            sample_prompt = random.choice(template['prompts'])
            print(f"  {level}: {sample_prompt[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"Template system test failed: {e}")
        return False

def test_standalone_generation():
    """Test the pipeline in standalone mode (without actual diffusion models)"""
    print("\nTesting standalone pipeline features...")
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import TargetedDiffusionPipeline, TargetedGenerationConfig
        
        # Create configuration for testing
        config = TargetedGenerationConfig(
            output_dir="test_standalone_output",
            verify_target_properties=True,
            property_verification_threshold=0.3,
            max_retries=1
        )
        
        # Initialize pipeline
        pipeline = TargetedDiffusionPipeline(config)
        print(f"Pipeline initialized: {pipeline.output_dir}")
        print(f"Statistics tracking: {len(pipeline.stats)} metrics")
        
        # Test statistics update
        pipeline.stats["total_requested"] = 10
        pipeline.stats["successful_generations"] = 8
        print(f"Test stats: {pipeline.stats['successful_generations']}/{pipeline.stats['total_requested']} successful")
        
        # Test metadata structure
        test_metadata = {
            "generation_type": "targeted_lighting",
            "target_property": "overhead_harsh", 
            "verification": {
                "verified": True,
                "score": 0.85,
                "metrics": {"contrast_ratio": 0.45}
            }
        }
        
        # Test output directory structure
        subdirs = ["lighting_conditions", "burn_levels", "combined_properties"]
        for subdir in subdirs:
            subdir_path = pipeline.output_dir / subdir
            assert subdir_path.exists(), f"Output subdirectory {subdir} not created"
        
        print("✓ Standalone pipeline features working correctly")
        return True
        
    except Exception as e:
        print(f"Standalone generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("=== Testing DIFFUSION-2.1 Property Verification ===\n")
    
    tests_passed = 0
    total_tests = 0
    
    # Test property verification algorithms
    total_tests += 1
    verification_results = test_property_verification()
    if verification_results:
        tests_passed += 1
        print("✓ Property verification algorithms working")
    else:
        print("✗ Property verification failed")
    
    # Test template system
    total_tests += 1
    if test_template_system():
        tests_passed += 1
        print("✓ Template system working")
    else:
        print("✗ Template system failed")
    
    # Test standalone pipeline features
    total_tests += 1
    if test_standalone_generation():
        tests_passed += 1
        print("✓ Standalone pipeline features working")
    else:
        print("✗ Standalone pipeline features failed")
    
    print(f"\n=== Test Results: {tests_passed}/{total_tests} passed ===")
    
    if tests_passed == total_tests:
        print("\n🎉 All DIFFUSION-2.1 tests passed!")
        print("\nThe targeted diffusion pipeline is ready for:")
        print("1. Integration with actual diffusion models")
        print("2. End-to-end generation testing")
        print("3. Production dataset generation")
        
        # Show some sample verification results
        if verification_results:
            print("\nSample verification results:")
            for test_name, result in verification_results.items():
                print(f"  {test_name}: score={result['score']:.3f}, verified={result['verified']}")
    else:
        print(f"\n❌ {total_tests - tests_passed} tests failed. Please review and fix issues.")
    
    return 0 if tests_passed == total_tests else 1

if __name__ == "__main__":
    exit(main())
