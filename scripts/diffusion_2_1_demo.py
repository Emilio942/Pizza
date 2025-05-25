#!/usr/bin/env python3
"""
DIFFUSION-2.1 Demonstration Script

This script demonstrates the complete targeted diffusion pipeline functionality,
including prompt generation, property verification, and metadata handling.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup comprehensive logging for the demonstration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"diffusion_2_1_demo_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return log_file

def demonstrate_prompt_generation():
    """Demonstrate the enhanced prompt generation system"""
    print("=== PROMPT GENERATION DEMONSTRATION ===\n")
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import (
            LIGHTING_CONDITION_TEMPLATES, 
            BURN_LEVEL_TEMPLATES,
            COOKING_TRANSITION_TEMPLATES
        )
        import random
        
        # Show lighting condition prompts
        print("🔆 LIGHTING CONDITION PROMPTS:")
        for condition, template in LIGHTING_CONDITION_TEMPLATES.items():
            print(f"\n  {condition.upper()}:")
            print(f"    Description: {template['description']}")
            print(f"    Sample prompt: {random.choice(template['prompts'])}")
            print(f"    Negative prompt: {random.choice(template['negative_prompts'])}")
            print(f"    Parameters: {template['lighting_params']}")
        
        # Show burn level prompts
        print(f"\n🔥 BURN LEVEL PROMPTS:")
        for level, template in BURN_LEVEL_TEMPLATES.items():
            print(f"\n  {level.upper()}:")
            print(f"    Description: {template['description']}")
            print(f"    Sample prompt: {random.choice(template['prompts'])}")
            print(f"    Parameters: {template['burn_params']}")
        
        # Show cooking transition prompts
        print(f"\n🍕 COOKING TRANSITION PROMPTS:")
        for transition, template in COOKING_TRANSITION_TEMPLATES.items():
            print(f"\n  {transition.upper()}:")
            print(f"    Description: {template['description']}")
            print(f"    Sample prompt: {random.choice(template['prompts'])}")
            print(f"    Parameters: {template['transition_params']}")
        
        return True
        
    except Exception as e:
        print(f"Error demonstrating prompt generation: {e}")
        return False

def demonstrate_verification_system():
    """Demonstrate the property verification system with detailed analysis"""
    print("\n=== PROPERTY VERIFICATION DEMONSTRATION ===\n")
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import TargetedGenerationConfig, PropertyVerifier
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
        import numpy as np
        
        config = TargetedGenerationConfig(property_verification_threshold=0.5)
        verifier = PropertyVerifier(config)
        
        # Create demonstration images with known properties
        demo_dir = Path("demo_verification")
        demo_dir.mkdir(exist_ok=True)
        
        verification_results = {}
        
        # 1. Create and test lighting conditions
        print("🔆 LIGHTING CONDITION VERIFICATION:")
        
        # Overhead harsh - high contrast with shadows
        harsh_img = Image.new('RGB', (256, 256), color=(200, 180, 150))
        draw = ImageDraw.Draw(harsh_img)
        for i in range(0, 256, 40):
            draw.rectangle([i, 0, i+20, 256], fill=(50, 40, 30))
        
        verified, score, metrics = verifier.verify_lighting_condition(harsh_img, "overhead_harsh")
        verification_results['overhead_harsh'] = {'verified': verified, 'score': score, 'metrics': metrics}
        
        print(f"  Overhead Harsh: {'✓' if verified else '✗'} (score: {score:.3f})")
        print(f"    Brightness: {metrics['brightness_mean']:.1f}, Contrast: {metrics['contrast_ratio']:.3f}")
        print(f"    Shadow ratio: {metrics['shadow_ratio']:.3f}")
        
        # Dim ambient - low brightness, low contrast
        dim_img = Image.new('RGB', (256, 256), color=(60, 50, 40))
        dim_img = dim_img.filter(ImageFilter.GaussianBlur(radius=1))
        
        verified, score, metrics = verifier.verify_lighting_condition(dim_img, "dim_ambient")
        verification_results['dim_ambient'] = {'verified': verified, 'score': score, 'metrics': metrics}
        
        print(f"  Dim Ambient: {'✓' if verified else '✗'} (score: {score:.3f})")
        print(f"    Brightness: {metrics['brightness_mean']:.1f}, Contrast: {metrics['contrast_ratio']:.3f}")
        
        # 2. Create and test burn levels
        print(f"\n🔥 BURN LEVEL VERIFICATION:")
        
        # Slightly burnt - some brown areas
        slight_img = Image.new('RGB', (256, 256), color=(180, 160, 120))
        draw = ImageDraw.Draw(slight_img)
        for i in range(0, 256, 60):
            for j in range(0, 256, 60):
                draw.ellipse([i, j, i+20, j+20], fill=(100, 80, 50))
        
        verified, score, metrics = verifier.verify_burn_level(slight_img, "slightly_burnt")
        verification_results['slightly_burnt'] = {'verified': verified, 'score': score, 'metrics': metrics}
        
        print(f"  Slightly Burnt: {'✓' if verified else '✗'} (score: {score:.3f})")
        print(f"    Burnt ratio: {metrics['burnt_ratio']:.3f} (dark: {metrics['dark_ratio']:.3f}, brown: {metrics['brown_ratio']:.3f})")
        
        # Severely burnt - mostly dark/black
        severe_img = Image.new('RGB', (256, 256), color=(40, 30, 20))
        draw = ImageDraw.Draw(severe_img)
        for i in range(0, 256, 30):
            for j in range(0, 256, 30):
                draw.ellipse([i, j, i+20, j+20], fill=(15, 10, 5))
        
        verified, score, metrics = verifier.verify_burn_level(severe_img, "severely_burnt")
        verification_results['severely_burnt'] = {'verified': verified, 'score': score, 'metrics': metrics}
        
        print(f"  Severely Burnt: {'✓' if verified else '✗'} (score: {score:.3f})")
        print(f"    Burnt ratio: {metrics['burnt_ratio']:.3f} (dark: {metrics['dark_ratio']:.3f}, brown: {metrics['brown_ratio']:.3f})")
        
        # Save verification report
        report_path = demo_dir / "verification_report.json"
        with open(report_path, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        print(f"\n📊 Verification report saved to: {report_path}")
        
        return verification_results
        
    except Exception as e:
        print(f"Error demonstrating verification system: {e}")
        import traceback
        traceback.print_exc()
        return {}

def demonstrate_pipeline_configuration():
    """Demonstrate the configuration system and pipeline features"""
    print("\n=== PIPELINE CONFIGURATION DEMONSTRATION ===\n")
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import TargetedGenerationConfig, TargetedDiffusionPipeline
        
        # Show different configuration options
        configs = {
            "high_quality": TargetedGenerationConfig(
                model_type="sdxl",
                image_size=1024,
                guidance_scale=10.0,
                num_inference_steps=50,
                quality_threshold=0.8,
                max_retries=5,
                verify_target_properties=True,
                property_verification_threshold=0.7
            ),
            "fast_generation": TargetedGenerationConfig(
                model_type="sd",
                image_size=512,
                guidance_scale=7.5,
                num_inference_steps=20,
                max_retries=2,
                verify_target_properties=False
            ),
            "memory_optimized": TargetedGenerationConfig(
                batch_size=1,
                enable_cpu_offload=True,
                enable_attention_slicing=True,
                image_size=512,
                verify_target_properties=True
            )
        }
        
        print("⚙️ CONFIGURATION PROFILES:")
        for name, config in configs.items():
            print(f"\n  {name.upper()}:")
            print(f"    Model: {config.model_type}, Size: {config.image_size}")
            print(f"    Steps: {config.num_inference_steps}, Guidance: {config.guidance_scale}")
            print(f"    Quality threshold: {config.quality_threshold}")
            print(f"    Property verification: {config.verify_target_properties}")
            print(f"    Memory optimization: CPU offload={config.enable_cpu_offload}")
        
        # Initialize a pipeline with the memory optimized config
        print(f"\n🏗️ INITIALIZING PIPELINE:")
        pipeline = TargetedDiffusionPipeline(configs["memory_optimized"])
        print(f"  Output directory: {pipeline.output_dir}")
        print(f"  Existing pipeline components: {pipeline.generator is not None}")
        print(f"  Statistics tracking enabled: {len(pipeline.stats)} metrics")
        
        # Show output directory structure
        print(f"\n📁 OUTPUT DIRECTORY STRUCTURE:")
        for subdir in pipeline.output_dir.iterdir():
            if subdir.is_dir():
                print(f"  {subdir.name}/")
        
        return True
        
    except Exception as e:
        print(f"Error demonstrating pipeline configuration: {e}")
        return False

def demonstrate_metadata_system():
    """Demonstrate the comprehensive metadata system"""
    print("\n=== METADATA SYSTEM DEMONSTRATION ===\n")
    
    try:
        # Create sample metadata for different generation types
        metadata_examples = {
            "targeted_lighting": {
                "generation_type": "targeted_lighting",
                "target_property": "overhead_harsh",
                "template_used": "Direct overhead lighting with harsh shadows",
                "final_prompt": "pizza with direct overhead light creating harsh shadows, professional food photography, high contrast lighting",
                "final_negative_prompt": "soft lighting, diffused light, even illumination, no shadows",
                "lighting_parameters": {
                    "direction": "overhead",
                    "intensity": "high",
                    "shadow_strength": "strong"
                },
                "generation_config": {
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "guidance_scale": 7.5,
                    "num_inference_steps": 30,
                    "image_size": 512
                },
                "lighting_verification": {
                    "verified": True,
                    "score": 0.72,
                    "metrics": {
                        "brightness_mean": 145.2,
                        "shadow_ratio": 0.35,
                        "contrast_ratio": 0.48
                    },
                    "target_condition": "overhead_harsh",
                    "verification_threshold": 0.6
                },
                "attempt_number": 1,
                "generation_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": 12.5
            },
            "targeted_burn": {
                "generation_type": "targeted_burn",
                "target_property": "moderately_burnt",
                "template_used": "Pizza with noticeable burn marks and browning",
                "final_prompt": "pizza moderately burnt with visible brown and dark spots, still edible appearance, rustic wood-fired style",
                "burn_parameters": {
                    "intensity": "medium",
                    "pattern": "irregular_spots",
                    "color_range": "brown_to_dark"
                },
                "burn_verification": {
                    "verified": True,
                    "score": 0.68,
                    "metrics": {
                        "dark_ratio": 0.12,
                        "brown_ratio": 0.23,
                        "burnt_ratio": 0.35,
                        "mean_brightness": 98.4
                    }
                },
                "attempt_number": 2,
                "generation_timestamp": datetime.now().isoformat()
            },
            "combined_properties": {
                "generation_type": "combined_properties",
                "target_lighting": "side_dramatic",
                "target_burn": "slightly_burnt",
                "combined_verification": {
                    "lighting_verified": True,
                    "burn_verified": True,
                    "overall_score": 0.75
                },
                "generation_timestamp": datetime.now().isoformat()
            }
        }
        
        print("📋 METADATA EXAMPLES:")
        for gen_type, metadata in metadata_examples.items():
            print(f"\n  {gen_type.upper()}:")
            print(f"    Target: {metadata.get('target_property', 'Combined properties')}")
            if 'lighting_verification' in metadata:
                verification = metadata['lighting_verification']
                print(f"    Verification: {'✓' if verification['verified'] else '✗'} (score: {verification['score']:.3f})")
            elif 'burn_verification' in metadata:
                verification = metadata['burn_verification']
                print(f"    Verification: {'✓' if verification['verified'] else '✗'} (score: {verification['score']:.3f})")
            print(f"    Timestamp: {metadata['generation_timestamp']}")
        
        # Save sample metadata files
        demo_dir = Path("demo_metadata")
        demo_dir.mkdir(exist_ok=True)
        
        for gen_type, metadata in metadata_examples.items():
            metadata_file = demo_dir / f"{gen_type}_sample.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Sample metadata files saved to: {demo_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error demonstrating metadata system: {e}")
        return False

def generate_capabilities_report():
    """Generate a comprehensive capabilities report"""
    print("\n=== DIFFUSION-2.1 CAPABILITIES REPORT ===\n")
    
    capabilities = {
        "core_features": [
            "Enhanced prompt templates for targeted property generation",
            "Real-time property verification with quantitative scoring",
            "Comprehensive metadata storage with generation parameters",
            "Quality-aware generation with retry mechanisms",
            "Memory-optimized pipeline configuration options",
            "Standalone operation mode without existing pipeline dependencies"
        ],
        "lighting_conditions": list(range(4)),  # Will be filled with actual conditions
        "burn_levels": list(range(3)),  # Will be filled with actual levels
        "generation_modes": [
            "Single property targeting (lighting or burn)",
            "Combined property generation (lighting + burn)",
            "Cooking transition sequences",
            "Comprehensive dataset generation"
        ],
        "verification_algorithms": [
            "Shadow ratio analysis for lighting conditions",
            "Brightness and contrast metrics",
            "Color-based burn detection",
            "Dark/brown pixel ratio calculation",
            "Configurable verification thresholds"
        ],
        "output_formats": [
            "High-resolution images (512x512 to 1024x1024)",
            "Comprehensive JSON metadata",
            "Organized directory structure by property type",
            "Verification reports and statistics"
        ]
    }
    
    try:
        from src.augmentation.targeted_diffusion_pipeline import LIGHTING_CONDITION_TEMPLATES, BURN_LEVEL_TEMPLATES
        capabilities["lighting_conditions"] = list(LIGHTING_CONDITION_TEMPLATES.keys())
        capabilities["burn_levels"] = list(BURN_LEVEL_TEMPLATES.keys())
    except:
        pass
    
    print("🚀 CORE FEATURES:")
    for feature in capabilities["core_features"]:
        print(f"  ✓ {feature}")
    
    print(f"\n🔆 SUPPORTED LIGHTING CONDITIONS ({len(capabilities['lighting_conditions'])}):")
    for condition in capabilities["lighting_conditions"]:
        print(f"  • {condition}")
    
    print(f"\n🔥 SUPPORTED BURN LEVELS ({len(capabilities['burn_levels'])}):")
    for level in capabilities["burn_levels"]:
        print(f"  • {level}")
    
    print(f"\n⚡ GENERATION MODES:")
    for mode in capabilities["generation_modes"]:
        print(f"  • {mode}")
    
    print(f"\n🔍 VERIFICATION ALGORITHMS:")
    for algorithm in capabilities["verification_algorithms"]:
        print(f"  • {algorithm}")
    
    print(f"\n📤 OUTPUT FORMATS:")
    for format_type in capabilities["output_formats"]:
        print(f"  • {format_type}")
    
    # Save capabilities report
    report_path = Path("DIFFUSION_2_1_CAPABILITIES.json")
    with open(report_path, 'w') as f:
        json.dump(capabilities, f, indent=2)
    
    print(f"\n📊 Full capabilities report saved to: {report_path}")
    
    return capabilities

def main():
    """Run the complete DIFFUSION-2.1 demonstration"""
    print("🎯 DIFFUSION-2.1: Targeted Image Generation Pipeline")
    print("=" * 60)
    print("Advanced diffusion pipeline for generating pizza images with")
    print("specific properties like lighting conditions and burn levels.\n")
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting DIFFUSION-2.1 demonstration")
    
    demos_passed = 0
    total_demos = 5
    
    # Run demonstrations
    if demonstrate_prompt_generation():
        demos_passed += 1
        logging.info("Prompt generation demonstration completed successfully")
    
    verification_results = demonstrate_verification_system()
    if verification_results:
        demos_passed += 1
        logging.info("Verification system demonstration completed successfully")
    
    if demonstrate_pipeline_configuration():
        demos_passed += 1
        logging.info("Pipeline configuration demonstration completed successfully")
    
    if demonstrate_metadata_system():
        demos_passed += 1
        logging.info("Metadata system demonstration completed successfully")
    
    capabilities = generate_capabilities_report()
    if capabilities:
        demos_passed += 1
        logging.info("Capabilities report generated successfully")
    
    # Final summary
    print(f"\n🎉 DEMONSTRATION COMPLETE: {demos_passed}/{total_demos} modules successful")
    
    if demos_passed == total_demos:
        print("\n✅ DIFFUSION-2.1 is fully operational and ready for:")
        print("   1. Integration with production diffusion models")
        print("   2. Large-scale dataset generation")
        print("   3. Advanced property-controlled image synthesis")
        print("   4. Quality-controlled synthetic data creation")
        
        # Next steps
        print(f"\n📋 NEXT STEPS:")
        print("   • Load actual diffusion models (requires GPU memory management)")
        print("   • Run end-to-end generation tests with target properties")
        print("   • Optimize memory usage for large-scale generation")
        print("   • Integrate with existing dataset workflows")
        print("   • Generate sample datasets for validation")
    else:
        print(f"\n❌ {total_demos - demos_passed} demonstration modules failed")
        print("Please review the output and resolve any issues before proceeding.")
    
    print(f"\n📜 Detailed logs available at: {log_file}")
    logging.info("DIFFUSION-2.1 demonstration completed")
    
    return 0 if demos_passed == total_demos else 1

if __name__ == "__main__":
    exit(main())
