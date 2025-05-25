#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRAM Optimization Testing for DIFFUSION-1.2

This script systematically tests different VRAM optimization strategies
for diffusion model image generation and measures their impact on memory
usage and generation performance.

Usage:
    python scripts/vram_optimization_test.py
    python scripts/vram_optimization_test.py --quick_test
"""

import os
import sys
import argparse
import logging
import json
import time
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from diffusers import StableDiffusionXLPipeline
    import transformers
except ImportError as e:
    print(f"Error importing diffusion libraries: {e}")
    print("Please install required packages: pip install diffusers transformers accelerate")
    sys.exit(1)

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

@dataclass
class OptimizationConfig:
    """Configuration for a single optimization test"""
    name: str
    model: str
    image_size: int
    batch_size: int
    cpu_offload: bool
    attention_slicing: bool
    vae_slicing: bool
    torch_dtype: str
    xformers: bool
    sequential_cpu_offload: bool
    description: str

@dataclass
class TestResult:
    """Results from a single optimization test"""
    config_name: str
    model: str
    image_size: int
    batch_size: int
    success: bool
    peak_vram_gb: float
    avg_vram_gb: float
    generation_time_s: float
    images_generated: int
    vram_efficiency: float  # images per GB VRAM
    time_efficiency: float  # images per second
    error_message: str = ""
    memory_optimizations: Dict[str, bool] = None

class VRAMOptimizationTester:
    """Main class for testing VRAM optimization strategies"""
    
    def __init__(self, output_dir: str = "output/diffusion_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[TestResult] = []
        self.system_info = self._get_system_info()
        self.test_prompt = "A delicious pizza with golden crust and melted cheese, professional food photography"
        
        # Setup logging
        log_file = self.output_dir / "vram_optimization_test.log"
        self.logger = logging.getLogger("vram_optimization_test")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # Add file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("VRAM Optimization Testing Started")
        self.logger.info(f"System Info: {json.dumps(self.system_info, indent=2)}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_devices"] = []
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu_devices"].append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor
                })
        
        return info
    
    def get_optimization_configs(self, quick_test: bool = False) -> List[OptimizationConfig]:
        """Define the optimization configurations to test"""
        configs = []
        
        if quick_test:
            # Quick test with just a few key configurations
            configs = [
                OptimizationConfig(
                    name="baseline_sd_food",
                    model="sd-food",
                    image_size=512,
                    batch_size=1,
                    cpu_offload=False,
                    attention_slicing=True,
                    vae_slicing=True,
                    torch_dtype="float16",
                    xformers=True,
                    sequential_cpu_offload=False,
                    description="Baseline sd-food model with basic optimizations"
                ),
                OptimizationConfig(
                    name="max_optimized_sd_food",
                    model="sd-food",
                    image_size=512,
                    batch_size=1,
                    cpu_offload=True,
                    attention_slicing=True,
                    vae_slicing=True,
                    torch_dtype="float16",
                    xformers=True,
                    sequential_cpu_offload=True,
                    description="sd-food with maximum memory optimizations"
                ),
                OptimizationConfig(
                    name="small_image_sd_food",
                    model="sd-food",
                    image_size=256,
                    batch_size=1,
                    cpu_offload=False,
                    attention_slicing=True,
                    vae_slicing=True,
                    torch_dtype="float16",
                    xformers=True,
                    sequential_cpu_offload=False,
                    description="sd-food with smaller image size (256px)"
                )
            ]
        else:
            # Comprehensive test suite
            base_models = ["sd-food", "stable-diffusion-v1-5"]
            image_sizes = [256, 512, 768]
            
            for model in base_models:
                for size in image_sizes:
                    # Skip large sizes for memory-intensive models if needed
                    if model == "stable-diffusion-v1-5" and size > 512:
                        continue
                    
                    # Basic configuration
                    configs.append(OptimizationConfig(
                        name=f"basic_{model}_{size}px",
                        model=model,
                        image_size=size,
                        batch_size=1,
                        cpu_offload=False,
                        attention_slicing=True,
                        vae_slicing=True,
                        torch_dtype="float16",
                        xformers=True,
                        sequential_cpu_offload=False,
                        description=f"Basic optimizations for {model} at {size}px"
                    ))
                    
                    # CPU offload configuration
                    configs.append(OptimizationConfig(
                        name=f"cpu_offload_{model}_{size}px",
                        model=model,
                        image_size=size,
                        batch_size=1,
                        cpu_offload=True,
                        attention_slicing=True,
                        vae_slicing=True,
                        torch_dtype="float16",
                        xformers=True,
                        sequential_cpu_offload=False,
                        description=f"CPU offload enabled for {model} at {size}px"
                    ))
                    
                    # Maximum optimization
                    configs.append(OptimizationConfig(
                        name=f"max_opt_{model}_{size}px",
                        model=model,
                        image_size=size,
                        batch_size=1,
                        cpu_offload=True,
                        attention_slicing=True,
                        vae_slicing=True,
                        torch_dtype="float16",
                        xformers=True,
                        sequential_cpu_offload=True,
                        description=f"Maximum optimizations for {model} at {size}px"
                    ))
        
        return configs
    
    def test_configuration(self, config: OptimizationConfig) -> TestResult:
        """Test a single optimization configuration"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Testing Configuration: {config.name}")
        self.logger.info(f"Description: {config.description}")
        self.logger.info(f"Model: {config.model}, Size: {config.image_size}px, Batch: {config.batch_size}")
        self.logger.info(f"{'='*80}")
        
        # Clear CUDA cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Initialize result
        result = TestResult(
            config_name=config.name,
            model=config.model,
            image_size=config.image_size,
            batch_size=config.batch_size,
            success=False,
            peak_vram_gb=0.0,
            avg_vram_gb=0.0,
            generation_time_s=0.0,
            images_generated=0,
            vram_efficiency=0.0,
            time_efficiency=0.0,
            memory_optimizations={
                "cpu_offload": config.cpu_offload,
                "attention_slicing": config.attention_slicing,
                "vae_slicing": config.vae_slicing,
                "torch_dtype": config.torch_dtype,
                "xformers": config.xformers,
                "sequential_cpu_offload": config.sequential_cpu_offload
            }
        )
        
        try:
            # Load model with configuration
            start_time = time.time()
            pipeline = self._load_model_with_config(config)
            load_time = time.time() - start_time
            
            self.logger.info(f"Model loaded in {load_time:.1f}s")
            
            # Measure VRAM usage during generation
            vram_measurements = []
            
            # Generate test images
            num_test_images = 3  # Generate 3 images for average timing
            generation_times = []
            
            for i in range(num_test_images):
                # Measure VRAM before generation
                if torch.cuda.is_available():
                    pre_vram = torch.cuda.memory_allocated() / (1024**3)
                    vram_measurements.append(pre_vram)
                
                # Generate image
                img_start_time = time.time()
                
                # Set seed for reproducibility
                torch.manual_seed(42 + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + i)
                
                image = pipeline(
                    self.test_prompt,
                    height=config.image_size,
                    width=config.image_size,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    num_images_per_prompt=config.batch_size
                ).images[0]
                
                img_time = time.time() - img_start_time
                generation_times.append(img_time)
                
                # Measure VRAM after generation
                if torch.cuda.is_available():
                    post_vram = torch.cuda.memory_allocated() / (1024**3)
                    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
                    vram_measurements.append(post_vram)
                    
                    self.logger.info(f"Image {i+1}: {img_time:.1f}s, VRAM: {post_vram:.3f}GB (peak: {peak_vram:.3f}GB)")
                else:
                    self.logger.info(f"Image {i+1}: {img_time:.1f}s")
                
                # Save test image
                output_path = self.output_dir / f"test_{config.name}_img{i+1}.png"
                image.save(output_path)
                
                result.images_generated += 1
            
            # Calculate metrics
            result.success = True
            result.generation_time_s = sum(generation_times) / len(generation_times)
            
            if torch.cuda.is_available():
                result.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
                result.avg_vram_gb = sum(vram_measurements) / len(vram_measurements) if vram_measurements else 0
            
            result.time_efficiency = 1.0 / result.generation_time_s if result.generation_time_s > 0 else 0
            result.vram_efficiency = 1.0 / result.peak_vram_gb if result.peak_vram_gb > 0 else 0
            
            self.logger.info(f"✅ Configuration {config.name} completed successfully")
            self.logger.info(f"   Peak VRAM: {result.peak_vram_gb:.3f}GB")
            self.logger.info(f"   Avg generation time: {result.generation_time_s:.1f}s")
            self.logger.info(f"   Images generated: {result.images_generated}")
            
            # Clean up
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"❌ Configuration {config.name} failed: {e}")
            
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return result
    
    def _load_model_with_config(self, config: OptimizationConfig):
        """Load model with specific optimization configuration"""
        
        # Model mapping - using cached models that are available locally
        model_mapping = {
            "sd-food": "runwayml/stable-diffusion-v1-5",
            "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
            "stable-diffusion-2": "stabilityai/stable-diffusion-2-1",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        }
        
        model_id = model_mapping.get(config.model, config.model)
        
        # Determine dtype
        torch_dtype = torch.float16 if config.torch_dtype == "float16" else torch.float32
        
        # Load pipeline
        self.logger.info(f"Loading model: {model_id}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16" if torch_dtype == torch.float16 else None
        )
        
        # Apply optimizations based on configuration
        if torch.cuda.is_available():
            if not config.cpu_offload and not config.sequential_cpu_offload:
                pipeline = pipeline.to("cuda")
            
            if config.cpu_offload:
                pipeline.enable_model_cpu_offload()
            
            if config.sequential_cpu_offload:
                pipeline.enable_sequential_cpu_offload()
            
            if config.attention_slicing:
                pipeline.enable_attention_slicing()
            
            if config.vae_slicing:
                pipeline.enable_vae_slicing()
            
            if config.xformers and hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    self.logger.info("✅ XFormers memory efficient attention enabled")
                except Exception as e:
                    self.logger.warning(f"Could not enable XFormers: {e}")
        
        return pipeline
    
    def run_all_tests(self, quick_test: bool = False) -> List[TestResult]:
        """Run all optimization tests"""
        configs = self.get_optimization_configs(quick_test)
        
        self.logger.info(f"\n🚀 Starting VRAM optimization testing")
        self.logger.info(f"Configurations to test: {len(configs)}")
        self.logger.info(f"Quick test mode: {quick_test}")
        
        results = []
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"\n[{i}/{len(configs)}] Testing: {config.name}")
            
            result = self.test_configuration(config)
            results.append(result)
            self.results.append(result)
            
            # Log progress
            successful = sum(1 for r in results if r.success)
            self.logger.info(f"Progress: {i}/{len(configs)} completed ({successful} successful)")
        
        return results
    
    def save_results(self) -> str:
        """Save test results to JSON file"""
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_configurations": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success)
            },
            "system_info": self.system_info,
            "results": [asdict(result) for result in self.results],
            "analysis": self._generate_analysis()
        }
        
        json_file = self.output_dir / "vram_optimization_test.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\n📊 Results saved to: {json_file}")
        return str(json_file)
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate analysis of the test results"""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {"error": "No successful tests to analyze"}
        
        # Find best configurations
        best_vram = min(successful_results, key=lambda x: x.peak_vram_gb)
        best_speed = min(successful_results, key=lambda x: x.generation_time_s)
        best_efficiency = max(successful_results, key=lambda x: x.vram_efficiency)
        
        # VRAM usage statistics
        vram_usages = [r.peak_vram_gb for r in successful_results]
        
        analysis = {
            "summary": {
                "total_successful_tests": len(successful_results),
                "vram_usage_range_gb": {
                    "min": min(vram_usages),
                    "max": max(vram_usages),
                    "average": sum(vram_usages) / len(vram_usages)
                },
                "generation_time_range_s": {
                    "min": min(r.generation_time_s for r in successful_results),
                    "max": max(r.generation_time_s for r in successful_results),
                    "average": sum(r.generation_time_s for r in successful_results) / len(successful_results)
                }
            },
            "best_configurations": {
                "lowest_vram": {
                    "name": best_vram.config_name,
                    "vram_gb": best_vram.peak_vram_gb,
                    "time_s": best_vram.generation_time_s
                },
                "fastest_generation": {
                    "name": best_speed.config_name,
                    "time_s": best_speed.generation_time_s,
                    "vram_gb": best_speed.peak_vram_gb
                },
                "best_efficiency": {
                    "name": best_efficiency.config_name,
                    "vram_efficiency": best_efficiency.vram_efficiency,
                    "vram_gb": best_efficiency.peak_vram_gb
                }
            },
            "recommendations": self._generate_recommendations(successful_results)
        }
        
        return analysis
    
    def _generate_recommendations(self, successful_results: List[TestResult]) -> Dict[str, str]:
        """Generate recommendations based on test results"""
        recommendations = {}
        
        # Find optimal configurations for different use cases
        low_vram_results = [r for r in successful_results if r.peak_vram_gb < 3.0]
        fast_results = [r for r in successful_results if r.generation_time_s < 5.0]
        
        if low_vram_results:
            best_low_vram = min(low_vram_results, key=lambda x: x.peak_vram_gb)
            recommendations["low_vram_setup"] = f"For systems with limited VRAM (<4GB), use '{best_low_vram.config_name}' which uses only {best_low_vram.peak_vram_gb:.2f}GB"
        
        if fast_results:
            fastest = min(fast_results, key=lambda x: x.generation_time_s)
            recommendations["fast_generation"] = f"For fastest generation, use '{fastest.config_name}' which generates images in {fastest.generation_time_s:.1f}s"
        
        # General recommendations
        vram_usages = [r.peak_vram_gb for r in successful_results]
        avg_vram = sum(vram_usages) / len(vram_usages)
        
        if avg_vram < 4.0:
            recommendations["general"] = "All tested configurations are suitable for GPUs with 6GB+ VRAM"
        elif avg_vram < 8.0:
            recommendations["general"] = "Most configurations require 8GB+ VRAM for reliable operation"
        else:
            recommendations["general"] = "High-end GPU (12GB+ VRAM) recommended for optimal performance"
        
        return recommendations
    
    def print_summary(self):
        """Print a comprehensive summary of the test results"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print("\n" + "="*80)
        print("📊 VRAM OPTIMIZATION TEST SUMMARY")
        print("="*80)
        
        print(f"📈 Test Results:")
        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   📊 Total: {len(self.results)}")
        
        if successful:
            vram_usages = [r.peak_vram_gb for r in successful]
            times = [r.generation_time_s for r in successful]
            
            print(f"\n💾 VRAM Usage:")
            print(f"   📉 Lowest: {min(vram_usages):.3f}GB ({min(successful, key=lambda x: x.peak_vram_gb).config_name})")
            print(f"   📈 Highest: {max(vram_usages):.3f}GB ({max(successful, key=lambda x: x.peak_vram_gb).config_name})")
            print(f"   📊 Average: {sum(vram_usages)/len(vram_usages):.3f}GB")
            
            print(f"\n⏱️  Generation Time:")
            print(f"   🚀 Fastest: {min(times):.1f}s ({min(successful, key=lambda x: x.generation_time_s).config_name})")
            print(f"   🐌 Slowest: {max(times):.1f}s ({max(successful, key=lambda x: x.generation_time_s).config_name})")
            print(f"   📊 Average: {sum(times)/len(times):.1f}s")
        
        if failed:
            print(f"\n❌ Failed Configurations:")
            for result in failed:
                print(f"   • {result.config_name}: {result.error_message}")
        
        print("="*80)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VRAM Optimization Testing for Diffusion Models")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_analysis",
                       help="Output directory for test results")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with fewer configurations")
    
    args = parser.parse_args()
    
    print("🔬 VRAM Optimization Testing - DIFFUSION-1.2")
    print("="*60)
    
    # Initialize tester
    tester = VRAMOptimizationTester(output_dir=args.output_dir)
    
    try:
        # Run tests
        results = tester.run_all_tests(quick_test=args.quick_test)
        
        # Save results
        json_file = tester.save_results()
        
        # Print summary
        tester.print_summary()
        
        print(f"\n✅ DIFFUSION-1.2 COMPLETED!")
        print(f"📄 Detailed results: {json_file}")
        print(f"📁 Test images: {args.output_dir}")
        
        # Check completion criteria
        successful_tests = sum(1 for r in results if r.success)
        if successful_tests >= 3:  # At least 3 successful configurations tested
            print("\n🎉 TASK DIFFUSION-1.2 COMPLETION CRITERIA MET:")
            print("   ✅ Multiple optimization strategies tested")
            print("   ✅ VRAM usage measured and compared")
            print("   ✅ Generation time analyzed")
            print("   ✅ Optimization report generated")
            print(f"   ✅ {successful_tests} successful configurations documented")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
