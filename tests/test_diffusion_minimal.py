#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Diffusion Test Script for DIFFUSION-1.1 Task

This script performs the minimal required test to complete DIFFUSION-1.1:
- Set up diffusion environment
- Load a model (sd-food option)
- Generate one test image
- Monitor and log VRAM usage
"""

import os
import sys
import time
import gc
import json
import psutil
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

def get_memory_info():
    """Get current memory usage information"""
    ram = psutil.virtual_memory()
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "system_ram_used_gb": round(ram.used / (1024**3), 2),
        "system_ram_percent": round(ram.percent, 1),
        "gpu_allocated_gb": 0,
        "gpu_cached_gb": 0,
        "gpu_max_allocated_gb": 0
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
            "gpu_cached_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
            "gpu_max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        })
    
    return info

def log_memory(stage, description, log_file, generation_time=None):
    """Log memory usage to file and console"""
    info = get_memory_info()
    
    time_info = f" (took {generation_time:.1f}s)" if generation_time else ""
    log_line = f"[{info['timestamp']}] {stage}: {description}{time_info}"
    
    print(log_line)
    print(f"  System RAM: {info['system_ram_used_gb']}GB used ({info['system_ram_percent']}%)")
    print(f"  GPU_0: {info['gpu_allocated_gb']}GB allocated, {info['gpu_cached_gb']}GB cached (max: {info['gpu_max_allocated_gb']}GB)")
    print()
    
    with open(log_file, 'a') as f:
        f.write(log_line + "\n")
        f.write(f"  System RAM: {info['system_ram_used_gb']}GB used ({info['system_ram_percent']}%)\n")
        f.write(f"  GPU_0: {info['gpu_allocated_gb']}GB allocated, {info['gpu_cached_gb']}GB cached (max: {info['gpu_max_allocated_gb']}GB)\n")
        f.write("\n")
    
    return info

def main():
    """Run minimal diffusion test for DIFFUSION-1.1"""
    
    # Setup output directory
    output_dir = Path("output/diffusion_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "initial_vram_usage.log"
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write("VRAM Usage Monitor - Diffusion Model Analysis\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        
        # System info
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            system_info["gpu_name"] = props.name
            system_info["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        
        f.write(f"System Info: {json.dumps(system_info, indent=2)}\n")
        f.write("=" * 80 + "\n\n")
    
    print("🎨 Minimal Diffusion Model Test - DIFFUSION-1.1")
    print("=" * 50)
    print("🎯 Goal: Set up basic diffusion environment and analyze VRAM usage")
    print()
    
    # Step 1: Initial state
    log_memory("start", "Initial memory state", log_file)
    
    # Step 2: Import libraries
    print("📦 Libraries already imported...")
    log_memory("imports", "After importing diffusion libraries", log_file)
    
    # Step 3: Load model (sd-food option = standard SD 1.5 for food)
    print("🔄 Loading model: stable-diffusion-v1-5 (sd-food option)...")
    log_memory("pre_model_load", "Before loading model stable-diffusion-v1-5", log_file)
    
    start_time = time.time()
    
    try:
        # Load with memory optimizations for sd-food
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # Memory optimization
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            # Enable memory efficient features
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        
        load_time = time.time() - start_time
        log_memory("model_loaded", f"After loading model (took {load_time:.1f}s)", log_file, load_time)
        
        print(f"✅ Model loaded successfully in {load_time:.1f}s")
        
        # Step 4: Generate test image
        print("🖼️  Generating test image...")
        log_memory("pre_generation_0", "Before generating image 1", log_file)
        
        gen_start = time.time()
        
        # Generate with pizza prompt
        prompt = "a delicious pizza with tomato sauce and cheese, food photography, high quality"
        
        with torch.no_grad():
            image = pipe(
                prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator(device="cuda").manual_seed(42) if torch.cuda.is_available() else None
            ).images[0]
        
        gen_time = time.time() - gen_start
        log_memory("post_generation_0", f"After generating image 1 (took {gen_time:.1f}s)", log_file, gen_time)
        
        # Save the generated image
        image_path = output_dir / "test_image_1_sd-food_512px.png"
        image.save(image_path)
        
        print(f"✅ Image generated and saved: {image_path}")
        print(f"⏱️  Generation time: {gen_time:.1f}s")
        
        # Step 5: Final analysis
        final_info = get_memory_info()
        
        print("\n" + "="*50)
        print("📊 VRAM ANALYSIS SUMMARY")
        print("="*50)
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"🖥️  GPU: {props.name}")
            print(f"💾 Total VRAM: {round(props.total_memory / (1024**3), 2)} GB")
            print(f"🚀 Peak VRAM used: {final_info['gpu_max_allocated_gb']:.3f} GB")
            print(f"🔢 Current RAM usage: {final_info['system_ram_used_gb']:.2f} GB")
            print(f"⏱️  Model load time: {load_time:.1f}s")
            print(f"⏱️  Image generation time: {gen_time:.1f}s")
            
            # Hardware assessment
            peak_vram = final_info['gpu_max_allocated_gb']
            total_vram = round(props.total_memory / (1024**3), 2)
            
            print(f"\n🎯 HARDWARE REQUIREMENTS:")
            print(f"   - Model: sd-food (Stable Diffusion 1.5)")
            print(f"   - Image size: 512x512")
            print(f"   - Peak VRAM usage: {peak_vram:.1f} GB")
            print(f"   - GPU capacity: {total_vram} GB")
            
            if peak_vram <= total_vram * 0.8:
                print(f"   - Status: ✅ SUCCESS - Sufficient VRAM")
            else:
                print(f"   - Status: ⚠️  MARGINAL - Close to limit")
        
        # Save final analysis
        analysis = {
            "task": "DIFFUSION-1.1",
            "model": "sd-food",
            "system_info": system_info,
            "peak_vram_gb": final_info['gpu_max_allocated_gb'],
            "peak_ram_gb": final_info['system_ram_used_gb'],
            "model_load_time_s": load_time,
            "generation_time_s": gen_time,
            "generated_image": str(image_path),
            "success": True
        }
        
        with open(output_dir / "initial_vram_usage.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✅ DIFFUSION-1.1 TASK COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Log file: {log_file}")
        print(f"🖼️  Generated image: {image_path}")
        
    except Exception as e:
        print(f"❌ Error during model loading or generation: {e}")
        log_memory("error", f"Error occurred: {e}", log_file)
        
        # Save error analysis
        analysis = {
            "task": "DIFFUSION-1.1",
            "model": "sd-food",
            "error": str(e),
            "success": False
        }
        
        with open(output_dir / "initial_vram_usage.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
