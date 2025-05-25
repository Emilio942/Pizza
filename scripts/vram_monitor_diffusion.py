#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced VRAM Monitor for Diffusion Model Generation

This improved script provides better error handling, more detailed monitoring,
and enhanced performance optimization for diffusion model VRAM analysis.

Usage:
    python scripts/vram_monitor_diffusion.py --model sd-food --image_size 512 --num_images 1
"""

import os
import sys
import argparse
import logging
import json
import time
import gc
import psutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/diffusion_analysis/monitor.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Data class for memory measurements"""
    timestamp: str
    stage: str
    description: str
    system_memory: Dict[str, Union[float, int]]
    gpu_memory: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'stage': self.stage,
            'description': self.description,
            'system_memory': self.system_memory,
            'gpu_memory': self.gpu_memory or {}
        }

class EnhancedVRAMMonitor:
    """Enhanced VRAM monitoring with real-time tracking and better analysis"""
    
    def __init__(self, log_file: str = "output/diffusion_analysis/enhanced_vram_usage.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.measurements: List[MemorySnapshot] = []
        self.system_info = self._get_comprehensive_system_info()
        self.monitoring_thread = None
        self.monitoring_active = False
        self.real_time_data = []
        
        self._initialize_log_file()
        logger.info("Enhanced VRAM Monitor initialized")
    
    def _get_comprehensive_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system and GPU information"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": os.name,
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:3]  # First 3 paths only
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "backends": {
                    "cudnn": torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
                    "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
                }
            }
        }
        
        if torch.cuda.is_available():
            info["cuda"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multiprocessor_count,
                    "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor
                }
                info["cuda"]["devices"].append(device_info)
        
        return info
    
    def _initialize_log_file(self):
        """Initialize log file with comprehensive header"""
        with open(self.log_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("ENHANCED VRAM USAGE MONITOR - DIFFUSION MODEL ANALYSIS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Session Started: {datetime.now().isoformat()}\n")
            f.write(f"Log File: {self.log_file}\n\n")
            
            f.write("SYSTEM INFORMATION:\n")
            f.write("-" * 50 + "\n")
            f.write(json.dumps(self.system_info, indent=2))
            f.write("\n\n" + "=" * 100 + "\n\n")
    
    def start_continuous_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring in background thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.real_time_data = []
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    snapshot = self._create_memory_snapshot("continuous", "Real-time monitoring")
                    self.real_time_data.append(snapshot)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in continuous monitoring: {e}")
                    break
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started continuous monitoring (interval: {interval}s)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous memory monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            logger.info(f"Stopped continuous monitoring. Collected {len(self.real_time_data)} data points")
    
    def _create_memory_snapshot(self, stage: str, description: str = "") -> MemorySnapshot:
        """Create a comprehensive memory snapshot"""
        # System memory
        vm = psutil.virtual_memory()
        system_memory = {
            "used_gb": round(vm.used / (1024**3), 3),
            "available_gb": round(vm.available / (1024**3), 3),
            "percent": round(vm.percent, 1),
            "cached_gb": round(vm.cached / (1024**3), 3) if hasattr(vm, 'cached') else 0,
            "buffers_gb": round(vm.buffers / (1024**3), 3) if hasattr(vm, 'buffers') else 0
        }
        
        # GPU memory
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(i)  # Ensure accurate measurements
                    
                    # Get memory info
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    max_allocated = torch.cuda.max_memory_allocated(i)
                    max_reserved = torch.cuda.max_memory_reserved(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    
                    gpu_memory[f"gpu_{i}"] = {
                        "allocated_gb": round(allocated / (1024**3), 4),
                        "reserved_gb": round(reserved / (1024**3), 4),
                        "max_allocated_gb": round(max_allocated / (1024**3), 4),
                        "max_reserved_gb": round(max_reserved / (1024**3), 4),
                        "total_gb": round(total / (1024**3), 2),
                        "utilization_percent": round((allocated / total) * 100, 2),
                        "efficiency_percent": round((allocated / reserved * 100) if reserved > 0 else 0, 1)
                    }
                except Exception as e:
                    logger.warning(f"Could not get GPU {i} memory info: {e}")
                    gpu_memory[f"gpu_{i}"] = {"error": str(e)}
        
        return MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            description=description,
            system_memory=system_memory,
            gpu_memory=gpu_memory
        )
    
    def measure_memory(self, stage: str, description: str = "") -> MemorySnapshot:
        """Take a comprehensive memory measurement"""
        snapshot = self._create_memory_snapshot(stage, description)
        self.measurements.append(snapshot)
        
        # Log to file
        self._log_measurement(snapshot)
        
        # Log to console
        self._console_log_measurement(snapshot)
        
        return snapshot
    
    def _log_measurement(self, snapshot: MemorySnapshot):
        """Log measurement to file"""
        with open(self.log_file, 'a') as f:
            f.write(f"[{snapshot.timestamp}] {snapshot.stage.upper()}: {snapshot.description}\n")
            
            # System memory
            sm = snapshot.system_memory
            f.write(f"  System RAM: {sm['used_gb']:.3f}GB used, {sm['available_gb']:.3f}GB available "
                   f"({sm['percent']:.1f}% utilization)\n")
            
            # GPU memory
            if snapshot.gpu_memory:
                for gpu_id, gpu_data in snapshot.gpu_memory.items():
                    if "error" not in gpu_data:
                        f.write(f"  {gpu_id.upper()}: {gpu_data['allocated_gb']:.4f}GB allocated, "
                               f"{gpu_data['reserved_gb']:.4f}GB reserved "
                               f"({gpu_data['utilization_percent']:.2f}% GPU utilization, "
                               f"{gpu_data['efficiency_percent']:.1f}% efficiency)\n")
                    else:
                        f.write(f"  {gpu_id.upper()}: Error - {gpu_data['error']}\n")
            
            f.write("\n")
    
    def _console_log_measurement(self, snapshot: MemorySnapshot):
        """Log key metrics to console"""
        logger.info(f"Memory checkpoint - {snapshot.stage}: {snapshot.description}")
        
        if snapshot.gpu_memory:
            for gpu_id, gpu_data in snapshot.gpu_memory.items():
                if "error" not in gpu_data:
                    logger.info(f"  {gpu_id.upper()}: {gpu_data['allocated_gb']:.3f}GB "
                               f"({gpu_data['utilization_percent']:.1f}%)")
    
    @contextmanager
    def memory_context(self, stage: str, description: str = ""):
        """Context manager for automatic before/after memory measurement"""
        self.measure_memory(f"{stage}_start", f"Before {description}")
        try:
            yield
        finally:
            self.measure_memory(f"{stage}_end", f"After {description}")
    
    def reset_peak_memory_stats(self):
        """Reset peak memory statistics for all devices"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.empty_cache()
            logger.info("Reset peak memory statistics for all GPUs")
    
    def force_garbage_collection(self):
        """Force comprehensive garbage collection"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Forced garbage collection")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        if not self.measurements:
            return {"error": "No measurements available"}
        
        analysis = {
            "session_info": {
                "start_time": self.measurements[0].timestamp,
                "end_time": self.measurements[-1].timestamp,
                "total_measurements": len(self.measurements),
                "continuous_data_points": len(self.real_time_data)
            }
        }
        
        # Calculate duration
        if len(self.measurements) > 1:
            start = datetime.fromisoformat(self.measurements[0].timestamp)
            end = datetime.fromisoformat(self.measurements[-1].timestamp)
            analysis["session_info"]["duration_seconds"] = (end - start).total_seconds()
        
        # System memory analysis
        sys_mem_data = [m.system_memory for m in self.measurements]
        analysis["system_memory"] = {
            "peak_usage_gb": max(m["used_gb"] for m in sys_mem_data),
            "min_available_gb": min(m["available_gb"] for m in sys_mem_data),
            "max_utilization_percent": max(m["percent"] for m in sys_mem_data),
            "average_utilization_percent": round(sum(m["percent"] for m in sys_mem_data) / len(sys_mem_data), 1)
        }
        
        # GPU memory analysis
        if torch.cuda.is_available() and self.measurements[0].gpu_memory:
            gpu_analysis = {}
            
            for i in range(torch.cuda.device_count()):
                gpu_key = f"gpu_{i}"
                gpu_measurements = []
                
                for m in self.measurements:
                    if m.gpu_memory and gpu_key in m.gpu_memory and "error" not in m.gpu_memory[gpu_key]:
                        gpu_measurements.append(m.gpu_memory[gpu_key])
                
                if gpu_measurements:
                    gpu_analysis[gpu_key] = {
                        "peak_allocated_gb": max(g["allocated_gb"] for g in gpu_measurements),
                        "peak_reserved_gb": max(g["reserved_gb"] for g in gpu_measurements),
                        "max_utilization_percent": max(g["utilization_percent"] for g in gpu_measurements),
                        "average_efficiency_percent": round(sum(g["efficiency_percent"] for g in gpu_measurements) / len(gpu_measurements), 1),
                        "total_capacity_gb": gpu_measurements[0]["total_gb"],
                        "final_allocated_gb": gpu_measurements[-1]["allocated_gb"],
                        "memory_leak_detected": gpu_measurements[-1]["allocated_gb"] > gpu_measurements[0]["allocated_gb"] + 0.1
                    }
            
            analysis["gpu_analysis"] = gpu_analysis
        
        return analysis
    
    def save_comprehensive_report(self, output_file: str = None):
        """Save comprehensive analysis report"""
        if output_file is None:
            output_file = self.log_file.with_suffix('.json')
        
        report = {
            "metadata": {
                "report_generated": datetime.now().isoformat(),
                "monitor_version": "enhanced_v2.0",
                "log_file": str(self.log_file)
            },
            "system_info": self.system_info,
            "measurements": [m.to_dict() for m in self.measurements],
            "continuous_monitoring": [m.to_dict() for m in self.real_time_data],
            "analysis": self.get_memory_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive VRAM analysis report saved to {output_file}")
        return report


def get_optimized_pipeline_config(device: torch.device, model_id: str) -> Dict[str, Any]:
    """Get optimized pipeline configuration based on available hardware"""
    config = {
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
        "use_safetensors": True,
        "safety_checker": None,
        "requires_safety_checker": False
    }
    
    # Add variant for CUDA with sufficient memory
    if device.type == "cuda":
        try:
            # Check available memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            if free_memory > 4 * (1024**3):  # More than 4GB available
                config["variant"] = "fp16"
        except:
            pass
    
    return config


def load_diffusion_model_with_monitoring(monitor: EnhancedVRAMMonitor, model_name: str, device: torch.device):
    """Load diffusion model with comprehensive monitoring"""
    
    # Model mapping
    model_mapping = {
        "sd-food": "runwayml/stable-diffusion-v1-5",
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "sd21": "stabilityai/stable-diffusion-2-1"
    }
    
    model_id = model_mapping.get(model_name, model_name)
    logger.info(f"Loading model: {model_id}")
    
    with monitor.memory_context("model_loading", f"Loading {model_id}"):
        try:
            from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline
            
            # Get optimized configuration
            config = get_optimized_pipeline_config(device, model_id)
            
            # Load appropriate pipeline
            if "xl" in model_id.lower():
                pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **config)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_id, **config)
            
            # Move to device
            pipe = pipe.to(device)
            
            # Apply memory optimizations
            optimizations_applied = []
            
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
                optimizations_applied.append("attention_slicing")
            
            if hasattr(pipe, 'enable_memory_efficient_attention'):
                try:
                    pipe.enable_memory_efficient_attention()
                    optimizations_applied.append("memory_efficient_attention")
                except:
                    pass
            
            if device.type == "cuda":
                try:
                    pipe.enable_model_cpu_offload()
                    optimizations_applied.append("cpu_offload")
                except:
                    try:
                        pipe.enable_sequential_cpu_offload()
                        optimizations_applied.append("sequential_cpu_offload")
                    except:
                        pass
            
            logger.info(f"Applied optimizations: {', '.join(optimizations_applied)}")
            return pipe, optimizations_applied
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise


def enhanced_diffusion_test(model_name: str = "sd-food", image_size: int = 512, 
                           num_images: int = 1, output_dir: str = "output/diffusion_analysis",
                           continuous_monitoring: bool = True):
    """Enhanced diffusion model test with comprehensive monitoring"""
    
    # Initialize enhanced monitor
    monitor = EnhancedVRAMMonitor()
    monitor.measure_memory("initialization", "Monitor initialized")
    
    try:
        # Start continuous monitoring if requested
        if continuous_monitoring:
            monitor.start_continuous_monitoring(interval=0.5)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Reset memory stats
        monitor.reset_peak_memory_stats()
        
        # Load model with monitoring
        pipe, optimizations = load_diffusion_model_with_monitoring(monitor, model_name, device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhanced prompts for testing
        enhanced_prompts = [
            "a gourmet pizza with melted mozzarella cheese, fresh basil leaves, and tomato sauce on a wooden cutting board, professional food photography, high quality, detailed",
            "freshly baked artisanal pizza with colorful vegetables, golden crust, warm restaurant lighting, appetizing presentation",
            "homemade margherita pizza with fresh ingredients, rustic kitchen setting, natural lighting, mouth-watering appearance"
        ]
        
        # Generate images with detailed monitoring
        generation_times = []
        
        for i in range(num_images):
            logger.info(f"Generating image {i+1}/{num_images}")
            
            prompt = enhanced_prompts[i % len(enhanced_prompts)]
            
            with monitor.memory_context(f"generation_{i}", f"Generating image {i+1}"):
                gen_start = time.time()
                
                # Enhanced generation parameters
                generation_params = {
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, deformed, ugly, distorted, watermark, text, signature",
                    "height": image_size,
                    "width": image_size,
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "generator": torch.Generator(device=device).manual_seed(42 + i)
                }
                
                # Add SDXL-specific parameters if needed
                if "xl" in str(type(pipe)).lower():
                    generation_params.update({
                        "denoising_end": 0.8,
                        "output_type": "pil"
                    })
                
                # Generate with memory monitoring
                with torch.no_grad():
                    image = pipe(**generation_params).images[0]
                
                gen_time = time.time() - gen_start
                generation_times.append(gen_time)
                
                logger.info(f"Generated image {i+1} in {gen_time:.2f}s")
            
            # Save image with metadata
            output_path = Path(output_dir) / f"enhanced_test_{i+1}_{model_name}_{image_size}px.png"
            
            # Add metadata to image
            metadata = {
                "prompt": prompt,
                "model": model_name,
                "size": f"{image_size}x{image_size}",
                "generation_time": f"{gen_time:.2f}s",
                "optimizations": optimizations
            }
            
            # Save with metadata
            image.save(output_path)
            
            # Save metadata separately
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved image and metadata to {output_path}")
            
            # Force cleanup between generations
            monitor.force_garbage_collection()
        
        # Final measurements
        monitor.measure_memory("generation_complete", "All images generated")
        
        # Stop continuous monitoring
        if continuous_monitoring:
            monitor.stop_continuous_monitoring()
        
        # Cleanup
        del pipe
        monitor.force_garbage_collection()
        monitor.measure_memory("cleanup_complete", "Pipeline cleaned up")
        
        # Generate comprehensive report
        report = monitor.save_comprehensive_report()
        
        # Print summary
        analysis = report["analysis"]
        logger.info("=" * 60)
        logger.info("ENHANCED VRAM ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        if "gpu_analysis" in analysis:
            for gpu_id, gpu_data in analysis["gpu_analysis"].items():
                logger.info(f"{gpu_id.upper()}:")
                logger.info(f"  Peak allocated: {gpu_data['peak_allocated_gb']:.3f}GB")
                logger.info(f"  Peak reserved: {gpu_data['peak_reserved_gb']:.3f}GB")
                logger.info(f"  Max utilization: {gpu_data['max_utilization_percent']:.1f}%")
                logger.info(f"  Average efficiency: {gpu_data['average_efficiency_percent']:.1f}%")
                logger.info(f"  Memory leak detected: {'Yes' if gpu_data['memory_leak_detected'] else 'No'}")
        
        logger.info(f"Average generation time: {sum(generation_times)/len(generation_times):.2f}s")
        logger.info("=" * 60)
        
        return True, report
        
    except Exception as e:
        logger.error(f"Error during enhanced diffusion test: {e}")
        if continuous_monitoring:
            monitor.stop_continuous_monitoring()
        
        monitor.measure_memory("error", f"Error occurred: {str(e)}")
        report = monitor.save_comprehensive_report()
        return False, report


def main():
    """Enhanced main function with better argument handling"""
    parser = argparse.ArgumentParser(
        description="Enhanced VRAM Monitor for Diffusion Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vram_monitor_enhanced.py --model sd15 --image_size 512 --num_images 3
  python vram_monitor_enhanced.py --model sdxl --image_size 1024 --continuous_monitoring
  python vram_monitor_enhanced.py --model sd-food --output_dir ./my_analysis
        """
    )
    
    parser.add_argument("--model", type=str, default="sd-food",
                       choices=["sd-food", "sd15", "sd21", "sdxl", "sdxl-turbo"],
                       help="Diffusion model to test")
    parser.add_argument("--image_size", type=int, default=512,
                       choices=[256, 512, 768, 1024],
                       help="Size of generated images")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of test images to generate")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_analysis",
                       help="Output directory for generated images and logs")
    parser.add_argument("--continuous_monitoring", action="store_true",
                       help="Enable continuous memory monitoring during generation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("ENHANCED VRAM ANALYSIS FOR DIFFUSION MODELS")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")
    logger.info(f"Number of images: {args.num_images}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Continuous monitoring: {'Enabled' if args.continuous_monitoring else 'Disabled'}")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        success, report = enhanced_diffusion_test(
            model_name=args.model,
            image_size=args.image_size,
            num_images=args.num_images,
            output_dir=args.output_dir,
            continuous_monitoring=args.continuous_monitoring
        )
        
        total_time = time.time() - start_time
        
        if success:
            logger.info(f"Enhanced VRAM analysis completed successfully in {total_time:.2f}s!")
            return 0
        else:
            logger.error(f"Enhanced VRAM analysis failed after {total_time:.2f}s!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())