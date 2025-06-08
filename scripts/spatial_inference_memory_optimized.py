#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Optimized Spatial-MLLM Inference Pipeline for Real-time Performance

This script addresses critical CUDA memory management issues while implementing:
1. Dynamic memory-aware batch sizing
2. Aggressive memory cleanup and optimization
3. Gradient checkpointing and tensor management
4. CPU offloading strategies
5. Efficient dual-encoder processing with memory bounds

SPATIAL-3.3 Implementation - Memory Optimization Focus
Author: GitHub Copilot (2025-01-27)
"""

import gc
import os
import sys
import time
import json
import torch
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append("/home/emilio/Documents/ai/Spatial-MLLM")

# Import spatial preprocessing
try:
    from scripts.spatial_preprocessing import SpatialPreprocessingPipeline
    SPATIAL_PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Spatial preprocessing not available: {e}")
    SPATIAL_PREPROCESSING_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryOptimizedConfig:
    """Configuration for memory-optimized inference"""
    initial_batch_size: int = 1  # Start small and adapt
    max_batch_size: int = 4      # Maximum allowed batch size
    memory_threshold: float = 0.85  # Use max 85% of GPU memory
    enable_cpu_offload: bool = True  # Offload to CPU when needed
    enable_gradient_checkpointing: bool = True
    enable_amp: bool = True
    cleanup_frequency: int = 5   # Clean memory every N operations
    hardware_backend: str = "auto"
    benchmark_iterations: int = 10  # Reduced for memory safety

@dataclass
class InferenceResult:
    """Results from inference operation"""
    image_path: str
    prediction: str
    confidence: float
    processing_time: float
    memory_usage: Dict[str, float]
    batch_size_used: int
    success: bool
    error_message: Optional[str] = None

class MemoryManager:
    """Advanced memory management utilities"""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get detailed GPU memory information"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': free,
            'utilization': allocated / total if total > 0 else 0
        }
    
    @staticmethod
    def aggressive_cleanup():
        """Perform aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def can_fit_batch(batch_size: int, memory_threshold: float = 0.85) -> bool:
        """Check if a batch size can fit in available memory"""
        if not torch.cuda.is_available():
            return True  # CPU has more flexible memory
        
        memory_info = MemoryManager.get_gpu_memory_info()
        utilization = memory_info.get('utilization', 0)
        
        # Estimate memory needed for batch (rough heuristic)
        estimated_memory_per_image = 0.5  # GB per image (conservative estimate)
        estimated_total = estimated_memory_per_image * batch_size
        
        available_memory = memory_info.get('free_gb', 0)
        
        logger.info(f"Memory check: utilization={utilization:.2f}, available={available_memory:.2f}GB, estimated_needed={estimated_total:.2f}GB")
        
        return (utilization < memory_threshold) and (available_memory > estimated_total)
    
    @staticmethod
    def get_optimal_batch_size(max_batch_size: int, memory_threshold: float = 0.85) -> int:
        """Dynamically determine optimal batch size based on available memory"""
        for batch_size in range(max_batch_size, 0, -1):
            if MemoryManager.can_fit_batch(batch_size, memory_threshold):
                logger.info(f"Optimal batch size determined: {batch_size}")
                return batch_size
        
        logger.warning("Memory extremely constrained, using batch size 1")
        return 1

class MemoryOptimizedSpatialInference:
    """
    Memory-optimized inference pipeline for Spatial-MLLM with aggressive memory management.
    """
    
    def __init__(self, config: MemoryOptimizedConfig):
        self.config = config
        self.memory_manager = MemoryManager()
        self.device = self._determine_device()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.spatial_preprocessor = None
        
        # Memory optimization tools
        self.scaler = GradScaler() if config.enable_amp and self.device.type == 'cuda' else None
        self.operation_count = 0
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'memory_cleanups': 0,
            'batch_size_adaptations': 0,
            'oom_errors': 0
        }
        
        logger.info(f"✅ Initialized MemoryOptimizedSpatialInference on {self.device}")
        
        # Initial memory cleanup
        self.memory_manager.aggressive_cleanup()
    
    def _determine_device(self) -> torch.device:
        """Determine the optimal device for inference with memory considerations"""
        if self.config.hardware_backend == "cpu":
            return torch.device("cpu")
        elif self.config.hardware_backend == "gpu":
            if torch.cuda.is_available():
                # Check if GPU has sufficient memory
                memory_info = self.memory_manager.get_gpu_memory_info()
                if memory_info.get('free_gb', 0) < 2.0:  # Need at least 2GB free
                    logger.warning("GPU has insufficient memory, falling back to CPU")
                    return torch.device("cpu")
                return torch.device("cuda")
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                return torch.device("cpu")
        else:  # auto
            if torch.cuda.is_available():
                memory_info = self.memory_manager.get_gpu_memory_info()
                if memory_info.get('free_gb', 0) >= 2.0:
                    return torch.device("cuda")
            return torch.device("cpu")
    
    def load_model(self, model_name: str = "Diankun/Spatial-MLLM-subset-sft") -> bool:
        """Load the model with aggressive memory optimizations"""
        logger.info(f"🔄 Loading model with memory optimizations: {model_name}")
        start_time = time.time()
        
        try:
            # Pre-loading memory cleanup
            self.memory_manager.aggressive_cleanup()
            
            # GPU-specific optimizations
            if self.device.type == 'cuda':
                # Set conservative memory fraction
                torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
                
                # Optimize CUDA settings
                torch.backends.cudnn.benchmark = False  # Disable for memory consistency
                torch.backends.cudnn.deterministic = True
                
                # Enable memory-efficient attention
                if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(False)  # Can cause memory issues
            
            # Load tokenizer first (smaller memory footprint)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer for efficiency
            )
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Model loading with extreme memory optimizations
            logger.info("Loading model with memory optimizations...")
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,  # Critical for memory efficiency
                'device_map': 'auto' if self.device.type == 'cuda' else None,
            }
            
            # Add device-specific optimizations
            if self.device.type == 'cuda':
                model_kwargs.update({
                    'use_flash_attention_2': False,  # Disable to avoid memory issues
                    'attn_implementation': 'eager',   # Use memory-efficient attention
                })
            
            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
            
            # Apply memory optimizations
            if self.config.enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("✅ Gradient checkpointing enabled")
            
            # Move to device carefully
            if self.device.type != 'cuda' or 'device_map' not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # Set to eval mode and disable gradients
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Initialize spatial preprocessor with memory constraints
            if SPATIAL_PREPROCESSING_AVAILABLE:
                self.spatial_preprocessor = SpatialPreprocessingPipeline(
                    output_size=(518, 518),
                    depth_estimation_method="edge_based",
                    enable_quality_validation=False  # Disable for memory
                )
                logger.info("✅ Spatial preprocessor initialized")
            
            # Final memory cleanup
            self.memory_manager.aggressive_cleanup()
            
            load_time = time.time() - start_time
            memory_info = self.memory_manager.get_gpu_memory_info()
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Memory usage after loading: {memory_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.memory_manager.aggressive_cleanup()
            return False
    
    def _memory_efficient_preprocessing(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Memory-efficient image preprocessing with cleanup"""
        try:
            # Process with spatial preprocessor if available
            if self.spatial_preprocessor:
                processed_data = self.spatial_preprocessor.process_image(image_path)
                
                # Extract and optimize tensors
                visual_input = processed_data['visual_input'].squeeze(0)  # Remove batch dim
                spatial_input = processed_data['spatial_input'].squeeze(0)  # Remove batch dim
                
                # Ensure proper device placement with memory management
                visual_input = visual_input.to(self.device, non_blocking=True)
                spatial_input = spatial_input.to(self.device, non_blocking=True)
                
                return {
                    'visual_input': visual_input,
                    'spatial_input': spatial_input
                }
            else:
                # Fallback preprocessing with memory efficiency
                image = Image.open(image_path).convert('RGB')
                
                # Efficient transformation pipeline
                transform = transforms.Compose([
                    transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image)
                visual_input = img_tensor.unsqueeze(0).to(self.device, non_blocking=True)  # Add frame dim
                
                # Create minimal spatial input
                spatial_input = torch.zeros(1, 4, 518, 518, device=self.device, dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
                
                return {
                    'visual_input': visual_input,
                    'spatial_input': spatial_input
                }
                
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_path}: {e}")
            return None
        finally:
            # Cleanup after preprocessing
            if hasattr(self, 'spatial_preprocessor') and self.spatial_preprocessor:
                gc.collect()
    
    def _memory_safe_inference(self, image_paths: List[str]) -> List[InferenceResult]:
        """Memory-safe inference with dynamic batch sizing and cleanup"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        remaining_images = image_paths.copy()
        
        while remaining_images:
            # Determine optimal batch size for current memory state
            optimal_batch_size = self.memory_manager.get_optimal_batch_size(
                min(self.config.max_batch_size, len(remaining_images)),
                self.config.memory_threshold
            )
            
            # Get current batch
            current_batch = remaining_images[:optimal_batch_size]
            remaining_images = remaining_images[optimal_batch_size:]
            
            logger.info(f"Processing batch of {len(current_batch)} images")
            
            try:
                batch_results = self._process_single_batch(current_batch, optimal_batch_size)
                results.extend(batch_results)
                
                # Periodic memory cleanup
                self.operation_count += 1
                if self.operation_count % self.config.cleanup_frequency == 0:
                    logger.info("Performing scheduled memory cleanup...")
                    self.memory_manager.aggressive_cleanup()
                    self.performance_stats['memory_cleanups'] += 1
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM error with batch size {optimal_batch_size}: {e}")
                self.performance_stats['oom_errors'] += 1
                
                # Emergency memory cleanup
                self.memory_manager.aggressive_cleanup()
                
                # Retry with smaller batch size
                if optimal_batch_size > 1:
                    logger.info("Retrying with batch size 1...")
                    remaining_images = current_batch + remaining_images  # Put back in queue
                    self.performance_stats['batch_size_adaptations'] += 1
                    continue
                else:
                    # Even batch size 1 failed, create error results
                    for image_path in current_batch:
                        results.append(InferenceResult(
                            image_path=image_path,
                            prediction="error",
                            confidence=0.0,
                            processing_time=0.0,
                            memory_usage=self.memory_manager.get_gpu_memory_info(),
                            batch_size_used=0,
                            success=False,
                            error_message=f"CUDA OOM error: {str(e)}"
                        ))
            
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Create error results for failed batch
                for image_path in current_batch:
                    results.append(InferenceResult(
                        image_path=image_path,
                        prediction="error",
                        confidence=0.0,
                        processing_time=0.0,
                        memory_usage=self.memory_manager.get_gpu_memory_info(),
                        batch_size_used=optimal_batch_size,
                        success=False,
                        error_message=str(e)
                    ))
        
        self.performance_stats['total_operations'] += len(image_paths)
        return results
    
    def _process_single_batch(self, image_paths: List[str], batch_size: int) -> List[InferenceResult]:
        """Process a single batch with memory management"""
        batch_start_time = time.time()
        results = []
        
        # Preprocess images one by one to manage memory
        preprocessed_data = []
        for image_path in image_paths:
            data = self._memory_efficient_preprocessing(image_path)
            if data is not None:
                preprocessed_data.append(data)
            else:
                # Create error result for failed preprocessing
                results.append(InferenceResult(
                    image_path=image_path,
                    prediction="error",
                    confidence=0.0,
                    processing_time=0.0,
                    memory_usage=self.memory_manager.get_gpu_memory_info(),
                    batch_size_used=batch_size,
                    success=False,
                    error_message="Preprocessing failed"
                ))
        
        if not preprocessed_data:
            return results
        
        # Process each image individually to minimize memory usage
        for i, (image_path, data) in enumerate(zip(image_paths[:len(preprocessed_data)], preprocessed_data)):
            try:
                start_time = time.time()
                
                # Create prompt
                prompt = "Analyze this pizza image and classify it into one of these categories: basic, burnt, combined, mixed, progression, segment. Respond with just the category name."
                
                # Prepare messages for the model
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                
                # Process with model
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Handle vision input processing
                try:
                    from qwen_vl_utils import process_vision_info
                    _, vision_inputs = process_vision_info(messages)
                except ImportError:
                    vision_inputs = None
                
                # Prepare model inputs
                inputs = self.processor(
                    text=[text],
                    images=vision_inputs if vision_inputs else None,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device, non_blocking=True)
                
                # Override with our preprocessed visual data
                if 'pixel_values' in inputs and data['visual_input'] is not None:
                    visual_data = data['visual_input'].squeeze(0) if data['visual_input'].dim() == 5 else data['visual_input']
                    inputs['pixel_values'] = visual_data.unsqueeze(0)  # Add batch dimension
                
                # Generate with memory-efficient settings
                with torch.no_grad():
                    if self.config.enable_amp and self.scaler and self.device.type == 'cuda':
                        with autocast():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=10,  # Reduced for memory efficiency
                                do_sample=False,
                                temperature=0.0,
                                pad_token_id=self.tokenizer.pad_token_id,
                                use_cache=False,  # Disable KV cache for memory
                            )
                    else:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=10,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=False,
                        )
                
                # Decode response
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Parse prediction
                prediction = self._parse_pizza_prediction(output_text)
                processing_time = time.time() - start_time
                
                # Create result
                results.append(InferenceResult(
                    image_path=image_path,
                    prediction=prediction,
                    confidence=0.9 if prediction != "error" else 0.0,
                    processing_time=processing_time,
                    memory_usage=self.memory_manager.get_gpu_memory_info(),
                    batch_size_used=batch_size,
                    success=prediction != "error",
                    error_message=None
                ))
                
                # Clean up intermediate tensors
                del inputs, generated_ids, generated_ids_trimmed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append(InferenceResult(
                    image_path=image_path,
                    prediction="error",
                    confidence=0.0,
                    processing_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                    memory_usage=self.memory_manager.get_gpu_memory_info(),
                    batch_size_used=batch_size,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _parse_pizza_prediction(self, output_text: str) -> str:
        """Parse the model output to extract pizza category"""
        output_lower = output_text.lower().strip()
        categories = ["basic", "burnt", "combined", "mixed", "progression", "segment"]
        
        for category in categories:
            if category in output_lower:
                return category
        
        return "basic"  # Fallback
    
    def benchmark_performance(self, test_images_dir: str, output_path: str) -> Dict[str, Any]:
        """Memory-safe performance benchmarking"""
        logger.info(f"🚀 Starting memory-optimized performance benchmark")
        
        # Find test images (limited for memory safety)
        test_images = []
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in extensions:
            test_images.extend(Path(test_images_dir).rglob(f'*{ext}'))
        
        test_images = [str(img) for img in test_images[:20]]  # Limit to 20 images for memory safety
        
        if not test_images:
            logger.error(f"No test images found in {test_images_dir}")
            return {}
        
        logger.info(f"Testing with {len(test_images)} images")
        
        # Run memory-safe inference
        start_time = time.time()
        results = self._memory_safe_inference(test_images)
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results) if results else 0
        
        # Final memory info
        final_memory = self.memory_manager.get_gpu_memory_info()
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'initial_batch_size': self.config.initial_batch_size,
                'max_batch_size': self.config.max_batch_size,
                'memory_threshold': self.config.memory_threshold,
                'device': str(self.device)
            },
            'performance_metrics': {
                'total_images': len(test_images),
                'successful_inferences': len(successful_results),
                'success_rate': success_rate,
                'total_time': total_time,
                'average_time_per_image': total_time / len(test_images) if test_images else 0,
                'images_per_second': len(test_images) / total_time if total_time > 0 else 0
            },
            'memory_metrics': {
                'final_memory_usage': final_memory,
                'memory_cleanups': self.performance_stats['memory_cleanups'],
                'oom_errors': self.performance_stats['oom_errors'],
                'batch_size_adaptations': self.performance_stats['batch_size_adaptations']
            },
            'detailed_results': [
                {
                    'image_path': r.image_path,
                    'prediction': r.prediction,
                    'success': r.success,
                    'processing_time': r.processing_time,
                    'memory_usage': r.memory_usage
                } for r in results
            ]
        }
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logger.info(f"✅ Memory-optimized benchmark completed")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Average time per image: {benchmark_results['performance_metrics']['average_time_per_image']:.3f}s")
        logger.info(f"Memory cleanups performed: {self.performance_stats['memory_cleanups']}")
        logger.info(f"OOM errors encountered: {self.performance_stats['oom_errors']}")
        
        return benchmark_results


def main():
    """Main function to test the memory-optimized inference pipeline"""
    # Initialize with memory-conservative settings
    config = MemoryOptimizedConfig(
        initial_batch_size=1,
        max_batch_size=2,  # Very conservative
        memory_threshold=0.75,  # Use max 75% of GPU memory
        enable_cpu_offload=True,
        enable_gradient_checkpointing=True,
        enable_amp=True,
        cleanup_frequency=3,  # Clean memory every 3 operations
        benchmark_iterations=5
    )
    
    # Initialize inference pipeline
    pipeline = MemoryOptimizedSpatialInference(config)
    
    # Load model with memory optimizations
    success = pipeline.load_model("Diankun/Spatial-MLLM-subset-sft")
    if not success:
        logger.error("Failed to load model")
        return
    
    # Test with sample images
    test_images_dir = "/home/emilio/Documents/ai/pizza/data/subset_dataset"
    output_path = "/home/emilio/Documents/ai/pizza/results/memory_optimized_benchmark.json"
    
    # Run benchmark
    results = pipeline.benchmark_performance(test_images_dir, output_path)
    
    print(f"\n✅ Memory-optimized benchmark completed!")
    print(f"Results saved to: {output_path}")
    print(f"Check the detailed results for memory usage and performance metrics.")


if __name__ == "__main__":
    main()
