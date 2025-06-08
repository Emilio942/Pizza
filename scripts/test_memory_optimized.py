#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Optimized Spatial Inference Test Script

Quick test of the memory-optimized spatial inference pipeline with available images.
"""

import sys
import os
import time
import json
from pathlib import Path
from spatial_inference_memory_optimized import MemoryOptimizedSpatialInference, MemoryOptimizedConfig

def test_memory_optimized_inference():
    """Test the memory-optimized inference pipeline with available images"""
    print("🚀 Testing Memory-Optimized Spatial Inference Pipeline")
    print("=" * 60)
    
    # Initialize with conservative memory settings
    config = MemoryOptimizedConfig(
        initial_batch_size=1,
        max_batch_size=2,
        memory_threshold=0.75,
        enable_cpu_offload=True,
        enable_gradient_checkpointing=True,
        enable_amp=True,
        cleanup_frequency=2,
        benchmark_iterations=3
    )
    
    # Initialize pipeline
    pipeline = MemoryOptimizedSpatialInference(config)
    
    # Load model
    print("🔄 Loading model with memory optimizations...")
    success = pipeline.load_model("Diankun/Spatial-MLLM-subset-sft")
    if not success:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Get memory info after loading
    memory_info = pipeline.memory_manager.get_gpu_memory_info()
    print(f"📊 Memory after loading: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB ({memory_info['utilization']:.1%})")
    
    # Find test images
    test_images_dir = Path("/home/emilio/Documents/ai/pizza/data/test")
    test_images = []
    
    # Collect some test images from different categories
    categories = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']
    for category in categories:
        category_dir = test_images_dir / category
        if category_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                images = list(category_dir.glob(f'*{ext}'))
                if images:
                    test_images.append(str(images[0]))  # Take first image from each category
                    break
    
    # Add the sample image if it exists
    sample_image = test_images_dir / "sample_pizza_image.jpg"
    if sample_image.exists():
        test_images.append(str(sample_image))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"📁 Found {len(test_images)} test images")
    for img in test_images:
        print(f"  - {Path(img).name}")
    
    # Test inference
    print("\n🔄 Running memory-safe inference...")
    start_time = time.time()
    
    try:
        results = pipeline._memory_safe_inference(test_images)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results) if results else 0
        
        print(f"\n✅ Inference completed in {total_time:.2f}s")
        print(f"📊 Success rate: {success_rate:.1%} ({len(successful_results)}/{len(results)})")
        print(f"⚡ Average time per image: {total_time/len(test_images):.3f}s")
        
        # Show results
        print("\n📋 Results:")
        for result in results:
            status = "✅" if result.success else "❌"
            image_name = Path(result.image_path).name
            memory_usage = result.memory_usage.get('allocated_gb', 0)
            print(f"  {status} {image_name}: {result.prediction} (Memory: {memory_usage:.2f}GB)")
        
        # Final memory state
        final_memory = pipeline.memory_manager.get_gpu_memory_info()
        print(f"\n📊 Final memory usage: {final_memory['allocated_gb']:.2f}GB / {final_memory['total_gb']:.2f}GB ({final_memory['utilization']:.1%})")
        
        # Performance stats
        print(f"\n📈 Performance Stats:")
        print(f"  - Memory cleanups: {pipeline.performance_stats['memory_cleanups']}")
        print(f"  - OOM errors: {pipeline.performance_stats['oom_errors']}")
        print(f"  - Batch adaptations: {pipeline.performance_stats['batch_size_adaptations']}")
        
        # Save test results
        output_path = "/home/emilio/Documents/ai/pizza/results/memory_test_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        test_results = {
            'test_timestamp': time.time(),
            'config': {
                'max_batch_size': config.max_batch_size,
                'memory_threshold': config.memory_threshold,
                'cleanup_frequency': config.cleanup_frequency
            },
            'performance': {
                'total_time': total_time,
                'success_rate': success_rate,
                'avg_time_per_image': total_time / len(test_images),
                'images_processed': len(test_images)
            },
            'memory_usage': {
                'initial': memory_info,
                'final': final_memory
            },
            'stats': pipeline.performance_stats,
            'results': [
                {
                    'image': Path(r.image_path).name,
                    'prediction': r.prediction,
                    'success': r.success,
                    'processing_time': r.processing_time,
                    'memory_gb': r.memory_usage.get('allocated_gb', 0)
                } for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {output_path}")
        print("\n🎉 Memory-optimized inference test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_memory_optimized_inference()
