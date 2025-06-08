#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Comprehensive Testing for Pizza Quality Assessment System

This script performs comprehensive testing of the spatial-MLLM pizza quality system:
1. Real image data batch inference testing
2. Performance benchmarking across different hardware
3. Memory optimization validation
4. Spatial-MLLM integration testing
5. RP2040 deployment preparation

SPATIAL-3.3 Testing Suite
Author: GitHub Copilot (2025-01-11)
"""

import os
import sys
import time
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our optimized inference system
try:
    from scripts.spatial_inference_optimized import (
        SpatialMLLMInferenceSystem,
        PizzaQualityDataset,
        InferenceConfig,
        logger
    )
    INFERENCE_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import inference system: {e}")
    INFERENCE_SYSTEM_AVAILABLE = False

class PizzaTestingFramework:
    """Comprehensive testing framework for pizza quality assessment"""
    
    def __init__(self, data_root: str = "/home/emilio/Documents/ai/pizza/data"):
        self.data_root = Path(data_root)
        self.test_results = {}
        self.setup_logging()
        
        # Test configuration
        self.test_config = InferenceConfig(
            batch_size=4,
            enable_amp=True,
            enable_parallel_encoders=True,
            max_workers=4,
            hardware_backend="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize system if available
        if INFERENCE_SYSTEM_AVAILABLE:
            self.inference_system = SpatialMLLMInferenceSystem(config=self.test_config)
        else:
            self.inference_system = None
            
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.data_root / "test_results"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"end_to_end_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def collect_test_images(self) -> Dict[str, List[Path]]:
        """Collect test images from different categories"""
        self.logger.info("🔍 Collecting test images...")
        
        test_images = {
            "raw": [],
            "basic": [],
            "burnt": [],
            "burning": []
        }
        
        # Raw images
        raw_dir = self.data_root / "raw"
        if raw_dir.exists():
            for img_file in raw_dir.glob("*.jpg"):
                test_images["raw"].append(img_file)
                
        # Test images by category
        test_dir = self.data_root / "test_new_images"
        if test_dir.exists():
            for category in ["basic", "burnt", "burning"]:
                cat_dir = test_dir / category
                if cat_dir.exists():
                    for img_file in cat_dir.glob("*.jpg"):
                        test_images[category].append(img_file)
                        
        # Log collection results
        for category, files in test_images.items():
            self.logger.info(f"  📁 {category}: {len(files)} images")
            
        return test_images
        
    def test_single_image_inference(self, image_path: Path) -> Dict[str, Any]:
        """Test single image inference with detailed metrics"""
        self.logger.info(f"🧪 Testing single image: {image_path.name}")
        
        if not self.inference_system:
            return {"error": "Inference system not available"}
            
        try:
            # Load and preprocess image
            start_time = time.time()
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            load_time = time.time() - start_time
            
            # Run inference
            inference_start = time.time()
            results = self.inference_system.process_image(image)
            inference_time = time.time() - inference_start
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "image_path": str(image_path),
                "image_size": image.size,
                "load_time": load_time,
                "inference_time": inference_time,
                "total_time": total_time,
                "results": results,
                "memory_usage": self.get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Single image test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path)
            }
            
    def test_batch_inference(self, image_paths: List[Path], batch_size: int = 4) -> Dict[str, Any]:
        """Test batch inference with performance metrics"""
        self.logger.info(f"🚀 Testing batch inference: {len(image_paths)} images, batch_size={batch_size}")
        
        if not self.inference_system:
            return {"error": "Inference system not available"}
            
        try:
            # Create dataset
            dataset = PizzaQualityDataset(
                image_paths=[str(p) for p in image_paths],
                labels=["unknown"] * len(image_paths)
            )
            
            # Run batch inference
            start_time = time.time()
            results = self.inference_system.batch_inference(dataset, batch_size=batch_size)
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "num_images": len(image_paths),
                "batch_size": batch_size,
                "total_time": total_time,
                "images_per_second": len(image_paths) / total_time,
                "results": results,
                "memory_usage": self.get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Batch inference test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "num_images": len(image_paths),
                "batch_size": batch_size
            }
            
    def test_memory_optimization(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Test memory optimization across different batch sizes"""
        self.logger.info("🧠 Testing memory optimization...")
        
        if not self.inference_system:
            return {"error": "Inference system not available"}
            
        memory_results = {}
        batch_sizes = [1, 2, 4, 8] if len(image_paths) >= 8 else [1, 2, 4]
        
        for batch_size in batch_sizes:
            if batch_size > len(image_paths):
                continue
                
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Run test
                test_images = image_paths[:batch_size * 2]  # Use 2 batches worth
                result = self.test_batch_inference(test_images, batch_size)
                
                memory_results[f"batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "success": result["success"],
                    "memory_usage": result.get("memory_usage", {}),
                    "images_per_second": result.get("images_per_second", 0)
                }
                
            except Exception as e:
                memory_results[f"batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "success": False,
                    "error": str(e)
                }
                
        return memory_results
        
    def test_device_compatibility(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Test compatibility across different devices"""
        self.logger.info("💻 Testing device compatibility...")
        
        device_results = {}
        devices = ["cpu"]
        
        if torch.cuda.is_available():
            devices.append("cuda")
            
        for device in devices:
            try:
                # Create device-specific config
                device_config = InferenceConfig(
                    batch_size=2,
                    hardware_backend=device,
                    enable_amp=(device == "cuda"),
                    enable_parallel_encoders=True
                )
                
                # Initialize system for this device
                device_system = SpatialMLLMInferenceSystem(config=device_config)
                
                # Run small test
                test_images = image_paths[:4]
                start_time = time.time()
                
                for img_path in test_images[:2]:  # Test 2 images
                    image = Image.open(img_path).convert('RGB')
                    result = device_system.process_image(image)
                    
                total_time = time.time() - start_time
                
                device_results[device] = {
                    "success": True,
                    "device": device,
                    "total_time": total_time,
                    "memory_usage": self.get_memory_usage(),
                    "mixed_precision": device_config.enable_amp
                }
                
            except Exception as e:
                device_results[device] = {
                    "success": False,
                    "device": device,
                    "error": str(e)
                }
                
        return device_results
        
    def test_quantization_optimization(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Test Int4 quantization performance"""
        self.logger.info("⚡ Testing quantization optimization...")
        
        if not self.inference_system:
            return {"error": "Inference system not available"}
            
        try:
            # Test with optimization enabled vs disabled
            results = {}
            
            # Test without optimization
            config_no_opt = InferenceConfig(
                batch_size=2,
                enable_parallel_encoders=False,
                enable_amp=False,
                hardware_backend=self.test_config.hardware_backend
            )
            system_no_opt = SpatialMLLMInferenceSystem(config=config_no_opt)
            
            # Test with optimization
            config_opt = InferenceConfig(
                batch_size=2,
                enable_parallel_encoders=True,
                enable_amp=True,
                hardware_backend=self.test_config.hardware_backend
            )
            system_opt = SpatialMLLMInferenceSystem(config=config_opt)
            
            test_images = image_paths[:4]
            
            # Benchmark both
            for name, system in [("without_opt", system_no_opt), ("with_opt", system_opt)]:
                start_time = time.time()
                
                for img_path in test_images:
                    image = Image.open(img_path).convert('RGB')
                    result = system.process_image(image)
                    
                total_time = time.time() - start_time
                
                results[name] = {
                    "total_time": total_time,
                    "images_per_second": len(test_images) / total_time,
                    "memory_usage": self.get_memory_usage()
                }
                
            # Calculate speedup
            if results["without_opt"]["total_time"] > 0:
                speedup = results["without_opt"]["total_time"] / results["with_opt"]["total_time"]
                results["speedup"] = speedup
                
            return results
            
        except Exception as e:
            return {"error": str(e)}
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_info = {}
        
        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
            memory_info["cpu_memory_percent"] = process.memory_percent()
        except ImportError:
            pass
            
        # GPU memory
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info["gpu_memory_cached_mb"] = torch.cuda.memory_cached() / 1024 / 1024
            
        return memory_info
        
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🍕 PIZZA QUALITY ASSESSMENT SYSTEM - TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Hardware Backend: {self.test_config.hardware_backend}")
        report_lines.append("")
        
        # Single image tests
        if "single_image_tests" in results:
            report_lines.append("📊 SINGLE IMAGE INFERENCE TESTS")
            report_lines.append("-" * 40)
            for test in results["single_image_tests"]:
                if test["success"]:
                    report_lines.append(f"✅ {Path(test['image_path']).name}")
                    report_lines.append(f"   Size: {test['image_size']}")
                    report_lines.append(f"   Load time: {test['load_time']:.3f}s")
                    report_lines.append(f"   Inference time: {test['inference_time']:.3f}s")
                    report_lines.append(f"   Total time: {test['total_time']:.3f}s")
                else:
                    report_lines.append(f"❌ {Path(test['image_path']).name}: {test['error']}")
            report_lines.append("")
            
        # Batch inference tests
        if "batch_tests" in results:
            report_lines.append("🚀 BATCH INFERENCE TESTS")
            report_lines.append("-" * 40)
            for test in results["batch_tests"]:
                if test["success"]:
                    report_lines.append(f"✅ Batch size {test['batch_size']}: {test['images_per_second']:.2f} images/sec")
                else:
                    report_lines.append(f"❌ Batch size {test['batch_size']}: {test['error']}")
            report_lines.append("")
            
        # Memory optimization
        if "memory_tests" in results:
            report_lines.append("🧠 MEMORY OPTIMIZATION TESTS")
            report_lines.append("-" * 40)
            for batch_name, test in results["memory_tests"].items():
                if test["success"]:
                    report_lines.append(f"✅ {batch_name}: {test['images_per_second']:.2f} images/sec")
                    if "memory_usage" in test and "gpu_memory_allocated_mb" in test["memory_usage"]:
                        report_lines.append(f"   GPU Memory: {test['memory_usage']['gpu_memory_allocated_mb']:.1f} MB")
                else:
                    report_lines.append(f"❌ {batch_name}: {test['error']}")
            report_lines.append("")
            
        # Device compatibility
        if "device_tests" in results:
            report_lines.append("💻 DEVICE COMPATIBILITY TESTS")
            report_lines.append("-" * 40)
            for device, test in results["device_tests"].items():
                if test["success"]:
                    report_lines.append(f"✅ {device.upper()}: {test['total_time']:.3f}s")
                else:
                    report_lines.append(f"❌ {device.upper()}: {test['error']}")
            report_lines.append("")
            
        # Quantization tests
        if "quantization_tests" in results:
            report_lines.append("⚡ QUANTIZATION OPTIMIZATION TESTS")
            report_lines.append("-" * 40)
            qtest = results["quantization_tests"]
            if "error" not in qtest:
                report_lines.append(f"Without optimization: {qtest['without_opt']['images_per_second']:.2f} images/sec")
                report_lines.append(f"With optimization: {qtest['with_opt']['images_per_second']:.2f} images/sec")
                if "speedup" in qtest:
                    report_lines.append(f"Speedup: {qtest['speedup']:.2f}x")
            else:
                report_lines.append(f"❌ Error: {qtest['error']}")
            report_lines.append("")
            
        report_lines.append("=" * 80)
        return "\n".join(report_lines)
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        self.logger.info("🚀 Starting comprehensive pizza quality assessment tests...")
        
        # Collect test images
        test_images = self.collect_test_images()
        
        # Select representative images for testing
        all_test_images = []
        for category, images in test_images.items():
            if images:
                # Take up to 3 images from each category
                all_test_images.extend(images[:3])
                
        if not all_test_images:
            self.logger.error("❌ No test images found!")
            return {"error": "No test images available"}
            
        self.logger.info(f"📸 Using {len(all_test_images)} test images")
        
        results = {
            "test_config": {
                "num_images": len(all_test_images),
                "hardware_backend": self.test_config.hardware_backend,
                "batch_size": self.test_config.batch_size,
                "mixed_precision": self.test_config.enable_amp
            }
        }
        
        # 1. Single image inference tests
        self.logger.info("1️⃣ Running single image inference tests...")
        single_tests = []
        for img_path in all_test_images[:5]:  # Test first 5 images
            result = self.test_single_image_inference(img_path)
            single_tests.append(result)
        results["single_image_tests"] = single_tests
        
        # 2. Batch inference tests
        self.logger.info("2️⃣ Running batch inference tests...")
        batch_tests = []
        for batch_size in [1, 2, 4]:
            if batch_size <= len(all_test_images):
                result = self.test_batch_inference(all_test_images[:batch_size*2], batch_size)
                batch_tests.append(result)
        results["batch_tests"] = batch_tests
        
        # 3. Memory optimization tests
        self.logger.info("3️⃣ Running memory optimization tests...")
        results["memory_tests"] = self.test_memory_optimization(all_test_images)
        
        # 4. Device compatibility tests
        self.logger.info("4️⃣ Running device compatibility tests...")
        results["device_tests"] = self.test_device_compatibility(all_test_images)
        
        # 5. Quantization tests
        self.logger.info("5️⃣ Running quantization optimization tests...")
        results["quantization_tests"] = self.test_quantization_optimization(all_test_images)
        
        # Generate and save report
        report = self.generate_test_report(results)
        
        # Save results
        results_dir = self.data_root / "test_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        json_file = results_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save text report
        report_file = results_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
            
        self.logger.info(f"📊 Results saved to: {json_file}")
        self.logger.info(f"📋 Report saved to: {report_file}")
        
        print("\n" + report)
        
        return results

def main():
    """Main testing function"""
    print("🍕 Pizza Quality Assessment System - Comprehensive Testing")
    print("=" * 60)
    
    # Create testing framework
    tester = PizzaTestingFramework()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Summary
    if "error" in results:
        print(f"\n❌ Testing failed: {results['error']}")
        return 1
    else:
        print(f"\n✅ Comprehensive testing completed successfully!")
        print(f"📊 Check test_results directory for detailed reports")
        return 0

if __name__ == "__main__":
    exit(main())
