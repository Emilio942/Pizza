#!/usr/bin/env python3
"""
SPATIAL-4.2: Automated Testing for Spatial Features

This script implements comprehensive automated tests for the Spatial-MLLM
integration, covering functionality, performance, and integration testing.
"""

import os
import sys
import json
import time
import torch
import pytest
import asyncio
import logging
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
import unittest.mock as mock

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append("/home/emilio/Documents/ai/Spatial-MLLM")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SpatialTestResult:
    """Test result for spatial feature tests"""
    test_name: str
    status: str  # pass, fail, skip
    duration: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None

class SpatialFeatureTests:
    """Comprehensive test suite for Spatial-MLLM features"""
    
    def __init__(self, test_config: Optional[Dict[str, Any]] = None):
        self.config = test_config or {}
        self.results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = project_root
        
        # Test data paths
        self.test_data_dir = self.project_root / "data" / "test"
        self.spatial_data_dir = self.project_root / "data" / "spatial_processed"
        self.model_dir = self.project_root / "models" / "spatial_mllm"
        
        # API endpoints
        self.api_base_url = self.config.get("api_base_url", "http://localhost:8001")
        
        logger.info(f"🧪 Spatial Feature Tests initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Test data: {self.test_data_dir}")
        logger.info(f"API URL: {self.api_base_url}")

    def run_test(self, test_func, test_name: str) -> SpatialTestResult:
        """Run a single test with error handling and timing"""
        logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if isinstance(result, dict):
                metrics = result.get("metrics", {})
                details = result.get("details", {})
                status = result.get("status", "pass")
            else:
                metrics = {}
                details = {}
                status = "pass"
            
            test_result = SpatialTestResult(
                test_name=test_name,
                status=status,
                duration=duration,
                metrics=metrics,
                details=details
            )
            
            logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = SpatialTestResult(
                test_name=test_name,
                status="fail",
                duration=duration,
                error_message=str(e),
                details={"exception_type": type(e).__name__}
            )
            
            logger.error(f"❌ {test_name} - FAILED: {e}")
        
        self.results.append(test_result)
        return test_result

    def test_spatial_model_loading(self) -> Dict[str, Any]:
        """Test 1: Spatial-MLLM model loading and initialization"""
        try:
            # Import spatial modules
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            model_id = "Diankun/Spatial-MLLM-subset-sft"
            
            # Test model loading
            start_time = time.time()
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map={"": self.device} if self.device.type == "cuda" else None,
            )
            processor = AutoProcessor.from_pretrained(model_id)
            load_time = time.time() - start_time
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            return {
                "status": "pass",
                "metrics": {
                    "load_time": load_time,
                    "total_parameters": total_params,
                    "model_size_mb": total_params * 4 / 1024 / 1024  # Estimate
                },
                "details": {
                    "model_id": model_id,
                    "device": str(self.device),
                    "model_type": type(model).__name__
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def test_spatial_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Test 2: Spatial preprocessing pipeline functionality"""
        try:
            from scripts.spatial_preprocessing import SpatialPreprocessingPipeline
            
            # Initialize pipeline
            pipeline = SpatialPreprocessingPipeline(
                output_size=(518, 518),
                depth_estimation_method="edge_based"
            )
            
            # Test with sample image
            test_images = list(self.test_data_dir.glob("*.jpg"))[:3]
            if not test_images:
                return {
                    "status": "skip",
                    "details": {"reason": "No test images found"}
                }
            
            processing_times = []
            quality_scores = []
            
            for img_path in test_images:
                image = Image.open(img_path)
                
                start_time = time.time()
                result = pipeline.process_image(image)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                quality_scores.append(result.get("quality_score", 0.0))
            
            return {
                "status": "pass",
                "metrics": {
                    "avg_processing_time": np.mean(processing_times),
                    "avg_quality_score": np.mean(quality_scores),
                    "images_processed": len(test_images)
                },
                "details": {
                    "pipeline_config": {
                        "output_size": (518, 518),
                        "depth_estimation_method": "edge_based"
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def test_spatial_api_integration(self) -> Dict[str, Any]:
        """Test 3: Spatial-MLLM API integration"""
        try:
            # Test API health check
            health_response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if health_response.status_code != 200:
                return {
                    "status": "fail",
                    "details": {"error": f"Health check failed: {health_response.status_code}"}
                }
            
            # Test spatial prediction endpoint
            test_image_path = next(self.test_data_dir.glob("*.jpg"), None)
            if not test_image_path:
                return {
                    "status": "skip",
                    "details": {"reason": "No test images found"}
                }
            
            with open(test_image_path, "rb") as f:
                files = {"file": f}
                
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/predict/spatial",
                    files=files,
                    timeout=30
                )
                inference_time = time.time() - start_time
            
            if response.status_code != 200:
                return {
                    "status": "fail",
                    "details": {"error": f"API request failed: {response.status_code}"}
                }
            
            result = response.json()
            
            return {
                "status": "pass",
                "metrics": {
                    "inference_time": inference_time,
                    "confidence": result.get("confidence", 0.0),
                    "response_size": len(str(result))
                },
                "details": {
                    "prediction": result.get("prediction"),
                    "spatial_features": bool(result.get("spatial_features"))
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def test_dual_encoder_functionality(self) -> Dict[str, Any]:
        """Test 4: Dual encoder (2D + 3D) functionality"""
        try:
            # Test if spatial inference pipeline works
            from scripts.spatial_inference_optimized import OptimizedSpatialInference, InferenceConfig
            
            # Create config for the inference
            config = InferenceConfig(
                batch_size=1,
                max_workers=2,
                enable_amp=False,  # Disable AMP for testing to avoid potential issues
                hardware_backend="auto"
            )
            
            inference = OptimizedSpatialInference(config)
            
            # Test with sample data
            test_image_path = next(self.test_data_dir.glob("*.jpg"), None)
            if not test_image_path:
                return {
                    "status": "skip",
                    "details": {"reason": "No test images found"}
                }
            
            image = Image.open(test_image_path)
            
            start_time = time.time()
            result = inference.predict_single(image)
            processing_time = time.time() - start_time
            
            return {
                "status": "pass",
                "metrics": {
                    "processing_time": processing_time,
                    "has_spatial_features": bool(result.get("spatial_features")),
                    "has_visual_features": bool(result.get("visual_features"))
                },
                "details": {
                    "prediction": result.get("prediction"),
                    "confidence": result.get("confidence", 0.0)
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def test_memory_optimization(self) -> Dict[str, Any]:
        """Test 5: Memory optimization for Spatial-MLLM"""
        try:
            if not torch.cuda.is_available():
                return {
                    "status": "skip",
                    "details": {"reason": "CUDA not available"}
                }
            
            # Test memory-optimized inference using OptimizedSpatialInference with memory_efficient=True
            from scripts.spatial_inference_optimized import OptimizedSpatialInference, InferenceConfig
            
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Create memory-optimized config
            config = InferenceConfig(
                batch_size=1,
                memory_efficient=True,
                enable_amp=True,
                hardware_backend="gpu"
            )
            
            inference = OptimizedSpatialInference(config)
            
            # Test multiple inferences to check memory stability
            test_images = list(self.test_data_dir.glob("*.jpg"))[:3]
            if not test_images:
                return {
                    "status": "skip",
                    "details": {"reason": "No test images found"}
                }
            
            memory_usage = []
            inference_times = []
            
            for img_path in test_images:
                image = Image.open(img_path)
                
                start_time = time.time()
                result = inference.predict(image)
                inference_time = time.time() - start_time
                
                current_memory = torch.cuda.memory_allocated()
                memory_usage.append(current_memory)
                inference_times.append(inference_time)
            
            max_memory = max(memory_usage)
            final_memory = torch.cuda.memory_allocated()
            
            return {
                "status": "pass",
                "metrics": {
                    "initial_memory_mb": initial_memory / 1024 / 1024,
                    "max_memory_mb": max_memory / 1024 / 1024,
                    "final_memory_mb": final_memory / 1024 / 1024,
                    "avg_inference_time": np.mean(inference_times),
                    "memory_leak_mb": (final_memory - initial_memory) / 1024 / 1024
                },
                "details": {
                    "images_tested": len(test_images),
                    "memory_stable": abs(final_memory - initial_memory) < 100 * 1024 * 1024  # 100MB threshold
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def test_model_versioning(self) -> Dict[str, Any]:
        """Test 6: Model versioning for dual-encoder models"""
        try:
            # Check if model files exist and have proper metadata
            model_files = list(self.model_dir.glob("*.pth"))
            
            if not model_files:
                return {
                    "status": "skip",
                    "details": {"reason": "No model files found"}
                }
            
            model_info = []
            for model_path in model_files:
                try:
                    # Load model metadata
                    checkpoint = torch.load(model_path, map_location="cpu")
                    
                    info = {
                        "path": str(model_path),
                        "size_mb": model_path.stat().st_size / 1024 / 1024,
                        "has_metadata": "metadata" in checkpoint,
                        "has_state_dict": "state_dict" in checkpoint or any(k.startswith("model.") for k in checkpoint.keys())
                    }
                    
                    if "metadata" in checkpoint:
                        metadata = checkpoint["metadata"]
                        info.update({
                            "version": metadata.get("version"),
                            "architecture": metadata.get("architecture"),
                            "training_date": metadata.get("training_date")
                        })
                    
                    model_info.append(info)
                    
                except Exception as e:
                    model_info.append({
                        "path": str(model_path),
                        "error": str(e)
                    })
            
            return {
                "status": "pass",
                "metrics": {
                    "total_models": len(model_files),
                    "valid_models": len([m for m in model_info if "error" not in m]),
                    "total_size_mb": sum(m.get("size_mb", 0) for m in model_info if "size_mb" in m)
                },
                "details": {
                    "models": model_info
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def test_multi_environment_compatibility(self) -> Dict[str, Any]:
        """Test 7: Multi-environment deployment compatibility"""
        try:
            compatibility_results = {}
            
            # Test CUDA compatibility
            compatibility_results["cuda"] = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "version": torch.version.cuda if torch.cuda.is_available() else None
            }
            
            # Test CPU-only mode
            device_backup = self.device
            self.device = torch.device("cpu")
            
            try:
                # Test basic model loading on CPU
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("Diankun/Spatial-MLLM-subset-sft")
                compatibility_results["cpu"] = {
                    "model_loading": True,
                    "tokenizer_loading": True
                }
            except Exception as e:
                compatibility_results["cpu"] = {
                    "model_loading": False,
                    "error": str(e)
                }
            finally:
                self.device = device_backup
            
            # Test memory requirements
            import psutil
            memory_info = psutil.virtual_memory()
            compatibility_results["memory"] = {
                "total_gb": memory_info.total / 1024 / 1024 / 1024,
                "available_gb": memory_info.available / 1024 / 1024 / 1024,
                "sufficient": memory_info.available > 8 * 1024 * 1024 * 1024  # 8GB minimum
            }
            
            # Test dependencies
            required_packages = ["torch", "transformers", "PIL", "numpy"]
            dependency_check = {}
            
            for package in required_packages:
                try:
                    __import__(package)
                    dependency_check[package] = True
                except ImportError:
                    dependency_check[package] = False
            
            compatibility_results["dependencies"] = dependency_check
            
            # Overall compatibility score
            score = 0
            if compatibility_results["cuda"]["available"]: score += 30
            if compatibility_results["cpu"]["model_loading"]: score += 20
            if compatibility_results["memory"]["sufficient"]: score += 30
            if all(dependency_check.values()): score += 20
            
            return {
                "status": "pass",
                "metrics": {
                    "compatibility_score": score,
                    "max_score": 100
                },
                "details": compatibility_results
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "details": {"error": str(e)}
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all spatial feature tests"""
        logger.info("🚀 Starting comprehensive Spatial-MLLM feature testing...")
        
        test_suite = [
            (self.test_spatial_model_loading, "Spatial Model Loading"),
            (self.test_spatial_preprocessing_pipeline, "Spatial Preprocessing Pipeline"),
            (self.test_spatial_api_integration, "Spatial API Integration"),
            (self.test_dual_encoder_functionality, "Dual Encoder Functionality"),
            (self.test_memory_optimization, "Memory Optimization"),
            (self.test_model_versioning, "Model Versioning"),
            (self.test_multi_environment_compatibility, "Multi-Environment Compatibility")
        ]
        
        for test_func, test_name in test_suite:
            self.run_test(test_func, test_name)
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "pass"])
        failed_tests = len([r for r in self.results if r.status == "fail"])
        skipped_tests = len([r for r in self.results if r.status == "skip"])
        
        summary = {
            "test_run_timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": sum(r.duration for r in self.results),
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "duration": r.duration,
                    "error_message": r.error_message,
                    "metrics": r.metrics,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        logger.info(f"📊 Test Summary: {passed_tests}/{total_tests} passed, {failed_tests} failed, {skipped_tests} skipped")
        
        return summary

def main():
    """Main function for SPATIAL-4.2 automated testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPATIAL-4.2: Automated Spatial Feature Testing")
    parser.add_argument("--output-dir", type=str, default="output/spatial_tests", help="Output directory for test results")
    parser.add_argument("--api-url", type=str, default="http://localhost:8001", help="Base URL for spatial API")
    parser.add_argument("--config", type=str, help="Path to test configuration JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    test_config = {"api_base_url": args.api_url}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            test_config.update(json.load(f))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tester = SpatialFeatureTests(test_config)
    summary = tester.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"spatial_tests_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📄 Test results saved to: {output_file}")
    
    # Exit with appropriate code
    if summary["failed"] > 0:
        logger.error("❌ Some tests failed!")
        return 1
    elif summary["passed"] == 0:
        logger.warning("⚠️ No tests passed!")
        return 1
    else:
        logger.info("✅ All tests passed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
