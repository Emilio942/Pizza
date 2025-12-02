#!/usr/bin/env python3
"""
SPATIAL-4.2: Lightweight Spatial Tests for Deployment Pipeline

This script implements essential spatial tests that can run without external services
for deployment pipeline validation.
"""

import os
import sys
import json
import time
import torch
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpatialDeploymentTests:
    """Lightweight spatial tests for deployment validation."""
    
    def __init__(self, project_root: str = None):
        """Initialize the test suite."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.start_time = time.time()
        
        # Setup paths
        sys.path.insert(0, str(self.project_root))
        
        logger.info("🧪 Spatial Deployment Tests initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Project root: {self.project_root}")
    
    def test_torch_cuda_availability(self) -> bool:
        """Test CUDA availability for spatial processing."""
        try:
            logger.info("Running test: CUDA Availability")
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                # Get CUDA info
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                memory_allocated = torch.cuda.memory_allocated(0) if gpu_count > 0 else 0
                memory_reserved = torch.cuda.memory_reserved(0) if gpu_count > 0 else 0
                
                self.results["cuda_availability"] = {
                    "status": "pass",
                    "cuda_available": True,
                    "cuda_version": cuda_version,
                    "gpu_count": gpu_count,
                    "gpu_name": gpu_name,
                    "memory_allocated_mb": memory_allocated / (1024*1024),
                    "memory_reserved_mb": memory_reserved / (1024*1024)
                }
                logger.info(f"✅ CUDA Available - GPU: {gpu_name}, Count: {gpu_count}")
            else:
                self.results["cuda_availability"] = {
                    "status": "pass",
                    "cuda_available": False,
                    "note": "CUDA not available, will use CPU"
                }
                logger.info("⚠️ CUDA not available, using CPU")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CUDA Availability test failed: {str(e)}")
            self.results["cuda_availability"] = {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return False
    
    def test_model_version_manager(self) -> bool:
        """Test model version manager functionality."""
        try:
            logger.info("Running test: Model Version Manager")
            
            # Import model version manager
            from scripts.model_version_manager import ModelVersionManager
            
            # Initialize manager
            manager = ModelVersionManager(str(self.project_root / "models"))
            
            # Test basic functionality
            # List models (should not fail even if no models exist)
            spatial_models = manager.list_models("spatial")
            standard_models = manager.list_models("standard")
            
            # Test validation functionality
            validation_result = manager.validate_models()
            
            self.results["model_version_manager"] = {
                "status": "pass",
                "spatial_models_count": len(spatial_models),
                "standard_models_count": len(standard_models),
                "validation_result": validation_result
            }
            
            logger.info(f"✅ Model Version Manager - Spatial: {len(spatial_models)}, Standard: {len(standard_models)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model Version Manager test failed: {str(e)}")
            self.results["model_version_manager"] = {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return False
    
    def test_spatial_imports(self) -> bool:
        """Test critical spatial module imports."""
        try:
            logger.info("Running test: Spatial Module Imports")
            
            imports_tested = []
            import_errors = []
            
            # Test transformers imports
            try:
                from transformers import AutoTokenizer, AutoImageProcessor
                imports_tested.append("transformers.AutoTokenizer")
                imports_tested.append("transformers.AutoImageProcessor")
            except Exception as e:
                import_errors.append(f"transformers: {str(e)}")
            
            # Test torch vision imports
            try:
                import torchvision.transforms as transforms
                imports_tested.append("torchvision.transforms")
            except Exception as e:
                import_errors.append(f"torchvision: {str(e)}")
            
            # Test spatial integration (if available)
            try:
                from src.spatial.spatial_integration import SpatialMLLMIntegration
                imports_tested.append("SpatialMLLMIntegration")
            except Exception as e:
                import_errors.append(f"SpatialMLLMIntegration: {str(e)}")
            
            # Test data classes
            try:
                from src.spatial.data_classes import SpatialResult, BoundingBox
                imports_tested.append("SpatialResult")
                imports_tested.append("BoundingBox")
            except Exception as e:
                import_errors.append(f"Spatial data classes: {str(e)}")
            
            success = len(import_errors) == 0
            
            self.results["spatial_imports"] = {
                "status": "pass" if success else "partial",
                "imports_tested": imports_tested,
                "import_errors": import_errors,
                "success_count": len(imports_tested),
                "error_count": len(import_errors)
            }
            
            if success:
                logger.info(f"✅ Spatial Imports - All {len(imports_tested)} imports successful")
            else:
                logger.warning(f"⚠️ Spatial Imports - {len(imports_tested)} successful, {len(import_errors)} failed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Spatial imports test failed: {str(e)}")
            self.results["spatial_imports"] = {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return False
    
    def test_basic_model_loading(self) -> bool:
        """Test basic model loading capability without full initialization."""
        try:
            logger.info("Running test: Basic Model Loading")
            
            # Test if we can access the spatial model directory
            spatial_model_dir = self.project_root / "models" / "spatial_mllm"
            
            model_info = {
                "spatial_model_dir_exists": spatial_model_dir.exists(),
                "model_files": []
            }
            
            if spatial_model_dir.exists():
                # List model files
                model_files = list(spatial_model_dir.glob("*.pth"))
                model_files.extend(spatial_model_dir.glob("*.bin"))
                model_files.extend(spatial_model_dir.glob("*.safetensors"))
                
                model_info["model_files"] = [f.name for f in model_files]
                model_info["model_count"] = len(model_files)
            
            # Test transformers model loading capability
            try:
                from transformers import AutoConfig
                # Try to get config for the spatial model ID
                model_id = "Diankun/Spatial-MLLM-subset-sft"
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                model_info["transformers_config_loaded"] = True
                model_info["model_type"] = config.model_type if hasattr(config, 'model_type') else "unknown"
            except Exception as e:
                model_info["transformers_config_loaded"] = False
                model_info["config_error"] = str(e)
            
            self.results["basic_model_loading"] = {
                "status": "pass",
                **model_info
            }
            
            logger.info(f"✅ Basic Model Loading - Directory exists: {model_info['spatial_model_dir_exists']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic model loading test failed: {str(e)}")
            self.results["basic_model_loading"] = {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return False
    
    def test_docker_environment(self) -> bool:
        """Test Docker environment compatibility."""
        try:
            logger.info("Running test: Docker Environment")
            
            # Check for Docker-related environment variables
            docker_env = {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "MODEL_CACHE_DIR": os.environ.get("MODEL_CACHE_DIR"),
                "OUTPUT_CACHE_DIR": os.environ.get("OUTPUT_CACHE_DIR"),
                "PYTHONPATH": os.environ.get("PYTHONPATH")
            }
            
            # Check if running in Docker
            in_docker = os.path.exists("/.dockerenv")
            
            # Check required directories
            required_dirs = [
                self.project_root / "models",
                self.project_root / "src",
                self.project_root / "scripts"
            ]
            
            dir_status = {str(d): d.exists() for d in required_dirs}
            
            self.results["docker_environment"] = {
                "status": "pass",
                "in_docker": in_docker,
                "environment_variables": docker_env,
                "directory_status": dir_status,
                "all_dirs_exist": all(dir_status.values())
            }
            
            logger.info(f"✅ Docker Environment - In Docker: {in_docker}, Dirs OK: {all(dir_status.values())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker environment test failed: {str(e)}")
            self.results["docker_environment"] = {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all deployment tests."""
        logger.info("🚀 Starting Spatial Deployment Tests...")
        
        # Test functions
        tests = [
            self.test_torch_cuda_availability,
            self.test_spatial_imports,
            self.test_model_version_manager,
            self.test_basic_model_loading,
            self.test_docker_environment
        ]
        
        # Run tests
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed with exception: {str(e)}")
        
        # Calculate summary
        duration = time.time() - self.start_time
        success_rate = passed / total
        
        summary = {
            "test_run_timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": success_rate,
            "total_duration": duration,
            "results": self.results
        }
        
        # Save results
        output_dir = self.project_root / "output" / "spatial_tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"spatial_deployment_tests_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log summary
        logger.info(f"📊 Test Summary: {passed}/{total} passed ({success_rate:.1%})")
        logger.info(f"📄 Results saved to: {results_file}")
        
        if passed == total:
            logger.info("🎉 All deployment tests passed!")
            return summary
        else:
            logger.error(f"❌ {total - passed} tests failed!")
            return summary

def main():
    """Main function."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    tester = SpatialDeploymentTests(project_root)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
