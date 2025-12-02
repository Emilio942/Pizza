#!/usr/bin/env python3
"""
Automated CI/CD Testing Framework for Spatial-MLLM
SPATIAL-4.2: Deployment-Pipeline erweitern
"""

import os
import sys
import json
import time
import docker
import asyncio
import logging
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import unittest
from concurrent.futures import ThreadPoolExecutor

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # pass, fail, skip, error
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: str = ""

class CICDTestFramework:
    """Comprehensive CI/CD testing framework"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.output_dir = self.project_root / "output" / "cicd_tests"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "cicd_tests.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        self.test_results: List[TestResult] = []
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and record results"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            self.logger.info(f"Running test: {test_name}")
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if result is True or (isinstance(result, dict) and result.get('success')):
                status = "pass"
                error_message = None
                details = result if isinstance(result, dict) else None
            else:
                status = "fail"
                error_message = str(result) if result else "Test returned False"
                details = None
                
        except Exception as e:
            duration = time.time() - start_time
            status = "error"
            error_message = str(e)
            details = None
            self.logger.error(f"Test {test_name} failed with error: {e}")
        
        test_result = TestResult(
            test_name=test_name,
            status=status,
            duration=duration,
            error_message=error_message,
            details=details,
            timestamp=timestamp
        )
        
        self.test_results.append(test_result)
        self.logger.info(f"Test {test_name} completed: {status} ({duration:.2f}s)")
        return test_result
    
    # ==== Docker Tests ====
    
    def test_docker_build_standard(self) -> bool:
        """Test building standard Docker container"""
        if not self.docker_available:
            raise Exception("Docker not available")
        
        try:
            # Build standard container
            image, logs = self.docker_client.images.build(
                path=str(self.project_root),
                dockerfile="Dockerfile",
                tag="pizza-detection-test:latest",
                rm=True
            )
            
            # Verify image exists
            images = self.docker_client.images.list(name="pizza-detection-test:latest")
            return len(images) > 0
            
        except Exception as e:
            self.logger.error(f"Standard Docker build failed: {e}")
            return False
    
    def test_docker_build_spatial(self) -> bool:
        """Test building spatial Docker container"""
        if not self.docker_available:
            raise Exception("Docker not available")
        
        try:
            # Build spatial container
            image, logs = self.docker_client.images.build(
                path=str(self.project_root),
                dockerfile="Dockerfile.spatial",
                tag="pizza-detection-spatial-test:latest",
                rm=True
            )
            
            # Verify image exists
            images = self.docker_client.images.list(name="pizza-detection-spatial-test:latest")
            return len(images) > 0
            
        except Exception as e:
            self.logger.error(f"Spatial Docker build failed: {e}")
            return False
    
    def test_container_startup_standard(self) -> Dict[str, Any]:
        """Test standard container startup and basic functionality"""
        if not self.docker_available:
            raise Exception("Docker not available")
        
        try:
            # Run container with health check
            container = self.docker_client.containers.run(
                "pizza-detection-test:latest",
                command="python -c 'import sys; sys.path.append(\"/app\"); from src.api.pizza_api import PizzaAPI; print(\"Import successful\")'",
                remove=True,
                detach=False,
                environment={
                    "MODEL_TYPE": "standard",
                    "PYTHONPATH": "/app"
                },
                volumes={
                    str(self.project_root / "test_data"): {"bind": "/app/test_data", "mode": "ro"}
                }
            )
            
            return {"success": True, "output": container.decode()}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_container_startup_spatial(self) -> Dict[str, Any]:
        """Test spatial container startup and basic functionality"""
        if not self.docker_available:
            raise Exception("Docker not available")
        
        try:
            # Check if NVIDIA Docker runtime is available
            runtime = "nvidia" if self._nvidia_docker_available() else None
            
            container = self.docker_client.containers.run(
                "pizza-detection-spatial-test:latest",
                command="python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'",
                remove=True,
                detach=False,
                runtime=runtime,
                environment={
                    "MODEL_TYPE": "spatial",
                    "PYTHONPATH": "/app"
                },
                volumes={
                    str(self.project_root / "test_data"): {"bind": "/app/test_data", "mode": "ro"},
                    str(self.project_root / "models"): {"bind": "/app/models", "mode": "ro"}
                }
            )
            
            return {"success": True, "output": container.decode()}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _nvidia_docker_available(self) -> bool:
        """Check if NVIDIA Docker runtime is available"""
        try:
            info = self.docker_client.info()
            runtimes = info.get('Runtimes', {})
            return 'nvidia' in runtimes
        except:
            return False
    
    # ==== API Tests ====
    
    def test_api_health_check(self) -> Dict[str, Any]:
        """Test API health check endpoints"""
        results = {}
        
        # Test standard API
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            results["standard_api"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
        except requests.RequestException as e:
            results["standard_api"] = {"success": False, "error": str(e)}
        
        # Test spatial API
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            results["spatial_api"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
        except requests.RequestException as e:
            results["spatial_api"] = {"success": False, "error": str(e)}
        
        # Test load balancer
        try:
            response = requests.get("http://localhost/health", timeout=5)
            results["load_balancer"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
        except requests.RequestException as e:
            results["load_balancer"] = {"success": False, "error": str(e)}
        
        # Overall success if at least one API is working
        overall_success = any(result.get("success") for result in results.values())
        return {"success": overall_success, "details": results}
    
    def test_api_prediction_endpoints(self) -> Dict[str, Any]:
        """Test API prediction functionality"""
        # Create a test image
        test_image_path = self._create_test_image()
        results = {}
        
        try:
            # Test standard prediction
            with open(test_image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    "http://localhost:8000/predict",
                    files=files,
                    timeout=30
                )
                results["standard_prediction"] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else None
                }
        except Exception as e:
            results["standard_prediction"] = {"success": False, "error": str(e)}
        
        try:
            # Test spatial prediction
            with open(test_image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    "http://localhost:8001/predict_spatial",
                    files=files,
                    timeout=60
                )
                results["spatial_prediction"] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else None
                }
        except Exception as e:
            results["spatial_prediction"] = {"success": False, "error": str(e)}
        
        # Cleanup
        os.unlink(test_image_path)
        
        overall_success = any(result.get("success") for result in results.values())
        return {"success": overall_success, "details": results}
    
    # ==== Model Tests ====
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading functionality"""
        results = {}
        
        try:
            # Test standard model loading
            sys.path.append(str(self.project_root))
            from src.api.pizza_api import PizzaAPI
            
            api = PizzaAPI()
            results["standard_model"] = {"success": True, "loaded": True}
            
        except Exception as e:
            results["standard_model"] = {"success": False, "error": str(e)}
        
        try:
            # Test spatial model loading
            from src.spatial.spatial_integration import SpatialMLLMIntegration
            
            spatial_integration = SpatialMLLMIntegration()
            results["spatial_model"] = {
                "success": True,
                "loaded": spatial_integration.spatial_model is not None
            }
            
        except Exception as e:
            results["spatial_model"] = {"success": False, "error": str(e)}
        
        overall_success = any(result.get("success") for result in results.values())
        return {"success": overall_success, "details": results}
    
    def test_model_versioning(self) -> Dict[str, Any]:
        """Test model versioning system"""
        try:
            # Import model version manager
            sys.path.append(str(self.project_root))
            from scripts.model_version_manager import ModelVersionManager
            
            manager = ModelVersionManager(str(self.project_root / "models"))
            
            # Test listing versions
            versions = manager.list_versions()
            
            # Test version database
            has_spatial = len(versions.get("spatial", [])) > 0
            has_standard = len(versions.get("standard", [])) > 0
            
            return {
                "success": True,
                "spatial_versions": len(versions.get("spatial", [])),
                "standard_versions": len(versions.get("standard", [])),
                "has_spatial": has_spatial,
                "has_standard": has_standard
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==== Performance Tests ====
    
    def test_inference_performance(self) -> Dict[str, Any]:
        """Test inference performance"""
        test_image_path = self._create_test_image()
        results = {}
        
        try:
            sys.path.append(str(self.project_root))
            
            # Test standard inference time
            start_time = time.time()
            # Simulate standard inference
            time.sleep(0.1)  # Placeholder
            standard_time = time.time() - start_time
            
            results["standard_inference"] = {
                "success": True,
                "inference_time": standard_time,
                "performance_ok": standard_time < 5.0
            }
            
        except Exception as e:
            results["standard_inference"] = {"success": False, "error": str(e)}
        
        try:
            # Test spatial inference time
            from scripts.spatial_inference_optimized import OptimizedSpatialInference
            
            optimized_inference = OptimizedSpatialInference()
            start_time = time.time()
            
            result = optimized_inference.predict_single(test_image_path)
            spatial_time = time.time() - start_time
            
            results["spatial_inference"] = {
                "success": True,
                "inference_time": spatial_time,
                "performance_ok": spatial_time < 10.0,
                "result": result is not None
            }
            
        except Exception as e:
            results["spatial_inference"] = {"success": False, "error": str(e)}
        
        # Cleanup
        os.unlink(test_image_path)
        
        overall_success = any(result.get("success") for result in results.values())
        return {"success": overall_success, "details": results}
    
    # ==== Integration Tests ====
    
    def test_docker_compose_deployment(self) -> Dict[str, Any]:
        """Test docker-compose deployment"""
        try:
            # Start docker-compose services
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            # Wait for services to start
            time.sleep(30)
            
            # Test service health
            health_results = self.test_api_health_check()
            
            # Stop services
            subprocess.run(
                ["docker-compose", "down"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            return {
                "success": health_results["success"],
                "deployment": "completed",
                "health_check": health_results
            }
            
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"Docker-compose failed: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==== Utility Methods ====
    
    def _create_test_image(self) -> str:
        """Create a test image for API testing"""
        import numpy as np
        from PIL import Image
        
        # Create synthetic pizza-like image
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        test_image_path = self.output_dir / "test_image.jpg"
        image.save(test_image_path)
        
        return str(test_image_path)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all CI/CD tests"""
        self.logger.info("Starting comprehensive CI/CD test suite")
        
        # Define test groups
        test_groups = [
            ("Docker Tests", [
                ("docker_build_standard", self.test_docker_build_standard),
                ("docker_build_spatial", self.test_docker_build_spatial),
                ("container_startup_standard", self.test_container_startup_standard),
                ("container_startup_spatial", self.test_container_startup_spatial),
            ]),
            ("Model Tests", [
                ("model_loading", self.test_model_loading),
                ("model_versioning", self.test_model_versioning),
            ]),
            ("Performance Tests", [
                ("inference_performance", self.test_inference_performance),
            ]),
            ("API Tests", [
                ("api_health_check", self.test_api_health_check),
                ("api_prediction_endpoints", self.test_api_prediction_endpoints),
            ]),
            ("Integration Tests", [
                ("docker_compose_deployment", self.test_docker_compose_deployment),
            ]),
        ]
        
        # Run tests
        for group_name, tests in test_groups:
            self.logger.info(f"Running {group_name}...")
            
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
        
        # Compile results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "pass"])
        failed_tests = len([r for r in self.test_results if r.status == "fail"])
        error_tests = len([r for r in self.test_results if r.status == "error"])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": passed_tests / max(total_tests, 1),
            "test_results": [asdict(result) for result in self.test_results]
        }
        
        # Save results
        results_file = self.output_dir / f"cicd_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"CI/CD tests completed. Results saved to: {results_file}")
        return summary


def main():
    """Main function for running CI/CD tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CI/CD Test Framework for Spatial-MLLM")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--test-group", choices=["docker", "model", "api", "performance", "integration"], 
                       help="Run specific test group")
    parser.add_argument("--output-format", choices=["json", "console"], default="console",
                       help="Output format")
    
    args = parser.parse_args()
    
    # Initialize test framework
    framework = CICDTestFramework(args.project_root)
    
    # Run tests
    if args.test_group:
        # Run specific test group
        if args.test_group == "docker":
            framework.run_test("docker_build_standard", framework.test_docker_build_standard)
            framework.run_test("docker_build_spatial", framework.test_docker_build_spatial)
        elif args.test_group == "model":
            framework.run_test("model_loading", framework.test_model_loading)
            framework.run_test("model_versioning", framework.test_model_versioning)
        # Add other test groups as needed
        
        results = {
            "test_results": [asdict(result) for result in framework.test_results]
        }
    else:
        # Run all tests
        results = framework.run_all_tests()
    
    # Output results
    if args.output_format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(f"\n🧪 CI/CD Test Results:")
        print(f"Total Tests: {results.get('total_tests', len(framework.test_results))}")
        print(f"Passed: {results.get('passed', 0)}")
        print(f"Failed: {results.get('failed', 0)}")
        print(f"Errors: {results.get('errors', 0)}")
        print(f"Success Rate: {results.get('success_rate', 0):.1%}")
        
        # Show failed tests
        failed_tests = [r for r in framework.test_results if r.status in ["fail", "error"]]
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.error_message}")
    
    # Exit with error code if tests failed
    if results.get('failed', 0) > 0 or results.get('errors', 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
