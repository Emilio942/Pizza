#!/usr/bin/env python3
"""
Spatial-MLLM Integration Test Suite
Part of SPATIAL-4.2: Deployment-Pipeline erweitern

This script performs end-to-end integration testing of the complete
Spatial-MLLM deployment pipeline across different environments.
"""

import os
import sys
import json
import time
import requests
import asyncio
import aiohttp
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import docker
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class SpatialIntegrationTester:
    """Comprehensive integration tester for Spatial-MLLM deployment"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.test_results = {}
        self.logger = self._setup_logging()
        self.docker_client = docker.from_env()
        
        # Environment-specific configurations
        self.configs = {
            "development": {
                "api_base_url": "http://localhost:8001",
                "nginx_url": "http://localhost:80",
                "prometheus_url": "http://localhost:9090",
                "grafana_url": "http://localhost:3000",
                "timeout": 30
            },
            "staging": {
                "api_base_url": "http://staging.spatial.pizza.local:8001",
                "nginx_url": "http://staging.spatial.pizza.local",
                "prometheus_url": "http://staging.spatial.pizza.local:9090",
                "grafana_url": "http://staging.spatial.pizza.local:3000",
                "timeout": 60
            },
            "production": {
                "api_base_url": "https://api.spatial.pizza.local",
                "nginx_url": "https://spatial.pizza.local",
                "prometheus_url": "https://monitoring.spatial.pizza.local",
                "grafana_url": "https://grafana.spatial.pizza.local",
                "timeout": 45
            }
        }
        
        self.config = self.configs.get(environment, self.configs["development"])
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SpatialIntegrationTester')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def test_infrastructure_health(self) -> Dict[str, Any]:
        """Test basic infrastructure health"""
        self.logger.info("Testing infrastructure health...")
        
        results = {
            "containers_running": False,
            "services_accessible": False,
            "load_balancer_working": False,
            "monitoring_active": False,
            "details": {}
        }
        
        try:
            # Check Docker containers
            containers = self.docker_client.containers.list()
            spatial_containers = [c for c in containers if 'spatial' in c.name.lower()]
            results["details"]["container_count"] = len(spatial_containers)
            results["containers_running"] = len(spatial_containers) >= 3  # Expect at least 3 spatial containers
            
            # Test service accessibility
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # Test API health endpoint
                try:
                    async with session.get(f"{self.config['api_base_url']}/health") as response:
                        if response.status == 200:
                            results["details"]["api_health"] = "OK"
                        else:
                            results["details"]["api_health"] = f"ERROR_{response.status}"
                except Exception as e:
                    results["details"]["api_health"] = f"FAILED_{str(e)}"
                
                # Test load balancer
                try:
                    async with session.get(f"{self.config['nginx_url']}/health") as response:
                        if response.status == 200:
                            results["details"]["nginx_health"] = "OK"
                            results["load_balancer_working"] = True
                        else:
                            results["details"]["nginx_health"] = f"ERROR_{response.status}"
                except Exception as e:
                    results["details"]["nginx_health"] = f"FAILED_{str(e)}"
                
                # Test monitoring services
                try:
                    async with session.get(f"{self.config['prometheus_url']}/api/v1/query?query=up") as response:
                        if response.status == 200:
                            results["details"]["prometheus_health"] = "OK"
                            results["monitoring_active"] = True
                        else:
                            results["details"]["prometheus_health"] = f"ERROR_{response.status}"
                except Exception as e:
                    results["details"]["prometheus_health"] = f"FAILED_{str(e)}"
            
            # Overall service accessibility
            results["services_accessible"] = all([
                results["details"].get("api_health") == "OK",
                results["details"].get("nginx_health") == "OK"
            ])
            
        except Exception as e:
            self.logger.error(f"Infrastructure health test failed: {e}")
            results["details"]["error"] = str(e)
        
        return results
    
    async def test_api_functionality(self) -> Dict[str, Any]:
        """Test API functionality with actual requests"""
        self.logger.info("Testing API functionality...")
        
        results = {
            "basic_endpoints": False,
            "spatial_features": False,
            "file_upload": False,
            "load_balancing": False,
            "response_times": {},
            "details": {}
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                # Test basic endpoints
                basic_tests = []
                
                # Health check
                start_time = time.time()
                try:
                    async with session.get(f"{self.config['api_base_url']}/health") as response:
                        basic_tests.append(response.status == 200)
                        results["response_times"]["health"] = time.time() - start_time
                except Exception:
                    basic_tests.append(False)
                
                # API info
                start_time = time.time()
                try:
                    async with session.get(f"{self.config['api_base_url']}/api/info") as response:
                        basic_tests.append(response.status == 200)
                        results["response_times"]["info"] = time.time() - start_time
                except Exception:
                    basic_tests.append(False)
                
                results["basic_endpoints"] = all(basic_tests)
                
                # Test spatial features with synthetic data
                start_time = time.time()
                try:
                    # Create a simple test image data
                    test_data = {
                        "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...",  # Placeholder
                        "query": "What type of pizza is this?"
                    }
                    
                    async with session.post(
                        f"{self.config['api_base_url']}/api/spatial/analyze",
                        json=test_data
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            results["spatial_features"] = "response" in response_data
                            results["details"]["spatial_response"] = response_data
                        else:
                            results["details"]["spatial_error"] = response.status
                        
                        results["response_times"]["spatial_analyze"] = time.time() - start_time
                        
                except Exception as e:
                    results["details"]["spatial_exception"] = str(e)
                
                # Test load balancing by making multiple requests
                response_servers = set()
                for i in range(5):
                    try:
                        async with session.get(f"{self.config['api_base_url']}/api/server-info") as response:
                            if response.status == 200:
                                server_info = await response.json()
                                if "server_id" in server_info:
                                    response_servers.add(server_info["server_id"])
                    except Exception:
                        pass
                
                results["load_balancing"] = len(response_servers) > 1
                results["details"]["unique_servers"] = len(response_servers)
                
        except Exception as e:
            self.logger.error(f"API functionality test failed: {e}")
            results["details"]["error"] = str(e)
        
        return results
    
    async def test_model_performance(self) -> Dict[str, Any]:
        """Test model performance and accuracy"""
        self.logger.info("Testing model performance...")
        
        results = {
            "model_loading": False,
            "inference_working": False,
            "dual_encoder_functional": False,
            "performance_acceptable": False,
            "metrics": {}
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                # Test model status
                try:
                    async with session.get(f"{self.config['api_base_url']}/api/model/status") as response:
                        if response.status == 200:
                            status_data = await response.json()
                            results["model_loading"] = status_data.get("loaded", False)
                            results["details"] = status_data
                except Exception as e:
                    results["details"]["model_status_error"] = str(e)
                
                # Test inference performance
                test_cases = [
                    {"query": "What type of pizza is this?", "expected_keywords": ["pizza", "type"]},
                    {"query": "Describe the visual quality", "expected_keywords": ["quality", "visual"]},
                    {"query": "What ingredients can you see?", "expected_keywords": ["ingredients"]}
                ]
                
                inference_times = []
                successful_inferences = 0
                
                for i, test_case in enumerate(test_cases):
                    start_time = time.time()
                    try:
                        test_data = {
                            "image_data": self._generate_test_image_data(),
                            "query": test_case["query"]
                        }
                        
                        async with session.post(
                            f"{self.config['api_base_url']}/api/spatial/inference",
                            json=test_data
                        ) as response:
                            inference_time = time.time() - start_time
                            inference_times.append(inference_time)
                            
                            if response.status == 200:
                                response_data = await response.json()
                                successful_inferences += 1
                                
                                # Check if response contains expected keywords
                                response_text = response_data.get("response", "").lower()
                                keyword_matches = sum(1 for kw in test_case["expected_keywords"] 
                                                    if kw in response_text)
                                
                                results["metrics"][f"test_{i}_keywords"] = keyword_matches
                                results["metrics"][f"test_{i}_inference_time"] = inference_time
                                
                    except Exception as e:
                        results["details"][f"inference_error_{i}"] = str(e)
                
                # Calculate performance metrics
                if inference_times:
                    results["metrics"]["avg_inference_time"] = sum(inference_times) / len(inference_times)
                    results["metrics"]["max_inference_time"] = max(inference_times)
                    results["performance_acceptable"] = results["metrics"]["avg_inference_time"] < 5.0
                
                results["inference_working"] = successful_inferences > 0
                results["dual_encoder_functional"] = successful_inferences >= len(test_cases) * 0.8
                
        except Exception as e:
            self.logger.error(f"Model performance test failed: {e}")
            results["details"]["error"] = str(e)
        
        return results
    
    def _generate_test_image_data(self) -> str:
        """Generate base64 encoded test image data"""
        # This is a minimal placeholder - in real implementation, 
        # you would generate or load actual test images
        return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAAAAAAB/8QAFhEBAQEAAAAAAAAAAAAAAAAAAAER/9oADAMBAAIRAxEAPwCdABmLFrLYOjHQA="
    
    async def test_monitoring_integration(self) -> Dict[str, Any]:
        """Test monitoring and alerting integration"""
        self.logger.info("Testing monitoring integration...")
        
        results = {
            "prometheus_collecting": False,
            "grafana_accessible": False,
            "alerts_configured": False,
            "metrics_available": False,
            "details": {}
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # Test Prometheus metrics collection
                try:
                    async with session.get(f"{self.config['prometheus_url']}/api/v1/targets") as response:
                        if response.status == 200:
                            targets_data = await response.json()
                            active_targets = sum(1 for target in targets_data.get("data", {}).get("activeTargets", [])
                                               if target.get("health") == "up")
                            results["details"]["active_targets"] = active_targets
                            results["prometheus_collecting"] = active_targets > 0
                except Exception as e:
                    results["details"]["prometheus_error"] = str(e)
                
                # Test specific metrics availability
                test_metrics = [
                    "up{job=\"spatial-api\"}",
                    "http_requests_total{job=\"spatial-api\"}",
                    "model_inference_duration_seconds"
                ]
                
                available_metrics = 0
                for metric in test_metrics:
                    try:
                        async with session.get(f"{self.config['prometheus_url']}/api/v1/query?query={metric}") as response:
                            if response.status == 200:
                                metric_data = await response.json()
                                if metric_data.get("data", {}).get("result"):
                                    available_metrics += 1
                    except Exception:
                        pass
                
                results["metrics_available"] = available_metrics >= len(test_metrics) * 0.7
                results["details"]["available_metrics"] = f"{available_metrics}/{len(test_metrics)}"
                
                # Test Grafana accessibility
                try:
                    async with session.get(f"{self.config['grafana_url']}/api/health") as response:
                        results["grafana_accessible"] = response.status == 200
                        results["details"]["grafana_status"] = response.status
                except Exception as e:
                    results["details"]["grafana_error"] = str(e)
                
                # Test alert rules
                try:
                    async with session.get(f"{self.config['prometheus_url']}/api/v1/rules") as response:
                        if response.status == 200:
                            rules_data = await response.json()
                            rule_groups = rules_data.get("data", {}).get("groups", [])
                            total_rules = sum(len(group.get("rules", [])) for group in rule_groups)
                            results["alerts_configured"] = total_rules > 0
                            results["details"]["alert_rules_count"] = total_rules
                except Exception as e:
                    results["details"]["alerts_error"] = str(e)
                
        except Exception as e:
            self.logger.error(f"Monitoring integration test failed: {e}")
            results["details"]["error"] = str(e)
        
        return results
    
    async def test_multi_environment_compatibility(self) -> Dict[str, Any]:
        """Test compatibility across different environments"""
        self.logger.info("Testing multi-environment compatibility...")
        
        results = {
            "environment_config": self.environment,
            "docker_compatibility": False,
            "network_connectivity": False,
            "resource_requirements": False,
            "details": {}
        }
        
        try:
            # Test Docker compatibility
            try:
                # Check if all required images are available
                required_images = [
                    "spatial-mllm-api",
                    "nginx",
                    "prometheus/prometheus",
                    "grafana/grafana",
                    "redis"
                ]
                
                available_images = []
                for image_name in required_images:
                    try:
                        self.docker_client.images.get(image_name)
                        available_images.append(image_name)
                    except docker.errors.ImageNotFound:
                        pass
                
                results["details"]["available_images"] = f"{len(available_images)}/{len(required_images)}"
                results["docker_compatibility"] = len(available_images) >= len(required_images) * 0.8
                
            except Exception as e:
                results["details"]["docker_error"] = str(e)
            
            # Test network connectivity
            network_tests = []
            test_urls = [
                self.config["api_base_url"],
                self.config["nginx_url"],
                self.config["prometheus_url"]
            ]
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for url in test_urls:
                    try:
                        async with session.get(f"{url}/health", allow_redirects=True) as response:
                            network_tests.append(response.status in [200, 404])  # 404 is acceptable for health endpoint
                    except Exception:
                        network_tests.append(False)
            
            results["network_connectivity"] = any(network_tests)
            results["details"]["network_tests"] = f"{sum(network_tests)}/{len(network_tests)}"
            
            # Test resource requirements
            try:
                # Check available system resources
                import psutil
                
                memory_gb = psutil.virtual_memory().total / (1024**3)
                cpu_count = psutil.cpu_count()
                disk_gb = psutil.disk_usage('/').free / (1024**3)
                
                resource_requirements = {
                    "memory_sufficient": memory_gb >= 8,  # Minimum 8GB RAM
                    "cpu_sufficient": cpu_count >= 4,    # Minimum 4 CPU cores
                    "disk_sufficient": disk_gb >= 10     # Minimum 10GB free space
                }
                
                results["details"]["system_resources"] = {
                    "memory_gb": round(memory_gb, 2),
                    "cpu_count": cpu_count,
                    "disk_gb": round(disk_gb, 2)
                }
                
                results["resource_requirements"] = all(resource_requirements.values())
                
            except Exception as e:
                results["details"]["resource_error"] = str(e)
                
        except Exception as e:
            self.logger.error(f"Multi-environment compatibility test failed: {e}")
            results["details"]["error"] = str(e)
        
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        self.logger.info(f"Starting comprehensive integration tests for {self.environment} environment...")
        
        start_time = time.time()
        
        # Run all test suites
        test_suites = {
            "infrastructure": await self.test_infrastructure_health(),
            "api_functionality": await self.test_api_functionality(),
            "model_performance": await self.test_model_performance(),
            "monitoring": await self.test_monitoring_integration(),
            "multi_environment": await self.test_multi_environment_compatibility()
        }
        
        # Calculate overall results
        total_time = time.time() - start_time
        
        # Determine overall status
        critical_checks = [
            test_suites["infrastructure"].get("services_accessible", False),
            test_suites["api_functionality"].get("basic_endpoints", False),
            test_suites["model_performance"].get("inference_working", False),
            test_suites["multi_environment"].get("network_connectivity", False)
        ]
        
        overall_status = "PASSED" if all(critical_checks) else "FAILED"
        
        # Compile final report
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "overall_status": overall_status,
            "total_duration_seconds": round(total_time, 2),
            "test_suites": test_suites,
            "summary": {
                "total_tests": len(test_suites),
                "passed_tests": sum(1 for suite in test_suites.values() 
                                  if any(v for k, v in suite.items() if isinstance(v, bool) and v)),
                "critical_checks_passed": sum(critical_checks),
                "critical_checks_total": len(critical_checks)
            },
            "recommendations": []
        }
        
        # Generate recommendations based on failed tests
        if not test_suites["infrastructure"].get("containers_running", False):
            final_report["recommendations"].append("Start all required Docker containers")
        
        if not test_suites["api_functionality"].get("load_balancing", False):
            final_report["recommendations"].append("Check load balancer configuration")
        
        if not test_suites["model_performance"].get("performance_acceptable", False):
            final_report["recommendations"].append("Optimize model performance")
        
        if not test_suites["monitoring"].get("prometheus_collecting", False):
            final_report["recommendations"].append("Fix Prometheus metrics collection")
        
        self.logger.info(f"Integration tests completed: {overall_status}")
        self.logger.info(f"Total duration: {total_time:.2f} seconds")
        
        return final_report
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save test report to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Integration test report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Spatial-MLLM integration tests')
    parser.add_argument('--environment', choices=['development', 'staging', 'production'], 
                       default='development', help='Target environment')
    parser.add_argument('--output-report', help='Output path for test report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = SpatialIntegrationTester(args.environment)
    
    # Run tests
    report = await tester.run_comprehensive_tests()
    
    # Save report if requested
    if args.output_report:
        tester.save_report(report, args.output_report)
    else:
        # Print report to stdout
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    if report['overall_status'] == 'PASSED':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
