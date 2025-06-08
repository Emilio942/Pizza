#!/usr/bin/env python3
"""
SPATIAL-4.2: Performance Benchmarking for Spatial-MLLM Deployment

This script provides comprehensive performance benchmarking for Spatial-MLLM
deployment across different environments and configurations.
"""

import os
import sys
import json
import time
import torch
import psutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import gc
import threading
from contextlib import contextmanager

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark measurement result"""
    test_name: str
    environment: str
    model_type: str
    inference_time: float
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_utilization: float
    accuracy: Optional[float] = None
    throughput_images_per_sec: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class SystemInfo:
    """System configuration information"""
    cpu_model: str
    cpu_cores: int
    total_ram_gb: float
    gpu_model: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_version: Optional[str]
    pytorch_version: str
    python_version: str

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start monitoring system resources"""
        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.measurements:
            return {}
        
        cpu_usage = [m['cpu'] for m in self.measurements]
        memory_usage = [m['memory'] for m in self.measurements]
        gpu_memory = [m['gpu_memory'] for m in self.measurements if m['gpu_memory'] is not None]
        
        stats = {
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'measurement_count': len(self.measurements)
        }
        
        if gpu_memory:
            stats.update({
                'avg_gpu_memory_mb': np.mean(gpu_memory),
                'max_gpu_memory_mb': np.max(gpu_memory)
            })
        
        return stats
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent()
                memory_mb = psutil.virtual_memory().used / 1024 / 1024
                
                gpu_memory_mb = None
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                
                self.measurements.append({
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory': memory_mb,
                    'gpu_memory': gpu_memory_mb
                })
                
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break

class SpatialPerformanceBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, output_dir: str = "output/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.system_info = self._get_system_info()
        
        logger.info(f"🚀 Performance Benchmark initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU: {self.system_info.gpu_model}")
        logger.info(f"Output: {self.output_dir}")

    def _get_system_info(self) -> SystemInfo:
        """Collect system information"""
        import platform
        
        # CPU info
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count()
        total_ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        
        # GPU info
        gpu_model = None
        gpu_memory_gb = None
        cuda_version = None
        
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            cuda_version = torch.version.cuda
        
        return SystemInfo(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            total_ram_gb=total_ram_gb,
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory_gb,
            cuda_version=cuda_version,
            pytorch_version=torch.__version__,
            python_version=platform.python_version()
        )

    @contextmanager
    def _benchmark_context(self, test_name: str):
        """Context manager for benchmark measurements"""
        # Cleanup before test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Start monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        initial_gpu_memory = 0
        
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        try:
            yield
        finally:
            # Stop timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Stop monitoring
            monitor_stats = monitor.stop_monitoring()
            
            # Calculate final metrics
            final_memory = psutil.virtual_memory().used / 1024 / 1024
            final_gpu_memory = 0
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Store results
            self._current_result = {
                'inference_time': end_time - start_time,
                'memory_usage_mb': final_memory - initial_memory,
                'gpu_memory_mb': final_gpu_memory - initial_gpu_memory,
                'cpu_utilization': monitor_stats.get('avg_cpu_percent', 0),
                'monitor_stats': monitor_stats
            }

    def benchmark_model_loading(self, model_id: str = "Diankun/Spatial-MLLM-subset-sft") -> BenchmarkResult:
        """Benchmark model loading performance"""
        logger.info(f"🔄 Benchmarking model loading: {model_id}")
        
        try:
            with self._benchmark_context("model_loading"):
                from transformers import AutoModelForVision2Seq, AutoProcessor
                
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map={"": self.device} if self.device.type == "cuda" else None,
                )
                processor = AutoProcessor.from_pretrained(model_id)
            
            result = BenchmarkResult(
                test_name="model_loading",
                environment=f"{self.device}",
                model_type="spatial_mllm",
                inference_time=self._current_result['inference_time'],
                memory_usage_mb=self._current_result['memory_usage_mb'],
                gpu_memory_mb=self._current_result['gpu_memory_mb'],
                cpu_utilization=self._current_result['cpu_utilization']
            )
            
            logger.info(f"✅ Model loading: {result.inference_time:.2f}s, {result.gpu_memory_mb:.1f}MB GPU")
            
        except Exception as e:
            result = BenchmarkResult(
                test_name="model_loading",
                environment=f"{self.device}",
                model_type="spatial_mllm",
                inference_time=0,
                memory_usage_mb=0,
                gpu_memory_mb=0,
                cpu_utilization=0,
                error_message=str(e)
            )
            logger.error(f"❌ Model loading failed: {e}")
        
        self.results.append(result)
        return result

    def benchmark_single_inference(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark single image inference"""
        logger.info(f"🔄 Benchmarking single inference ({iterations} iterations)")
        
        try:
            # Load test image
            from PIL import Image
            test_data_dir = project_root / "data" / "test"
            test_images = list(test_data_dir.glob("*.jpg"))
            
            if not test_images:
                raise FileNotFoundError("No test images found")
            
            test_image = Image.open(test_images[0])
            
            # Load spatial inference system
            from scripts.spatial_inference_optimized import OptimizedSpatialInference
            inference = OptimizedSpatialInference()
            
            # Warm-up run
            inference.predict_single(test_image)
            
            # Benchmark runs
            inference_times = []
            
            for i in range(iterations):
                with self._benchmark_context(f"single_inference_{i}"):
                    result = inference.predict_single(test_image)
                
                inference_times.append(self._current_result['inference_time'])
            
            # Calculate statistics
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            throughput = 1.0 / avg_inference_time
            
            result = BenchmarkResult(
                test_name="single_inference",
                environment=f"{self.device}",
                model_type="spatial_mllm",
                inference_time=avg_inference_time,
                memory_usage_mb=self._current_result['memory_usage_mb'],
                gpu_memory_mb=self._current_result['gpu_memory_mb'],
                cpu_utilization=self._current_result['cpu_utilization'],
                throughput_images_per_sec=throughput
            )
            
            logger.info(f"✅ Single inference: {avg_inference_time:.3f}±{std_inference_time:.3f}s, {throughput:.1f} img/s")
            
        except Exception as e:
            result = BenchmarkResult(
                test_name="single_inference",
                environment=f"{self.device}",
                model_type="spatial_mllm",
                inference_time=0,
                memory_usage_mb=0,
                gpu_memory_mb=0,
                cpu_utilization=0,
                error_message=str(e)
            )
            logger.error(f"❌ Single inference failed: {e}")
        
        self.results.append(result)
        return result

    def benchmark_batch_inference(self, batch_sizes: List[int] = [1, 2, 4, 8]) -> List[BenchmarkResult]:
        """Benchmark batch inference with different batch sizes"""
        logger.info(f"🔄 Benchmarking batch inference: {batch_sizes}")
        
        batch_results = []
        
        # Load test images
        from PIL import Image
        test_data_dir = project_root / "data" / "test"
        test_images = list(test_data_dir.glob("*.jpg"))[:max(batch_sizes)]
        
        if len(test_images) < max(batch_sizes):
            logger.warning(f"Only {len(test_images)} test images available")
        
        for batch_size in batch_sizes:
            if batch_size > len(test_images):
                logger.warning(f"Skipping batch size {batch_size} - not enough test images")
                continue
            
            try:
                logger.info(f"Testing batch size: {batch_size}")
                
                # Prepare batch
                batch_images = [Image.open(img_path) for img_path in test_images[:batch_size]]
                
                # Load spatial inference system
                from scripts.spatial_inference_optimized import OptimizedSpatialInference
                inference = OptimizedSpatialInference()
                
                with self._benchmark_context(f"batch_inference_{batch_size}"):
                    results = inference.predict_batch(batch_images)
                
                throughput = batch_size / self._current_result['inference_time']
                
                result = BenchmarkResult(
                    test_name=f"batch_inference_size_{batch_size}",
                    environment=f"{self.device}",
                    model_type="spatial_mllm",
                    inference_time=self._current_result['inference_time'],
                    memory_usage_mb=self._current_result['memory_usage_mb'],
                    gpu_memory_mb=self._current_result['gpu_memory_mb'],
                    cpu_utilization=self._current_result['cpu_utilization'],
                    throughput_images_per_sec=throughput
                )
                
                logger.info(f"✅ Batch {batch_size}: {result.inference_time:.3f}s, {throughput:.1f} img/s")
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"batch_inference_size_{batch_size}",
                    environment=f"{self.device}",
                    model_type="spatial_mllm",
                    inference_time=0,
                    memory_usage_mb=0,
                    gpu_memory_mb=0,
                    cpu_utilization=0,
                    error_message=str(e)
                )
                logger.error(f"❌ Batch {batch_size} failed: {e}")
            
            batch_results.append(result)
            self.results.append(result)
        
        return batch_results

    def benchmark_memory_optimization(self) -> BenchmarkResult:
        """Benchmark memory-optimized inference"""
        logger.info(f"🔄 Benchmarking memory optimization")
        
        try:
            # Load test image
            from PIL import Image
            test_data_dir = project_root / "data" / "test"
            test_images = list(test_data_dir.glob("*.jpg"))
            
            if not test_images:
                raise FileNotFoundError("No test images found")
            
            test_image = Image.open(test_images[0])
            
            # Load memory-optimized inference
            from scripts.spatial_inference_memory_optimized import SpatialInferenceMemoryOptimized
            inference = SpatialInferenceMemoryOptimized()
            
            # Test multiple inferences to check memory stability
            max_memory = 0
            inference_times = []
            
            for i in range(5):
                with self._benchmark_context(f"memory_optimized_{i}"):
                    result = inference.predict(test_image)
                
                inference_times.append(self._current_result['inference_time'])
                max_memory = max(max_memory, self._current_result['gpu_memory_mb'])
            
            avg_inference_time = np.mean(inference_times)
            
            result = BenchmarkResult(
                test_name="memory_optimized_inference",
                environment=f"{self.device}",
                model_type="spatial_mllm_optimized",
                inference_time=avg_inference_time,
                memory_usage_mb=self._current_result['memory_usage_mb'],
                gpu_memory_mb=max_memory,
                cpu_utilization=self._current_result['cpu_utilization'],
                throughput_images_per_sec=1.0 / avg_inference_time
            )
            
            logger.info(f"✅ Memory optimized: {avg_inference_time:.3f}s, {max_memory:.1f}MB peak GPU")
            
        except Exception as e:
            result = BenchmarkResult(
                test_name="memory_optimized_inference",
                environment=f"{self.device}",
                model_type="spatial_mllm_optimized",
                inference_time=0,
                memory_usage_mb=0,
                gpu_memory_mb=0,
                cpu_utilization=0,
                error_message=str(e)
            )
            logger.error(f"❌ Memory optimization failed: {e}")
        
        self.results.append(result)
        return result

    def benchmark_api_endpoints(self, api_url: str = "http://localhost:8001") -> List[BenchmarkResult]:
        """Benchmark API endpoint performance"""
        logger.info(f"🔄 Benchmarking API endpoints: {api_url}")
        
        import requests
        from PIL import Image
        import io
        
        api_results = []
        
        # Test data
        test_data_dir = project_root / "data" / "test"
        test_images = list(test_data_dir.glob("*.jpg"))
        
        if not test_images:
            logger.warning("No test images found for API testing")
            return api_results
        
        endpoints = [
            ("/health", "GET", None),
            ("/predict/spatial", "POST", test_images[0]),
            ("/predict/standard", "POST", test_images[0])
        ]
        
        for endpoint, method, image_path in endpoints:
            try:
                logger.info(f"Testing {method} {endpoint}")
                
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(f"{api_url}{endpoint}", timeout=10)
                elif method == "POST" and image_path:
                    with open(image_path, "rb") as f:
                        files = {"file": f}
                        response = requests.post(f"{api_url}{endpoint}", files=files, timeout=30)
                else:
                    continue
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                result = BenchmarkResult(
                    test_name=f"api_{endpoint.replace('/', '_')}",
                    environment="api",
                    model_type="api_endpoint",
                    inference_time=inference_time,
                    memory_usage_mb=0,
                    gpu_memory_mb=0,
                    cpu_utilization=0,
                    accuracy=1.0 if response.status_code == 200 else 0.0
                )
                
                logger.info(f"✅ {endpoint}: {inference_time:.3f}s, status {response.status_code}")
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"api_{endpoint.replace('/', '_')}",
                    environment="api",
                    model_type="api_endpoint",
                    inference_time=0,
                    memory_usage_mb=0,
                    gpu_memory_mb=0,
                    cpu_utilization=0,
                    error_message=str(e)
                )
                logger.error(f"❌ {endpoint} failed: {e}")
            
            api_results.append(result)
            self.results.append(result)
        
        return api_results

    def run_comprehensive_benchmark(self, 
                                  iterations: int = 10,
                                  batch_sizes: List[int] = [1, 2, 4],
                                  include_api: bool = False,
                                  api_url: str = "http://localhost:8001") -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info(f"🚀 Starting comprehensive benchmark suite")
        
        benchmark_start = time.time()
        
        # Clear previous results
        self.results = []
        
        # Run benchmarks
        logger.info("📊 Running model loading benchmark...")
        self.benchmark_model_loading()
        
        logger.info("📊 Running single inference benchmark...")
        self.benchmark_single_inference(iterations)
        
        logger.info("📊 Running batch inference benchmark...")
        self.benchmark_batch_inference(batch_sizes)
        
        logger.info("📊 Running memory optimization benchmark...")
        self.benchmark_memory_optimization()
        
        if include_api:
            logger.info("📊 Running API endpoint benchmark...")
            self.benchmark_api_endpoints(api_url)
        
        benchmark_duration = time.time() - benchmark_start
        
        # Generate summary
        summary = self._generate_benchmark_summary(benchmark_duration)
        
        # Save results
        self._save_results(summary)
        
        logger.info(f"🏁 Benchmark completed in {benchmark_duration:.1f}s")
        
        return summary

    def _generate_benchmark_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary"""
        successful_results = [r for r in self.results if r.error_message is None]
        failed_results = [r for r in self.results if r.error_message is not None]
        
        summary = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "system_info": asdict(self.system_info),
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "failed_tests": len(failed_results),
            "success_rate": len(successful_results) / len(self.results) if self.results else 0,
            "performance_metrics": {},
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        if successful_results:
            # Calculate average metrics
            avg_inference_time = np.mean([r.inference_time for r in successful_results])
            avg_memory_usage = np.mean([r.memory_usage_mb for r in successful_results])
            avg_gpu_memory = np.mean([r.gpu_memory_mb for r in successful_results])
            avg_cpu_utilization = np.mean([r.cpu_utilization for r in successful_results])
            
            throughputs = [r.throughput_images_per_sec for r in successful_results if r.throughput_images_per_sec is not None]
            avg_throughput = np.mean(throughputs) if throughputs else 0
            
            summary["performance_metrics"] = {
                "avg_inference_time": avg_inference_time,
                "avg_memory_usage_mb": avg_memory_usage,
                "avg_gpu_memory_mb": avg_gpu_memory,
                "avg_cpu_utilization": avg_cpu_utilization,
                "avg_throughput_images_per_sec": avg_throughput,
                "best_single_inference_time": min(r.inference_time for r in successful_results),
                "peak_memory_usage_mb": max(r.memory_usage_mb for r in successful_results),
                "peak_gpu_memory_mb": max(r.gpu_memory_mb for r in successful_results)
            }
        
        return summary

    def _save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save human-readable summary
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("SPATIAL-MLLM PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Benchmark Date: {summary['benchmark_timestamp']}\n")
            f.write(f"Total Duration: {summary['total_duration']:.1f}s\n")
            f.write(f"Tests Run: {summary['total_tests']} ({summary['successful_tests']} successful)\n")
            f.write(f"Success Rate: {summary['success_rate']:.1%}\n\n")
            
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 20 + "\n")
            sys_info = summary['system_info']
            f.write(f"CPU: {sys_info['cpu_model']} ({sys_info['cpu_cores']} cores)\n")
            f.write(f"RAM: {sys_info['total_ram_gb']:.1f} GB\n")
            f.write(f"GPU: {sys_info['gpu_model']} ({sys_info['gpu_memory_gb']:.1f} GB)\n")
            f.write(f"CUDA: {sys_info['cuda_version']}\n")
            f.write(f"PyTorch: {sys_info['pytorch_version']}\n\n")
            
            if summary['performance_metrics']:
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")
                metrics = summary['performance_metrics']
                f.write(f"Average Inference Time: {metrics['avg_inference_time']:.3f}s\n")
                f.write(f"Best Inference Time: {metrics['best_single_inference_time']:.3f}s\n")
                f.write(f"Average Throughput: {metrics['avg_throughput_images_per_sec']:.1f} images/sec\n")
                f.write(f"Average Memory Usage: {metrics['avg_memory_usage_mb']:.1f} MB\n")
                f.write(f"Peak GPU Memory: {metrics['peak_gpu_memory_mb']:.1f} MB\n")
                f.write(f"Average CPU Utilization: {metrics['avg_cpu_utilization']:.1f}%\n\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            for result in summary['detailed_results']:
                f.write(f"{result['test_name']}: ")
                if result['error_message']:
                    f.write(f"FAILED - {result['error_message']}\n")
                else:
                    f.write(f"{result['inference_time']:.3f}s")
                    if result['throughput_images_per_sec']:
                        f.write(f" ({result['throughput_images_per_sec']:.1f} img/s)")
                    f.write(f" | GPU: {result['gpu_memory_mb']:.1f}MB\n")
        
        logger.info(f"📄 Results saved to: {json_file}")
        logger.info(f"📄 Summary saved to: {summary_file}")

def main():
    """Main function for performance benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPATIAL-4.2: Performance Benchmarking")
    parser.add_argument("--output-dir", type=str, default="output/benchmarks", help="Output directory")
    parser.add_argument("--iterations", type=int, default=10, help="Number of inference iterations")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4], help="Batch sizes to test")
    parser.add_argument("--include-api", action="store_true", help="Include API endpoint testing")
    parser.add_argument("--api-url", type=str, default="http://localhost:8001", help="API base URL")
    parser.add_argument("--include-memory-profiling", action="store_true", help="Include detailed memory profiling")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer iterations")
    
    args = parser.parse_args()
    
    if args.quick:
        args.iterations = 3
        args.batch_sizes = [1, 2]
    
    # Initialize benchmark
    benchmark = SpatialPerformanceBenchmark(args.output_dir)
    
    # Run comprehensive benchmark
    try:
        summary = benchmark.run_comprehensive_benchmark(
            iterations=args.iterations,
            batch_sizes=args.batch_sizes,
            include_api=args.include_api,
            api_url=args.api_url
        )
        
        # Print summary
        print(f"\n🏁 Benchmark completed successfully!")
        print(f"📊 {summary['successful_tests']}/{summary['total_tests']} tests passed")
        
        if summary['performance_metrics']:
            metrics = summary['performance_metrics']
            print(f"⚡ Best inference time: {metrics['best_single_inference_time']:.3f}s")
            print(f"🚀 Average throughput: {metrics['avg_throughput_images_per_sec']:.1f} images/sec")
            print(f"💾 Peak GPU memory: {metrics['peak_gpu_memory_mb']:.1f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
