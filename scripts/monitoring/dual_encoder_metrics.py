#!/usr/bin/env python3
"""
Dual-Encoder Performance Metrics System
Part of SPATIAL-4.3: Monitoring und Logging erweitern

This module implements comprehensive performance monitoring for the dual-encoder
architecture, including visual encoder, spatial encoder, and connector metrics.
"""

import time
import psutil
import torch
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict, deque
import numpy as np
import json
from pathlib import Path

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def time(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def time(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Info:
        def __init__(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
    
    class CollectorRegistry:
        def __init__(self):
            pass
    
    REGISTRY = CollectorRegistry()


@dataclass
class EncoderMetrics:
    """Individual encoder performance metrics"""
    encoder_name: str
    inference_time: float
    memory_usage: float
    input_shape: tuple
    output_shape: tuple
    parameter_count: int
    flops: Optional[float] = None
    throughput: float = 0.0
    efficiency_score: float = 0.0


@dataclass
class ConnectorMetrics:
    """Connector performance between visual and spatial encoders"""
    processing_time: float
    input_visual_size: int
    input_spatial_size: int
    output_size: int
    fusion_method: str
    memory_overhead: float
    transformation_loss: float


@dataclass
class BatchMetrics:
    """Batch processing performance metrics"""
    batch_size: int
    total_time: float
    per_sample_time: float
    memory_peak: float
    gpu_utilization: float
    cpu_utilization: float
    throughput: float
    bottleneck_component: str


class GPUMonitor:
    """GPU performance monitoring"""
    
    def __init__(self):
        self.available = NVIDIA_ML_AVAILABLE
        if self.available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
            except:
                self.available = False
                self.device_count = 0
    
    def get_gpu_metrics(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU metrics for specified device"""
        if not self.available or device_id >= self.device_count:
            return {}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used / 1024**3  # GB
            memory_total = mem_info.total / 1024**3  # GB
            memory_util = (mem_info.used / mem_info.total) * 100
            
            # GPU utilization
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_info.gpu
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            
            return {
                'memory_used_gb': memory_used,
                'memory_total_gb': memory_total,
                'memory_utilization': memory_util,
                'gpu_utilization': gpu_util,
                'temperature': temp,
                'power_watts': power
            }
            
        except Exception as e:
            return {'error': str(e)}


class DualEncoderProfiler:
    """Profiler for dual-encoder architecture performance"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.gpu_monitor = GPUMonitor()
        
        # Performance history
        self.encoder_history = defaultdict(lambda: deque(maxlen=1000))
        self.connector_history = deque(maxlen=1000)
        self.batch_history = deque(maxlen=1000)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.setup_prometheus_metrics()
        
        # Baseline measurements
        self.baseline_metrics = {}
        self.performance_targets = {
            'visual_encoder_time': 0.05,  # 50ms target
            'spatial_encoder_time': 0.03,  # 30ms target
            'connector_time': 0.01,       # 10ms target
            'total_inference_time': 0.1,  # 100ms target
            'memory_efficiency': 0.8,     # 80% efficiency target
            'throughput': 10.0            # 10 samples/sec target
        }
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for dual-encoder monitoring"""
        # Encoder latency metrics
        self.encoder_latency = Histogram(
            'dual_encoder_latency_seconds',
            'Latency for individual encoders',
            ['encoder_name', 'model_version'],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Memory usage metrics
        self.encoder_memory = Gauge(
            'dual_encoder_memory_usage_bytes',
            'Memory usage for encoders',
            ['encoder_name', 'memory_type']
        )
        
        # Throughput metrics
        self.encoder_throughput = Gauge(
            'dual_encoder_throughput_samples_per_second',
            'Throughput for encoder processing',
            ['encoder_name']
        )
        
        # Efficiency metrics
        self.encoder_efficiency = Gauge(
            'dual_encoder_efficiency_score',
            'Efficiency score for encoder (0-1)',
            ['encoder_name']
        )
        
        # Connector metrics
        self.connector_overhead = Histogram(
            'connector_processing_seconds',
            'Time spent in connector between encoders',
            ['fusion_method'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        # GPU metrics
        self.gpu_utilization = Gauge(
            'dual_encoder_gpu_utilization_percent',
            'GPU utilization during dual-encoder processing',
            ['device_id', 'component']
        )
        
        # Performance target achievement
        self.target_achievement = Gauge(
            'dual_encoder_target_achievement_ratio',
            'Ratio of actual performance to target',
            ['metric_name']
        )
    
    def profile_encoder(self, encoder_name: str, model_version: str = "v1") -> Callable:
        """Decorator for profiling individual encoders"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Pre-execution metrics
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                gpu_metrics_start = self.gpu_monitor.get_gpu_metrics()
                
                try:
                    # Execute encoder
                    result = func(*args, **kwargs)
                    
                    # Post-execution metrics
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    gpu_metrics_end = self.gpu_monitor.get_gpu_metrics()
                    
                    # Calculate metrics
                    inference_time = end_time - start_time
                    memory_usage = end_memory.get('rss', 0) - start_memory.get('rss', 0)
                    
                    # Determine input/output shapes
                    input_shape = self._extract_shape(args, kwargs)
                    output_shape = self._extract_shape([result])
                    
                    # Create encoder metrics
                    metrics = EncoderMetrics(
                        encoder_name=encoder_name,
                        inference_time=inference_time,
                        memory_usage=memory_usage,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        parameter_count=self._count_parameters(func),
                        throughput=1.0 / inference_time if inference_time > 0 else 0,
                        efficiency_score=self._calculate_efficiency(inference_time, memory_usage)
                    )
                    
                    # Store metrics
                    self.encoder_history[encoder_name].append(metrics)
                    
                    # Update Prometheus metrics
                    if self.enable_prometheus:
                        self.encoder_latency.labels(
                            encoder_name=encoder_name,
                            model_version=model_version
                        ).observe(inference_time)
                        
                        self.encoder_memory.labels(
                            encoder_name=encoder_name,
                            memory_type='rss'
                        ).set(memory_usage)
                        
                        self.encoder_throughput.labels(
                            encoder_name=encoder_name
                        ).set(metrics.throughput)
                        
                        self.encoder_efficiency.labels(
                            encoder_name=encoder_name
                        ).set(metrics.efficiency_score)
                        
                        # Update target achievement
                        target_key = f"{encoder_name}_time"
                        if target_key in self.performance_targets:
                            achievement = self.performance_targets[target_key] / inference_time
                            self.target_achievement.labels(
                                metric_name=target_key
                            ).set(min(achievement, 2.0))  # Cap at 2x target
                    
                    return result
                    
                except Exception as e:
                    # Log error metrics
                    error_metrics = EncoderMetrics(
                        encoder_name=encoder_name,
                        inference_time=-1,
                        memory_usage=-1,
                        input_shape=(),
                        output_shape=(),
                        parameter_count=0
                    )
                    self.encoder_history[encoder_name].append(error_metrics)
                    raise e
            
            return wrapper
        return decorator
    
    def profile_connector(self, fusion_method: str = "concatenation") -> Callable:
        """Decorator for profiling connector between encoders"""
        def decorator(func):
            def wrapper(visual_features, spatial_features, *args, **kwargs):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                # Input sizes
                visual_size = self._get_tensor_size(visual_features)
                spatial_size = self._get_tensor_size(spatial_features)
                
                try:
                    result = func(visual_features, spatial_features, *args, **kwargs)
                    
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    
                    # Calculate metrics
                    processing_time = end_time - start_time
                    memory_overhead = end_memory.get('rss', 0) - start_memory.get('rss', 0)
                    output_size = self._get_tensor_size(result)
                    
                    # Calculate transformation loss (simplified)
                    transformation_loss = self._calculate_transformation_loss(
                        visual_features, spatial_features, result
                    )
                    
                    metrics = ConnectorMetrics(
                        processing_time=processing_time,
                        input_visual_size=visual_size,
                        input_spatial_size=spatial_size,
                        output_size=output_size,
                        fusion_method=fusion_method,
                        memory_overhead=memory_overhead,
                        transformation_loss=transformation_loss
                    )
                    
                    self.connector_history.append(metrics)
                    
                    # Update Prometheus metrics
                    if self.enable_prometheus:
                        self.connector_overhead.labels(
                            fusion_method=fusion_method
                        ).observe(processing_time)
                    
                    return result
                    
                except Exception as e:
                    raise e
            
            return wrapper
        return decorator
    
    def profile_batch(self, batch_size: int) -> Callable:
        """Decorator for profiling batch processing"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                gpu_start = self.gpu_monitor.get_gpu_metrics()
                cpu_start = psutil.cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    gpu_end = self.gpu_monitor.get_gpu_metrics()
                    cpu_end = psutil.cpu_percent()
                    
                    # Calculate metrics
                    total_time = end_time - start_time
                    per_sample_time = total_time / batch_size if batch_size > 0 else 0
                    memory_peak = max(end_memory.get('rss', 0), start_memory.get('rss', 0))
                    
                    gpu_util = gpu_end.get('gpu_utilization', 0) if gpu_end else 0
                    cpu_util = (cpu_start + cpu_end) / 2
                    
                    throughput = batch_size / total_time if total_time > 0 else 0
                    
                    # Identify bottleneck
                    bottleneck = self._identify_bottleneck(
                        cpu_util, gpu_util, memory_peak, total_time
                    )
                    
                    metrics = BatchMetrics(
                        batch_size=batch_size,
                        total_time=total_time,
                        per_sample_time=per_sample_time,
                        memory_peak=memory_peak,
                        gpu_utilization=gpu_util,
                        cpu_utilization=cpu_util,
                        throughput=throughput,
                        bottleneck_component=bottleneck
                    )
                    
                    self.batch_history.append(metrics)
                    
                    # Update Prometheus metrics
                    if self.enable_prometheus:
                        self.gpu_utilization.labels(
                            device_id='0',
                            component='batch_processing'
                        ).set(gpu_util)
                        
                        # Update throughput target achievement
                        throughput_achievement = throughput / self.performance_targets['throughput']
                        self.target_achievement.labels(
                            metric_name='throughput'
                        ).set(min(throughput_achievement, 2.0))
                    
                    return result
                    
                except Exception as e:
                    raise e
            
            return wrapper
        return decorator
    
    def profile_batch_processing(self, batch_size: int):
        """Context manager for profiling batch processing"""
        class BatchProfilingContext:
            def __init__(self, profiler, batch_size):
                self.profiler = profiler
                self.batch_size = batch_size
                self.start_time = None
                self.start_memory = None
                self.gpu_start = None
                self.cpu_start = None
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                self.start_memory = self.profiler._get_memory_usage()
                self.gpu_start = self.profiler.gpu_monitor.get_gpu_metrics()
                self.cpu_start = psutil.cpu_percent()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                end_memory = self.profiler._get_memory_usage()
                gpu_end = self.profiler.gpu_monitor.get_gpu_metrics()
                cpu_end = psutil.cpu_percent()
                
                # Calculate metrics
                total_time = end_time - self.start_time
                per_sample_time = total_time / self.batch_size if self.batch_size > 0 else 0
                memory_peak = max(end_memory.get('rss', 0), self.start_memory.get('rss', 0))
                
                gpu_util = gpu_end.get('gpu_utilization', 0) if gpu_end else 0
                cpu_util = (self.cpu_start + cpu_end) / 2
                
                throughput = self.batch_size / total_time if total_time > 0 else 0
                
                # Identify bottleneck
                bottleneck = self.profiler._identify_bottleneck(
                    cpu_util, gpu_util, memory_peak, total_time
                )
                
                metrics = BatchMetrics(
                    batch_size=self.batch_size,
                    total_time=total_time,
                    per_sample_time=per_sample_time,
                    memory_peak=memory_peak,
                    gpu_utilization=gpu_util,
                    cpu_utilization=cpu_util,
                    throughput=throughput,
                    bottleneck_component=bottleneck
                )
                
                self.profiler.batch_history.append(metrics)
                
                # Update Prometheus metrics
                if self.profiler.enable_prometheus:
                    self.profiler.gpu_utilization.labels(
                        device_id='0',
                        component='batch_processing'
                    ).set(gpu_util)
                    
                    # Update throughput target achievement
                    throughput_achievement = throughput / self.profiler.performance_targets['throughput']
                    self.profiler.target_achievement.labels(
                        metric_name='throughput'
                    ).set(min(throughput_achievement, 2.0))
                
        return BatchProfilingContext(self, batch_size)
    
    def log_visual_encoder_performance(self, duration: float, gpu_utilization: float):
        """Log visual encoder performance metrics"""
        try:
            # Create encoder metrics for visual encoder
            metrics = EncoderMetrics(
                encoder_name="visual_encoder",
                inference_time=duration,
                memory_usage=self._get_memory_usage().get('rss', 0),
                input_shape=(),  # Shape will be filled if available
                output_shape=(),  # Shape will be filled if available
                parameter_count=0,  # Will be filled if model available
                throughput=1.0 / duration if duration > 0 else 0,
                efficiency_score=self._calculate_efficiency(duration, gpu_utilization)
            )
            
            # Store metrics
            if "visual_encoder" not in self.encoder_history:
                self.encoder_history["visual_encoder"] = deque(maxlen=self.max_history_size)
            self.encoder_history["visual_encoder"].append(metrics)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.encoder_latency.labels(
                    encoder_name="visual_encoder",
                    model_version="v2.1"
                ).observe(duration)
                
                self.encoder_throughput.labels(
                    encoder_name="visual_encoder"
                ).set(metrics.throughput)
                
                self.encoder_efficiency.labels(
                    encoder_name="visual_encoder"
                ).set(metrics.efficiency_score)
                
        except Exception as e:
            print(f"Warning: Failed to log visual encoder performance: {e}")
    
    def log_spatial_encoder_performance(self, duration: float, gpu_utilization: float):
        """Log spatial encoder performance metrics"""
        try:
            # Create encoder metrics for spatial encoder
            metrics = EncoderMetrics(
                encoder_name="spatial_encoder",
                inference_time=duration,
                memory_usage=self._get_memory_usage().get('rss', 0),
                input_shape=(),  # Shape will be filled if available
                output_shape=(),  # Shape will be filled if available
                parameter_count=0,  # Will be filled if model available
                throughput=1.0 / duration if duration > 0 else 0,
                efficiency_score=self._calculate_efficiency(duration, gpu_utilization)
            )
            
            # Store metrics
            if "spatial_encoder" not in self.encoder_history:
                self.encoder_history["spatial_encoder"] = deque(maxlen=self.max_history_size)
            self.encoder_history["spatial_encoder"].append(metrics)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.encoder_latency.labels(
                    encoder_name="spatial_encoder",
                    model_version="v2.1"
                ).observe(duration)
                
                self.encoder_throughput.labels(
                    encoder_name="spatial_encoder"
                ).set(metrics.throughput)
                
                self.encoder_efficiency.labels(
                    encoder_name="spatial_encoder"
                ).set(metrics.efficiency_score)
                
        except Exception as e:
            print(f"Warning: Failed to log spatial encoder performance: {e}")

    def log_connector_performance(self, duration: float, fusion_method: str = "attention_fusion"):
        """Log connector performance metrics between encoders"""
        try:
            # Create connector metrics
            memory_stats = self._get_memory_usage()
            
            metrics = ConnectorMetrics(
                processing_time=duration,
                input_visual_size=0,  # Will be filled if size data available
                input_spatial_size=0,  # Will be filled if size data available
                output_size=0,  # Will be filled if size data available
                fusion_method=fusion_method,
                memory_overhead=memory_stats.get('rss', 0),
                transformation_loss=0.0   # Minimal loss assumed
            )
            
            # Store metrics
            self.connector_history.append(metrics)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.connector_overhead.labels(
                    fusion_method=fusion_method
                ).observe(duration)
                
                # Update efficiency based on processing time vs target
                target_time = self.performance_targets.get('connector_time', 0.01)
                efficiency = min(1.0, target_time / max(duration, 0.001))
                
                self.encoder_efficiency.labels(
                    encoder_name="connector"
                ).set(efficiency)
                
        except Exception as e:
            print(f"Warning: Failed to log connector performance: {e}")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_stats = {}
        
        if torch.cuda.is_available():
            try:
                memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
            except Exception:
                memory_stats['gpu_allocated'] = 0
                memory_stats['gpu_reserved'] = 0
                memory_stats['gpu_max_allocated'] = 0
        
        # System memory
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_stats['rss'] = memory_info.rss / 1024**3  # GB
            memory_stats['vms'] = memory_info.vms / 1024**3  # GB
        except Exception:
            memory_stats['rss'] = 0
            memory_stats['vms'] = 0
        
        return memory_stats

    def _calculate_efficiency(self, inference_time: float, memory_usage: float) -> float:
        """Calculate efficiency score based on time and memory usage"""
        try:
            # Normalize based on targets (lower is better for time and memory)
            time_efficiency = min(1.0, self.performance_targets.get('visual_encoder_time', 0.05) / max(inference_time, 0.001))
            memory_efficiency = min(1.0, 1.0 / max(memory_usage / 1024**3, 0.001))  # Assume 1GB is baseline
            
            # Weighted average (favor time efficiency)
            efficiency = 0.7 * time_efficiency + 0.3 * memory_efficiency
            return max(0.0, min(1.0, efficiency))
        except Exception:
            return 0.5  # Default efficiency

    def _extract_shape(self, data, additional_data=None) -> tuple:
        """Extract tensor shapes from input data"""
        try:
            shapes = []
            
            # Process main data
            if isinstance(data, (list, tuple)):
                for item in data:
                    if hasattr(item, 'shape'):
                        shapes.append(tuple(item.shape))
                    elif isinstance(item, (int, float)):
                        shapes.append((1,))
            elif hasattr(data, 'shape'):
                shapes.append(tuple(data.shape))
            
            # Process additional data (kwargs)
            if additional_data and isinstance(additional_data, dict):
                for value in additional_data.values():
                    if hasattr(value, 'shape'):
                        shapes.append(tuple(value.shape))
            
            return tuple(shapes) if shapes else ()
        except Exception:
            return ()

    def _count_parameters(self, func) -> int:
        """Count parameters in a function/model (simplified)"""
        try:
            # If it's a torch module, count parameters
            if hasattr(func, '__self__') and hasattr(func.__self__, 'parameters'):
                return sum(p.numel() for p in func.__self__.parameters())
            return 0
        except Exception:
            return 0

    def _get_tensor_size(self, tensor) -> int:
        """Get tensor size in bytes"""
        try:
            if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                return tensor.numel() * tensor.element_size()
            elif hasattr(tensor, 'nbytes'):
                return tensor.nbytes
            return 0
        except Exception:
            return 0

    def _calculate_transformation_loss(self, visual_features, spatial_features, result) -> float:
        """Calculate information loss during feature transformation (simplified)"""
        try:
            # Simple heuristic: compare input and output magnitudes
            if hasattr(visual_features, 'norm') and hasattr(spatial_features, 'norm') and hasattr(result, 'norm'):
                input_magnitude = (visual_features.norm() + spatial_features.norm()) / 2
                output_magnitude = result.norm()
                loss = abs(input_magnitude - output_magnitude) / max(input_magnitude, 0.001)
                return min(1.0, loss.item() if hasattr(loss, 'item') else loss)
            return 0.0
        except Exception:
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'encoder_stats': {},
                'connector_stats': {},
                'batch_stats': {},
                'performance_targets': self.performance_targets,
                'target_achievements': {}
            }
            
            # Encoder statistics
            for encoder_name, history in self.encoder_history.items():
                if history:
                    times = [m.inference_time for m in history if m.inference_time > 0]
                    memory_usage = [m.memory_usage for m in history if m.memory_usage > 0]
                    throughputs = [m.throughput for m in history if hasattr(m, 'throughput') and m.throughput > 0]
                    
                    summary['encoder_stats'][encoder_name] = {
                        'count': len(history),
                        'avg_inference_time': sum(times) / len(times) if times else 0,
                        'min_inference_time': min(times) if times else 0,
                        'max_inference_time': max(times) if times else 0,
                        'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                        'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0
                    }
            
            # Connector statistics
            if self.connector_history:
                times = [m.processing_time for m in self.connector_history]
                memory_overhead = [m.memory_overhead for m in self.connector_history if hasattr(m, 'memory_overhead')]
                
                summary['connector_stats'] = {
                    'count': len(self.connector_history),
                    'avg_processing_time': sum(times) / len(times) if times else 0,
                    'min_processing_time': min(times) if times else 0,
                    'max_processing_time': max(times) if times else 0,
                    'avg_memory_overhead': sum(memory_overhead) / len(memory_overhead) if memory_overhead else 0
                }
            
            # Batch statistics
            if self.batch_history:
                times = [m.total_processing_time for m in self.batch_history]
                throughputs = [m.throughput for m in self.batch_history]
                
                summary['batch_stats'] = {
                    'count': len(self.batch_history),
                    'avg_processing_time': sum(times) / len(times) if times else 0,
                    'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0
                }
            
            return summary
        except Exception as e:
            return {'error': f'Failed to generate performance summary: {e}'}

    # Add max_history_size property if not defined
    @property
    def max_history_size(self) -> int:
        """Maximum number of metrics to keep in history"""
        return 1000

    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        metrics_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'encoder_history': {
                name: [asdict(m) for m in deque_data]
                for name, deque_data in self.encoder_history.items()
            },
            'connector_history': [asdict(m) for m in self.connector_history],
            'batch_history': [asdict(m) for m in self.batch_history],
            'performance_summary': self.get_performance_summary(),
            'performance_targets': self.performance_targets
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

    def _identify_bottleneck(self, cpu_util: float, gpu_util: float, memory_peak: float, total_time: float) -> str:
        """Identify the primary bottleneck based on system metrics"""
        try:
            # Define thresholds for bottleneck detection
            HIGH_CPU_THRESHOLD = 80.0    # 80% CPU utilization
            HIGH_GPU_THRESHOLD = 80.0    # 80% GPU utilization  
            HIGH_MEMORY_THRESHOLD = 8.0  # 8GB memory usage
            SLOW_TIME_THRESHOLD = 1.0    # 1 second total time
            
            # Check for memory bottleneck first (most critical)
            if memory_peak > HIGH_MEMORY_THRESHOLD:
                return "memory"
            
            # Check for GPU bottleneck
            if gpu_util > HIGH_GPU_THRESHOLD:
                if cpu_util < HIGH_CPU_THRESHOLD:
                    return "gpu"
                else:
                    return "gpu_cpu"  # Both GPU and CPU are high
            
            # Check for CPU bottleneck
            if cpu_util > HIGH_CPU_THRESHOLD:
                return "cpu"
            
            # Check for I/O or other bottleneck (slow but low resource usage)
            if total_time > SLOW_TIME_THRESHOLD and cpu_util < 50.0 and gpu_util < 50.0:
                return "io"
            
            # If moderate resource usage across the board
            if cpu_util > 50.0 and gpu_util > 50.0:
                return "balanced"
            
            # Default case - no clear bottleneck identified
            return "unknown"
            
        except Exception as e:
            # Return a safe default in case of any errors
            return "unknown"


# Global profiler instance
_profiler_instance = None

def get_dual_encoder_profiler(**kwargs) -> DualEncoderProfiler:
    """Get or create dual-encoder profiler singleton"""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = DualEncoderProfiler(**kwargs)
    return _profiler_instance


if __name__ == "__main__":
    # Demo usage
    profiler = get_dual_encoder_profiler()
    
    # Example encoder functions
    @profiler.profile_encoder("visual_encoder", "v2.1")
    def mock_visual_encoder(image_tensor):
        time.sleep(0.02)  # Simulate processing
        return torch.randn(1, 512)
    
    @profiler.profile_encoder("spatial_encoder", "v2.1")
    def mock_spatial_encoder(spatial_tensor):
        time.sleep(0.015)  # Simulate processing
        return torch.randn(1, 256)
    
    @profiler.profile_connector("attention_fusion")
    def mock_connector(visual_features, spatial_features):
        time.sleep(0.005)  # Simulate processing
        return torch.cat([visual_features, spatial_features], dim=1)
    
    @profiler.profile_batch(batch_size=4)
    def mock_batch_process(batch_data):
        time.sleep(0.1)  # Simulate batch processing
        return [torch.randn(1, 768) for _ in range(4)]
    
    # Run demo
    print("🔧 Running dual-encoder profiler demo...")
    
    for i in range(5):
        # Simulate processing
        image = torch.randn(1, 3, 224, 224)
        spatial = torch.randn(1, 4, 224, 224)
        
        visual_out = mock_visual_encoder(image)
        spatial_out = mock_spatial_encoder(spatial)
        fused_out = mock_connector(visual_out, spatial_out)
        
        batch_out = mock_batch_process([image] * 4)
    
    # Get performance summary
    summary = profiler.get_performance_summary()
    print("📊 Performance Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Export metrics
    profiler.export_metrics("output/dual_encoder_metrics.json")
    print("✅ Metrics exported to output/dual_encoder_metrics.json")
