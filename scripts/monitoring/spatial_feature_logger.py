#!/usr/bin/env python3
"""
Enhanced Spatial Feature Logging System
Part of SPATIAL-4.3: Monitoring und Logging erweitern

This module implements comprehensive logging for spatial feature extraction,
dual-encoder performance metrics, and anomaly detection for spatial features.
"""

import logging
import json
import time
import numpy as np
import torch
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import deque
import pickle
from contextlib import contextmanager

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
    
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
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
    
    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def time(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class CollectorRegistry:
        def __init__(self):
            pass
    
    REGISTRY = CollectorRegistry()


@dataclass
class SpatialFeatureMetrics:
    """Metrics for spatial feature extraction"""
    timestamp: str
    image_id: str
    extraction_time: float
    feature_dimensions: Tuple[int, ...]
    feature_quality: float
    spatial_complexity: float
    visual_encoder_time: float
    spatial_encoder_time: float
    connector_time: float
    memory_usage: float
    gpu_utilization: float
    error_count: int = 0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class DualEncoderPerformance:
    """Performance metrics for dual-encoder architecture"""
    timestamp: str
    session_id: str
    visual_encoder_latency: float
    spatial_encoder_latency: float
    total_inference_time: float
    visual_feature_size: int
    spatial_feature_size: int
    connector_overhead: float
    batch_size: int
    model_accuracy: float
    confidence_score: float
    throughput: float
    resource_efficiency: float


@dataclass
class SpatialAnomaly:
    """Anomaly detection results for spatial features"""
    timestamp: str
    image_id: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    confidence: float
    spatial_location: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    feature_deviation: float
    expected_range: Tuple[float, float]
    actual_value: float
    recommendation: str


class SpatialMetricsCollector:
    """Prometheus metrics collector for spatial features"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.setup_metrics()
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # Feature extraction metrics
        self.feature_extraction_time = Histogram(
            'spatial_feature_extraction_seconds',
            'Time spent on spatial feature extraction',
            ['encoder_type', 'image_type'],
            registry=self.registry
        )
        
        self.feature_quality = Gauge(
            'spatial_feature_quality',
            'Quality metric for extracted spatial features',
            ['image_id', 'feature_type'],
            registry=self.registry
        )
        
        # Dual-encoder performance
        self.dual_encoder_latency = Histogram(
            'dual_encoder_inference_seconds',
            'Dual-encoder inference latency',
            ['encoder', 'batch_size'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'spatial_model_accuracy',
            'Current model accuracy score',
            ['model_version', 'dataset'],
            registry=self.registry
        )
        
        # Anomaly detection
        self.anomaly_count = Counter(
            'spatial_anomalies_total',
            'Total number of spatial anomalies detected',
            ['severity', 'type'],
            registry=self.registry
        )
        
        # Resource utilization
        self.gpu_utilization = Gauge(
            'spatial_gpu_utilization_percent',
            'GPU utilization for spatial processing',
            ['device_id'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'spatial_memory_usage_bytes',
            'Memory usage for spatial processing',
            ['memory_type'],
            registry=self.registry
        )


class SpatialLogger:
    """Enhanced logging system for spatial features"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 metrics_file: Optional[str] = None,
                 enable_prometheus: bool = True):
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file or "logs/spatial_features.log"
        self.metrics_file = metrics_file or "logs/spatial_metrics.json"
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Create directories
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup metrics collection
        if self.enable_prometheus:
            self.metrics_collector = SpatialMetricsCollector()
        
        # Thread-safe collections
        self.metrics_buffer = deque(maxlen=10000)
        self.anomaly_buffer = deque(maxlen=1000)
        self.performance_buffer = deque(maxlen=5000)
        self.buffer_lock = threading.Lock()
        
        # Background thread for metrics processing
        self.metrics_thread = None
        self.running = False
        
        self.logger.info("🔧 Spatial logging system initialized")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Metrics file: {self.metrics_file}")
        self.logger.info(f"Prometheus enabled: {self.enable_prometheus}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger('spatial_mllm')
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_background_processing(self):
        """Start background thread for metrics processing"""
        if self.metrics_thread is None or not self.metrics_thread.is_alive():
            self.running = True
            self.metrics_thread = threading.Thread(target=self._process_metrics_background)
            self.metrics_thread.daemon = True
            self.metrics_thread.start()
            self.logger.info("🚀 Background metrics processing started")
    
    def stop_background_processing(self):
        """Stop background metrics processing"""
        self.running = False
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5.0)
            self.logger.info("⏹️  Background metrics processing stopped")
    
    def _process_metrics_background(self):
        """Background processing of metrics"""
        while self.running:
            try:
                # Process buffered metrics
                self._flush_metrics_to_file()
                time.sleep(5)  # Process every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in background metrics processing: {e}")
    
    @contextmanager
    def log_spatial_extraction(self, image_id: str, image_type: str = "pizza"):
        """Context manager for logging spatial feature extraction"""
        start_time = time.time()
        metrics = SpatialFeatureMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_id=image_id,
            extraction_time=0.0,
            feature_dimensions=(0,),
            feature_quality=0.0,
            spatial_complexity=0.0,
            visual_encoder_time=0.0,
            spatial_encoder_time=0.0,
            connector_time=0.0,
            memory_usage=0.0,
            gpu_utilization=0.0
        )
        
        try:
            self.logger.info(f"🔍 Starting spatial feature extraction for {image_id}")
            yield metrics
            
        except Exception as e:
            metrics.error_count += 1
            metrics.warnings.append(f"Extraction error: {str(e)}")
            self.logger.error(f"❌ Spatial extraction failed for {image_id}: {e}")
            raise
        
        finally:
            # Update timing
            metrics.extraction_time = time.time() - start_time
            
            # Log completion
            self.logger.info(
                f"✅ Spatial extraction completed for {image_id} "
                f"in {metrics.extraction_time:.3f}s"
            )
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.metrics_collector.feature_extraction_time.labels(
                    encoder_type='dual',
                    image_type=image_type
                ).observe(metrics.extraction_time)
                
                self.metrics_collector.feature_quality.labels(
                    image_id=image_id,
                    feature_type='spatial'
                ).set(metrics.feature_quality)
            
            # Buffer metrics
            with self.buffer_lock:
                self.metrics_buffer.append(metrics)
    
    def log_dual_encoder_performance(self, performance: DualEncoderPerformance):
        """Log dual-encoder performance metrics"""
        self.logger.info(
            f"📊 Dual-encoder performance - "
            f"Visual: {performance.visual_encoder_latency:.3f}s, "
            f"Spatial: {performance.spatial_encoder_latency:.3f}s, "
            f"Total: {performance.total_inference_time:.3f}s, "
            f"Accuracy: {performance.model_accuracy:.3f}"
        )
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.metrics_collector.dual_encoder_latency.labels(
                encoder='visual',
                batch_size=str(performance.batch_size)
            ).observe(performance.visual_encoder_latency)
            
            self.metrics_collector.dual_encoder_latency.labels(
                encoder='spatial',
                batch_size=str(performance.batch_size)
            ).observe(performance.spatial_encoder_latency)
            
            self.metrics_collector.model_accuracy.labels(
                model_version='spatial_v1',
                dataset='pizza'
            ).set(performance.model_accuracy)
        
        # Buffer performance data
        with self.buffer_lock:
            self.performance_buffer.append(performance)
    
    def log_spatial_anomaly(self, anomaly: SpatialAnomaly):
        """Log spatial anomaly detection"""
        severity_emoji = {
            'low': '🟡',
            'medium': '🟠', 
            'high': '🔴',
            'critical': '🚨'
        }
        
        emoji = severity_emoji.get(anomaly.severity, '⚠️')
        
        self.logger.warning(
            f"{emoji} Spatial anomaly detected - "
            f"Type: {anomaly.anomaly_type}, "
            f"Severity: {anomaly.severity}, "
            f"Confidence: {anomaly.confidence:.3f}, "
            f"Image: {anomaly.image_id}"
        )
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.metrics_collector.anomaly_count.labels(
                severity=anomaly.severity,
                type=anomaly.anomaly_type
            ).inc()
        
        # Buffer anomaly data
        with self.buffer_lock:
            self.anomaly_buffer.append(anomaly)
    
    def log_resource_usage(self, gpu_utilization: float, memory_usage: Dict[str, float]):
        """Log resource utilization"""
        self.logger.debug(
            f"💻 Resource usage - GPU: {gpu_utilization:.1f}%, "
            f"Memory: {memory_usage.get('total', 0):.1f}MB"
        )
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.metrics_collector.gpu_utilization.labels(device_id='0').set(gpu_utilization)
            
            for mem_type, usage in memory_usage.items():
                self.metrics_collector.memory_usage.labels(
                    memory_type=mem_type
                ).set(usage * 1024 * 1024)  # Convert MB to bytes
    
    def _flush_metrics_to_file(self):
        """Flush buffered metrics to file"""
        if not any([self.metrics_buffer, self.performance_buffer, self.anomaly_buffer]):
            return
        
        try:
            # Load existing metrics
            metrics_data = {}
            if Path(self.metrics_file).exists():
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            
            # Initialize sections
            if 'spatial_features' not in metrics_data:
                metrics_data['spatial_features'] = []
            if 'dual_encoder_performance' not in metrics_data:
                metrics_data['dual_encoder_performance'] = []
            if 'anomalies' not in metrics_data:
                metrics_data['anomalies'] = []
            
            # Add buffered data
            with self.buffer_lock:
                # Spatial features
                while self.metrics_buffer:
                    metrics = self.metrics_buffer.popleft()
                    metrics_data['spatial_features'].append(asdict(metrics))
                
                # Performance data
                while self.performance_buffer:
                    perf = self.performance_buffer.popleft()
                    metrics_data['dual_encoder_performance'].append(asdict(perf))
                
                # Anomalies
                while self.anomaly_buffer:
                    anomaly = self.anomaly_buffer.popleft()
                    metrics_data['anomalies'].append(asdict(anomaly))
            
            # Keep only recent data (last 24 hours worth)
            cutoff_time = datetime.now(timezone.utc).timestamp() - 86400
            
            for section in ['spatial_features', 'dual_encoder_performance', 'anomalies']:
                metrics_data[section] = [
                    item for item in metrics_data[section]
                    if datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
                ]
            
            # Write back to file
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error flushing metrics to file: {e}")
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of metrics for the last N hours"""
        try:
            if not Path(self.metrics_file).exists():
                return {}
            
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
            summary = {}
            
            # Spatial features summary
            features = [
                item for item in data.get('spatial_features', [])
                if datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
            ]
            
            if features:
                extraction_times = [f['extraction_time'] for f in features]
                quality_scores = [f['feature_quality'] for f in features if f['feature_quality'] > 0]
                
                summary['spatial_features'] = {
                    'count': len(features),
                    'avg_extraction_time': np.mean(extraction_times),
                    'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                    'error_rate': sum(f['error_count'] for f in features) / len(features)
                }
            
            # Performance summary
            performance = [
                item for item in data.get('dual_encoder_performance', [])
                if datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
            ]
            
            if performance:
                summary['dual_encoder'] = {
                    'count': len(performance),
                    'avg_total_time': np.mean([p['total_inference_time'] for p in performance]),
                    'avg_accuracy': np.mean([p['model_accuracy'] for p in performance]),
                    'avg_throughput': np.mean([p['throughput'] for p in performance])
                }
            
            # Anomalies summary
            anomalies = [
                item for item in data.get('anomalies', [])
                if datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
            ]
            
            if anomalies:
                severity_counts = {}
                for anomaly in anomalies:
                    severity = anomaly['severity']
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                summary['anomalies'] = {
                    'total_count': len(anomalies),
                    'by_severity': severity_counts
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating metrics summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_background_processing()
        self._flush_metrics_to_file()
        self.logger.info("🧹 Spatial logging system cleaned up")


# Singleton instance
_spatial_logger_instance = None

def get_spatial_logger(**kwargs) -> SpatialLogger:
    """Get or create spatial logger singleton"""
    global _spatial_logger_instance
    if _spatial_logger_instance is None:
        _spatial_logger_instance = SpatialLogger(**kwargs)
        _spatial_logger_instance.start_background_processing()
    return _spatial_logger_instance


class SpatialFeatureLogger:
    """
    Simplified interface for spatial feature extraction logging.
    This class provides the interface expected by the monitoring tests
    while wrapping the more comprehensive SpatialLogger functionality.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 buffer_size: int = 1000,
                 debug: bool = False):
        """
        Initialize the SpatialFeatureLogger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
            buffer_size: Size of internal buffer for metrics
            debug: Enable debug mode
        """
        self.buffer_size = buffer_size
        self.debug = debug
        
        # Initialize the underlying spatial logger
        if debug:
            actual_log_level = "DEBUG"
        else:
            actual_log_level = log_level
            
        self.spatial_logger = get_spatial_logger(
            log_level=actual_log_level,
            log_file=log_file
        )
        
        # Internal counters for testing
        self.extraction_count = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        
        if self.debug:
            self.spatial_logger.logger.debug("🔧 SpatialFeatureLogger initialized in debug mode")
    
    @contextmanager
    def log_extraction(self, pizza_type: str, feature_count: int):
        """
        Context manager for logging spatial feature extraction.
        
        Args:
            pizza_type: Type of pizza being processed (e.g., "margherita", "pepperoni")
            feature_count: Number of features expected/extracted
            
        Yields:
            extraction_info: Dictionary with extraction information
        """
        # Generate unique image ID for this extraction
        image_id = f"{pizza_type}_{int(time.time() * 1000)}_{self.extraction_count}"
        self.extraction_count += 1
        
        # Use the underlying spatial logger's context manager
        try:
            with self.spatial_logger.log_spatial_extraction(image_id, pizza_type) as metrics:
                # Create extraction info object for compatibility
                extraction_info = {
                    'image_id': image_id,
                    'pizza_type': pizza_type,
                    'feature_count': feature_count,
                    'start_time': time.time(),
                    'metrics': metrics
                }
                
                yield extraction_info
                
                # Update metrics with feature count information
                metrics.feature_dimensions = (feature_count,)
                metrics.feature_quality = min(1.0, feature_count / 100.0)  # Simple quality heuristic
                metrics.spatial_complexity = min(1.0, feature_count / 200.0)  # Complexity heuristic
                
                # Simulate realistic encoder timings based on feature count
                base_time = metrics.extraction_time
                metrics.visual_encoder_time = base_time * 0.4
                metrics.spatial_encoder_time = base_time * 0.35
                metrics.connector_time = base_time * 0.25
                
                # Simulate memory usage based on feature count
                metrics.memory_usage = feature_count * 0.1  # MB per feature
                metrics.gpu_utilization = min(95.0, 30.0 + (feature_count / 10.0))
                
                self.successful_extractions += 1
                
                if self.debug:
                    self.spatial_logger.logger.debug(
                        f"🎯 Extraction completed: {pizza_type}, "
                        f"Features: {feature_count}, "
                        f"Time: {metrics.extraction_time:.3f}s"
                    )
                
        except Exception as e:
            self.failed_extractions += 1
            if self.debug:
                self.spatial_logger.logger.error(f"❌ Extraction failed for {pizza_type}: {e}")
            raise
    
    def log_feature_extraction(self, feature_data: Dict[str, Any], pizza_type: str = "unknown"):
        """
        Log feature extraction data (non-context manager version)
        
        Args:
            feature_data: Dictionary containing feature extraction information
            pizza_type: Type of pizza being processed
        """
        image_id = f"{pizza_type}_{int(time.time() * 1000)}"
        
        # Create metrics object
        metrics = SpatialFeatureMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_id=image_id,
            extraction_time=feature_data.get('processing_time', 0.0),
            feature_dimensions=tuple(feature_data.get('dimensions', [0])),
            feature_quality=feature_data.get('quality', 0.0),
            spatial_complexity=feature_data.get('complexity', 0.0),
            visual_encoder_time=feature_data.get('visual_time', 0.0),
            spatial_encoder_time=feature_data.get('spatial_time', 0.0),
            connector_time=feature_data.get('connector_time', 0.0),
            memory_usage=feature_data.get('memory_mb', 0.0),
            gpu_utilization=feature_data.get('gpu_percent', 0.0),
            error_count=feature_data.get('errors', 0),
            warnings=feature_data.get('warnings', [])
        )
        
        # Add to buffer
        with self.spatial_logger.buffer_lock:
            self.spatial_logger.metrics_buffer.append(metrics)
        
        if self.debug:
            self.spatial_logger.logger.debug(f"📊 Feature extraction logged for {pizza_type}")
    
    def log_spatial_analysis(self, analysis_data: Dict[str, Any], pizza_type: str = "unknown"):
        """
        Log spatial analysis results
        
        Args:
            analysis_data: Dictionary containing spatial analysis information
            pizza_type: Type of pizza being analyzed
        """
        # Create performance object
        performance = DualEncoderPerformance(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=f"analysis_{int(time.time())}",
            visual_encoder_latency=analysis_data.get('visual_latency', 0.0),
            spatial_encoder_latency=analysis_data.get('spatial_latency', 0.0),
            total_inference_time=analysis_data.get('total_time', 0.0),
            visual_feature_size=analysis_data.get('visual_features', 0),
            spatial_feature_size=analysis_data.get('spatial_features', 0),
            connector_overhead=analysis_data.get('connector_overhead', 0.0),
            batch_size=analysis_data.get('batch_size', 1),
            model_accuracy=analysis_data.get('accuracy', 0.0),
            confidence_score=analysis_data.get('confidence', 0.0),
            throughput=analysis_data.get('throughput', 0.0),
            resource_efficiency=analysis_data.get('efficiency', 0.0)
        )
        
        self.spatial_logger.log_dual_encoder_performance(performance)
        
        if self.debug:
            self.spatial_logger.logger.debug(f"🔬 Spatial analysis logged for {pizza_type}")
    
    def log_feature_quality(self, quality_score: float):
        """
        Log feature quality score for spatial features
        
        Args:
            quality_score: Quality score as a float (typically 0.0 to 1.0)
        """
        # Validate quality score
        if not isinstance(quality_score, (int, float)):
            raise ValueError(f"Quality score must be a number, got {type(quality_score)}")
        
        # Clamp quality score to valid range
        quality_score = max(0.0, min(1.0, float(quality_score)))
        
        # Log the quality score using the underlying spatial logger
        try:
            self.spatial_logger.logger.info(
                f"📊 Feature quality logged: {quality_score:.3f}",
                extra={
                    'quality_score': quality_score,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'component': 'spatial_feature_quality'
                }
            )
            
            # Update Prometheus metrics if available
            if hasattr(self.spatial_logger, 'metrics') and hasattr(self.spatial_logger.metrics, 'feature_quality'):
                self.spatial_logger.metrics.feature_quality.set(quality_score)
            
            if self.debug:
                self.spatial_logger.logger.debug(f"✅ Feature quality {quality_score:.3f} logged successfully")
                
        except Exception as e:
            if self.debug:
                self.spatial_logger.logger.error(f"❌ Failed to log feature quality: {e}")
            raise
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics
        
        Returns:
            Dictionary containing extraction statistics
        """
        total_extractions = self.successful_extractions + self.failed_extractions
        success_rate = (self.successful_extractions / total_extractions) if total_extractions > 0 else 0.0
        
        return {
            'total_extractions': total_extractions,
            'successful_extractions': self.successful_extractions,
            'failed_extractions': self.failed_extractions,
            'success_rate': success_rate,
            'buffer_utilization': len(self.spatial_logger.metrics_buffer) / self.buffer_size
        }
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.extraction_count = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        
        if self.debug:
            self.spatial_logger.logger.debug("🔄 Extraction statistics reset")
    
    def cleanup(self):
        """Cleanup resources"""
        self.spatial_logger.cleanup()
        
        if self.debug:
            self.spatial_logger.logger.debug("🧹 SpatialFeatureLogger cleanup completed")


# Convenience function for backwards compatibility
def create_spatial_feature_logger(**kwargs) -> SpatialFeatureLogger:
    """Create a new SpatialFeatureLogger instance"""
    return SpatialFeatureLogger(**kwargs)


if __name__ == "__main__":
    # Demo usage
    logger = get_spatial_logger(log_level="DEBUG")
    
    # Test spatial feature extraction logging
    with logger.log_spatial_extraction("test_pizza_001", "margherita") as metrics:
        time.sleep(0.1)  # Simulate processing
        metrics.feature_dimensions = (512, 256, 128)
        metrics.feature_quality = 0.87
        metrics.spatial_complexity = 0.65
        metrics.visual_encoder_time = 0.05
        metrics.spatial_encoder_time = 0.03
        metrics.connector_time = 0.02
        metrics.memory_usage = 512.5
        metrics.gpu_utilization = 75.2
    
    # Test performance logging
    performance = DualEncoderPerformance(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id="session_001",
        visual_encoder_latency=0.045,
        spatial_encoder_latency=0.032,
        total_inference_time=0.089,
        visual_feature_size=512,
        spatial_feature_size=256,
        connector_overhead=0.012,
        batch_size=1,
        model_accuracy=0.94,
        confidence_score=0.87,
        throughput=11.2,
        resource_efficiency=0.82
    )
    logger.log_dual_encoder_performance(performance)
    
    # Test anomaly logging
    anomaly = SpatialAnomaly(
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_id="test_pizza_001",
        anomaly_type="unusual_spatial_pattern",
        severity="medium",
        confidence=0.73,
        spatial_location=(120, 80, 150, 120),
        feature_deviation=2.1,
        expected_range=(0.4, 0.8),
        actual_value=1.2,
        recommendation="Review spatial preprocessing parameters"
    )
    logger.log_spatial_anomaly(anomaly)
    
    # Test resource logging
    logger.log_resource_usage(
        gpu_utilization=78.5,
        memory_usage={"total": 1024.5, "gpu": 512.3, "system": 512.2}
    )
    
    print("✅ Spatial logging demo completed")
    print(f"📊 Metrics summary: {logger.get_metrics_summary()}")
    
    # Cleanup
    time.sleep(1)
    logger.cleanup()

    # Demo usage of SpatialFeatureLogger
    feature_logger = SpatialFeatureLogger(log_level="DEBUG", debug=True)
    
    # Test spatial feature extraction logging
    with feature_logger.log_extraction("margherita", 42) as extraction_info:
        time.sleep(0.2)  # Simulate processing
        metrics = extraction_info['metrics']
        metrics.visual_encoder_time = 0.04
        metrics.spatial_encoder_time = 0.035
        metrics.connector_time = 0.025
        metrics.memory_usage = 4.2
        metrics.gpu_utilization = 80.1
    
    # Test feature extraction logging (non-context manager)
    feature_logger.log_feature_extraction({
        'processing_time': 0.075,
        'dimensions': [512, 256, 128],
        'quality': 0.92,
        'complexity': 0.75,
        'visual_time': 0.03,
        'spatial_time': 0.025,
        'connector_time': 0.02,
        'memory_mb': 6.5,
        'gpu_percent': 85.0,
        'errors': 0,
        'warnings': []
    }, pizza_type="pepperoni")
    
    # Test spatial analysis logging
    feature_logger.log_spatial_analysis({
        'visual_latency': 0.038,
        'spatial_latency': 0.028,
        'total_time': 0.078,
        'visual_features': 512,
        'spatial_features': 256,
        'connector_overhead': 0.015,
        'batch_size': 1,
        'accuracy': 0.96,
        'confidence': 0.91,
        'throughput': 12.5,
        'efficiency': 0.85
    }, pizza_type="margherita")
    
    print("✅ SpatialFeatureLogger demo completed")
    print(f"📊 Extraction stats: {feature_logger.get_extraction_stats()}")
    
    # Cleanup
    feature_logger.cleanup()
