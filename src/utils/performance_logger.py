"""
Performance Logger for Pizza RL Training System
===============================================

Comprehensive performance logging system for the Pizza RL training process.
Tracks training metrics, system resources, and model performance.
"""

import time
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import psutil
import os

# Try to import GPU monitoring
try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None


@dataclass
class TrainingMetrics:
    """RL training-specific metrics"""
    timestamp: str
    episode: int
    timestep: int
    mean_reward: float
    episode_length: float
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    approx_kl: Optional[float] = None
    fps: Optional[float] = None


class PerformanceLogger:
    """Comprehensive performance logger for Pizza RL training"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_gpu_monitoring: bool = True,
                 enable_memory_tracking: bool = True,
                 enable_energy_tracking: bool = True,
                 buffer_size: int = 1000):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_energy_tracking = enable_energy_tracking
        
        # Setup logging
        self.logger = logging.getLogger('pizza_rl_performance')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handlers
        log_file = self.log_dir / "performance.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.performance_buffer = deque(maxlen=buffer_size)
        self.training_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # System monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # GPU setup
        if self.enable_gpu_monitoring:
            try:
                self.gpu_count = nvml.nvmlDeviceGetCount()
                self.gpu_handles = []
                for i in range(self.gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                self.logger.info(f"GPU monitoring enabled for {self.gpu_count} devices")
            except Exception as e:
                self.logger.warning(f"GPU monitoring setup failed: {e}")
                self.enable_gpu_monitoring = False
        
        # Background thread
        self.monitoring_thread = None
        self.is_monitoring = False
        
        self.logger.info("Performance logger initialized")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_background,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Background monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Background monitoring stopped")
    
    def _monitor_background(self, interval: float):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                with self.buffer_lock:
                    self.performance_buffer.append(metrics)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Process specific
        try:
            process_memory = self.process.memory_info()
            memory_used_mb = process_memory.rss / 1024 / 1024
        except:
            memory_used_mb = 0.0
        
        # GPU metrics
        gpu_util = None
        gpu_memory = None
        if self.enable_gpu_monitoring and self.gpu_handles:
            try:
                # Use first GPU for monitoring
                handle = self.gpu_handles[0]
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_util = util.gpu
                gpu_memory = (memory_info.used / memory_info.total) * 100
            except Exception as e:
                self.logger.debug(f"GPU monitoring error: {e}")
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else None
        disk_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else None
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / 1024 / 1024 if net_io else None
        net_recv_mb = net_io.bytes_recv / 1024 / 1024 if net_io else None
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_memory,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb
        )
    
    def log_training_metrics(self, 
                           episode: int,
                           timestep: int,
                           mean_reward: float,
                           episode_length: float,
                           policy_loss: Optional[float] = None,
                           value_loss: Optional[float] = None,
                           approx_kl: Optional[float] = None,
                           fps: Optional[float] = None):
        """Log training-specific metrics"""
        
        metrics = TrainingMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            episode=episode,
            timestep=timestep,
            mean_reward=mean_reward,
            episode_length=episode_length,
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            fps=fps
        )
        
        with self.buffer_lock:
            self.training_buffer.append(metrics)
        
        # Log to file
        self.logger.info(
            f"Training - Episode: {episode}, Timestep: {timestep}, "
            f"Reward: {mean_reward:.3f}, Length: {episode_length:.1f}, "
            f"FPS: {fps:.1f}" if fps else ""
        )
    
    def log_performance_snapshot(self) -> PerformanceMetrics:
        """Log current performance snapshot"""
        metrics = self._collect_system_metrics()
        
        with self.buffer_lock:
            self.performance_buffer.append(metrics)
        
        # Log summary
        gpu_info = ""
        if metrics.gpu_utilization is not None:
            gpu_info = f", GPU: {metrics.gpu_utilization:.1f}%"
        
        self.logger.info(
            f"System - CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.1f}MB)"
            f"{gpu_info}"
        )
        
        return metrics
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary"""
        uptime = time.time() - self.start_time
        
        with self.buffer_lock:
            perf_metrics = list(self.performance_buffer)
            train_metrics = list(self.training_buffer)
        
        summary = {
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m",
            "total_performance_samples": len(perf_metrics),
            "total_training_samples": len(train_metrics)
        }
        
        if perf_metrics:
            latest = perf_metrics[-1]
            summary.update({
                "current_cpu_percent": latest.cpu_percent,
                "current_memory_percent": latest.memory_percent,
                "current_memory_mb": latest.memory_used_mb,
                "current_gpu_utilization": latest.gpu_utilization,
                "current_gpu_memory_percent": latest.gpu_memory_percent
            })
            
            # Averages
            cpu_avg = sum(m.cpu_percent for m in perf_metrics[-10:]) / min(10, len(perf_metrics))
            memory_avg = sum(m.memory_percent for m in perf_metrics[-10:]) / min(10, len(perf_metrics))
            
            summary.update({
                "avg_cpu_percent_10min": cpu_avg,
                "avg_memory_percent_10min": memory_avg
            })
        
        if train_metrics:
            latest = train_metrics[-1]
            summary.update({
                "latest_episode": latest.episode,
                "latest_timestep": latest.timestep,
                "latest_mean_reward": latest.mean_reward,
                "latest_episode_length": latest.episode_length
            })
        
        return summary
    
    def export_metrics(self, file_prefix: str = "pizza_rl_metrics"):
        """Export all metrics to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self.buffer_lock:
            perf_data = [asdict(m) for m in self.performance_buffer]
            train_data = [asdict(m) for m in self.training_buffer]
        
        # Export performance metrics
        perf_file = self.log_dir / f"{file_prefix}_performance_{timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_data, f, indent=2)
        
        # Export training metrics
        train_file = self.log_dir / f"{file_prefix}_training_{timestamp}.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {perf_file} and {train_file}")
        
        return perf_file, train_file
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        self.export_metrics()
        self.logger.info("Performance logger cleanup completed")


# Convenience functions
def create_performance_logger(**kwargs) -> PerformanceLogger:
    """Create a new PerformanceLogger instance"""
    return PerformanceLogger(**kwargs)


# Global instance for convenience
_global_logger = None

def get_performance_logger(**kwargs) -> PerformanceLogger:
    """Get or create global performance logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = PerformanceLogger(**kwargs)
    return _global_logger


if __name__ == "__main__":
    # Demo usage
    logger = create_performance_logger(log_level="DEBUG")
    logger.start_monitoring(interval=2.0)
    
    # Simulate training metrics
    for i in range(5):
        logger.log_training_metrics(
            episode=i,
            timestep=i*100,
            mean_reward=10.0 + i*2.5,
            episode_length=50.0 + i*5,
            fps=100.0
        )
        time.sleep(3)
    
    # Get summary
    summary = logger.get_system_summary()
    print(f"System Summary: {json.dumps(summary, indent=2)}")
    
    # Cleanup
    logger.cleanup()
