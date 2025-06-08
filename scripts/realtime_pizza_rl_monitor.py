#!/usr/bin/env python3
"""
Real-time Pizza RL Training Monitor with Live Dashboard
Aufgabe 4.1 - Enhanced monitoring for Pizza RL training progress

This script provides:
- Real-time training progress monitoring
- Live performance metrics visualization
- Multi-objective optimization tracking
- System resource monitoring
- Training completion prediction
- Automatic checkpoint analysis
"""

import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import os
import re
from datetime import datetime, timedelta
import logging
import threading
from collections import deque
import subprocess

class RealTimePizzaRLMonitor:
    """Real-time monitor for Pizza RL training with live updates"""
    
    def __init__(self, training_dir: str, update_interval: int = 10):
        self.training_dir = Path(training_dir)
        self.logs_dir = self.training_dir / "logs"
        self.update_interval = update_interval
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Data storage for live plotting
        self.max_points = 100  # Keep last 100 data points for plotting
        self.training_metrics = {
            'iterations': deque(maxlen=self.max_points),
            'rewards': deque(maxlen=self.max_points),
            'accuracy': deque(maxlen=self.max_points),
            'energy_efficiency': deque(maxlen=self.max_points),
            'time_per_iteration': deque(maxlen=self.max_points),
            'progress_percent': deque(maxlen=self.max_points)
        }
        
        # System metrics
        self.system_metrics = {
            'timestamps': deque(maxlen=self.max_points),
            'cpu_usage': deque(maxlen=self.max_points),
            'memory_usage': deque(maxlen=self.max_points),
            'gpu_usage': deque(maxlen=self.max_points),
            'gpu_memory': deque(maxlen=self.max_points),
            'gpu_power': deque(maxlen=self.max_points)
        }
        
        # Training status
        self.training_active = True
        self.last_log_size = 0
        self.training_start_time = None
        self.estimated_completion = None
        
        # Setup matplotlib for real-time plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Pizza RL Training - Live Monitoring Dashboard', fontsize=16, fontweight='bold')
        
    def parse_latest_training_metrics(self):
        """Parse the latest training metrics from log file"""
        log_file = self.logs_dir / "training.log"
        
        if not log_file.exists():
            return False
            
        try:
            # Check if log file has new content
            current_size = os.path.getsize(log_file)
            if current_size == self.last_log_size:
                return False  # No new data
                
            self.last_log_size = current_size
            
            # Read and parse the latest training iterations
            pattern = r"Iteration (\d+)/(\d+) \(Steps: (\d+)/(\d+)\) - Reward: ([\d.]+) - Accuracy: ([\d.]+) - Energy Eff: ([\d.]+) - Time: ([\d.]+)s"
            
            latest_entries = []
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    match = re.search(pattern, line)
                    if match:
                        iteration, total_iter, steps, total_steps, reward, accuracy, energy_eff, time_taken = match.groups()
                        latest_entries.append({
                            'iteration': int(iteration),
                            'total_iterations': int(total_iter),
                            'steps': int(steps),
                            'total_steps': int(total_steps),
                            'reward': float(reward),
                            'accuracy': float(accuracy),
                            'energy_efficiency': float(energy_eff),
                            'time_per_iteration': float(time_taken),
                            'progress_percent': (int(steps) / int(total_steps)) * 100
                        })
            
            # Update data with latest entries
            for entry in latest_entries[-5:]:  # Take last 5 entries to avoid duplicates
                if not self.training_metrics['iterations'] or entry['iteration'] > max(self.training_metrics['iterations']):
                    self.training_metrics['iterations'].append(entry['iteration'])
                    self.training_metrics['rewards'].append(entry['reward'])
                    self.training_metrics['accuracy'].append(entry['accuracy'])
                    self.training_metrics['energy_efficiency'].append(entry['energy_efficiency'])
                    self.training_metrics['time_per_iteration'].append(entry['time_per_iteration'])
                    self.training_metrics['progress_percent'].append(entry['progress_percent'])
                    
                    # Update training status
                    if self.training_start_time is None and len(self.training_metrics['iterations']) > 1:
                        self.training_start_time = datetime.now()
                    
                    # Estimate completion time
                    if len(self.training_metrics['iterations']) > 2:
                        self.estimate_completion_time(entry)
            
            return len(latest_entries) > 0
            
        except Exception as e:
            self.logger.error(f"Error parsing training metrics: {e}")
            return False
    
    def estimate_completion_time(self, latest_entry):
        """Estimate training completion time"""
        try:
            current_iteration = latest_entry['iteration']
            total_iterations = latest_entry['total_iterations']
            avg_time_per_iteration = np.mean(list(self.training_metrics['time_per_iteration']))
            
            remaining_iterations = total_iterations - current_iteration
            remaining_seconds = avg_time_per_iteration * remaining_iterations
            
            self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
            
        except Exception as e:
            self.logger.warning(f"Could not estimate completion time: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # Get current timestamp
            current_time = datetime.now()
            
            # CPU and Memory from system_usage.csv
            system_file = self.logs_dir / "system_usage.csv"
            if system_file.exists():
                try:
                    # Read last line of system usage
                    with open(system_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            parts = last_line.split(',')
                            if len(parts) >= 3:
                                self.system_metrics['timestamps'].append(current_time)
                                self.system_metrics['cpu_usage'].append(float(parts[1]))
                                self.system_metrics['memory_usage'].append(float(parts[2]))
                except Exception as e:
                    self.logger.warning(f"Error reading system metrics: {e}")
            
            # GPU metrics from gpu_usage.csv
            gpu_file = self.logs_dir / "gpu_usage.csv"
            if gpu_file.exists():
                try:
                    with open(gpu_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            parts = last_line.split(', ')
                            if len(parts) >= 5:
                                # Extract GPU utilization, memory, and power
                                gpu_util_str = parts[1].replace(' %', '')
                                gpu_memory_str = parts[2].replace(' MiB', '')
                                power_str = parts[4].replace(' W', '')
                                
                                self.system_metrics['gpu_usage'].append(float(gpu_util_str))
                                self.system_metrics['gpu_memory'].append(float(gpu_memory_str))
                                self.system_metrics['gpu_power'].append(float(power_str))
                except Exception as e:
                    self.logger.warning(f"Error reading GPU metrics: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Error updating system metrics: {e}")
    
    def check_training_status(self):
        """Check if training process is still active"""
        try:
            result = subprocess.run(['pgrep', '-f', 'train_pizza_rl.py'], 
                                   capture_output=True, text=True)
            self.training_active = bool(result.stdout.strip())
            return self.training_active
        except Exception:
            return False
    
    def update_plots(self):
        """Update all live plots"""
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot 1: Reward progression
            if self.training_metrics['iterations'] and self.training_metrics['rewards']:
                self.axes[0, 0].plot(list(self.training_metrics['iterations']), 
                                   list(self.training_metrics['rewards']), 
                                   'b-', linewidth=2, marker='o', markersize=4)
                self.axes[0, 0].set_title('Multi-Objective Reward Progression')
                self.axes[0, 0].set_xlabel('Iteration')
                self.axes[0, 0].set_ylabel('Reward')
                self.axes[0, 0].grid(True, alpha=0.3)
                
                # Add trend line
                if len(self.training_metrics['rewards']) > 1:
                    x = np.array(list(self.training_metrics['iterations']))
                    y = np.array(list(self.training_metrics['rewards']))
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    self.axes[0, 0].plot(x, p(x), "r--", alpha=0.8, linewidth=1, label='Trend')
                    self.axes[0, 0].legend()
            
            # Plot 2: Individual metrics
            if self.training_metrics['iterations']:
                iterations = list(self.training_metrics['iterations'])
                self.axes[0, 1].plot(iterations, list(self.training_metrics['accuracy']), 
                                   'g-', label='Accuracy', linewidth=2)
                self.axes[0, 1].plot(iterations, list(self.training_metrics['energy_efficiency']), 
                                   'r-', label='Energy Efficiency', linewidth=2)
                self.axes[0, 1].set_title('Performance Metrics')
                self.axes[0, 1].set_xlabel('Iteration')
                self.axes[0, 1].set_ylabel('Metric Value')
                self.axes[0, 1].legend()
                self.axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Training progress percentage
            if self.training_metrics['progress_percent']:
                self.axes[0, 2].plot(list(self.training_metrics['iterations']), 
                                   list(self.training_metrics['progress_percent']), 
                                   'c-', linewidth=3, marker='s', markersize=4)
                self.axes[0, 2].set_title('Training Progress')
                self.axes[0, 2].set_xlabel('Iteration')
                self.axes[0, 2].set_ylabel('Progress (%)')
                self.axes[0, 2].set_ylim(0, 100)
                self.axes[0, 2].grid(True, alpha=0.3)
                
                # Add completion estimate
                if self.estimated_completion:
                    current_progress = list(self.training_metrics['progress_percent'])[-1] if self.training_metrics['progress_percent'] else 0
                    time_remaining = self.estimated_completion - datetime.now()
                    hours, remainder = divmod(time_remaining.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    self.axes[0, 2].text(0.05, 0.95, f'Progress: {current_progress:.1f}%\nETA: {int(hours)}h {int(minutes)}m', 
                                       transform=self.axes[0, 2].transAxes, fontsize=10, 
                                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 4: System CPU/Memory usage
            if self.system_metrics['cpu_usage'] and self.system_metrics['memory_usage']:
                time_indices = list(range(len(self.system_metrics['cpu_usage'])))
                self.axes[1, 0].plot(time_indices, list(self.system_metrics['cpu_usage']), 
                                   'orange', label='CPU %', linewidth=2)
                self.axes[1, 0].plot(time_indices, list(self.system_metrics['memory_usage']), 
                                   'purple', label='Memory %', linewidth=2)
                self.axes[1, 0].set_title('System Resource Usage')
                self.axes[1, 0].set_xlabel('Time (samples)')
                self.axes[1, 0].set_ylabel('Usage (%)')
                self.axes[1, 0].legend()
                self.axes[1, 0].grid(True, alpha=0.3)
                self.axes[1, 0].set_ylim(0, 100)
            
            # Plot 5: GPU metrics
            if self.system_metrics['gpu_usage'] and self.system_metrics['gpu_power']:
                time_indices = list(range(len(self.system_metrics['gpu_usage'])))
                ax1 = self.axes[1, 1]
                ax2 = ax1.twinx()
                
                line1 = ax1.plot(time_indices, list(self.system_metrics['gpu_usage']), 
                               'blue', label='GPU Util %', linewidth=2)
                line2 = ax2.plot(time_indices, list(self.system_metrics['gpu_power']), 
                               'red', label='Power (W)', linewidth=2)
                
                ax1.set_xlabel('Time (samples)')
                ax1.set_ylabel('GPU Utilization (%)', color='blue')
                ax2.set_ylabel('Power (W)', color='red')
                ax1.set_title('GPU Performance')
                ax1.grid(True, alpha=0.3)
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
            
            # Plot 6: Training efficiency (reward per time)
            if self.training_metrics['rewards'] and self.training_metrics['time_per_iteration']:
                efficiency = [r/t for r, t in zip(self.training_metrics['rewards'], 
                                                 self.training_metrics['time_per_iteration'])]
                self.axes[1, 2].plot(list(self.training_metrics['iterations']), efficiency, 
                                   'm-', linewidth=2, marker='d', markersize=4)
                self.axes[1, 2].set_title('Training Efficiency (Reward/Time)')
                self.axes[1, 2].set_xlabel('Iteration')
                self.axes[1, 2].set_ylabel('Efficiency')
                self.axes[1, 2].grid(True, alpha=0.3)
            
            # Update plot
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
        except Exception as e:
            self.logger.error(f"Error updating plots: {e}")
    
    def print_status_summary(self):
        """Print current training status summary"""
        if not self.training_metrics['iterations']:
            print("No training data available yet...")
            return
            
        latest_iteration = list(self.training_metrics['iterations'])[-1]
        latest_reward = list(self.training_metrics['rewards'])[-1]
        latest_accuracy = list(self.training_metrics['accuracy'])[-1]
        latest_energy = list(self.training_metrics['energy_efficiency'])[-1]
        latest_progress = list(self.training_metrics['progress_percent'])[-1]
        
        print(f"\n{'='*60}")
        print(f"Pizza RL Training Status - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Iteration: {latest_iteration}/122 ({latest_progress:.1f}% complete)")
        print(f"Reward: {latest_reward:.3f}")
        print(f"Accuracy: {latest_accuracy:.3f}")
        print(f"Energy Efficiency: {latest_energy:.3f}")
        
        if self.estimated_completion:
            time_remaining = self.estimated_completion - datetime.now()
            hours, remainder = divmod(time_remaining.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"Estimated completion: {int(hours)}h {int(minutes)}m")
        
        # CPU/GPU status
        if self.system_metrics['cpu_usage']:
            latest_cpu = list(self.system_metrics['cpu_usage'])[-1]
            latest_memory = list(self.system_metrics['memory_usage'])[-1]
            print(f"System: CPU {latest_cpu:.1f}%, Memory {latest_memory:.1f}%")
            
        if self.system_metrics['gpu_usage']:
            latest_gpu = list(self.system_metrics['gpu_usage'])[-1]
            latest_power = list(self.system_metrics['gpu_power'])[-1]
            print(f"GPU: Util {latest_gpu:.1f}%, Power {latest_power:.1f}W")
        
        print(f"Training Active: {'✓' if self.training_active else '✗'}")
        print(f"{'='*60}")
    
    def run_monitoring(self, show_plots: bool = True):
        """Run the real-time monitoring loop"""
        self.logger.info("Starting real-time Pizza RL training monitoring...")
        
        try:
            while True:
                # Check if training is still active
                if not self.check_training_status():
                    self.logger.warning("Training process not detected. Continuing to monitor for completion...")
                
                # Update metrics
                metrics_updated = self.parse_latest_training_metrics()
                self.update_system_metrics()
                
                # Update visualizations
                if show_plots:
                    self.update_plots()
                
                # Print status
                if metrics_updated:
                    self.print_status_summary()
                
                # Check completion
                if self.training_metrics['progress_percent'] and list(self.training_metrics['progress_percent'])[-1] >= 99.9:
                    self.logger.info("Training appears to be complete!")
                    break
                
                # Wait for next update
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        finally:
            if show_plots:
                plt.ioff()
                plt.show()  # Keep final plot open


def main():
    parser = argparse.ArgumentParser(description="Real-time Pizza RL training monitor")
    parser.add_argument("training_dir", help="Path to training directory")
    parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds (default: 10)")
    parser.add_argument("--no-plots", action="store_true", help="Disable live plotting")
    
    args = parser.parse_args()
    
    monitor = RealTimePizzaRLMonitor(args.training_dir, args.interval)
    monitor.run_monitoring(show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
