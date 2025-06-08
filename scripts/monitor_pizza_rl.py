#!/usr/bin/env python3
"""
Pizza RL Training Monitor Dashboard (Aufgabe 4.1)
=================================================

Real-time monitoring dashboard for Pizza RL training progress.
Provides visualization of training metrics, system resources, and performance.

Features:
- Real-time training metrics visualization
- Multi-objective reward tracking
- System resource monitoring
- Performance trend analysis
- Interactive plots and statistics
"""

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import logging

logger = logging.getLogger(__name__)


class PizzaRLMonitor:
    """Real-time monitoring dashboard for Pizza RL training."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the training monitor.
        
        Args:
            output_dir: Directory containing training outputs and logs
        """
        self.output_dir = Path(output_dir)
        self.running = True
        self.data_queue = queue.Queue()
        
        # Data storage
        self.training_data = {
            'timestamps': [],
            'rewards': [],
            'accuracy': [],
            'energy_efficiency': [],
            'inference_speed': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_lengths': []
        }
        
        self.system_data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': []
        }
        
        # Setup GUI
        self.setup_gui()
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self.collect_data, daemon=True)
        self.data_thread.start()
        
        logger.info(f"Pizza RL Monitor initialized for: {output_dir}")
    
    def setup_gui(self):
        """Setup the GUI dashboard."""
        self.root = tk.Tk()
        self.root.title("Pizza RL Training Monitor")
        self.root.geometry("1400x900")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Training Metrics Tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Training Metrics")
        
        # System Resources Tab
        self.system_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.system_frame, text="System Resources")
        
        # Statistics Tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Setup training metrics plots
        self.setup_training_plots()
        
        # Setup system resource plots
        self.setup_system_plots()
        
        # Setup statistics display
        self.setup_statistics()
        
        # Control panel
        self.setup_control_panel()
    
    def setup_training_plots(self):
        """Setup training metrics visualization."""
        # Create figure with subplots
        self.training_fig, self.training_axes = plt.subplots(2, 3, figsize=(15, 8))
        self.training_fig.suptitle('Pizza RL Training Metrics', fontsize=16)
        
        # Configure subplots
        self.reward_ax = self.training_axes[0, 0]
        self.accuracy_ax = self.training_axes[0, 1]
        self.energy_ax = self.training_axes[0, 2]
        self.loss_ax = self.training_axes[1, 0]
        self.entropy_ax = self.training_axes[1, 1]
        self.episode_length_ax = self.training_axes[1, 2]
        
        # Setup subplot properties
        self.reward_ax.set_title('Episode Reward')
        self.reward_ax.set_ylabel('Reward')
        self.reward_ax.grid(True, alpha=0.3)
        
        self.accuracy_ax.set_title('Pizza Recognition Accuracy')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.grid(True, alpha=0.3)
        
        self.energy_ax.set_title('Energy Efficiency')
        self.energy_ax.set_ylabel('Efficiency')
        self.energy_ax.grid(True, alpha=0.3)
        
        self.loss_ax.set_title('Training Losses')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True, alpha=0.3)
        
        self.entropy_ax.set_title('Policy Entropy')
        self.entropy_ax.set_ylabel('Entropy')
        self.entropy_ax.grid(True, alpha=0.3)
        
        self.episode_length_ax.set_title('Episode Length')
        self.episode_length_ax.set_ylabel('Steps')
        self.episode_length_ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, self.training_frame)
        self.training_canvas.get_tk_widget().pack(expand=True, fill='both')
    
    def setup_system_plots(self):
        """Setup system resource monitoring plots."""
        self.system_fig, self.system_axes = plt.subplots(2, 2, figsize=(12, 8))
        self.system_fig.suptitle('System Resource Usage', fontsize=16)
        
        # Configure system plots
        self.cpu_ax = self.system_axes[0, 0]
        self.memory_ax = self.system_axes[0, 1]
        self.gpu_usage_ax = self.system_axes[1, 0]
        self.gpu_memory_ax = self.system_axes[1, 1]
        
        self.cpu_ax.set_title('CPU Usage')
        self.cpu_ax.set_ylabel('CPU %')
        self.cpu_ax.grid(True, alpha=0.3)
        
        self.memory_ax.set_title('Memory Usage')
        self.memory_ax.set_ylabel('Memory %')
        self.memory_ax.grid(True, alpha=0.3)
        
        self.gpu_usage_ax.set_title('GPU Usage')
        self.gpu_usage_ax.set_ylabel('GPU %')
        self.gpu_usage_ax.grid(True, alpha=0.3)
        
        self.gpu_memory_ax.set_title('GPU Memory')
        self.gpu_memory_ax.set_ylabel('Memory MB')
        self.gpu_memory_ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.system_canvas = FigureCanvasTkAgg(self.system_fig, self.system_frame)
        self.system_canvas.get_tk_widget().pack(expand=True, fill='both')
    
    def setup_statistics(self):
        """Setup statistics display."""
        # Statistics text widget
        stats_text_frame = ttk.Frame(self.stats_frame)
        stats_text_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_text_frame, font=('Courier', 10))
        stats_scrollbar = ttk.Scrollbar(stats_text_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side='left', expand=True, fill='both')
        stats_scrollbar.pack(side='right', fill='y')
    
    def setup_control_panel(self):
        """Setup control panel."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=self.force_refresh).pack(side='left', padx=5)
        
        # Export data button
        ttk.Button(control_frame, text="Export Data", command=self.export_data).pack(side='left', padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Monitoring...")
        self.status_label.pack(side='right', padx=5)
    
    def collect_data(self):
        """Background thread for collecting training data."""
        while self.running:
            try:
                # Read training logs
                self.read_training_logs()
                
                # Read system monitoring data
                self.read_system_logs()
                
                # Update GUI
                self.root.after(0, self.update_plots)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
                time.sleep(10)
    
    def read_training_logs(self):
        """Read training logs and extract metrics."""
        log_file = self.output_dir / 'logs' / 'training.log'
        results_file = self.output_dir / 'final_results.json'
        
        # Check if results file exists (training completed)
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                training_metrics = results.get('training_info', {}).get('training_metrics', [])
                
                if training_metrics:
                    # Extract data from training metrics
                    for i, metric in enumerate(training_metrics):
                        if isinstance(metric, dict):
                            timestamp = datetime.now() - timedelta(minutes=len(training_metrics) - i)
                            self.training_data['timestamps'].append(timestamp)
                            self.training_data['rewards'].append(metric.get('average_reward', 0))
                            self.training_data['accuracy'].append(metric.get('average_accuracy', 0))
                            self.training_data['energy_efficiency'].append(metric.get('average_energy_efficiency', 0))
                            self.training_data['policy_loss'].append(metric.get('policy_loss', 0))
                            self.training_data['value_loss'].append(metric.get('value_loss', 0))
                            self.training_data['entropy'].append(metric.get('entropy', 0))
                            
            except Exception as e:
                logger.error(f"Error reading results file: {e}")
        
        # Also try to read from ongoing log file
        if log_file.exists():
            try:
                # Simple log parsing - in a real implementation, this would be more sophisticated
                self.parse_log_file(log_file)
            except Exception as e:
                logger.error(f"Error reading log file: {e}")
    
    def parse_log_file(self, log_file):
        """Parse log file for training metrics."""
        # This is a simplified implementation
        # In practice, you'd want more robust log parsing
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for recent training iteration logs
        for line in reversed(lines[-100:]):  # Last 100 lines
            if 'Iteration' in line and 'Reward:' in line:
                try:
                    # Extract metrics from log line
                    # Format: "Iteration X/Y - Reward: Z.ZZZ - Accuracy: Y.YYY ..."
                    parts = line.split(' - ')
                    if len(parts) >= 4:
                        reward = float(parts[1].split(': ')[1])
                        accuracy = float(parts[2].split(': ')[1])
                        energy_eff = float(parts[3].split(': ')[1])
                        
                        # Add to data if not already present
                        if not self.training_data['timestamps'] or \
                           reward != self.training_data['rewards'][-1]:
                            self.training_data['timestamps'].append(datetime.now())
                            self.training_data['rewards'].append(reward)
                            self.training_data['accuracy'].append(accuracy)
                            self.training_data['energy_efficiency'].append(energy_eff)
                            
                except (ValueError, IndexError):
                    continue
    
    def read_system_logs(self):
        """Read system monitoring logs."""
        system_log = self.output_dir / 'logs' / 'system_usage.csv'
        gpu_log = self.output_dir / 'logs' / 'gpu_usage.csv'
        
        # Read system usage
        if system_log.exists():
            try:
                df = pd.read_csv(system_log)
                if not df.empty:
                    latest_row = df.iloc[-1]
                    self.system_data['timestamps'].append(datetime.now())
                    self.system_data['cpu_usage'].append(float(latest_row['cpu_percent']))
                    self.system_data['memory_usage'].append(float(latest_row['memory_percent']))
            except Exception as e:
                logger.error(f"Error reading system log: {e}")
        
        # Read GPU usage
        if gpu_log.exists():
            try:
                df = pd.read_csv(gpu_log)
                if not df.empty:
                    latest_row = df.iloc[-1]
                    if len(self.system_data['gpu_usage']) < len(self.system_data['timestamps']):
                        self.system_data['gpu_usage'].append(float(latest_row['utilization.gpu [%]']))
                        self.system_data['gpu_memory'].append(float(latest_row['memory.used [MiB]']))
            except Exception as e:
                logger.error(f"Error reading GPU log: {e}")
    
    def update_plots(self):
        """Update all plots with latest data."""
        try:
            self.update_training_plots()
            self.update_system_plots()
            self.update_statistics()
            
            # Update status
            self.status_label.config(text=f"Status: Last updated {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
    
    def update_training_plots(self):
        """Update training metric plots."""
        if not self.training_data['timestamps']:
            return
        
        timestamps = self.training_data['timestamps']
        
        # Clear and plot rewards
        self.reward_ax.clear()
        if self.training_data['rewards']:
            self.reward_ax.plot(timestamps, self.training_data['rewards'], 'b-', linewidth=2)
            self.reward_ax.set_title('Episode Reward')
            self.reward_ax.grid(True, alpha=0.3)
        
        # Plot accuracy
        self.accuracy_ax.clear()
        if self.training_data['accuracy']:
            self.accuracy_ax.plot(timestamps, self.training_data['accuracy'], 'g-', linewidth=2)
            self.accuracy_ax.set_title('Pizza Recognition Accuracy')
            self.accuracy_ax.grid(True, alpha=0.3)
            self.accuracy_ax.set_ylim(0, 1)
        
        # Plot energy efficiency
        self.energy_ax.clear()
        if self.training_data['energy_efficiency']:
            self.energy_ax.plot(timestamps, self.training_data['energy_efficiency'], 'orange', linewidth=2)
            self.energy_ax.set_title('Energy Efficiency')
            self.energy_ax.grid(True, alpha=0.3)
            self.energy_ax.set_ylim(0, 1)
        
        # Plot losses
        self.loss_ax.clear()
        if self.training_data['policy_loss']:
            self.loss_ax.plot(timestamps, self.training_data['policy_loss'], 'r-', label='Policy Loss', linewidth=2)
        if self.training_data['value_loss']:
            self.loss_ax.plot(timestamps, self.training_data['value_loss'], 'purple', label='Value Loss', linewidth=2)
        self.loss_ax.set_title('Training Losses')
        self.loss_ax.legend()
        self.loss_ax.grid(True, alpha=0.3)
        
        # Plot entropy
        self.entropy_ax.clear()
        if self.training_data['entropy']:
            self.entropy_ax.plot(timestamps, self.training_data['entropy'], 'teal', linewidth=2)
            self.entropy_ax.set_title('Policy Entropy')
            self.entropy_ax.grid(True, alpha=0.3)
        
        self.training_canvas.draw()
    
    def update_system_plots(self):
        """Update system resource plots."""
        if not self.system_data['timestamps']:
            return
        
        timestamps = self.system_data['timestamps']
        
        # CPU usage
        self.cpu_ax.clear()
        if self.system_data['cpu_usage']:
            self.cpu_ax.plot(timestamps, self.system_data['cpu_usage'], 'b-', linewidth=2)
            self.cpu_ax.set_title('CPU Usage')
            self.cpu_ax.set_ylim(0, 100)
            self.cpu_ax.grid(True, alpha=0.3)
        
        # Memory usage
        self.memory_ax.clear()
        if self.system_data['memory_usage']:
            self.memory_ax.plot(timestamps, self.system_data['memory_usage'], 'g-', linewidth=2)
            self.memory_ax.set_title('Memory Usage')
            self.memory_ax.set_ylim(0, 100)
            self.memory_ax.grid(True, alpha=0.3)
        
        # GPU usage
        self.gpu_usage_ax.clear()
        if self.system_data['gpu_usage']:
            self.gpu_usage_ax.plot(timestamps, self.system_data['gpu_usage'], 'orange', linewidth=2)
            self.gpu_usage_ax.set_title('GPU Usage')
            self.gpu_usage_ax.set_ylim(0, 100)
            self.gpu_usage_ax.grid(True, alpha=0.3)
        
        # GPU memory
        self.gpu_memory_ax.clear()
        if self.system_data['gpu_memory']:
            self.gpu_memory_ax.plot(timestamps, self.system_data['gpu_memory'], 'red', linewidth=2)
            self.gpu_memory_ax.set_title('GPU Memory')
            self.gpu_memory_ax.grid(True, alpha=0.3)
        
        self.system_canvas.draw()
    
    def update_statistics(self):
        """Update statistics display."""
        self.stats_text.delete(1.0, tk.END)
        
        stats = []
        stats.append("Pizza RL Training Statistics")
        stats.append("=" * 40)
        stats.append(f"Monitoring Directory: {self.output_dir}")
        stats.append(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stats.append("")
        
        # Training statistics
        if self.training_data['rewards']:
            stats.append("Training Metrics:")
            stats.append(f"  Latest Reward: {self.training_data['rewards'][-1]:.3f}")
            stats.append(f"  Average Reward: {np.mean(self.training_data['rewards']):.3f}")
            stats.append(f"  Best Reward: {max(self.training_data['rewards']):.3f}")
            
            if self.training_data['accuracy']:
                stats.append(f"  Latest Accuracy: {self.training_data['accuracy'][-1]:.3f}")
                stats.append(f"  Average Accuracy: {np.mean(self.training_data['accuracy']):.3f}")
            
            if self.training_data['energy_efficiency']:
                stats.append(f"  Latest Energy Eff: {self.training_data['energy_efficiency'][-1]:.3f}")
                stats.append(f"  Average Energy Eff: {np.mean(self.training_data['energy_efficiency']):.3f}")
        
        stats.append("")
        
        # System statistics
        if self.system_data['cpu_usage']:
            stats.append("System Resources:")
            stats.append(f"  CPU Usage: {self.system_data['cpu_usage'][-1]:.1f}%")
            stats.append(f"  Memory Usage: {self.system_data['memory_usage'][-1]:.1f}%")
            
            if self.system_data['gpu_usage']:
                stats.append(f"  GPU Usage: {self.system_data['gpu_usage'][-1]:.1f}%")
            if self.system_data['gpu_memory']:
                stats.append(f"  GPU Memory: {self.system_data['gpu_memory'][-1]:.0f} MB")
        
        self.stats_text.insert(tk.END, "\n".join(stats))
    
    def force_refresh(self):
        """Force refresh of all data."""
        self.collect_data_once()
        self.update_plots()
    
    def collect_data_once(self):
        """Collect data once (for manual refresh)."""
        try:
            self.read_training_logs()
            self.read_system_logs()
        except Exception as e:
            logger.error(f"Error in manual data collection: {e}")
    
    def export_data(self):
        """Export collected data to files."""
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        export_path = Path(export_dir)
        
        # Export training data
        training_df = pd.DataFrame(self.training_data)
        training_df.to_csv(export_path / 'training_metrics.csv', index=False)
        
        # Export system data
        system_df = pd.DataFrame(self.system_data)
        system_df.to_csv(export_path / 'system_metrics.csv', index=False)
        
        self.status_label.config(text=f"Data exported to {export_path}")
    
    def run(self):
        """Start the monitoring dashboard."""
        try:
            self.root.mainloop()
        finally:
            self.running = False


def main():
    """Main entry point for the monitoring dashboard."""
    parser = argparse.ArgumentParser(description="Pizza RL Training Monitor")
    parser.add_argument('output_dir', help='Training output directory to monitor')
    parser.add_argument('--refresh-interval', type=int, default=5, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run monitor
    monitor = PizzaRLMonitor(args.output_dir)
    monitor.run()


if __name__ == "__main__":
    main()
