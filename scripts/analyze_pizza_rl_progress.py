#!/usr/bin/env python3
"""
Pizza RL Training Progress Analyzer
Aufgabe 4.1 - Real-time training analysis and progress monitoring

This script provides comprehensive analysis of ongoing Pizza RL training including:
- Training progress visualization
- Multi-objective optimization tracking
- Performance metrics analysis
- System resource utilization
- Training convergence analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os
import re
from datetime import datetime
import logging

class PizzaRLProgressAnalyzer:
    """Comprehensive analyzer for Pizza RL training progress"""
    
    def __init__(self, training_dir: str):
        self.training_dir = Path(training_dir)
        self.logs_dir = self.training_dir / "logs"
        self.checkpoints_dir = self.training_dir / "checkpoints"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize data containers
        self.training_data = []
        self.system_data = None
        self.gpu_data = None
        
    def parse_training_log(self):
        """Parse training log file to extract metrics"""
        log_file = self.logs_dir / "training.log"
        
        if not log_file.exists():
            self.logger.error(f"Training log not found: {log_file}")
            return
            
        pattern = r"Iteration (\d+)/(\d+) \(Steps: (\d+)/(\d+)\) - Reward: ([\d.]+) - Accuracy: ([\d.]+) - Energy Eff: ([\d.]+) - Time: ([\d.]+)s"
        
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    iteration, total_iter, steps, total_steps, reward, accuracy, energy_eff, time_taken = match.groups()
                    
                    self.training_data.append({
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
    
    def load_system_metrics(self):
        """Load system resource usage data"""
        system_file = self.logs_dir / "system_usage.csv"
        gpu_file = self.logs_dir / "gpu_usage.csv"
        
        try:
            if system_file.exists():
                self.system_data = pd.read_csv(system_file, 
                    names=['timestamp', 'cpu_percent', 'memory_percent', 'swap_percent'])
                self.system_data['timestamp'] = pd.to_datetime(self.system_data['timestamp'])
                
            if gpu_file.exists():
                self.gpu_data = pd.read_csv(gpu_file,
                    names=['timestamp', 'gpu_util', 'memory_used', 'memory_total', 'temp', 'power'])
                self.gpu_data['timestamp'] = pd.to_datetime(self.gpu_data['timestamp'])
                
        except Exception as e:
            self.logger.warning(f"Error loading system metrics: {e}")
    
    def analyze_training_progress(self):
        """Analyze training progress and convergence"""
        if not self.training_data:
            self.logger.warning("No training data available for analysis")
            return {}
            
        df = pd.DataFrame(self.training_data)
        
        analysis = {
            'current_progress': {
                'iteration': df['iteration'].iloc[-1] if len(df) > 0 else 0,
                'total_iterations': df['total_iterations'].iloc[-1] if len(df) > 0 else 0,
                'steps_completed': df['steps'].iloc[-1] if len(df) > 0 else 0,
                'total_steps': df['total_steps'].iloc[-1] if len(df) > 0 else 0,
                'progress_percent': df['progress_percent'].iloc[-1] if len(df) > 0 else 0,
                'estimated_time_remaining': self._estimate_remaining_time(df)
            },
            'performance_trends': {
                'reward_trend': self._analyze_trend(df['reward'].values) if len(df) > 1 else 'insufficient_data',
                'accuracy_trend': self._analyze_trend(df['accuracy'].values) if len(df) > 1 else 'insufficient_data',
                'energy_efficiency_trend': self._analyze_trend(df['energy_efficiency'].values) if len(df) > 1 else 'insufficient_data',
                'current_metrics': {
                    'reward': df['reward'].iloc[-1] if len(df) > 0 else 0,
                    'accuracy': df['accuracy'].iloc[-1] if len(df) > 0 else 0,
                    'energy_efficiency': df['energy_efficiency'].iloc[-1] if len(df) > 0 else 0
                }
            },
            'multi_objective_analysis': self._analyze_multi_objective(df) if len(df) > 0 else {}
        }
        
        return analysis
    
    def _estimate_remaining_time(self, df):
        """Estimate remaining training time"""
        if len(df) < 2:
            return "Insufficient data"
            
        avg_time_per_iteration = df['time_per_iteration'].mean()
        remaining_iterations = df['total_iterations'].iloc[-1] - df['iteration'].iloc[-1]
        remaining_seconds = avg_time_per_iteration * remaining_iterations
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def _analyze_trend(self, values):
        """Analyze trend in values (improving, degrading, stable)"""
        if len(values) < 3:
            return "insufficient_data"
            
        # Calculate trend using linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_multi_objective(self, df):
        """Analyze multi-objective optimization balance"""
        if len(df) == 0:
            return {}
            
        # Define reward weights from training config
        weights = {'accuracy': 0.55, 'energy': 0.25, 'speed': 0.20}
        
        latest = df.iloc[-1]
        
        return {
            'weighted_components': {
                'accuracy_contribution': latest['accuracy'] * weights['accuracy'],
                'energy_contribution': latest['energy_efficiency'] * weights['energy'],
                'speed_contribution': (1.0 - (latest['time_per_iteration'] / 100.0)) * weights['speed']  # Normalize time
            },
            'balance_analysis': {
                'accuracy_vs_efficiency_correlation': df['accuracy'].corr(df['energy_efficiency']) if len(df) > 1 else 0,
                'reward_stability': df['reward'].std() if len(df) > 1 else 0
            }
        }
    
    def generate_progress_plots(self, save_path: str = None):
        """Generate comprehensive progress visualization plots"""
        if not self.training_data:
            self.logger.warning("No training data available for plotting")
            return
            
        df = pd.DataFrame(self.training_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pizza RL Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Reward progression
        axes[0, 0].plot(df['iteration'], df['reward'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('Multi-Objective Reward Progression')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual metrics
        axes[0, 1].plot(df['iteration'], df['accuracy'], 'g-', label='Accuracy', linewidth=2)
        axes[0, 1].plot(df['iteration'], df['energy_efficiency'], 'r-', label='Energy Efficiency', linewidth=2)
        axes[0, 1].set_title('Individual Performance Metrics')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Metric Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Training speed
        axes[0, 2].plot(df['iteration'], df['time_per_iteration'], 'm-', linewidth=2, marker='s', markersize=4)
        axes[0, 2].set_title('Training Speed (Time per Iteration)')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Progress percentage
        axes[1, 0].plot(df['iteration'], df['progress_percent'], 'c-', linewidth=3)
        axes[1, 0].set_title('Training Progress')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Progress (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
        
        # Plot 5: Multi-objective balance
        if len(df) > 1:
            axes[1, 1].scatter(df['accuracy'], df['energy_efficiency'], 
                             c=df['iteration'], cmap='viridis', s=50, alpha=0.7)
            axes[1, 1].set_title('Accuracy vs Energy Efficiency')
            axes[1, 1].set_xlabel('Accuracy')
            axes[1, 1].set_ylabel('Energy Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
            cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
            cbar.set_label('Iteration')
        
        # Plot 6: System resources (if available)
        if self.system_data is not None and len(self.system_data) > 0:
            recent_system = self.system_data.tail(100)  # Last 100 measurements
            axes[1, 2].plot(recent_system.index, recent_system['cpu_percent'], 'orange', label='CPU %', linewidth=2)
            axes[1, 2].plot(recent_system.index, recent_system['memory_percent'], 'purple', label='Memory %', linewidth=2)
            axes[1, 2].set_title('System Resource Usage')
            axes[1, 2].set_xlabel('Measurement')
            axes[1, 2].set_ylabel('Usage (%)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'System data\nnot available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_title('System Resource Usage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Progress plots saved to: {save_path}")
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive training progress report"""
        analysis = self.analyze_training_progress()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_directory': str(self.training_dir),
            'analysis': analysis,
            'summary': self._generate_summary(analysis)
        }
        
        return report
    
    def _generate_summary(self, analysis):
        """Generate human-readable summary"""
        if not analysis.get('current_progress'):
            return "No training progress data available"
            
        progress = analysis['current_progress']
        performance = analysis.get('performance_trends', {})
        
        summary = f"""
Pizza RL Training Progress Summary
================================

Training Status:
- Progress: {progress.get('progress_percent', 0):.1f}% complete
- Iteration: {progress.get('iteration', 0)}/{progress.get('total_iterations', 0)}
- Steps: {progress.get('steps_completed', 0):,}/{progress.get('total_steps', 0):,}
- Estimated time remaining: {progress.get('estimated_time_remaining', 'Unknown')}

Current Performance:
- Reward: {performance.get('current_metrics', {}).get('reward', 0):.3f}
- Accuracy: {performance.get('current_metrics', {}).get('accuracy', 0):.3f}
- Energy Efficiency: {performance.get('current_metrics', {}).get('energy_efficiency', 0):.3f}

Performance Trends:
- Reward: {performance.get('reward_trend', 'unknown')}
- Accuracy: {performance.get('accuracy_trend', 'unknown')}
- Energy Efficiency: {performance.get('energy_efficiency_trend', 'unknown')}

Multi-Objective Status:
The training is optimizing for pizza recognition accuracy (55% weight), 
energy efficiency (25% weight), and inference speed (20% weight).
        """.strip()
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Pizza RL training progress")
    parser.add_argument("training_dir", help="Path to training directory")
    parser.add_argument("--plots", help="Save plots to file", default=None)
    parser.add_argument("--report", help="Save report to file", default=None)
    parser.add_argument("--live", action="store_true", help="Live monitoring mode")
    
    args = parser.parse_args()
    
    analyzer = PizzaRLProgressAnalyzer(args.training_dir)
    
    # Load and analyze data
    analyzer.parse_training_log()
    analyzer.load_system_metrics()
    
    # Generate analysis
    report = analyzer.generate_report()
    
    # Print summary
    print(report['summary'])
    print("\n" + "="*50)
    print(f"Analysis timestamp: {report['timestamp']}")
    
    # Generate plots if requested
    if args.plots:
        fig = analyzer.generate_progress_plots(args.plots)
        plt.show()
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to: {args.report}")


if __name__ == "__main__":
    main()
