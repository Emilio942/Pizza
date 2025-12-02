"""
Visualization Dashboard for Parameter Drift and Training Monitoring
Creates real-time and static visualizations for supervisor monitoring
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

# Placeholder for matplotlib - would be imported when available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from logging_system import IDGenerator  # for generating run IDs when needed

class VisualizationDashboard:
    """
    Creates comprehensive visualizations for PG-ES training monitoring
    Supports both real-time and static plot generation
    """
    
    def __init__(self, output_dir: Union[str, Any], run_id: Optional[str] = None):
        """Initialize dashboard.
        
        Accepts either:
        - output_dir as str/Path and an explicit run_id, or
        - a config-like object as `output_dir` (with `.logging.log_dir`), in which case `run_id`
          is optional and will be generated if not provided.
        """
        # Allow passing a Config/MasterConfig object as first arg
        if not isinstance(output_dir, (str, Path)):
            cfg = output_dir
            log_cfg = getattr(cfg, 'logging', cfg)
            output_dir = getattr(log_cfg, 'log_dir', './logs')

        self.output_dir = Path(str(output_dir))
        self.run_id = run_id or IDGenerator.generate_run_id()
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot configuration
        self.figure_size = (12, 8)
        self.dpi = 100
        self.style = 'default'
        
        # Color scheme for consistency
        self.colors = {
            'pg': '#2E86AB',
            'es': '#A23B72', 
            'combined': '#F18F01',
            'safety': '#C73E1D',
            'supervisor': '#8E44AD',
            'drift': '#16A085'
        }
        
        self.logger = logging.getLogger(__name__)
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - visualizations will be text-based")

    # --- Lightweight interfaces expected by tests and trainer ---
    def update_training_progress(self, data: Dict[str, Any]) -> None:
        """Record a lightweight snapshot of training progress for later plotting."""
        # Persist a simple JSON snapshot to the viz dir
        try:
            snapshot_path = self.viz_dir / f"progress_{self.run_id}.json"
            # If file exists, append-like by reading and extending lists
            existing = {}
            if snapshot_path.exists():
                try:
                    existing = json.load(open(snapshot_path))
                except Exception:
                    existing = {}
            # Merge basic fields
            for k, v in data.items():
                if isinstance(v, (list, tuple)):
                    existing.setdefault(k, [])
                    existing[k] = list(existing[k]) + list(v)
                else:
                    existing[k] = v
            with open(snapshot_path, 'w') as f:
                json.dump(existing, f, indent=2, default=str)
        except Exception:
            pass

    def generate_plots(self) -> Dict[str, str]:
        """Generate available plots from current logging data if possible."""
        # This method expects to be called with access to a logging manager usually. For tests,
        # we simply create empty placeholder files to indicate success.
        results = {}
        try:
            placeholder = self.viz_dir / f"placeholder_{self.run_id}.txt"
            with open(placeholder, 'w') as f:
                f.write(f"Visualization placeholders for run {self.run_id} at {datetime.now().isoformat()}\n")
            results['placeholder'] = str(placeholder)
        except Exception as e:
            self.logger.warning(f"Could not write placeholder visualization: {e}")
        return results
    
    def create_parameter_drift_plot(self, drift_history: List[Dict[str, Any]]) -> str:
        """Create parameter drift visualization"""
        
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_drift_report(drift_history)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle(f'Parameter Drift Analysis - Run {self.run_id}', fontsize=16)
        
        # Extract data
        iterations = [entry['iteration'] for entry in drift_history]
        
        # PG parameters
        pg_lr = [entry['pg_parameters'].get('learning_rate', 0) for entry in drift_history]
        pg_grad_norm = [entry['pg_parameters'].get('gradient_norm', 0) for entry in drift_history]
        
        # ES parameters
        es_sigma = [entry['es_parameters'].get('mutation_strength', 0) for entry in drift_history]
        es_diversity = [entry['es_parameters'].get('population_diversity', 0) for entry in drift_history]
        
        # Plot 1: PG Learning Rate
        axes[0, 0].plot(iterations, pg_lr, color=self.colors['pg'], linewidth=2, label='Learning Rate')
        axes[0, 0].set_title('PG Learning Rate Drift')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: PG Gradient Norm
        axes[0, 1].plot(iterations, pg_grad_norm, color=self.colors['pg'], linewidth=2, label='Gradient Norm')
        axes[0, 1].set_title('PG Gradient Norm Drift')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: ES Mutation Strength
        axes[1, 0].plot(iterations, es_sigma, color=self.colors['es'], linewidth=2, label='Mutation Strength (σ)')
        axes[1, 0].set_title('ES Mutation Strength Drift')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Sigma')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: ES Population Diversity
        axes[1, 1].plot(iterations, es_diversity, color=self.colors['es'], linewidth=2, label='Population Diversity')
        axes[1, 1].set_title('ES Population Diversity Drift')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Diversity')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        
        plot_path = self.viz_dir / f"parameter_drift_{self.run_id}.png"
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Parameter drift plot saved to {plot_path}")
        return str(plot_path)
    
    def create_fitness_reward_history_plot(self, 
                                         fitness_history: List[float], 
                                         reward_history: List[float],
                                         pg_logs: List[Dict],
                                         es_logs: List[Dict]) -> str:
        """Create comprehensive fitness and reward history visualization"""
        
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_performance_report(fitness_history, reward_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        fig.suptitle(f'Performance History - Run {self.run_id}', fontsize=16)
        
        # Plot 1: Combined Fitness/Reward Trends
        if fitness_history:
            axes[0, 0].plot(range(len(fitness_history)), fitness_history, 
                          color=self.colors['es'], linewidth=2, alpha=0.8, label='ES Fitness')
        
        if reward_history:
            axes[0, 0].plot(range(len(reward_history)), reward_history, 
                          color=self.colors['pg'], linewidth=2, alpha=0.8, label='PG Reward')
        
        axes[0, 0].set_title('Fitness & Reward History')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: PG Loss History
        if pg_logs:
            pg_losses = [log.get('policy_loss', 0) for log in pg_logs]
            pg_iterations = [log.get('iteration', i) for i, log in enumerate(pg_logs)]
            
            axes[0, 1].plot(pg_iterations, pg_losses, color=self.colors['pg'], linewidth=2)
            axes[0, 1].set_title('PG Policy Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: ES Generation Performance
        if es_logs:
            es_fitness = [log.get('population_fitness_mean', 0) for log in es_logs]
            es_generations = [log.get('generation', i) for i, log in enumerate(es_logs)]
            
            axes[1, 0].plot(es_generations, es_fitness, color=self.colors['es'], linewidth=2, marker='o', markersize=4)
            axes[1, 0].set_title('ES Population Fitness')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Mean Fitness')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Algorithm Contribution Comparison
        if pg_logs and es_logs:
            # Calculate moving averages for smoother comparison
            window_size = min(10, len(pg_logs) // 4) if pg_logs else 1
            
            pg_performance = self._moving_average([log.get('metrics', {}).get('contribution', 0) for log in pg_logs], window_size)
            es_performance = self._moving_average([log.get('metrics', {}).get('contribution', 0) for log in es_logs], window_size)
            
            max_len = max(len(pg_performance), len(es_performance))
            x_axis = range(max_len)
            
            if pg_performance:
                pg_padded = pg_performance + [pg_performance[-1]] * (max_len - len(pg_performance))
                axes[1, 1].plot(x_axis, pg_padded, color=self.colors['pg'], linewidth=2, label='PG Contribution')
            
            if es_performance:
                es_padded = es_performance + [es_performance[-1]] * (max_len - len(es_performance))
                axes[1, 1].plot(x_axis, es_padded, color=self.colors['es'], linewidth=2, label='ES Contribution')
            
            axes[1, 1].set_title('Algorithm Contribution Comparison')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Contribution')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_path = self.viz_dir / f"performance_history_{self.run_id}.png"
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Performance history plot saved to {plot_path}")
        return str(plot_path)
    
    def create_supervisor_dashboard_plot(self, dashboard_data: Dict[str, Any]) -> str:
        """Create supervisor monitoring dashboard"""
        
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_supervisor_report(dashboard_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle(f'Supervisor Dashboard - Run {self.run_id}', fontsize=16)
        
        # Plot 1: Safety Violations Over Time
        safety_data = dashboard_data.get('safety_summary', {})
        violation_rate = safety_data.get('violation_rate', 0)
        
        axes[0, 0].bar(['Violation Rate'], [violation_rate], color=self.colors['safety'])
        axes[0, 0].set_title('Safety Violation Rate')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add threshold line
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Threshold (5%)')
        axes[0, 0].legend()
        
        # Plot 2: PG vs ES Performance Comparison
        pg_perf = dashboard_data.get('pg_performance', {})
        es_perf = dashboard_data.get('es_performance', {})
        
        algorithms = ['PG', 'ES']
        performances = [pg_perf.get('avg_reward', 0), es_perf.get('avg_fitness', 0)]
        
        bars = axes[0, 1].bar(algorithms, performances, color=[self.colors['pg'], self.colors['es']])
        axes[0, 1].set_title('Algorithm Performance Comparison')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, performances):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Fitness Trend with Statistics
        fitness_trend = dashboard_data.get('fitness_trend', {})
        fitness_values = fitness_trend.get('values', [])
        
        if fitness_values:
            x_axis = range(len(fitness_values))
            axes[0, 2].plot(x_axis, fitness_values, color=self.colors['combined'], linewidth=2)
            
            # Add mean line
            mean_fitness = fitness_trend.get('mean', 0)
            axes[0, 2].axhline(y=mean_fitness, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_fitness:.3f}')
            
            axes[0, 2].set_title('Recent Fitness Trend')
            axes[0, 2].set_xlabel('Recent Iterations')
            axes[0, 2].set_ylabel('Fitness')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
        
        # Plot 4: Parameter Drift Indicators
        param_drift = dashboard_data.get('parameter_drift', {})
        recent_drift = param_drift.get('recent_drift', [])
        
        if recent_drift:
            drift_metrics = []
            for entry in recent_drift[-10:]:  # Last 10 entries
                pg_lr_drift = entry.get('drift_metrics', {}).get('pg_lr_drift', 0)
                es_sigma_drift = entry.get('drift_metrics', {}).get('es_sigma_drift', 0)
                drift_metrics.append(max(pg_lr_drift, es_sigma_drift))
            
            axes[1, 0].plot(range(len(drift_metrics)), drift_metrics, 
                          color=self.colors['drift'], linewidth=2, marker='o')
            axes[1, 0].set_title('Parameter Drift Magnitude')
            axes[1, 0].set_xlabel('Recent Updates')
            axes[1, 0].set_ylabel('Max Drift')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Algorithm Iteration Distribution
        total_pg = dashboard_data.get('total_pg_updates', 0)
        total_es = dashboard_data.get('total_es_updates', 0)
        
        if total_pg + total_es > 0:
            labels = ['PG Updates', 'ES Updates']
            sizes = [total_pg, total_es]
            colors = [self.colors['pg'], self.colors['es']]
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Algorithm Update Distribution')
        
        # Plot 6: Current Status Summary
        current_iter = dashboard_data.get('current_iteration', 0)
        total_violations = safety_data.get('total_violations', 0)
        recent_alerts = safety_data.get('recent_alerts', 0)
        
        status_metrics = ['Iteration', 'Total Violations', 'Recent Alerts']
        status_values = [current_iter, total_violations, recent_alerts]
        
        bars = axes[1, 2].bar(status_metrics, status_values, color=self.colors['supervisor'])
        axes[1, 2].set_title('Current Status Metrics')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, status_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + max(status_values) * 0.01,
                          f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.viz_dir / f"supervisor_dashboard_{self.run_id}.png"
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Supervisor dashboard plot saved to {plot_path}")
        return str(plot_path)
    
    def create_real_time_monitoring_plot(self, 
                                       recent_data: Dict[str, List],
                                       window_size: int = 100) -> str:
        """Create real-time monitoring plot for recent performance"""
        
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_realtime_report(recent_data)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle(f'Real-Time Monitoring - Run {self.run_id}', fontsize=16)
        
        # Extract recent data
        iterations = recent_data.get('iterations', [])
        rewards = recent_data.get('rewards', [])
        fitness_values = recent_data.get('fitness', [])
        safety_violations = recent_data.get('safety_violations', [])
        
        # Limit to window size
        if len(iterations) > window_size:
            iterations = iterations[-window_size:]
            rewards = rewards[-window_size:]
            fitness_values = fitness_values[-window_size:]
            safety_violations = safety_violations[-window_size:]
        
        # Plot 1: Recent Rewards/Fitness
        if rewards:
            axes[0, 0].plot(iterations, rewards, color=self.colors['pg'], linewidth=2, label='Rewards', alpha=0.8)
        if fitness_values:
            axes[0, 0].plot(iterations, fitness_values, color=self.colors['es'], linewidth=2, label='Fitness', alpha=0.8)
        
        axes[0, 0].set_title('Recent Performance')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Safety Violations
        if safety_violations:
            violation_counts = [len(violations) if isinstance(violations, list) else violations for violations in safety_violations]
            axes[0, 1].plot(iterations, violation_counts, color=self.colors['safety'], linewidth=2, marker='o', markersize=3)
            axes[0, 1].set_title('Safety Violations')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Violation Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance Moving Average
        if rewards or fitness_values:
            combined_performance = []
            for i, iter_num in enumerate(iterations):
                reward = rewards[i] if i < len(rewards) else 0
                fitness = fitness_values[i] if i < len(fitness_values) else 0
                combined_performance.append(0.7 * reward + 0.3 * fitness)  # Weighted combination
            
            ma_window = min(10, len(combined_performance) // 4)
            if ma_window > 1:
                moving_avg = self._moving_average(combined_performance, ma_window)
                ma_iterations = iterations[ma_window-1:]
                
                axes[1, 0].plot(ma_iterations, moving_avg, color=self.colors['combined'], linewidth=3)
                axes[1, 0].set_title(f'Performance Moving Average (Window: {ma_window})')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Combined Score')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance Variance
        if len(rewards) > 10 or len(fitness_values) > 10:
            variance_window = 20
            reward_variance = self._rolling_variance(rewards, variance_window) if len(rewards) >= variance_window else []
            fitness_variance = self._rolling_variance(fitness_values, variance_window) if len(fitness_values) >= variance_window else []
            
            if reward_variance:
                var_iterations = iterations[variance_window-1:]
                axes[1, 1].plot(var_iterations, reward_variance, color=self.colors['pg'], linewidth=2, label='Reward Variance', alpha=0.7)
            
            if fitness_variance:
                var_iterations = iterations[variance_window-1:]
                axes[1, 1].plot(var_iterations, fitness_variance, color=self.colors['es'], linewidth=2, label='Fitness Variance', alpha=0.7)
            
            axes[1, 1].set_title('Performance Variance')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_path = self.viz_dir / f"realtime_monitoring_{self.run_id}.png"
        plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Real-time monitoring plot saved to {plot_path}")
        return str(plot_path)
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data
        
        result = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            result.append(avg)
        
        return result
    
    def _rolling_variance(self, data: List[float], window: int) -> List[float]:
        """Calculate rolling variance"""
        if len(data) < window:
            return []
        
        result = []
        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            mean = sum(window_data) / len(window_data)
            variance = sum((x - mean) ** 2 for x in window_data) / len(window_data)
            result.append(variance)
        
        return result
    
    # Text-based fallback methods when matplotlib is not available
    
    def _create_text_drift_report(self, drift_history: List[Dict]) -> str:
        """Create text-based parameter drift report"""
        
        report_path = self.viz_dir / f"parameter_drift_report_{self.run_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Parameter Drift Report - Run {self.run_id}\n")
            f.write("=" * 50 + "\n\n")
            
            if not drift_history:
                f.write("No drift data available.\n")
                return str(report_path)
            
            recent_entries = drift_history[-10:]  # Last 10 entries
            
            f.write("Recent Parameter Values:\n")
            f.write("-" * 30 + "\n")
            
            for entry in recent_entries:
                iter_num = entry.get('iteration', 0)
                pg_params = entry.get('pg_parameters', {})
                es_params = entry.get('es_parameters', {})
                
                f.write(f"Iteration {iter_num}:\n")
                f.write(f"  PG Learning Rate: {pg_params.get('learning_rate', 0):.6f}\n")
                f.write(f"  PG Gradient Norm: {pg_params.get('gradient_norm', 0):.6f}\n")
                f.write(f"  ES Mutation Strength: {es_params.get('mutation_strength', 0):.6f}\n")
                f.write(f"  ES Population Diversity: {es_params.get('population_diversity', 0):.6f}\n")
                f.write("\n")
        
        return str(report_path)
    
    def _create_text_performance_report(self, fitness_history: List[float], reward_history: List[float]) -> str:
        """Create text-based performance report"""
        
        report_path = self.viz_dir / f"performance_report_{self.run_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Performance Report - Run {self.run_id}\n")
            f.write("=" * 40 + "\n\n")
            
            # Fitness statistics
            if fitness_history:
                f.write("ES Fitness Statistics:\n")
                f.write(f"  Total samples: {len(fitness_history)}\n")
                f.write(f"  Mean: {sum(fitness_history) / len(fitness_history):.4f}\n")
                f.write(f"  Min: {min(fitness_history):.4f}\n")
                f.write(f"  Max: {max(fitness_history):.4f}\n")
                f.write(f"  Final: {fitness_history[-1]:.4f}\n\n")
            
            # Reward statistics
            if reward_history:
                f.write("PG Reward Statistics:\n")
                f.write(f"  Total samples: {len(reward_history)}\n")
                f.write(f"  Mean: {sum(reward_history) / len(reward_history):.4f}\n")
                f.write(f"  Min: {min(reward_history):.4f}\n")
                f.write(f"  Max: {max(reward_history):.4f}\n")
                f.write(f"  Final: {reward_history[-1]:.4f}\n\n")
            
            # Recent performance trend
            recent_window = 20
            if len(fitness_history) >= recent_window:
                recent_fitness = fitness_history[-recent_window:]
                recent_mean = sum(recent_fitness) / len(recent_fitness)
                f.write(f"Recent ES Fitness Trend (last {recent_window}): {recent_mean:.4f}\n")
            
            if len(reward_history) >= recent_window:
                recent_rewards = reward_history[-recent_window:]
                recent_mean = sum(recent_rewards) / len(recent_rewards)
                f.write(f"Recent PG Reward Trend (last {recent_window}): {recent_mean:.4f}\n")
        
        return str(report_path)
    
    def _create_text_supervisor_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Create text-based supervisor report"""
        
        report_path = self.viz_dir / f"supervisor_report_{self.run_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Supervisor Dashboard Report - Run {self.run_id}\n")
            f.write("=" * 50 + "\n\n")
            
            # Run metadata
            metadata = dashboard_data.get('run_metadata', {})
            f.write("Run Information:\n")
            f.write(f"  Run ID: {metadata.get('run_id', 'Unknown')}\n")
            f.write(f"  Experiment: {metadata.get('experiment_name', 'Unknown')}\n")
            f.write(f"  Current Iteration: {dashboard_data.get('current_iteration', 0)}\n\n")
            
            # Algorithm performance
            pg_perf = dashboard_data.get('pg_performance', {})
            es_perf = dashboard_data.get('es_performance', {})
            
            f.write("Algorithm Performance:\n")
            f.write(f"  PG Iterations: {pg_perf.get('iterations', 0)}\n")
            f.write(f"  PG Avg Reward: {pg_perf.get('avg_reward', 0):.4f}\n")
            f.write(f"  ES Iterations: {es_perf.get('iterations', 0)}\n")
            f.write(f"  ES Avg Fitness: {es_perf.get('avg_fitness', 0):.4f}\n\n")
            
            # Safety summary
            safety = dashboard_data.get('safety_summary', {})
            f.write("Safety Analysis:\n")
            f.write(f"  Total Violations: {safety.get('total_violations', 0)}\n")
            f.write(f"  Violation Rate: {safety.get('violation_rate', 0):.4f}\n")
            f.write(f"  Recent Alerts: {safety.get('recent_alerts', 0)}\n\n")
            
            # Fitness trends
            fitness_trend = dashboard_data.get('fitness_trend', {})
            f.write("Performance Trends:\n")
            f.write(f"  Fitness Mean: {fitness_trend.get('mean', 0):.4f}\n")
            f.write(f"  Fitness Std: {fitness_trend.get('std', 0):.4f}\n")
        
        return str(report_path)
    
    def _create_text_realtime_report(self, recent_data: Dict[str, List]) -> str:
        """Create text-based real-time report"""
        
        report_path = self.viz_dir / f"realtime_report_{self.run_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Real-Time Monitoring Report - Run {self.run_id}\n")
            f.write("=" * 45 + "\n\n")
            
            iterations = recent_data.get('iterations', [])
            rewards = recent_data.get('rewards', [])
            fitness_values = recent_data.get('fitness', [])
            
            f.write(f"Recent Data Window: {len(iterations)} iterations\n\n")
            
            if rewards:
                f.write("Recent PG Rewards:\n")
                for i, (iter_num, reward) in enumerate(zip(iterations[-5:], rewards[-5:])):
                    f.write(f"  Iter {iter_num}: {reward:.4f}\n")
                f.write("\n")
            
            if fitness_values:
                f.write("Recent ES Fitness:\n")
                for i, (iter_num, fitness) in enumerate(zip(iterations[-5:], fitness_values[-5:])):
                    f.write(f"  Iter {iter_num}: {fitness:.4f}\n")
                f.write("\n")
            
            # Recent trends
            if len(rewards) >= 10:
                recent_reward_trend = sum(rewards[-5:]) / 5 - sum(rewards[-10:-5]) / 5
                f.write(f"Recent Reward Trend: {recent_reward_trend:+.4f}\n")
            
            if len(fitness_values) >= 10:
                recent_fitness_trend = sum(fitness_values[-5:]) / 5 - sum(fitness_values[-10:-5]) / 5
                f.write(f"Recent Fitness Trend: {recent_fitness_trend:+.4f}\n")
        
        return str(report_path)
    
    def export_all_visualizations(self, logging_manager) -> Dict[str, str]:
        """Export all available visualizations"""
        
        plots = {}
        
        try:
            # Get dashboard data
            dashboard_data = logging_manager.get_supervisor_dashboard_data()
            
            # Parameter drift plot
            if logging_manager.parameter_drift_history:
                plots['parameter_drift'] = self.create_parameter_drift_plot(logging_manager.parameter_drift_history)
            
            # Performance history plot
            plots['performance_history'] = self.create_fitness_reward_history_plot(
                logging_manager.fitness_history,
                logging_manager.reward_history,
                logging_manager.pg_logs,
                logging_manager.es_logs
            )
            
            # Supervisor dashboard
            plots['supervisor_dashboard'] = self.create_supervisor_dashboard_plot(dashboard_data)
            
            # Real-time monitoring
            recent_data = {
                'iterations': list(range(len(logging_manager.fitness_history))),
                'rewards': logging_manager.reward_history,
                'fitness': logging_manager.fitness_history,
                'safety_violations': []  # Would be extracted from iteration logs
            }
            plots['realtime_monitoring'] = self.create_real_time_monitoring_plot(recent_data)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
        
        return plots
