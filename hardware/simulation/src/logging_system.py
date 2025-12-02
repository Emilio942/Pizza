"""
Comprehensive Logging and Versioning System
Provides unique ID generation, separate PG/ES logging, and data persistence
"""

import os
import json
import time
import uuid
import hashlib
from datetime import datetime, timezone
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging
from pathlib import Path

@dataclass
class RunMetadata:
    """Metadata for each training run"""
    run_id: str
    experiment_name: str
    start_time: datetime
    config_hash: str
    git_commit: Optional[str] = None
    python_version: str = ""
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class IterationLog:
    """Log entry for each training iteration"""
    iteration_id: str
    run_id: str
    iteration_number: int
    timestamp: datetime
    wall_time: float
    algorithm: str  # 'pg', 'es', 'hybrid'
    
    # Algorithm-specific data
    pg_data: Optional[Dict[str, Any]] = None
    es_data: Optional[Dict[str, Any]] = None
    supervisor_data: Optional[Dict[str, Any]] = None
    
    # Simulation state
    simulation_state: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    reward: Optional[float] = None
    fitness: Optional[float] = None
    combined_score: Optional[float] = None
    
    # Safety and validation
    safety_violations: List[str] = field(default_factory=list)
    supervisor_alerts: List[Dict[str, Any]] = field(default_factory=list)

class IDGenerator:
    """Generates unique IDs for runs and iterations"""
    
    @staticmethod
    def generate_run_id() -> str:
        """Generate unique run ID with timestamp and random component"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        random_part = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{random_part}"
    
    @staticmethod
    def generate_iteration_id(run_id: str, iteration: int, algorithm: str) -> str:
        """Generate unique iteration ID"""
        return f"{run_id}_iter_{iteration:06d}_{algorithm}"
    
    @staticmethod
    def generate_config_hash(config: Dict[str, Any]) -> str:
        """Generate hash of configuration for reproducibility"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

class LoggingManager:
    """Central logging and versioning manager
    
    Accepts either:
    - a config-like object as first argument (with `.logging.log_dir` and `.logging.experiment_name`), or
    - a string/Path for the base_log_dir and optional experiment_name.
    """
    
    def __init__(self, base_log_dir: Union[str, Path, Any] = "./logs", experiment_name: Optional[str] = None):
        # Allow passing a Config/MasterConfig object as the first argument
        if not isinstance(base_log_dir, (str, os.PathLike)):
            cfg = base_log_dir
            # Try to extract from a nested logging config
            log_dir = None
            exp_name = None
            try:
                logging_cfg = getattr(cfg, 'logging', None) or cfg
                log_dir = getattr(logging_cfg, 'log_dir', None)
                exp_name = getattr(logging_cfg, 'experiment_name', None)
            except Exception:
                pass
            base_log_dir = log_dir or "./logs"
            if experiment_name is None:
                experiment_name = exp_name or "pg_es_experiment"
        
        if experiment_name is None:
            experiment_name = "pg_es_experiment"
        
        self.base_log_dir = Path(str(base_log_dir))
        self.experiment_name = experiment_name
        
        # Create directory structure
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.base_log_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        # Current run tracking
        self.current_run_id: Optional[str] = None
        self.current_run_dir: Optional[Path] = None
        self.run_metadata: Optional[RunMetadata] = None
        
        # Logging buffers
        self.iteration_logs: List[IterationLog] = []
        self.buffer_size = 100  # Flush to disk every N iterations
        
        # Separate algorithm logs
        self.pg_logs: List[Dict[str, Any]] = []
        self.es_logs: List[Dict[str, Any]] = []
        self.supervisor_logs: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.fitness_history: List[float] = []
        self.reward_history: List[float] = []
        self.parameter_drift_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging manager initialized - Base dir: {self.base_log_dir}")

    # --- Backward-compatibility helpers expected by tests ---
    def generate_iteration_id(self) -> str:
        """Generate a simple iteration id even without an active run (for tests)."""
        run_id = self.current_run_id or IDGenerator.generate_run_id()
        # Use wall-time seconds as a lightweight counter surrogate
        iter_num = int(time.time() * 1000) % 1_000_000
        return IDGenerator.generate_iteration_id(run_id, iter_num, 'test')

    def _ensure_run_dir(self):
        """Ensure there's a run directory if test wrappers need to write files."""
        if not self.current_run_id:
            # Start a lightweight run to ensure directories exist
            try:
                self.start_run(config={"source": "test-wrapper"}, description="auto-started for test logging")
            except Exception:
                # As a fallback, just ensure base dirs
                self.base_log_dir.mkdir(parents=True, exist_ok=True)
                self.runs_dir = self.base_log_dir / "runs"
                self.runs_dir.mkdir(exist_ok=True)

    def _extract_iteration(self, iteration_identifier: Union[int, str, None], default: int = 0) -> int:
        if isinstance(iteration_identifier, int):
            return iteration_identifier
        if isinstance(iteration_identifier, str):
            m = re.search(r"_iter_(\d+)_", iteration_identifier)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return default

    def _write_metric_entry(self, filename: str, payload: Dict[str, Any]) -> None:
        """Append a metric entry to a jsonl file in the current run directory."""

        if not self.current_run_dir:
            return

        metrics_dir = self.current_run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        file_path = metrics_dir / filename
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            # Best-effort logging – never raise during training/tests
            self.logger.debug("Failed writing metric entry", exc_info=True)

    @staticmethod
    def _coerce_scalar(value: Any) -> Optional[float]:
        """Convert value to float if possible (handling numpy scalars)."""

        if value is None:
            return None
        try:
            if hasattr(value, "item") and callable(value.item):
                value = value.item()
            return float(value)
        except (TypeError, ValueError):
            return None

    def log_reward(self, reward: Any, iteration: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a reward value for analysis dashboards and history tracking."""

        self._ensure_run_dir()
        reward_value = self._coerce_scalar(reward)
        if reward_value is None:
            return

        self.reward_history.append(reward_value)
        if len(self.reward_history) > 100_000:
            self.reward_history.pop(0)

        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'iteration': iteration,
            'reward': reward_value,
        }
        if metadata:
            entry['metadata'] = metadata

        self._write_metric_entry("reward_history.jsonl", entry)

    def log_fitness(self, fitness: Any, iteration: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a fitness value for analysis dashboards and history tracking."""

        self._ensure_run_dir()
        fitness_value = self._coerce_scalar(fitness)
        if fitness_value is None:
            return

        self.fitness_history.append(fitness_value)
        if len(self.fitness_history) > 100_000:
            self.fitness_history.pop(0)

        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'iteration': iteration,
            'fitness': fitness_value,
        }
        if metadata:
            entry['metadata'] = metadata

        self._write_metric_entry("fitness_history.jsonl", entry)

    def log_pg_update(self, iteration_identifier: Union[int, str, None], pg_metrics: Dict[str, Any]):
        """Public wrapper used by tests. Accepts iteration number or id string."""
        self._ensure_run_dir()
        iteration = self._extract_iteration(iteration_identifier, pg_metrics.get('iteration', 0))
        self._log_pg_update(iteration, pg_metrics)

    def log_es_update(self, iteration_identifier: Union[int, str, None], es_metrics: Dict[str, Any]):
        """Public wrapper used by tests. Accepts iteration number or id string."""
        self._ensure_run_dir()
        iteration = self._extract_iteration(iteration_identifier, es_metrics.get('iteration', 0))
        self._log_es_update(iteration, es_metrics)

    def log_supervisor_alert(self, alert: Dict[str, Any]):
        """Public wrapper to log supervisor alerts used by tests."""
        self._ensure_run_dir()
        alert_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **alert,
        }
        self.supervisor_logs.append(alert_entry)
        if self.current_run_dir:
            file_path = self.current_run_dir / "supervisor_logs" / f"alert_{int(time.time()*1000)}.json"
            try:
                with open(file_path, 'w') as f:
                    json.dump(alert_entry, f, indent=2)
            except Exception:
                pass

    def log_initialization(self, info: Dict[str, Any]):
        """Lightweight initialization log used by trainer/tests."""
        self._ensure_run_dir()
        try:
            if self.current_run_dir:
                with open(self.current_run_dir / "init.json", 'w') as f:
                    json.dump({'timestamp': datetime.now(timezone.utc).isoformat(), **info}, f, indent=2)
        except Exception:
            pass

    def log_event(self, event: Dict[str, Any]) -> None:
        """Generic event logger."""
        self._ensure_run_dir()
        try:
            if self.current_run_dir:
                with open(self.current_run_dir / "events.log", 'a') as f:
                    f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

    def log_progress(self, progress: Dict[str, Any]) -> None:
        """Progress logger for periodic training updates."""
        self._ensure_run_dir()
        try:
            if self.current_run_dir:
                with open(self.current_run_dir / "progress.log", 'a') as f:
                    f.write(json.dumps(progress, default=str) + "\n")
        except Exception:
            pass
    
    def start_run(self, config: Dict[str, Any], description: str = "", tags: List[str] = None) -> str:
        """Start a new training run"""
        
        # Generate run ID and metadata
        self.current_run_id = IDGenerator.generate_run_id()
        
        self.run_metadata = RunMetadata(
            run_id=self.current_run_id,
            experiment_name=self.experiment_name,
            start_time=datetime.now(timezone.utc),
            config_hash=IDGenerator.generate_config_hash(config),
            python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            tags=tags or [],
            description=description
        )
        
        # Create run directory
        self.current_run_dir = self.runs_dir / self.current_run_id
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.current_run_dir / "iterations").mkdir(exist_ok=True)
        (self.current_run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.current_run_dir / "visualizations").mkdir(exist_ok=True)
        (self.current_run_dir / "pg_logs").mkdir(exist_ok=True)
        (self.current_run_dir / "es_logs").mkdir(exist_ok=True)
        (self.current_run_dir / "supervisor_logs").mkdir(exist_ok=True)
        
        # Save run metadata
        self._save_run_metadata()
        
        # Save configuration
        with open(self.current_run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Started new run: {self.current_run_id}")
        return self.current_run_id
    
    def log_iteration(self, 
                     iteration: int, 
                     algorithm: str,
                     pg_metrics: Optional[Dict] = None,
                     es_metrics: Optional[Dict] = None,
                     supervisor_data: Optional[Dict] = None,
                     simulation_state: Optional[Dict] = None,
                     reward: Optional[float] = None,
                     fitness: Optional[float] = None) -> str:
        """Log a single training iteration"""
        
        if not self.current_run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        # Generate iteration ID
        iteration_id = IDGenerator.generate_iteration_id(self.current_run_id, iteration, algorithm)
        
        # Create iteration log
        iteration_log = IterationLog(
            iteration_id=iteration_id,
            run_id=self.current_run_id,
            iteration_number=iteration,
            timestamp=datetime.now(timezone.utc),
            wall_time=time.time(),
            algorithm=algorithm,
            pg_data=pg_metrics,
            es_data=es_metrics,
            supervisor_data=supervisor_data,
            simulation_state=simulation_state,
            reward=reward,
            fitness=fitness,
            safety_violations=simulation_state.get('safety_violations', []) if simulation_state else [],
            supervisor_alerts=supervisor_data.get('alerts', []) if supervisor_data else []
        )
        
        # Calculate combined score
        if reward is not None and fitness is not None:
            iteration_log.combined_score = 0.7 * reward + 0.3 * fitness
        elif reward is not None:
            iteration_log.combined_score = reward
        elif fitness is not None:
            iteration_log.combined_score = fitness
        
        # Add to buffers
        self.iteration_logs.append(iteration_log)
        
        # Separate algorithm logging
        if algorithm == 'pg' and pg_metrics:
            self._log_pg_update(iteration, pg_metrics)
        elif algorithm == 'es' and es_metrics:
            self._log_es_update(iteration, es_metrics)
        
        # Update histories
        if reward is not None:
            self.reward_history.append(reward)
        if fitness is not None:
            self.fitness_history.append(fitness)
        
        # Log parameter drift
        self._log_parameter_drift(iteration, pg_metrics, es_metrics)
        
        # Flush buffer if needed
        if len(self.iteration_logs) >= self.buffer_size:
            self._flush_iteration_logs()
        
        return iteration_id
    
    def _log_pg_update(self, iteration: int, pg_metrics: Dict[str, Any]):
        """Log Policy Gradient specific update"""
        
        pg_log = {
            'iteration': iteration,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'pg',
            'metrics': pg_metrics.copy(),
            'learning_rate': pg_metrics.get('learning_rate', 0.0),
            'policy_loss': pg_metrics.get('policy_loss', 0.0),
            'value_loss': pg_metrics.get('value_loss', 0.0),
            'entropy_loss': pg_metrics.get('entropy_loss', 0.0),
            'advantage_mean': pg_metrics.get('advantage_mean', 0.0),
            'advantage_std': pg_metrics.get('advantage_std', 0.0),
            'gradient_norm': pg_metrics.get('gradient_norm', 0.0),
            'episodes_completed': pg_metrics.get('episodes_completed', 0)
        }
        
        self.pg_logs.append(pg_log)
        
        # Save immediately to separate PG log file
        if self.current_run_dir:
            pg_log_file = self.current_run_dir / "pg_logs" / f"pg_iteration_{iteration:06d}.json"
            with open(pg_log_file, 'w') as f:
                json.dump(pg_log, f, indent=2)
    
    def _log_es_update(self, iteration: int, es_metrics: Dict[str, Any]):
        """Log Evolution Strategy specific update"""
        
        es_log = {
            'iteration': iteration,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'es',
            'metrics': es_metrics.copy(),
            'generation': es_metrics.get('generation', 0),
            'population_fitness_mean': es_metrics.get('population_fitness_mean', 0.0),
            'population_fitness_std': es_metrics.get('population_fitness_std', 0.0),
            'population_diversity': es_metrics.get('population_diversity', 0.0),
            'mutation_strength': es_metrics.get('mutation_strength', 0.0),
            'elite_fitness': es_metrics.get('elite_fitness', 0.0),
            'improvement_rate': es_metrics.get('improvement_rate', 0.0),
            'evaluations': es_metrics.get('evaluations', 0)
        }
        
        self.es_logs.append(es_log)
        
        # Save immediately to separate ES log file
        if self.current_run_dir:
            es_log_file = self.current_run_dir / "es_logs" / f"es_generation_{es_metrics.get('generation', 0):06d}.json"
            with open(es_log_file, 'w') as f:
                json.dump(es_log, f, indent=2)
    
    def _log_parameter_drift(self, iteration: int, pg_metrics: Optional[Dict], es_metrics: Optional[Dict]):
        """Track parameter drift over time"""
        
        drift_entry = {
            'iteration': iteration,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'pg_parameters': {},
            'es_parameters': {},
            'drift_metrics': {}
        }
        
        # Extract PG parameters
        if pg_metrics:
            drift_entry['pg_parameters'] = {
                'learning_rate': pg_metrics.get('learning_rate', 0.0),
                'gradient_norm': pg_metrics.get('gradient_norm', 0.0),
                'advantage_std': pg_metrics.get('advantage_std', 0.0)
            }
        
        # Extract ES parameters
        if es_metrics:
            drift_entry['es_parameters'] = {
                'mutation_strength': es_metrics.get('mutation_strength', 0.0),
                'population_diversity': es_metrics.get('population_diversity', 0.0),
                'improvement_rate': es_metrics.get('improvement_rate', 0.0)
            }
        
        # Calculate drift metrics
        if len(self.parameter_drift_history) > 0:
            prev_entry = self.parameter_drift_history[-1]
            
            # PG parameter drift
            if pg_metrics and prev_entry['pg_parameters']:
                pg_lr_drift = abs(drift_entry['pg_parameters']['learning_rate'] - prev_entry['pg_parameters']['learning_rate'])
                drift_entry['drift_metrics']['pg_lr_drift'] = pg_lr_drift
            
            # ES parameter drift
            if es_metrics and prev_entry['es_parameters']:
                es_sigma_drift = abs(drift_entry['es_parameters']['mutation_strength'] - prev_entry['es_parameters']['mutation_strength'])
                drift_entry['drift_metrics']['es_sigma_drift'] = es_sigma_drift
        
        self.parameter_drift_history.append(drift_entry)
        
        # Keep history manageable
        if len(self.parameter_drift_history) > 10000:
            self.parameter_drift_history.pop(0)
    
    def get_supervisor_dashboard_data(self) -> Dict[str, Any]:
        """Get data for supervisor monitoring dashboard"""
        
        if not self.current_run_id:
            return {"error": "No active run"}
        
        # Recent iteration data
        recent_iterations = self.iteration_logs[-50:] if len(self.iteration_logs) >= 50 else self.iteration_logs
        
        # Fitness and reward trends
        recent_fitness = self.fitness_history[-100:] if len(self.fitness_history) >= 100 else self.fitness_history
        recent_rewards = self.reward_history[-100:] if len(self.reward_history) >= 100 else self.reward_history
        
        # PG vs ES contribution analysis
        pg_iterations = [log for log in recent_iterations if log.algorithm == 'pg']
        es_iterations = [log for log in recent_iterations if log.algorithm == 'es']
        
        dashboard_data = {
            'run_metadata': asdict(self.run_metadata) if self.run_metadata else {},
            'current_iteration': len(self.iteration_logs),
            'total_pg_updates': len(self.pg_logs),
            'total_es_updates': len(self.es_logs),
            
            # Performance trends
            'fitness_trend': {
                'values': recent_fitness,
                'mean': sum(recent_fitness) / len(recent_fitness) if recent_fitness else 0.0,
                'std': (sum((x - sum(recent_fitness) / len(recent_fitness)) ** 2 for x in recent_fitness) / len(recent_fitness)) ** 0.5 if len(recent_fitness) > 1 else 0.0
            },
            
            'reward_trend': {
                'values': recent_rewards,
                'mean': sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0,
                'std': (sum((x - sum(recent_rewards) / len(recent_rewards)) ** 2 for x in recent_rewards) / len(recent_rewards)) ** 0.5 if len(recent_rewards) > 1 else 0.0
            },
            
            # Algorithm performance comparison
            'pg_performance': {
                'iterations': len(pg_iterations),
                'avg_reward': sum(log.reward for log in pg_iterations if log.reward is not None) / len([log for log in pg_iterations if log.reward is not None]) if pg_iterations else 0.0
            },
            
            'es_performance': {
                'iterations': len(es_iterations),
                'avg_fitness': sum(log.fitness for log in es_iterations if log.fitness is not None) / len([log for log in es_iterations if log.fitness is not None]) if es_iterations else 0.0
            },
            
            # Safety analysis
            'safety_summary': {
                'total_violations': sum(len(log.safety_violations) for log in recent_iterations),
                'violation_rate': sum(1 for log in recent_iterations if log.safety_violations) / len(recent_iterations) if recent_iterations else 0.0,
                'recent_alerts': sum(len(log.supervisor_alerts) for log in recent_iterations[-10:]) if len(recent_iterations) >= 10 else 0
            },
            
            # Parameter drift summary
            'parameter_drift': {
                'recent_drift': self.parameter_drift_history[-10:] if len(self.parameter_drift_history) >= 10 else self.parameter_drift_history
            }
        }
        
        return dashboard_data
    
    def _flush_iteration_logs(self):
        """Flush iteration logs to disk"""
        
        if not self.current_run_dir or not self.iteration_logs:
            return
        
        # Save batch of iterations
        batch_file = self.current_run_dir / "iterations" / f"batch_{len(self.iteration_logs):06d}.json"
        
        serializable_logs = []
        for log in self.iteration_logs:
            log_dict = asdict(log)
            log_dict['timestamp'] = log_dict['timestamp'].isoformat()
            serializable_logs.append(log_dict)
        
        with open(batch_file, 'w') as f:
            json.dump(serializable_logs, f, indent=2)
        
        # Clear buffer
        self.iteration_logs.clear()
        
        self.logger.debug(f"Flushed {len(serializable_logs)} iteration logs to {batch_file}")
    
    def _save_run_metadata(self):
        """Save run metadata to disk"""
        
        if not self.current_run_dir or not self.run_metadata:
            return
        
        metadata_dict = asdict(self.run_metadata)
        metadata_dict['start_time'] = metadata_dict['start_time'].isoformat()
        
        with open(self.current_run_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def finalize_run(self):
        """Finalize and close current run"""
        
        if not self.current_run_id:
            return
        
        # Flush remaining logs
        if self.iteration_logs:
            self._flush_iteration_logs()
        
        # Save final summaries
        self._save_final_summaries()
        
        # Update run metadata with end time
        if self.run_metadata:
            self.run_metadata.tags.append("completed")
            self._save_run_metadata()
        
        self.logger.info(f"Finalized run: {self.current_run_id}")
        
        # Reset state
        self.current_run_id = None
        self.current_run_dir = None
        self.run_metadata = None
        self.iteration_logs.clear()
        self.pg_logs.clear()
        self.es_logs.clear()
        self.parameter_drift_history.clear()
    
    def _save_final_summaries(self):
        """Save final summary files"""
        
        if not self.current_run_dir:
            return
        
        # PG summary
        if self.pg_logs:
            with open(self.current_run_dir / "pg_logs" / "pg_summary.json", 'w') as f:
                json.dump({
                    'total_updates': len(self.pg_logs),
                    'final_metrics': self.pg_logs[-1] if self.pg_logs else {},
                    'performance_trend': [log['metrics'].get('loss', 0) for log in self.pg_logs[-50:]]
                }, f, indent=2)
        
        # ES summary
        if self.es_logs:
            with open(self.current_run_dir / "es_logs" / "es_summary.json", 'w') as f:
                json.dump({
                    'total_generations': len(self.es_logs),
                    'final_metrics': self.es_logs[-1] if self.es_logs else {},
                    'fitness_trend': [log['metrics'].get('fitness', 0) for log in self.es_logs[-50:]]
                }, f, indent=2)
        
        # Combined summary
        summary = {
            'run_id': self.current_run_id,
            'total_iterations': len(self.pg_logs) + len(self.es_logs),
            'pg_iterations': len(self.pg_logs),
            'es_iterations': len(self.es_logs),
            'final_fitness_history': self.fitness_history[-100:],
            'final_reward_history': self.reward_history[-100:],
            'parameter_drift_summary': self.parameter_drift_history[-50:]
        }
        
        with open(self.current_run_dir / "run_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

# Alias for backward compatibility and clearer naming
PGESLoggingSystem = LoggingManager
