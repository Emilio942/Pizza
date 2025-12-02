"""
Hybrid PG-ES Training System
Combines Policy Gradient and Evolution Strategy for robust hardware optimization
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os
from enum import Enum

# Local imports
from hardware_simulator import HardwareSimulator, SimulationState
from pg_optimizer import PolicyGradientOptimizer, PGMetrics
from es_optimizer import EvolutionStrategyOptimizer, ESMetrics
from supervisor import FailSafeSupervisor, SupervisorAlert, AlertLevel
from logging_system import PGESLoggingSystem
from visualization_dashboard import VisualizationDashboard
from fitness_normalizer import FitnessNormalizer
# Enhanced production-ready components
from advanced_temperature_controller import AdvancedTemperatureController
from enhanced_safety_monitor import EnhancedSafetyMonitor

@dataclass
class HybridMetrics:
    """Combined metrics from both optimizers"""
    pg_metrics: PGMetrics = field(default_factory=PGMetrics)
    es_metrics: ESMetrics = field(default_factory=ESMetrics)
    combined_fitness: float = 0.0
    pg_contribution_ratio: float = 0.0
    es_contribution_ratio: float = 0.0
    total_episodes: int = 0
    total_evaluations: int = 0
    wall_time: float = 0.0
    supervisor_alerts: int = 0

class HybridPGESTrainer:
    """
    Main training system combining PG and ES optimization
    Implements Phase B requirements from aufgabenliste.md
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize logging system
        self.logging_system = PGESLoggingSystem(config)
        
        # Initialize visualization dashboard
        self.dashboard = VisualizationDashboard(config)
        
        # Initialize simulator
        self.simulator = HardwareSimulator(config)
        state_dim = self.simulator.get_observation_space_size()
        action_dim = self.simulator.get_action_space_size()
        
        # Initialize optimizers
        self.pg_optimizer = PolicyGradientOptimizer(config, state_dim, action_dim)
        self.es_optimizer = EvolutionStrategyOptimizer(config, action_dim)  # ES optimizes action parameters
        
        # Initialize supervisor
        self.supervisor = FailSafeSupervisor(config)
        
        # Training parameters
        self.pg_es_ratio = config.hybrid.pg_es_ratio
        self.alternating_mode = config.hybrid.alternating_mode
        self.warm_start_pg_iterations = config.hybrid.warm_start_pg_iterations
        self.share_experience = config.hybrid.share_experience
        self.adaptive_ratio = config.hybrid.adaptive_ratio
        
        # Training state
        self.iteration = 0
        self.start_time = time.time()
        self.metrics = HybridMetrics()
        self.training_log = []
        
        # Experience sharing
        if self.share_experience:
            self.shared_experience_buffer = []
            self.buffer_size = config.hybrid.experience_buffer_size
        
        # Current training mode
        self.current_mode = "warmup"  # warmup, pg, es, hybrid
        self.mode_iteration = 0
        
        # Initialize fitness normalization
        self.fitness_normalizer = FitnessNormalizer(config.logging.fitness_normalization_method)
        
        # Initialize enhanced components for production-ready optimization
        self.temperature_controller = None
        self.safety_monitor = None
        
        if config.hybrid.get('enable_temperature_optimization', True):
            temp_config = config.simulation.to_dict()
            temp_config.update(config.hybrid.get('temperature_config', {}))
            self.temperature_controller = AdvancedTemperatureController(temp_config)
            self.logger.info("Advanced temperature controller initialized")
        
        if config.hybrid.get('enable_enhanced_safety', True):
            safety_config = config.simulation.to_dict()
            safety_config.update(config.hybrid.get('safety_config', {}))
            self.safety_monitor = EnhancedSafetyMonitor(safety_config)
            self.logger.info("Enhanced safety monitor initialized")
        
    # Note: EnhancedSafetyManager not implemented; monitor is sufficient for tests

        # Enhanced training parameters
        self.optimization_targets = {
            'temperature_stability_target': config.hybrid.get('temperature_stability_target', 0.95),
            'safety_compliance_target': config.hybrid.get('safety_compliance_target', 0.99),
            'enable_adaptive_optimization': config.hybrid.get('enable_adaptive_optimization', True)
        }
        
        self.logger.info(f"Hybrid PG-ES Trainer initialized - State dim: {state_dim}, Action dim: {action_dim}")
        
        # Log initialization
        self.logging_system.log_initialization({
            'state_dim': state_dim,
            'action_dim': action_dim,
            'pg_es_ratio': self.pg_es_ratio,
            'warm_start_iterations': self.warm_start_pg_iterations
        })
    
    def train(self, total_iterations: int = 10000) -> HybridMetrics:
        """
        Main training loop
        Implements hybrid PG-ES optimization strategy
        """
        
        self.logger.info(f"Starting hybrid training for {total_iterations} iterations")
        
        # Initialize ES population
        initial_action = np.zeros(self.simulator.get_action_space_size())
        self.es_optimizer.initialize_population(initial_action)
        
        try:
            for self.iteration in range(total_iterations):
                
                # Check supervisor fail-safes
                if self._check_supervisor_systems():
                    self.logger.error("Training stopped by supervisor system")
                    break
                
                # Determine training mode for this iteration
                training_mode = self._determine_training_mode()
                
                # Execute training step
                if training_mode == "pg":
                    self._train_pg_step()
                elif training_mode == "es":
                    self._train_es_step()
                elif training_mode == "hybrid":
                    self._train_hybrid_step()
                
                # Update metrics
                self._update_metrics()
                
                # Logging and checkpointing
                if self.iteration % 100 == 0:
                    self._log_progress()
                    
                    # Update visualization dashboard
                    if hasattr(self, 'dashboard'):
                        try:
                            self.dashboard.update_training_progress({
                                'iteration': self.iteration,
                                'pg_metrics': self.metrics.pg_metrics,
                                'es_metrics': self.metrics.es_metrics,
                                'supervisor_status': self.supervisor.get_status(),
                                'training_mode': training_mode
                            })
                        except Exception as e:
                            self.logger.warning(f"Dashboard update failed: {e}")
                
                if self.iteration % 1000 == 0:
                    self._save_checkpoint()
                    
                    # Generate visualization plots
                    if hasattr(self, 'dashboard'):
                        try:
                            self.dashboard.generate_plots()
                        except Exception as e:
                            self.logger.warning(f"Plot generation failed: {e}")
                
                # Adaptive ratio adjustment
                if self.adaptive_ratio and self.iteration % 500 == 0:
                    self._adjust_pg_es_ratio()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        
        finally:
            self._save_final_results()
        
        return self.metrics
    
    def _determine_training_mode(self) -> str:
        """Determine which optimization method to use this iteration"""
        
        # Warm-up phase - only PG
        if self.iteration < self.warm_start_pg_iterations:
            return "pg"
        
        # Alternating mode
        if self.alternating_mode:
            if (self.iteration // 100) % 2 == 0:
                return "pg"
            else:
                return "es"
        
        # Probabilistic mixing based on ratio
        if np.random.random() < self.pg_es_ratio:
            return "pg"
        else:
            return "es"
    
    def _train_pg_step(self):
        """Execute one Policy Gradient training step"""
        
        # Generate unique iteration ID
        iteration_id = self.logging_system.generate_iteration_id()
        
        # Log PG step start
        self.logging_system.log_pg_update(iteration_id, {
            'iteration': self.iteration,
            'mode': 'pg_step_start',
            'timestamp': time.time()
        })
        
        # Run episode with current policy
        state = self.simulator.reset_state()
        episode_reward = 0.0
        episode_length = 0
        
        # Enhanced PG training with temperature control and safety monitoring
        temperature_stability_bonus = 0.0
        safety_compliance_bonus = 0.0
        
        for step in range(100):  # Max episode length
            
            # Get action from PG policy
            state_vector = state.get_state_vector() if hasattr(state, 'get_state_vector') else np.zeros(self.simulator.get_observation_space_size())
            action, action_info = self.pg_optimizer.select_action(state_vector)
            
            # Convert action to simulator control inputs
            control_inputs = self._action_to_control_inputs(action)
            
            # Apply enhanced temperature control guidance
            if self.temperature_controller is not None:
                temp_control = self.temperature_controller.update_control(
                    current_temp=state.temperature,
                    target_temp=65.0,  # Target temperature
                    dt=state.dt,
                    power_limit=1.0,
                    environmental_disturbances={
                        'ambient_temp_delta': 0.0,
                        'humidity_thermal_effect': state.humidity * 0.01,
                        'power_variation': 0.0
                    }
                )
                # Use temperature control as guidance for reward shaping
                temp_stability = self.temperature_controller.get_stability_metrics()
                stability_score = temp_stability.get('temperature_stability_percentage', 0.0)
                temperature_stability_bonus += stability_score * 0.001  # Small bonus per step
            
            # Step simulator
            next_state, reward, done, sim_info = self.simulator.step(control_inputs)
            
            # Enhanced safety monitoring with reward shaping
            if self.safety_monitor is not None:
                is_safe, violations = self.safety_monitor.check_safety(
                    simulation_state=next_state.__dict__,
                    control_inputs=control_inputs
                )
                
                if is_safe:
                    safety_compliance_bonus += 0.001  # Small bonus for safe operation
                else:
                    # Penalty for violations
                    violation_penalty = sum(0.01 if v.severity.value == 'warning' else 
                                          0.05 if v.severity.value == 'critical' else 
                                          0.1 for v in violations)
                    reward -= violation_penalty
            
            # Enhanced reward with stability and safety bonuses
            enhanced_reward = reward + temperature_stability_bonus + safety_compliance_bonus
            
            # Store experience with enhanced reward
            next_state_vector = next_state.get_state_vector() if hasattr(next_state, 'get_state_vector') else np.zeros(self.simulator.get_observation_space_size())
            self.pg_optimizer.store_experience(state_vector, action, enhanced_reward, next_state_vector, done, action_info)
            
            # Share experience if enabled
            if self.share_experience:
                self._add_to_shared_experience(state_vector, action, enhanced_reward, next_state_vector, done, action_info)
            
            episode_reward += enhanced_reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update policy
        self.metrics.pg_metrics = self.pg_optimizer.update_policy()
        self.pg_optimizer.episode_end(episode_reward, episode_length)
        
        # Log PG step completion with metrics
        self.logging_system.log_pg_update(iteration_id, {
            'iteration': self.iteration,
            'mode': 'pg_step_complete',
            'reward': episode_reward,
            'episode_length': episode_length,
            'loss': self.metrics.pg_metrics.loss,
            # Some codebases name this 'policy_entropy'; our metrics expose 'entropy_loss'.
            # Provide both keys for compatibility, using safe defaults.
            'policy_entropy': getattr(self.metrics.pg_metrics, 'entropy_loss', getattr(self.metrics.pg_metrics, 'policy_entropy', 0.0)),
            'entropy_loss': getattr(self.metrics.pg_metrics, 'entropy_loss', 0.0),
            'value_loss': self.metrics.pg_metrics.value_loss,
            'timestamp': time.time()
        })
        
        # Log reward to history
        self.logging_system.log_reward(episode_reward)
        
        self.logger.debug(f"PG step completed - Reward: {episode_reward:.3f}, Loss: {self.metrics.pg_metrics.loss:.6f}")
    
    def _train_es_step(self):
        """Execute one Evolution Strategy training step"""
        
        # Generate unique iteration ID
        iteration_id = self.logging_system.generate_iteration_id()
        
        # Log ES step start
        self.logging_system.log_es_update(iteration_id, {
            'iteration': self.iteration,
            'mode': 'es_step_start',
            'timestamp': time.time()
        })
        
        # Check if we need to start new generation
        if self.es_optimizer.all_evaluated():
            # Update population based on fitness
            generation_data = {
                'generation': self.es_optimizer.generation,
                'fitness_stats': {
                    'mean': np.mean(self.es_optimizer.fitness_values),
                    'std': np.std(self.es_optimizer.fitness_values),
                    'max': np.max(self.es_optimizer.fitness_values),
                    'min': np.min(self.es_optimizer.fitness_values)
                }
            }
            
            # Apply fitness normalization
            if hasattr(self, 'fitness_normalizer'):
                normalized_fitness = self.fitness_normalizer.normalize(self.es_optimizer.fitness_values)
                generation_data['fitness_stats']['normalized_mean'] = np.mean(normalized_fitness)
                generation_data['fitness_stats']['normalized_std'] = np.std(normalized_fitness)
            
            self.logging_system.log_es_update(iteration_id, {
                'iteration': self.iteration,
                'mode': 'es_generation_complete',
                'generation_data': generation_data,
                'timestamp': time.time()
            })
            
            self.metrics.es_metrics = self.es_optimizer.update_population()
        
        # Get next individual to evaluate
        try:
            individual, individual_index = self.es_optimizer.get_next_individual()
        except ValueError:
            # No individuals to evaluate - this shouldn't happen
            self.logger.warning("No ES individuals to evaluate")
            return
        
        # Evaluate individual
        fitness = self._evaluate_es_individual(individual)
        
        # Set fitness
        self.es_optimizer.set_fitness(individual_index, fitness)
        
        # Log fitness
        self.logging_system.log_fitness(fitness)
        
        # Log ES step completion
        self.logging_system.log_es_update(iteration_id, {
            'iteration': self.iteration,
            'mode': 'es_individual_evaluated',
            'individual_index': individual_index,
            'fitness': fitness,
            'timestamp': time.time()
        })
        
        self.logger.debug(f"ES evaluation completed - Individual {individual_index}, Fitness: {fitness:.6f}")
    
    def _train_hybrid_step(self):
        """Execute combined PG-ES training step"""
        
        # For now, run both sequentially
        # In advanced implementation, could run in parallel
        
        if np.random.random() < 0.5:
            self._train_pg_step()
        else:
            self._train_es_step()
    
    def _action_to_control_inputs(self, action: np.ndarray) -> Dict[str, float]:
        """Convert ML action to simulator control inputs"""
        
        # Map normalized action [-1, 1] to control ranges
        control_inputs = {
            'target_power': 0.1 + 0.9 * (action[0] + 1) / 2,  # 0.1 to 1.0 W
            'cooling_effort': max(0, min(1, (action[1] + 1) / 2)),  # 0 to 1
            'voltage_regulation': 3.0 + 0.6 * action[2]  # 3.0 to 3.6 V
        }
        
        return control_inputs
    
    def _evaluate_es_individual(self, individual: np.ndarray) -> float:
        """Evaluate ES individual (parameter set) in simulation"""
        
        # Reset simulator
        state = self.simulator.reset_state()
        total_reward = 0.0
        
        # Enhanced evaluation with temperature control and safety monitoring
        temperature_stability_score = 0.0
        safety_compliance_score = 0.0
        
        # Run simulation with fixed parameters
        for step in range(100):  # Episode length
            
            # Use individual as action parameters
            control_inputs = self._action_to_control_inputs(individual)
            
            # Apply enhanced temperature control if available
            if self.temperature_controller is not None:
                temp_control = self.temperature_controller.update_control(
                    current_temp=state.temperature,
                    target_temp=65.0,  # Target temperature
                    dt=state.dt,
                    power_limit=1.0,
                    environmental_disturbances={
                        'ambient_temp_delta': 0.0,
                        'humidity_thermal_effect': state.humidity * 0.01,
                        'power_variation': 0.0
                    }
                )
                # Merge temperature control with individual parameters
                for key, value in temp_control.items():
                    if key in control_inputs:
                        control_inputs[key] = 0.7 * control_inputs[key] + 0.3 * value
            
            # Step simulator
            next_state, reward, done, sim_info = self.simulator.step(control_inputs)
            
            # Enhanced safety monitoring
            if self.safety_monitor is not None:
                is_safe, violations = self.safety_monitor.check_safety(
                    simulation_state=next_state.__dict__,
                    control_inputs=control_inputs
                )
                
                # Calculate safety compliance score
                safety_compliance_score += 1.0 if is_safe else 0.0
                
                # Penalize violations
                if not is_safe:
                    violation_penalty = sum(0.1 if v.severity.value == 'warning' else 
                                          0.5 if v.severity.value == 'critical' else 
                                          1.0 for v in violations)
                    reward -= violation_penalty
            
            # Calculate temperature stability
            if self.temperature_controller is not None:
                stability_metrics = self.temperature_controller.get_stability_metrics()
                temp_stability = stability_metrics.get('temperature_stability_percentage', 0.0)
                temperature_stability_score += temp_stability / 100.0
                
                # Bonus for high stability
                if temp_stability > 90.0:
                    reward += 0.1
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Calculate enhanced fitness with stability and safety bonuses
        base_fitness = total_reward / max(1, step + 1)  # Average reward per step
        
        # Temperature stability bonus (up to 20% improvement)
        temp_bonus = 0.0
        if self.temperature_controller is not None and step > 0:
            avg_temp_stability = temperature_stability_score / (step + 1)
            temp_bonus = 0.2 * avg_temp_stability  # Up to 20% bonus
        
        # Safety compliance bonus (up to 15% improvement)
        safety_bonus = 0.0
        if self.safety_monitor is not None and step > 0:
            avg_safety_compliance = safety_compliance_score / (step + 1)
            safety_bonus = 0.15 * avg_safety_compliance  # Up to 15% bonus
        
        enhanced_fitness = base_fitness * (1.0 + temp_bonus + safety_bonus)
        
        return enhanced_fitness
    
    def _add_to_shared_experience(self, state, action, reward, next_state, done, info):
        """Add experience to shared buffer"""
        
        if not self.share_experience:
            return
        
        experience = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'info': info.copy(),
            'source': 'pg' if self.current_mode == 'pg' else 'es',
            'iteration': self.iteration
        }
        
        self.shared_experience_buffer.append(experience)
        
        # Keep buffer size manageable
        if len(self.shared_experience_buffer) > self.buffer_size:
            self.shared_experience_buffer.pop(0)
    
    def _check_supervisor_systems(self) -> bool:
        """Check all supervisor fail-safe systems"""
        
        # Get current state
        current_state = self.simulator.state
        
        # Get domain randomization state (placeholder)
        domain_rand_state = {
            'noise_levels': {'temperature': 0.1, 'humidity': 0.05},
            'parameter_drift': 0.02,
            'extreme_events': 5
        }
        
        # Run all checks
        alerts = self.supervisor.check_all_systems(
            simulation_state=current_state,
            pg_metrics=self.metrics.pg_metrics.__dict__,
            es_metrics=self.metrics.es_metrics.__dict__,
            domain_rand_state=domain_rand_state
        )
        
        # Count alerts by severity
        critical_alerts = [a for a in alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
        self.metrics.supervisor_alerts = len(alerts)
        
        # Log alerts to both system logger and logging system
        for alert in alerts:
            # Log to system logger
            if alert.level == AlertLevel.EMERGENCY:
                self.logger.error(f"EMERGENCY: {alert.message}")
            elif alert.level == AlertLevel.CRITICAL:
                self.logger.error(f"CRITICAL: {alert.message}")
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(f"WARNING: {alert.message}")
            else:
                self.logger.info(f"INFO: {alert.message}")
            
            # Log to logging system
            if hasattr(self, 'logging_system'):
                self.logging_system.log_supervisor_alert({
                    'iteration': self.iteration,
                    'alert_level': alert.level.name,
                    'check_type': getattr(alert, 'category', ''),
                    'message': alert.message,
                    'timestamp': time.time()
                })
        
        # Log supervisor status
        if hasattr(self, 'logging_system'):
            supervisor_status = self.supervisor.get_status()
            self.logging_system.log_event({
                'iteration': self.iteration,
                'event_type': 'supervisor_check',
                'supervisor_status': supervisor_status,
                'total_alerts': len(alerts),
                'critical_alerts': len(critical_alerts),
                'system_frozen': self.supervisor.system_frozen,
                'timestamp': time.time()
            })
        
        # Return True if system should stop
        return self.supervisor.system_frozen
    
    def _update_metrics(self):
        """Update combined training metrics"""
        
        # Calculate contributions
        pg_contrib = self.metrics.pg_metrics.contribution
        es_contrib = self.metrics.es_metrics.contribution
        total_contrib = pg_contrib + es_contrib
        
        if total_contrib > 0:
            self.metrics.pg_contribution_ratio = pg_contrib / total_contrib
            self.metrics.es_contribution_ratio = es_contrib / total_contrib
        
        # Combined fitness (weighted average)
        pg_fitness = self.metrics.pg_metrics.advantage_mean if hasattr(self.metrics.pg_metrics, 'advantage_mean') else 0.0
        es_fitness = self.metrics.es_metrics.fitness if hasattr(self.metrics.es_metrics, 'fitness') else 0.0
        
        self.metrics.combined_fitness = (self.pg_es_ratio * pg_fitness + 
                                       (1 - self.pg_es_ratio) * es_fitness)
        
        # Totals
        self.metrics.total_episodes = self.metrics.pg_metrics.episodes_completed
        self.metrics.total_evaluations = self.metrics.es_metrics.evaluations
        self.metrics.wall_time = time.time() - self.start_time
    
    def _adjust_pg_es_ratio(self):
        """Adaptively adjust PG-ES ratio based on performance"""
        
        if not self.adaptive_ratio:
            return
        
        # Compare recent performance of PG vs ES
        pg_performance = self.metrics.pg_metrics.contribution
        es_performance = self.metrics.es_metrics.contribution
        
        adaptation_rate_default = 0.05
        if hasattr(self.config.hybrid, 'get'):
            adaptation_rate = self.config.hybrid.get('ratio_adaptation_rate', adaptation_rate_default)
        else:
            adaptation_rate = getattr(self.config.hybrid, 'ratio_adaptation_rate', adaptation_rate_default)
        
        if pg_performance > es_performance:
            # PG is performing better, increase its share
            self.pg_es_ratio = min(0.9, self.pg_es_ratio + adaptation_rate)
        elif es_performance > pg_performance:
            # ES is performing better, increase its share
            self.pg_es_ratio = max(0.1, self.pg_es_ratio - adaptation_rate)
        
        self.logger.debug(f"Adjusted PG-ES ratio to {self.pg_es_ratio:.3f}")
    
    def _log_progress(self):
        """Log training progress"""
        
        log_entry = {
            'iteration': self.iteration,
            'wall_time': self.metrics.wall_time,
            'combined_fitness': self.metrics.combined_fitness,
            'pg_contribution': self.metrics.pg_contribution_ratio,
            'es_contribution': self.metrics.es_contribution_ratio,
            'pg_loss': self.metrics.pg_metrics.loss,
            'es_fitness': self.metrics.es_metrics.fitness,
            'supervisor_alerts': self.metrics.supervisor_alerts,
            'current_mode': self.current_mode
        }
        
        self.training_log.append(log_entry)
        
        # Log to system logger
        self.logger.info(
            f"Iter {self.iteration:5d} | "
            f"Fitness: {self.metrics.combined_fitness:8.4f} | "
            f"PG: {self.metrics.pg_contribution_ratio:5.3f} | "
            f"ES: {self.metrics.es_contribution_ratio:5.3f} | "
            f"Alerts: {self.metrics.supervisor_alerts:2d}"
        )
        
        # Log to logging system
        if hasattr(self, 'logging_system'):
            self.logging_system.log_progress({
                'iteration': self.iteration,
                'timestamp': time.time(),
                'metrics': log_entry,
                'parameter_drift': self._calculate_parameter_drift(),
                'performance_trend': self._calculate_performance_trend()
            })
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        
        checkpoint_dir = os.path.join(self.config.logging.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.iteration:06d}.json")
        
        checkpoint = {
            'iteration': self.iteration,
            'metrics': {
                'combined_fitness': self.metrics.combined_fitness,
                'pg_metrics': self.metrics.pg_metrics.__dict__,
                'es_metrics': self.metrics.es_metrics.__dict__,
                'wall_time': self.metrics.wall_time
            },
            'config': {
                'pg_es_ratio': self.pg_es_ratio,
                'current_mode': self.current_mode
            },
            'training_log': self.training_log[-100:]  # Keep recent history
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save optimizer checkpoints
        self.pg_optimizer.save_checkpoint(os.path.join(checkpoint_dir, f"pg_{self.iteration:06d}.pkl"))
        self.es_optimizer.save_checkpoint(os.path.join(checkpoint_dir, f"es_{self.iteration:06d}.pkl"))
        
        self.logger.debug(f"Checkpoint saved at iteration {self.iteration}")
    
    def _save_final_results(self):
        """Save final training results and analysis"""
        
        results_dir = os.path.join(self.config.logging.log_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Final metrics
        final_results = {
            'training_completed': True,
            'total_iterations': self.iteration,
            'wall_time': self.metrics.wall_time,
            'final_metrics': {
                'combined_fitness': self.metrics.combined_fitness,
                'pg_contribution_ratio': self.metrics.pg_contribution_ratio,
                'es_contribution_ratio': self.metrics.es_contribution_ratio,
                'total_episodes': self.metrics.total_episodes,
                'total_evaluations': self.metrics.total_evaluations
            },
            'best_parameters': {
                'pg_best': self.pg_optimizer.get_performance_metrics(),
                'es_best': self.es_optimizer.get_best_parameters()
            },
            'supervisor_summary': self.supervisor.get_system_status(),
            'training_log': self.training_log
        }
        
        # Save results (ensure JSON-safe types)
        def _json_safe(obj):
            try:
                import numpy as _np  # local import to avoid global dependency in tests
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, _np.generic):
                    return obj.item()
            except Exception:
                pass

            if isinstance(obj, Enum):
                return getattr(obj, 'name', getattr(obj, 'value', str(obj)))

            try:
                from dataclasses import asdict as _asdict, is_dataclass as _is_dataclass
                if _is_dataclass(obj):
                    return _json_safe(_asdict(obj))
            except Exception:
                pass

            if isinstance(obj, dict):
                safe_dict = {}
                for k, v in obj.items():
                    if isinstance(k, Enum):
                        key = getattr(k, 'name', getattr(k, 'value', str(k)))
                    elif isinstance(k, (str, int, float, bool)) or k is None:
                        key = k
                    else:
                        key = str(k)
                    safe_dict[key] = _json_safe(v)
                return safe_dict

            if isinstance(obj, (list, tuple, set)):
                return [_json_safe(x) for x in obj]

            try:
                from pathlib import Path as _Path
                if isinstance(obj, _Path):
                    return str(obj)
            except Exception:
                pass

            if hasattr(obj, '__dict__'):
                return _json_safe(vars(obj))

            return obj
        with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
            json.dump(_json_safe(final_results), f, indent=2)
        
        # Save supervisor report
        self.supervisor.export_alerts_report(os.path.join(results_dir, 'supervisor_alerts.json'))
        
        self.logger.info(f"Final results saved to {results_dir}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current training status for monitoring"""
        
        return {
            'iteration': self.iteration,
            'wall_time': self.metrics.wall_time,
            'current_mode': self.current_mode,
            'pg_es_ratio': self.pg_es_ratio,
            'metrics': {
                'combined_fitness': self.metrics.combined_fitness,
                'pg_contribution': self.metrics.pg_contribution_ratio,
                'es_contribution': self.metrics.es_contribution_ratio,
                'supervisor_alerts': self.metrics.supervisor_alerts
            },
            'supervisor_status': self.supervisor.get_system_status(),
            'pg_performance': self.pg_optimizer.get_performance_metrics(),
            'es_statistics': self.es_optimizer.get_population_statistics()
        }
    
    def _calculate_parameter_drift(self) -> Dict[str, float]:
        """Calculate parameter drift metrics for logging"""
        try:
            # Get current parameters
            pg_params = self.pg_optimizer.get_current_parameters()
            es_params = self.es_optimizer.get_current_parameters()
            
            # Calculate drift from initial values (placeholder)
            drift_metrics = {
                'pg_param_drift': np.mean(np.abs(pg_params)) if pg_params is not None else 0.0,
                'es_param_drift': np.mean(np.abs(es_params)) if es_params is not None else 0.0,
                'combined_drift': 0.0
            }
            
            drift_metrics['combined_drift'] = (drift_metrics['pg_param_drift'] + drift_metrics['es_param_drift']) / 2
            
            return drift_metrics
        except Exception as e:
            self.logger.warning(f"Parameter drift calculation failed: {e}")
            return {'pg_param_drift': 0.0, 'es_param_drift': 0.0, 'combined_drift': 0.0}
    
    def _calculate_performance_trend(self) -> Dict[str, float]:
        """Calculate performance trend metrics"""
        try:
            if len(self.training_log) < 2:
                return {'fitness_trend': 0.0, 'loss_trend': 0.0, 'convergence_rate': 0.0}
            
            # Get recent performance data
            recent_fitness = [entry['combined_fitness'] for entry in self.training_log[-10:]]
            recent_losses = [entry['pg_loss'] for entry in self.training_log[-10:]]
            
            # Calculate trends
            fitness_trend = np.mean(np.diff(recent_fitness)) if len(recent_fitness) > 1 else 0.0
            loss_trend = np.mean(np.diff(recent_losses)) if len(recent_losses) > 1 else 0.0
            
            # Estimate convergence rate
            convergence_rate = np.std(recent_fitness) if len(recent_fitness) > 1 else 1.0
            
            return {
                'fitness_trend': float(fitness_trend),
                'loss_trend': float(loss_trend),
                'convergence_rate': float(convergence_rate)
            }
        except Exception as e:
            self.logger.warning(f"Performance trend calculation failed: {e}")
            return {'fitness_trend': 0.0, 'loss_trend': 0.0, 'convergence_rate': 0.0}
