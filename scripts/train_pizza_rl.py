#!/usr/bin/env python3
"""
Pizza RL Training Script (Aufgabe 4.1)
=====================================

Comprehensive training script for the Pizza RL system with multi-objective optimization.
Implements adaptive pizza recognition with energy efficiency and inference speed optimization.

Features:
- Multi-objective reward optimization (Accuracy + Energy + Speed)
- Integration with existing pizza datasets
- Realistic pizza recognition scenarios
- Comprehensive logging and monitoring
- Model checkpointing and evaluation
- Performance metrics tracking
"""

import argparse
import logging
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import wandb

# Local imports
from src.rl.ppo_pizza_agent import PPOPizzaAgent, PPOHyperparameters
from src.rl.pizza_rl_environment import PizzaRLEnvironment
from src.verification.pizza_verifier import PizzaVerifier
from src.utils.performance_logger import PerformanceLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/emilio/Documents/ai/pizza/logs/pizza_rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PizzaRLTrainer:
    """
    Comprehensive trainer for the Pizza RL system with multi-objective optimization.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_wandb: bool = True,
        device: str = 'auto'
    ):
        """
        Initialize the Pizza RL trainer.
        
        Args:
            config_path: Path to training configuration file
            use_wandb: Whether to use Weights & Biases for logging
            device: Device to use for training ('auto', 'cpu', 'cuda')
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device(device)
        self.use_wandb = use_wandb
        
        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.environment = None
        self.agent = None
        self.performance_logger = None
        
        # Training state
        self.training_start_time = None
        self.best_reward = float('-inf')
        self.training_metrics = []
        
        logger.info(f"Pizza RL Trainer initialized on device: {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration."""
        default_config = {
            'output_dir': '/home/emilio/Documents/ai/pizza/results/pizza_rl_training',
            'total_timesteps': 1000000,
            'save_interval': 10,
            'eval_interval': 20,
            'eval_episodes': 50,
            'max_episodes_per_eval': 100,
            
            # Environment configuration
            'environment': {
                'max_steps_per_episode': 10,
                'battery_capacity_mah': 2000.0,
                'initial_battery_level': 1.0,
                'task_dataset_path': '/home/emilio/Documents/ai/pizza/test_data',
                'enable_logging': True,
                'energy_weight': 0.3,
                'accuracy_weight': 0.5,
                'speed_weight': 0.2
            },
            
            # PPO hyperparameters
            'ppo_hyperparams': {
                'learning_rate': 3e-4,
                'batch_size': 256,
                'ppo_epochs': 10,
                'rollout_steps': 2048,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'advantage_normalization': True,
                'reward_normalization': True,
                'adaptive_entropy': True
            },
            
            # Multi-objective optimization
            'multi_objective': {
                'enable': True,
                'accuracy_target': 0.85,
                'energy_efficiency_target': 0.7,
                'inference_speed_target': 100.0,  # ms
                'safety_penalty_weight': 2.0,
                'adaptive_weights': True
            },
            
            # Evaluation configuration
            'evaluation': {
                'scenarios': [
                    'standard_lighting',
                    'low_light',
                    'high_contrast',
                    'temporal_sequence',
                    'energy_constrained',
                    'speed_critical'
                ],
                'deterministic': True,
                'save_trajectories': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configurations
            self._deep_update(default_config, user_config)
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _setup_device(self, device: str) -> str:
        """Setup training device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        return device
    
    def initialize_components(self):
        """Initialize the RL environment and agent."""
        logger.info("Initializing Pizza RL components...")
        
        # Initialize environment
        env_config = self.config['environment']
        self.environment = PizzaRLEnvironment(
            max_steps_per_episode=env_config['max_steps_per_episode'],
            battery_capacity_mah=env_config['battery_capacity_mah'],
            initial_battery_level=env_config['initial_battery_level'],
            task_dataset_path=env_config['task_dataset_path'],
            device=self.device,
            enable_logging=env_config['enable_logging']
        )
        
        # Configure multi-objective rewards
        if self.config['multi_objective']['enable']:
            self.environment.set_reward_weights(
                accuracy_weight=env_config['accuracy_weight'],
                energy_weight=env_config['energy_weight'],
                speed_weight=env_config['speed_weight']
            )
        
        # Initialize PPO hyperparameters
        ppo_config = self.config['ppo_hyperparams']
        hyperparams = PPOHyperparameters(
            learning_rate=ppo_config['learning_rate'],
            batch_size=ppo_config['batch_size'],
            ppo_epochs=ppo_config['ppo_epochs'],
            rollout_steps=ppo_config['rollout_steps'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_ratio=ppo_config['clip_ratio'],
            value_loss_coef=ppo_config['value_loss_coef'],
            entropy_coef=ppo_config['entropy_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            advantage_normalization=ppo_config['advantage_normalization'],
            reward_normalization=ppo_config['reward_normalization'],
            adaptive_entropy=ppo_config['adaptive_entropy']
        )
        
        # Initialize PPO agent
        self.agent = PPOPizzaAgent(
            environment=self.environment,
            hyperparams=hyperparams,
            device=self.device,
            model_save_dir=self.checkpoints_dir
        )
        
        # Initialize performance logger
        self.performance_logger = PerformanceLogger(
            log_dir=str(self.logs_dir),
            enable_memory_tracking=True,
            enable_energy_tracking=True
        )
        
        logger.info("✓ All components initialized successfully")
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.use_wandb:
            return
        
        try:
            wandb.init(
                project="pizza-rl-training",
                name=f"pizza_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                save_code=True,
                dir=str(self.logs_dir)
            )
            logger.info("✓ Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def evaluate_agent(self, num_episodes: int = None) -> Dict:
        """
        Comprehensive evaluation of the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        if num_episodes is None:
            num_episodes = self.config['eval_episodes']
        
        logger.info(f"Evaluating agent for {num_episodes} episodes...")
        
        # Standard evaluation
        eval_results = self.agent.evaluate(
            num_episodes=num_episodes,
            deterministic=self.config['evaluation']['deterministic']
        )
        
        # Scenario-based evaluation
        scenario_results = {}
        for scenario in self.config['evaluation']['scenarios']:
            scenario_results[scenario] = self._evaluate_scenario(scenario, num_episodes // len(self.config['evaluation']['scenarios']))
        
        # Combine results
        comprehensive_results = {
            'standard_evaluation': eval_results,
            'scenario_evaluation': scenario_results,
            'evaluation_timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes
        }
        
        # Log key metrics
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Reward: {eval_results['mean_reward']:.3f}")
        logger.info(f"  Mean Episode Length: {eval_results['mean_episode_length']:.1f}")
        logger.info(f"  Success Rate: {eval_results['success_rate']:.2%}")
        logger.info(f"  Average Accuracy: {eval_results['average_accuracy']:.3f}")
        logger.info(f"  Average Energy Efficiency: {eval_results['average_energy_efficiency']:.3f}")
        
        return comprehensive_results
    
    def _evaluate_scenario(self, scenario: str, num_episodes: int) -> Dict:
        """Evaluate agent on specific scenario."""
        logger.info(f"Evaluating scenario: {scenario}")
        
        # Configure environment for scenario
        original_config = self.environment.get_config()
        self._configure_scenario(scenario)
        
        try:
            # Run evaluation
            results = self.agent.evaluate(num_episodes=num_episodes, deterministic=True)
            results['scenario'] = scenario
            
        finally:
            # Restore original configuration
            self.environment.set_config(original_config)
        
        return results
    
    def _configure_scenario(self, scenario: str):
        """Configure environment for specific evaluation scenario."""
        if scenario == 'low_light':
            # Simulate low light conditions
            self.environment.set_image_difficulty_modifier(0.3)
        elif scenario == 'high_contrast':
            # Simulate high contrast conditions
            self.environment.set_image_difficulty_modifier(-0.2)
        elif scenario == 'energy_constrained':
            # Start with low battery
            self.environment.reset_battery_level(0.2)
        elif scenario == 'speed_critical':
            # Set strict time constraints
            self.environment.set_time_constraint_modifier(0.5)
        elif scenario == 'temporal_sequence':
            # Enable temporal sequence evaluation
            self.environment.enable_temporal_sequence_mode(True)
        # Add more scenarios as needed
    
    def train(self):
        """
        Main training loop with multi-objective optimization.
        """
        logger.info("Starting Pizza RL Training...")
        self.training_start_time = time.time()
        
        # Setup logging
        self.setup_wandb()
        
        # Training configuration
        total_timesteps = self.config['total_timesteps']
        save_interval = self.config['save_interval']
        eval_interval = self.config['eval_interval']
        
        # Training loop
        try:
            training_info = self.agent.train(
                total_timesteps=total_timesteps,
                save_interval=save_interval
            )
            
            # Log final training results
            self._log_training_results(training_info)
            
            # Final comprehensive evaluation
            logger.info("Performing final comprehensive evaluation...")
            final_eval = self.evaluate_agent(self.config['max_episodes_per_eval'])
            
            # Save final results
            self._save_final_results(training_info, final_eval)
            
            logger.info("✓ Pizza RL Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint("interrupted")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _log_training_results(self, training_info: Dict):
        """Log training results to various outputs."""
        logger.info("Training Summary:")
        logger.info(f"  Total Timesteps: {training_info['total_timesteps']}")
        logger.info(f"  Training Episodes: {training_info['training_episodes']}")
        logger.info(f"  Iterations Completed: {training_info['iterations_completed']}")
        
        final_eval = training_info['final_evaluation']
        logger.info(f"Final Evaluation:")
        logger.info(f"  Mean Reward: {final_eval['mean_reward']:.3f}")
        logger.info(f"  Success Rate: {final_eval['success_rate']:.2%}")
        logger.info(f"  Average Accuracy: {final_eval['average_accuracy']:.3f}")
        logger.info(f"  Average Energy Efficiency: {final_eval['average_energy_efficiency']:.3f}")
        
        # Log to wandb if available
        if self.use_wandb:
            wandb.log({
                'final/total_timesteps': training_info['total_timesteps'],
                'final/training_episodes': training_info['training_episodes'],
                'final/mean_reward': final_eval['mean_reward'],
                'final/success_rate': final_eval['success_rate'],
                'final/average_accuracy': final_eval['average_accuracy'],
                'final/average_energy_efficiency': final_eval['average_energy_efficiency']
            })
    
    def _save_final_results(self, training_info: Dict, final_eval: Dict):
        """Save final training and evaluation results."""
        results = {
            'training_info': training_info,
            'final_evaluation': final_eval,
            'training_duration': time.time() - self.training_start_time,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to: {results_path}")
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        if self.agent:
            self.agent.save_checkpoint(checkpoint_name)
            logger.info(f"Checkpoint saved: {checkpoint_name}")


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Pizza RL Training Script")
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Training device')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--timesteps', type=int, help='Total training timesteps (overrides config)')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = PizzaRLTrainer(
        config_path=args.config,
        use_wandb=not args.no_wandb,
        device=args.device
    )
    
    # Override config with command line arguments
    if args.timesteps:
        trainer.config['total_timesteps'] = args.timesteps
    if args.output_dir:
        trainer.config['output_dir'] = args.output_dir
    
    # Initialize components
    trainer.initialize_components()
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
