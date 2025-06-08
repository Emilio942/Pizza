#!/usr/bin/env python3
"""
PPO Agent for Pizza RL Training (Aufgabe 3.2)
==============================================

Implements a Proximal Policy Optimization (PPO) agent specifically designed
for adaptive pizza recognition with multi-objective optimization.

Features:
- PPO algorithm implementation with clipped surrogate objective
- Multi-objective reward handling (accuracy + energy + speed + safety)
- Integration with Pizza RL Environment
- Support for continuous training and evaluation
- Model checkpointing and restoration
- Performance monitoring and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt

# Local imports
from .adaptive_pizza_policy import AdaptivePizzaRecognitionPolicy, SystemState, InferenceStrategy
from .pizza_rl_environment import PizzaRLEnvironment, PizzaRLEnvironmentFactory

logger = logging.getLogger(__name__)


@dataclass
class PPOHyperparameters:
    """PPO hyperparameters for pizza recognition training."""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_ratio: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training schedule
    ppo_epochs: int = 4  # PPO update epochs per iteration
    batch_size: int = 64
    rollout_steps: int = 2048  # Steps per rollout
    
    # Multi-objective specific
    adaptive_entropy: bool = True  # Adapt entropy coefficient
    reward_normalization: bool = True  # Normalize rewards
    advantage_normalization: bool = True  # Normalize advantages


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring."""
    episode: int = 0
    total_steps: int = 0
    average_reward: float = 0.0
    average_accuracy: float = 0.0
    average_energy_efficiency: float = 0.0
    average_inference_time: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    explained_variance: float = 0.0
    clipfrac: float = 0.0
    lr: float = 0.0


class PPOBuffer:
    """
    Buffer for storing rollout data for PPO training.
    """
    
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = 'cpu'):
        """
        Initialize PPO buffer.
        
        Args:
            capacity: Maximum number of steps to store
            obs_dim: Observation dimension
            act_dim: Action dimension (for discrete actions)
            device: Device for tensors
        """
        self.capacity = capacity
        self.device = device
        
        # Storage buffers
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.long, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # GAE buffers
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        # Pointer and size
        self.ptr = 0
        self.size = 0
    
    def store(self, obs: np.ndarray, act: np.ndarray, log_prob: float, 
              reward: float, value: float, done: bool) -> None:
        """Store a single step."""
        if self.ptr >= self.capacity:
            logger.warning("Buffer overflow, overwriting old data")
            self.ptr = 0
        
        self.observations[self.ptr] = torch.from_numpy(obs).float()
        self.actions[self.ptr] = torch.from_numpy(act).long()
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
    
    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute Generalized Advantage Estimation."""
        last_advantage = 0.0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_advantage = (
                delta + gamma * gae_lambda * next_non_terminal * last_advantage
            )
        
        self.returns = self.advantages + self.values[:self.size]
    
    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get random batches for training."""
        if self.size < batch_size:
            # Return single batch with all data if buffer is smaller
            batch_size = self.size
        
        indices = torch.randperm(self.size, device=self.device)
        batches = []
        
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]
            
            batch = {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
                'values': self.values[batch_indices]
            }
            batches.append(batch)
        
        return batches
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOPizzaAgent:
    """
    PPO Agent for adaptive pizza recognition training.
    
    This agent uses the Proximal Policy Optimization algorithm to train
    an adaptive policy for pizza recognition that balances accuracy,
    energy efficiency, and speed constraints.
    """
    
    def __init__(
        self,
        environment: PizzaRLEnvironment,
        hyperparams: Optional[PPOHyperparameters] = None,
        device: str = 'cpu',
        model_save_dir: str = 'models/rl_checkpoints'
    ):
        """
        Initialize PPO Pizza Agent.
        
        Args:
            environment: Pizza RL environment
            hyperparams: PPO hyperparameters
            device: Device for training
            model_save_dir: Directory to save model checkpoints
        """
        self.env = environment
        self.device = device
        self.hyperparams = hyperparams or PPOHyperparameters()
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize policy network
        self.policy = AdaptivePizzaRecognitionPolicy(
            state_dim=13,  # SystemState dimension
            hidden_dim=128
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=self.hyperparams.learning_rate
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Initialize buffer
        obs_dim = environment.observation_space.shape[0]
        act_dim = len(environment.action_space.nvec)  # MultiDiscrete action space
        self.buffer = PPOBuffer(
            capacity=self.hyperparams.rollout_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device
        )
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.total_env_steps = 0
        
        # Metrics tracking
        self.training_metrics = []
        self.episode_rewards = deque(maxlen=100)
        self.episode_accuracies = deque(maxlen=100)
        self.episode_energy_efficiency = deque(maxlen=100)
        
        # Adaptive entropy coefficient
        self.entropy_coef = self.hyperparams.entropy_coef
        
        logger.info(f"PPO Pizza Agent initialized with {sum(p.numel() for p in self.policy.parameters())} parameters")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action using current policy.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            
            # Get policy outputs
            policy_outputs = self.policy(obs_tensor)
            
            actions = []
            log_probs = []
            
            # Sample model variant (3 options)
            model_dist = Categorical(probs=policy_outputs["model_probs"])
            if deterministic:
                model_action = torch.argmax(policy_outputs["model_probs"], dim=-1)
            else:
                model_action = model_dist.sample()
            actions.append(model_action.item())
            log_probs.append(model_dist.log_prob(model_action))
            
            # Sample processing intensity (3 options)
            intensity_dist = Categorical(probs=policy_outputs["intensity_probs"])
            if deterministic:
                intensity_action = torch.argmax(policy_outputs["intensity_probs"], dim=-1)
            else:
                intensity_action = intensity_dist.sample()
            actions.append(intensity_action.item())
            log_probs.append(intensity_dist.log_prob(intensity_action))
            
            # Sample binary decisions (CMSIS-NN and temporal smoothing)
            binary_probs = policy_outputs["binary_decisions"].squeeze(0)
            
            # CMSIS-NN decision
            cmsis_prob = binary_probs[0].item()
            if deterministic:
                cmsis_action = 1 if cmsis_prob > 0.5 else 0
            else:
                cmsis_action = 1 if torch.rand(1).item() < cmsis_prob else 0
            actions.append(cmsis_action)
            
            # Calculate log probability for binary decision
            cmsis_log_prob = torch.log(torch.tensor(cmsis_prob if cmsis_action == 1 else (1 - cmsis_prob), device=self.device) + 1e-8)
            log_probs.append(cmsis_log_prob)
            
            # Temporal smoothing decision
            temporal_prob = binary_probs[1].item()
            if deterministic:
                temporal_action = 1 if temporal_prob > 0.5 else 0
            else:
                temporal_action = 1 if torch.rand(1).item() < temporal_prob else 0
            actions.append(temporal_action)
            
            # Calculate log probability for binary decision
            temporal_log_prob = torch.log(torch.tensor(temporal_prob if temporal_action == 1 else (1 - temporal_prob), device=self.device) + 1e-8)
            log_probs.append(temporal_log_prob)
            
            # Confidence threshold (convert to discrete action)
            confidence_value = policy_outputs["confidence_threshold"].item()
            # Map to discrete confidence levels: [0.6, 0.7, 0.8, 0.85, 0.9]
            confidence_levels = [0.6, 0.7, 0.8, 0.85, 0.9]
            confidence_action = min(range(len(confidence_levels)), key=lambda i: abs(confidence_levels[i] - confidence_value))
            actions.append(confidence_action)
            
            # For continuous values, we use a different approach for log prob
            # We'll approximate it as a uniform distribution over the discrete choices
            confidence_log_prob = torch.log(torch.tensor(1.0 / len(confidence_levels), device=self.device))
            log_probs.append(confidence_log_prob)
            
            # Combined log probability
            total_log_prob = sum(log_probs).item()
            
            value = policy_outputs["state_value"]
            
            return np.array(actions), total_log_prob, value.squeeze().item()
    
    def collect_rollout(self) -> Dict[str, Any]:
        """
        Collect a rollout of experience.
        
        Returns:
            Dictionary with rollout statistics
        """
        self.buffer.clear()
        
        episode_rewards = []
        episode_accuracies = []
        episode_energy_efficiencies = []
        episode_inference_times = []
        
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(self.hyperparams.rollout_steps):
            # Select action
            action, log_prob, value = self.select_action(obs, deterministic=False)
            
            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Store in buffer
            self.buffer.store(obs, action, log_prob, reward, value, done)
            
            # Update tracking
            episode_reward += reward
            episode_steps += 1
            self.total_env_steps += 1
            
            # Track metrics from info
            if 'result' in info:
                result = info['result']
                episode_accuracies.append(result['accuracy_achieved'])
                episode_energy_efficiencies.append(
                    max(0.0, 1.0 - result['energy_consumed_mj'] / 200.0)
                )
                episode_inference_times.append(result['inference_time_ms'])
            
            obs = next_obs
            
            if done:
                # Episode finished
                self.episode_rewards.append(episode_reward)
                if episode_accuracies:
                    self.episode_accuracies.append(np.mean(episode_accuracies))
                if episode_energy_efficiencies:
                    self.episode_energy_efficiency.append(np.mean(episode_energy_efficiencies))
                
                self.episode_count += 1
                
                # Reset for new episode
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_accuracies = []
                episode_energy_efficiencies = []
                episode_inference_times = []
        
        # Compute final value for GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            policy_outputs = self.policy(obs_tensor)
            last_value = policy_outputs["state_value"].squeeze().item()
        
        # Compute advantages and returns
        self.buffer.compute_gae(
            last_value=last_value,
            gamma=self.hyperparams.gamma,
            gae_lambda=self.hyperparams.gae_lambda
        )
        
        # Calculate rollout statistics
        rollout_stats = {
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'average_accuracy': np.mean(self.episode_accuracies) if self.episode_accuracies else 0.0,
            'average_energy_efficiency': np.mean(self.episode_energy_efficiency) if self.episode_energy_efficiency else 0.0,
            'total_steps': self.total_env_steps,
            'episodes_collected': len(self.episode_rewards)
        }
        
        # Return actual rollout data for testing/inspection
        rollout_data = {
            'observations': self.buffer.observations[:self.buffer.size].cpu().numpy(),
            'actions': self.buffer.actions[:self.buffer.size].cpu().numpy(),
            'rewards': self.buffer.rewards[:self.buffer.size].cpu().numpy(),
            'values': self.buffer.values[:self.buffer.size].cpu().numpy(),
            'log_probs': self.buffer.log_probs[:self.buffer.size].cpu().numpy(),
            'returns': self.buffer.returns[:self.buffer.size].cpu().numpy(),
            'advantages': self.buffer.advantages[:self.buffer.size].cpu().numpy(),
            'statistics': rollout_stats
        }

        return rollout_data
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary with training statistics
        """
        # Normalize advantages
        if self.hyperparams.advantage_normalization:
            advantages = self.buffer.advantages[:self.buffer.size]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.buffer.advantages[:self.buffer.size] = advantages
        
        # Normalize returns
        if self.hyperparams.reward_normalization:
            returns = self.buffer.returns[:self.buffer.size]
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            self.buffer.returns[:self.buffer.size] = returns
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        clipfracs = []
        approx_kls = []
        
        # PPO update epochs
        for epoch in range(self.hyperparams.ppo_epochs):
            batches = self.buffer.get_batches(self.hyperparams.batch_size)
            
            for batch in batches:
                # Get current policy outputs
                policy_outputs = self.policy(batch['observations'])
                values = policy_outputs["state_value"]
                
                # Calculate log probabilities and entropy for each action component
                log_probs = []
                entropies_components = []
                
                # Model variant (3 options)
                model_dist = Categorical(probs=policy_outputs["model_probs"])
                model_actions = batch['actions'][:, 0]
                log_probs.append(model_dist.log_prob(model_actions))
                entropies_components.append(model_dist.entropy())
                
                # Processing intensity (3 options)
                intensity_dist = Categorical(probs=policy_outputs["intensity_probs"])
                intensity_actions = batch['actions'][:, 1]
                log_probs.append(intensity_dist.log_prob(intensity_actions))
                entropies_components.append(intensity_dist.entropy())
                
                # Binary decisions (CMSIS-NN and temporal smoothing)
                binary_probs = policy_outputs["binary_decisions"]
                
                # CMSIS-NN log prob
                cmsis_actions = batch['actions'][:, 2]
                cmsis_probs = binary_probs[:, 0]
                cmsis_log_probs = torch.where(
                    cmsis_actions.bool(),
                    torch.log(cmsis_probs + 1e-8),
                    torch.log(1 - cmsis_probs + 1e-8)
                )
                log_probs.append(cmsis_log_probs)
                
                # CMSIS entropy approximation
                cmsis_entropy = -(cmsis_probs * torch.log(cmsis_probs + 1e-8) + 
                                  (1 - cmsis_probs) * torch.log(1 - cmsis_probs + 1e-8))
                entropies_components.append(cmsis_entropy)
                
                # Temporal smoothing log prob
                temporal_actions = batch['actions'][:, 3]
                temporal_probs = binary_probs[:, 1]
                temporal_log_probs = torch.where(
                    temporal_actions.bool(),
                    torch.log(temporal_probs + 1e-8),
                    torch.log(1 - temporal_probs + 1e-8)
                )
                log_probs.append(temporal_log_probs)
                
                # Temporal entropy approximation
                temporal_entropy = -(temporal_probs * torch.log(temporal_probs + 1e-8) + 
                                     (1 - temporal_probs) * torch.log(1 - temporal_probs + 1e-8))
                entropies_components.append(temporal_entropy)
                
                # Confidence threshold (treat as uniform over discrete choices)
                confidence_actions = batch['actions'][:, 4]
                confidence_log_probs = torch.log(torch.tensor(1.0 / 5.0, device=self.device).expand_as(confidence_actions).float())  # 5 confidence levels
                log_probs.append(confidence_log_probs)
                
                # Confidence entropy (uniform distribution)
                confidence_entropy = torch.log(torch.tensor(5.0, device=self.device)).expand(batch['actions'].shape[0])
                entropies_components.append(confidence_entropy)
                
                current_log_probs = sum(log_probs)
                entropy = sum(entropies_components).mean()
                
                # PPO policy loss
                ratio = torch.exp(current_log_probs - batch['log_probs'])
                
                surrogate1 = ratio * batch['advantages']
                surrogate2 = torch.clamp(
                    ratio, 
                    1.0 - self.hyperparams.clip_ratio, 
                    1.0 + self.hyperparams.clip_ratio
                ) * batch['advantages']
                
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch['returns'])
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.hyperparams.value_loss_coef * value_loss - 
                    self.entropy_coef * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.hyperparams.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Track statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                
                # Clip fraction
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.hyperparams.clip_ratio).float().mean().item()
                    clipfracs.append(clipfrac)
                    
                    # Approximate KL divergence
                    approx_kl = (current_log_probs - batch['log_probs']).mean().item()
                    approx_kls.append(approx_kl)
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Adaptive entropy coefficient
        if self.hyperparams.adaptive_entropy:
            target_entropy = 0.5  # Target entropy level
            current_entropy = np.mean(entropies)
            if current_entropy < target_entropy:
                self.entropy_coef = min(0.1, self.entropy_coef * 1.01)
            else:
                self.entropy_coef = max(0.001, self.entropy_coef * 0.99)
        
        # Calculate explained variance
        with torch.no_grad():
            y_true = self.buffer.returns[:self.buffer.size]
            y_pred = self.buffer.values[:self.buffer.size]
            var_y = torch.var(y_true)
            if var_y > 0:
                explained_var = max(0.0, (1 - torch.var(y_true - y_pred) / var_y).item())
            else:
                explained_var = 0.0
        
        self.training_step += 1
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clipfrac': np.mean(clipfracs),
            'approx_kl': np.mean(approx_kls),
            'explained_variance': explained_var,
            'lr': self.optimizer.param_groups[0]['lr'],
            'entropy_coef': self.entropy_coef
        }
    
    def train(self, total_timesteps: int, save_interval: int = 10) -> Dict[str, Any]:
        """
        Train the agent for specified timesteps.
        
        Args:
            total_timesteps: Total environment steps to train
            save_interval: Save model every N iterations
            
        Returns:
            List of training metrics
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        metrics_list = []
        iterations = total_timesteps // self.hyperparams.rollout_steps
        
        for iteration in range(iterations):
            iteration_start_time = time.time()
            
            # Collect rollout
            rollout_data = self.collect_rollout()
            rollout_stats = rollout_data['statistics']
            
            # Update policy
            update_stats = self.update_policy()
            
            # Create training metrics
            metrics = TrainingMetrics(
                episode=self.episode_count,
                total_steps=self.total_env_steps,
                average_reward=rollout_stats['average_reward'],
                average_accuracy=rollout_stats['average_accuracy'],
                average_energy_efficiency=rollout_stats['average_energy_efficiency'],
                policy_loss=update_stats['policy_loss'],
                value_loss=update_stats['value_loss'],
                entropy=update_stats['entropy'],
                explained_variance=update_stats['explained_variance'],
                clipfrac=update_stats['clipfrac'],
                lr=update_stats['lr']
            )
            
            metrics_list.append(metrics)
            self.training_metrics.append(metrics)
            
            # Logging
            iteration_time = time.time() - iteration_start_time
            
            logger.info(
                f"Iteration {iteration + 1}/{iterations} "
                f"(Steps: {self.total_env_steps}/{total_timesteps}) "
                f"- Reward: {metrics.average_reward:.3f} "
                f"- Accuracy: {metrics.average_accuracy:.3f} "
                f"- Energy Eff: {metrics.average_energy_efficiency:.3f} "
                f"- Time: {iteration_time:.1f}s"
            )
            
            # Save model checkpoint
            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_iter_{iteration + 1}")
        
        logger.info("Training completed")
        
        # Run final evaluation
        final_evaluation = self.evaluate(num_episodes=5, deterministic=True)
        
        # Return training summary as dictionary
        training_summary = {
            'total_timesteps': self.total_env_steps,
            'training_episodes': self.episode_count,
            'iterations_completed': len(metrics_list),
            'final_evaluation': final_evaluation,
            'training_metrics': metrics_list
        }
        
        return training_summary
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Use deterministic action selection
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        eval_rewards = []
        eval_accuracies = []
        eval_energy_efficiencies = []
        eval_inference_times = []
        eval_episode_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_accuracies = []
            episode_energy_effs = []
            episode_times = []
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.select_action(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if 'result' in info:
                    result = info['result']
                    episode_accuracies.append(result['accuracy_achieved'])
                    episode_energy_effs.append(
                        max(0.0, 1.0 - result['energy_consumed_mj'] / 200.0)
                    )
                    episode_times.append(result['inference_time_ms'])
            
            eval_rewards.append(episode_reward)
            eval_episode_lengths.append(episode_length)
            if episode_accuracies:
                eval_accuracies.append(np.mean(episode_accuracies))
            if episode_energy_effs:
                eval_energy_efficiencies.append(np.mean(episode_energy_effs))
            if episode_times:
                eval_inference_times.append(np.mean(episode_times))
        
        evaluation_results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_episode_length': np.mean(eval_episode_lengths),
            'average_accuracy': np.mean(eval_accuracies) if eval_accuracies else 0.0,
            'average_energy_efficiency': np.mean(eval_energy_efficiencies) if eval_energy_efficiencies else 0.0,
            'average_inference_time': np.mean(eval_inference_times) if eval_inference_times else 0.0,
            'success_rate': np.mean([r > 0 for r in eval_rewards])
        }
        
        logger.info(f"Evaluation results: {evaluation_results}")
        return evaluation_results
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.model_save_dir / f"{checkpoint_name}.pth"
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_env_steps': self.total_env_steps,
            'hyperparameters': self.hyperparams.__dict__,
            'training_metrics': [m.__dict__ for m in self.training_metrics[-10:]]  # Last 10 metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.total_env_steps = checkpoint['total_env_steps']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """Plot training progress."""
        if not self.training_metrics:
            logger.warning("No training metrics available for plotting")
            return
        
        metrics_data = {
            'rewards': [m.average_reward for m in self.training_metrics],
            'accuracies': [m.average_accuracy for m in self.training_metrics],
            'energy_efficiencies': [m.average_energy_efficiency for m in self.training_metrics],
            'policy_losses': [m.policy_loss for m in self.training_metrics],
            'value_losses': [m.value_loss for m in self.training_metrics],
            'entropies': [m.entropy for m in self.training_metrics]
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PPO Pizza Agent Training Progress')
        
        titles = ['Average Reward', 'Average Accuracy', 'Energy Efficiency',
                 'Policy Loss', 'Value Loss', 'Entropy']
        
        for i, (key, title) in enumerate(zip(metrics_data.keys(), titles)):
            row, col = i // 3, i % 3
            axes[row, col].plot(metrics_data[key])
            axes[row, col].set_title(title)
            axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Test the PPO agent
    logging.basicConfig(level=logging.INFO)
    
    # Create environment
    env = PizzaRLEnvironmentFactory.create_training_environment()
    
    # Create agent
    hyperparams = PPOHyperparameters(
        learning_rate=3e-4,
        rollout_steps=512,  # Smaller for testing
        ppo_epochs=4,
        batch_size=64
    )
    
    agent = PPOPizzaAgent(
        environment=env,
        hyperparams=hyperparams,
        device='cpu'
    )
    
    # Test action selection
    obs, _ = env.reset()
    action, log_prob, value = agent.select_action(obs)
    print(f"Test action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
    
    # Short training test
    print("Running short training test...")
    metrics = agent.train(total_timesteps=2048, save_interval=1)  # Small test
    
    # Evaluation test
    print("Running evaluation test...")
    eval_results = agent.evaluate(num_episodes=3)
    
    print("PPO agent test completed successfully!")
