#!/usr/bin/env python3
"""
PPO Agent for training adaptive pizza recognition policies.

This module implements a Proximal Policy Optimization agent specifically
designed for multi-objective optimization of pizza recognition systems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
import json
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("stable-baselines3 not available, using custom PPO implementation")

from .adaptive_policy import AdaptivePizzaRecognitionPolicy
from .environment import PizzaRLEnvironment

logger = logging.getLogger(__name__)


class PizzaTrainingCallback(BaseCallback):
    """Custom callback for monitoring pizza RL training."""
    
    def __init__(self, eval_freq: int = 1000, save_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_accuracies = deque(maxlen=100)
        self.episode_energy_costs = deque(maxlen=100)
        
    def _on_step(self) -> bool:
        """Called at each step of training."""
        # Log training metrics
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(info['r'])
            
            # Extract pizza-specific metrics if available
            if 'pizza_metrics' in self.locals.get('infos', [{}])[0]:
                metrics = self.locals['infos'][0]['pizza_metrics']
                self.episode_accuracies.append(metrics.get('mean_accuracy', 0))
                self.episode_energy_costs.append(metrics.get('total_energy_cost', 0))
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            self._log_metrics()
        
        # Periodic model saving
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        return True
    
    def _log_metrics(self):
        """Log training metrics."""
        if self.episode_rewards:
            logger.info(f"Step {self.n_calls}: "
                       f"Mean reward: {np.mean(self.episode_rewards):.2f}, "
                       f"Mean accuracy: {np.mean(self.episode_accuracies):.3f}, "
                       f"Mean energy cost: {np.mean(self.episode_energy_costs):.2f}")
    
    def _save_model(self):
        """Save current model checkpoint."""
        if hasattr(self.model, 'save'):
            save_path = f"pizza_rl_checkpoint_{self.n_calls}.zip"
            self.model.save(save_path)
            logger.info(f"Model checkpoint saved: {save_path}")


class CustomPPOAgent:
    """
    Custom PPO implementation for pizza recognition optimization.
    
    This implementation provides more control over the training process
    and handles multi-objective optimization specifically for pizza detection.
    """
    
    def __init__(
        self,
        policy: AdaptivePizzaRecognitionPolicy,
        env: PizzaRLEnvironment,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.policy = policy.to(device)
        self.env = env
        self.device = device
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training metrics
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'accuracies': [],
            'energy_costs': [],
            'policy_losses': [],
            'value_losses': []
        }
        
        logger.info(f"Custom PPO agent initialized on {device}")
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards [num_steps]
            values: Tensor of value estimates [num_steps]
            dones: Tensor of done flags [num_steps]
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            states: State observations
            actions: Actions taken
            old_log_probs: Log probabilities from old policy
            advantages: Computed advantages
            returns: Computed returns
            old_values: Value estimates from old policy
            
        Returns:
            Dictionary of training metrics
        """
        # Forward pass through current policy
        action_logits, values = self.policy(states)
        
        # Compute action log probabilities
        # Note: This is simplified - actual implementation needs to handle
        # the complex action space with mixed discrete/continuous actions
        log_probs = torch.distributions.Normal(action_logits, 1.0).log_prob(actions).sum(dim=-1)
        
        # Policy loss (PPO clip objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy = torch.distributions.Normal(action_logits, 1.0).entropy().sum(dim=-1).mean()
        entropy_loss = -self.ent_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def collect_experience(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """
        Collect experience by running the policy in the environment.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary of collected experience
        """
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state = self.env.reset()
        
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, value = self.policy(state_tensor)
                # Simplified action sampling
                action = torch.randn_like(action_logits)
                log_prob = torch.distributions.Normal(action_logits, 1.0).log_prob(action).sum()
            
            # Convert action to numpy for environment
            action_np = action.squeeze().cpu().numpy()
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action_np)
            
            # Store experience
            states.append(state)
            actions.append(action.squeeze().cpu())
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze().cpu())
            log_probs.append(log_prob.cpu())
            
            state = next_state
            
            if done:
                state = self.env.reset()
        
        return {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.stack(actions).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'dones': torch.FloatTensor(dones).to(self.device),
            'values': torch.stack(values).to(self.device),
            'log_probs': torch.stack(log_probs).to(self.device)
        }
    
    def train(
        self,
        num_episodes: int,
        steps_per_episode: int = 100,
        ppo_epochs: int = 4,
        batch_size: int = 64
    ):
        """
        Train the PPO agent.
        
        Args:
            num_episodes: Number of training episodes
            steps_per_episode: Steps to collect per episode
            ppo_epochs: Number of PPO update epochs per collection
            batch_size: Batch size for training
        """
        logger.info(f"Starting PPO training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Collect experience
            experience = self.collect_experience(steps_per_episode)
            
            # Compute advantages
            advantages, returns = self.compute_advantages(
                experience['rewards'],
                experience['values'],
                experience['dones']
            )
            
            # PPO updates
            dataset_size = len(experience['states'])
            epoch_losses = {'policy_loss': [], 'value_loss': []}
            
            for epoch in range(ppo_epochs):
                # Shuffle data
                indices = torch.randperm(dataset_size)
                
                for start in range(0, dataset_size, batch_size):
                    end = min(start + batch_size, dataset_size)
                    batch_indices = indices[start:end]
                    
                    # Get batch
                    batch_states = experience['states'][batch_indices]
                    batch_actions = experience['actions'][batch_indices]
                    batch_old_log_probs = experience['log_probs'][batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_old_values = experience['values'][batch_indices]
                    
                    # Training step
                    step_metrics = self.train_step(
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_advantages,
                        batch_returns,
                        batch_old_values
                    )
                    
                    epoch_losses['policy_loss'].append(step_metrics['policy_loss'])
                    epoch_losses['value_loss'].append(step_metrics['value_loss'])
            
            # Record episode metrics
            episode_metrics = self.env.get_episode_metrics()
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_metrics.get('total_reward', 0))
            self.training_history['accuracies'].append(episode_metrics.get('mean_accuracy', 0))
            self.training_history['energy_costs'].append(episode_metrics.get('total_energy_cost', 0))
            self.training_history['policy_losses'].append(np.mean(epoch_losses['policy_loss']))
            self.training_history['value_losses'].append(np.mean(epoch_losses['value_loss']))
            
            # Logging
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"Reward: {episode_metrics.get('total_reward', 0):.2f}, "
                    f"Accuracy: {episode_metrics.get('mean_accuracy', 0):.3f}, "
                    f"Energy: {episode_metrics.get('total_energy_cost', 0):.2f}mJ"
                )
    
    def save(self, path: str):
        """Save the trained agent."""
        save_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }
        torch.save(save_data, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load a trained agent."""
        save_data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(save_data['policy_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        self.training_history = save_data['training_history']
        logger.info(f"Agent loaded from {path}")


class PPOPizzaAgent:
    """
    Pizza RL agent using either stable-baselines3 or custom PPO implementation.
    
    This class provides a unified interface for training adaptive pizza recognition
    policies using reinforcement learning.
    """
    
    def __init__(
        self,
        env: PizzaRLEnvironment,
        use_sb3: bool = True,
        device: str = 'cpu',
        **kwargs
    ):
        self.env = env
        self.device = device
        self.use_sb3 = use_sb3 and SB3_AVAILABLE
        
        if self.use_sb3:
            self._init_sb3_agent(**kwargs)
        else:
            self._init_custom_agent(**kwargs)
    
    def _init_sb3_agent(self, **kwargs):
        """Initialize stable-baselines3 PPO agent."""
        # Wrap environment for stable-baselines3
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO agent
        self.agent = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=kwargs.get('learning_rate', 3e-4),
            gamma=kwargs.get('gamma', 0.99),
            gae_lambda=kwargs.get('gae_lambda', 0.95),
            clip_range=kwargs.get('clip_range', 0.2),
            ent_coef=kwargs.get('ent_coef', 0.01),
            vf_coef=kwargs.get('vf_coef', 0.5),
            max_grad_norm=kwargs.get('max_grad_norm', 0.5),
            device=self.device,
            verbose=1
        )
        
        logger.info("Stable-baselines3 PPO agent initialized")
    
    def _init_custom_agent(self, **kwargs):
        """Initialize custom PPO agent."""
        policy = AdaptivePizzaRecognitionPolicy()
        self.agent = CustomPPOAgent(
            policy=policy,
            env=self.env,
            device=self.device,
            **kwargs
        )
        
        logger.info("Custom PPO agent initialized")
    
    def train(self, total_timesteps: int, **kwargs):
        """Train the agent."""
        if self.use_sb3:
            callback = PizzaTrainingCallback()
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                **kwargs
            )
        else:
            num_episodes = total_timesteps // kwargs.get('steps_per_episode', 100)
            self.agent.train(num_episodes=num_episodes, **kwargs)
    
    def predict(self, observation, deterministic: bool = True):
        """Make a prediction using the trained agent."""
        if self.use_sb3:
            return self.agent.predict(observation, deterministic=deterministic)
        else:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.agent.policy.sample_action(obs_tensor)
            return action, None
    
    def save(self, path: str):
        """Save the trained agent."""
        if self.use_sb3:
            self.agent.save(path)
        else:
            self.agent.save(path)
    
    def load(self, path: str):
        """Load a trained agent."""
        if self.use_sb3:
            self.agent = PPO.load(path)
        else:
            self.agent.load(path)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        if hasattr(self.agent, 'training_history'):
            return self.agent.training_history
        else:
            return {}
