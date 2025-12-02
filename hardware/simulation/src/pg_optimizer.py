"""
Policy Gradient Optimizer
Implementation of Actor-Critic method for hardware parameter optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class PGMetrics:
    """Metrics tracking for Policy Gradient optimization"""
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    contribution: float = 0.0  # Contribution to overall optimization
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    episodes_completed: int = 0

class PolicyGradientOptimizer:
    """
    Policy Gradient optimizer using Actor-Critic method
    Optimizes differentiable hardware control parameters
    """
    
    def __init__(self, config, state_dim: int, action_dim: int):
        self.config = config.pg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger(__name__)
        
        # Network architecture
        self.hidden_sizes = self.config.hidden_sizes
        self.activation = self.config.activation
        
        # Training parameters
        self.learning_rate = self.config.learning_rate
        self.discount_factor = self.config.discount_factor
        self.entropy_coeff = self.config.entropy_coeff
        self.value_loss_coeff = self.config.value_loss_coeff
        self.max_grad_norm = self.config.max_grad_norm
        
        # Experience buffer
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size
        self.experience_buffer = []
        
        # Networks (simplified implementation - would use PyTorch/TensorFlow in practice)
        self.policy_network = self._create_policy_network()
        self.value_network = self._create_value_network()
        
        # Optimization state
        self.episode_count = 0
        self.step_count = 0
        self.metrics = PGMetrics()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        
        self.logger.info(f"Policy Gradient optimizer initialized - state_dim={state_dim}, action_dim={action_dim}")
    
    def _create_policy_network(self) -> Dict:
        """Create policy network (Actor) - simplified implementation"""
        # In real implementation, this would create PyTorch/TensorFlow networks
        network = {
            'type': 'policy',
            'input_dim': self.state_dim,
            'hidden_dims': self.hidden_sizes,
            'output_dim': self.action_dim,
            'activation': self.activation,
            'weights': self._initialize_weights('policy'),
            'optimizer': 'adam'
        }
        return network
    
    def _create_value_network(self) -> Dict:
        """Create value network (Critic) - simplified implementation"""
        network = {
            'type': 'value',
            'input_dim': self.state_dim,
            'hidden_dims': self.hidden_sizes,
            'output_dim': 1,
            'activation': self.activation,
            'weights': self._initialize_weights('value'),
            'optimizer': 'adam'
        }
        return network
    
    def _initialize_weights(self, network_type: str) -> Dict:
        """Initialize network weights - simplified implementation"""
        np.random.seed(42)  # For reproducibility
        
        weights = {}
        if network_type == 'policy':
            # Policy network weights
            weights['layer1'] = np.random.normal(0, 0.1, (self.state_dim, self.hidden_sizes[0]))
            weights['layer2'] = np.random.normal(0, 0.1, (self.hidden_sizes[0], self.hidden_sizes[1]))
            weights['output'] = np.random.normal(0, 0.1, (self.hidden_sizes[1], self.action_dim))
            
            # Biases
            weights['bias1'] = np.zeros(self.hidden_sizes[0])
            weights['bias2'] = np.zeros(self.hidden_sizes[1])
            weights['output_bias'] = np.zeros(self.action_dim)
            
        elif network_type == 'value':
            # Value network weights
            weights['layer1'] = np.random.normal(0, 0.1, (self.state_dim, self.hidden_sizes[0]))
            weights['layer2'] = np.random.normal(0, 0.1, (self.hidden_sizes[0], self.hidden_sizes[1]))
            weights['output'] = np.random.normal(0, 0.1, (self.hidden_sizes[1], 1))
            
            # Biases
            weights['bias1'] = np.zeros(self.hidden_sizes[0])
            weights['bias2'] = np.zeros(self.hidden_sizes[1])
            weights['output_bias'] = np.zeros(1)
        
        return weights
    
    def _forward_policy(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through policy network"""
        weights = self.policy_network['weights']
        
        # Layer 1
        x = np.dot(state, weights['layer1']) + weights['bias1']
        x = np.tanh(x)  # Activation
        
        # Layer 2
        x = np.dot(x, weights['layer2']) + weights['bias2']
        x = np.tanh(x)
        
        # Output layer
        logits = np.dot(x, weights['output']) + weights['output_bias']
        
        # For continuous control, output mean and std
        mean = np.tanh(logits)  # Bounded actions
        log_std = np.zeros_like(mean) - 1.0  # Fixed log std for simplicity
        std = np.exp(log_std)
        
        return mean, std
    
    def _forward_value(self, state: np.ndarray) -> float:
        """Forward pass through value network"""
        weights = self.value_network['weights']
        
        # Layer 1
        x = np.dot(state, weights['layer1']) + weights['bias1']
        x = np.tanh(x)
        
        # Layer 2
        x = np.dot(x, weights['layer2']) + weights['bias2']
        x = np.tanh(x)
        
        # Output
        value = np.dot(x, weights['output']) + weights['output_bias']
        return float(value[0])
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Select action using current policy
        
        Returns:
            action: Selected action
            info: Additional information (log_prob, value, etc.)
        """
        
        # Get policy distribution
        mean, std = self._forward_policy(state)
        
        # Sample action from Gaussian distribution
        action = np.random.normal(mean, std)
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Calculate log probability (simplified)
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2) - 0.5 * np.sum(np.log(2 * np.pi * std ** 2))
        
        # Get state value
        value = self._forward_value(state)
        
        info = {
            'log_prob': log_prob,
            'value': value,
            'entropy': 0.5 * np.sum(np.log(2 * np.pi * np.e * std ** 2))  # Gaussian entropy
        }
        
        return action, info
    
    def store_experience(self, 
                        state: np.ndarray, 
                        action: np.ndarray, 
                        reward: float, 
                        next_state: np.ndarray, 
                        done: bool, 
                        info: Dict):
        """Store experience in replay buffer"""
        
        experience = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'log_prob': info.get('log_prob', 0.0),
            'value': info.get('value', 0.0),
            'entropy': info.get('entropy', 0.0)
        }
        
        self.experience_buffer.append(experience)
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def update_policy(self) -> PGMetrics:
        """
        Update policy using stored experiences
        Implements Actor-Critic with advantage estimation
        """
        
        if len(self.experience_buffer) < self.batch_size:
            return self.metrics
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), 
                                       size=min(self.batch_size, len(self.experience_buffer)), 
                                       replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Extract batch data
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        log_probs = np.array([exp['log_prob'] for exp in batch])
        values = np.array([exp['value'] for exp in batch])
        
        # Calculate advantages using GAE (simplified)
        advantages = self._calculate_advantages(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Calculate losses
        policy_loss = -np.mean(log_probs * advantages)
        value_loss = np.mean((returns - values) ** 2)
        entropy_loss = -np.mean([exp['entropy'] for exp in batch])
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coeff * value_loss + 
                     self.entropy_coeff * entropy_loss)
        
        # Simplified gradient update (in practice, use automatic differentiation)
        self._update_networks(total_loss, policy_loss, value_loss)
        
        # Update metrics
        self.metrics.loss = total_loss
        self.metrics.policy_loss = policy_loss
        self.metrics.value_loss = value_loss
        self.metrics.entropy_loss = entropy_loss
        self.metrics.advantage_mean = np.mean(advantages)
        self.metrics.advantage_std = np.std(advantages)
        self.metrics.learning_rate = self.learning_rate
        self.metrics.gradient_norm = self._calculate_gradient_norm()
        
        # Track loss history
        self.loss_history.append(total_loss)
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
        
        # Calculate contribution (simplified metric)
        self.metrics.contribution = abs(np.mean(advantages))
        
        self.logger.debug(f"PG Update - Loss: {total_loss:.6f}, Policy: {policy_loss:.6f}, Value: {value_loss:.6f}")
        
        return self.metrics
    
    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Calculate advantages using TD error (simplified GAE)"""
        
        advantages = np.zeros_like(rewards)
        
        for t in range(len(rewards)):
            if t == len(rewards) - 1 or dones[t]:
                # Terminal state
                advantages[t] = rewards[t] - values[t]
            else:
                # TD error
                td_error = rewards[t] + self.discount_factor * values[t + 1] - values[t]
                advantages[t] = td_error
        
        return advantages
    
    def _update_networks(self, total_loss: float, policy_loss: float, value_loss: float):
        """Update network weights (simplified implementation)"""
        
        # In real implementation, this would use PyTorch/TensorFlow optimizers
        # Here we just apply a simple gradient descent approximation
        
        update_magnitude = self.learning_rate * 0.01  # Simplified
        
        # Update policy network weights (simplified)
        policy_weights = self.policy_network['weights']
        for key in policy_weights:
            # Add small random perturbation as gradient approximation
            gradient_approx = np.random.normal(0, update_magnitude, policy_weights[key].shape)
            if policy_loss > 0:  # Gradient descent
                policy_weights[key] -= gradient_approx * policy_loss
            
        # Update value network weights
        value_weights = self.value_network['weights']
        for key in value_weights:
            gradient_approx = np.random.normal(0, update_magnitude, value_weights[key].shape)
            if value_loss > 0:
                value_weights[key] -= gradient_approx * value_loss
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate gradient norm for monitoring"""
        # Simplified implementation
        total_norm = 0.0
        
        for weights in [self.policy_network['weights'], self.value_network['weights']]:
            for weight_matrix in weights.values():
                total_norm += np.sum(weight_matrix ** 2)
        
        return np.sqrt(total_norm)
    
    def episode_end(self, episode_reward: float, episode_length: int):
        """Called at end of episode for tracking"""
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.metrics.episodes_completed = self.episode_count
        
        # Keep history manageable
        if len(self.episode_rewards) > 1000:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
        
        self.logger.debug(f"Episode {self.episode_count} completed - Reward: {episode_reward:.3f}, Length: {episode_length}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        
        if len(self.episode_rewards) == 0:
            return {"episodes": 0}
        
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards[-100:]),  # Last 100 episodes
            "std_reward": np.std(self.episode_rewards[-100:]),
            "mean_episode_length": np.mean(self.episode_lengths[-100:]),
            "recent_loss": np.mean(self.loss_history[-10:]) if self.loss_history else 0.0,
            "loss_trend": np.polyfit(range(len(self.loss_history[-50:])), self.loss_history[-50:], 1)[0] if len(self.loss_history) >= 50 else 0.0,
            "gradient_norm": self.metrics.gradient_norm,
            "learning_rate": self.learning_rate
        }
    
    def save_checkpoint(self, filepath: str):
        """Save optimizer state"""
        checkpoint = {
            'policy_weights': self.policy_network['weights'],
            'value_weights': self.value_network['weights'],
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'metrics': self.metrics.__dict__,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }
        
        # In real implementation, use pickle or torch.save
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load optimizer state"""
        # In real implementation, load from pickle or torch.load
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def adapt_learning_rate(self, performance_trend: float):
        """Adapt learning rate based on performance"""
        if performance_trend < -0.1:  # Performance degrading
            self.learning_rate *= 0.9  # Reduce learning rate
        elif performance_trend > 0.1:  # Performance improving
            self.learning_rate *= 1.05  # Slightly increase learning rate
        
        # Clamp learning rate
        self.learning_rate = np.clip(self.learning_rate, 1e-6, 1e-2)
        self.metrics.learning_rate = self.learning_rate
