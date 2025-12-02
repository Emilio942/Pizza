#!/usr/bin/env python3
"""
Adaptive Pizza Recognition Policy for RL-based inference optimization.

This module implements an adaptive policy that selects optimal inference strategies
based on energy constraints, image complexity, and quality requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..constants import CLASS_NAMES
from ..pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE


class ModelVariant(Enum):
    """Available pizza detection model variants."""
    MICRO_PIZZA_NET = "MicroPizzaNet"
    MICRO_PIZZA_NET_V2 = "MicroPizzaNetV2" 
    MICRO_PIZZA_NET_SE = "MicroPizzaNetWithSE"


class ProcessingIntensity(Enum):
    """Processing intensity levels for adaptive inference."""
    LOW = 0.5      # Reduced image resolution, basic preprocessing
    MEDIUM = 0.75  # Standard resolution, standard preprocessing
    HIGH = 1.0     # Full resolution, enhanced preprocessing


@dataclass
class InferenceStrategy:
    """Represents a complete inference strategy configuration."""
    model_variant: ModelVariant
    processing_intensity: ProcessingIntensity
    confidence_threshold: float
    use_cmsis_nn: bool
    enable_temporal_smoothing: bool


@dataclass
class SystemState:
    """Current system state for policy decision making."""
    battery_level: float  # 0.0 to 1.0
    image_complexity: float  # Estimated complexity metric
    required_accuracy: float  # Minimum required accuracy
    time_constraints: float  # Available processing time (ms)
    temperature: float  # System temperature
    memory_usage: float  # Current memory usage ratio


class AdaptivePizzaRecognitionPolicy(nn.Module):
    """
    Neural network policy for adaptive pizza recognition strategy selection.
    
    Maps system state to optimal inference configuration for energy-efficient
    pizza detection while maintaining required quality levels.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 128,
        num_model_variants: int = 3,
        num_intensity_levels: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Policy network layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Output heads for different strategy components
        self.model_head = nn.Linear(hidden_dim, num_model_variants)
        self.intensity_head = nn.Linear(hidden_dim, num_intensity_levels)
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Output confidence threshold
        self.cmsis_head = nn.Linear(hidden_dim, 1)  # Binary: use CMSIS-NN or not
        self.temporal_head = nn.Linear(hidden_dim, 1)  # Binary: use temporal smoothing
        
        # Value head for critic (PPO)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: System state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Model selection logits
        model_logits = self.model_head(features)
        
        # Processing intensity logits
        intensity_logits = self.intensity_head(features)
        
        # Confidence threshold (sigmoid to keep in [0,1])
        confidence_logits = torch.sigmoid(self.confidence_head(features))
        
        # CMSIS-NN usage (sigmoid for binary decision)
        cmsis_logits = torch.sigmoid(self.cmsis_head(features))
        
        # Temporal smoothing (sigmoid for binary decision)
        temporal_logits = torch.sigmoid(self.temporal_head(features))
        
        # Combine all action components
        action_logits = torch.cat([
            model_logits,
            intensity_logits, 
            confidence_logits,
            cmsis_logits,
            temporal_logits
        ], dim=-1)
        
        # State value estimate
        value = self.value_head(features)
        
        return action_logits, value
    
    def get_action_distribution(self, state: torch.Tensor):
        """Get action distribution for sampling."""
        action_logits, value = self.forward(state)
        
        # Split action logits
        model_logits = action_logits[:, :3]  # 3 model variants
        intensity_logits = action_logits[:, 3:6]  # 3 intensity levels
        confidence_logits = action_logits[:, 6:7]  # 1 confidence value
        cmsis_logits = action_logits[:, 7:8]  # 1 binary value
        temporal_logits = action_logits[:, 8:9]  # 1 binary value
        
        # Create categorical distributions for discrete choices
        model_dist = torch.distributions.Categorical(logits=model_logits)
        intensity_dist = torch.distributions.Categorical(logits=intensity_logits)
        
        # Beta distributions for continuous values in [0,1]
        confidence_dist = torch.distributions.Beta(
            confidence_logits + 1e-8, 
            1 - confidence_logits + 1e-8
        )
        
        # Bernoulli distributions for binary choices
        cmsis_dist = torch.distributions.Bernoulli(logits=cmsis_logits)
        temporal_dist = torch.distributions.Bernoulli(logits=temporal_logits)
        
        return {
            'model': model_dist,
            'intensity': intensity_dist,
            'confidence': confidence_dist,
            'cmsis': cmsis_dist,
            'temporal': temporal_dist,
            'value': value
        }
    
    def sample_action(self, state: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Sample action from policy."""
        distributions = self.get_action_distribution(state)
        
        action = {
            'model': distributions['model'].sample(),
            'intensity': distributions['intensity'].sample(),
            'confidence': distributions['confidence'].sample(),
            'cmsis': distributions['cmsis'].sample(),
            'temporal': distributions['temporal'].sample()
        }
        
        return action, distributions['value']
    
    def get_strategy(self, state: SystemState, device: str = 'cpu') -> InferenceStrategy:
        """
        Convert system state to concrete inference strategy.
        
        Args:
            state: Current system state
            device: Device for computation
            
        Returns:
            Selected inference strategy
        """
        # Convert state to tensor
        state_tensor = torch.tensor([
            state.battery_level,
            state.image_complexity,
            state.required_accuracy,
            state.time_constraints / 1000.0,  # Normalize to seconds
            state.temperature / 100.0,  # Normalize temperature
            state.memory_usage
        ], dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.sample_action(state_tensor)
        
        # Convert action indices to concrete strategy
        model_variants = list(ModelVariant)
        intensity_levels = list(ProcessingIntensity)
        
        strategy = InferenceStrategy(
            model_variant=model_variants[action['model'].item()],
            processing_intensity=intensity_levels[action['intensity'].item()],
            confidence_threshold=action['confidence'].item(),
            use_cmsis_nn=bool(action['cmsis'].item()),
            enable_temporal_smoothing=bool(action['temporal'].item())
        )
        
        return strategy
    
    def estimate_energy_cost(self, strategy: InferenceStrategy, image_size: Tuple[int, int]) -> float:
        """
        Estimate energy cost for a given inference strategy.
        
        Args:
            strategy: Inference strategy configuration
            image_size: Input image dimensions
            
        Returns:
            Estimated energy cost in mJ (millijoules)
        """
        base_cost = 10.0  # Base energy cost in mJ
        
        # Model complexity factor
        model_factors = {
            ModelVariant.MICRO_PIZZA_NET: 1.0,
            ModelVariant.MICRO_PIZZA_NET_V2: 1.2, 
            ModelVariant.MICRO_PIZZA_NET_SE: 1.5
        }
        
        # Processing intensity factor
        intensity_factors = {
            ProcessingIntensity.LOW: 0.5,
            ProcessingIntensity.MEDIUM: 0.75,
            ProcessingIntensity.HIGH: 1.0
        }
        
        cost = base_cost * model_factors[strategy.model_variant]
        cost *= intensity_factors[strategy.processing_intensity]
        
        # CMSIS-NN reduces energy cost
        if strategy.use_cmsis_nn:
            cost *= 0.7
        
        # Temporal smoothing adds minimal cost
        if strategy.enable_temporal_smoothing:
            cost *= 1.05
        
        # Scale by image size
        pixel_count = image_size[0] * image_size[1]
        cost *= (pixel_count / (48 * 48))  # Normalize to base 48x48
        
        return cost
    
    def estimate_accuracy(self, strategy: InferenceStrategy, image_complexity: float) -> float:
        """
        Estimate accuracy for a given inference strategy.
        
        Args:
            strategy: Inference strategy configuration
            image_complexity: Estimated image complexity [0,1]
            
        Returns:
            Estimated accuracy [0,1]
        """
        # Base accuracy for each model variant
        base_accuracies = {
            ModelVariant.MICRO_PIZZA_NET: 0.85,
            ModelVariant.MICRO_PIZZA_NET_V2: 0.88,
            ModelVariant.MICRO_PIZZA_NET_SE: 0.91
        }
        
        accuracy = base_accuracies[strategy.model_variant]
        
        # Processing intensity affects accuracy
        intensity_factors = {
            ProcessingIntensity.LOW: 0.9,
            ProcessingIntensity.MEDIUM: 0.95,
            ProcessingIntensity.HIGH: 1.0
        }
        
        accuracy *= intensity_factors[strategy.processing_intensity]
        
        # Complex images reduce accuracy
        complexity_penalty = image_complexity * 0.1
        accuracy *= (1.0 - complexity_penalty)
        
        # Temporal smoothing improves accuracy
        if strategy.enable_temporal_smoothing:
            accuracy *= 1.03
        
        return min(accuracy, 1.0)
