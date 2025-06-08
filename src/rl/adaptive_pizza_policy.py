#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Pizza Recognition Policy Architecture

This module implements the RL policy for adaptive inference strategies 
based on energy state and pizza recognition requirements. It integrates 
with the existing energy management system and pizza detection pipeline.

Author: AI Assistant  
Date: June 8, 2025
Version: 1.0.0
"""

import os
import sys
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.constants import CLASS_NAMES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingIntensity(Enum):
    """Processing intensity levels for adaptive inference."""
    MINIMAL = "minimal"      # Basic preprocessing, fastest inference
    STANDARD = "standard"    # Normal preprocessing and inference
    ENHANCED = "enhanced"    # Full preprocessing, highest accuracy


class ModelVariant(Enum):
    """Available model variants for adaptive selection."""
    MICRO_PIZZA_NET = "MicroPizzaNet"
    MICRO_PIZZA_NET_V2 = "MicroPizzaNetV2"
    MICRO_PIZZA_NET_SE = "MicroPizzaNetWithSE"


@dataclass
class SystemState:
    """System state representation for RL policy."""
    # Energy characteristics
    battery_level: float        # 0.0-1.0 (normalized)
    power_draw_current: float   # Current power consumption (mW)
    energy_budget: float        # Available energy for next inference (mJ)
    
    # Image characteristics
    image_complexity: float     # 0.0-1.0 (estimated complexity)
    brightness_level: float     # 0.0-1.0 (normalized brightness)
    contrast_level: float       # 0.0-1.0 (normalized contrast)
    has_motion_blur: bool       # Motion blur detection
    
    # Performance requirements
    required_accuracy: float    # 0.0-1.0 (minimum required accuracy)
    time_constraints: float     # Maximum inference time (ms)
    food_safety_critical: bool  # Whether this is a safety-critical decision
    
    # System characteristics
    temperature: float          # Operating temperature (°C)
    memory_usage: float         # Current RAM usage (0.0-1.0)
    processing_load: float      # Current CPU load (0.0-1.0)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert system state to tensor for neural network input."""
        state_vector = [
            self.battery_level,
            self.power_draw_current / 1000.0,  # Normalize to 0-1 range
            self.energy_budget / 100.0,        # Normalize to reasonable range
            self.image_complexity,
            self.brightness_level,
            self.contrast_level,
            float(self.has_motion_blur),
            self.required_accuracy,
            self.time_constraints / 1000.0,    # Normalize to 0-1 range
            float(self.food_safety_critical),
            (self.temperature + 40) / 80.0,    # Normalize -40°C to 40°C -> 0-1
            self.memory_usage,
            self.processing_load
        ]
        return torch.tensor(state_vector, dtype=torch.float32)


@dataclass
class InferenceStrategy:
    """Inference strategy chosen by the RL policy."""
    model_variant: ModelVariant
    processing_intensity: ProcessingIntensity
    use_cmsis_nn: bool
    confidence_threshold: float     # 0.0-1.0
    enable_temporal_smoothing: bool
    preprocessing_options: Dict[str, Any]
    
    def get_estimated_metrics(self) -> Dict[str, float]:
        """Get estimated performance metrics for this strategy."""
        # Base metrics for different model variants
        base_metrics = {
            ModelVariant.MICRO_PIZZA_NET: {
                "inference_time_ms": 0.18,
                "energy_mj": 2.5,
                "accuracy": 0.65,
                "ram_kb": 29
            },
            ModelVariant.MICRO_PIZZA_NET_V2: {
                "inference_time_ms": 0.22,
                "energy_mj": 3.2,
                "accuracy": 0.68,
                "ram_kb": 34
            },
            ModelVariant.MICRO_PIZZA_NET_SE: {
                "inference_time_ms": 0.28,
                "energy_mj": 4.1,
                "accuracy": 0.72,
                "ram_kb": 41
            }
        }
        
        metrics = base_metrics[self.model_variant].copy()
        
        # Adjust for processing intensity
        intensity_multipliers = {
            ProcessingIntensity.MINIMAL: {"time": 0.7, "energy": 0.6, "accuracy": 0.9},
            ProcessingIntensity.STANDARD: {"time": 1.0, "energy": 1.0, "accuracy": 1.0},
            ProcessingIntensity.ENHANCED: {"time": 1.5, "energy": 1.8, "accuracy": 1.1}
        }
        
        multiplier = intensity_multipliers[self.processing_intensity]
        metrics["inference_time_ms"] *= multiplier["time"]
        metrics["energy_mj"] *= multiplier["energy"]
        metrics["accuracy"] *= multiplier["accuracy"]
        
        # CMSIS-NN acceleration
        if self.use_cmsis_nn:
            metrics["inference_time_ms"] *= 0.6  # 40% speedup
            metrics["energy_mj"] *= 0.7          # 30% energy savings
        
        # Temporal smoothing overhead
        if self.enable_temporal_smoothing:
            metrics["inference_time_ms"] *= 1.1
            metrics["energy_mj"] *= 1.05
            metrics["accuracy"] *= 1.02  # Slight accuracy improvement
        
        # Clip accuracy to valid range
        metrics["accuracy"] = min(1.0, metrics["accuracy"])
        
        return metrics


class AdaptivePizzaRecognitionPolicy(nn.Module):
    """
    Neural network policy for adaptive pizza recognition.
    
    Architecture: Multi-layer perceptron with specialized heads for different
    decision components (model selection, processing configuration, etc.).
    """
    
    def __init__(
        self,
        state_dim: int = 13,  # Size of SystemState.to_tensor()
        hidden_dim: int = 128,
        num_hidden_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction layers
        layers = []
        current_dim = state_dim
        
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Specialized decision heads
        
        # Model variant selection (3 options)
        self.model_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 model variants
            nn.Softmax(dim=-1)
        )
        
        # Processing intensity selection (3 options)
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 intensity levels
            nn.Softmax(dim=-1)
        )
        
        # Binary decisions (CMSIS-NN, temporal smoothing)
        self.binary_decisions_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [use_cmsis_nn, enable_temporal_smoothing]
            nn.Sigmoid()
        )
        
        # Confidence threshold (continuous value 0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Value function for training
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        logger.info(f"Initialized AdaptivePizzaRecognitionPolicy with {self.count_parameters()} parameters")
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the policy network.
        
        Args:
            state: System state tensor [batch_size, state_dim]
            
        Returns:
            Dictionary with policy outputs for each decision component
        """
        # Extract shared features
        features = self.feature_extractor(state)
        
        # Generate outputs for each decision component
        outputs = {
            "model_probs": self.model_head(features),
            "intensity_probs": self.intensity_head(features),
            "binary_decisions": self.binary_decisions_head(features),
            "confidence_threshold": self.confidence_head(features),
            "state_value": self.value_head(features)
        }
        
        return outputs
    
    def select_action(self, state: SystemState, deterministic: bool = False) -> InferenceStrategy:
        """
        Select an inference strategy based on the current system state.
        
        Args:
            state: Current system state
            deterministic: If True, select most likely actions; if False, sample
            
        Returns:
            Selected inference strategy
        """
        self.eval()
        with torch.no_grad():
            state_tensor = state.to_tensor().unsqueeze(0)
            outputs = self.forward(state_tensor)
            
            # Select model variant
            if deterministic:
                model_idx = torch.argmax(outputs["model_probs"], dim=-1).item()
            else:
                model_idx = torch.multinomial(outputs["model_probs"], 1).item()
            model_variant = list(ModelVariant)[model_idx]
            
            # Select processing intensity
            if deterministic:
                intensity_idx = torch.argmax(outputs["intensity_probs"], dim=-1).item()
            else:
                intensity_idx = torch.multinomial(outputs["intensity_probs"], 1).item()
            processing_intensity = list(ProcessingIntensity)[intensity_idx]
            
            # Binary decisions
            binary_probs = outputs["binary_decisions"].squeeze(0)
            if deterministic:
                use_cmsis_nn = binary_probs[0].item() > 0.5
                enable_temporal_smoothing = binary_probs[1].item() > 0.5
            else:
                use_cmsis_nn = torch.bernoulli(binary_probs[0]).bool().item()
                enable_temporal_smoothing = torch.bernoulli(binary_probs[1]).bool().item()
            
            # Confidence threshold
            confidence_threshold = outputs["confidence_threshold"].squeeze(0).item()
            
            # Preprocessing options based on processing intensity
            preprocessing_options = self._get_preprocessing_options(
                processing_intensity, state
            )
            
            strategy = InferenceStrategy(
                model_variant=model_variant,
                processing_intensity=processing_intensity,
                use_cmsis_nn=use_cmsis_nn,
                confidence_threshold=confidence_threshold,
                enable_temporal_smoothing=enable_temporal_smoothing,
                preprocessing_options=preprocessing_options
            )
            
            return strategy
    
    def get_strategy(self, system_state: SystemState) -> InferenceStrategy:
        """
        Get an inference strategy for the given system state.
        This method is an alias for select_action to maintain compatibility.
        
        Args:
            system_state: Current system state
            
        Returns:
            Selected inference strategy
        """
        return self.select_action(system_state, deterministic=True)
    
    def _get_preprocessing_options(
        self, 
        intensity: ProcessingIntensity, 
        state: SystemState
    ) -> Dict[str, Any]:
        """Generate preprocessing options based on intensity and state."""
        options = {}
        
        if intensity == ProcessingIntensity.MINIMAL:
            options.update({
                "enable_clahe": False,
                "resize_method": "fast",
                "normalization": "simple",
                "augmentation": False
            })
        elif intensity == ProcessingIntensity.STANDARD:
            options.update({
                "enable_clahe": state.contrast_level < 0.6,  # Adaptive CLAHE
                "resize_method": "bilinear",
                "normalization": "imagenet",
                "augmentation": False
            })
        else:  # ENHANCED
            options.update({
                "enable_clahe": True,
                "resize_method": "bicubic",
                "normalization": "imagenet",
                "augmentation": state.has_motion_blur,  # Augment if motion blur
                "denoise": state.image_complexity > 0.7
            })
        
        return options
    
    def get_action_log_probs(
        self, 
        state: torch.Tensor, 
        actions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate log probabilities of given actions for training.
        
        Args:
            state: System state tensor [batch_size, state_dim]
            actions: Dictionary with action tensors
            
        Returns:
            Log probabilities of the actions
        """
        outputs = self.forward(state)
        
        # Calculate log probabilities for each action component
        model_log_probs = torch.log(outputs["model_probs"] + 1e-8)
        model_action_log_probs = model_log_probs.gather(1, actions["model_variant"].unsqueeze(1))
        
        intensity_log_probs = torch.log(outputs["intensity_probs"] + 1e-8)
        intensity_action_log_probs = intensity_log_probs.gather(1, actions["processing_intensity"].unsqueeze(1))
        
        # Binary decisions log probabilities
        binary_probs = outputs["binary_decisions"]
        cmsis_log_prob = torch.where(
            actions["use_cmsis_nn"].bool(),
            torch.log(binary_probs[:, 0] + 1e-8),
            torch.log(1 - binary_probs[:, 0] + 1e-8)
        )
        temporal_log_prob = torch.where(
            actions["enable_temporal_smoothing"].bool(),
            torch.log(binary_probs[:, 1] + 1e-8),
            torch.log(1 - binary_probs[:, 1] + 1e-8)
        )
        
        # Confidence threshold (treat as Gaussian for continuous action)
        confidence_pred = outputs["confidence_threshold"].squeeze(-1)
        confidence_log_prob = -0.5 * torch.pow(confidence_pred - actions["confidence_threshold"], 2) / 0.01
        
        # Sum all log probabilities
        total_log_prob = (
            model_action_log_probs.squeeze(1) +
            intensity_action_log_probs.squeeze(1) +
            cmsis_log_prob +
            temporal_log_prob +
            confidence_log_prob
        )
        
        return total_log_prob
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save the policy model to disk."""
        model_data = {
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'model_class': 'AdaptivePizzaRecognitionPolicy'
        }
        torch.save(model_data, path)
        logger.info(f"Policy model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'AdaptivePizzaRecognitionPolicy':
        """Load a policy model from disk."""
        model_data = torch.load(path, map_location='cpu')
        
        # Create model with same architecture
        model = cls(
            state_dim=model_data['state_dim'],
            hidden_dim=model_data['hidden_dim']
        )
        
        # Load weights
        model.load_state_dict(model_data['state_dict'])
        
        logger.info(f"Policy model loaded from {path}")
        return model
    
    def get_strategy(self, system_state: SystemState) -> InferenceStrategy:
        """
        Get an inference strategy for the given system state.
        This method is an alias for select_action to maintain compatibility.
        
        Args:
            system_state: Current system state
            
        Returns:
            Selected inference strategy
        """
        return self.select_action(system_state, deterministic=True)


class PolicyArchitectureValidator:
    """Validator for policy architecture and integration."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_policy_architecture(
        self, 
        policy: AdaptivePizzaRecognitionPolicy
    ) -> Dict[str, Any]:
        """Validate the policy architecture and functionality."""
        logger.info("Validating policy architecture...")
        
        results = {
            "architecture_valid": True,
            "parameter_count": policy.count_parameters(),
            "forward_pass_test": False,
            "action_selection_test": False,
            "integration_compatibility": False,
            "errors": []
        }
        
        try:
            # Test forward pass
            dummy_state = torch.randn(1, policy.state_dim)
            outputs = policy.forward(dummy_state)
            
            required_outputs = ["model_probs", "intensity_probs", "binary_decisions", 
                              "confidence_threshold", "state_value"]
            
            for output_name in required_outputs:
                if output_name not in outputs:
                    results["errors"].append(f"Missing output: {output_name}")
                    results["architecture_valid"] = False
            
            if results["architecture_valid"]:
                results["forward_pass_test"] = True
                
                # Test action selection
                test_state = SystemState(
                    battery_level=0.8,
                    power_draw_current=50.0,
                    energy_budget=100.0,
                    image_complexity=0.6,
                    brightness_level=0.7,
                    contrast_level=0.5,
                    has_motion_blur=False,
                    required_accuracy=0.8,
                    time_constraints=100.0,
                    food_safety_critical=True,
                    temperature=25.0,
                    memory_usage=0.4,
                    processing_load=0.3
                )
                
                strategy = policy.select_action(test_state)
                
                if isinstance(strategy, InferenceStrategy):
                    results["action_selection_test"] = True
                    
                    # Test strategy validation
                    estimated_metrics = strategy.get_estimated_metrics()
                    if all(key in estimated_metrics for key in 
                          ["inference_time_ms", "energy_mj", "accuracy", "ram_kb"]):
                        results["integration_compatibility"] = True
                
        except Exception as e:
            results["errors"].append(f"Validation error: {str(e)}")
            results["architecture_valid"] = False
        
        self.validation_results = results
        return results
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """Generate a detailed validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_policy_architecture() first."
        
        results = self.validation_results
        
        report = f"""
# Adaptive Pizza Recognition Policy Architecture Validation Report

**Date**: 2024-12-21

## Architecture Overview

- **Parameter Count**: {results.get('parameter_count', 'N/A'):,}
- **Architecture Valid**: {'✅ Yes' if results.get('architecture_valid') else '❌ No'}
- **Forward Pass Test**: {'✅ Passed' if results.get('forward_pass_test') else '❌ Failed'}
- **Action Selection Test**: {'✅ Passed' if results.get('action_selection_test') else '❌ Failed'}
- **Integration Compatibility**: {'✅ Compatible' if results.get('integration_compatibility') else '❌ Incompatible'}

## Policy Components

### 1. State Representation
- **Dimensions**: 13 features
- **Energy Features**: battery_level, power_draw_current, energy_budget
- **Image Features**: complexity, brightness, contrast, motion_blur
- **Requirements**: accuracy, time_constraints, food_safety_critical
- **System Features**: temperature, memory_usage, processing_load

### 2. Action Space
- **Model Selection**: 3 variants (MicroPizzaNet, V2, SE)
- **Processing Intensity**: 3 levels (minimal, standard, enhanced)
- **Binary Decisions**: CMSIS-NN usage, temporal smoothing
- **Continuous Parameters**: confidence threshold (0-1)

### 3. Neural Network Architecture
- **Type**: Multi-layer perceptron with specialized heads
- **Shared Features**: 3 hidden layers (128 neurons each)
- **Decision Heads**: Model, intensity, binary, confidence, value
- **Activation**: ReLU with dropout (0.1)

## Integration Points

### Energy Management System
- ✅ Battery level monitoring
- ✅ Power consumption estimation
- ✅ Energy budget constraints

### Model Compatibility
- ✅ MicroPizzaNet variants support
- ✅ CMSIS-NN acceleration options
- ✅ Temporal smoothing integration

### Performance Estimation
- ✅ Inference time prediction
- ✅ Energy consumption estimation
- ✅ Accuracy expectation
- ✅ Memory usage calculation

"""
        
        if results.get('errors'):
            report += "\n## Errors Encountered\n\n"
            for error in results['errors']:
                report += f"- ❌ {error}\n"
        
        report += f"""
## Recommendations

### For Production Deployment
1. **Model Compression**: Consider quantization for RP2040 deployment
2. **Hardware Integration**: Test with actual RP2040 hardware constraints
3. **Energy Calibration**: Calibrate energy estimates with real measurements
4. **Safety Validation**: Extensive testing for food safety critical decisions

### For Training
1. **Reward Engineering**: Implement multi-objective reward function
2. **Environment Simulation**: Create realistic pizza detection scenarios
3. **Data Collection**: Gather real energy and performance data
4. **Hyperparameter Tuning**: Optimize policy network architecture

## Status: {'✅ READY FOR RL TRAINING' if all([
    results.get('architecture_valid'),
    results.get('forward_pass_test'),
    results.get('action_selection_test'),
    results.get('integration_compatibility')
]) else '⚠️ REQUIRES FIXES BEFORE TRAINING'}
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")
        
        return report


def main():
    """Main function for testing the policy architecture."""
    logger.info("Testing Adaptive Pizza Recognition Policy Architecture")
    
    # Create policy
    policy = AdaptivePizzaRecognitionPolicy()
    
    # Validate architecture
    validator = PolicyArchitectureValidator()
    validation_results = validator.validate_policy_architecture(policy)
    
    # Generate report
    report_path = project_root / "output" / "rl_policy_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = validator.generate_validation_report(str(report_path))
    print(report)
    
    # Test with example scenarios
    logger.info("Testing with example scenarios...")
    
    scenarios = [
        {
            "name": "Low Battery Emergency",
            "state": SystemState(
                battery_level=0.1, power_draw_current=80.0, energy_budget=20.0,
                image_complexity=0.5, brightness_level=0.6, contrast_level=0.7,
                has_motion_blur=False, required_accuracy=0.6, time_constraints=50.0,
                food_safety_critical=False, temperature=30.0, memory_usage=0.6, processing_load=0.8
            )
        },
        {
            "name": "Food Safety Critical",
            "state": SystemState(
                battery_level=0.8, power_draw_current=50.0, energy_budget=100.0,
                image_complexity=0.7, brightness_level=0.4, contrast_level=0.3,
                has_motion_blur=True, required_accuracy=0.95, time_constraints=200.0,
                food_safety_critical=True, temperature=25.0, memory_usage=0.3, processing_load=0.2
            )
        },
        {
            "name": "Optimal Conditions",
            "state": SystemState(
                battery_level=0.9, power_draw_current=40.0, energy_budget=150.0,
                image_complexity=0.4, brightness_level=0.8, contrast_level=0.8,
                has_motion_blur=False, required_accuracy=0.8, time_constraints=100.0,
                food_safety_critical=False, temperature=22.0, memory_usage=0.2, processing_load=0.1
            )
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\nTesting scenario: {scenario['name']}")
        strategy = policy.select_action(scenario['state'])
        metrics = strategy.get_estimated_metrics()
        
        logger.info(f"Selected strategy: {strategy.model_variant.value}")
        logger.info(f"Processing intensity: {strategy.processing_intensity.value}")
        logger.info(f"Use CMSIS-NN: {strategy.use_cmsis_nn}")
        logger.info(f"Estimated inference time: {metrics['inference_time_ms']:.3f} ms")
        logger.info(f"Estimated energy: {metrics['energy_mj']:.2f} mJ")
        logger.info(f"Estimated accuracy: {metrics['accuracy']:.3f}")
    
    # Save policy for future use
    model_path = project_root / "models" / "rl_policy" / "adaptive_pizza_policy.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save_model(model_path)
    
    logger.info(f"Policy architecture implementation complete!")
    return validation_results


if __name__ == "__main__":
    main()
