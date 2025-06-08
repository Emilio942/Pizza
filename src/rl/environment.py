#!/usr/bin/env python3
"""
Pizza RL Environment for adaptive recognition training.

This environment simulates pizza recognition scenarios with varying
energy constraints and quality requirements for RL training.
"""

import gym
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .adaptive_policy import (
    SystemState, InferenceStrategy, ModelVariant, 
    ProcessingIntensity, AdaptivePizzaRecognitionPolicy
)
from src.verification.pizza_verifier import PizzaVerifier
from src.pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE
from src.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class PizzaScenario:
    """Represents a pizza recognition scenario for training."""
    image_path: str
    true_class: str
    complexity: float  # Estimated image complexity [0,1]
    lighting_condition: str  # 'normal', 'dim', 'bright'
    urgency: float  # Time constraint factor [0,1]


class PizzaRLEnvironment(gym.Env):
    """
    Gym environment for training adaptive pizza recognition policies.
    
    State space: [battery_level, image_complexity, required_accuracy, 
                  time_constraints, temperature, memory_usage]
    Action space: [model_variant, processing_intensity, confidence_threshold,
                   use_cmsis_nn, enable_temporal_smoothing]
    """
    
    def __init__(
        self,
        pizza_scenarios: List[PizzaScenario],
        verifier: Optional[PizzaVerifier] = None,
        max_episodes: int = 1000,
        energy_budget: float = 100.0,  # mJ per episode
        accuracy_threshold: float = 0.8,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.scenarios = pizza_scenarios
        self.verifier = verifier
        self.max_episodes = max_episodes
        self.energy_budget = energy_budget
        self.accuracy_threshold = accuracy_threshold
        self.device = device
        
        # Environment state
        self.current_step = 0
        self.current_scenario_idx = 0
        self.remaining_energy = energy_budget
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_energy_costs = []
        
        # Load pizza detection models
        self.models = self._load_models()
        
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: [model(3), intensity(3), confidence(1), cmsis(1), temporal(1)]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0.0, 0, 0]),
            high=np.array([2, 2, 1.0, 1, 1]),
            dtype=np.float32
        )
        
        logger.info(f"Pizza RL Environment initialized with {len(self.scenarios)} scenarios")
    
    def _load_models(self) -> Dict[ModelVariant, torch.nn.Module]:
        """Load available pizza detection models."""
        models = {}
        
        try:
            # Try to load pre-trained models
            model_paths = {
                ModelVariant.MICRO_PIZZA_NET: "models/pizza_model_float32.pth",
                ModelVariant.MICRO_PIZZA_NET_V2: "models/pizza_model_v2.pth",
                ModelVariant.MICRO_PIZZA_NET_SE: "models/pizza_model_with_se.pth"
            }
            
            for variant, path in model_paths.items():
                try:
                    if variant == ModelVariant.MICRO_PIZZA_NET:
                        model = MicroPizzaNet(num_classes=len(CLASS_NAMES))
                    elif variant == ModelVariant.MICRO_PIZZA_NET_V2:
                        model = MicroPizzaNetV2(num_classes=len(CLASS_NAMES))
                    elif variant == ModelVariant.MICRO_PIZZA_NET_SE:
                        model = MicroPizzaNetWithSE(num_classes=len(CLASS_NAMES))
                    
                    # Try to load weights if available
                    try:
                        model.load_state_dict(torch.load(path, map_location=self.device))
                        logger.info(f"Loaded {variant.value} from {path}")
                    except:
                        logger.warning(f"Could not load weights for {variant.value}, using random initialization")
                    
                    model.to(self.device)
                    model.eval()
                    models[variant] = model
                    
                except Exception as e:
                    logger.error(f"Failed to load {variant.value}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
        # Ensure we have at least one model
        if not models:
            logger.warning("No pre-trained models available, creating default MicroPizzaNet")
            models[ModelVariant.MICRO_PIZZA_NET] = MicroPizzaNet(num_classes=6).to(self.device)
            
        return models
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_scenario_idx = 0
        self.remaining_energy = self.energy_budget
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_energy_costs = []
        
        # Sample initial system state
        state = self._generate_system_state()
        return self._state_to_observation(state)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action array [model_idx, intensity_idx, confidence, cmsis, temporal]
            
        Returns:
            Tuple of (next_observation, reward, done, info)
        """
        # Convert action to inference strategy
        strategy = self._action_to_strategy(action)
        
        # Get current scenario
        scenario = self.scenarios[self.current_scenario_idx]
        
        # Simulate pizza recognition with chosen strategy
        result = self._simulate_recognition(scenario, strategy)
        
        # Calculate reward
        reward = self._calculate_reward(result, strategy)
        
        # Update environment state
        self.current_step += 1
        self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenarios)
        self.remaining_energy -= result['energy_cost']
        
        # Record metrics
        self.episode_rewards.append(reward)
        self.episode_accuracies.append(result['accuracy'])
        self.episode_energy_costs.append(result['energy_cost'])
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_episodes or
            self.remaining_energy <= 0 or
            result['accuracy'] < self.accuracy_threshold * 0.5  # Severe accuracy drop
        )
        
        # Generate next state
        next_state = self._generate_system_state()
        next_observation = self._state_to_observation(next_state)
        
        # Info dictionary
        info = {
            'scenario': scenario,
            'strategy': strategy,
            'result': result,
            'remaining_energy': self.remaining_energy,
            'episode_step': self.current_step
        }
        
        return next_observation, reward, done, info
    
    def _generate_system_state(self) -> SystemState:
        """Generate realistic system state for current scenario."""
        scenario = self.scenarios[self.current_scenario_idx]
        
        # Battery level decreases over time
        battery_level = max(0.1, self.remaining_energy / self.energy_budget)
        
        # Image complexity from scenario
        image_complexity = scenario.complexity
        
        # Required accuracy varies based on scenario urgency
        base_accuracy = self.accuracy_threshold
        urgency_bonus = scenario.urgency * 0.1
        required_accuracy = min(1.0, base_accuracy + urgency_bonus)
        
        # Time constraints based on urgency and battery
        max_time = 200.0  # ms
        time_multiplier = (1.0 - scenario.urgency) * battery_level
        time_constraints = max_time * time_multiplier
        
        # Temperature simulation (affects battery and performance)
        base_temp = 25.0  # Celsius
        load_temp = (1.0 - battery_level) * 20.0  # Higher load -> higher temp
        temperature = base_temp + load_temp
        
        # Memory usage simulation
        base_memory = 0.3
        complexity_memory = image_complexity * 0.2
        memory_usage = min(0.9, base_memory + complexity_memory)
        
        return SystemState(
            battery_level=battery_level,
            image_complexity=image_complexity,
            required_accuracy=required_accuracy,
            time_constraints=time_constraints,
            temperature=temperature,
            memory_usage=memory_usage
        )
    
    def _state_to_observation(self, state: SystemState) -> np.ndarray:
        """Convert system state to observation vector."""
        return np.array([
            state.battery_level,
            state.image_complexity,
            state.required_accuracy,
            state.time_constraints / 200.0,  # Normalize
            state.temperature / 100.0,  # Normalize
            state.memory_usage
        ], dtype=np.float32)
    
    def _action_to_strategy(self, action: np.ndarray) -> InferenceStrategy:
        """Convert action vector to inference strategy."""
        model_variants = list(ModelVariant)
        intensity_levels = list(ProcessingIntensity)
        
        # Clip and convert actions
        model_idx = int(np.clip(action[0], 0, len(model_variants) - 1))
        intensity_idx = int(np.clip(action[1], 0, len(intensity_levels) - 1))
        confidence = float(np.clip(action[2], 0.0, 1.0))
        use_cmsis = bool(action[3] > 0.5)
        use_temporal = bool(action[4] > 0.5)
        
        return InferenceStrategy(
            model_variant=model_variants[model_idx],
            processing_intensity=intensity_levels[intensity_idx],
            confidence_threshold=confidence,
            use_cmsis_nn=use_cmsis,
            enable_temporal_smoothing=use_temporal
        )
    
    def _simulate_recognition(
        self, 
        scenario: PizzaScenario, 
        strategy: InferenceStrategy
    ) -> Dict[str, Any]:
        """
        Simulate pizza recognition with given strategy.
        
        Args:
            scenario: Pizza recognition scenario
            strategy: Inference strategy to use
            
        Returns:
            Dictionary with recognition results
        """
        # Get model for strategy
        model = self.models.get(strategy.model_variant)
        if model is None:
            # Fallback to first available model
            model = list(self.models.values())[0]
        
        # Simulate recognition latency
        base_latency = 50.0  # ms
        
        # Model complexity affects latency
        model_latency_factors = {
            ModelVariant.MICRO_PIZZA_NET: 1.0,
            ModelVariant.MICRO_PIZZA_NET_V2: 1.2,
            ModelVariant.MICRO_PIZZA_NET_SE: 1.5
        }
        
        # Processing intensity affects latency
        intensity_latency_factors = {
            ProcessingIntensity.LOW: 0.7,
            ProcessingIntensity.MEDIUM: 0.85,
            ProcessingIntensity.HIGH: 1.0
        }
        
        latency = base_latency * model_latency_factors[strategy.model_variant]
        latency *= intensity_latency_factors[strategy.processing_intensity]
        
        # CMSIS-NN reduces latency
        if strategy.use_cmsis_nn:
            latency *= 0.6
        
        # Estimate accuracy based on strategy and scenario
        policy = AdaptivePizzaRecognitionPolicy()
        estimated_accuracy = policy.estimate_accuracy(strategy, scenario.complexity)
        
        # Add some randomness to accuracy
        actual_accuracy = np.clip(
            estimated_accuracy + np.random.normal(0, 0.05),
            0.0, 1.0
        )
        
        # Estimate energy cost
        estimated_energy = policy.estimate_energy_cost(strategy, (48, 48))
        
        # Use verifier if available
        quality_score = actual_accuracy
        if self.verifier is not None:
            try:
                # Create mock prediction data for verifier
                verifier_data = {
                    'pizza_image_path': scenario.image_path,
                    'model_prediction': scenario.true_class,  # Simplified
                    'ground_truth_class': scenario.true_class,
                    'confidence_score': actual_accuracy,
                    'quality_score': actual_accuracy
                }
                quality_score = self.verifier.predict_quality(verifier_data)
            except Exception as e:
                logger.warning(f"Verifier error: {e}")
        
        return {
            'accuracy': actual_accuracy,
            'quality_score': quality_score,
            'energy_cost': estimated_energy,
            'latency': latency,
            'predicted_class': scenario.true_class,  # Simplified
            'confidence': actual_accuracy
        }
    
    def _calculate_reward(
        self, 
        result: Dict[str, Any], 
        strategy: InferenceStrategy
    ) -> float:
        """
        Calculate reward for the recognition result.
        
        Multi-objective reward balancing accuracy, energy efficiency, and speed.
        """
        # Accuracy reward (most important)
        accuracy_reward = result['accuracy'] * 10.0
        
        # Energy efficiency reward
        max_energy = 20.0  # mJ
        energy_efficiency = max(0, (max_energy - result['energy_cost']) / max_energy)
        energy_reward = energy_efficiency * 5.0
        
        # Speed reward
        max_latency = 100.0  # ms
        speed_efficiency = max(0, (max_latency - result['latency']) / max_latency)
        speed_reward = speed_efficiency * 3.0
        
        # Quality bonus from verifier
        quality_bonus = (result['quality_score'] - result['accuracy']) * 2.0
        
        # Penalty for accuracy below threshold
        accuracy_penalty = 0.0
        if result['accuracy'] < self.accuracy_threshold:
            accuracy_penalty = (self.accuracy_threshold - result['accuracy']) * 20.0
        
        # Total reward
        total_reward = (
            accuracy_reward + 
            energy_reward + 
            speed_reward + 
            quality_bonus - 
            accuracy_penalty
        )
        
        return total_reward
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get metrics for the current episode."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_reward': sum(self.episode_rewards),
            'mean_accuracy': np.mean(self.episode_accuracies),
            'total_energy_cost': sum(self.episode_energy_costs),
            'energy_efficiency': 1.0 - (sum(self.episode_energy_costs) / self.energy_budget),
            'steps_completed': len(self.episode_rewards)
        }


def create_pizza_scenarios(data_dir: str, num_scenarios: int = 100) -> List[PizzaScenario]:
    """
    Create pizza scenarios from available data.
    
    Args:
        data_dir: Directory containing pizza images
        num_scenarios: Number of scenarios to create
        
    Returns:
        List of pizza scenarios
    """
    from ..constants import CLASS_NAMES
    import os
    
    scenarios = []
    data_path = Path(data_dir)
    
    # Collect image files
    image_files = []
    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if class_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(class_dir.glob(ext)))
    
    # If no organized structure, try flat structure
    if not image_files:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(data_path.glob(f"**/{ext}")))
    
    # Create scenarios
    for i in range(min(num_scenarios, len(image_files))):
        img_path = image_files[i % len(image_files)]
        
        # Extract class from filename or directory
        true_class = 'basic'  # Default
        for class_name in CLASS_NAMES:
            if class_name in str(img_path).lower():
                true_class = class_name
                break
        
        # Estimate complexity based on filename patterns
        complexity = 0.5  # Default
        if 'burnt' in str(img_path).lower():
            complexity = 0.8  # Burnt pizzas are more complex
        elif 'progression' in str(img_path).lower():
            complexity = 0.9  # Progression tracking is complex
        elif 'basic' in str(img_path).lower():
            complexity = 0.3  # Basic pizzas are simpler
        
        # Random lighting and urgency
        lighting_conditions = ['normal', 'dim', 'bright']
        lighting = random.choice(lighting_conditions)
        urgency = random.uniform(0.2, 0.8)
        
        scenarios.append(PizzaScenario(
            image_path=str(img_path),
            true_class=true_class,
            complexity=complexity,
            lighting_condition=lighting,
            urgency=urgency
        ))
    
    logger.info(f"Created {len(scenarios)} pizza recognition scenarios")
    return scenarios
