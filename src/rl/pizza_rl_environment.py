#!/usr/bin/env python3
"""
Pizza RL Environment Implementation (Aufgabe 3.2)
==================================================

Implements a gym-compatible RL environment for adaptive pizza recognition
with integration to existing RP2040 emulator infrastructure and PPO training.

Features:
- Pizza-specific state representation with image complexity analysis
- Integration with existing RP2040 emulator and performance logging
- Multi-objective reward function (accuracy + energy + speed)
- Support for different model variants and CMSIS-NN acceleration
- Realistic battery and energy constraints
- Pizza verifier quality assessment integration
"""

import gym
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import cv2

# Project imports
from .adaptive_pizza_policy import (
    AdaptivePizzaRecognitionPolicy, SystemState, InferenceStrategy,
    ModelVariant, ProcessingIntensity, PolicyArchitectureValidator
)
from ..constants import CLASS_NAMES

# Import emulator infrastructure
try:
    from ..emulation.emulator import RP2040Emulator
    from ..emulation.simple_power_manager import AdaptiveMode
    EMULATOR_AVAILABLE = True
except ImportError:
    EMULATOR_AVAILABLE = False
    logging.warning("RP2040 emulator not available, using mock implementation")

# Import verifier integration
try:
    from ..verification.pizza_verifier import PizzaVerifier, VerifierData
    from ..integration.compatibility import get_verifier_integration, get_compatibility_manager
    VERIFIER_AVAILABLE = True
except ImportError:
    VERIFIER_AVAILABLE = False
    logging.warning("Pizza verifier not available, using mock implementation")

# Import pizza detection models
try:
    from ..pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Pizza detection models not available, using mock implementation")

logger = logging.getLogger(__name__)


@dataclass
class PizzaTask:
    """Represents a pizza recognition task with specific requirements."""
    image_path: str
    image_data: Optional[np.ndarray] = None
    required_accuracy: float = 0.8
    max_inference_time_ms: float = 100.0
    food_safety_critical: bool = False
    ground_truth_class: Optional[int] = None
    complexity_score: Optional[float] = None


@dataclass
class EnvironmentState:
    """Current state of the RL environment."""
    battery_level: float  # 0.0 to 1.0
    energy_budget: float  # Available energy in mJ
    current_task: Optional[PizzaTask] = None
    system_temperature: float = 25.0
    memory_usage: float = 0.5  # 0.0 to 1.0
    time_pressure: float = 0.0  # 0.0 to 1.0
    recent_accuracy: float = 0.8  # Recent model performance
    inference_queue_length: int = 0
    uptime_hours: float = 0.0


@dataclass
class ActionResult:
    """Result of executing an action in the environment."""
    success: bool
    inference_time_ms: float
    energy_consumed_mj: float
    accuracy_achieved: float
    predicted_class: int
    confidence: float
    verifier_quality: float
    reward: float
    info: Dict[str, Any]


class ImageComplexityAnalyzer:
    """Analyzes image complexity for adaptive processing decisions."""
    
    def __init__(self):
        self.device = torch.device('cpu')
    
    def analyze_complexity(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image complexity metrics.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Dictionary with complexity metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture complexity using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_complexity = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Color variance (if color image)
        if len(image.shape) == 3:
            color_var = np.var(image, axis=(0, 1)).mean()
            color_complexity = min(color_var / 10000.0, 1.0)  # Normalize
        else:
            color_complexity = 0.0
        
        # Overall complexity score
        complexity_score = (edge_density * 0.4 + 
                          texture_complexity * 0.4 + 
                          color_complexity * 0.2)
        
        return {
            'edge_density': edge_density,
            'texture_complexity': texture_complexity,
            'color_complexity': color_complexity,
            'overall_complexity': complexity_score
        }


class MockPizzaVerifier:
    """Mock verifier when real verifier is not available."""
    
    def predict_quality(self, verifier_data) -> float:
        """Mock quality prediction based on confidence."""
        base_quality = verifier_data.confidence_score * 0.8
        # Add some variance
        noise = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_quality + noise))


class MockEmulator:
    """Mock emulator when real emulator is not available."""
    
    def __init__(self):
        self.battery_level = 0.8
        self.temperature = 25.0
        self.memory_usage = 0.5
        
    def get_battery_level(self) -> float:
        return self.battery_level
        
    def get_temperature(self) -> float:
        return self.temperature
        
    def get_ram_usage(self) -> int:
        return int(self.memory_usage * 256 * 1024)  # 256KB total
        
    def simulate_inference(self, image: np.ndarray) -> Dict[str, Any]:
        """Mock inference simulation."""
        time.sleep(0.05)  # Simulate processing time
        return {
            'success': True,
            'inference_time': random.uniform(20, 150),
            'class_id': random.randint(0, 5),
            'confidence': random.uniform(0.5, 0.95),
            'energy_consumed': random.uniform(50, 200)
        }


class PizzaRLEnvironment(gym.Env):
    """
    RL Environment for adaptive pizza recognition training.
    
    This environment simulates pizza recognition tasks with energy constraints,
    integrating with existing RP2040 emulator infrastructure and pizza verifier.
    """
    
    def __init__(
        self,
        max_steps_per_episode: int = 100,
        battery_capacity_mah: float = 1500.0,
        initial_battery_level: float = 0.8,
        task_dataset_path: Optional[str] = None,
        device: str = 'cpu',
        enable_logging: bool = True,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Pizza RL Environment.
        
        Args:
            max_steps_per_episode: Maximum steps per episode
            battery_capacity_mah: Battery capacity in mAh
            initial_battery_level: Initial battery level (0.0-1.0)
            task_dataset_path: Path to pizza image dataset
            device: Device for computations
            enable_logging: Enable performance logging
            reward_weights: Weights for multi-objective reward
        """
        super().__init__()
        
        self.max_steps_per_episode = max_steps_per_episode
        self.battery_capacity_mah = battery_capacity_mah
        self.initial_battery_level = initial_battery_level
        self.device = device
        self.enable_logging = enable_logging
        
        # Reward function weights
        self.reward_weights = reward_weights or {
            'accuracy': 0.4,
            'energy_efficiency': 0.3,
            'speed': 0.2,
            'safety': 0.1
        }
        
        # Initialize components
        self._init_emulator()
        self._init_verifier()
        self._init_models()
        self._init_task_dataset(task_dataset_path)
        
        # Environment state
        self.current_step = 0
        self.env_state = EnvironmentState(
            battery_level=initial_battery_level,
            energy_budget=battery_capacity_mah * initial_battery_level * 3.6  # Convert to mJ
        )
        
        # Image complexity analyzer
        self.complexity_analyzer = ImageComplexityAnalyzer()
        
        # Performance tracking
        self.episode_metrics = {
            'total_reward': 0.0,
            'accuracy_scores': [],
            'energy_consumed': 0.0,
            'inference_times': [],
            'verifier_qualities': [],
            'classification_results': []
        }
        
        # Training scenario modifiers
        self.image_difficulty_modifier = 0.0
        self.time_constraint_modifier = 1.0
        self.temporal_sequence_mode = False
        self.previous_predictions = []
        
        # Define action and observation spaces
        self._define_spaces()
        
        logger.info(f"Pizza RL Environment initialized with {self.max_steps_per_episode} max steps")
    
    def _init_emulator(self) -> None:
        """Initialize RP2040 emulator."""
        if EMULATOR_AVAILABLE:
            try:
                self.emulator = RP2040Emulator(
                    battery_capacity_mah=self.battery_capacity_mah,
                    adaptive_mode=AdaptiveMode.BALANCED
                )
                logger.info("RP2040 emulator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize real emulator: {e}, using mock")
                self.emulator = MockEmulator()
        else:
            self.emulator = MockEmulator()
    
    def _init_verifier(self) -> None:
        """Initialize pizza verifier."""
        if VERIFIER_AVAILABLE:
            try:
                self.verifier_integration = get_verifier_integration(device=self.device)
                self.verifier = self.verifier_integration.verifier
                logger.info("Pizza verifier initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize real verifier: {e}, using mock")
                self.verifier = MockPizzaVerifier()
        else:
            self.verifier = MockPizzaVerifier()
    
    def _init_models(self) -> None:
        """Initialize pizza detection models."""
        self.models = {}
        
        if MODELS_AVAILABLE:
            try:
                # Initialize different model variants
                self.models['MicroPizzaNet'] = MicroPizzaNet(num_classes=6)
                self.models['MicroPizzaNetV2'] = MicroPizzaNetV2(num_classes=6)
                self.models['MicroPizzaNetWithSE'] = MicroPizzaNetWithSE(num_classes=6)
                
                # Load pretrained weights if available
                for model_name, model in self.models.items():
                    model.eval()
                    logger.info(f"Initialized {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize real models: {e}")
                self.models = {}
    
    def _init_task_dataset(self, dataset_path: Optional[str]) -> None:
        """Initialize pizza task dataset."""
        self.pizza_tasks = []
        
        if dataset_path and Path(dataset_path).exists():
            # Load real pizza images
            dataset_dir = Path(dataset_path)
            image_files = list(dataset_dir.glob('**/*.jpg')) + list(dataset_dir.glob('**/*.png'))
            
            for img_path in image_files[:50]:  # Limit for demo
                # Determine complexity based on file path or metadata
                required_accuracy = 0.8
                food_safety_critical = 'burnt' in str(img_path).lower()
                
                task = PizzaTask(
                    image_path=str(img_path),
                    required_accuracy=required_accuracy,
                    max_inference_time_ms=random.uniform(50, 200),
                    food_safety_critical=food_safety_critical
                )
                self.pizza_tasks.append(task)
        
        # Add synthetic tasks if no real dataset
        if not self.pizza_tasks:
            for i in range(20):
                task = PizzaTask(
                    image_path=f"synthetic_pizza_{i}.jpg",
                    required_accuracy=random.uniform(0.7, 0.95),
                    max_inference_time_ms=random.uniform(50, 200),
                    food_safety_critical=random.choice([True, False])
                )
                self.pizza_tasks.append(task)
        
        logger.info(f"Loaded {len(self.pizza_tasks)} pizza tasks")
    
    def _define_spaces(self) -> None:
        """Define action and observation spaces."""
        # State space: 13-dimensional as defined in SystemState
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )
        
        # Action space: Discrete actions for the strategy components
        # [model_variant (3), processing_intensity (3), use_cmsis_nn (2), 
        #  use_temporal_smoothing (2), confidence_threshold (5)]
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 5])
    
    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        self.current_step = 0
        
        # Reset battery and energy state
        self.env_state.battery_level = self.initial_battery_level
        self.env_state.energy_budget = (
            self.battery_capacity_mah * self.initial_battery_level * 3.6
        )
        self.env_state.system_temperature = 25.0
        self.env_state.memory_usage = 0.5
        self.env_state.time_pressure = 0.0
        self.env_state.recent_accuracy = 0.8
        self.env_state.inference_queue_length = 0
        self.env_state.uptime_hours = 0.0
        
        # Reset episode metrics
        self.episode_metrics = {
            'total_reward': 0.0,
            'accuracy_scores': [],
            'energy_consumed': 0.0,
            'inference_times': [],
            'verifier_qualities': []
        }
        
        # Select a new pizza task
        self._select_new_task()
        
        # Return initial observation and info dict
        info = {
            'episode_step': self.current_step,
            'battery_level': self.env_state.battery_level,
            'current_task': self.env_state.current_task.image_path if self.env_state.current_task else None
        }
        return self._get_observation(), info
    
    def _select_new_task(self) -> None:
        """Select a new pizza recognition task."""
        if self.pizza_tasks:
            task = random.choice(self.pizza_tasks)
            
            # Load image if it's a real file
            if Path(task.image_path).exists():
                image = cv2.imread(task.image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    task.image_data = cv2.resize(image, (48, 48))  # Resize to model input
                    
                    # Analyze complexity
                    complexity_metrics = self.complexity_analyzer.analyze_complexity(task.image_data)
                    task.complexity_score = complexity_metrics['overall_complexity']
            
            # Generate synthetic image if no real image available
            if task.image_data is None:
                task.image_data = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
                task.complexity_score = random.uniform(0.3, 0.8)
            
            self.env_state.current_task = task
        else:
            # Fallback synthetic task
            task = PizzaTask(
                image_path="synthetic.jpg",
                image_data=np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8),
                required_accuracy=random.uniform(0.7, 0.9),
                max_inference_time_ms=random.uniform(50, 200),
                food_safety_critical=random.choice([True, False]),
                complexity_score=random.uniform(0.3, 0.8)
            )
            self.env_state.current_task = task
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation as SystemState vector."""
        if self.env_state.current_task is None:
            self._select_new_task()
        
        task = self.env_state.current_task
        
        # Create SystemState representation with correct parameters
        system_state = SystemState(
            battery_level=self.env_state.battery_level,
            power_draw_current=15.0,  # Mock power consumption
            energy_budget=self.env_state.energy_budget,
            image_complexity=task.complexity_score or 0.5,
            brightness_level=0.6,  # Mock brightness
            contrast_level=0.7,    # Mock contrast
            has_motion_blur=False, # Mock motion blur
            required_accuracy=task.required_accuracy,
            time_constraints=task.max_inference_time_ms,
            food_safety_critical=task.food_safety_critical,
            temperature=self.env_state.system_temperature,
            memory_usage=self.env_state.memory_usage,
            processing_load=self.env_state.inference_queue_length / 10.0
        )
        
        return system_state.to_tensor().numpy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action vector [model_variant, processing_intensity, 
                   use_cmsis_nn, use_temporal_smoothing, confidence_threshold]
                   
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1
        
        # Decode action to InferenceStrategy
        strategy = self._decode_action(action)
        
        # Execute inference with chosen strategy
        result = self._execute_inference(strategy)
        
        # Update environment state
        self._update_environment_state(result)
        
        # Calculate reward
        reward = self._calculate_reward(result, strategy)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Update episode metrics
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['accuracy_scores'].append(result.accuracy_achieved)
        self.episode_metrics['energy_consumed'] += result.energy_consumed_mj
        self.episode_metrics['inference_times'].append(result.inference_time_ms)
        self.episode_metrics['verifier_qualities'].append(result.verifier_quality)
        
        # Select new task for next step (if not done)
        if not done:
            self._select_new_task()
        
        # Prepare info dictionary
        info = {
            'strategy': asdict(strategy),
            'result': asdict(result),
            'battery_level': self.env_state.battery_level,
            'energy_budget': self.env_state.energy_budget,
            'episode_metrics': self.episode_metrics.copy()
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _decode_action(self, action: np.ndarray) -> InferenceStrategy:
        """Decode action vector to InferenceStrategy."""
        model_variants = [ModelVariant.MICRO_PIZZA_NET,
                         ModelVariant.MICRO_PIZZA_NET_V2,
                         ModelVariant.MICRO_PIZZA_NET_SE]

        processing_intensities = [ProcessingIntensity.MINIMAL,
                                ProcessingIntensity.STANDARD,
                                ProcessingIntensity.ENHANCED]

        confidence_thresholds = [0.6, 0.7, 0.8, 0.85, 0.9]

        # Clip action values to valid ranges to handle out-of-bounds gracefully
        model_idx = max(0, min(action[0], len(model_variants) - 1))
        intensity_idx = max(0, min(action[1], len(processing_intensities) - 1))
        confidence_idx = max(0, min(action[4], len(confidence_thresholds) - 1))

        strategy = InferenceStrategy(
            model_variant=model_variants[model_idx],
            processing_intensity=processing_intensities[intensity_idx],
            use_cmsis_nn=bool(action[2]),
            enable_temporal_smoothing=bool(action[3]),
            confidence_threshold=confidence_thresholds[confidence_idx],
            preprocessing_options={}
        )
        
        return strategy
    
    def _execute_inference(self, strategy: InferenceStrategy) -> ActionResult:
        """Execute pizza inference with given strategy."""
        task = self.env_state.current_task
        
        # Simulate inference execution
        start_time = time.time()
        
        if EMULATOR_AVAILABLE and hasattr(self.emulator, 'simulate_inference'):
            # Use real emulator
            emulator_result = self.emulator.simulate_inference(task.image_data)
            inference_time_ms = emulator_result.get('inference_time', 100.0)
            predicted_class = emulator_result.get('class_id', 0)
            confidence = emulator_result.get('confidence', 0.8)
            energy_consumed_mj = emulator_result.get('energy_consumed', 100.0)
            success = emulator_result.get('success', True)
        else:
            # Mock simulation
            base_time = {
                ProcessingIntensity.MINIMAL: 80.0,
                ProcessingIntensity.STANDARD: 120.0,
                ProcessingIntensity.ENHANCED: 180.0
            }[strategy.processing_intensity]
            
            # CMSIS-NN acceleration
            if strategy.use_cmsis_nn:
                base_time *= 0.7
            
            # Add complexity factor
            complexity_factor = 1.0 + (task.complexity_score or 0.5) * 0.5
            inference_time_ms = base_time * complexity_factor
            
            # Energy consumption (proportional to time and processing intensity)
            energy_factor = {
                ProcessingIntensity.MINIMAL: 0.8,
                ProcessingIntensity.STANDARD: 1.0,
                ProcessingIntensity.ENHANCED: 1.4
            }[strategy.processing_intensity]
            
            energy_consumed_mj = inference_time_ms * 0.8 * energy_factor
            
            # Mock prediction results
            predicted_class = random.randint(0, 5)
            base_confidence = random.uniform(0.6, 0.9)
            
            # Higher processing intensity -> higher confidence
            intensity_boost = {
                ProcessingIntensity.MINIMAL: 0.0,
                ProcessingIntensity.STANDARD: 0.05,
                ProcessingIntensity.ENHANCED: 0.1
            }[strategy.processing_intensity]
            
            confidence = min(0.99, base_confidence + intensity_boost)
            success = True
        
        # Calculate accuracy (mock for now)
        accuracy_achieved = confidence * 0.9  # Simplified accuracy estimation
        
        # Apply temporal smoothing effect
        if strategy.enable_temporal_smoothing:
            accuracy_achieved *= 1.05  # Slight boost
            confidence *= 1.02
        
        # Pizza verifier quality assessment
        if VERIFIER_AVAILABLE and hasattr(self.verifier, 'predict_quality'):
            verifier_data = VerifierData(
                pizza_image_path=task.image_path,
                model_prediction=CLASS_NAMES[predicted_class],
                ground_truth_class=task.ground_truth_class or CLASS_NAMES[predicted_class],
                confidence_score=confidence,
                quality_score=accuracy_achieved,
                model_variant=strategy.model_variant.value
            )
            verifier_quality = self.verifier.predict_quality(verifier_data)
        else:
            # Mock verifier
            verifier_quality = confidence * 0.8 + random.uniform(-0.1, 0.1)
            verifier_quality = max(0.0, min(1.0, verifier_quality))
        
        # Prepare result
        result = ActionResult(
            success=success,
            inference_time_ms=inference_time_ms,
            energy_consumed_mj=energy_consumed_mj,
            accuracy_achieved=accuracy_achieved,
            predicted_class=predicted_class,
            confidence=confidence,
            verifier_quality=verifier_quality,
            reward=0.0,  # Will be calculated separately
            info={
                'strategy_used': asdict(strategy),
                'task_requirements': {
                    'required_accuracy': task.required_accuracy,
                    'max_time_ms': task.max_inference_time_ms,
                    'safety_critical': task.food_safety_critical
                }
            }
        )
        
        return result
    
    def _update_environment_state(self, result: ActionResult) -> None:
        """Update environment state after inference execution."""
        # Update battery and energy
        energy_consumed_wh = result.energy_consumed_mj / 3600.0  # Convert mJ to Wh
        battery_consumed = energy_consumed_wh / (self.battery_capacity_mah / 1000.0)
        
        self.env_state.battery_level = max(0.0, 
            self.env_state.battery_level - battery_consumed)
        self.env_state.energy_budget = max(0.0, 
            self.env_state.energy_budget - result.energy_consumed_mj)
        
        # Update system temperature (simplified thermal model)
        heat_generated = result.energy_consumed_mj / 100.0  # Simplified
        self.env_state.system_temperature += heat_generated
        
        # Natural cooling
        ambient_temp = 25.0
        cooling_rate = 0.1
        self.env_state.system_temperature = (
            self.env_state.system_temperature * (1 - cooling_rate) +
            ambient_temp * cooling_rate
        )
        
        # Update recent accuracy (moving average)
        alpha = 0.3
        self.env_state.recent_accuracy = (
            alpha * result.accuracy_achieved + 
            (1 - alpha) * self.env_state.recent_accuracy
        )
        
        # Update uptime
        self.env_state.uptime_hours += result.inference_time_ms / (1000.0 * 3600.0)
        
        # Memory usage fluctuation
        self.env_state.memory_usage += random.uniform(-0.05, 0.05)
        self.env_state.memory_usage = max(0.2, min(0.9, self.env_state.memory_usage))
    
    def _calculate_reward(self, result: ActionResult, strategy: InferenceStrategy) -> float:
        """Calculate multi-objective reward based on result and strategy."""
        task = self.env_state.current_task
        weights = self.reward_weights
        
        # Accuracy component
        accuracy_met = result.accuracy_achieved >= task.required_accuracy
        accuracy_reward = weights['accuracy'] * (
            result.accuracy_achieved if accuracy_met else 
            result.accuracy_achieved * 0.5  # Penalty for not meeting requirement
        )
        
        # Energy efficiency component
        max_energy_budget = 200.0  # mJ, reasonable for one inference
        energy_efficiency = max(0.0, 1.0 - (result.energy_consumed_mj / max_energy_budget))
        energy_reward = weights['energy_efficiency'] * energy_efficiency
        
        # Speed component
        time_met = result.inference_time_ms <= task.max_inference_time_ms
        time_efficiency = max(0.0, 1.0 - (result.inference_time_ms / task.max_inference_time_ms))
        speed_reward = weights['speed'] * (
            time_efficiency if time_met else 
            time_efficiency * 0.5  # Penalty for exceeding time limit
        )
        
        # Safety component (extra reward for food safety critical tasks)
        safety_reward = weights['safety'] * (
            result.verifier_quality * 2.0 if task.food_safety_critical else 
            result.verifier_quality
        )
        
        # Combine rewards
        total_reward = accuracy_reward + energy_reward + speed_reward + safety_reward
        
        # Additional penalties
        penalties = 0.0
        
        # Battery depletion penalty
        if self.env_state.battery_level < 0.2:
            penalties += 0.1  # Low battery penalty
        
        # Temperature penalty
        if self.env_state.system_temperature > 60.0:
            penalties += 0.05  # Overheating penalty
        
        final_reward = total_reward - penalties
        
        # Store reward in result
        result.reward = final_reward
        
        return final_reward
    
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate."""
        # Maximum steps reached
        if self.current_step >= self.max_steps_per_episode:
            return True
        
        # Battery depleted
        if self.env_state.battery_level <= 0.05:
            return True
        
        # System overheating
        if self.env_state.system_temperature > 80.0:
            return True
        
        return False
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode performance."""
        metrics = self.episode_metrics
        
        summary = {
            'total_steps': self.current_step,
            'total_reward': metrics['total_reward'],
            'average_accuracy': np.mean(metrics['accuracy_scores']) if metrics['accuracy_scores'] else 0.0,
            'total_energy_consumed': metrics['energy_consumed'],
            'average_inference_time': np.mean(metrics['inference_times']) if metrics['inference_times'] else 0.0,
            'average_verifier_quality': np.mean(metrics['verifier_qualities']) if metrics['verifier_qualities'] else 0.0,
            'final_battery_level': self.env_state.battery_level,
            'final_temperature': self.env_state.system_temperature
        }
        
        return summary
    
    def set_reward_weights(self, accuracy_weight: float, energy_weight: float, speed_weight: float):
        """Set the multi-objective reward weights."""
        total_weight = accuracy_weight + energy_weight + speed_weight
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            accuracy_weight /= total_weight
            energy_weight /= total_weight  
            speed_weight /= total_weight
            
        self.reward_weights['accuracy'] = accuracy_weight
        self.reward_weights['energy_efficiency'] = energy_weight
        self.reward_weights['speed'] = speed_weight
        
        logger.info(f"Updated reward weights: accuracy={accuracy_weight:.3f}, energy={energy_weight:.3f}, speed={speed_weight:.3f}")
    
    def get_config(self) -> Dict:
        """Get current environment configuration."""
        return {
            'max_steps_per_episode': self.max_steps_per_episode,
            'battery_capacity_mah': self.battery_capacity_mah,
            'initial_battery_level': self.initial_battery_level,
            'reward_weights': self.reward_weights.copy(),
            'device': str(self.device),
            'enable_logging': self.enable_logging
        }
    
    def set_config(self, config: Dict):
        """Set environment configuration."""
        if 'reward_weights' in config:
            self.reward_weights.update(config['reward_weights'])
        if 'max_steps_per_episode' in config:
            self.max_steps_per_episode = config['max_steps_per_episode']
        if 'battery_capacity_mah' in config:
            self.battery_capacity_mah = config['battery_capacity_mah']
        if 'initial_battery_level' in config:
            self.initial_battery_level = config['initial_battery_level']
            
        logger.info("Environment configuration updated")
    
    def set_image_difficulty_modifier(self, modifier: float):
        """Set image difficulty modifier for scenario testing."""
        self.image_difficulty_modifier = modifier
        logger.info(f"Image difficulty modifier set to: {modifier}")
    
    def reset_battery_level(self, level: float):
        """Reset battery level for scenario testing."""
        self.env_state.battery_level = max(0.0, min(1.0, level))
        logger.info(f"Battery level reset to: {level:.3f}")
    
    def set_time_constraint_modifier(self, modifier: float):
        """Set time constraint modifier for speed-critical scenarios."""
        self.time_constraint_modifier = modifier
        logger.info(f"Time constraint modifier set to: {modifier}")
    
    def enable_temporal_sequence_mode(self, enable: bool):
        """Enable/disable temporal sequence evaluation mode."""
        self.temporal_sequence_mode = enable
        logger.info(f"Temporal sequence mode: {'enabled' if enable else 'disabled'}")
    
    def get_pizza_specific_metrics(self) -> Dict:
        """Get pizza-specific performance metrics."""
        if not hasattr(self, 'episode_metrics'):
            return {}
            
        metrics = self.episode_metrics
        
        # Calculate pizza-specific derived metrics
        pizza_metrics = {
            'food_safety_accuracy': self._calculate_food_safety_accuracy(metrics),
            'burnt_detection_precision': self._calculate_burnt_detection_precision(metrics),
            'temporal_consistency': self._calculate_temporal_consistency(metrics),
            'energy_per_classification': self._calculate_energy_efficiency(metrics),
            'real_time_performance': self._calculate_real_time_performance(metrics)
        }
        
        return pizza_metrics
    
    def _calculate_food_safety_accuracy(self, metrics: Dict) -> float:
        """Calculate accuracy for food safety critical classifications."""
        # Focus on raw vs cooked detection accuracy
        safety_critical_results = [
            result for result in metrics.get('classification_results', [])
            if result.get('true_class') in ['raw', 'cooked', 'burnt']
        ]
        
        if not safety_critical_results:
            return 0.0
            
        correct = sum(1 for r in safety_critical_results if r['predicted_class'] == r['true_class'])
        return correct / len(safety_critical_results)
    
    def _calculate_burnt_detection_precision(self, metrics: Dict) -> float:
        """Calculate precision for burnt pizza detection."""
        burnt_predictions = [
            r for r in metrics.get('classification_results', [])
            if r.get('predicted_class') == 'burnt'
        ]
        
        if not burnt_predictions:
            return 0.0
            
        correct_burnt = sum(1 for r in burnt_predictions if r['true_class'] == 'burnt')
        return correct_burnt / len(burnt_predictions)
    
    def _calculate_temporal_consistency(self, metrics: Dict) -> float:
        """Calculate temporal consistency of predictions."""
        if not hasattr(self, 'previous_predictions'):
            return 1.0
            
        # Simple temporal consistency metric
        # In a real implementation, this would analyze prediction stability over time
        return 0.85 + random.uniform(-0.1, 0.1)  # Placeholder implementation
    
    def _calculate_energy_efficiency(self, metrics: Dict) -> float:
        """Calculate energy consumption per classification."""
        total_energy = metrics.get('energy_consumed', 0.0)
        num_classifications = len(metrics.get('classification_results', []))
        
        if num_classifications == 0:
            return 0.0
            
        return total_energy / num_classifications
    
    def _calculate_real_time_performance(self, metrics: Dict) -> float:
        """Calculate real-time performance score."""
        inference_times = metrics.get('inference_times', [])
        if not inference_times:
            return 0.0
            
        avg_time = np.mean(inference_times)
        target_time = 100.0  # Target 100ms for real-time performance
        
        # Performance score: 1.0 if under target, decreasing exponentially
        return min(1.0, np.exp(-(avg_time - target_time) / target_time))


class PizzaRLEnvironmentFactory:
    """Factory for creating Pizza RL Environment instances with different configurations."""
    
    @staticmethod
    def create_training_environment(
        dataset_path: Optional[str] = None,
        difficulty: str = 'medium'
    ) -> PizzaRLEnvironment:
        """Create environment for training."""
        configs = {
            'easy': {
                'max_steps_per_episode': 50,
                'initial_battery_level': 0.9,
                'reward_weights': {'accuracy': 0.5, 'energy_efficiency': 0.2, 'speed': 0.2, 'safety': 0.1}
            },
            'medium': {
                'max_steps_per_episode': 100,
                'initial_battery_level': 0.8,
                'reward_weights': {'accuracy': 0.4, 'energy_efficiency': 0.3, 'speed': 0.2, 'safety': 0.1}
            },
            'hard': {
                'max_steps_per_episode': 150,
                'initial_battery_level': 0.6,
                'reward_weights': {'accuracy': 0.3, 'energy_efficiency': 0.4, 'speed': 0.2, 'safety': 0.1}
            }
        }
        
        config = configs.get(difficulty, configs['medium'])
        
        return PizzaRLEnvironment(
            task_dataset_path=dataset_path,
            **config
        )
    
    @staticmethod
    def create_evaluation_environment(
        dataset_path: Optional[str] = None
    ) -> PizzaRLEnvironment:
        """Create environment for evaluation."""
        return PizzaRLEnvironment(
            max_steps_per_episode=200,
            initial_battery_level=1.0,
            task_dataset_path=dataset_path,
            reward_weights={'accuracy': 0.4, 'energy_efficiency': 0.3, 'speed': 0.2, 'safety': 0.1}
        )


if __name__ == "__main__":
    # Test the environment
    logging.basicConfig(level=logging.INFO)
    
    env = PizzaRLEnvironmentFactory.create_training_environment()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few test steps
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Battery: {info['battery_level']:.3f}")
        print(f"  Done: {done}")
        
        if done:
            break
    
    summary = env.get_episode_summary()
    print(f"\nEpisode Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
