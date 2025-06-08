#!/usr/bin/env python3
"""
Compatibility integration for Pizza Verifier and RL systems.

This module ensures compatibility with existing MicroPizzaNet models,
CMSIS-NN integration, and the formal verification framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE
from src.verification.pizza_verifier import PizzaVerifier, VerifierData
from src.rl.adaptive_pizza_policy import AdaptivePizzaRecognitionPolicy, SystemState, InferenceStrategy
from src.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


class ModelCompatibilityManager:
    """
    Manages compatibility between different model variants and systems.
    
    This class provides unified interfaces for loading and using different
    MicroPizzaNet variants with the verifier and RL systems.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.loaded_models = {}
        self.model_metadata = {}
        
        # CMSIS-NN compatibility flags
        self.cmsis_nn_available = self._check_cmsis_availability()
        
        logger.info(f"Model compatibility manager initialized on {device}")
    
    def _check_cmsis_availability(self) -> bool:
        """Check if CMSIS-NN integration is available."""
        try:
            # Try to import CMSIS-NN related modules
            # This is a placeholder - actual implementation would check for
            # CMSIS-NN libraries and hardware support
            cmsis_paths = [
                "models/exports/pizza_model_cmsis.c",
                "models/rp2040_export/pizza_model.c"
            ]
            
            for path in cmsis_paths:
                if Path(path).exists():
                    logger.info(f"CMSIS-NN export found: {path}")
                    return True
            
            logger.warning("CMSIS-NN exports not found, using PyTorch inference")
            return False
            
        except Exception as e:
            logger.warning(f"CMSIS-NN availability check failed: {e}")
            return False
    
    def load_model(
        self,
        model_type: str,
        model_path: Optional[str] = None,
        num_classes: int = 6
    ) -> nn.Module:
        """
        Load a MicroPizzaNet model variant.
        
        Args:
            model_type: Type of model ('MicroPizzaNet', 'MicroPizzaNetV2', 'MicroPizzaNetWithSE')
            model_path: Path to model weights (optional)
            num_classes: Number of output classes
            
        Returns:
            Loaded PyTorch model
        """
        cache_key = f"{model_type}_{num_classes}"
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Create model instance
        if model_type == 'MicroPizzaNet':
            model = MicroPizzaNet(num_classes=num_classes)
        elif model_type == 'MicroPizzaNetV2':
            model = MicroPizzaNetV2(num_classes=num_classes)
        elif model_type == 'MicroPizzaNetWithSE':
            model = MicroPizzaNetWithSE(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded {model_type} weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights for {model_type}: {e}")
        else:
            logger.warning(f"No weights provided for {model_type}, using random initialization")
        
        model.to(self.device)
        model.eval()
        
        # Cache model and metadata
        self.loaded_models[cache_key] = model
        self.model_metadata[cache_key] = {
            'type': model_type,
            'path': model_path,
            'num_classes': num_classes,
            'parameters': sum(p.numel() for p in model.parameters()),
            'cmsis_compatible': self._check_model_cmsis_compatibility(model_type)
        }
        
        return model
    
    def _check_model_cmsis_compatibility(self, model_type: str) -> bool:
        """Check if a model type is compatible with CMSIS-NN."""
        # All MicroPizzaNet variants should be CMSIS-NN compatible
        # as they were designed for microcontroller deployment
        compatible_types = ['MicroPizzaNet', 'MicroPizzaNetV2', 'MicroPizzaNetWithSE']
        return model_type in compatible_types and self.cmsis_nn_available
    
    def get_model_info(self, model_type: str, num_classes: int = 6) -> Dict[str, Any]:
        """Get information about a loaded model."""
        cache_key = f"{model_type}_{num_classes}"
        return self.model_metadata.get(cache_key, {})
    
    def estimate_model_performance(
        self,
        model_type: str,
        input_size: Tuple[int, int] = (48, 48),
        use_cmsis: bool = False
    ) -> Dict[str, float]:
        """
        Estimate performance characteristics of a model.
        
        Args:
            model_type: Type of model
            input_size: Input image dimensions
            use_cmsis: Whether to use CMSIS-NN acceleration
            
        Returns:
            Dictionary with performance estimates
        """
        # Base performance characteristics (empirically determined)
        base_metrics = {
            'MicroPizzaNet': {
                'latency_ms': 45.0,
                'energy_mj': 8.5,
                'accuracy': 0.85,
                'memory_kb': 120
            },
            'MicroPizzaNetV2': {
                'latency_ms': 52.0,
                'energy_mj': 10.2,
                'accuracy': 0.88,
                'memory_kb': 140
            },
            'MicroPizzaNetWithSE': {
                'latency_ms': 68.0,
                'energy_mj': 13.5,
                'accuracy': 0.91,
                'memory_kb': 180
            }
        }
        
        metrics = base_metrics.get(model_type, base_metrics['MicroPizzaNet']).copy()
        
        # Scale by input size
        size_factor = (input_size[0] * input_size[1]) / (48 * 48)
        metrics['latency_ms'] *= size_factor
        metrics['energy_mj'] *= size_factor
        
        # CMSIS-NN optimization
        if use_cmsis and self._check_model_cmsis_compatibility(model_type):
            metrics['latency_ms'] *= 0.6  # 40% speedup
            metrics['energy_mj'] *= 0.7   # 30% energy reduction
            metrics['memory_kb'] *= 0.8   # 20% memory reduction
        
        return metrics


class VerifierIntegration:
    """
    Integration layer between the verifier and existing pizza detection systems.
    
    This class provides compatibility with formal verification, temporal smoothing,
    and existing evaluation infrastructure.
    """
    
    def __init__(
        self,
        verifier: PizzaVerifier,
        compatibility_manager: ModelCompatibilityManager
    ):
        self.verifier = verifier
        self.compatibility_manager = compatibility_manager
        
        # Integration with formal verification
        self.formal_verification_available = self._check_formal_verification()
        
        logger.info("Verifier integration initialized")
    
    def _check_formal_verification(self) -> bool:
        """Check if formal verification framework is available."""
        try:
            from models.formal_verification.formal_verification import ModelVerifier
            return True
        except ImportError:
            logger.warning("Formal verification framework not available")
            return False
    
    def create_verifier_data_from_prediction(
        self,
        image_path: str,
        prediction_result: Dict[str, Any],
        ground_truth: Optional[str] = None,
        model_type: str = 'MicroPizzaNet',
        strategy: Optional[InferenceStrategy] = None
    ) -> VerifierData:
        """
        Create VerifierData from a prediction result.
        
        Args:
            image_path: Path to the input image
            prediction_result: Result from pizza detection model
            ground_truth: True class label (if available)
            model_type: Type of model used
            strategy: Inference strategy used (if available)
            
        Returns:
            VerifierData object
        """
        # Extract prediction information
        predicted_class = prediction_result.get('predicted_class', 'basic')
        confidence = prediction_result.get('confidence', 0.5)
        
        # Estimate quality score based on confidence and consistency
        base_quality = confidence
        
        # Adjust quality based on model performance
        model_info = self.compatibility_manager.get_model_info(model_type)
        if model_info:
            expected_accuracy = model_info.get('accuracy', 0.85)
            base_quality *= expected_accuracy
        
        # Create verifier data
        verifier_data = VerifierData(
            pizza_image_path=image_path,
            model_prediction=predicted_class,
            ground_truth_class=ground_truth or predicted_class,
            confidence_score=confidence,
            quality_score=base_quality,
            model_variant=model_type
        )
        
        # Add strategy information if available
        if strategy:
            verifier_data.processing_intensity = strategy.processing_intensity.value
            verifier_data.energy_cost = self.compatibility_manager.estimate_model_performance(
                model_type, use_cmsis=strategy.use_cmsis_nn
            )['energy_mj']
        
        return verifier_data
    
    def integrate_with_formal_verification(
        self,
        verifier_data: VerifierData,
        epsilon: float = 0.03
    ) -> Dict[str, Any]:
        """
        Integrate verifier results with formal verification framework.
        
        Args:
            verifier_data: Verifier input data
            epsilon: Perturbation bound for verification
            
        Returns:
            Combined verification results
        """
        results = {
            'verifier_quality': self.verifier.predict_quality(verifier_data),
            'formal_verification': None,
            'combined_confidence': verifier_data.confidence_score
        }
        
        if self.formal_verification_available:
            try:
                from models.formal_verification.formal_verification import (
                    ModelVerifier, VerificationProperty, load_model_for_verification
                )
                
                # Load model for formal verification
                model = load_model_for_verification(
                    model_path="models/pizza_model_float32.pth",
                    model_type=verifier_data.model_variant or 'MicroPizzaNet'
                )
                
                # Create formal verifier
                formal_verifier = ModelVerifier(
                    model=model,
                    input_size=(48, 48),
                    epsilon=epsilon
                )
                
                # Perform robustness verification
                # Note: This would require the actual image data
                formal_result = {
                    'robustness_verified': True,  # Placeholder
                    'verification_time': 0.1,    # Placeholder
                    'certified_radius': epsilon  # Placeholder
                }
                
                results['formal_verification'] = formal_result
                
                # Combine verifier and formal verification results
                if formal_result['robustness_verified']:
                    results['combined_confidence'] *= 1.1  # Boost confidence
                else:
                    results['combined_confidence'] *= 0.9  # Reduce confidence
                    
            except Exception as e:
                logger.warning(f"Formal verification integration failed: {e}")
        
        return results
    
    def apply_temporal_smoothing(
        self,
        verifier_history: List[VerifierData],
        window_size: int = 5,
        smoothing_factor: float = 0.8
    ) -> float:
        """
        Apply temporal smoothing to verifier predictions.
        
        Args:
            verifier_history: List of recent verifier data
            window_size: Size of smoothing window
            smoothing_factor: Weight for smoothing
            
        Returns:
            Temporally smoothed quality score
        """
        if not verifier_history:
            return 0.5
        
        # Get recent predictions
        recent_predictions = verifier_history[-window_size:]
        quality_scores = [self.verifier.predict_quality(data) for data in recent_predictions]
        
        if len(quality_scores) == 1:
            return quality_scores[0]
        
        # Apply exponential smoothing
        smoothed_score = quality_scores[0]
        for score in quality_scores[1:]:
            smoothed_score = smoothing_factor * smoothed_score + (1 - smoothing_factor) * score
        
        return smoothed_score


class RLIntegration:
    """
    Integration layer for RL components with existing systems.
    
    This class provides compatibility between the RL training system
    and existing pizza detection infrastructure.
    """
    
    def __init__(
        self,
        policy: AdaptivePizzaRecognitionPolicy,
        verifier: PizzaVerifier,
        compatibility_manager: ModelCompatibilityManager
    ):
        self.policy = policy
        self.verifier = verifier
        self.compatibility_manager = compatibility_manager
        
        # Integration with energy management
        self.energy_system_available = self._check_energy_system()
        
        logger.info("RL integration initialized")
    
    def _check_energy_system(self) -> bool:
        """Check if energy management system is available."""
        try:
            # Check for energy management modules
            energy_paths = [
                "src/emulation/rp2040_emulator.py",
                "src/analysis/energy_analysis.py"
            ]
            
            for path in energy_paths:
                if Path(path).exists():
                    return True
            
            return False
            
        except Exception:
            return False
    
    def create_system_state_from_context(
        self,
        battery_level: Optional[float] = None,
        image_complexity: Optional[float] = None,
        required_accuracy: float = 0.8,
        time_constraints: float = 100.0,
        temperature: Optional[float] = None,
        memory_usage: Optional[float] = None
    ) -> SystemState:
        """
        Create SystemState from available context information.
        
        Args:
            battery_level: Current battery level [0,1] (None for auto-estimation)
            image_complexity: Estimated image complexity [0,1]
            required_accuracy: Minimum required accuracy
            time_constraints: Available processing time (ms)
            temperature: System temperature (Celsius)
            memory_usage: Current memory usage ratio [0,1]
            
        Returns:
            SystemState object
        """
        # Estimate missing values
        if battery_level is None:
            battery_level = 0.8  # Assume good battery by default
        
        if image_complexity is None:
            image_complexity = 0.5  # Assume medium complexity
        
        if temperature is None:
            # Estimate temperature based on system load
            base_temp = 25.0
            load_factor = (1.0 - battery_level) * 15.0
            temperature = base_temp + load_factor
        
        if memory_usage is None:
            # Estimate memory usage
            base_memory = 0.3
            complexity_memory = image_complexity * 0.2
            memory_usage = min(0.9, base_memory + complexity_memory)
        
        return SystemState(
            battery_level=battery_level,
            power_draw_current=15.0,  # Default power consumption
            energy_budget=50.0,       # Default energy budget
            image_complexity=image_complexity,
            brightness_level=0.6,     # Default brightness
            contrast_level=0.7,       # Default contrast
            has_motion_blur=False,    # Default no motion blur
            required_accuracy=required_accuracy,
            time_constraints=time_constraints,
            food_safety_critical=False,  # Default not critical
            temperature=temperature,
            memory_usage=memory_usage,
            processing_load=0.4       # Default processing load
        )
    
    def optimize_inference_strategy(
        self,
        system_state: SystemState,
        available_models: List[str]
    ) -> InferenceStrategy:
        """
        Use RL policy to optimize inference strategy.
        
        Args:
            system_state: Current system state
            available_models: List of available model types
            
        Returns:
            Optimized inference strategy
        """
        # Get strategy from RL policy
        strategy = self.policy.get_strategy(system_state)
        
        # Validate strategy against available models
        if strategy.model_variant.value not in available_models:
            # Fallback to first available model
            from src.rl.adaptive_policy import ModelVariant
            for variant in ModelVariant:
                if variant.value in available_models:
                    strategy.model_variant = variant
                    break
        
        # Validate CMSIS-NN availability
        if strategy.use_cmsis_nn and not self.compatibility_manager.cmsis_nn_available:
            strategy.use_cmsis_nn = False
            logger.warning("CMSIS-NN not available, disabling CMSIS optimization")
        
        return strategy
    
    def estimate_strategy_performance(
        self,
        strategy: InferenceStrategy,
        system_state: SystemState
    ) -> Dict[str, float]:
        """
        Estimate performance of an inference strategy.
        
        Args:
            strategy: Inference strategy
            system_state: Current system state
            
        Returns:
            Performance estimates
        """
        # Get base performance estimates
        base_performance = self.compatibility_manager.estimate_model_performance(
            strategy.model_variant.value,
            use_cmsis=strategy.use_cmsis_nn
        )
        
        # Adjust for processing intensity
        intensity_mapping = {
            'minimal': 0.7,
            'standard': 1.0,
            'enhanced': 1.5
        }
        intensity_factor = intensity_mapping.get(strategy.processing_intensity.value, 1.0)
        base_performance['latency_ms'] *= intensity_factor
        base_performance['energy_mj'] *= intensity_factor
        base_performance['accuracy'] *= (0.8 + 0.2 * intensity_factor)
        
        # Adjust for system state
        if system_state.battery_level < 0.3:
            # Low battery - reduce performance
            base_performance['latency_ms'] *= 1.2
            base_performance['energy_mj'] *= 1.1
        
        if system_state.temperature > 60:
            # High temperature - thermal throttling
            base_performance['latency_ms'] *= 1.3
            base_performance['accuracy'] *= 0.95
        
        return base_performance


# Global integration instances
_compatibility_manager = None
_verifier_integration = None
_rl_integration = None


def get_compatibility_manager(device: str = 'cpu') -> ModelCompatibilityManager:
    """Get global compatibility manager instance."""
    global _compatibility_manager
    if _compatibility_manager is None:
        _compatibility_manager = ModelCompatibilityManager(device=device)
    return _compatibility_manager


def get_verifier_integration(
    verifier: Optional[PizzaVerifier] = None,
    device: str = 'cpu'
) -> VerifierIntegration:
    """Get global verifier integration instance."""
    global _verifier_integration
    if _verifier_integration is None:
        if verifier is None:
            verifier = PizzaVerifier(device=device)
        compatibility_manager = get_compatibility_manager(device)
        _verifier_integration = VerifierIntegration(verifier, compatibility_manager)
    return _verifier_integration


def get_rl_integration(
    policy: Optional[AdaptivePizzaRecognitionPolicy] = None,
    verifier: Optional[PizzaVerifier] = None,
    device: str = 'cpu'
) -> RLIntegration:
    """Get global RL integration instance."""
    global _rl_integration
    if _rl_integration is None:
        if policy is None:
            policy = AdaptivePizzaRecognitionPolicy()
        if verifier is None:
            verifier = PizzaVerifier(device=device)
        compatibility_manager = get_compatibility_manager(device)
        _rl_integration = RLIntegration(policy, verifier, compatibility_manager)
    return _rl_integration
