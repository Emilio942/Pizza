#!/usr/bin/env python3
"""
Pizza Recognition Quality Data Structure

This module defines the data structures and utilities for working with
pizza-specific verifier data that includes temporal smoothing and CMSIS-NN
performance considerations.

Implements Aufgabe 1.2: Pizza-Erkennungs-Qualitätsdatenstruktur
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Define class names directly to avoid import issues
CLASS_NAMES = ["basic", "burnt", "combined", "mixed", "progression", "segment"]


class ErrorType(Enum):
    """Types of prediction errors for classification."""
    NONE = "none"
    BURNT_VS_BASIC = "burnt_vs_basic"
    COMBINED_VS_MIXED = "combined_vs_mixed"
    PROGRESSION_STAGE = "progression_stage"
    SAFETY_CRITICAL = "safety_critical"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"


class HardwarePlatform(Enum):
    """Hardware platforms for inference."""
    RP2040_EMULATOR = "rp2040_emulator"
    RP2040_HARDWARE = "rp2040_hardware"
    HOST_SIMULATION = "host_simulation"


@dataclass
class PredictionMetadata:
    """Metadata about the prediction process."""
    model_version: Optional[str] = None
    preprocessing_version: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PizzaQualityData:
    """
    Data structure for pizza recognition quality assessment.
    
    Maps [Pizza-Bild, Model-Vorhersage, Ground-Truth] to a Qualitätsscore [0.0-1.0]
    with additional considerations for temporal smoothing and CMSIS-NN performance.
    """
    pizza_image_path: str
    model_prediction: str
    ground_truth_class: str
    confidence_score: float
    quality_score: float
    
    # Enhanced features for temporal smoothing and CMSIS-NN
    temporal_smoothing_applied: bool = False
    temporal_smoothing_factor: Optional[float] = None
    cmsis_nn_optimized: bool = False
    inference_time_ms: Optional[float] = None
    energy_consumption_mj: Optional[float] = None
    hardware_platform: HardwarePlatform = HardwarePlatform.HOST_SIMULATION
    
    # Additional metadata
    prediction_metadata: Optional[PredictionMetadata] = None
    food_safety_critical: bool = False
    error_type: ErrorType = ErrorType.NONE
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Validate class names
        if self.model_prediction not in CLASS_NAMES:
            raise ValueError(f"Invalid model_prediction: {self.model_prediction}. Must be one of {CLASS_NAMES}")
        
        if self.ground_truth_class not in CLASS_NAMES:
            raise ValueError(f"Invalid ground_truth_class: {self.ground_truth_class}. Must be one of {CLASS_NAMES}")
        
        # Validate scores
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}")
        
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(f"quality_score must be between 0.0 and 1.0, got {self.quality_score}")
        
        if self.temporal_smoothing_factor is not None:
            if not (0.0 <= self.temporal_smoothing_factor <= 1.0):
                raise ValueError(f"temporal_smoothing_factor must be between 0.0 and 1.0, got {self.temporal_smoothing_factor}")
        
        # Auto-detect food safety critical cases
        if not self.food_safety_critical:
            self.food_safety_critical = self._is_food_safety_critical()
        
        # Initialize metadata if not provided
        if self.prediction_metadata is None:
            self.prediction_metadata = PredictionMetadata()
    
    def _is_food_safety_critical(self) -> bool:
        """
        Determine if this prediction involves food safety critical decisions.
        
        Returns:
            bool: True if the prediction involves food safety considerations
        """
        # Raw vs cooked detection is critical
        safety_critical_pairs = [
            ("basic", "burnt"),
            ("basic", "combined"),
            ("basic", "mixed"),
            ("basic", "progression")
        ]
        
        prediction_pair = (self.ground_truth_class, self.model_prediction)
        reverse_pair = (self.model_prediction, self.ground_truth_class)
        
        return prediction_pair in safety_critical_pairs or reverse_pair in safety_critical_pairs
    
    def calculate_quality_score(self) -> float:
        """
        Calculate quality score based on various factors.
        
        Returns:
            float: Calculated quality score between 0.0 and 1.0
        """
        base_score = 1.0 if self.model_prediction == self.ground_truth_class else 0.0
        
        # Adjust based on confidence
        confidence_factor = self.confidence_score
        
        # Penalty for food safety critical errors
        safety_penalty = 0.0
        if self.food_safety_critical and base_score < 1.0:
            safety_penalty = 0.3
        
        # Bonus for temporal consistency
        temporal_bonus = 0.0
        if self.temporal_smoothing_applied and base_score > 0.5:
            temporal_bonus = 0.1
        
        # CMSIS-NN optimization consideration
        cmsis_factor = 1.0
        if self.cmsis_nn_optimized and self.inference_time_ms is not None:
            # Bonus for efficient inference
            if self.inference_time_ms < 50:  # ms
                cmsis_factor = 1.05
        
        quality_score = (base_score * confidence_factor + temporal_bonus) * cmsis_factor - safety_penalty
        return max(0.0, min(1.0, quality_score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        
        # Convert enums to strings
        result['hardware_platform'] = self.hardware_platform.value
        result['error_type'] = self.error_type.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PizzaQualityData':
        """Create instance from dictionary."""
        # Convert string enums back to enum objects
        if 'hardware_platform' in data:
            data['hardware_platform'] = HardwarePlatform(data['hardware_platform'])
        
        if 'error_type' in data:
            data['error_type'] = ErrorType(data['error_type'])
        
        # Handle metadata
        if 'prediction_metadata' in data and data['prediction_metadata']:
            data['prediction_metadata'] = PredictionMetadata(**data['prediction_metadata'])
        
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PizzaQualityData':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class PizzaQualityDataset:
    """
    Collection of pizza quality data samples.
    """
    
    def __init__(self):
        self.samples: List[PizzaQualityData] = []
    
    def add_sample(self, sample: PizzaQualityData):
        """Add a quality data sample."""
        self.samples.append(sample)
    
    def add_samples(self, samples: List[PizzaQualityData]):
        """Add multiple quality data samples."""
        self.samples.extend(samples)
    
    def filter_by_platform(self, platform: HardwarePlatform) -> 'PizzaQualityDataset':
        """Filter samples by hardware platform."""
        filtered = PizzaQualityDataset()
        filtered.samples = [s for s in self.samples if s.hardware_platform == platform]
        return filtered
    
    def filter_by_error_type(self, error_type: ErrorType) -> 'PizzaQualityDataset':
        """Filter samples by error type."""
        filtered = PizzaQualityDataset()
        filtered.samples = [s for s in self.samples if s.error_type == error_type]
        return filtered
    
    def filter_food_safety_critical(self) -> 'PizzaQualityDataset':
        """Filter samples that are food safety critical."""
        filtered = PizzaQualityDataset()
        filtered.samples = [s for s in self.samples if s.food_safety_critical]
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.samples:
            return {}
        
        return {
            "total_samples": len(self.samples),
            "average_quality_score": sum(s.quality_score for s in self.samples) / len(self.samples),
            "average_confidence": sum(s.confidence_score for s in self.samples) / len(self.samples),
            "temporal_smoothing_samples": sum(1 for s in self.samples if s.temporal_smoothing_applied),
            "cmsis_nn_samples": sum(1 for s in self.samples if s.cmsis_nn_optimized),
            "food_safety_critical_samples": sum(1 for s in self.samples if s.food_safety_critical),
            "platform_distribution": self._get_platform_distribution(),
            "error_type_distribution": self._get_error_type_distribution(),
            "class_distribution": self._get_class_distribution()
        }
    
    def _get_platform_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by platform."""
        distribution = {}
        for sample in self.samples:
            platform = sample.hardware_platform.value
            distribution[platform] = distribution.get(platform, 0) + 1
        return distribution
    
    def _get_error_type_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by error type."""
        distribution = {}
        for sample in self.samples:
            error_type = sample.error_type.value
            distribution[error_type] = distribution.get(error_type, 0) + 1
        return distribution
    
    def _get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of ground truth vs predicted classes."""
        distribution = {"ground_truth": {}, "predictions": {}}
        
        for sample in self.samples:
            # Ground truth distribution
            gt = sample.ground_truth_class
            distribution["ground_truth"][gt] = distribution["ground_truth"].get(gt, 0) + 1
            
            # Prediction distribution
            pred = sample.model_prediction
            distribution["predictions"][pred] = distribution["predictions"].get(pred, 0) + 1
        
        return distribution
    
    def save_to_json(self, filepath: str):
        """Save dataset to JSON file."""
        data = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "samples": [sample.to_dict() for sample in self.samples]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'PizzaQualityDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls()
        if 'samples' in data:
            dataset.samples = [PizzaQualityData.from_dict(sample_data) for sample_data in data['samples']]
        
        return dataset


def create_sample_data():
    """Create sample pizza quality data for testing."""
    samples = []
    
    # Positive example - correct prediction
    samples.append(PizzaQualityData(
        pizza_image_path="test_images/basic_001.jpg",
        model_prediction="basic",
        ground_truth_class="basic",
        confidence_score=0.95,
        quality_score=0.95,
        temporal_smoothing_applied=True,
        temporal_smoothing_factor=0.8,
        cmsis_nn_optimized=True,
        inference_time_ms=45.2,
        energy_consumption_mj=12.5,
        hardware_platform=HardwarePlatform.RP2040_EMULATOR,
        prediction_metadata=PredictionMetadata(
            model_version="MicroPizzaNetV2_1.0",
            preprocessing_version="1.2",
            session_id="test_session_001"
        )
    ))
    
    # Hard negative example - food safety critical error
    samples.append(PizzaQualityData(
        pizza_image_path="test_images/basic_002.jpg",
        model_prediction="burnt",
        ground_truth_class="basic",
        confidence_score=0.72,
        quality_score=0.1,  # Low quality for safety-critical error
        food_safety_critical=True,
        error_type=ErrorType.SAFETY_CRITICAL,
        hardware_platform=HardwarePlatform.RP2040_HARDWARE,
        inference_time_ms=52.1,
        energy_consumption_mj=15.2
    ))
    
    return samples


if __name__ == "__main__":
    # Test the data structure
    print("Testing Pizza Quality Data Structure...")
    
    # Create sample data
    samples = create_sample_data()
    
    # Create dataset
    dataset = PizzaQualityDataset()
    dataset.add_samples(samples)
    
    # Print statistics
    stats = dataset.get_statistics()
    print(f"Dataset Statistics: {json.dumps(stats, indent=2)}")
    
    # Test JSON serialization
    test_sample = samples[0]
    json_str = test_sample.to_json()
    print(f"Sample JSON: {json_str}")
    
    # Test JSON deserialization
    restored_sample = PizzaQualityData.from_json(json_str)
    print(f"Restored sample quality score: {restored_sample.quality_score}")
    
    print("✅ Pizza Quality Data Structure implementation completed successfully!")
