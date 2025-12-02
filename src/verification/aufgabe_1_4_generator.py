#!/usr/bin/env python3
"""
Aufgabe 1.4: Generierung von "Pizza-spezifischen Hard Negatives" (40%)

This module generates challenging pizza-specific recognition errors for robust verifier training.
Focus on subtle errors that are difficult to detect but critical for pizza recognition quality.
"""

import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

# Pizza Quality Data Structures (standalone version)
class ErrorType(Enum):
    MISCLASSIFICATION = "misclassification"
    LOW_CONFIDENCE = "low_confidence"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    FOOD_SAFETY_CRITICAL = "food_safety_critical"

class HardwarePlatform(Enum):
    RP2040 = "rp2040"
    JETSON_NANO = "jetson_nano"
    ESP32 = "esp32"
    ARDUINO_NANO33 = "arduino_nano33"
    GENERIC = "generic"

@dataclass
class PizzaQualityData:
    pizza_image_path: str
    model_prediction: str
    ground_truth_class: str
    confidence_score: float
    quality_score: float
    temporal_smoothing_applied: bool = False
    cmsis_nn_optimized: bool = False
    inference_time_ms: float = 0.0
    energy_consumption_mj: float = 0.0
    hardware_platform: HardwarePlatform = HardwarePlatform.GENERIC
    prediction_metadata: Dict[str, Any] = None
    food_safety_critical: bool = False
    error_type: Optional[ErrorType] = None
    
    def __post_init__(self):
        if self.prediction_metadata is None:
            self.prediction_metadata = {}

class PizzaVerifierDataset:
    def __init__(self, samples: List[PizzaQualityData]):
        self.samples = samples
    
    def get_class_distribution(self) -> Dict[str, int]:
        dist = {}
        for sample in self.samples:
            dist[sample.model_prediction] = dist.get(sample.model_prediction, 0) + 1
        return dist
    
    def get_quality_statistics(self) -> Dict[str, float]:
        if not self.samples:
            return {}
        quality_scores = [s.quality_score for s in self.samples]
        return {
            "mean": round(sum(quality_scores) / len(quality_scores), 4),
            "min": round(min(quality_scores), 4),
            "max": round(max(quality_scores), 4),
            "count": len(quality_scores)
        }

class PizzaHardNegativesGenerator:
    """Generates challenging pizza-specific recognition errors for verifier training."""
    
    # Pizza class mapping
    CLASS_NAMES = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']
    
    # Critical confusion pairs for food safety
    FOOD_SAFETY_CRITICAL_CONFUSIONS = [
        ('basic', 'burnt'),      # Raw vs cooked detection critical
        ('burnt', 'basic'),      # Overcooked vs undercooked
        ('progression', 'basic'), # Cooking progress assessment
    ]
    
    # Common pizza classification confusions
    COMMON_CONFUSIONS = [
        ('combined', 'mixed'),    # Similar multi-ingredient pizzas
        ('mixed', 'combined'),    # Reverse confusion
        ('segment', 'progression'), # Spatial vs temporal features
        ('progression', 'segment'), # Reverse
        ('basic', 'combined'),    # Simple vs complex toppings
        ('combined', 'basic'),    # Reverse
    ]
    
    def __init__(self, project_root: str = "/home/emilio/Documents/ai/pizza"):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / "output" / "verification_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load diffusion analysis data for base images
        self.diffusion_analysis_file = self.project_root / "output" / "diffusion_analysis" / "full_synthetic_evaluation_20250524_025813.json"
        
        # Load existing positive examples for contrast
        self.positive_examples_file = self.output_dir / "pizza_positive_examples_comprehensive.json"
        
    def load_diffusion_analysis_data(self) -> Dict[str, Any]:
        """Load the comprehensive diffusion analysis evaluation data."""
        try:
            with open(self.diffusion_analysis_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Diffusion analysis file not found: {self.diffusion_analysis_file}")
            return {"detailed_metrics": []}
    
    def load_positive_examples(self) -> List[Dict[str, Any]]:
        """Load existing positive examples for reference."""
        try:
            with open(self.positive_examples_file, 'r') as f:
                data = json.load(f)
                return data.get('samples', [])
        except FileNotFoundError:
            print(f"Warning: Positive examples file not found: {self.positive_examples_file}")
            return []
    
    def generate_food_safety_critical_error(self, base_metrics: Dict[str, Any]) -> PizzaQualityData:
        """Generate a food safety critical error (e.g., burnt vs basic confusion)."""
        
        # Select a critical confusion pair
        ground_truth, wrong_prediction = random.choice(self.FOOD_SAFETY_CRITICAL_CONFUSIONS)
        
        # Generate characteristics that could lead to confusion
        if ground_truth == 'basic' and wrong_prediction == 'burnt':
            # Model incorrectly predicts burnt when pizza is basic (false alarm)
            burn_level = random.uniform(0.1, 0.3)  # Actually low burn
            confidence = random.uniform(0.6, 0.8)  # Moderate confidence in wrong prediction
            quality_score = random.uniform(0.1, 0.3)  # Very low quality
        elif ground_truth == 'burnt' and wrong_prediction == 'basic':
            # Model incorrectly predicts basic when pizza is burnt (dangerous miss)
            burn_level = random.uniform(0.7, 0.9)   # Actually high burn
            confidence = random.uniform(0.5, 0.7)   # Low confidence but still wrong
            quality_score = random.uniform(0.0, 0.2)  # Extremely low quality - dangerous
        else:
            # Progression vs basic confusion
            burn_level = random.uniform(0.4, 0.6)   # Intermediate burn level
            confidence = random.uniform(0.6, 0.8)   # Moderate confidence
            quality_score = random.uniform(0.1, 0.3)  # Low quality
        
        return PizzaQualityData(
            pizza_image_path=f"data/synthetic/hard_negative_{base_metrics.get('filename', 'unknown')}.jpg",
            model_prediction=wrong_prediction,
            ground_truth_class=ground_truth,
            confidence_score=confidence,
            quality_score=quality_score,
            temporal_smoothing_applied=random.choice([True, False]),
            cmsis_nn_optimized=random.choice([True, False]),
            inference_time_ms=round(random.uniform(80, 250), 2),  # Longer inference for difficult cases
            energy_consumption_mj=round(random.uniform(1.0, 3.0), 3),
            hardware_platform=random.choice(list(HardwarePlatform)),
            prediction_metadata={
                "burn_level_consistency": burn_level,
                "confusion_type": "food_safety_critical",
                "ground_truth_class": ground_truth,
                "predicted_class": wrong_prediction,
                "original_pizza_likelihood": base_metrics.get('pizza_likelihood', 0.5),
                "texture_quality": base_metrics.get('texture_quality', 0.5),
                "source": "hard_negative_generation"
            },
            food_safety_critical=True,  # Mark as critical
            error_type=ErrorType.FOOD_SAFETY_CRITICAL
        )
    
    def generate_subtle_misclassification(self, base_metrics: Dict[str, Any]) -> PizzaQualityData:
        """Generate subtle misclassification between similar pizza classes."""
        
        # Select a common confusion pair
        ground_truth, wrong_prediction = random.choice(self.COMMON_CONFUSIONS)
        
        # Generate characteristics that make the error subtle
        pizza_likelihood = max(0.7, base_metrics.get('pizza_likelihood', 0.8))  # High pizza likelihood
        texture_quality = base_metrics.get('texture_quality', 0.6)
        
        # Confidence should be high (makes it a hard negative)
        confidence = random.uniform(0.75, 0.92)
        
        # Quality score moderate to low (harder to detect error)
        quality_score = random.uniform(0.2, 0.5)
        
        return PizzaQualityData(
            pizza_image_path=f"data/synthetic/subtle_{base_metrics.get('filename', 'unknown')}.jpg",
            model_prediction=wrong_prediction,
            ground_truth_class=ground_truth,
            confidence_score=confidence,
            quality_score=quality_score,
            temporal_smoothing_applied=random.choice([True, False]),
            cmsis_nn_optimized=random.choice([True, False]),
            inference_time_ms=round(random.uniform(60, 180), 2),
            energy_consumption_mj=round(random.uniform(0.8, 2.5), 3),
            hardware_platform=random.choice(list(HardwarePlatform)),
            prediction_metadata={
                "pizza_likelihood": pizza_likelihood,
                "texture_quality": texture_quality,
                "confusion_type": "subtle_misclassification",
                "ground_truth_class": ground_truth,
                "predicted_class": wrong_prediction,
                "brightness": base_metrics.get('brightness', 50),
                "contrast": base_metrics.get('contrast', 50),
                "source": "hard_negative_generation"
            },
            food_safety_critical=False,
            error_type=ErrorType.MISCLASSIFICATION
        )
    
    def generate_low_confidence_error(self, base_metrics: Dict[str, Any]) -> PizzaQualityData:
        """Generate low confidence prediction errors."""
        
        # Random class confusion
        ground_truth = random.choice(self.CLASS_NAMES)
        wrong_prediction = random.choice([c for c in self.CLASS_NAMES if c != ground_truth])
        
        # Low confidence but still wrong
        confidence = random.uniform(0.45, 0.65)  # Around threshold area
        quality_score = random.uniform(0.15, 0.4)  # Low quality due to uncertainty
        
        return PizzaQualityData(
            pizza_image_path=f"data/synthetic/low_conf_{base_metrics.get('filename', 'unknown')}.jpg",
            model_prediction=wrong_prediction,
            ground_truth_class=ground_truth,
            confidence_score=confidence,
            quality_score=quality_score,
            temporal_smoothing_applied=random.choice([True, False]),
            cmsis_nn_optimized=random.choice([True, False]),
            inference_time_ms=round(random.uniform(90, 220), 2),
            energy_consumption_mj=round(random.uniform(1.2, 2.8), 3),
            hardware_platform=random.choice(list(HardwarePlatform)),
            prediction_metadata={
                "confidence_range": "low",
                "uncertainty": True,
                "confusion_type": "low_confidence",
                "ground_truth_class": ground_truth,
                "predicted_class": wrong_prediction,
                "pizza_likelihood": base_metrics.get('pizza_likelihood', 0.6),
                "source": "hard_negative_generation"
            },
            food_safety_critical=False,
            error_type=ErrorType.LOW_CONFIDENCE
        )
    
    def generate_temporal_inconsistency_error(self, base_metrics: Dict[str, Any]) -> PizzaQualityData:
        """Generate temporal inconsistency errors (progression class specific)."""
        
        # Focus on progression-related errors
        if random.random() < 0.5:
            ground_truth = 'progression'
            wrong_prediction = random.choice(['basic', 'burnt', 'combined'])
        else:
            ground_truth = random.choice(['basic', 'burnt'])
            wrong_prediction = 'progression'
        
        confidence = random.uniform(0.65, 0.85)  # Moderate to high confidence
        quality_score = random.uniform(0.1, 0.35)  # Low quality due to temporal issues
        
        return PizzaQualityData(
            pizza_image_path=f"data/synthetic/temporal_{base_metrics.get('filename', 'unknown')}.jpg",
            model_prediction=wrong_prediction,
            ground_truth_class=ground_truth,
            confidence_score=confidence,
            quality_score=quality_score,
            temporal_smoothing_applied=True,  # Temporal smoothing was applied but failed
            cmsis_nn_optimized=random.choice([True, False]),
            inference_time_ms=round(random.uniform(100, 280), 2),
            energy_consumption_mj=round(random.uniform(1.5, 3.5), 3),
            hardware_platform=random.choice(list(HardwarePlatform)),
            prediction_metadata={
                "temporal_context": "inconsistent",
                "confusion_type": "temporal_inconsistency",
                "ground_truth_class": ground_truth,
                "predicted_class": wrong_prediction,
                "burn_level_consistency": base_metrics.get('burn_level_consistency', 0.5),
                "texture_quality": base_metrics.get('texture_quality', 0.5),
                "source": "hard_negative_generation"
            },
            food_safety_critical=(ground_truth in ['basic', 'burnt'] or wrong_prediction in ['basic', 'burnt']),
            error_type=ErrorType.TEMPORAL_INCONSISTENCY
        )
    
    def generate_cmsis_nn_vs_standard_discrepancy(self, base_metrics: Dict[str, Any]) -> PizzaQualityData:
        """Generate errors from CMSIS-NN vs standard model discrepancies."""
        
        # Different predictions between CMSIS-NN (edge) and standard model
        ground_truth = random.choice(self.CLASS_NAMES)
        wrong_prediction = random.choice([c for c in self.CLASS_NAMES if c != ground_truth])
        
        # Moderate confidence (optimization artifacts)
        confidence = random.uniform(0.6, 0.8)
        quality_score = random.uniform(0.15, 0.35)  # Low due to model discrepancy
        
        return PizzaQualityData(
            pizza_image_path=f"data/synthetic/cmsis_discrepancy_{base_metrics.get('filename', 'unknown')}.jpg",
            model_prediction=wrong_prediction,
            ground_truth_class=ground_truth,
            confidence_score=confidence,
            quality_score=quality_score,
            temporal_smoothing_applied=random.choice([True, False]),
            cmsis_nn_optimized=True,  # Error comes from CMSIS-NN optimization
            inference_time_ms=round(random.uniform(30, 80), 2),  # Faster but less accurate
            energy_consumption_mj=round(random.uniform(0.3, 1.0), 3),  # Lower energy consumption
            hardware_platform=HardwarePlatform.RP2040,  # Embedded platform
            prediction_metadata={
                "model_type": "cmsis_nn",
                "optimization_artifacts": True,
                "confusion_type": "model_discrepancy",
                "ground_truth_class": ground_truth,
                "predicted_class": wrong_prediction,
                "quantization_error": True,
                "source": "hard_negative_generation"
            },
            food_safety_critical=(ground_truth in ['basic', 'burnt'] or wrong_prediction in ['basic', 'burnt']),
            error_type=ErrorType.MISCLASSIFICATION
        )
    
    def generate_hard_negatives(self, target_count: int = 100) -> List[PizzaQualityData]:
        """
        Generate challenging pizza-specific hard negative examples.
        
        Args:
            target_count: Target number of hard negative examples (40% of total dataset)
            
        Returns:
            List of challenging hard negative examples
        """
        hard_negatives = []
        
        print(f"🔥 Generating pizza-specific hard negative examples...")
        
        # Load base data for realistic image characteristics
        diffusion_data = self.load_diffusion_analysis_data()
        base_images = diffusion_data.get('detailed_metrics', [])
        
        if not base_images:
            print("Warning: No base images found, generating synthetic characteristics")
            base_images = [{'filename': f'synthetic_{i}.jpg'} for i in range(target_count)]
        
        # Distribution of hard negative types
        food_safety_count = int(target_count * 0.4)    # 40% food safety critical
        subtle_count = int(target_count * 0.3)         # 30% subtle misclassifications
        low_conf_count = int(target_count * 0.15)      # 15% low confidence errors
        temporal_count = int(target_count * 0.1)       # 10% temporal inconsistency
        cmsis_count = target_count - (food_safety_count + subtle_count + low_conf_count + temporal_count)  # Remainder
        
        print(f"   Distribution: {food_safety_count} food safety, {subtle_count} subtle, {low_conf_count} low confidence, {temporal_count} temporal, {cmsis_count} CMSIS-NN")
        
        # Sample base images for generation
        sampled_images = random.sample(base_images, min(target_count, len(base_images)))
        if len(sampled_images) < target_count:
            # Repeat sampling if needed
            while len(sampled_images) < target_count:
                sampled_images.extend(random.sample(base_images, min(target_count - len(sampled_images), len(base_images))))
        
        # Generate food safety critical errors
        print("⚠️  Generating food safety critical errors...")
        for i in range(food_safety_count):
            base_metrics = sampled_images[i % len(sampled_images)]
            hard_negative = self.generate_food_safety_critical_error(base_metrics)
            hard_negatives.append(hard_negative)
        
        # Generate subtle misclassifications
        print("🎯 Generating subtle misclassifications...")
        for i in range(subtle_count):
            base_metrics = sampled_images[(food_safety_count + i) % len(sampled_images)]
            hard_negative = self.generate_subtle_misclassification(base_metrics)
            hard_negatives.append(hard_negative)
        
        # Generate low confidence errors
        print("❓ Generating low confidence errors...")
        for i in range(low_conf_count):
            base_metrics = sampled_images[(food_safety_count + subtle_count + i) % len(sampled_images)]
            hard_negative = self.generate_low_confidence_error(base_metrics)
            hard_negatives.append(hard_negative)
        
        # Generate temporal inconsistency errors
        print("⏱️  Generating temporal inconsistency errors...")
        for i in range(temporal_count):
            base_metrics = sampled_images[(food_safety_count + subtle_count + low_conf_count + i) % len(sampled_images)]
            hard_negative = self.generate_temporal_inconsistency_error(base_metrics)
            hard_negatives.append(hard_negative)
        
        # Generate CMSIS-NN discrepancy errors
        print("🔧 Generating CMSIS-NN optimization errors...")
        for i in range(cmsis_count):
            base_metrics = sampled_images[(food_safety_count + subtle_count + low_conf_count + temporal_count + i) % len(sampled_images)]
            hard_negative = self.generate_cmsis_nn_vs_standard_discrepancy(base_metrics)
            hard_negatives.append(hard_negative)
        
        print(f"✨ Generated {len(hard_negatives)} hard negative examples")
        
        return hard_negatives
    
    def save_hard_negatives(self, hard_negatives: List[PizzaQualityData]) -> str:
        """Save hard negative examples to JSON file."""
        
        # Create dataset
        dataset = PizzaVerifierDataset(hard_negatives)
        
        # Count error types
        error_type_dist = {}
        food_safety_count = 0
        for example in hard_negatives:
            error_type = example.error_type.value if example.error_type else "unknown"
            error_type_dist[error_type] = error_type_dist.get(error_type, 0) + 1
            if example.food_safety_critical:
                food_safety_count += 1
        
        # Save to file
        output_file = self.output_dir / "pizza_hard_negatives_comprehensive.json"
        
        # Convert dataclasses to dict with enum handling
        serializable_samples = []
        for example in hard_negatives:
            sample_dict = asdict(example)
            # Convert enums to strings
            if 'hardware_platform' in sample_dict:
                sample_dict['hardware_platform'] = sample_dict['hardware_platform'].value
            if 'error_type' in sample_dict and sample_dict['error_type'] is not None:
                sample_dict['error_type'] = sample_dict['error_type'].value
            serializable_samples.append(sample_dict)
        
        with open(output_file, 'w') as f:
            json.dump({
                "dataset_info": {
                    "task": "Aufgabe 1.4: Generierung von Pizza-spezifischen Hard Negatives",
                    "description": "Challenging pizza recognition errors for robust verifier training",
                    "total_samples": len(hard_negatives),
                    "food_safety_critical_count": food_safety_count,
                    "error_type_distribution": error_type_dist,
                    "quality_threshold": "<=0.5 (hard negatives)",
                    "class_distribution": dataset.get_class_distribution(),
                    "quality_statistics": dataset.get_quality_statistics()
                },
                "samples": serializable_samples
            }, f, indent=2)
        
        print(f"💾 Saved {len(hard_negatives)} hard negative examples to: {output_file}")
        print(f"📈 Dataset statistics:")
        print(f"   Class distribution: {dataset.get_class_distribution()}")
        print(f"   Quality statistics: {dataset.get_quality_statistics()}")
        print(f"   Error type distribution: {error_type_dist}")
        print(f"   Food safety critical: {food_safety_count}/{len(hard_negatives)} ({100*food_safety_count/len(hard_negatives):.1f}%)")
        
        return str(output_file)

def main():
    """Main execution for Aufgabe 1.4"""
    print("🚀 Starting Aufgabe 1.4: Generierung von Pizza-spezifischen Hard Negatives")
    
    generator = PizzaHardNegativesGenerator()
    
    # Generate hard negatives (targeting 40% of total dataset)
    hard_negatives = generator.generate_hard_negatives(target_count=63)  # 40% of ~160 total samples
    
    # Save to file
    output_file = generator.save_hard_negatives(hard_negatives)
    
    print(f"✅ Aufgabe 1.4 completed successfully!")
    print(f"📄 Output file: {output_file}")
    print(f"🔄 Next task: Aufgabe 2.1 - Vorbereitung des gesamten Pizza-Verifier-Datensatzes")

if __name__ == "__main__":
    main()
