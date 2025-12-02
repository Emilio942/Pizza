#!/usr/bin/env python3
"""
Pizza Quality Data Integration Utilities

This module provides utilities to integrate the pizza quality data structure
with the existing verification system and prepare data for the verifier training.

Part of Aufgabe 1.2: Pizza-Erkennungs-Qualitätsdatenstruktur
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Define required classes inline to avoid import issues
CLASS_NAMES = ["basic", "burnt", "combined", "mixed", "progression", "segment"]

# Import after ensuring path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

try:
    from src.verification.pizza_quality_data import (
        PizzaQualityData, PizzaQualityDataset, ErrorType, HardwarePlatform, PredictionMetadata
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PizzaQualityDataGenerator:
    """
    Generator for pizza quality data from existing model predictions and evaluations.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parents[2]
        self.output_dir = self.project_root / "output" / "verification_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_test_results(self, test_results_path: str) -> PizzaQualityDataset:
        """
        Generate quality data from existing test results.
        
        Args:
            test_results_path: Path to test results JSON file
            
        Returns:
            PizzaQualityDataset with converted data
        """
        dataset = PizzaQualityDataset()
        
        try:
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
            
            # Extract predictions from test results
            if 'predictions' in test_results:
                for prediction in test_results['predictions']:
                    quality_data = self._convert_prediction_to_quality_data(prediction)
                    if quality_data:
                        dataset.add_sample(quality_data)
            
            logger.info(f"Generated {len(dataset.samples)} quality data samples from {test_results_path}")
            
        except Exception as e:
            logger.error(f"Error processing test results {test_results_path}: {e}")
        
        return dataset
    
    def _convert_prediction_to_quality_data(self, prediction: Dict) -> Optional[PizzaQualityData]:
        """
        Convert a single prediction to quality data format.
        
        Args:
            prediction: Dictionary containing prediction data
            
        Returns:
            PizzaQualityData object or None if conversion fails
        """
        try:
            # Extract required fields
            image_path = prediction.get('image_path', '')
            model_pred = prediction.get('prediction', '')
            ground_truth = prediction.get('ground_truth', '')
            confidence = prediction.get('confidence', 0.0)
            
            # Calculate quality score based on correctness and confidence
            is_correct = model_pred == ground_truth
            base_quality = 1.0 if is_correct else 0.0
            quality_score = base_quality * confidence
            
            # Determine if this is a food safety critical case
            food_safety_critical = self._is_food_safety_critical(ground_truth, model_pred)
            
            # Determine error type
            error_type = self._determine_error_type(ground_truth, model_pred)
            
            # Create quality data object
            quality_data = PizzaQualityData(
                pizza_image_path=image_path,
                model_prediction=model_pred,
                ground_truth_class=ground_truth,
                confidence_score=confidence,
                quality_score=quality_score,
                food_safety_critical=food_safety_critical,
                error_type=error_type,
                hardware_platform=HardwarePlatform.HOST_SIMULATION,
                prediction_metadata=PredictionMetadata(
                    model_version="MicroPizzaNetV2",
                    preprocessing_version="1.0"
                )
            )
            
            return quality_data
            
        except Exception as e:
            logger.warning(f"Failed to convert prediction to quality data: {e}")
            return None
    
    def _is_food_safety_critical(self, ground_truth: str, prediction: str) -> bool:
        """
        Determine if a prediction error is food safety critical.
        
        Args:
            ground_truth: True class
            prediction: Predicted class
            
        Returns:
            bool: True if food safety critical
        """
        # Critical cases: confusing raw/undercooked with cooked states
        critical_confusions = [
            ("basic", "burnt"),
            ("basic", "combined"),
            ("basic", "mixed"),
            ("basic", "progression")
        ]
        
        pair = (ground_truth, prediction)
        reverse_pair = (prediction, ground_truth)
        
        return pair in critical_confusions or reverse_pair in critical_confusions
    
    def _determine_error_type(self, ground_truth: str, prediction: str) -> ErrorType:
        """
        Determine the type of prediction error.
        
        Args:
            ground_truth: True class
            prediction: Predicted class
            
        Returns:
            ErrorType: Type of error
        """
        if ground_truth == prediction:
            return ErrorType.NONE
        
        # Food safety critical errors
        if self._is_food_safety_critical(ground_truth, prediction):
            return ErrorType.SAFETY_CRITICAL
        
        # Specific error patterns
        if {ground_truth, prediction} == {"burnt", "basic"}:
            return ErrorType.BURNT_VS_BASIC
        elif {ground_truth, prediction} == {"combined", "mixed"}:
            return ErrorType.COMBINED_VS_MIXED
        elif "progression" in {ground_truth, prediction}:
            return ErrorType.PROGRESSION_STAGE
        else:
            return ErrorType.SAFETY_CRITICAL  # Default for unmatched errors
    
    def generate_positive_examples(self) -> PizzaQualityDataset:
        """
        Generate positive examples from verified pizza recognition data.
        
        Returns:
            PizzaQualityDataset with positive examples
        """
        dataset = PizzaQualityDataset()
        
        # Look for test results in various locations
        test_result_paths = [
            self.project_root / "output" / "evaluation",
            self.project_root / "results",
            self.project_root / "test_data"
        ]
        
        for test_path in test_result_paths:
            if test_path.exists():
                for json_file in test_path.glob("*.json"):
                    try:
                        partial_dataset = self.generate_from_test_results(str(json_file))
                        # Filter for high-quality examples (quality_score > 0.7)
                        positive_samples = [s for s in partial_dataset.samples if s.quality_score > 0.7]
                        dataset.add_samples(positive_samples)
                    except Exception as e:
                        logger.warning(f"Failed to process {json_file}: {e}")
        
        logger.info(f"Generated {len(dataset.samples)} positive examples")
        return dataset
    
    def generate_hard_negatives(self, base_dataset: PizzaQualityDataset) -> PizzaQualityDataset:
        """
        Generate hard negative examples from base dataset.
        
        Args:
            base_dataset: Base dataset to derive negatives from
            
        Returns:
            PizzaQualityDataset with hard negative examples
        """
        hard_negatives = PizzaQualityDataset()
        
        for sample in base_dataset.samples:
            # Create hard negatives by introducing specific errors
            negative_samples = self._create_hard_negatives_for_sample(sample)
            hard_negatives.add_samples(negative_samples)
        
        logger.info(f"Generated {len(hard_negatives.samples)} hard negative examples")
        return hard_negatives
    
    def _create_hard_negatives_for_sample(self, sample: PizzaQualityData) -> List[PizzaQualityData]:
        """
        Create hard negative examples from a single sample.
        
        Args:
            sample: Original quality data sample
            
        Returns:
            List of hard negative samples
        """
        negatives = []
        gt_class = sample.ground_truth_class
        
        # Define hard confusion patterns
        confusion_patterns = {
            "basic": ["burnt", "combined"],  # Critical safety errors
            "burnt": ["basic"],  # Overcooking detection failure
            "combined": ["mixed"],  # State confusion
            "mixed": ["combined"],  # State confusion
            "progression": ["basic", "burnt"],  # Stage confusion
            "segment": ["mixed"]  # Segmentation confusion
        }
        
        if gt_class in confusion_patterns:
            for wrong_prediction in confusion_patterns[gt_class]:
                # Create hard negative with low quality score
                negative = PizzaQualityData(
                    pizza_image_path=sample.pizza_image_path,
                    model_prediction=wrong_prediction,
                    ground_truth_class=gt_class,
                    confidence_score=0.8,  # High confidence but wrong prediction
                    quality_score=0.1,  # Low quality score
                    food_safety_critical=self._is_food_safety_critical(gt_class, wrong_prediction),
                    error_type=self._determine_error_type(gt_class, wrong_prediction),
                    hardware_platform=sample.hardware_platform,
                    prediction_metadata=PredictionMetadata(
                        model_version="MicroPizzaNetV2_corrupted",
                        preprocessing_version="1.0"
                    )
                )
                negatives.append(negative)
        
        return negatives
    
    def save_datasets(self, positive_dataset: PizzaQualityDataset, 
                     negative_dataset: PizzaQualityDataset):
        """
        Save positive and negative datasets to JSON files.
        
        Args:
            positive_dataset: Dataset with positive examples
            negative_dataset: Dataset with hard negative examples
        """
        # Save positive examples
        positive_path = self.output_dir / "pizza_positive_examples.json"
        positive_dataset.save_to_json(str(positive_path))
        logger.info(f"Saved {len(positive_dataset.samples)} positive examples to {positive_path}")
        
        # Save hard negatives
        negative_path = self.output_dir / "pizza_hard_negatives.json"
        negative_dataset.save_to_json(str(negative_path))
        logger.info(f"Saved {len(negative_dataset.samples)} hard negatives to {negative_path}")
    
    def validate_schema_compliance(self, dataset: PizzaQualityDataset) -> Dict:
        """
        Validate that dataset samples comply with the schema.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            "valid_samples": 0,
            "invalid_samples": 0,
            "errors": []
        }
        
        for i, sample in enumerate(dataset.samples):
            try:
                # Test serialization/deserialization
                json_str = sample.to_json()
                restored = PizzaQualityData.from_json(json_str)
                validation_results["valid_samples"] += 1
            except Exception as e:
                validation_results["invalid_samples"] += 1
                validation_results["errors"].append(f"Sample {i}: {str(e)}")
        
        return validation_results


def main():
    """
    Main function to demonstrate the pizza quality data integration.
    """
    print("🍕 Pizza Quality Data Integration Demo")
    
    generator = PizzaQualityDataGenerator()
    
    # Generate positive examples
    print("\n1. Generating positive examples...")
    positive_dataset = generator.generate_positive_examples()
    
    # Generate hard negatives
    print("\n2. Generating hard negative examples...")
    hard_negatives = generator.generate_hard_negatives(positive_dataset)
    
    # Display statistics
    print("\n3. Dataset Statistics:")
    print(f"Positive examples: {len(positive_dataset.samples)}")
    print(f"Hard negatives: {len(hard_negatives.samples)}")
    
    if positive_dataset.samples:
        pos_stats = positive_dataset.get_statistics()
        print(f"Positive dataset stats: {json.dumps(pos_stats, indent=2)}")
    
    if hard_negatives.samples:
        neg_stats = hard_negatives.get_statistics()
        print(f"Hard negatives stats: {json.dumps(neg_stats, indent=2)}")
    
    # Validate schema compliance
    print("\n4. Validating schema compliance...")
    pos_validation = generator.validate_schema_compliance(positive_dataset)
    neg_validation = generator.validate_schema_compliance(hard_negatives)
    
    print(f"Positive validation: {pos_validation}")
    print(f"Negative validation: {neg_validation}")
    
    # Save datasets
    print("\n5. Saving datasets...")
    generator.save_datasets(positive_dataset, hard_negatives)
    
    print("\n✅ Aufgabe 1.2 - Pizza Quality Data Structure completed successfully!")


if __name__ == "__main__":
    main()
