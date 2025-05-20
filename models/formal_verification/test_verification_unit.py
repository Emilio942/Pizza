#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the formal verification framework.
"""

import os
import sys
import unittest
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import mock verification for testing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mock_verification import (
    ModelVerifier,
    VerificationProperty,
    VerificationResult,
    load_model_for_verification,
    CLASS_NAMES
)


class MockModel(torch.nn.Module):
    """Simple mock model for testing."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, x):
        # For testing, create predictable outputs
        batch_size = x.shape[0]
        # Create logits where the highest value is at index that matches batch index % num_classes
        logits = torch.zeros((batch_size, self.num_classes))
        for i in range(batch_size):
            prediction_idx = i % self.num_classes
            logits[i, prediction_idx] = 10.0  # Strong prediction
            # Add small values to other classes
            for j in range(self.num_classes):
                if j != prediction_idx:
                    logits[i, j] = -5.0
        return logits


class TestVerificationFramework(unittest.TestCase):
    """Test cases for the verification framework."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock model
        self.model = MockModel(num_classes=len(CLASS_NAMES))
        
        # Create verifier
        self.verifier = ModelVerifier(
            model=self.model,
            input_size=(48, 48),
            device='cpu',
            epsilon=0.03
        )
        
        # Create test image
        self.test_image = np.random.rand(3, 48, 48)
        
    def test_initialization(self):
        """Test verifier initialization."""
        self.assertIsInstance(self.verifier, ModelVerifier)
        
    def test_robustness_verification(self):
        """Test robustness verification."""
        # Test for multiple classes
        for class_idx in range(len(CLASS_NAMES)):
            result = self.verifier.verify_robustness(
                input_image=self.test_image,
                true_class=class_idx,
                epsilon=0.03
            )
            
            # Verify result type
            self.assertIsInstance(result, VerificationResult)
            self.assertEqual(result.property_type, VerificationProperty.ROBUSTNESS)
            
            # Check required fields
            self.assertIn("epsilon", result.details)
            self.assertIn("norm", result.details)
            
    def test_brightness_invariance(self):
        """Test brightness invariance verification."""
        for class_idx in range(len(CLASS_NAMES)):
            result = self.verifier.verify_brightness_invariance(
                input_image=self.test_image,
                true_class=class_idx,
                brightness_range=(0.8, 1.2)
            )
            
            # Verify result type
            self.assertIsInstance(result, VerificationResult)
            self.assertEqual(result.property_type, VerificationProperty.BRIGHTNESS_INVARIANCE)
            
            # Check required fields
            self.assertIn("brightness_range", result.details)
            
    def test_class_separation(self):
        """Test class separation verification."""
        # Test for a pair of classes
        class1, class2 = 0, 1
        
        # Create multiple test images
        test_images = [np.random.rand(3, 48, 48) for _ in range(3)]
        
        result = self.verifier.verify_class_separation(
            class1=class1,
            class2=class2,
            examples=test_images,
            robustness_eps=0.03
        )
        
        # Verify result type
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.property_type, VerificationProperty.CLASS_SEPARATION)
        
        # Check required fields
        self.assertIn("class1", result.details)
        self.assertIn("class2", result.details)
        self.assertIn("results", result.details)
        
    def test_verify_all_properties(self):
        """Test verification of all properties."""
        # Create test data
        num_test_images = 3
        images = [np.random.rand(3, 48, 48) for _ in range(num_test_images)]
        classes = [i % len(CLASS_NAMES) for i in range(num_test_images)]
        
        # Define critical pairs
        critical_pairs = [(0, 1), (2, 3)]
        
        # Run verification
        results = self.verifier.verify_all_properties(
            input_images=images,
            true_classes=classes,
            critical_class_pairs=critical_pairs,
            robustness_eps=0.02,
            brightness_range=(0.9, 1.1)
        )
        
        # Check results structure
        self.assertIn(VerificationProperty.ROBUSTNESS.value, results)
        self.assertIn(VerificationProperty.BRIGHTNESS_INVARIANCE.value, results)
        self.assertIn(VerificationProperty.CLASS_SEPARATION.value, results)
        
        # Check number of results
        self.assertEqual(len(results[VerificationProperty.ROBUSTNESS.value]), num_test_images)
        self.assertEqual(len(results[VerificationProperty.BRIGHTNESS_INVARIANCE.value]), num_test_images)
        
    def test_report_generation(self):
        """Test report generation."""
        # Create sample results
        sample_results = {
            VerificationProperty.ROBUSTNESS.value: [
                VerificationResult(
                    verified=True,
                    property_type=VerificationProperty.ROBUSTNESS,
                    time_seconds=0.5,
                    details={"epsilon": 0.03, "norm": "L_inf"}
                )
            ],
            VerificationProperty.BRIGHTNESS_INVARIANCE.value: [
                VerificationResult(
                    verified=False,
                    property_type=VerificationProperty.BRIGHTNESS_INVARIANCE,
                    time_seconds=0.3,
                    details={"brightness_range": (0.8, 1.2)}
                )
            ]
        }
        
        # Generate report
        report = self.verifier.generate_verification_report(sample_results)
        
        # Check report structure
        self.assertIn("model_name", report)
        self.assertIn("properties", report)
        self.assertIn("summary", report)
        
        # Check properties
        self.assertIn(VerificationProperty.ROBUSTNESS.value, report["properties"])
        self.assertIn(VerificationProperty.BRIGHTNESS_INVARIANCE.value, report["properties"])
        
        # Check summary calculations
        self.assertEqual(report["summary"]["total_properties_checked"], 2)
        self.assertEqual(report["summary"]["total_verified"], 1)
        self.assertEqual(report["summary"]["total_failed"], 1)
        self.assertAlmostEqual(report["summary"]["overall_verification_rate"], 0.5)
        
    def test_load_model_function(self):
        """Test model loading function."""
        # Mock model path
        model_path = "mock_path.pth"
        
        # Load model
        model = load_model_for_verification(
            model_path=model_path,
            model_type="MicroPizzaNet",
            num_classes=len(CLASS_NAMES)
        )
        
        # Check model
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
