#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the formal verification framework.
These tests validate that the verification functionality works correctly
with the pizza detection models.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Try to import verification dependencies
try:
    from auto_LiRPA import BoundedModule
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

# Import verification module
from models.formal_verification.formal_verification import (
    ModelVerifier, VerificationProperty, VerificationResult, load_model_for_verification
)
from src.pizza_detector import MicroPizzaNet

# Skip tests if verification dependencies are not available
pytestmark = pytest.mark.skipif(
    not VERIFICATION_AVAILABLE,
    reason="Verification dependencies not installed. Run: pip install auto_LiRPA"
)

@pytest.fixture
def small_model():
    """Create a small model for testing."""
    model = MicroPizzaNet(num_classes=3, dropout_rate=0.0)
    model.eval()
    return model

@pytest.fixture
def dummy_image():
    """Create a dummy input image."""
    np.random.seed(42)
    return np.random.rand(48, 48, 3)

class TestFormalVerification:
    """Test cases for the formal verification framework."""
    
    def test_verifier_initialization(self, small_model):
        """Test that the verifier can be initialized with a model."""
        verifier = ModelVerifier(
            model=small_model,
            input_size=(48, 48),
            epsilon=0.01
        )
        assert verifier.model is small_model
        assert verifier.input_size == (48, 48)
        assert verifier.epsilon == 0.01
    
    def test_verification_result(self):
        """Test the VerificationResult class."""
        result = VerificationResult(
            verified=True,
            property_type=VerificationProperty.ROBUSTNESS,
            time_seconds=1.5,
            details={"epsilon": 0.01}
        )
        
        assert result.verified is True
        assert result.property_type == VerificationProperty.ROBUSTNESS
        assert result.time_seconds == 1.5
        assert result.details == {"epsilon": 0.01}
        
        # Test conversion to dict
        result_dict = result.to_dict()
        assert result_dict["verified"] is True
        assert result_dict["property_type"] == "robustness"
        assert result_dict["time_seconds"] == 1.5
        assert result_dict["details"] == {"epsilon": 0.01}
    
    def test_robustness_verification(self, small_model, dummy_image):
        """Test robustness verification with a small model."""
        verifier = ModelVerifier(
            model=small_model,
            input_size=(48, 48),
            epsilon=0.01
        )
        
        # For test purposes, we don't need the result to be verified,
        # we just need to ensure the verification process runs without errors
        result = verifier.verify_robustness(
            input_image=dummy_image,
            true_class=0,
            epsilon=0.01
        )
        
        assert isinstance(result, VerificationResult)
        assert result.property_type == VerificationProperty.ROBUSTNESS
        assert isinstance(result.verified, bool)
        assert result.time_seconds > 0
        
    def test_brightness_invariance(self, small_model, dummy_image):
        """Test brightness invariance verification."""
        verifier = ModelVerifier(
            model=small_model,
            input_size=(48, 48),
            epsilon=0.01
        )
        
        result = verifier.verify_brightness_invariance(
            input_image=dummy_image,
            true_class=0,
            brightness_range=(0.8, 1.2)
        )
        
        assert isinstance(result, VerificationResult)
        assert result.property_type == VerificationProperty.BRIGHTNESS_INVARIANCE
        assert isinstance(result.verified, bool)
        assert result.time_seconds > 0
        
    def test_class_separation(self, small_model, dummy_image):
        """Test class separation verification."""
        verifier = ModelVerifier(
            model=small_model,
            input_size=(48, 48),
            epsilon=0.01
        )
        
        # Create multiple test images
        np.random.seed(42)
        test_images = [np.random.rand(48, 48, 3) for _ in range(3)]
        
        result = verifier.verify_class_separation(
            class1=0,
            class2=1,
            examples=test_images,
            robustness_eps=0.01
        )
        
        assert isinstance(result, VerificationResult)
        assert result.property_type == VerificationProperty.CLASS_SEPARATION
        assert isinstance(result.verified, bool)
        assert result.time_seconds > 0
        
    def test_generate_report(self, small_model, dummy_image, tmp_path):
        """Test report generation."""
        verifier = ModelVerifier(
            model=small_model,
            input_size=(48, 48),
            epsilon=0.01
        )
        
        # Create some verification results
        robustness_result = verifier.verify_robustness(
            input_image=dummy_image,
            true_class=0
        )
        
        brightness_result = verifier.verify_brightness_invariance(
            input_image=dummy_image,
            true_class=0
        )
        
        results = {
            VerificationProperty.ROBUSTNESS.value: [robustness_result],
            VerificationProperty.BRIGHTNESS_INVARIANCE.value: [brightness_result]
        }
        
        # Generate report
        report_path = tmp_path / "test_report.json"
        report = verifier.generate_verification_report(
            results=results,
            output_path=str(report_path)
        )
        
        # Check that the report was created
        assert os.path.exists(report_path)
        
        # Check report structure
        assert "model_name" in report
        assert "properties" in report
        assert "summary" in report
        assert "robustness" in report["properties"]
        assert "brightness" in report["properties"]
        
    def test_load_model_for_verification(self, tmp_path, small_model):
        """Test the model loading function."""
        # Save a test model
        model_path = tmp_path / "test_model.pth"
        torch.save(small_model.state_dict(), model_path)
        
        # Load the model
        loaded_model = load_model_for_verification(
            model_path=str(model_path),
            model_type="MicroPizzaNet",
            num_classes=3
        )
        
        # Check that the model was loaded correctly
        assert isinstance(loaded_model, MicroPizzaNet)
        assert sum(p.numel() for p in loaded_model.parameters()) == \
               sum(p.numel() for p in small_model.parameters())
        
        # Ensure the model is in eval mode
        assert not loaded_model.training

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
