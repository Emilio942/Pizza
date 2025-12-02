#!/usr/bin/env python3
"""
Spatial MLLM Model for Pizza Classification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the actual implementation
from scripts.spatial_inference_optimized import SpatialMLLMInferenceSystem


class SpatialMLLM:
    """Wrapper class for SpatialMLLM compatibility"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the spatial MLLM wrapper"""
        self.inference_system = SpatialMLLMInferenceSystem()
    
    def load_model(self, model_name: str = "Diankun/Spatial-MLLM-subset-sft"):
        """Load the spatial MLLM model"""
        return self.inference_system.load_model(model_name)
    
    def evaluate(self, *args, **kwargs):
        """Evaluate using the spatial MLLM system"""
        if hasattr(self.inference_system, 'evaluate'):
            return self.inference_system.evaluate(*args, **kwargs)
        else:
            # Fallback implementation
            return {"status": "evaluation not implemented"}
    
    def predict(self, *args, **kwargs):
        """Make predictions using the spatial MLLM system"""
        if hasattr(self.inference_system, 'predict'):
            return self.inference_system.predict(*args, **kwargs)
        else:
            # Fallback implementation
            return {"prediction": "unknown", "confidence": 0.0}


# Backward compatibility
__all__ = ['SpatialMLLM']
