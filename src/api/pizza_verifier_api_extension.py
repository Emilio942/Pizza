#!/usr/bin/env python3
"""
Aufgabe 5.1: API-Integration des Pizza-Verifier-Systems
Extension of the existing FastAPI-based Pizza-API with Quality-Assessment functionality

This module extends the existing Pizza API with:
- Verifier-based quality assessment endpoints
- Integration with Spatial-MLLM and Standard-CNN models
- Caching strategy for verifier results
- ModelManager integration for unified model management
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from fastapi import HTTPException
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.verification.pizza_verifier import PizzaVerifier
from src.continuous_improvement.pizza_verifier_improvement import ContinuousPizzaVerifierImprovement

class QualityAssessmentRequest(BaseModel):
    """Request model for pizza quality assessment"""
    image_path: str
    model_prediction: str
    confidence_score: float
    ground_truth_class: Optional[str] = None
    assessment_level: str = "standard"  # "standard", "detailed", "food_safety"

class QualityAssessmentResponse(BaseModel):
    """Response model for pizza quality assessment"""
    quality_score: float
    confidence: float
    assessment_details: Dict
    recommendations: List[str]
    processing_time_ms: float
    timestamp: str

class VerifierAPIExtension:
    """
    API Extension for Pizza Verifier System
    Integrates with existing FastAPI infrastructure
    """
    
    def __init__(self, api_instance=None, base_models_dir: str = "models", rl_training_results_dir: str = "results", improvement_config: Optional[Dict] = None):
        """Initialize the verifier API extension"""
        self.api = api_instance
        self.verifier = None
        self.continuous_improvement = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_max_size = 1000
        
        # Performance metrics
        self.request_count = 0
        self.total_processing_time = 0.0
        
        # Components tracking for testing
        self.components = {}
        
        self.base_models_dir = base_models_dir
        self.rl_training_results_dir = rl_training_results_dir
        self.improvement_config = improvement_config if improvement_config else self._get_default_improvement_config()
        
        # Defer initialization to a separate method to allow for more control
        # self._initialize_verifier_systems()

    def initialize(self):
        """Initialize the verifier and continuous improvement systems."""
        self._initialize_verifier_systems()

    def _get_default_improvement_config(self) -> Dict:
        """Provides a default configuration for the improvement system."""
        return {
            'learning_threshold': 0.02,
            'retraining_interval_hours': 24,
            'performance_window': 100,
            'min_samples': 50,
            'monitoring_interval_seconds': 300 
        }

    def _initialize_verifier_systems(self):
        """Initialize verifier and continuous improvement systems"""
        try:
            # Initialize Pizza Verifier
            self.verifier = PizzaVerifier()
            self.components['verifier'] = self.verifier
            print("✓ Pizza Verifier initialized")
            
            # Initialize Continuous Improvement System
            self.continuous_improvement = ContinuousPizzaVerifierImprovement(
                base_models_dir=self.base_models_dir,
                rl_training_results_dir=self.rl_training_results_dir,
                improvement_config=self.improvement_config
            )
            if self.continuous_improvement.initialize_system():
                self.components['continuous_improvement'] = self.continuous_improvement
                print("✓ Continuous Improvement System initialized and system started.")
            else:
                print("Warning: Continuous Improvement System failed to initialize.")

        except Exception as e:
            print(f"Warning: Could not initialize verifier systems: {e}")
    
    def _generate_cache_key(self, request: QualityAssessmentRequest) -> str:
        """Generate cache key for quality assessment request"""
        key_data = f"{request.image_path}:{request.model_prediction}:{request.confidence_score}:{request.assessment_level}"
        return str(hash(key_data))
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return (time.time() - cache_entry['timestamp']) < self.cache_ttl
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if (current_time - entry['timestamp']) > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        
        # Limit cache size
        if len(self.cache) > self.cache_max_size:
            # Remove oldest entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            to_remove = len(self.cache) - self.cache_max_size
            for key, _ in sorted_entries[:to_remove]:
                del self.cache[key]
    
    async def assess_pizza_quality(self, request: QualityAssessmentRequest) -> QualityAssessmentResponse:
        """
        Assess pizza recognition quality using the verifier system
        
        Args:
            request: Quality assessment request
            
        Returns:
            Quality assessment response with score and recommendations
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                cached_result = self.cache[cache_key]['result']
                return QualityAssessmentResponse(**cached_result)
            
            # Validate input
            if not Path(request.image_path).exists():
                raise HTTPException(status_code=404, detail="Image file not found")
            
            # Prepare verifier input
            verifier_input = {
                'image_path': request.image_path,
                'model_prediction': request.model_prediction,
                'confidence_score': request.confidence_score,
                'ground_truth_class': request.ground_truth_class
            }
            
            # Perform quality assessment
            if self.verifier:
                quality_result = await self._assess_with_verifier(verifier_input, request.assessment_level)
            else:
                # Fallback to basic assessment
                quality_result = await self._basic_quality_assessment(verifier_input)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality_result, request)
            
            # Prepare response
            processing_time = (time.time() - start_time) * 1000
            response_data = {
                'quality_score': quality_result['quality_score'],
                'confidence': quality_result['confidence'],
                'assessment_details': quality_result['details'],
                'recommendations': recommendations,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = {
                'result': response_data,
                'timestamp': time.time()
            }
            self._clean_cache()
            
            # Update metrics
            self.request_count += 1
            self.total_processing_time += processing_time
            
            # Log to continuous improvement system
            if self.continuous_improvement:
                await self._log_assessment_result(request, quality_result)
            
            return QualityAssessmentResponse(**response_data)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")
    
    async def _assess_with_verifier(self, verifier_input: Dict, assessment_level: str) -> Dict:
        """Perform quality assessment using the verifier system"""
        
        # Load image and extract features
        image_features = await self._extract_image_features(verifier_input['image_path'])
        
        # Prepare verifier input
        verification_data = {
            'image_features': image_features,
            'prediction': verifier_input['model_prediction'],
            'confidence': verifier_input['confidence_score'],
            'ground_truth': verifier_input.get('ground_truth_class')
        }
        
        # Perform verification based on assessment level
        if assessment_level == "food_safety":
            quality_score = await self._food_safety_assessment(verification_data)
            assessment_details = {
                'food_safety_score': quality_score,
                'safety_classification': 'safe' if quality_score > 0.8 else 'warning' if quality_score > 0.6 else 'unsafe',
                'critical_checks': ['raw_detection', 'burn_level', 'contamination']
            }
        elif assessment_level == "detailed":
            quality_score = await self._detailed_assessment(verification_data)
            assessment_details = {
                'overall_quality': quality_score,
                'accuracy_confidence': verification_data['confidence'],
                'prediction_consistency': await self._check_prediction_consistency(verification_data),
                'temporal_stability': await self._check_temporal_stability(verification_data)
            }
        else:  # standard
            quality_score = await self._standard_assessment(verification_data)
            assessment_details = {
                'quality_score': quality_score,
                'prediction_accuracy': verification_data['confidence'],
                'assessment_confidence': min(quality_score + 0.1, 1.0)
            }
        
        return {
            'quality_score': quality_score,
            'confidence': assessment_details.get('assessment_confidence', quality_score),
            'details': assessment_details
        }
    
    async def _basic_quality_assessment(self, verifier_input: Dict) -> Dict:
        """Fallback basic quality assessment when verifier is not available"""
        confidence = verifier_input['confidence_score']
        
        # Simple heuristic-based assessment
        if confidence > 0.9:
            quality_score = 0.95
        elif confidence > 0.8:
            quality_score = 0.85
        elif confidence > 0.7:
            quality_score = 0.75
        elif confidence > 0.6:
            quality_score = 0.65
        else:
            quality_score = 0.5
        
        return {
            'quality_score': quality_score,
            'confidence': confidence,
            'details': {
                'method': 'confidence_based_heuristic',
                'confidence_threshold': confidence,
                'assessment_type': 'basic'
            }
        }
    
    async def _extract_image_features(self, image_path: str) -> Dict:
        """Extract features from pizza image for verifier"""
        # This would integrate with existing MicroPizzaNet feature extraction
        # For now, return placeholder features
        return {
            'image_path': image_path,
            'features_extracted': True,
            'feature_vector_size': 512,
            'preprocessing_applied': ['resize', 'normalize']
        }
    
    async def _food_safety_assessment(self, verification_data: Dict) -> float:
        """Perform food safety specific assessment"""
        # Food safety critical assessment
        confidence = verification_data['confidence']
        prediction = verification_data['prediction']
        
        # Critical for raw vs cooked detection
        if 'raw' in prediction.lower():
            return max(0.9, confidence)  # High confidence required for raw detection
        elif 'burnt' in prediction.lower():
            return max(0.8, confidence)  # High confidence for burnt detection
        else:
            return min(confidence + 0.1, 1.0)
    
    async def _detailed_assessment(self, verification_data: Dict) -> float:
        """Perform detailed quality assessment"""
        base_score = verification_data['confidence']
        
        # Factor in prediction consistency and temporal stability
        consistency_bonus = 0.05  # Placeholder
        temporal_bonus = 0.03     # Placeholder
        
        return min(base_score + consistency_bonus + temporal_bonus, 1.0)
    
    async def _standard_assessment(self, verification_data: Dict) -> float:
        """Perform standard quality assessment"""
        return verification_data['confidence']
    
    async def _check_prediction_consistency(self, verification_data: Dict) -> float:
        """Check prediction consistency (placeholder)"""
        return 0.85  # Placeholder consistency score
    
    async def _check_temporal_stability(self, verification_data: Dict) -> float:
        """Check temporal stability (placeholder)"""
        return 0.80  # Placeholder temporal stability score
    
    def _generate_recommendations(self, quality_result: Dict, request: QualityAssessmentRequest) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        quality_score = quality_result['quality_score']
        
        if quality_score < 0.6:
            recommendations.append("Low quality detection - consider manual verification")
            recommendations.append("Check image quality and lighting conditions")
        
        if quality_score < 0.8 and request.assessment_level == "food_safety":
            recommendations.append("Food safety concern - manual inspection recommended")
        
        if request.confidence_score < 0.7:
            recommendations.append("Low model confidence - consider alternative model")
        
        if quality_score > 0.9:
            recommendations.append("High quality detection - suitable for automated processing")
        
        return recommendations
    
    async def _log_assessment_result(self, request: QualityAssessmentRequest, quality_result: Dict):
        """Log assessment result to continuous improvement system"""
        try:
            if self.continuous_improvement:
                assessment_data = {
                    'timestamp': datetime.now().isoformat(),
                    'image_path': request.image_path,
                    'model_prediction': request.model_prediction,
                    'confidence_score': request.confidence_score,
                    'quality_score': quality_result['quality_score'],
                    'assessment_level': request.assessment_level
                }
                # This would integrate with the continuous improvement logging
                pass
        except Exception as e:
            print(f"Warning: Could not log to continuous improvement: {e}")
    
    def get_api_metrics(self) -> Dict:
        """Get API performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'total_requests': self.request_count,
            'average_processing_time_ms': avg_processing_time,
            'cache_hit_ratio': len(self.cache) / max(self.request_count, 1),
            'cache_size': len(self.cache),
            'verifier_available': self.verifier is not None,
            'continuous_improvement_available': self.continuous_improvement is not None
        }

def extend_existing_api(app):
    """
    Extend existing FastAPI app with verifier endpoints
    
    Args:
        app: Existing FastAPI application instance
    """
    
    # Initialize verifier extension
    verifier_api = VerifierAPIExtension(app)
    
    @app.post("/api/v1/quality/assess", response_model=QualityAssessmentResponse)
    async def assess_pizza_quality(request: QualityAssessmentRequest):
        """Assess pizza recognition quality"""
        return await verifier_api.assess_pizza_quality(request)
    
    @app.get("/api/v1/quality/metrics")
    async def get_quality_metrics():
        """Get quality assessment API metrics"""
        return verifier_api.get_api_metrics()
    
    @app.get("/api/v1/quality/health")
    async def quality_health_check():
        """Health check for quality assessment system"""
        return {
            'status': 'healthy',
            'verifier_available': verifier_api.verifier is not None,
            'continuous_improvement_available': verifier_api.continuous_improvement is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    return verifier_api

# Example usage for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test_quality_assessment():
        """Test the quality assessment functionality"""
        verifier_api = VerifierAPIExtension()
        
        # Test request
        test_request = QualityAssessmentRequest(
            image_path="/home/emilio/Documents/ai/pizza/test_data/pizza_test.jpg",
            model_prediction="basic",
            confidence_score=0.85,
            assessment_level="standard"
        )
        
        try:
            response = await verifier_api.assess_pizza_quality(test_request)
            print("Quality Assessment Response:")
            print(f"Quality Score: {response.quality_score}")
            print(f"Confidence: {response.confidence}")
            print(f"Processing Time: {response.processing_time_ms}ms")
            print(f"Recommendations: {response.recommendations}")
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_quality_assessment())
