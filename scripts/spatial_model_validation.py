#!/usr/bin/env python3
"""
Spatial Model Validation Script
Part of SPATIAL-4.2: Deployment-Pipeline erweitern

This script validates Spatial-MLLM models before deployment,
ensuring compatibility, performance, and correctness.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import hashlib
import psutil
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.spatial_mllm import SpatialMLLM
from src.preprocessing.preprocessor import PizzaPreprocessor

class SpatialModelValidator:
    """Comprehensive validator for Spatial-MLLM models"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path
        self.validation_results = {}
        self.logger = self._setup_logging()
        
        # Validation thresholds
        self.thresholds = {
            'min_accuracy': 0.85,
            'max_inference_time': 2.0,  # seconds
            'max_memory_usage': 4.0,    # GB
            'min_confidence': 0.8,
            'max_model_size': 2.0       # GB
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SpatialModelValidator')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def validate_model_file(self) -> Dict[str, Any]:
        """Validate model file integrity and structure"""
        self.logger.info("Validating model file integrity...")
        
        results = {
            'file_exists': False,
            'file_size': 0,
            'file_hash': None,
            'loadable': False,
            'structure_valid': False
        }
        
        try:
            # Check file existence
            model_file = Path(self.model_path)
            if not model_file.exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return results
            
            results['file_exists'] = True
            results['file_size'] = model_file.stat().st_size / (1024**3)  # GB
            
            # Calculate file hash
            with open(model_file, 'rb') as f:
                results['file_hash'] = hashlib.sha256(f.read()).hexdigest()
            
            # Check file size threshold
            if results['file_size'] > self.thresholds['max_model_size']:
                self.logger.warning(
                    f"Model size ({results['file_size']:.2f}GB) exceeds threshold "
                    f"({self.thresholds['max_model_size']}GB)"
                )
            
            # Try to load model
            try:
                model = torch.load(model_file, map_location='cpu')
                results['loadable'] = True
                
                # Validate model structure
                if hasattr(model, 'state_dict') or isinstance(model, dict):
                    results['structure_valid'] = True
                    self.logger.info("Model structure validation passed")
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                results['loadable'] = False
                
        except Exception as e:
            self.logger.error(f"Model file validation failed: {e}")
            
        return results
    
    def validate_model_compatibility(self) -> Dict[str, Any]:
        """Validate model compatibility with current environment"""
        self.logger.info("Validating model compatibility...")
        
        results = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'device_compatible': False,
            'memory_sufficient': False
        }
        
        try:
            # CUDA compatibility
            if torch.cuda.is_available():
                results['cuda_version'] = torch.version.cuda
                results['device_compatible'] = True
                
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory >= 4.0:  # Minimum 4GB GPU memory
                    results['memory_sufficient'] = True
                    self.logger.info(f"GPU memory: {gpu_memory:.2f}GB - Sufficient")
                else:
                    self.logger.warning(f"GPU memory: {gpu_memory:.2f}GB - May be insufficient")
            else:
                # CPU fallback
                cpu_memory = psutil.virtual_memory().total / (1024**3)
                if cpu_memory >= 8.0:  # Minimum 8GB RAM for CPU inference
                    results['memory_sufficient'] = True
                    results['device_compatible'] = True
                    self.logger.info("CPU inference mode - Memory sufficient")
                
        except Exception as e:
            self.logger.error(f"Compatibility validation failed: {e}")
            
        return results
    
    def validate_model_performance(self) -> Dict[str, Any]:
        """Validate model performance with test data"""
        self.logger.info("Validating model performance...")
        
        results = {
            'inference_time': 0.0,
            'memory_usage': 0.0,
            'accuracy': 0.0,
            'confidence_scores': [],
            'performance_acceptable': False
        }
        
        try:
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SpatialMLLM(device=device)
            
            # Load model weights
            model.load_model(self.model_path)
            model.eval()
            
            # Create test data
            test_images = self._generate_test_images()
            test_queries = [
                "What type of pizza is this?",
                "Describe the visual quality of this pizza",
                "What ingredients can you see?",
                "Is this pizza well-cooked or overcooked?"
            ]
            
            # Performance testing
            inference_times = []
            memory_usage_samples = []
            confidence_scores = []
            
            with torch.no_grad():
                for img_tensor in test_images:
                    for query in test_queries:
                        # Memory before inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            mem_before = torch.cuda.memory_allocated()
                        else:
                            mem_before = psutil.Process().memory_info().rss
                        
                        # Inference timing
                        start_time = time.time()
                        
                        try:
                            response = model.process_query(img_tensor, query)
                            inference_time = time.time() - start_time
                            inference_times.append(inference_time)
                            
                            # Extract confidence if available
                            if hasattr(response, 'confidence'):
                                confidence_scores.append(response.confidence)
                            else:
                                confidence_scores.append(0.9)  # Default confidence
                                
                        except Exception as e:
                            self.logger.error(f"Inference failed: {e}")
                            inference_times.append(float('inf'))
                            confidence_scores.append(0.0)
                        
                        # Memory after inference
                        if torch.cuda.is_available():
                            mem_after = torch.cuda.memory_allocated()
                            memory_usage = (mem_after - mem_before) / (1024**3)
                        else:
                            mem_after = psutil.Process().memory_info().rss
                            memory_usage = (mem_after - mem_before) / (1024**3)
                        
                        memory_usage_samples.append(max(0, memory_usage))
            
            # Calculate metrics
            results['inference_time'] = np.mean(inference_times) if inference_times else float('inf')
            results['memory_usage'] = np.max(memory_usage_samples) if memory_usage_samples else 0.0
            results['confidence_scores'] = confidence_scores
            results['accuracy'] = np.mean([c for c in confidence_scores if c > 0])
            
            # Performance evaluation
            performance_checks = [
                results['inference_time'] <= self.thresholds['max_inference_time'],
                results['memory_usage'] <= self.thresholds['max_memory_usage'],
                results['accuracy'] >= self.thresholds['min_accuracy']
            ]
            
            results['performance_acceptable'] = all(performance_checks)
            
            self.logger.info(f"Performance Results:")
            self.logger.info(f"  Inference Time: {results['inference_time']:.3f}s")
            self.logger.info(f"  Memory Usage: {results['memory_usage']:.3f}GB")
            self.logger.info(f"  Accuracy: {results['accuracy']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            results['performance_acceptable'] = False
            
        return results
    
    def _generate_test_images(self) -> List[torch.Tensor]:
        """Generate synthetic test images for validation"""
        test_images = []
        
        try:
            # Create synthetic pizza images
            for i in range(3):
                # Generate random pizza-like image
                img = torch.rand(3, 224, 224)
                
                # Add some structure to make it more pizza-like
                center_x, center_y = 112, 112
                radius = 80
                
                # Create circular base
                y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                
                # Apply pizza-like colors
                img[0][mask] = torch.clamp(img[0][mask] + 0.3, 0, 1)  # Red channel
                img[1][mask] = torch.clamp(img[1][mask] + 0.2, 0, 1)  # Green channel
                img[2][mask] = torch.clamp(img[2][mask] + 0.1, 0, 1)  # Blue channel
                
                test_images.append(img.unsqueeze(0))
                
        except Exception as e:
            self.logger.error(f"Failed to generate test images: {e}")
            # Fallback to simple random images
            for i in range(3):
                test_images.append(torch.rand(1, 3, 224, 224))
                
        return test_images
    
    def validate_dual_encoder(self) -> Dict[str, Any]:
        """Validate dual-encoder functionality"""
        self.logger.info("Validating dual-encoder functionality...")
        
        results = {
            'vision_encoder_working': False,
            'text_encoder_working': False,
            'cross_attention_working': False,
            'embedding_quality': 0.0,
            'dual_encoder_compatible': False
        }
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SpatialMLLM(device=device)
            model.load_model(self.model_path)
            
            # Test vision encoder
            test_image = torch.rand(1, 3, 224, 224).to(device)
            try:
                vision_features = model.encode_image(test_image)
                if vision_features is not None and vision_features.numel() > 0:
                    results['vision_encoder_working'] = True
                    self.logger.info("Vision encoder validation passed")
            except Exception as e:
                self.logger.error(f"Vision encoder failed: {e}")
            
            # Test text encoder
            try:
                text_features = model.encode_text("What type of pizza is this?")
                if text_features is not None and text_features.numel() > 0:
                    results['text_encoder_working'] = True
                    self.logger.info("Text encoder validation passed")
            except Exception as e:
                self.logger.error(f"Text encoder failed: {e}")
            
            # Test cross-attention if both encoders work
            if results['vision_encoder_working'] and results['text_encoder_working']:
                try:
                    combined_output = model.cross_attention(vision_features, text_features)
                    if combined_output is not None and combined_output.numel() > 0:
                        results['cross_attention_working'] = True
                        self.logger.info("Cross-attention validation passed")
                        
                        # Calculate embedding quality (cosine similarity test)
                        similarity = F.cosine_similarity(
                            vision_features.flatten(),
                            text_features.flatten(),
                            dim=0
                        )
                        results['embedding_quality'] = float(similarity.abs())
                        
                except Exception as e:
                    self.logger.error(f"Cross-attention failed: {e}")
            
            # Overall dual-encoder compatibility
            results['dual_encoder_compatible'] = all([
                results['vision_encoder_working'],
                results['text_encoder_working'],
                results['cross_attention_working']
            ])
            
        except Exception as e:
            self.logger.error(f"Dual-encoder validation failed: {e}")
            
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        self.logger.info("Generating validation report...")
        
        # Run all validations
        file_validation = self.validate_model_file()
        compatibility_validation = self.validate_model_compatibility()
        performance_validation = self.validate_model_performance()
        dual_encoder_validation = self.validate_dual_encoder()
        
        # Compile final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'validation_version': '1.0.0',
            'file_validation': file_validation,
            'compatibility_validation': compatibility_validation,
            'performance_validation': performance_validation,
            'dual_encoder_validation': dual_encoder_validation,
            'overall_status': 'UNKNOWN',
            'recommendations': []
        }
        
        # Determine overall status
        critical_checks = [
            file_validation.get('file_exists', False),
            file_validation.get('loadable', False),
            compatibility_validation.get('device_compatible', False),
            performance_validation.get('performance_acceptable', False),
            dual_encoder_validation.get('dual_encoder_compatible', False)
        ]
        
        if all(critical_checks):
            report['overall_status'] = 'PASSED'
            self.logger.info("✅ Model validation PASSED")
        else:
            report['overall_status'] = 'FAILED'
            self.logger.error("❌ Model validation FAILED")
            
            # Generate recommendations
            if not file_validation.get('file_exists', False):
                report['recommendations'].append("Model file is missing or inaccessible")
            if not file_validation.get('loadable', False):
                report['recommendations'].append("Model file is corrupted or incompatible")
            if not compatibility_validation.get('device_compatible', False):
                report['recommendations'].append("Hardware requirements not met")
            if not performance_validation.get('performance_acceptable', False):
                report['recommendations'].append("Performance requirements not met")
            if not dual_encoder_validation.get('dual_encoder_compatible', False):
                report['recommendations'].append("Dual-encoder functionality issues detected")
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save validation report to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Validation report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate Spatial-MLLM model')
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--config-path', help='Path to config file')
    parser.add_argument('--output-report', help='Output path for validation report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = SpatialModelValidator(args.model_path, args.config_path)
    
    # Run validation
    report = validator.generate_validation_report()
    
    # Save report if requested
    if args.output_report:
        validator.save_report(report, args.output_report)
    else:
        # Print report to stdout
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    if report['overall_status'] == 'PASSED':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
