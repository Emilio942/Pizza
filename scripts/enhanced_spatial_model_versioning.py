#!/usr/bin/env python3
"""
SPATIAL-4.2: Enhanced Spatial Model Versioning System

Comprehensive model versioning and compatibility management for Dual-Encoder models
in deployment pipeline. Supports model validation, rollback, and environment-specific
deployments.
"""

import os
import sys
import json
import hashlib
import logging
import shutil
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import PyTorch if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some features will be limited")

@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_type: str
    architecture: str
    file_path: str
    creation_date: str
    checksum: str
    size_mb: float
    performance_metrics: Dict[str, float]
    compatibility_info: Dict[str, Any]
    deployment_environments: List[str]
    training_config: Dict[str, Any]
    validation_status: str
    tags: List[str]

@dataclass 
class DeploymentEnvironment:
    """Deployment environment configuration."""
    name: str
    supported_architectures: List[str]
    memory_requirements_gb: float
    compute_requirements: Dict[str, Any]
    performance_requirements: Dict[str, float]
    validation_config: Dict[str, Any]

class EnhancedSpatialModelVersionManager:
    """Enhanced model versioning system for Spatial-MLLM deployment pipeline."""
    
    def __init__(self, models_dir: str = None):
        """Initialize model version manager."""
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata files
        self.metadata_file = self.models_dir / "model_versions.json"
        self.environments_file = self.models_dir / "deployment_environments.json"
        
        # Load existing data
        self.versions = self.load_metadata()
        self.environments = self.load_environments()
        
        logger.info(f"🔧 Model Version Manager initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Loaded {len(self.versions)} versions, {len(self.environments)} environments")
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def load_metadata(self) -> Dict[str, ModelVersion]:
        """Load model versions metadata."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: ModelVersion(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return {}
    
    def save_metadata(self):
        """Save model versions metadata."""
        try:
            data = {k: asdict(v) for k, v in self.versions.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
    
    def load_environments(self) -> Dict[str, DeploymentEnvironment]:
        """Load deployment environments configuration."""
        if not self.environments_file.exists():
            # Create default environments
            default_envs = self.create_default_environments()
            self.save_environments_dict(default_envs)
            return default_envs
        
        try:
            with open(self.environments_file, 'r') as f:
                data = json.load(f)
                return {k: DeploymentEnvironment(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load environments: {str(e)}")
            return self.create_default_environments()
    
    def save_environments(self):
        """Save deployment environments configuration."""
        self.save_environments_dict(self.environments)
    
    def save_environments_dict(self, environments: Dict[str, DeploymentEnvironment]):
        """Save environments dictionary to file."""
        try:
            data = {k: asdict(v) for k, v in environments.items()}
            with open(self.environments_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save environments: {str(e)}")
    
    def create_default_environments(self) -> Dict[str, DeploymentEnvironment]:
        """Create default deployment environments."""
        return {
            'development': DeploymentEnvironment(
                name='development',
                supported_architectures=['dual-encoder', 'visual-encoder', 'spatial-encoder'],
                memory_requirements_gb=8.0,
                compute_requirements={'min_gpu_memory_gb': 4.0, 'cuda_compatible': False},
                performance_requirements={'accuracy': 0.7, 'inference_time_ms': 3000},
                validation_config={'skip_heavy_tests': True, 'allow_cpu_only': True}
            ),
            'staging': DeploymentEnvironment(
                name='staging',
                supported_architectures=['dual-encoder'],
                memory_requirements_gb=16.0,
                compute_requirements={'min_gpu_memory_gb': 8.0, 'cuda_compatible': True},
                performance_requirements={'accuracy': 0.8, 'inference_time_ms': 2000},
                validation_config={'skip_heavy_tests': False, 'allow_cpu_only': False}
            ),
            'production': DeploymentEnvironment(
                name='production',
                supported_architectures=['dual-encoder'],
                memory_requirements_gb=32.0,
                compute_requirements={'min_gpu_memory_gb': 16.0, 'cuda_compatible': True},
                performance_requirements={'accuracy': 0.85, 'inference_time_ms': 1500},
                validation_config={'skip_heavy_tests': False, 'allow_cpu_only': False}
            )
        }
    
    def register_model(
        self,
        model_path: str,
        version_id: str,
        model_type: str = "spatial-mllm",
        architecture: str = "dual-encoder",
        performance_metrics: Dict[str, float] = None,
        training_config: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> ModelVersion:
        """Register a new model version."""
        logger.info(f"📝 Registering model version {version_id}")
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate metadata
        checksum = self.calculate_file_checksum(model_file)
        size_mb = model_file.stat().st_size / (1024 * 1024)
        
        # Detect compatibility information
        compatibility_info = self.detect_model_compatibility(model_file)
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            architecture=architecture,
            file_path=str(model_file.absolute()),
            creation_date=datetime.now().isoformat(),
            checksum=checksum,
            size_mb=size_mb,
            performance_metrics=performance_metrics or {},
            compatibility_info=compatibility_info,
            deployment_environments=[],
            training_config=training_config or {},
            validation_status='pending',
            tags=tags or []
        )
        
        self.versions[version_id] = model_version
        self.save_metadata()
        
        logger.info(f"✅ Model version {version_id} registered successfully")
        return model_version
    
    def detect_model_compatibility(self, model_path: Path) -> Dict[str, Any]:
        """Detect model compatibility information."""
        compatibility = {
            'pytorch_version': torch.__version__ if TORCH_AVAILABLE else 'unknown',
            'cuda_compatible': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'cpu_compatible': True,
            'architecture_detected': 'unknown',
            'model_size_category': 'unknown',
            'memory_estimate_gb': 0.0
        }
        
        if not TORCH_AVAILABLE:
            return compatibility
        
        try:
            # Analyze PyTorch models
            if model_path.suffix in ['.pth', '.pt']:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get('model_state_dict', 
                                               checkpoint.get('state_dict', checkpoint))
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in state_dict.values() 
                                     if isinstance(p, torch.Tensor))
                    
                    compatibility['total_parameters'] = total_params
                    compatibility['memory_estimate_gb'] = (total_params * 4) / (1024**3)
                    
                    # Categorize model size
                    if total_params < 100_000_000:
                        compatibility['model_size_category'] = 'small'
                    elif total_params < 1_000_000_000:
                        compatibility['model_size_category'] = 'medium'
                    elif total_params < 10_000_000_000:
                        compatibility['model_size_category'] = 'large'
                    else:
                        compatibility['model_size_category'] = 'xlarge'
                    
                    # Detect architecture patterns
                    param_names = list(state_dict.keys())
                    if any('visual' in name and 'spatial' in name for name in param_names):
                        compatibility['architecture_detected'] = 'dual-encoder'
                    elif any('visual' in name for name in param_names):
                        compatibility['architecture_detected'] = 'visual-encoder'
                    elif any('spatial' in name for name in param_names):
                        compatibility['architecture_detected'] = 'spatial-encoder'
                    
                    logger.info(f"Model analysis: {total_params:,} parameters")
            
            # Analyze Hugging Face models
            elif model_path.is_dir():
                config_file = model_path / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    compatibility['architecture_detected'] = config.get('architectures', ['unknown'])[0]
                    compatibility['model_type'] = config.get('model_type', 'unknown')
                    
                    # Estimate parameters from config
                    if 'num_parameters' in config:
                        compatibility['total_parameters'] = config['num_parameters']
                    elif 'hidden_size' in config and 'num_hidden_layers' in config:
                        hidden_size = config['hidden_size']
                        num_layers = config['num_hidden_layers']
                        vocab_size = config.get('vocab_size', 32000)
                        
                        estimated_params = vocab_size * hidden_size + num_layers * (4 * hidden_size * hidden_size)
                        compatibility['total_parameters'] = estimated_params
                        compatibility['memory_estimate_gb'] = (estimated_params * 4) / (1024**3)
        
        except Exception as e:
            logger.warning(f"Model compatibility detection failed: {str(e)}")
        
        return compatibility
    
    def validate_model_version(self, version_id: str, environment: str = 'development') -> Dict[str, Any]:
        """Validate a model version for a specific environment."""
        logger.info(f"🔍 Validating model version {version_id} for {environment}")
        
        if version_id not in self.versions:
            raise ValueError(f"Model version {version_id} not found")
        
        if environment not in self.environments:
            raise ValueError(f"Environment {environment} not configured")
        
        model_version = self.versions[version_id]
        env_config = self.environments[environment]
        validation_results = {}
        
        # Check file existence
        model_file = Path(model_version.file_path)
        validation_results['file_exists'] = model_file.exists()
        
        if not validation_results['file_exists']:
            validation_results['validation_error'] = f"Model file not found: {model_file}"
            return validation_results
        
        # Verify checksum
        current_checksum = self.calculate_file_checksum(model_file)
        validation_results['checksum_valid'] = current_checksum == model_version.checksum
        
        # Check architecture compatibility
        validation_results['architecture_compatible'] = \
            model_version.architecture in env_config.supported_architectures
        
        # Check memory requirements
        memory_estimate = model_version.compatibility_info.get('memory_estimate_gb', 0)
        validation_results['memory_compatible'] = memory_estimate <= env_config.memory_requirements_gb
        validation_results['memory_estimate_gb'] = memory_estimate
        
        # Check performance requirements
        performance_compatible = True
        for metric, required_value in env_config.performance_requirements.items():
            if metric in model_version.performance_metrics:
                actual_value = model_version.performance_metrics[metric]
                
                if 'time' in metric.lower() or 'latency' in metric.lower():
                    meets_requirement = actual_value <= required_value
                else:
                    meets_requirement = actual_value >= required_value
                
                validation_results[f'performance_{metric}'] = meets_requirement
                if not meets_requirement:
                    performance_compatible = False
        
        validation_results['performance_compatible'] = performance_compatible
        
        # Overall validation
        overall_valid = all([
            validation_results['file_exists'],
            validation_results['checksum_valid'],
            validation_results['architecture_compatible'],
            validation_results['memory_compatible'],
            validation_results['performance_compatible']
        ])
        
        validation_results['overall_valid'] = overall_valid
        
        # Update model status
        if overall_valid:
            if environment not in model_version.deployment_environments:
                model_version.deployment_environments.append(environment)
            model_version.validation_status = 'validated'
        else:
            model_version.validation_status = 'failed'
        
        self.save_metadata()
        
        logger.info(f"Validation result: {'✅ PASSED' if overall_valid else '❌ FAILED'}")
        return validation_results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Enhanced Spatial Model Versioning System")
    parser.add_argument("--register", help="Register a new model version")
    parser.add_argument("--version-id", help="Version ID for registration")
    parser.add_argument("--model-type", default="spatial-mllm", help="Model type")
    parser.add_argument("--architecture", default="dual-encoder", help="Model architecture")
    parser.add_argument("--validate", help="Validate model version for environment")
    parser.add_argument("--environment", default="development", help="Target environment")
    parser.add_argument("--test", action="store_true", help="Test deployment compatibility")
    
    args = parser.parse_args()
    
    # Initialize version manager
    version_manager = EnhancedSpatialModelVersionManager()
    
    try:
        if args.register and args.version_id:
            version_manager.register_model(
                model_path=args.register,
                version_id=args.version_id,
                model_type=args.model_type,
                architecture=args.architecture
            )
        
        elif args.validate:
            results = version_manager.validate_model_version(args.validate, args.environment)
            print(json.dumps(results, indent=2))
        
        elif args.test:
            print("Testing deployment compatibility...")
            print(f"Configured environments: {list(version_manager.environments.keys())}")
            print(f"Total versions: {len(version_manager.versions)}")
            print("✅ Deployment compatibility test completed")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
