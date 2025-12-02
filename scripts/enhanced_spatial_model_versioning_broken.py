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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_type: str  # 'spatial-mllm', 'standard', 'hybrid'
    architecture: str  # 'dual-encoder', 'single-encoder'
    creation_date: str
    file_path: str
    checksum: str
    size_mb: float
    performance_metrics: Dict[str, float]
    compatibility_info: Dict[str, Any]
    deployment_environments: List[str]
    training_config: Dict[str, Any]
    validation_status: str  # 'validated', 'pending', 'failed'
    tags: List[str]

@dataclass
class DeploymentEnvironment:
    """Deployment environment configuration."""
    name: str
    cuda_version: str
    pytorch_version: str
    memory_requirements_gb: float
    supported_architectures: List[str]
    performance_requirements: Dict[str, float]
    security_requirements: List[str]

class EnhancedSpatialModelVersionManager:
    """Enhanced model versioning system for Spatial-MLLM deployment."""
    
    def __init__(self, project_root: str = None):
        """Initialize the model version manager."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models" / "spatial_mllm"
        self.versions_dir = self.models_dir / "versions"
        self.metadata_file = self.models_dir / "version_metadata.json"
        self.environments_file = self.models_dir / "deployment_environments.json"
        
        # Create directories if they don't exist
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata and environments
        self.load_metadata()
        self.load_environments()
        
        logger.info("🔧 Enhanced Spatial Model Version Manager initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Versions directory: {self.versions_dir}")
    
    def load_metadata(self) -> None:
        """Load model version metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.versions = {v_id: ModelVersion(**v_data) for v_id, v_data in metadata.items()}
        else:
            self.versions = {}
            logger.info("No existing metadata found, starting fresh")
    
    def save_metadata(self) -> None:
        """Save model version metadata."""
        metadata = {v_id: asdict(version) for v_id, version in self.versions.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {self.metadata_file}")
    
    def load_environments(self) -> None:
        """Load deployment environment configurations."""
        if self.environments_file.exists():
            with open(self.environments_file, 'r') as f:
                env_data = json.load(f)
                self.environments = {name: DeploymentEnvironment(**env_info) 
                                   for name, env_info in env_data.items()}
        else:
            # Initialize with default environments
            self.environments = {
                'development': DeploymentEnvironment(
                    name='development',
                    cuda_version='12.4',
                    pytorch_version='2.6.0',
                    memory_requirements_gb=4.0,
                    supported_architectures=['dual-encoder', 'single-encoder'],
                    performance_requirements={'inference_time_ms': 3000, 'accuracy': 0.70},
                    security_requirements=['basic_validation']
                ),
                'staging': DeploymentEnvironment(
                    name='staging',
                    cuda_version='12.4',
                    pytorch_version='2.6.0',
                    memory_requirements_gb=6.0,
                    supported_architectures=['dual-encoder'],
                    performance_requirements={'inference_time_ms': 2000, 'accuracy': 0.75},
                    security_requirements=['comprehensive_validation', 'security_scan']
                ),
                'production': DeploymentEnvironment(
                    name='production',
                    cuda_version='12.4',
                    pytorch_version='2.6.0',
                    memory_requirements_gb=8.0,
                    supported_architectures=['dual-encoder'],
                    performance_requirements={'inference_time_ms': 1500, 'accuracy': 0.80},
                    security_requirements=['comprehensive_validation', 'security_scan', 'performance_validation']
                )
            }
            self.save_environments()
    
    def save_environments(self) -> None:
        """Save deployment environment configurations."""
        env_data = {name: asdict(env) for name, env in self.environments.items()}
        with open(self.environments_file, 'w') as f:
            json.dump(env_data, f, indent=2)
        logger.info(f"Environment configurations saved to {self.environments_file}")
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def register_model(self, 
                      model_path: Union[str, Path],
                      version_id: str,
                      model_type: str = 'spatial-mllm',
                      architecture: str = 'dual-encoder',
                      performance_metrics: Dict[str, float] = None,
                      training_config: Dict[str, Any] = None,
                      tags: List[str] = None) -> ModelVersion:
        """Register a new model version."""
        logger.info(f"📦 Registering model version: {version_id}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate file metadata
        checksum = self.calculate_file_checksum(model_path)
        size_mb = self.get_file_size_mb(model_path)
        
        # Copy model to versions directory
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        version_file = version_dir / model_path.name
        
        if not version_file.exists():
            shutil.copy2(model_path, version_file)
            logger.info(f"Model copied to {version_file}")
        
        # Detect compatibility info
        compatibility_info = self.detect_model_compatibility(model_path)
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            architecture=architecture,
            creation_date=datetime.now().isoformat(),
            file_path=str(version_file),
            checksum=checksum,
            size_mb=size_mb,
            performance_metrics=performance_metrics or {},
            compatibility_info=compatibility_info,
            deployment_environments=[],
            training_config=training_config or {},
            validation_status='pending',
            tags=tags or []
        )\n        
        self.versions[version_id] = model_version
        self.save_metadata()
        
        logger.info(f"✅ Model version {version_id} registered successfully")
        return model_version\n    \n    def detect_model_compatibility(self, model_path: Path) -> Dict[str, Any]:\n        \"\"\"Detect model compatibility information.\"\"\"\n        compatibility = {\n            'pytorch_version': torch.__version__,\n            'cuda_compatible': torch.cuda.is_available(),\n            'cpu_compatible': True,\n            'architecture_detected': 'unknown',\n            'model_size_category': 'unknown',\n            'memory_estimate_gb': 0.0\n        }\n        \n        try:\n            # Try to load model metadata\n            if model_path.suffix == '.pth' or model_path.suffix == '.pt':\n                # PyTorch model\n                try:\n                    checkpoint = torch.load(model_path, map_location='cpu')\n                    \n                    if isinstance(checkpoint, dict):\n                        # Check for common keys\n                        if 'model_state_dict' in checkpoint:\n                            state_dict = checkpoint['model_state_dict']\n                        elif 'state_dict' in checkpoint:\n                            state_dict = checkpoint['state_dict']\n                        else:\n                            state_dict = checkpoint\n                        \n                        # Estimate model size\n                        total_params = 0\n                        for param_name, param_tensor in state_dict.items():\n                            if isinstance(param_tensor, torch.Tensor):\n                                total_params += param_tensor.numel()\n                        \n                        compatibility['total_parameters'] = total_params\n                        compatibility['memory_estimate_gb'] = (total_params * 4) / (1024**3)  # Assuming float32\n                        \n                        # Determine model size category\n                        if total_params < 100_000_000:  # < 100M\n                            compatibility['model_size_category'] = 'small'\n                        elif total_params < 1_000_000_000:  # < 1B\n                            compatibility['model_size_category'] = 'medium'\n                        elif total_params < 10_000_000_000:  # < 10B\n                            compatibility['model_size_category'] = 'large'\n                        else:\n                            compatibility['model_size_category'] = 'xlarge'\n                        \n                        # Try to detect architecture\n                        param_names = list(state_dict.keys())\n                        if any('visual' in name and 'spatial' in name for name in param_names):\n                            compatibility['architecture_detected'] = 'dual-encoder'\n                        elif any('visual' in name for name in param_names):\n                            compatibility['architecture_detected'] = 'visual-encoder'\n                        elif any('spatial' in name for name in param_names):\n                            compatibility['architecture_detected'] = 'spatial-encoder'\n                        \n                        logger.info(f\"Model analysis: {total_params:,} parameters, {compatibility['model_size_category']} size\")\n                    \n                except Exception as e:\n                    logger.warning(f\"Could not analyze PyTorch model: {str(e)}\")\n            \n            elif model_path.is_dir():\n                # Hugging Face model directory\n                config_file = model_path / 'config.json'\n                if config_file.exists():\n                    with open(config_file, 'r') as f:\n                        config = json.load(f)\n                    \n                    compatibility['architecture_detected'] = config.get('architectures', ['unknown'])[0]\n                    compatibility['model_type'] = config.get('model_type', 'unknown')\n                    \n                    # Try to estimate parameters from config\n                    if 'num_parameters' in config:\n                        compatibility['total_parameters'] = config['num_parameters']\n                    elif 'hidden_size' in config and 'num_hidden_layers' in config:\n                        # Rough estimate for transformer models\n                        hidden_size = config['hidden_size']\n                        num_layers = config['num_hidden_layers']\n                        vocab_size = config.get('vocab_size', 32000)\n                        \n                        # Rough parameter count estimation\n                        estimated_params = vocab_size * hidden_size + num_layers * (4 * hidden_size * hidden_size)\n                        compatibility['total_parameters'] = estimated_params\n                        compatibility['memory_estimate_gb'] = (estimated_params * 4) / (1024**3)\n                    \n                    logger.info(f\"Hugging Face model detected: {compatibility['architecture_detected']}\")\n        \n        except Exception as e:\n            logger.warning(f\"Model compatibility detection failed: {str(e)}\")\n        \n        return compatibility\n    \n    def validate_model_version(self, version_id: str, environment: str = 'development') -> Dict[str, Any]:\n        \"\"\"Validate a model version for a specific environment.\"\"\"\n        logger.info(f\"🔍 Validating model version {version_id} for {environment}\")\n        \n        if version_id not in self.versions:\n            raise ValueError(f\"Model version {version_id} not found\")\n        \n        if environment not in self.environments:\n            raise ValueError(f\"Environment {environment} not configured\")\n        \n        model_version = self.versions[version_id]\n        env_config = self.environments[environment]\n        validation_results = {}\n        \n        # Check file integrity\n        model_file = Path(model_version.file_path)\n        if not model_file.exists():\n            validation_results['file_exists'] = False\n            validation_results['validation_error'] = f\"Model file not found: {model_file}\"\n            return validation_results\n        \n        validation_results['file_exists'] = True\n        \n        # Verify checksum\n        current_checksum = self.calculate_file_checksum(model_file)\n        validation_results['checksum_valid'] = current_checksum == model_version.checksum\n        \n        if not validation_results['checksum_valid']:\n            validation_results['checksum_error'] = f\"Checksum mismatch: expected {model_version.checksum}, got {current_checksum}\"\n        \n        # Check architecture compatibility\n        validation_results['architecture_compatible'] = model_version.architecture in env_config.supported_architectures\n        \n        # Check memory requirements\n        memory_estimate = model_version.compatibility_info.get('memory_estimate_gb', 0)\n        validation_results['memory_compatible'] = memory_estimate <= env_config.memory_requirements_gb\n        validation_results['memory_estimate_gb'] = memory_estimate\n        validation_results['memory_limit_gb'] = env_config.memory_requirements_gb\n        \n        # Check performance requirements\n        performance_compatible = True\n        for metric, required_value in env_config.performance_requirements.items():\n            if metric in model_version.performance_metrics:\n                actual_value = model_version.performance_metrics[metric]\n                \n                # Different metrics have different comparison logic\n                if 'time' in metric.lower() or 'latency' in metric.lower():\n                    # Lower is better for time metrics\n                    meets_requirement = actual_value <= required_value\n                else:\n                    # Higher is better for accuracy metrics\n                    meets_requirement = actual_value >= required_value\n                \n                validation_results[f'performance_{metric}'] = meets_requirement\n                if not meets_requirement:\n                    performance_compatible = False\n            else:\n                validation_results[f'performance_{metric}'] = False\n                performance_compatible = False\n        \n        validation_results['performance_compatible'] = performance_compatible\n        \n        # Overall validation status\n        overall_valid = all([\n            validation_results['file_exists'],\n            validation_results['checksum_valid'],\n            validation_results['architecture_compatible'],\n            validation_results['memory_compatible'],\n            validation_results['performance_compatible']\n        ])\n        \n        validation_results['overall_valid'] = overall_valid\n        \n        # Update model validation status\n        if overall_valid:\n            if environment not in model_version.deployment_environments:\n                model_version.deployment_environments.append(environment)\n            model_version.validation_status = 'validated'\n        else:\n            model_version.validation_status = 'failed'\n        \n        self.save_metadata()\n        \n        logger.info(f\"Validation result: {'✅ PASSED' if overall_valid else '❌ FAILED'}\")\n        return validation_results\n    \n    def list_available_versions(self, environment: str = None) -> List[str]:\n        \"\"\"List available model versions, optionally filtered by environment.\"\"\"\n        if environment:\n            return [v_id for v_id, version in self.versions.items() \n                   if environment in version.deployment_environments]\n        else:\n            return list(self.versions.keys())\n    \n    def get_model_info(self, version_id: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Get detailed information about a model version.\"\"\"\n        if version_id in self.versions:\n            return asdict(self.versions[version_id])\n        return None\n    \n    def get_latest_version(self, model_type: str = None, environment: str = None) -> Optional[str]:\n        \"\"\"Get the latest model version for a specific type and environment.\"\"\"\n        candidates = []\n        \n        for v_id, version in self.versions.items():\n            # Filter by model type if specified\n            if model_type and version.model_type != model_type:\n                continue\n            \n            # Filter by environment if specified\n            if environment and environment not in version.deployment_environments:\n                continue\n            \n            # Only consider validated models\n            if version.validation_status != 'validated':\n                continue\n            \n            candidates.append((v_id, version.creation_date))\n        \n        if not candidates:\n            return None\n        \n        # Sort by creation date and return the latest\n        candidates.sort(key=lambda x: x[1], reverse=True)\n        return candidates[0][0]\n    \n    def create_rollback_plan(self, current_version: str, environment: str) -> Dict[str, Any]:\n        \"\"\"Create a rollback plan for a deployment.\"\"\"\n        logger.info(f\"📋 Creating rollback plan for {current_version} in {environment}\")\n        \n        # Find the previous stable version\n        stable_versions = []\n        for v_id, version in self.versions.items():\n            if (v_id != current_version and \n                environment in version.deployment_environments and\n                version.validation_status == 'validated'):\n                stable_versions.append((v_id, version.creation_date))\n        \n        # Sort by date and get the most recent stable version\n        stable_versions.sort(key=lambda x: x[1], reverse=True)\n        \n        rollback_plan = {\n            'current_version': current_version,\n            'environment': environment,\n            'rollback_available': len(stable_versions) > 0,\n            'rollback_candidates': [v[0] for v in stable_versions[:3]],  # Top 3 candidates\n            'recommended_rollback': stable_versions[0][0] if stable_versions else None,\n            'rollback_steps': [\n                'Stop current service',\n                'Backup current model',\n                'Deploy rollback model',\n                'Validate rollback deployment',\n                'Update model version metadata',\n                'Monitor service health'\n            ],\n            'estimated_downtime_minutes': 5\n        }\n        \n        logger.info(f\"Rollback plan created: {'✅ Available' if rollback_plan['rollback_available'] else '❌ No stable versions'}\")\n        return rollback_plan\n    \n    def export_model_registry(self, output_path: str = None) -> str:\n        \"\"\"Export model registry for backup or migration.\"\"\"\n        if output_path is None:\n            output_path = self.models_dir / f\"model_registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n        \n        export_data = {\n            'export_timestamp': datetime.now().isoformat(),\n            'versions': {v_id: asdict(version) for v_id, version in self.versions.items()},\n            'environments': {name: asdict(env) for name, env in self.environments.items()}\n        }\n        \n        with open(output_path, 'w') as f:\n            json.dump(export_data, f, indent=2)\n        \n        logger.info(f\"📤 Model registry exported to {output_path}\")\n        return str(output_path)\n    \n    def import_model_registry(self, import_path: str) -> None:\n        \"\"\"Import model registry from backup.\"\"\"\n        logger.info(f\"📥 Importing model registry from {import_path}\")\n        \n        with open(import_path, 'r') as f:\n            import_data = json.load(f)\n        \n        # Import versions\n        imported_versions = {v_id: ModelVersion(**v_data) \n                           for v_id, v_data in import_data['versions'].items()}\n        \n        # Import environments\n        imported_environments = {name: DeploymentEnvironment(**env_data) \n                               for name, env_data in import_data['environments'].items()}\n        \n        # Merge with existing data (import takes precedence)\n        self.versions.update(imported_versions)\n        self.environments.update(imported_environments)\n        \n        # Save merged data\n        self.save_metadata()\n        self.save_environments()\n        \n        logger.info(f\"✅ Imported {len(imported_versions)} versions and {len(imported_environments)} environments\")\n    \n    def cleanup_old_versions(self, keep_count: int = 5, environment: str = None) -> int:\n        \"\"\"Clean up old model versions, keeping only the most recent ones.\"\"\"\n        logger.info(f\"🧹 Cleaning up old model versions (keeping {keep_count})\")\n        \n        # Get versions to consider for cleanup\n        versions_to_consider = []\n        for v_id, version in self.versions.items():\n            if environment and environment not in version.deployment_environments:\n                continue\n            versions_to_consider.append((v_id, version.creation_date))\n        \n        # Sort by creation date (newest first)\n        versions_to_consider.sort(key=lambda x: x[1], reverse=True)\n        \n        # Identify versions to remove\n        versions_to_remove = versions_to_consider[keep_count:]\n        removed_count = 0\n        \n        for v_id, _ in versions_to_remove:\n            version = self.versions[v_id]\n            \n            # Don't remove if it's currently deployed in any environment\n            if len(version.deployment_environments) > 0:\n                continue\n            \n            # Remove model file\n            model_file = Path(version.file_path)\n            if model_file.exists():\n                model_file.unlink()\n                logger.info(f\"Removed model file: {model_file}\")\n            \n            # Remove version directory if empty\n            version_dir = model_file.parent\n            if version_dir.exists() and not any(version_dir.iterdir()):\n                version_dir.rmdir()\n            \n            # Remove from metadata\n            del self.versions[v_id]\n            removed_count += 1\n            logger.info(f\"Removed version: {v_id}\")\n        \n        # Save updated metadata\n        if removed_count > 0:\n            self.save_metadata()\n        \n        logger.info(f\"🧹 Cleanup completed: {removed_count} versions removed\")\n        return removed_count\n    \n    def generate_deployment_report(self, environment: str) -> Dict[str, Any]:\n        \"\"\"Generate deployment report for an environment.\"\"\"\n        logger.info(f\"📊 Generating deployment report for {environment}\")\n        \n        env_versions = [v for v in self.versions.values() \n                       if environment in v.deployment_environments]\n        \n        report = {\n            'environment': environment,\n            'timestamp': datetime.now().isoformat(),\n            'total_versions': len(env_versions),\n            'model_types': {},\n            'architectures': {},\n            'validation_status': {},\n            'performance_summary': {},\n            'latest_version': self.get_latest_version(environment=environment),\n            'storage_usage_mb': sum(v.size_mb for v in env_versions)\n        }\n        \n        # Analyze model types and architectures\n        for version in env_versions:\n            # Count model types\n            if version.model_type not in report['model_types']:\n                report['model_types'][version.model_type] = 0\n            report['model_types'][version.model_type] += 1\n            \n            # Count architectures\n            if version.architecture not in report['architectures']:\n                report['architectures'][version.architecture] = 0\n            report['architectures'][version.architecture] += 1\n            \n            # Count validation statuses\n            if version.validation_status not in report['validation_status']:\n                report['validation_status'][version.validation_status] = 0\n            report['validation_status'][version.validation_status] += 1\n        \n        # Calculate performance summary\n        if env_versions:\n            all_metrics = set()\n            for version in env_versions:\n                all_metrics.update(version.performance_metrics.keys())\n            \n            for metric in all_metrics:\n                values = [v.performance_metrics.get(metric) for v in env_versions \n                         if metric in v.performance_metrics]\n                if values:\n                    report['performance_summary'][metric] = {\n                        'min': min(values),\n                        'max': max(values),\n                        'avg': sum(values) / len(values),\n                        'count': len(values)\n                    }\n        \n        logger.info(f\"📊 Report generated: {report['total_versions']} versions in {environment}\")\n        return report\n\ndef main():\n    \"\"\"Main function for command-line usage.\"\"\"\n    parser = argparse.ArgumentParser(description=\"Enhanced Spatial Model Versioning System\")\n    parser.add_argument(\"--register\", help=\"Register a new model version\")\n    parser.add_argument(\"--version-id\", help=\"Version ID for registration\")\n    parser.add_argument(\"--model-type\", default=\"spatial-mllm\", help=\"Model type\")\n    parser.add_argument(\"--architecture\", default=\"dual-encoder\", help=\"Model architecture\")\n    parser.add_argument(\"--validate\", help=\"Validate model version for environment\")\n    parser.add_argument(\"--environment\", default=\"development\", help=\"Target environment\")\n    parser.add_argument(\"--list\", action=\"store_true\", help=\"List available versions\")\n    parser.add_argument(\"--info\", help=\"Get model version info\")\n    parser.add_argument(\"--latest\", action=\"store_true\", help=\"Get latest version\")\n    parser.add_argument(\"--rollback-plan\", help=\"Create rollback plan for version\")\n    parser.add_argument(\"--export\", help=\"Export model registry\")\n    parser.add_argument(\"--import\", help=\"Import model registry\")\n    parser.add_argument(\"--cleanup\", type=int, help=\"Clean up old versions (keep N)\")\n    parser.add_argument(\"--report\", action=\"store_true\", help=\"Generate deployment report\")\n    parser.add_argument(\"--test-deployment\", action=\"store_true\", help=\"Test deployment compatibility\")\n    \n    args = parser.parse_args()\n    \n    # Initialize version manager\n    version_manager = EnhancedSpatialModelVersionManager()\n    \n    try:\n        if args.register and args.version_id:\n            version_manager.register_model(\n                model_path=args.register,\n                version_id=args.version_id,\n                model_type=args.model_type,\n                architecture=args.architecture\n            )\n        \n        elif args.validate:\n            results = version_manager.validate_model_version(args.validate, args.environment)\n            print(json.dumps(results, indent=2))\n        \n        elif args.list:\n            versions = version_manager.list_available_versions(args.environment)\n            print(f\"Available versions for {args.environment}:\")\n            for version in versions:\n                print(f\"  - {version}\")\n        \n        elif args.info:\n            info = version_manager.get_model_info(args.info)\n            if info:\n                print(json.dumps(info, indent=2))\n            else:\n                print(f\"Version {args.info} not found\")\n        \n        elif args.latest:\n            latest = version_manager.get_latest_version(environment=args.environment)\n            print(f\"Latest version for {args.environment}: {latest}\")\n        \n        elif args.rollback_plan:\n            plan = version_manager.create_rollback_plan(args.rollback_plan, args.environment)\n            print(json.dumps(plan, indent=2))\n        \n        elif getattr(args, 'export'):\n            export_path = version_manager.export_model_registry(getattr(args, 'export'))\n            print(f\"Registry exported to: {export_path}\")\n        \n        elif getattr(args, 'import'):\n            version_manager.import_model_registry(getattr(args, 'import'))\n            print(\"Registry imported successfully\")\n        \n        elif args.cleanup:\n            removed = version_manager.cleanup_old_versions(args.cleanup, args.environment)\n            print(f\"Cleaned up {removed} old versions\")\n        \n        elif args.report:\n            report = version_manager.generate_deployment_report(args.environment)\n            print(json.dumps(report, indent=2))\n        \n        elif args.test_deployment:\n            # Test basic functionality\n            print(\"Testing deployment compatibility...\")\n            \n            # List environments\n            print(f\"Configured environments: {list(version_manager.environments.keys())}\")\n            \n            # List versions\n            versions = version_manager.list_available_versions()\n            print(f\"Total versions: {len(versions)}\")\n            \n            # Test validation for each environment\n            for env_name in version_manager.environments.keys():\n                env_versions = version_manager.list_available_versions(env_name)\n                print(f\"Versions for {env_name}: {len(env_versions)}\")\n            \n            print(\"✅ Deployment compatibility test completed\")\n        \n        else:\n            parser.print_help()\n    \n    except Exception as e:\n        logger.error(f\"Operation failed: {str(e)}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()
