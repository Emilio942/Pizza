#!/usr/bin/env python3
"""
SPATIAL-4.2: Model Versioning System for Dual-Encoder Models

This script implements a comprehensive model versioning system specifically
designed for Spatial-MLLM dual-encoder architectures.
"""

import os
import sys
import json
import time
import torch
import hashlib
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import git
import semver

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    model_type: str  # "spatial_mllm", "standard_cnn", "dual_encoder"
    architecture: str
    training_date: str
    file_path: str
    file_size_mb: float
    checksum: str
    performance_metrics: Dict[str, float]
    dependencies: Dict[str, str]
    git_commit: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ModelRegistry:
    """Registry of all model versions"""
    models: List[ModelVersion]
    last_updated: str
    registry_version: str = "1.0.0"

class SpatialModelVersioning:
    """Model versioning system for Spatial-MLLM models"""
    
    def __init__(self, model_dir: str, registry_file: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = Path(registry_file) if registry_file else self.model_dir / "model_registry.json"
        self.versioned_dir = self.model_dir / "versioned"
        self.versioned_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.registry = self._load_registry()
        
        logger.info(f"🔢 Model Versioning System initialized")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Registry: {self.registry_file}")
        logger.info(f"Tracked models: {len(self.registry.models)}")

    def _load_registry(self) -> ModelRegistry:
        """Load model registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                models = []
                for model_data in data.get("models", []):
                    models.append(ModelVersion(**model_data))
                
                return ModelRegistry(
                    models=models,
                    last_updated=data.get("last_updated", ""),
                    registry_version=data.get("registry_version", "1.0.0")
                )
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        
        return ModelRegistry(
            models=[],
            last_updated=datetime.now().isoformat(),
            registry_version="1.0.0"
        )

    def _save_registry(self):
        """Save model registry to file"""
        self.registry.last_updated = datetime.now().isoformat()
        
        registry_data = {
            "models": [asdict(model) for model in self.registry.models],
            "last_updated": self.registry.last_updated,
            "registry_version": self.registry.registry_version
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"📝 Registry saved with {len(self.registry.models)} models")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_git_info(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            repo = git.Repo(project_root)
            return repo.head.commit.hexsha
        except:
            return None

    def _detect_model_type(self, model_path: Path) -> Tuple[str, str]:
        """Detect model type and architecture from file"""
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Check for Spatial-MLLM indicators
            if any("spatial" in key.lower() for key in checkpoint.keys()):
                return "spatial_mllm", "dual_encoder"
            
            # Check for dual encoder patterns
            if any("encoder" in key.lower() for key in checkpoint.keys()):
                encoder_count = len([k for k in checkpoint.keys() if "encoder" in k.lower()])
                if encoder_count >= 2:
                    return "dual_encoder", "multi_encoder"
            
            # Check for standard CNN patterns
            if any("conv" in key.lower() for key in checkpoint.keys()):
                return "standard_cnn", "convolutional"
            
            # Check metadata
            if "metadata" in checkpoint:
                metadata = checkpoint["metadata"]
                model_type = metadata.get("model_type", "unknown")
                architecture = metadata.get("architecture", "unknown")
                return model_type, architecture
            
            return "unknown", "unknown"
            
        except Exception as e:
            logger.warning(f"Failed to detect model type for {model_path}: {e}")
            return "unknown", "unknown"

    def _generate_version(self, model_type: str, existing_versions: List[str]) -> str:
        """Generate new version number"""
        # Find latest version for this model type
        type_versions = [v for v in existing_versions if v.startswith(f"{model_type}_")]
        
        if not type_versions:
            return f"{model_type}_1.0.0"
        
        # Extract version numbers and find max
        version_numbers = []
        for v in type_versions:
            try:
                version_part = v.split("_", 1)[1]  # Remove model_type prefix
                version_numbers.append(semver.VersionInfo.parse(version_part))
            except:
                continue
        
        if not version_numbers:
            return f"{model_type}_1.0.0"
        
        latest = max(version_numbers)
        next_version = latest.bump_minor()
        
        return f"{model_type}_{next_version}"

    def register_model(self, 
                      model_path: str, 
                      performance_metrics: Dict[str, float] = None,
                      dependencies: Dict[str, str] = None,
                      notes: str = None,
                      tags: List[str] = None) -> ModelVersion:
        """Register a new model version"""
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"📦 Registering model: {model_path.name}")
        
        # Detect model type and architecture
        model_type, architecture = self._detect_model_type(model_path)
        
        # Generate version
        existing_versions = [m.version for m in self.registry.models]
        version = self._generate_version(model_type, existing_versions)
        
        # Calculate file info
        file_size_mb = model_path.stat().st_size / 1024 / 1024
        checksum = self._calculate_checksum(model_path)
        
        # Copy model to versioned directory
        versioned_filename = f"{version}_{model_path.name}"
        versioned_path = self.versioned_dir / versioned_filename
        shutil.copy2(model_path, versioned_path)
        
        # Create model version
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            architecture=architecture,
            training_date=datetime.now().isoformat(),
            file_path=str(versioned_path),
            file_size_mb=file_size_mb,
            checksum=checksum,
            performance_metrics=performance_metrics or {},
            dependencies=dependencies or {},
            git_commit=self._get_git_info(),
            notes=notes,
            tags=tags or []
        )
        
        # Add to registry
        self.registry.models.append(model_version)
        self._save_registry()
        
        logger.info(f"✅ Model registered as version: {version}")
        logger.info(f"📊 File size: {file_size_mb:.1f} MB")
        logger.info(f"🔑 Checksum: {checksum[:16]}...")
        
        return model_version

    def get_model_by_version(self, version: str) -> Optional[ModelVersion]:
        """Get model by version string"""
        for model in self.registry.models:
            if model.version == version:
                return model
        return None

    def get_latest_model(self, model_type: str = None) -> Optional[ModelVersion]:
        """Get latest model, optionally filtered by type"""
        models = self.registry.models
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if not models:
            return None
        
        # Sort by training date
        models.sort(key=lambda m: m.training_date, reverse=True)
        return models[0]

    def list_models(self, model_type: str = None, tags: List[str] = None) -> List[ModelVersion]:
        """List models with optional filtering"""
        models = self.registry.models
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        return sorted(models, key=lambda m: m.training_date, reverse=True)

    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        model1 = self.get_model_by_version(version1)
        model2 = self.get_model_by_version(version2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "size_difference_mb": model2.file_size_mb - model1.file_size_mb,
            "architecture_same": model1.architecture == model2.architecture,
            "performance_comparison": {},
            "dependency_changes": {}
        }
        
        # Compare performance metrics
        all_metrics = set(model1.performance_metrics.keys()) | set(model2.performance_metrics.keys())
        for metric in all_metrics:
            val1 = model1.performance_metrics.get(metric, 0)
            val2 = model2.performance_metrics.get(metric, 0)
            comparison["performance_comparison"][metric] = {
                "version1": val1,
                "version2": val2,
                "difference": val2 - val1,
                "improvement": val2 > val1
            }
        
        # Compare dependencies
        all_deps = set(model1.dependencies.keys()) | set(model2.dependencies.keys())
        for dep in all_deps:
            ver1 = model1.dependencies.get(dep, "missing")
            ver2 = model2.dependencies.get(dep, "missing")
            if ver1 != ver2:
                comparison["dependency_changes"][dep] = {
                    "from": ver1,
                    "to": ver2
                }
        
        return comparison

    def validate_model(self, version: str) -> Dict[str, Any]:
        """Validate a registered model"""
        model = self.get_model_by_version(version)
        if not model:
            raise ValueError(f"Model version {version} not found")
        
        validation_result = {
            "version": version,
            "file_exists": False,
            "checksum_valid": False,
            "loadable": False,
            "metadata_valid": False,
            "errors": []
        }
        
        # Check file existence
        model_path = Path(model.file_path)
        if model_path.exists():
            validation_result["file_exists"] = True
            
            # Validate checksum
            try:
                current_checksum = self._calculate_checksum(model_path)
                validation_result["checksum_valid"] = current_checksum == model.checksum
                if not validation_result["checksum_valid"]:
                    validation_result["errors"].append("Checksum mismatch")
            except Exception as e:
                validation_result["errors"].append(f"Checksum calculation failed: {e}")
            
            # Test model loading
            try:
                checkpoint = torch.load(model_path, map_location="cpu")
                validation_result["loadable"] = True
                
                # Validate metadata if present
                if "metadata" in checkpoint:
                    metadata = checkpoint["metadata"]
                    validation_result["metadata_valid"] = True
                else:
                    validation_result["metadata_valid"] = False
                    validation_result["errors"].append("No metadata found in model")
                    
            except Exception as e:
                validation_result["errors"].append(f"Model loading failed: {e}")
        else:
            validation_result["errors"].append("Model file not found")
        
        return validation_result

    def cleanup_old_versions(self, keep_latest: int = 5, model_type: str = None) -> List[str]:
        """Remove old model versions, keeping the latest N versions"""
        models = self.list_models(model_type=model_type)
        
        if len(models) <= keep_latest:
            logger.info(f"Only {len(models)} models found, no cleanup needed")
            return []
        
        # Remove old versions
        to_remove = models[keep_latest:]
        removed_versions = []
        
        for model in to_remove:
            try:
                model_path = Path(model.file_path)
                if model_path.exists():
                    model_path.unlink()
                    logger.info(f"🗑️ Removed old model: {model.version}")
                
                # Remove from registry
                self.registry.models.remove(model)
                removed_versions.append(model.version)
                
            except Exception as e:
                logger.error(f"Failed to remove {model.version}: {e}")
        
        if removed_versions:
            self._save_registry()
            logger.info(f"🧹 Cleaned up {len(removed_versions)} old versions")
        
        return removed_versions

    def export_model(self, version: str, export_path: str) -> bool:
        """Export a model version to specified path"""
        model = self.get_model_by_version(version)
        if not model:
            raise ValueError(f"Model version {version} not found")
        
        source_path = Path(model.file_path)
        export_path = Path(export_path)
        
        try:
            shutil.copy2(source_path, export_path)
            
            # Also export metadata
            metadata = {
                "version": model.version,
                "model_type": model.model_type,
                "architecture": model.architecture,
                "training_date": model.training_date,
                "performance_metrics": model.performance_metrics,
                "dependencies": model.dependencies,
                "checksum": model.checksum,
                "notes": model.notes,
                "tags": model.tags
            }
            
            metadata_path = export_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"📤 Exported {version} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive versioning report"""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_models": len(self.registry.models),
            "model_types": {},
            "storage_usage": {},
            "performance_trends": {},
            "validation_status": {}
        }
        
        # Analyze by model type
        type_counts = {}
        type_sizes = {}
        
        for model in self.registry.models:
            model_type = model.model_type
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
            type_sizes[model_type] = type_sizes.get(model_type, 0) + model.file_size_mb
        
        report["model_types"] = {
            "counts": type_counts,
            "sizes_mb": type_sizes
        }
        
        # Storage usage
        total_size = sum(m.file_size_mb for m in self.registry.models)
        report["storage_usage"] = {
            "total_size_mb": total_size,
            "total_size_gb": total_size / 1024,
            "average_size_mb": total_size / len(self.registry.models) if self.registry.models else 0
        }
        
        # Performance trends (if available)
        for model in self.registry.models:
            for metric, value in model.performance_metrics.items():
                if metric not in report["performance_trends"]:
                    report["performance_trends"][metric] = []
                report["performance_trends"][metric].append({
                    "version": model.version,
                    "value": value,
                    "date": model.training_date
                })
        
        return report

def main():
    """Main function for model versioning operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPATIAL-4.2: Model Versioning System")
    parser.add_argument("--model-dir", type=str, default="models/spatial_mllm", help="Model directory")
    parser.add_argument("--registry-file", type=str, help="Registry file path")
    parser.add_argument("--action", type=str, choices=["register", "list", "validate", "compare", "cleanup", "export", "report"], 
                       default="list", help="Action to perform")
    
    # Register options
    parser.add_argument("--model-path", type=str, help="Path to model file to register")
    parser.add_argument("--notes", type=str, help="Notes for model registration")
    parser.add_argument("--tags", type=str, nargs="*", help="Tags for model registration")
    
    # Other options
    parser.add_argument("--version", type=str, help="Model version for operations")
    parser.add_argument("--version2", type=str, help="Second model version for comparison")
    parser.add_argument("--model-type", type=str, help="Filter by model type")
    parser.add_argument("--export-path", type=str, help="Export path for model")
    parser.add_argument("--keep-latest", type=int, default=5, help="Number of latest versions to keep during cleanup")
    parser.add_argument("--output-dir", type=str, help="Output directory for reports")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    # Initialize versioning system
    versioning = SpatialModelVersioning(args.model_dir, args.registry_file)
    
    try:
        if args.action == "register":
            if not args.model_path:
                parser.error("--model-path required for register action")
            
            # Mock performance metrics for testing
            performance_metrics = {
                "accuracy": 0.85,
                "f1_score": 0.82,
                "inference_time": 1.2
            } if args.test_mode else {}
            
            # Mock dependencies
            dependencies = {
                "torch": "2.6.0",
                "transformers": "4.51.3",
                "spatial_mllm": "1.0.0"
            } if args.test_mode else {}
            
            model = versioning.register_model(
                args.model_path,
                performance_metrics=performance_metrics,
                dependencies=dependencies,
                notes=args.notes,
                tags=args.tags
            )
            print(f"Registered model: {model.version}")
            
        elif args.action == "list":
            models = versioning.list_models(model_type=args.model_type)
            print(f"\n📋 Found {len(models)} models:")
            for model in models:
                print(f"  {model.version} ({model.model_type}) - {model.file_size_mb:.1f}MB - {model.training_date[:10]}")
                
        elif args.action == "validate":
            if not args.version:
                parser.error("--version required for validate action")
            
            result = versioning.validate_model(args.version)
            print(f"\n🔍 Validation for {args.version}:")
            print(f"  File exists: {result['file_exists']}")
            print(f"  Checksum valid: {result['checksum_valid']}")
            print(f"  Loadable: {result['loadable']}")
            print(f"  Metadata valid: {result['metadata_valid']}")
            if result['errors']:
                print(f"  Errors: {', '.join(result['errors'])}")
                
        elif args.action == "compare":
            if not args.version or not args.version2:
                parser.error("--version and --version2 required for compare action")
            
            comparison = versioning.compare_models(args.version, args.version2)
            print(f"\n🔄 Comparison: {args.version} vs {args.version2}")
            print(f"  Size difference: {comparison['size_difference_mb']:.1f}MB")
            print(f"  Same architecture: {comparison['architecture_same']}")
            
        elif args.action == "cleanup":
            removed = versioning.cleanup_old_versions(
                keep_latest=args.keep_latest,
                model_type=args.model_type
            )
            print(f"🧹 Removed {len(removed)} old versions")
            
        elif args.action == "export":
            if not args.version or not args.export_path:
                parser.error("--version and --export-path required for export action")
            
            success = versioning.export_model(args.version, args.export_path)
            print(f"Export {'successful' if success else 'failed'}")
            
        elif args.action == "report":
            report = versioning.generate_report()
            
            output_dir = Path(args.output_dir) if args.output_dir else Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / f"model_versioning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📊 Report saved to: {report_file}")
            print(f"Total models: {report['total_models']}")
            print(f"Storage usage: {report['storage_usage']['total_size_gb']:.1f}GB")
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
