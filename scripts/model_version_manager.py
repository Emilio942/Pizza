#!/usr/bin/env python3
"""
Model Versioning System for Dual-Encoder Models
SPATIAL-4.2: Deployment-Pipeline erweitern
"""

import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
from transformers import AutoTokenizer, AutoProcessor

class ModelVersionManager:
    """Manages versioning for Spatial-MLLM dual-encoder models"""
    
    def __init__(self, models_root: str = "./models"):
        self.models_root = Path(models_root)
        self.versions_db = self.models_root / "versions.json"
        self.spatial_models_dir = self.models_root / "spatial_mllm"
        self.standard_models_dir = self.models_root / "standard"
        
        # Create directories
        self.spatial_models_dir.mkdir(parents=True, exist_ok=True)
        self.standard_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing versions database
        self.versions_data = self._load_versions_db()
    
    def _load_versions_db(self) -> Dict[str, Any]:
        """Load the versions database"""
        if self.versions_db.exists():
            try:
                with open(self.versions_db, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load versions DB: {e}")
        
        # Initialize new database
        return {
            "spatial_models": {},
            "standard_models": {},
            "active_versions": {
                "spatial": None,
                "standard": None
            },
            "deployment_history": [],
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_versions_db(self):
        """Save the versions database"""
        self.versions_data["last_updated"] = datetime.now().isoformat()
        with open(self.versions_db, 'w') as f:
            json.dump(self.versions_data, f, indent=2)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash for model files"""
        hasher = hashlib.sha256()
        
        if model_path.is_file():
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        elif model_path.is_dir():
            # Hash all relevant files in directory
            for file_path in sorted(model_path.rglob("*")):
                if file_path.is_file() and file_path.suffix in ['.bin', '.safetensors', '.json', '.txt']:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()[:16]  # Short hash
    
    def register_spatial_model(
        self, 
        model_path: str, 
        version_name: str, 
        description: str = "",
        performance_metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register a new spatial model version"""
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Check if this exact model already exists
        for existing_version, data in self.versions_data["spatial_models"].items():
            if data.get("hash") == model_hash:
                self.logger.warning(f"Model with identical hash already exists: {existing_version}")
                return data
        
        # Create version entry
        timestamp = datetime.now().isoformat()
        version_data = {
            "version": version_name,
            "hash": model_hash,
            "description": description,
            "model_path": str(model_path.absolute()),
            "registered": timestamp,
            "performance_metrics": performance_metrics or {},
            "training_config": training_config or {},
            "model_type": "spatial_dual_encoder",
            "status": "registered"
        }
        
        # Validate model structure
        validation_result = self._validate_spatial_model(model_path)
        version_data["validation"] = validation_result
        
        if not validation_result["valid"]:
            raise ValueError(f"Model validation failed: {validation_result['errors']}")
        
        # Copy model to versioned storage
        versioned_path = self.spatial_models_dir / f"{version_name}_{model_hash}"
        if model_path.is_dir():
            shutil.copytree(model_path, versioned_path, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, versioned_path.with_suffix(model_path.suffix))
        
        version_data["versioned_path"] = str(versioned_path)
        
        # Register in database
        self.versions_data["spatial_models"][version_name] = version_data
        self._save_versions_db()
        
        self.logger.info(f"Registered spatial model version: {version_name}")
        return version_data
    
    def register_standard_model(
        self, 
        model_path: str, 
        version_name: str, 
        description: str = "",
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Register a new standard model version"""
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Create version entry
        timestamp = datetime.now().isoformat()
        version_data = {
            "version": version_name,
            "hash": model_hash,
            "description": description,
            "model_path": str(model_path.absolute()),
            "registered": timestamp,
            "performance_metrics": performance_metrics or {},
            "model_type": "standard_classifier",
            "status": "registered"
        }
        
        # Copy model to versioned storage
        versioned_path = self.standard_models_dir / f"{version_name}_{model_hash}"
        if model_path.is_dir():
            shutil.copytree(model_path, versioned_path, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, versioned_path.with_suffix(model_path.suffix))
        
        version_data["versioned_path"] = str(versioned_path)
        
        # Register in database
        self.versions_data["standard_models"][version_name] = version_data
        self._save_versions_db()
        
        self.logger.info(f"Registered standard model version: {version_name}")
        return version_data
    
    def _validate_spatial_model(self, model_path: Path) -> Dict[str, Any]:
        """Validate spatial model structure and compatibility"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "model_info": {}
        }
        
        try:
            # Check if it's a Hugging Face model directory
            if model_path.is_dir():
                config_path = model_path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            content = f.read().strip()
                            if not content:
                                validation_result["errors"].append("config.json is empty")
                                validation_result["valid"] = False
                                return validation_result
                            config = json.loads(content)
                    except json.JSONDecodeError as e:
                        validation_result["errors"].append(f"Invalid JSON in config.json: {e}")
                        validation_result["valid"] = False
                        return validation_result
                    
                    validation_result["model_info"]["architecture"] = config.get("model_type", "unknown")
                    validation_result["model_info"]["hidden_size"] = config.get("hidden_size", 0)
                    
                    # Check for Qwen2-VL architecture (expected for Spatial-MLLM)
                    if config.get("model_type") != "qwen2_vl":
                        validation_result["warnings"].append(
                            f"Expected qwen2_vl architecture, found: {config.get('model_type')}"
                        )
                
                # Check for required files
                required_files = ["config.json"]
                optional_files = ["model.safetensors", "pytorch_model.bin", "tokenizer.json"]
                
                for req_file in required_files:
                    if not (model_path / req_file).exists():
                        validation_result["errors"].append(f"Missing required file: {req_file}")
                        validation_result["valid"] = False
                
                # Check model size
                model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                validation_result["model_info"]["size_mb"] = model_size / (1024 * 1024)
                
                # Warn if model is very large
                if model_size > 10 * 1024 * 1024 * 1024:  # 10GB
                    validation_result["warnings"].append(f"Large model size: {model_size / (1024**3):.1f}GB")
            
            elif model_path.is_file():
                # Single file model (e.g., .pth, .pkl)
                model_size = model_path.stat().st_size
                validation_result["model_info"]["size_mb"] = model_size / (1024 * 1024)
                
                if model_path.suffix not in ['.pth', '.pt', '.pkl', '.bin', '.safetensors']:
                    validation_result["warnings"].append(f"Unexpected file extension: {model_path.suffix}")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    def set_active_version(self, model_type: str, version_name: str) -> bool:
        """Set the active version for a model type"""
        if model_type not in ["spatial", "standard"]:
            raise ValueError("model_type must be 'spatial' or 'standard'")
        
        models_key = f"{model_type}_models"
        if version_name not in self.versions_data[models_key]:
            raise ValueError(f"Version {version_name} not found for {model_type} models")
        
        # Record deployment
        deployment_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "version": version_name,
            "previous_version": self.versions_data["active_versions"][model_type],
            "deployment_reason": "manual_activation"
        }
        
        self.versions_data["active_versions"][model_type] = version_name
        self.versions_data["deployment_history"].append(deployment_record)
        
        # Update model status
        self.versions_data[models_key][version_name]["status"] = "active"
        
        # Mark previous version as inactive
        for version, data in self.versions_data[models_key].items():
            if version != version_name and data.get("status") == "active":
                data["status"] = "inactive"
        
        self._save_versions_db()
        
        self.logger.info(f"Set active {model_type} model version: {version_name}")
        return True
    
    def get_active_version(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get the currently active version for a model type"""
        if model_type not in ["spatial", "standard"]:
            raise ValueError("model_type must be 'spatial' or 'standard'")
        
        active_version = self.versions_data["active_versions"][model_type]
        if active_version is None:
            return None
        
        models_key = f"{model_type}_models"
        return self.versions_data[models_key].get(active_version)
    
    def list_versions(self, model_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """List all registered model versions"""
        if model_type is None:
            return {
                "spatial": list(self.versions_data["spatial_models"].values()),
                "standard": list(self.versions_data["standard_models"].values())
            }
        elif model_type == "spatial":
            return {"spatial": list(self.versions_data["spatial_models"].values())}
        elif model_type == "standard":
            return {"standard": list(self.versions_data["standard_models"].values())}
        else:
            raise ValueError("model_type must be 'spatial', 'standard', or None")
    
    def compare_versions(self, model_type: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        models_key = f"{model_type}_models"
        
        if version1 not in self.versions_data[models_key]:
            raise ValueError(f"Version {version1} not found")
        if version2 not in self.versions_data[models_key]:
            raise ValueError(f"Version {version2} not found")
        
        v1_data = self.versions_data[models_key][version1]
        v2_data = self.versions_data[models_key][version2]
        
        comparison = {
            "version1": {
                "name": version1,
                "registered": v1_data["registered"],
                "metrics": v1_data.get("performance_metrics", {}),
                "hash": v1_data["hash"]
            },
            "version2": {
                "name": version2,
                "registered": v2_data["registered"],
                "metrics": v2_data.get("performance_metrics", {}),
                "hash": v2_data["hash"]
            },
            "differences": {}
        }
        
        # Compare performance metrics
        v1_metrics = v1_data.get("performance_metrics", {})
        v2_metrics = v2_data.get("performance_metrics", {})
        
        all_metrics = set(v1_metrics.keys()) | set(v2_metrics.keys())
        for metric in all_metrics:
            v1_val = v1_metrics.get(metric)
            v2_val = v2_metrics.get(metric)
            
            if v1_val is not None and v2_val is not None:
                diff = v2_val - v1_val
                comparison["differences"][metric] = {
                    "v1": v1_val,
                    "v2": v2_val,
                    "difference": diff,
                    "improvement": diff > 0 if "accuracy" in metric.lower() or "f1" in metric.lower() else diff < 0
                }
        
        return comparison
    
    def rollback_to_version(self, model_type: str, version_name: str, reason: str = "") -> bool:
        """Rollback to a previous version"""
        current_active = self.versions_data["active_versions"][model_type]
        
        if current_active == version_name:
            self.logger.warning(f"Version {version_name} is already active")
            return False
        
        # Record rollback
        deployment_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "version": version_name,
            "previous_version": current_active,
            "deployment_reason": f"rollback: {reason}"
        }
        
        self.versions_data["deployment_history"].append(deployment_record)
        
        # Set new active version
        return self.set_active_version(model_type, version_name)
    
    def cleanup_old_versions(self, keep_last_n: int = 5) -> Dict[str, int]:
        """Clean up old model versions, keeping the last N versions"""
        cleanup_stats = {"spatial": 0, "standard": 0}
        
        for model_type in ["spatial", "standard"]:
            models_key = f"{model_type}_models"
            models = self.versions_data[models_key]
            
            # Sort by registration date
            sorted_versions = sorted(
                models.items(),
                key=lambda x: x[1]["registered"],
                reverse=True
            )
            
            # Keep active version + last N versions
            active_version = self.versions_data["active_versions"][model_type]
            versions_to_keep = {active_version} if active_version else set()
            
            for version, data in sorted_versions[:keep_last_n]:
                versions_to_keep.add(version)
            
            # Remove old versions
            for version, data in list(models.items()):
                if version not in versions_to_keep:
                    # Remove files
                    versioned_path = Path(data["versioned_path"])
                    if versioned_path.exists():
                        if versioned_path.is_dir():
                            shutil.rmtree(versioned_path)
                        else:
                            versioned_path.unlink()
                    
                    # Remove from database
                    del models[version]
                    cleanup_stats[model_type] += 1
        
        self._save_versions_db()
        self.logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def export_deployment_report(self, output_path: str) -> Dict[str, Any]:
        """Export a deployment report"""
        report = {
            "generated": datetime.now().isoformat(),
            "active_versions": self.versions_data["active_versions"],
            "total_spatial_models": len(self.versions_data["spatial_models"]),
            "total_standard_models": len(self.versions_data["standard_models"]),
            "deployment_history": self.versions_data["deployment_history"][-10:],  # Last 10 deployments
            "model_summaries": {}
        }
        
        # Add model summaries
        for model_type in ["spatial", "standard"]:
            models_key = f"{model_type}_models"
            active_version = self.versions_data["active_versions"][model_type]
            
            if active_version and active_version in self.versions_data[models_key]:
                active_data = self.versions_data[models_key][active_version]
                report["model_summaries"][model_type] = {
                    "active_version": active_version,
                    "registered": active_data["registered"],
                    "performance_metrics": active_data.get("performance_metrics", {}),
                    "model_info": active_data.get("validation", {}).get("model_info", {})
                }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Version Manager for Spatial-MLLM")
    parser.add_argument("--models-root", default="./models", help="Root directory for models")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Register spatial model
    register_spatial = subparsers.add_parser("register-spatial", help="Register spatial model")
    register_spatial.add_argument("model_path", help="Path to model")
    register_spatial.add_argument("version_name", help="Version name")
    register_spatial.add_argument("--description", default="", help="Description")
    
    # Register standard model
    register_standard = subparsers.add_parser("register-standard", help="Register standard model")
    register_standard.add_argument("model_path", help="Path to model")
    register_standard.add_argument("version_name", help="Version name")
    register_standard.add_argument("--description", default="", help="Description")
    
    # Set active version
    set_active = subparsers.add_parser("set-active", help="Set active version")
    set_active.add_argument("model_type", choices=["spatial", "standard"])
    set_active.add_argument("version_name", help="Version name")
    
    # List versions
    list_versions = subparsers.add_parser("list", help="List versions")
    list_versions.add_argument("--type", choices=["spatial", "standard"], help="Model type")
    
    # Export report
    export_report = subparsers.add_parser("export-report", help="Export deployment report")
    export_report.add_argument("output_path", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ModelVersionManager(args.models_root)
    
    if args.command == "register-spatial":
        result = manager.register_spatial_model(args.model_path, args.version_name, args.description)
        print(f"Registered spatial model: {result['version']}")
    
    elif args.command == "register-standard":
        result = manager.register_standard_model(args.model_path, args.version_name, args.description)
        print(f"Registered standard model: {result['version']}")
    
    elif args.command == "set-active":
        manager.set_active_version(args.model_type, args.version_name)
        print(f"Set active {args.model_type} version: {args.version_name}")
    
    elif args.command == "list":
        versions = manager.list_versions(args.type)
        for model_type, model_list in versions.items():
            print(f"\n{model_type.upper()} Models:")
            for model in model_list:
                status = "🟢" if model.get("status") == "active" else "⚪"
                print(f"  {status} {model['version']} - {model['registered']}")
    
    elif args.command == "export-report":
        report = manager.export_deployment_report(args.output_path)
        print(f"Exported deployment report to: {args.output_path}")


if __name__ == "__main__":
    main()
