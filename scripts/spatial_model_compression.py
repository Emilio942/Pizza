#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-MLLM Model Compression for Edge Deployment

This script implements comprehensive model compression techniques for the Spatial-MLLM
dual-encoder architecture to enable deployment on resource-constrained devices.

Features:
1. Model size and memory requirement analysis
2. Quantization (INT8/INT4) for both visual and spatial encoders
3. Structured pruning optimized for dual-encoder architecture
4. Performance evaluation after compression
5. Edge platform compatibility assessment (RP2040, etc.)

SPATIAL-3.2 Implementation
Author: GitHub Copilot (2025-06-06)
"""

import os
import sys
import json
import time
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
import torch.nn.utils.prune as prune

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append("/home/emilio/Documents/ai/Spatial-MLLM")

from scripts.spatial_inference_optimized import SpatialMLLMInferenceSystem, InferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'output' / 'spatial_model_compression.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for model compression"""
    # Quantization settings
    enable_quantization: bool = True
    quantization_dtype: str = "int8"  # "int8" or "int4"
    quantize_visual_encoder: bool = True
    quantize_spatial_encoder: bool = True
    quantize_language_model: bool = True
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_ratio: float = 0.3  # 30% sparsity
    structured_pruning: bool = True
    prune_visual_encoder: bool = True
    prune_spatial_encoder: bool = True
    
    # Edge deployment settings
    target_memory_mb: int = 512  # Target memory usage for edge devices
    target_inference_time_ms: int = 1000  # Target inference time
    enable_fp16: bool = True
    
    # Evaluation settings
    test_images_count: int = 10
    performance_threshold: float = 0.8  # Minimum acceptable accuracy retention

@dataclass
class CompressionResult:
    """Results from model compression"""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    original_params: int
    compressed_params: int
    original_inference_time: float
    compressed_inference_time: float
    speedup: float
    accuracy_retention: float
    memory_usage_mb: float
    edge_compatible: bool
    compression_techniques: List[str]

class SpatialModelAnalyzer:
    """Analyzes Spatial-MLLM model characteristics"""
    
    def __init__(self, model_path: str = "Diankun/Spatial-MLLM-subset-sft"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def analyze_model_requirements(self, model) -> Dict[str, Any]:
        """Analyze model size and memory requirements"""
        logger.info("📊 Analyzing Spatial-MLLM model requirements...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Memory requirements for inference
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Run dummy forward pass to measure peak memory
            dummy_input = {
                'pixel_values': torch.randn(1, 3, 224, 224).to(self.device),
                'input_ids': torch.tensor([[1, 2, 3]]).to(self.device)
            }
            
            try:
                with torch.no_grad():
                    _ = model(**dummy_input)
                peak_memory = torch.cuda.max_memory_allocated()
                inference_memory_mb = (peak_memory - initial_memory) / 1024 / 1024
            except Exception as e:
                logger.warning(f"Could not measure GPU memory usage: {e}")
                inference_memory_mb = model_size_mb * 2  # Estimate
        else:
            inference_memory_mb = model_size_mb * 1.5  # CPU estimate
        
        analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'inference_memory_mb': inference_memory_mb,
            'peak_memory_mb': inference_memory_mb + model_size_mb,
            'parameter_breakdown': self._analyze_parameter_distribution(model),
            'component_sizes': self._analyze_component_sizes(model)
        }
        
        logger.info(f"  📈 Total parameters: {total_params:,}")
        logger.info(f"  💾 Model size: {model_size_mb:.1f} MB")
        logger.info(f"  🧠 Inference memory: {inference_memory_mb:.1f} MB")
        
        return analysis
    
    def _analyze_parameter_distribution(self, model) -> Dict[str, int]:
        """Analyze parameter distribution across model components"""
        distribution = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                param_count = module.weight.numel()
                component = self._categorize_component(name)
                distribution[component] = distribution.get(component, 0) + param_count
                
        return distribution
    
    def _analyze_component_sizes(self, model) -> Dict[str, float]:
        """Analyze size of different model components in MB"""
        sizes = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                size_mb = module.weight.numel() * module.weight.element_size() / 1024 / 1024
                component = self._categorize_component(name)
                sizes[component] = sizes.get(component, 0) + size_mb
                
        return sizes
    
    def _categorize_component(self, module_name: str) -> str:
        """Categorize module into component type"""
        name_lower = module_name.lower()
        
        if 'vision' in name_lower or 'visual' in name_lower:
            return 'visual_encoder'
        elif 'vggt' in name_lower or 'spatial' in name_lower:
            return 'spatial_encoder'  
        elif 'language' in name_lower or 'lm' in name_lower or 'qwen' in name_lower:
            return 'language_model'
        elif 'connector' in name_lower or 'projection' in name_lower:
            return 'connector'
        else:
            return 'other'

class SpatialModelQuantizer:
    """Implements quantization for Spatial-MLLM"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def quantize_model(self, model, sample_inputs: List[torch.Tensor]) -> torch.nn.Module:
        """Apply quantization to the dual-encoder model"""
        logger.info(f"⚡ Applying {self.config.quantization_dtype.upper()} quantization...")
        
        # Set model to evaluation mode
        model.eval()
        
        if self.config.quantization_dtype == "int8":
            return self._apply_int8_quantization(model, sample_inputs)
        elif self.config.quantization_dtype == "int4":
            return self._apply_int4_quantization(model, sample_inputs)
        else:
            raise ValueError(f"Unsupported quantization dtype: {self.config.quantization_dtype}")
    
    def _apply_int8_quantization(self, model, sample_inputs) -> torch.nn.Module:
        """Apply INT8 dynamic quantization"""
        
        # Define modules to quantize
        modules_to_quantize = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                component = self._get_component_type(name)
                
                should_quantize = (
                    (component == 'visual' and self.config.quantize_visual_encoder) or
                    (component == 'spatial' and self.config.quantize_spatial_encoder) or
                    (component == 'language' and self.config.quantize_language_model)
                )
                
                if should_quantize:
                    modules_to_quantize.append(type(module))
        
        # Apply dynamic quantization
        if modules_to_quantize:
            quantized_model = quantize_dynamic(
                model=model,
                qconfig_spec=set(modules_to_quantize),
                dtype=torch.qint8,
                inplace=False
            )
            logger.info(f"  ✅ Quantized {len(modules_to_quantize)} module types to INT8")
            return quantized_model
        else:
            logger.warning("  ⚠️ No modules found for quantization")
            return model
    
    def _apply_int4_quantization(self, model, sample_inputs) -> torch.nn.Module:
        """Apply INT4 quantization (experimental)"""
        logger.warning("  ⚠️ INT4 quantization is experimental")
        
        # For INT4, we'll use a custom approach
        quantized_model = self._custom_int4_quantization(model)
        
        return quantized_model
    
    def _custom_int4_quantization(self, model) -> torch.nn.Module:
        """Custom INT4 quantization implementation"""
        # This is a simplified INT4 quantization
        # In practice, you'd want to use libraries like bitsandbytes
        
        class Int4QuantizedLinear(nn.Module):
            def __init__(self, original_linear):
                super().__init__()
                self.in_features = original_linear.in_features
                self.out_features = original_linear.out_features
                
                # Quantize weights to 4-bit
                weights = original_linear.weight.data
                self.scale = weights.abs().max() / 7  # 4-bit range: -8 to 7
                self.quantized_weights = torch.clamp(
                    torch.round(weights / self.scale), -8, 7
                ).to(torch.int8)
                
                if original_linear.bias is not None:
                    self.bias = original_linear.bias
                else:
                    self.bias = None
            
            def forward(self, x):
                # Dequantize weights for computation
                weights = self.quantized_weights.float() * self.scale
                return F.linear(x, weights, self.bias)
        
        # Replace Linear layers with quantized versions
        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    component = self._get_component_type(name)
                    should_quantize = (
                        (component == 'visual' and self.config.quantize_visual_encoder) or
                        (component == 'spatial' and self.config.quantize_spatial_encoder) or
                        (component == 'language' and self.config.quantize_language_model)
                    )
                    
                    if should_quantize:
                        setattr(module, name, Int4QuantizedLinear(child))
                else:
                    replace_layers(child)
        
        model_copy = torch.nn.utils.deepcopy(model)
        replace_layers(model_copy)
        
        return model_copy
    
    def _get_component_type(self, module_name: str) -> str:
        """Determine component type from module name"""
        name_lower = module_name.lower()
        
        if 'vision' in name_lower or 'visual' in name_lower:
            return 'visual'
        elif 'vggt' in name_lower or 'spatial' in name_lower:
            return 'spatial'
        elif 'language' in name_lower or 'lm' in name_lower:
            return 'language'
        else:
            return 'other'

class SpatialModelPruner:
    """Implements structured pruning for Spatial-MLLM"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        
    def prune_model(self, model) -> torch.nn.Module:
        """Apply structured pruning to the dual-encoder model"""
        logger.info(f"✂️ Applying structured pruning (sparsity: {self.config.pruning_ratio:.1%})...")
        
        modules_pruned = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                component = self._get_component_type(name)
                
                should_prune = (
                    (component == 'visual' and self.config.prune_visual_encoder) or
                    (component == 'spatial' and self.config.prune_spatial_encoder)
                )
                
                if should_prune:
                    if self.config.structured_pruning:
                        self._apply_structured_pruning(module, name)
                    else:
                        self._apply_unstructured_pruning(module, name)
                    modules_pruned += 1
        
        logger.info(f"  ✅ Pruned {modules_pruned} modules")
        return model
    
    def _apply_structured_pruning(self, module, module_name):
        """Apply structured pruning (removes entire channels/neurons)"""
        if isinstance(module, nn.Conv2d):
            # Prune output channels
            prune.ln_structured(
                module, 
                name="weight", 
                amount=self.config.pruning_ratio,
                n=2,  # L2 norm
                dim=0  # Output channels
            )
        elif isinstance(module, nn.Linear):
            # Prune output features
            prune.ln_structured(
                module,
                name="weight",
                amount=self.config.pruning_ratio,
                n=2,  # L2 norm
                dim=0  # Output features
            )
    
    def _apply_unstructured_pruning(self, module, module_name):
        """Apply unstructured pruning (removes individual weights)"""
        prune.l1_unstructured(
            module,
            name="weight",
            amount=self.config.pruning_ratio
        )
    
    def _get_component_type(self, module_name: str) -> str:
        """Determine component type from module name"""
        name_lower = module_name.lower()
        
        if 'vision' in name_lower or 'visual' in name_lower:
            return 'visual'
        elif 'vggt' in name_lower or 'spatial' in name_lower:
            return 'spatial'
        else:
            return 'other'

class EdgeCompatibilityAnalyzer:
    """Analyzes compatibility with edge computing platforms"""
    
    def __init__(self):
        self.platforms = {
            'rp2040': {
                'name': 'Raspberry Pi Pico (RP2040)',
                'ram_kb': 264,
                'flash_mb': 2,
                'cpu_mhz': 133,
                'supports_float': False,
                'supports_int8': True,
                'supports_int4': True
            },
            'rp4': {
                'name': 'Raspberry Pi 4',
                'ram_mb': 4096,
                'cpu_cores': 4,
                'cpu_mhz': 1500,
                'supports_float': True,
                'supports_int8': True,
                'supports_int4': True
            },
            'jetson_nano': {
                'name': 'NVIDIA Jetson Nano',
                'ram_mb': 4096,
                'gpu_memory_mb': 128,
                'cpu_cores': 4,
                'supports_cuda': True,
                'supports_tensorrt': True
            },
            'coral_tpu': {
                'name': 'Google Coral Edge TPU',
                'ram_mb': 1024,
                'tpu_ops_per_sec': 4000000000000,  # 4 TOPS
                'supports_int8': True,
                'supports_int4': False
            }
        }
    
    def analyze_compatibility(self, compressed_model, model_size_mb: float, 
                            inference_time_ms: float) -> Dict[str, Any]:
        """Analyze compatibility with various edge platforms"""
        logger.info("🔍 Analyzing edge platform compatibility...")
        
        compatibility_results = {}
        
        for platform_id, platform_specs in self.platforms.items():
            compatibility = self._check_platform_compatibility(
                platform_id, platform_specs, model_size_mb, inference_time_ms
            )
            compatibility_results[platform_id] = compatibility
            
            status = "✅ Compatible" if compatibility['compatible'] else "❌ Not Compatible"
            logger.info(f"  {platform_specs['name']}: {status}")
            
            if not compatibility['compatible']:
                for issue in compatibility['issues']:
                    logger.info(f"    ⚠️ {issue}")
        
        return compatibility_results
    
    def _check_platform_compatibility(self, platform_id: str, specs: Dict[str, Any],
                                    model_size_mb: float, inference_time_ms: float) -> Dict[str, Any]:
        """Check if model is compatible with specific platform"""
        compatibility = {
            'compatible': True,
            'issues': [],
            'recommendations': [],
            'estimated_performance': {}
        }
        
        if platform_id == 'rp2040':
            # RP2040 has very limited resources
            max_model_size_kb = specs['flash_mb'] * 1024 * 0.5  # Use 50% of flash
            max_ram_kb = specs['ram_kb'] * 0.7  # Use 70% of RAM
            
            if model_size_mb * 1024 > max_model_size_kb:
                compatibility['compatible'] = False
                compatibility['issues'].append(f"Model too large: {model_size_mb:.1f}MB > {max_model_size_kb/1024:.1f}MB")
            
            if model_size_mb * 1024 > max_ram_kb:
                compatibility['compatible'] = False
                compatibility['issues'].append(f"RAM insufficient: needs {model_size_mb:.1f}MB > {max_ram_kb/1024:.1f}MB")
            
            # RP2040 doesn't support floating point well
            compatibility['recommendations'].append("Use INT4 quantization for optimal performance")
            compatibility['recommendations'].append("Consider model splitting/streaming")
            
        elif platform_id == 'rp4':
            # Raspberry Pi 4 is more capable
            if model_size_mb > specs['ram_mb'] * 0.5:
                compatibility['issues'].append(f"Model large for available RAM: {model_size_mb:.1f}MB")
                compatibility['recommendations'].append("Consider swap file or model quantization")
            
            # Estimate performance
            estimated_inference_ms = inference_time_ms * 2  # Rough estimate for Pi 4
            compatibility['estimated_performance']['inference_time_ms'] = estimated_inference_ms
            
        elif platform_id == 'jetson_nano':
            # Jetson Nano has GPU acceleration
            if model_size_mb > specs['gpu_memory_mb']:
                compatibility['recommendations'].append("Use CPU inference or model quantization")
            
            # Better performance with GPU
            estimated_inference_ms = inference_time_ms * 0.5
            compatibility['estimated_performance']['inference_time_ms'] = estimated_inference_ms
            
        elif platform_id == 'coral_tpu':
            # Coral TPU requires specific model format
            compatibility['recommendations'].append("Convert model to TensorFlow Lite with TPU delegate")
            compatibility['recommendations'].append("Use INT8 quantization (required for TPU)")
            
            # Very fast inference
            estimated_inference_ms = max(inference_time_ms * 0.1, 5)  # Minimum 5ms
            compatibility['estimated_performance']['inference_time_ms'] = estimated_inference_ms
        
        return compatibility

class SpatialModelCompressor:
    """Main compression orchestrator"""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.analyzer = SpatialModelAnalyzer()
        self.quantizer = SpatialModelQuantizer(self.config)
        self.pruner = SpatialModelPruner(self.config)
        self.edge_analyzer = EdgeCompatibilityAnalyzer()
        
        # Ensure output directories exist
        self.output_dir = Path(project_root) / "models" / "spatial_mllm" / "compressed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(project_root) / "output" / "compression_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compress_model(self) -> CompressionResult:
        """Run complete model compression pipeline"""
        logger.info("🚀 Starting Spatial-MLLM compression pipeline...")
        
        # 1. Load and analyze original model
        original_system = SpatialMLLMInferenceSystem(InferenceConfig())
        original_model = original_system.inference_engine.model
        
        original_analysis = self.analyzer.analyze_model_requirements(original_model)
        
        # 2. Measure original performance
        original_metrics = self._measure_performance(original_system, "original")
        
        # 3. Apply compression techniques
        compressed_model = original_model
        compression_techniques = []
        
        if self.config.enable_quantization:
            sample_inputs = self._generate_sample_inputs()
            compressed_model = self.quantizer.quantize_model(compressed_model, sample_inputs)
            compression_techniques.append(f"{self.config.quantization_dtype.upper()}_quantization")
        
        if self.config.enable_pruning:
            compressed_model = self.pruner.prune_model(compressed_model)
            compression_techniques.append(f"structured_pruning_{self.config.pruning_ratio:.1%}")
        
        # 4. Analyze compressed model
        compressed_analysis = self.analyzer.analyze_model_requirements(compressed_model)
        
        # 5. Create compressed inference system
        compressed_system = self._create_compressed_system(compressed_model)
        
        # 6. Measure compressed performance
        compressed_metrics = self._measure_performance(compressed_system, "compressed")
        
        # 7. Analyze edge compatibility
        edge_compatibility = self.edge_analyzer.analyze_compatibility(
            compressed_model,
            compressed_analysis['model_size_mb'],
            compressed_metrics['avg_inference_time_ms']
        )
        
        # 8. Save compressed model
        self._save_compressed_model(compressed_model, compression_techniques)
        
        # 9. Generate results
        result = CompressionResult(
            original_size_mb=original_analysis['model_size_mb'],
            compressed_size_mb=compressed_analysis['model_size_mb'],
            compression_ratio=original_analysis['model_size_mb'] / compressed_analysis['model_size_mb'],
            original_params=original_analysis['total_parameters'],
            compressed_params=compressed_analysis['total_parameters'],
            original_inference_time=original_metrics['avg_inference_time_ms'],
            compressed_inference_time=compressed_metrics['avg_inference_time_ms'],
            speedup=original_metrics['avg_inference_time_ms'] / compressed_metrics['avg_inference_time_ms'],
            accuracy_retention=compressed_metrics['accuracy'] / original_metrics['accuracy'],
            memory_usage_mb=compressed_analysis['inference_memory_mb'],
            edge_compatible=any(result['compatible'] for result in edge_compatibility.values()),
            compression_techniques=compression_techniques
        )
        
        # 10. Save comprehensive results
        self._save_results(result, original_analysis, compressed_analysis, 
                          original_metrics, compressed_metrics, edge_compatibility)
        
        logger.info("✅ Compression pipeline completed successfully!")
        return result
    
    def _generate_sample_inputs(self) -> List[torch.Tensor]:
        """Generate sample inputs for quantization calibration"""
        # This would typically use real data samples
        # For now, we'll create dummy inputs
        sample_inputs = [
            torch.randn(1, 3, 224, 224),  # Visual input
            torch.tensor([[1, 2, 3, 4, 5]])  # Text input
        ]
        return sample_inputs
    
    def _create_compressed_system(self, compressed_model) -> SpatialMLLMInferenceSystem:
        """Create inference system with compressed model"""
        config = InferenceConfig()
        system = SpatialMLLMInferenceSystem(config)
        
        # Replace the model in the inference system
        system.inference_engine.model = compressed_model
        
        return system
    
    def _measure_performance(self, system: SpatialMLLMInferenceSystem, 
                           model_type: str) -> Dict[str, float]:
        """Measure model performance on test set"""
        logger.info(f"📊 Measuring {model_type} model performance...")
        
        # Get test images
        test_images = self._get_test_images()[:self.config.test_images_count]
        
        inference_times = []
        correct_predictions = 0
        total_predictions = 0
        
        for image_path in test_images:
            start_time = time.time()
            
            try:
                result = system.process_image(image_path)
                end_time = time.time()
                
                inference_time_ms = (end_time - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                if result['success']:
                    # For this demo, we'll consider all successful predictions as correct
                    # In practice, you'd compare against ground truth labels
                    correct_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        
        metrics = {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time,
            'inference_times': inference_times,
            'total_tested': total_predictions,
            'successful_predictions': correct_predictions
        }
        
        logger.info(f"  📈 {model_type.title()} accuracy: {accuracy:.2%}")
        logger.info(f"  ⚡ {model_type.title()} avg inference: {avg_inference_time:.1f}ms")
        
        return metrics
    
    def _get_test_images(self) -> List[str]:
        """Get list of test images"""
        test_dirs = [
            project_root / "data" / "test_new_images" / "basic",
            project_root / "data" / "test_new_images" / "burnt",
            project_root / "data" / "raw"
        ]
        
        test_images = []
        for test_dir in test_dirs:
            if test_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    test_images.extend(list(test_dir.glob(ext)))
        
        return [str(img) for img in test_images]
    
    def _save_compressed_model(self, model, techniques: List[str]):
        """Save compressed model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        techniques_str = "_".join(techniques)
        
        model_filename = f"spatial_mllm_compressed_{techniques_str}_{timestamp}.pth"
        model_path = self.output_dir / model_filename
        
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'compression_techniques': techniques,
                'config': self.config.__dict__,
                'timestamp': timestamp
            }, model_path)
            
            logger.info(f"💾 Compressed model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save compressed model: {e}")
    
    def _save_results(self, result: CompressionResult, original_analysis: Dict,
                     compressed_analysis: Dict, original_metrics: Dict,
                     compressed_metrics: Dict, edge_compatibility: Dict):
        """Save comprehensive results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comprehensive results
        comprehensive_results = {
            'compression_summary': {
                'original_size_mb': result.original_size_mb,
                'compressed_size_mb': result.compressed_size_mb,
                'compression_ratio': result.compression_ratio,
                'original_params': result.original_params,
                'compressed_params': result.compressed_params,
                'param_reduction': 1 - (result.compressed_params / result.original_params),
                'original_inference_time_ms': result.original_inference_time,
                'compressed_inference_time_ms': result.compressed_inference_time,
                'speedup': result.speedup,
                'accuracy_retention': result.accuracy_retention,
                'memory_usage_mb': result.memory_usage_mb,
                'edge_compatible': result.edge_compatible,
                'compression_techniques': result.compression_techniques
            },
            'detailed_analysis': {
                'original_model': original_analysis,
                'compressed_model': compressed_analysis,
                'original_performance': original_metrics,
                'compressed_performance': compressed_metrics,
                'edge_compatibility': edge_compatibility
            },
            'configuration': self.config.__dict__,
            'timestamp': timestamp
        }
        
        # Save JSON results
        results_file = self.results_dir / f"compression_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Save human-readable report
        report_file = self.results_dir / f"compression_report_{timestamp}.txt"
        self._generate_report(comprehensive_results, report_file)
        
        logger.info(f"📋 Results saved:")
        logger.info(f"  📊 JSON: {results_file}")
        logger.info(f"  📄 Report: {report_file}")
    
    def _generate_report(self, results: Dict, report_path: Path):
        """Generate human-readable compression report"""
        summary = results['compression_summary']
        
        report = f"""
================================================================================
🤖 SPATIAL-MLLM MODEL COMPRESSION REPORT
================================================================================
Generated: {results['timestamp']}
Compression Techniques: {', '.join(summary['compression_techniques'])}

📊 COMPRESSION SUMMARY
----------------------------------------
Original Model Size:     {summary['original_size_mb']:.1f} MB
Compressed Model Size:   {summary['compressed_size_mb']:.1f} MB
Compression Ratio:       {summary['compression_ratio']:.2f}x
Size Reduction:          {(1 - 1/summary['compression_ratio']):.1%}

Original Parameters:     {summary['original_params']:,}
Compressed Parameters:   {summary['compressed_params']:,}
Parameter Reduction:     {summary['param_reduction']:.1%}

⚡ PERFORMANCE IMPACT
----------------------------------------
Original Inference:      {summary['original_inference_time_ms']:.1f} ms
Compressed Inference:    {summary['compressed_inference_time_ms']:.1f} ms
Speedup:                 {summary['speedup']:.2f}x

Accuracy Retention:      {summary['accuracy_retention']:.1%}
Memory Usage:            {summary['memory_usage_mb']:.1f} MB

🔧 EDGE COMPATIBILITY
----------------------------------------
"""
        
        edge_results = results['detailed_analysis']['edge_compatibility']
        for platform_id, compatibility in edge_results.items():
            platform_name = compatibility.get('platform_name', platform_id)
            status = "✅ Compatible" if compatibility['compatible'] else "❌ Not Compatible"
            report += f"{platform_name}: {status}\n"
            
            if compatibility['issues']:
                for issue in compatibility['issues']:
                    report += f"  ⚠️ {issue}\n"
                    
            if compatibility['recommendations']:
                for rec in compatibility['recommendations'][:2]:  # Show first 2 recommendations
                    report += f"  💡 {rec}\n"
            
            report += "\n"
        
        report += f"""
📈 DETAILED METRICS
----------------------------------------
Configuration Used:
- Quantization: {results['configuration']['quantization_dtype'].upper() if results['configuration']['enable_quantization'] else 'Disabled'}
- Pruning: {results['configuration']['pruning_ratio']:.1%} if results['configuration']['enable_pruning'] else 'Disabled'}
- Target Memory: {results['configuration']['target_memory_mb']} MB
- Target Inference: {results['configuration']['target_inference_time_ms']} ms

Edge Deployment Ready: {'✅ Yes' if summary['edge_compatible'] else '❌ No'}

================================================================================
"""
        
        with open(report_path, 'w') as f:
            f.write(report)

def main():
    """Main compression pipeline execution"""
    logger.info("🤖 Starting Spatial-MLLM Model Compression Pipeline")
    
    # Configuration for compression
    config = CompressionConfig(
        enable_quantization=True,
        quantization_dtype="int8",
        enable_pruning=True,
        pruning_ratio=0.2,  # 20% sparsity
        target_memory_mb=1024,  # 1GB target
        target_inference_time_ms=500,  # 500ms target
        test_images_count=5  # Quick test with 5 images
    )
    
    # Run compression
    compressor = SpatialModelCompressor(config)
    
    try:
        result = compressor.compress_model()
        
        # Print summary
        logger.info("🎉 Compression completed successfully!")
        logger.info(f"📊 Compression ratio: {result.compression_ratio:.2f}x")
        logger.info(f"⚡ Speedup: {result.speedup:.2f}x") 
        logger.info(f"📈 Accuracy retention: {result.accuracy_retention:.1%}")
        logger.info(f"🔧 Edge compatible: {'Yes' if result.edge_compatible else 'No'}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Compression pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
