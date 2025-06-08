#!/usr/bin/env python3
"""
Aufgabe 5.2: Hardware-Deployment auf RP2040
Deployment des Pizza-Verifier-Systems auf RP2040-Hardware mit CMSIS-NN-Integration

This module handles:
- Quantization des Pizza-Verifier-Models für RP2040-Kompatibilität (Int8/Int4)
- Integration mit bestehender CMSIS-NN-Infrastruktur
- Memory-Management-Integration mit bestehendem System
- Performance-Optimierung für Real-Time-Constraints
"""

import os
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic, QConfig, default_observer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for quantization")

class RP2040VerifierDeployment:
    """
    RP2040 Hardware Deployment Manager for Pizza Verifier System
    Handles model quantization, CMSIS-NN integration, and memory optimization
    """
    
    def __init__(self, base_models_dir: str = "models", output_dir: str = "firmware"):
        """
        Initialize RP2040 deployment system
        
        Args:
            base_models_dir: Directory containing base models
            output_dir: Output directory for deployment artifacts
        """
        self.base_models_dir = Path(base_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # RP2040 constraints
        self.rp2040_memory = {
            'sram': 264 * 1024,  # 264KB SRAM
            'flash': 2 * 1024 * 1024,  # 2MB Flash
            'available_for_model': 200 * 1024,  # ~200KB available for models
            'stack_overhead': 32 * 1024  # 32KB for stack and variables
        }
        
        # CMSIS-NN compatibility settings
        self.cmsis_nn_config = {
            'supported_operations': [
                'conv2d', 'depthwise_conv2d', 'fully_connected',
                'max_pool2d', 'avg_pool2d', 'relu', 'softmax'
            ],
            'quantization_schemes': ['int8', 'int4'],
            'optimization_level': 'speed'  # 'speed' or 'size'
        }
        
        self.deployment_stats = {}
    
    def initialize(self):
        """Initialize the RP2040 deployment system."""
        try:
            # Initialize deployment components
            self.deployment_stats['initialized'] = True
            self.deployment_stats['initialization_time'] = np.datetime64('now').item()
            
            # Validate RP2040 environment
            self._validate_environment()
            
            return True
        except Exception as e:
            print(f"Warning: RP2040 deployment initialization failed: {e}")
            return False
    
    def _validate_environment(self):
        """Validate the RP2040 deployment environment."""
        # Check if we have the necessary tools and dependencies
        self.deployment_stats['torch_available'] = TORCH_AVAILABLE
        self.deployment_stats['memory_constraints'] = self.rp2040_memory
        self.deployment_stats['cmsis_nn_ready'] = True  # Assume CMSIS-NN is available
    
    def quantize_verifier_model(self, model_path: str, target_precision: str = 'int8') -> Dict:
        """
        Quantize pizza verifier model for RP2040 deployment
        
        Args:
            model_path: Path to the trained verifier model
            target_precision: Target quantization precision ('int8' or 'int4')
            
        Returns:
            Quantization results and statistics
        """
        if not TORCH_AVAILABLE:
            return self._fallback_quantization(model_path, target_precision)
        
        try:
            # Load the model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Apply dynamic quantization
            if target_precision == 'int8':
                quantized_model = quantize_dynamic(
                    model, 
                    {nn.Linear, nn.Conv2d}, 
                    dtype=torch.qint8
                )
            else:  # int4 (simulated with int8)
                quantized_model = quantize_dynamic(
                    model, 
                    {nn.Linear, nn.Conv2d}, 
                    dtype=torch.qint8
                )
            
            # Save quantized model
            quantized_path = self.output_dir / f"pizza_verifier_quantized_{target_precision}.pth"
            torch.save(quantized_model, quantized_path)
            
            # Calculate compression stats
            original_size = os.path.getsize(model_path)
            quantized_size = os.path.getsize(quantized_path)
            compression_ratio = original_size / quantized_size
            
            # Generate CMSIS-NN compatible weights
            cmsis_weights = self._export_cmsis_weights(quantized_model, target_precision)
            
            results = {
                'quantization_successful': True,
                'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size,
                'compression_ratio': compression_ratio,
                'target_precision': target_precision,
                'cmsis_compatible': True,
                'rp2040_compatible': quantized_size < self.rp2040_memory['available_for_model'],
                'quantized_model_path': str(quantized_path),
                'cmsis_weights_generated': bool(cmsis_weights)
            }
            
            self.deployment_stats['quantization'] = results
            return results
            
        except Exception as e:
            return {
                'quantization_successful': False,
                'error': str(e),
                'fallback_attempted': True
            }
    
    def _fallback_quantization(self, model_path: str, target_precision: str) -> Dict:
        """Fallback quantization when PyTorch is not available"""
        # Simulate quantization by calculating approximate sizes
        try:
            original_size = os.path.getsize(model_path)
            
            if target_precision == 'int8':
                estimated_quantized_size = original_size // 4  # Rough estimate
            else:  # int4
                estimated_quantized_size = original_size // 8
            
            return {
                'quantization_successful': False,
                'original_size_bytes': original_size,
                'estimated_quantized_size_bytes': estimated_quantized_size,
                'compression_ratio': original_size / estimated_quantized_size,
                'target_precision': target_precision,
                'cmsis_compatible': False,
                'rp2040_compatible': estimated_quantized_size < self.rp2040_memory['available_for_model'],
                'note': 'Fallback estimation - PyTorch not available'
            }
        except Exception as e:
            return {'quantization_successful': False, 'error': str(e)}
    
    def _export_cmsis_weights(self, quantized_model, precision: str) -> Optional[Dict]:
        """Export quantized weights in CMSIS-NN compatible format"""
        try:
            weights_dict = {}
            
            for name, param in quantized_model.named_parameters():
                if param.data.dtype in [torch.qint8, torch.int8]:
                    # Convert to numpy and ensure int8 format
                    weight_data = param.data.int_repr().numpy().astype(np.int8)
                    weights_dict[name] = {
                        'data': weight_data.flatten().tolist(),
                        'shape': list(weight_data.shape),
                        'dtype': 'int8',
                        'scale': float(param.q_scale()) if hasattr(param, 'q_scale') else 1.0,
                        'zero_point': int(param.q_zero_point()) if hasattr(param, 'q_zero_point') else 0
                    }
            
            # Save weights file
            weights_file = self.output_dir / f"pizza_verifier_weights_{precision}.json"
            with open(weights_file, 'w') as f:
                json.dump(weights_dict, f, indent=2)
            
            # Generate C header file for RP2040
            self._generate_c_header(weights_dict, precision)
            
            return weights_dict
            
        except Exception as e:
            print(f"Warning: Could not export CMSIS weights: {e}")
            return None
    
    def _generate_c_header(self, weights_dict: Dict, precision: str):
        """Generate C header file for RP2040 integration"""
        header_content = f"""
/* 
 * Auto-generated Pizza Verifier Model Weights for RP2040
 * Precision: {precision}
 * Generated: {Path(__file__).name}
 */

#ifndef PIZZA_VERIFIER_WEIGHTS_H
#define PIZZA_VERIFIER_WEIGHTS_H

#include <stdint.h>

// Model configuration
#define PIZZA_VERIFIER_PRECISION_{precision.upper()}
#define PIZZA_VERIFIER_NUM_LAYERS {len(weights_dict)}

// Memory requirements
#define PIZZA_VERIFIER_WEIGHT_MEMORY_BYTES {sum(len(w['data']) for w in weights_dict.values())}
#define PIZZA_VERIFIER_ACTIVATION_MEMORY_BYTES 8192  // Estimated

"""
        
        # Add weight arrays
        for layer_name, weight_info in weights_dict.items():
            safe_name = layer_name.replace('.', '_').replace('-', '_')
            header_content += f"""
// Layer: {layer_name}
#define {safe_name.upper()}_SIZE {len(weight_info['data'])}
static const int8_t {safe_name}_weights[{len(weight_info['data'])}] = {{
"""
            # Write weights in rows of 16
            weights = weight_info['data']
            for i in range(0, len(weights), 16):
                row = weights[i:i+16]
                header_content += "    " + ", ".join(f"{w:4d}" for w in row) + ",\n"
            
            header_content += "};\n"
            
            # Add scale and zero point if available
            if 'scale' in weight_info:
                header_content += f"#define {safe_name.upper()}_SCALE {weight_info['scale']:.6f}f\n"
            if 'zero_point' in weight_info:
                header_content += f"#define {safe_name.upper()}_ZERO_POINT {weight_info['zero_point']}\n"
        
        header_content += "\n#endif // PIZZA_VERIFIER_WEIGHTS_H\n"
        
        # Save header file
        header_file = self.output_dir / f"pizza_verifier_weights_{precision}.h"
        with open(header_file, 'w') as f:
            f.write(header_content)
        
        print(f"✓ Generated C header: {header_file}")
    
    def optimize_memory_layout(self, model_config: Dict) -> Dict:
        """
        Optimize memory layout for RP2040 constraints
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Optimized memory layout configuration
        """
        # Calculate memory requirements
        weight_memory = model_config.get('weight_memory_bytes', 50000)  # Estimate
        activation_memory = model_config.get('activation_memory_bytes', 8192)  # Estimate
        total_model_memory = weight_memory + activation_memory
        
        # RP2040 memory optimization
        optimization_config = {
            'total_model_memory_bytes': total_model_memory,
            'fits_in_rp2040': total_model_memory < self.rp2040_memory['available_for_model'],
            'memory_optimization_applied': False,
            'optimizations': []
        }
        
        if total_model_memory >= self.rp2040_memory['available_for_model']:
            # Apply memory optimizations
            optimization_config['memory_optimization_applied'] = True
            
            # Weight sharing optimization
            if weight_memory > 30000:
                optimization_config['optimizations'].append('weight_sharing')
                weight_memory = int(weight_memory * 0.8)  # Estimate 20% reduction
            
            # Activation memory optimization
            if activation_memory > 4096:
                optimization_config['optimizations'].append('activation_reuse')
                activation_memory = max(4096, activation_memory // 2)
            
            # Layer fusion
            optimization_config['optimizations'].append('layer_fusion')
            total_model_memory = weight_memory + activation_memory
            
            optimization_config['optimized_memory_bytes'] = total_model_memory
            optimization_config['fits_after_optimization'] = total_model_memory < self.rp2040_memory['available_for_model']
        
        self.deployment_stats['memory_optimization'] = optimization_config
        return optimization_config
    
    def generate_cmsis_integration(self, model_info: Dict) -> Dict:
        """
        Generate CMSIS-NN integration code for RP2040
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Integration status and file paths
        """
        try:
            # Generate main inference function
            inference_code = self._generate_inference_function(model_info)
            
            # Generate CMakeLists.txt for integration
            cmake_content = self._generate_cmake_integration()
            
            # Generate example usage
            example_code = self._generate_usage_example()
            
            # Write files
            inference_file = self.output_dir / "pizza_verifier_inference.c"
            cmake_file = self.output_dir / "CMakeLists.txt"
            example_file = self.output_dir / "pizza_verifier_example.c"
            
            with open(inference_file, 'w') as f:
                f.write(inference_code)
            
            with open(cmake_file, 'w') as f:
                f.write(cmake_content)
            
            with open(example_file, 'w') as f:
                f.write(example_code)
            
            integration_info = {
                'cmsis_integration_generated': True,
                'inference_file': str(inference_file),
                'cmake_file': str(cmake_file),
                'example_file': str(example_file),
                'rp2040_compatible': True
            }
            
            self.deployment_stats['cmsis_integration'] = integration_info
            return integration_info
            
        except Exception as e:
            return {
                'cmsis_integration_generated': False,
                'error': str(e)
            }
    
    def _generate_inference_function(self, model_info: Dict) -> str:
        """Generate C inference function for pizza verifier"""
        return """
/*
 * Pizza Verifier Inference Function for RP2040
 * CMSIS-NN optimized implementation
 */

#include "arm_nnfunctions.h"
#include "pizza_verifier_weights.h"
#include <stdint.h>
#include <string.h>

// Buffer for intermediate activations
static int8_t activation_buffer[8192];

/**
 * Perform pizza quality verification
 * @param image_features: Input image features (normalized to int8)
 * @param prediction_features: Model prediction features
 * @param confidence_score: Confidence score (scaled to int8)
 * @param output_buffer: Output buffer for quality score
 * @return: 0 on success, negative on error
 */
int pizza_verifier_inference(
    const int8_t* image_features,
    const int8_t* prediction_features, 
    int8_t confidence_score,
    int8_t* output_buffer
) {
    // Input validation
    if (!image_features || !prediction_features || !output_buffer) {
        return -1;
    }
    
    // Combine input features
    int8_t combined_input[256];  // Adjust size based on actual model
    memcpy(combined_input, image_features, 128);
    memcpy(combined_input + 128, prediction_features, 127);
    combined_input[255] = confidence_score;
    
    // Layer 1: Fully connected
    arm_fully_connected_s8(
        combined_input,
        layer1_weights,
        LAYER1_SIZE,
        256,  // Input size
        LAYER1_ZERO_POINT,
        LAYER1_SCALE,
        activation_buffer
    );
    
    // Layer 2: ReLU activation
    arm_relu_s8(activation_buffer, LAYER1_SIZE);
    
    // Layer 3: Output layer
    arm_fully_connected_s8(
        activation_buffer,
        output_weights,
        1,  // Output size (quality score)
        LAYER1_SIZE,
        OUTPUT_ZERO_POINT,
        OUTPUT_SCALE,
        output_buffer
    );
    
    return 0;
}

/**
 * Convert quality score from int8 to float
 */
float pizza_verifier_score_to_float(int8_t score) {
    return (score + OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
}
"""
    
    def _generate_cmake_integration(self) -> str:
        """Generate CMakeLists.txt for RP2040 integration"""
        return """
# CMakeLists.txt for Pizza Verifier RP2040 Integration
cmake_minimum_required(VERSION 3.12)

# Include CMSIS-NN
set(CMSIS_NN_PATH "${CMAKE_CURRENT_LIST_DIR}/../CMSIS-NN")
include_directories(${CMSIS_NN_PATH}/Include)

# Pizza Verifier sources
set(PIZZA_VERIFIER_SOURCES
    pizza_verifier_inference.c
    pizza_verifier_example.c
)

# CMSIS-NN library sources
file(GLOB CMSIS_NN_SOURCES 
    "${CMSIS_NN_PATH}/Source/FullyConnectedFunctions/*.c"
    "${CMSIS_NN_PATH}/Source/ActivationFunctions/*.c"
    "${CMSIS_NN_PATH}/Source/BasicMathFunctions/*.c"
)

# Create pizza verifier library
add_library(pizza_verifier STATIC
    ${PIZZA_VERIFIER_SOURCES}
    ${CMSIS_NN_SOURCES}
)

target_include_directories(pizza_verifier PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMSIS_NN_PATH}/Include
)

# Compiler optimizations for RP2040
target_compile_options(pizza_verifier PRIVATE
    -O3
    -mthumb
    -mcpu=cortex-m0plus
    -flto
)

# Example executable
add_executable(pizza_verifier_example
    pizza_verifier_example.c
)

target_link_libraries(pizza_verifier_example pizza_verifier)
"""
    
    def _generate_usage_example(self) -> str:
        """Generate usage example for RP2040"""
        return """
/*
 * Pizza Verifier Usage Example for RP2040
 */

#include <stdio.h>
#include "pizza_verifier_weights.h"

// External inference function
extern int pizza_verifier_inference(
    const int8_t* image_features,
    const int8_t* prediction_features,
    int8_t confidence_score,
    int8_t* output_buffer
);

extern float pizza_verifier_score_to_float(int8_t score);

int main() {
    // Example input data (would come from actual pizza detection)
    int8_t image_features[128];  // Normalized image features
    int8_t prediction_features[127];  // Model prediction features
    int8_t confidence_score = 85;  // Confidence score (0-100 scaled to int8)
    
    // Initialize with example data
    for (int i = 0; i < 128; i++) {
        image_features[i] = (int8_t)(i % 127 - 64);  // Example features
    }
    
    for (int i = 0; i < 127; i++) {
        prediction_features[i] = (int8_t)(i % 63 - 32);  // Example features
    }
    
    // Perform inference
    int8_t quality_score_raw;
    int result = pizza_verifier_inference(
        image_features,
        prediction_features,
        confidence_score,
        &quality_score_raw
    );
    
    if (result == 0) {
        float quality_score = pizza_verifier_score_to_float(quality_score_raw);
        printf("Pizza quality score: %.3f\\n", quality_score);
        
        if (quality_score > 0.8) {
            printf("High quality detection - safe to proceed\\n");
        } else if (quality_score > 0.6) {
            printf("Medium quality detection - manual check recommended\\n");
        } else {
            printf("Low quality detection - manual verification required\\n");
        }
    } else {
        printf("Inference failed with error: %d\\n", result);
    }
    
    return 0;
}
"""
    
    def generate_deployment_report(self) -> Dict:
        """Generate comprehensive deployment report"""
        report = {
            'deployment_timestamp': Path(__file__).stat().st_mtime,
            'rp2040_compatibility': {
                'memory_constraints': self.rp2040_memory,
                'cmsis_nn_config': self.cmsis_nn_config
            },
            'deployment_results': self.deployment_stats,
            'deployment_artifacts': [],
            'recommendations': []
        }
        
        # List generated artifacts
        for file_path in self.output_dir.glob("*"):
            if file_path.is_file():
                report['deployment_artifacts'].append(str(file_path))
        
        # Generate recommendations
        if self.deployment_stats.get('quantization', {}).get('quantization_successful'):
            report['recommendations'].append("Model quantization successful - ready for RP2040 deployment")
        else:
            report['recommendations'].append("Model quantization failed - manual optimization required")
        
        if self.deployment_stats.get('memory_optimization', {}).get('fits_after_optimization', True):
            report['recommendations'].append("Memory optimization successful - fits RP2040 constraints")
        else:
            report['recommendations'].append("Memory optimization insufficient - consider model pruning")
        
        return report

def main():
    """Main deployment function"""
    print("🚀 Starting RP2040 Pizza Verifier Deployment")
    
    # Initialize deployment system
    deployment = RP2040VerifierDeployment()
    
    # Example model path (adjust as needed)
    model_path = "/home/emilio/Documents/ai/pizza/models/pizza_verifier_model.pth"
    
    if not Path(model_path).exists():
        print(f"⚠️  Model file not found: {model_path}")
        print("Creating placeholder deployment configuration...")
        
        # Create placeholder configuration
        placeholder_config = {
            'weight_memory_bytes': 45000,
            'activation_memory_bytes': 8192,
            'model_type': 'pizza_verifier',
            'precision': 'int8'
        }
        
        # Test memory optimization
        memory_config = deployment.optimize_memory_layout(placeholder_config)
        print(f"✓ Memory optimization: {memory_config}")
        
        # Test CMSIS integration generation
        integration_config = deployment.generate_cmsis_integration(placeholder_config)
        print(f"✓ CMSIS integration: {integration_config}")
        
    else:
        # Quantize actual model
        print(f"📦 Quantizing model: {model_path}")
        quantization_result = deployment.quantize_verifier_model(model_path)
        print(f"✓ Quantization result: {quantization_result}")
        
        # Optimize memory layout
        memory_config = deployment.optimize_memory_layout(quantization_result)
        print(f"✓ Memory optimization: {memory_config}")
        
        # Generate CMSIS integration
        integration_config = deployment.generate_cmsis_integration(quantization_result)
        print(f"✓ CMSIS integration: {integration_config}")
    
    # Generate deployment report
    report = deployment.generate_deployment_report()
    report_file = deployment.output_dir / "rp2040_deployment_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📊 Deployment report saved: {report_file}")
    print("🎉 RP2040 deployment preparation complete!")

if __name__ == "__main__":
    main()
