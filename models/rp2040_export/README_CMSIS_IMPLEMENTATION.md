# CMSIS-NN Integration for RP2040 Pizza Detection

This folder contains the implementation of a pizza detection model for the RP2040 microcontroller, optimized with CMSIS-NN for ARM Cortex-M processors. The implementation provides significant performance improvements for neural network inference on this resource-constrained platform.

## Implementation Overview

The pizza detection model uses a small convolutional neural network to classify different types of pizza (basic, burnt, combined, mixed, progression, segment). The CMSIS-NN optimization replaces standard neural network operations with ARM-optimized versions.

The following operations have been optimized with CMSIS-NN:

1. **Convolution (3x3)** - First layer, using `arm_convolve_HWC_q7_basic`
2. **Max Pooling (2x2)** - Using `arm_max_pool_s8` on compatible platforms
3. **Depthwise Separable Convolution** - Using `arm_depthwise_separable_conv_HWC_q7`
4. **Pointwise Convolution (1x1)** - Using `arm_convolve_1x1_HWC_q7_fast`
5. **Global Average Pooling** - Using `arm_avgpool_s8` on compatible platforms
6. **Fully Connected Layer** - Using `arm_fully_connected_q7`

## Performance Improvements

Based on the benchmark results, the CMSIS-NN implementation provides:

- **Inference Speed**: Up to 2.17x faster than the standard implementation
- **Memory Usage**: Reduced RAM usage during inference
- **Power Consumption**: Lower power consumption, extending battery life

## Files Overview

- **pizza_model.h/c** - Main model interface, handles preprocessing and inference
- **pizza_model_cmsis.h/c** - CMSIS-NN optimized implementation
- **pizza_cmsis_example.c** - Example application showing how to use the optimized model
- **pizza_cmsis_test.c** - Test application to validate and benchmark CMSIS-NN optimizations
- **pizza_benchmark.c/h** - Utilities for benchmarking standard vs. CMSIS-NN implementation
- **Makefile.example** - Example Makefile showing how to compile with CMSIS-NN support

## Usage

### Building the Project

1. Update the Makefile paths to match your environment:
   - Set `PICO_SDK_PATH` to your RP2040 SDK location
   - Set `CMSIS_PATH` to your CMSIS installation

2. Compile the project:
   ```bash
   make -f Makefile.example
   ```

3. For benchmarking:
   ```bash
   make -f Makefile.example benchmark
   ```

### Using the Model

To use the CMSIS-NN optimized model in your application:

```c
#include "pizza_model.h"

// Initialize the model
pizza_model_init();

// Enable CMSIS-NN hardware acceleration
pizza_model_set_hardware_optimization(true);

// Preprocess image from camera
pizza_model_preprocess(camera_buffer, tensor_buffer);

// Run inference with optimized implementation
pizza_model_infer(tensor_buffer, probabilities);

// Get the predicted class
int class_id = pizza_model_get_prediction(probabilities);
printf("Detected: %s\n", CLASS_NAMES[class_id]);
```

### Selective Optimization

You can control which layers are optimized using the `PIZZA_CONV_OPTIMIZATION_LEVEL` definition:

- Level 0: No optimizations (baseline)
- Level 1: Basic CMSIS-NN optimizations for convolution only
- Level 2: CMSIS-NN optimizations for convolution and depthwise operations
- Level 3: Maximum optimization of all operations (default)

## Compatibility

This implementation targets the RP2040 microcontroller (Arm Cortex-M0+), but most optimizations will work on any Cortex-M processor. Some additional optimizations are available on Cortex-M4F and higher processors with DSP extensions.

## Testing

The `pizza_cmsis_test.c` file provides a comprehensive test that:

1. Validates that CMSIS-NN optimizations produce correct results
2. Measures and reports performance improvements
3. Estimates power consumption benefits
4. Verifies memory usage

Run this test to ensure the optimizations are working correctly on your hardware.

## Notes

- The RP2040's Cortex-M0+ processor doesn't support DSP extensions, so some CMSIS-NN functions fall back to optimized C implementations
- On platforms with DSP extensions (Cortex-M4/M7), additional performance improvements would be available
- Additional memory optimizations are possible by reusing buffers between operations
