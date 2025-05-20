# Final Model Configuration for RP2040

## Overview

The final model configuration has been determined to optimize memory usage while maintaining high accuracy for the Pizza Detection system on the RP2040 microcontroller.

**Model name:** micropizzanetv2_quantized_s30
**Configuration file:** `/models/final_rp2040_config.json`
**Verification date:** 2025-05-19

## Configuration Details

### 1. Model Architecture

- **Base architecture:** MicroPizzaNetV2
- **Input size:** 48x48 pixels (RGB)
- **Parameters:**
  - Original: 150,000
  - After pruning: 105,000

### 2. Optimizations Applied

1. **Structured Pruning (30%):**
   - Removed 30% of the least important filters/channels
   - Reduced parameter count from 150,000 to 105,000
   - Minimal impact on accuracy (-3.8%)

2. **Quantization:**
   - Int8 quantization with Quantization-Aware Training (QAT)
   - Reduced per-weight memory from 32 bits to 8 bits

3. **CMSIS-NN Integration:**
   - ARM Cortex-M optimized operations
   - Accelerated convolution, depthwise convolution, and pooling
   - Implemented in `pizza_model_cmsis.c`

4. **Input Size Optimization:**
   - Selected 48x48 input size
   - Best balance of accuracy vs. memory usage

## Performance Verification

### Memory Usage

- **Total RAM usage:** 170.6 KB (64.62% of available RAM)
- **Remaining RAM:** 33.4 KB
- **Breakdown:**
  - Framebuffer: 76.8 KB (45.0%)
  - Tensor Arena: 10.8 KB (6.3%)
  - Preprocessing buffer: 27.0 KB (15.8%)
  - System overhead: 40.0 KB (23.4%)
  - Stack: 8.0 KB (4.7%)
  - Heap: 5.0 KB (2.9%)
  - Static buffers: 3.0 KB (1.8%)

### Accuracy

- **Accuracy:** 90.20%
- **Precision:** 91.86%
- **Recall:** 91.88%
- **F1-Score:** 0.9185

### Inference Performance

- **Inference time:** 22 ms
- **Speed improvement:** 10% compared to non-pruned model

## Conclusion

The optimized model meets all requirements defined in the project specifications:
- RAM usage is well under the 204 KB limit
- Accuracy exceeds the minimum requirements
- Inference time is acceptable for real-time applications

This configuration represents the optimal balance between performance and memory efficiency for the Pizza Detection system on the RP2040 microcontroller.
