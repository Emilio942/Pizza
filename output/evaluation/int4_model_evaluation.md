# Int4 Quantization Evaluation Report

*Evaluation Date: 2025-05-18*

## Overview

This report presents the results of applying Int4 quantization directly to the original model,
without prior clustering or pruning. This evaluation provides a baseline for understanding
the effectiveness of Int4 quantization on its own.

## Model Information

**Original Model:**
- Model Path: `models/pizza_model_float32.pth`
- Model Size: 0.00 KB
- Accuracy: 0.00%
- Average Inference Time: 0.00 ms

**Int4 Quantized Model:**
- Model Path: `output/evaluation/int4_quantized/int4_model.pth`
- Model Size: 0.79 KB
- Accuracy: 0.00%
- Average Inference Time: 0.27 ms
- Compression Ratio: 69.03%

## Comparison

- **Size Reduction:** 69.03%
- **Accuracy Difference:** 0.00%
- **Evaluation Execution Time:** 1.85 seconds

## Conclusion

Direct Int4 quantization achieved a 69.03% reduction in model size with an accuracy change of 0.00%. This provides a baseline for comparing with other optimization techniques like clustering and pruning combined with Int4 quantization.

