# Input Size Evaluation Report
*Generated on: 2025-05-19 01:05:28*

## Overview

This report evaluates the impact of different input image sizes on:
- Model accuracy
- RAM usage (framebuffer size + tensor arena size)

The target is to find an optimal input size that balances accuracy and RAM usage, 
with a constraint of staying under 204KB total RAM.

## Results Summary

| Size | Accuracy | Framebuffer (KB) | Tensor Arena (KB) | Total RAM (KB) | % of RP2040 RAM |
|------|----------|-----------------|------------------|---------------|-----------------|
| 32x32 | 35.00% | 3.0 | 1.85 | 4.85 | 1.84% |
| 40x40 | 10.00% | 4.69 | 1.85 | 6.54 | 2.48% |
| 48x48 | 47.50% | 6.75 | 1.85 | 8.6 | 3.26% |

## Detailed Results by Size

### Size: 32x32

- **Accuracy**: 35.00%
- **RAM Usage**:
  - Framebuffer: 3.0 KB
  - Tensor Arena: 1.85 KB
  - **Total**: 4.85 KB (1.84% of RP2040's 264KB RAM)

### Size: 40x40

- **Accuracy**: 10.00%
- **RAM Usage**:
  - Framebuffer: 4.69 KB
  - Tensor Arena: 1.85 KB
  - **Total**: 6.54 KB (2.48% of RP2040's 264KB RAM)

### Size: 48x48

- **Accuracy**: 47.50%
- **RAM Usage**:
  - Framebuffer: 6.75 KB
  - Tensor Arena: 1.85 KB
  - **Total**: 8.6 KB (3.26% of RP2040's 264KB RAM)

## Analysis and Recommendation

### Recommendation

Based on the evaluation, the optimal input size is **48x48**:

- It achieves the best accuracy (47.50%) among sizes that fit within the RAM constraint.
- It uses 8.6 KB of RAM, which is below the 204KB limit.

### Trade-offs

## Conclusion

The evaluation demonstrates the trade-off between input image size, model accuracy, and RAM usage. 
Smaller input sizes significantly reduce RAM requirements but at the cost of some accuracy. 
For the RP2040 microcontroller with its 264KB RAM constraint, choosing the right input size is crucial 
to balance performance and memory usage.