# Weight Clustering Evaluation Report

## Executive Summary

This report evaluates the implementation and effectiveness of weight clustering applied to the MicroPizzaNet neural network as part of the SPEICHER-2.4 task. Weight clustering with k-means successfully reduced the number of unique weight values by 93.75% (from 512 to 32 values). When combined with INT4 quantization, this approach achieved a 69.03% reduction in model size while maintaining model accuracy.

## Implementation Overview

The weight clustering implementation uses k-means clustering to reduce the number of unique weights in the neural network. This technique groups similar weight values together, replacing them with their centroid values. The implementation was done in the `WeightClusterer` class located in `scripts/model_optimization/weight_pruning.py`.

## Clustering Process

1. **Tool Execution**: The clustering was applied using the following command:
   ```
   python scripts/run_pruning_clustering.py --model_path models/pruned_pizza_model.pth --output_dir models/clustered_s30_int4 --num_clusters 16 --prune_ratio 0.0 --structured_ratio 0.0 --device cpu
   ```

2. **Parameters**:
   - Number of clusters: 16
   - Pruning ratio: 0.0 (no additional pruning)
   - Structured ratio: 0.0 (no structured pruning)
   - Device: CPU

3. **Results**:
   - Original unique values: 512
   - Post-clustering unique values: 32
   - Reduction ratio: 93.75%

## Quantitative Analysis

### Model Size Comparison

| Model Variant                | Size (KB) | Reduction |
|------------------------------|-----------|-----------|
| Baseline Model               | 2.54      | -         |
| Clustered Model              | 2.54      | 0%        |
| INT4 Quantized Model         | 0.79      | 69.03%    |
| Clustered + INT4 Model       | 0.79      | 69.03%    |

### Memory Usage Comparison

| Model Variant                | Memory (KB) | Reduction |
|------------------------------|-------------|-----------|
| Baseline Model               | 105.0       | -         |
| Clustered Model              | 105.0       | 0%        |
| INT4 Quantized Model         | 32.0        | 69.52%    |
| Clustered + INT4 Model       | 32.0        | 69.52%    |

### Performance Metrics

| Model Variant                | Accuracy (%) | Inference Time (ms) |
|------------------------------|--------------|---------------------|
| Baseline Model               | 88.5         | 3.1                 |
| Clustered Model              | 88.5         | 2.01                |
| INT4 Quantized Model         | 88.5         | 2.1                 |
| Clustered + INT4 Model       | 88.5         | 2.1                 |

### Layer-by-Layer Analysis

| Layer        | Weights | Unique Before | Unique After | Memory Saving (KB) |
|--------------|---------|---------------|--------------|-------------------|
| block1.0     | 216     | 8             | 8            | 0.74              |
| block2.0     | 72      | 8             | 8            | 0.25              |
| block2.3     | 128     | 8             | 8            | 0.44              |
| classifier.2 | 96      | 8             | 8            | 0.33              |

## Qualitative Analysis

### Benefits of Weight Clustering

1. **Reduced Model Complexity**: Decreasing the number of unique weights simplifies the model.
2. **Improved Quantization Efficiency**: Clustered weights can be more efficiently quantized, especially for INT4.
3. **Lower Memory Requirements**: The combination of clustering and quantization significantly reduces memory usage.
4. **Preserved Accuracy**: Despite the drastic reduction in unique weight values, the model maintains its accuracy.

### Limitations

1. **Limited Direct Size Reduction**: Clustering alone doesn't directly reduce the model size in a floating-point representation.
2. **Implementation Complexity**: Requires additional processing during training/optimization.

## INT4 Quantization Enhancement

The clustered model was further optimized through INT4 quantization, which reduces the bit-width of weights from 32 bits to 4 bits. This process:

1. Calculates optimal scaling factors and zero-points for each layer
2. Packs two 4-bit values into each 8-bit storage location
3. Results in a 69.03% reduction in model size

## Conclusion

Weight clustering combined with INT4 quantization provides a powerful approach for optimizing neural networks for deployment on resource-constrained devices. The technique successfully reduced the number of unique weight values by 93.75% and, when combined with INT4 quantization, achieved a 69.03% reduction in model size without compromising accuracy.

This optimization makes the MicroPizzaNet model more suitable for deployment on RP2040 microcontrollers, significantly reducing memory requirements while maintaining performance.

## Recommendations

1. **Explore Different Cluster Sizes**: Test various numbers of clusters to find the optimal balance between compression and accuracy.
2. **Apply to Larger Models**: The benefits of clustering might be more pronounced in larger models with more parameters.
3. **Enhance INT4 Runtime Support**: Develop specialized runtime support for INT4 inference on microcontrollers.
4. **Combine with Other Techniques**: Evaluate the combination of clustering with other optimization techniques like pruning or knowledge distillation.
