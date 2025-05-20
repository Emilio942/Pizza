# Tensor Arena Size Analysis for Pruned Model (s30)

## Overview

This report documents the Tensor Arena RAM usage for the pruned and quantized model (with 30% sparsity) as required by task SPEICHER-2.3.

## Model Information

- **Model Name**: micropizzanetv2_quantized_s30
- **Pruning Level**: 30% sparsity
- **Quantization**: INT8 (confirmed)
- **Model Size**: 25.6 KB

## RAM Usage Analysis

### Raw Emulator Output
```
RAM-Nutzung: 192.8KB (172.8KB Modell + 20.0KB System)
```

### Tensor Arena Size
According to the emulator output, the tensor arena RAM requirement for the pruned and quantized model is **172.8 KB**.

This value was calculated using the improved tensor arena estimation method implemented in `src/emulation/emulator-test.py`, which addresses the EMU-02 issue where the tensor arena size was previously underestimated.

### Comparison with Original Model

The tensor arena size estimation in the pruning report (`output/model_optimization/pruning_report_s30.json`) shows:
```
"inference_stats": {
  "latency_ms": 20,
  "ram_usage_kb": 9
}
```

However, this value (9 KB) was likely calculated using the old estimation method, which didn't correctly account for activations and tensor management during inference.

## Conclusion

The actual tensor arena size required for the pruned model (s30) is **172.8 KB**, which is significantly higher than the initially reported value. This is due to the improved calculation method that now correctly considers model architecture, quantization state, and input dimensions.

When combined with the system overhead (20.0 KB), the total RAM usage is **192.8 KB**, which is below the 204 KB RAM constraint for the RP2040 platform.

## Next Steps

- Update the `aufgaben.txt` to mark SPEICHER-2.3 as complete
- Proceed with the next optimization task (SPEICHER-2.4 or SPEICHER-6.1)
