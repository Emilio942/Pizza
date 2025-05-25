# Model Architecture Mismatch Fix

This document explains how to resolve the model architecture mismatch between the original `MicroPizzaNetWithEarlyExit` and improved `ImprovedMicroPizzaNetWithEarlyExit` models.

## Problem Description

The original issue was an architecture mismatch error when trying to load weights trained with the `ImprovedMicroPizzaNetWithEarlyExit` model into a `MicroPizzaNetWithEarlyExit` model structure. The key differences between the models are:

1. The improved model has an enhanced early exit classifier with an extra layer (16→32→num_classes)
2. The original model has a simpler early exit classifier (16→num_classes) 
3. The improved model has a `forced_exit` parameter in its forward method
4. Dropout rates differ (0.3 in improved vs 0.2 in original)
5. Confidence thresholds differ (0.5 in improved vs 0.8 in original)

## Solution

We've implemented a model adapter that can:
1. Load models with either architecture
2. Convert models between architectures
3. Provide a unified evaluation script that works with both

This solution ensures compatibility while maintaining the benefits of the improved model architecture.

## How to Use

### Testing Model Loading

To test loading a model with the adapter:

```bash
python test_model_adapter.py --model-path models_optimized/improved_early_exit.pth --data-dir data/augmented
```

### Converting Between Model Architectures

To convert a model from one architecture to another:

```bash
# Convert improved model to original architecture
python convert_early_exit_model.py --input-path models_optimized/improved_early_exit.pth --output-path models_optimized/improved_early_exit_original_format.pth --target-arch original

# Convert original model to improved architecture
python convert_early_exit_model.py --input-path models_optimized/micropizzanet_early_exit.pth --output-path models_optimized/micropizzanet_early_exit_improved_format.pth --target-arch improved
```

### Using the Unified Evaluation Script

The unified evaluation script works with both model architectures:

```bash
python scripts/early_exit/unified_evaluate_forced_early_exit.py --model-path models_optimized/improved_early_exit.pth --data-dir data/augmented
```

### Running the Full Workflow

To run the complete improved early exit workflow:

```bash
python run_improved_early_exit.py --data-dir data/augmented --epochs 50 --patience 10
```

## Directory Structure

```
/scripts/early_exit/
  ├── improved_early_exit.py           # Improved model architecture
  ├── micropizzanet_early_exit.py      # Original model architecture
  ├── model_adapter.py                 # Adapter functions for compatibility
  ├── unified_evaluate_forced_early_exit.py  # Evaluation script for both models
  └── improved_evaluate_forced_early_exit.py # Original improved evaluation script

/
  ├── convert_early_exit_model.py      # Tool to convert between architectures  
  ├── test_model_adapter.py            # Test script for the adapter
  └── run_improved_early_exit.py       # Complete workflow script
```

## Technical Approach

The adapter works by:
1. Attempting to load the model directly with the original architecture
2. If that fails, trying the improved architecture
3. If needed, adapting the model by copying compatible weights between architectures

For evaluation, the unified script handles both architectures by detecting the presence of the `forced_exit` parameter in the forward method and adjusting its behavior accordingly.

## Next Steps

1. Train the improved model by running the complete workflow 
2. Evaluate actual energy savings on the RP2040 microcontroller
3. For best results, we recommend training a new model from scratch using the improved architecture
