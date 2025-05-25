#!/bin/bash
# Fix model architecture mismatch and train new model

# Ensure the script exits on any error
set -e

# Show commands being executed
set -x

# Step 1: Check if models directory exists and create if not
if [ ! -d "models_optimized" ]; then
    mkdir -p models_optimized
    echo "Created models_optimized directory"
fi

# Step 2: Test for existing improved model and convert it if found
if [ -f "models_optimized/improved_early_exit.pth" ]; then
    echo "Found existing improved model, testing..."
    python test_model_adapter.py --model-path models_optimized/improved_early_exit.pth --data-dir data/augmented
    
    # Convert to original format for compatibility
    echo "Converting improved model to original format for better compatibility..."
    python convert_early_exit_model.py \
        --input-path models_optimized/improved_early_exit.pth \
        --output-path models_optimized/improved_early_exit_original_format.pth \
        --target-arch original
fi

# Step 3: Test for existing original model and convert it if found
if [ -f "models_optimized/micropizzanet_early_exit.pth" ]; then
    echo "Found existing original model, testing..."
    python test_model_adapter.py --model-path models_optimized/micropizzanet_early_exit.pth --data-dir data/augmented
    
    # Convert to improved format
    echo "Converting original model to improved format..."
    python convert_early_exit_model.py \
        --input-path models_optimized/micropizzanet_early_exit.pth \
        --output-path models_optimized/micropizzanet_early_exit_improved_format.pth \
        --target-arch improved
fi

# Step 4: Run the complete workflow to train a new improved model
echo "Running the complete workflow to train a new improved model..."
python run_improved_early_exit.py \
    --data-dir data/augmented \
    --epochs 50 \
    --patience 10 \
    --model-name improved_early_exit_new \
    --compare-baseline

echo "================================================================="
echo "Process completed! Check the logs and results directories for outputs."
echo "If the new model training was successful, you'll find it at:"
echo "  models_optimized/improved_early_exit_new.pth"
echo "You can use this model for actual RP2040 deployment."
echo "================================================================="
