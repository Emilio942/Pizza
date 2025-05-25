#!/usr/bin/env bash
# This script runs the input size evaluation for multiple sizes

# Create output directory
mkdir -p output/evaluation

# Run the evaluation script for different image sizes
echo "Starting evaluation of different input image sizes..."
python scripts/evaluate_image_sizes.py --sizes 32 40 48 

echo "Evaluation complete. Results can be found in output/evaluation/"
