#!/usr/bin/env bash
# Script to evaluate different input image sizes for the pizza detection model

# Directory for output
mkdir -p output/evaluation

echo "===== Pizza Detection: Input Size Evaluation ====="
echo "Testing input sizes: 32x32, 40x40, 48x48"
echo "(This process may take some time as it trains multiple models)"

# Run the evaluation script
python scripts/evaluate_input_sizes.py --sizes 32 40 48

echo "===== Evaluation Complete ====="
echo "Results are available in output/evaluation/"
echo "Summary file: output/evaluation/input_size_summary.json"
