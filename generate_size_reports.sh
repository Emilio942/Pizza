#!/usr/bin/env bash
# Generate visualizations and reports from the input size evaluation results

# Create output directories
mkdir -p output/evaluation
mkdir -p output/visualization

echo "===== Generating Input Size Evaluation Reports and Visualizations ====="

# Generate visualization plots
echo "Generating visualization plots..."
python scripts/visualize_input_size_results.py

# Generate comprehensive report
echo "Generating comprehensive evaluation report..."
python scripts/generate_input_size_report.py

echo "===== Report Generation Complete ====="
echo "Results available at:"
echo "- Report: output/evaluation/input_size_report.md"
echo "- Plots: output/visualization/input_size_evaluation.png"
echo "        output/visualization/input_size_tradeoff.png"
