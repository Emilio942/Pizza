#!/bin/bash
# Clustered Model Evaluation Script
# This script evaluates weight clustering at different cluster sizes (16, 32, 64)
# and with different quantization methods (INT8, INT4)

# Set up logs and output directory
LOG_FILE="pruning_clustering.log"
OUTPUT_DIR="output/model_optimization"
mkdir -p $OUTPUT_DIR

# Echo with timestamp
log_message() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

log_message "=== Starting Weight Clustering Evaluation ==="

# Check for base model
BASE_MODEL_PATH=""
for model_path in "models/micro_pizza_model.pth" "models/pizza_model_float32.pth"; do
  if [ -f "$model_path" ]; then
    BASE_MODEL_PATH=$model_path
    break
  fi
done

if [ -z "$BASE_MODEL_PATH" ]; then
  log_message "ERROR: No base model found. Please provide a model path with --model_path."
  exit 1
else
  log_message "Using base model: $BASE_MODEL_PATH"
fi

# Run the evaluation script
log_message "Starting clustering evaluation..."
python scripts/evaluate_clustering.py --model_path="$BASE_MODEL_PATH" --output_dir="$OUTPUT_DIR"

# Check if successful
if [ $? -eq 0 ]; then
  log_message "Clustering evaluation completed successfully!"
  log_message "See report at: $OUTPUT_DIR/clustering_evaluation.md"
else
  log_message "ERROR: Clustering evaluation failed!"
  exit 1
fi

# Optional: Generate visualizations
if command -v python &> /dev/null; then
  log_message "Generating visualizations..."
  
  # Simple visualization script (inline)
  python -c "
import json
import matplotlib.pyplot as plt
import os

# Load results
results_path = '$OUTPUT_DIR/clustering_evaluation.json'
with open(results_path, 'r') as f:
    results = json.load(f)

# Create comparison plots
plt.figure(figsize=(16, 12))

# Accuracy comparison
plt.subplot(2, 2, 1)
labels = ['Baseline']
values = [results['baseline']['accuracy']]

for config in ['cluster_16_int4', 'cluster_32_int4', 'cluster_64_int4']:
    if config in results:
        labels.append(config.replace('_', ' ').title())
        values.append(results[config]['accuracy'])

plt.bar(labels, values)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.xticks(rotation=45)

# Model size comparison
plt.subplot(2, 2, 2)
labels = ['Baseline']
values = [results['baseline']['model_size_kb']]

for config in ['cluster_16_int4', 'cluster_32_int4', 'cluster_64_int4']:
    if config in results:
        labels.append(config.replace('_', ' ').title())
        values.append(results[config]['model_size_kb'])

plt.bar(labels, values)
plt.ylabel('Model Size (KB)')
plt.title('Model Size Comparison')
plt.xticks(rotation=45)

# RAM usage comparison
plt.subplot(2, 2, 3)
labels = ['Baseline']
values = [results['baseline']['total_ram_kb']]

for config in ['cluster_16_int4', 'cluster_32_int4', 'cluster_64_int4']:
    if config in results:
        labels.append(config.replace('_', ' ').title())
        values.append(results[config]['total_ram_kb'])

plt.bar(labels, values)
plt.ylabel('Total RAM (KB)')
plt.title('RAM Usage Comparison')
plt.xticks(rotation=45)

# Inference time comparison
plt.subplot(2, 2, 4)
labels = ['Baseline']
values = [results['baseline']['avg_inference_time_ms']]

for config in ['cluster_16_int4', 'cluster_32_int4', 'cluster_64_int4']:
    if config in results:
        labels.append(config.replace('_', ' ').title())
        values.append(results[config]['avg_inference_time_ms'])

plt.bar(labels, values)
plt.ylabel('Inference Time (ms)')
plt.title('Inference Time Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/clustering_comparison.png')
print('Visualization saved to $OUTPUT_DIR/clustering_comparison.png')
"
fi

log_message "=== Weight Clustering Evaluation Complete ==="
