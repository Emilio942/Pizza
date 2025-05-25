#!/bin/bash
# Structured Pruning Evaluation Script
# This script runs structured pruning at different sparsity rates and evaluates the results

# Set up logs and output directory
LOG_FILE="pruning_clustering.log"
OUTPUT_DIR="output/model_optimization"
mkdir -p $OUTPUT_DIR

# Echo with timestamp
log_message() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

log_message "=== Starting Structured Pruning Evaluation ==="

# Check for base model
BASE_MODEL_PATH=""
for model_path in "models/micro_pizza_model.pth" "models/pizza_model_float32.pth"; do
  if [ -f "$model_path" ]; then
    BASE_MODEL_PATH=$model_path
    break
  fi
done

if [ -z "$BASE_MODEL_PATH" ]; then
  log_message "WARNING: No base model found. Using default model."
else
  log_message "Using base model: $BASE_MODEL_PATH"
fi

# Define sparsity rates to evaluate
SPARSITY_RATES=(0.1 0.2 0.3)

# Function to run pruning
run_pruning() {
  sparsity=$1
  log_message "Running pruning with sparsity $sparsity"
  
  # Build command
  cmd="python scripts/pruning_tool.py --sparsity=$sparsity"
  
  # Add model path if available
  if [ ! -z "$BASE_MODEL_PATH" ]; then
    cmd="$cmd --model_path=$BASE_MODEL_PATH"
  fi
  
  # Add finetune and quantize options
  cmd="$cmd --fine_tune --quantize"
  
  # Execute command
  log_message "Executing: $cmd"
  output=$(eval $cmd 2>&1)
  
  # Log output
  log_message "Pruning output:"
  echo "$output" | tee -a $LOG_FILE
  
  # Check if the pruned model was created
  pruned_model="models/micropizzanetv2_pruned_s${sparsity/./}.pth"
  quantized_model="models/micropizzanetv2_quantized_s${sparsity/./}.pth"
  
  if [ -f "$pruned_model" ]; then
    log_message "Pruned model created: $pruned_model"
  else
    log_message "ERROR: Pruned model not created at $pruned_model"
  fi
  
  if [ -f "$quantized_model" ]; then
    log_message "Quantized model created: $quantized_model"
  else
    log_message "ERROR: Quantized model not created at $quantized_model"
  fi
}

# Function to evaluate a model
evaluate_model() {
  model_path=$1
  log_message "Evaluating model: $model_path"
  
  # Run test script
  cmd="python scripts/run_pizza_tests.py --model $model_path"
  
  log_message "Executing: $cmd"
  output=$(eval $cmd 2>&1)
  
  # Log output
  log_message "Evaluation output:"
  echo "$output" | tee -a $LOG_FILE
  
  # Extract accuracy (simplified, would need parsing in real implementation)
  accuracy=$(echo "$output" | grep -i "accuracy" | head -1 | grep -o -E "[0-9]+\.[0-9]+")
  
  if [ ! -z "$accuracy" ]; then
    log_message "Model accuracy: $accuracy%"
  else
    log_message "Could not determine accuracy"
    accuracy="N/A"
  fi
  
  # Get model size
  if [ -f "$model_path" ]; then
    size_bytes=$(stat -c%s "$model_path")
    size_kb=$(echo "scale=2; $size_bytes/1024" | bc)
    log_message "Model size: $size_kb KB"
  else
    log_message "Model not found: $model_path"
    size_kb="N/A"
  fi
  
  # Return results
  echo "{\"accuracy\": \"$accuracy\", \"size_kb\": \"$size_kb\"}"
}

# Create results JSON structure
results="{\"timestamp\": \"$(date +"%Y-%m-%d %H:%M:%S")\", \"base_model\": {}, \"pruned_models\": {}}"

# Run pruning for each sparsity rate
for sparsity in "${SPARSITY_RATES[@]}"; do
  log_message "Processing sparsity rate: $sparsity"
  
  # Run pruning
  run_pruning $sparsity
  
  # Paths to models
  pruned_model="models/micropizzanetv2_pruned_s${sparsity/./}.pth"
  quantized_model="models/micropizzanetv2_quantized_s${sparsity/./}.pth"
  
  # Evaluate models
  if [ -f "$pruned_model" ]; then
    pruned_eval=$(evaluate_model "$pruned_model")
  else
    pruned_eval="{\"error\": \"Model not found\"}"
  fi
  
  if [ -f "$quantized_model" ]; then
    quantized_eval=$(evaluate_model "$quantized_model")
  else
    quantized_eval="{\"error\": \"Model not found\"}"
  fi
  
  # Add to results
  sparsity_key="${sparsity/./}"
  results=$(echo $results | jq --arg key "sparsity_$sparsity_key" --arg sparsity "$sparsity" --argjson pruned "$pruned_eval" --argjson quantized "$quantized_eval" '.pruned_models[$key] = {"sparsity": $sparsity | tonumber, "pruned": $pruned, "quantized": $quantized}')
done

# Save results
results_file="$OUTPUT_DIR/pruning_evaluation.json"
echo $results | jq '.' > "$results_file"
log_message "Saved results to $results_file"

# Print summary
log_message "===== PRUNING EVALUATION SUMMARY ====="
for sparsity in "${SPARSITY_RATES[@]}"; do
  sparsity_key="${sparsity/./}"
  log_message "Sparsity $sparsity:"
  pruned_acc=$(echo $results | jq -r --arg key "sparsity_$sparsity_key" '.pruned_models[$key].pruned.accuracy')
  pruned_size=$(echo $results | jq -r --arg key "sparsity_$sparsity_key" '.pruned_models[$key].pruned.size_kb')
  quantized_acc=$(echo $results | jq -r --arg key "sparsity_$sparsity_key" '.pruned_models[$key].quantized.accuracy')
  quantized_size=$(echo $results | jq -r --arg key "sparsity_$sparsity_key" '.pruned_models[$key].quantized.size_kb')
  
  log_message "  Pruned model: $pruned_acc% accuracy, $pruned_size KB"
  log_message "  Quantized model: $quantized_acc% accuracy, $quantized_size KB"
done

log_message "=== Structured pruning evaluation completed ==="
