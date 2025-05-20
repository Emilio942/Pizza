#!/bin/bash
# Script to run the optimization pipeline for the Pizza Model
# This script handles executing all optimization scripts in the right order
# and with proper error handling.

set -e
set -o pipefail

# Log file
LOG_DIR="output/optimization/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/optimization_pipeline.log"

# Output directories
OUTPUT_DIR="output/optimization"
mkdir -p "$OUTPUT_DIR/model_comparison"
mkdir -p "$OUTPUT_DIR/pruning"
mkdir -p "$OUTPUT_DIR/distillation"
mkdir -p "$OUTPUT_DIR/quantization"

# Track script status
STATUS_FILE="$OUTPUT_DIR/script_status.json"
echo "{\"scripts\": []}" > "$STATUS_FILE"

# Maximum retry attempts
MAX_RETRIES=2

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Function to run a script with error handling and retries
run_script() {
    local script_path="$1"
    local output_dir="$2"
    shift 2  # Remove the first two arguments
    
    local script_name=$(basename "$script_path")
    local log_file="$LOG_DIR/${script_name}.log"
    local status="success"
    local start_time=$(date +%s)
    local retries=0
    
    log "Running $script_name..."
    
    # Try to run the script with retries
    while [ $retries -le $MAX_RETRIES ]; do
        if [ $retries -gt 0 ]; then
            log "Retry $retries for $script_name..."
        fi
        
        # Only pass the arguments, let each script handle its own output directory
        if python "$script_path" "$@" > "$log_file" 2>&1; then
            status="success"
            break
        else
            status="failed"
            retries=$((retries+1))
            
            if [ $retries -le $MAX_RETRIES ]; then
                log "Error running $script_name. Retrying ($retries/$MAX_RETRIES)..."
                sleep 2
            else
                log "Error running $script_name after $MAX_RETRIES retries. Check $log_file for details."
            fi
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Add to status file
    local tmp_file=$(mktemp)
    jq ".scripts += [{\"name\": \"$script_name\", \"status\": \"$status\", \"duration\": $duration, \"retries\": $retries, \"log\": \"$log_file\"}]" "$STATUS_FILE" > "$tmp_file"
    mv "$tmp_file" "$STATUS_FILE"
    
    log "Finished $script_name with status: $status (duration: ${duration}s)"
    
    if [ "$status" = "failed" ]; then
        return 1
    else
        return 0
    fi
}

# Main execution starts here
log "Starting optimization pipeline..."

# 1. CNN Model Comparison
log "=== Stage 1: CNN Model Comparison ==="

run_script "scripts/compare_tiny_cnns.py" "$OUTPUT_DIR/model_comparison" "--visualize" "--early-stopping" || \
    log "Warning: compare_tiny_cnns.py failed, but continuing..."

run_script "scripts/compare_se_models.py" "$OUTPUT_DIR/model_comparison" "--visualize" "--early-stopping" || \
    log "Warning: compare_se_models.py failed, but continuing..."

run_script "scripts/compare_inverted_residual.py" "$OUTPUT_DIR/model_comparison" "--output-dir=$OUTPUT_DIR/model_comparison" || \
    log "Warning: compare_inverted_residual.py failed, but continuing..."

run_script "scripts/compare_hard_swish.py" "$OUTPUT_DIR/model_comparison" "--output-dir=$OUTPUT_DIR/model_comparison" || \
    log "Warning: compare_hard_swish.py failed, but continuing..."

# 2. Model Pruning and Clustering
log "=== Stage 2: Model Pruning and Clustering ==="

# The model pruning script requires a model to prune
BEST_MODEL="models/pizza_model_float32.pth"
if [ -f "$OUTPUT_DIR/model_comparison/models/micropizzanet_original.pth" ]; then
    BEST_MODEL="$OUTPUT_DIR/model_comparison/models/micropizzanet_original.pth"
    log "Using model from previous stage: $BEST_MODEL"
fi

run_script "scripts/run_pruning_clustering.py" "$OUTPUT_DIR/pruning" "--model_path=$BEST_MODEL" "--output_dir=$OUTPUT_DIR/pruning" "--prune_ratio=0.3" "--structured_ratio=0.2" "--num_clusters=32" "--fine_tune_epochs=5" || \
    log "Warning: run_pruning_clustering.py failed, but continuing..."

# 3. Knowledge Distillation
log "=== Stage 3: Knowledge Distillation ==="

run_script "scripts/knowledge_distillation.py" "$OUTPUT_DIR/distillation" "--teacher-model=$BEST_MODEL" "--output_dir=$OUTPUT_DIR/distillation" "--epochs=10" || \
    log "Warning: knowledge_distillation.py failed, but continuing..."

# 4. Quantization
log "=== Stage 4: Quantization ==="

run_script "scripts/quantization_aware_training.py" "$OUTPUT_DIR/quantization" "--model-path=$BEST_MODEL" "--output_dir=$OUTPUT_DIR/quantization" "--epochs=5" || \
    log "Warning: quantization_aware_training.py failed, but continuing..."

# Generate optimization summary
log "Generating optimization summary..."
python scripts/ci/generate_optimization_summary.py --status-file "$STATUS_FILE" --output-dir "$OUTPUT_DIR"

log "Optimization pipeline completed."
