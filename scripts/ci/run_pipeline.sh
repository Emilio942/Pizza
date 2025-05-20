#!/bin/bash
# Master CI/CD Pipeline for Pizza Detection Project
# This script orchestrates the entire pipeline from validation through testing and reporting

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Directories
OUTPUT_DIR="$PROJECT_ROOT/output/pipeline_runs/$TIMESTAMP"
LOG_DIR="$OUTPUT_DIR/logs"
REPORTS_DIR="$OUTPUT_DIR/reports"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$REPORTS_DIR"
mkdir -p "$OUTPUT_DIR/artifacts"

# Main log file
MAIN_LOG="$LOG_DIR/pipeline.log"
FAILED_SCRIPTS_LOG="$LOG_DIR/failed_scripts.log"
VALIDATION_REPORT="$REPORTS_DIR/validation_report.json"
INTEGRATION_REPORT="$REPORTS_DIR/integration_report.json"
SCRIPT_STATUS_JSON="$REPORTS_DIR/script_status.json"

# Initialize JSON for tracking script status
echo '{"pipeline_start": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'", "scripts": []}' > "$SCRIPT_STATUS_JSON"

# Function to log messages
log() {
    local message="$1"
    local level="${2:-INFO}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$MAIN_LOG"
}

# Function to log error messages
error_log() {
    log "$1" "ERROR"
}

# Function to log warning messages
warning_log() {
    log "$1" "WARNING"
}

# Function to log success messages
success_log() {
    log "$1" "SUCCESS"
}

# Function to update script status in JSON file
update_script_status() {
    local script_name="$1"
    local status="$2"
    local duration="$3"
    local retries="${4:-0}"
    local log_file="${5:-N/A}"
    local details="${6:-{}}"
    
    local tmp_file=$(mktemp)
    
    # Create a properly formatted JSON string for details
    if [ "$details" = "{}" ]; then
        details_json="{}"
    else
        details_json="$details"
    fi
    
    jq ".scripts += [{\"name\": \"$script_name\", \"status\": \"$status\", \"duration\": $duration, \"retries\": $retries, \"log\": \"$log_file\", \"details\": $details_json}]" \
       "$SCRIPT_STATUS_JSON" > "$tmp_file"
    mv "$tmp_file" "$SCRIPT_STATUS_JSON"
}

# Function to run a script with proper error handling and logging
run_script() {
    local script_path="$1"
    shift  # Remove the first argument (script_path)
    local critical="${1:-false}"
    shift || true  # Remove the critical flag if present
    local max_retries="${1:-2}"
    shift || true  # Remove the max_retries if present
    
    local script_name=$(basename "$script_path")
    local log_file="$LOG_DIR/${script_name%.py}.log"
    local status="success"
    local start_time=$(date +%s)
    local retries=0
    
    log "Running $script_name..."
    
    while [ $retries -le $max_retries ]; do
        if [ $retries -gt 0 ]; then
            log "Retry $retries for $script_name..."
        fi
        
        if $script_path "$@" > "$log_file" 2>&1; then
            status="success"
            break
        else
            status="failed"
            retries=$((retries+1))
            
            if [ $retries -le $max_retries ]; then
                warning_log "Error running $script_name. Retrying ($retries/$max_retries)..."
                sleep 2
            else
                error_log "Error running $script_name after $max_retries retries. Check $log_file for details."
                echo "$script_path $script_args" >> "$FAILED_SCRIPTS_LOG"
            fi
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Update status in JSON
    update_script_status "$script_name" "$status" "$duration" "$retries" "$log_file"
    
    log "Finished $script_name with status: $status (duration: ${duration}s)"
    
    if [ "$status" = "failed" ] && [ "$critical" = "true" ]; then
        error_log "Critical script $script_name failed. Stopping pipeline."
        exit 1
    elif [ "$status" = "failed" ]; then
        return 1
    else
        return 0
    fi
}

# Phase 1: Script Validation
log "======= PHASE 1: SCRIPT VALIDATION ======="
log "Validating scripts..."

# Run validation script
"$SCRIPT_DIR/validate_scripts.py" --output "$VALIDATION_REPORT" --scripts-dir "$PROJECT_ROOT/scripts" --log-dir "$LOG_DIR"
if [ $? -ne 0 ]; then
    error_log "Script validation failed. Check $LOG_DIR/validate_scripts.log for details."
    exit 1
fi
success_log "Script validation completed successfully."

# Phase 2: Preprocessing
log "======= PHASE 2: PREPROCESSING ======="

# Run CLAHE preprocessing
log "Running CLAHE preprocessing..."
run_script "$PROJECT_ROOT/scripts/test_image_preprocessing.py" "false" "2" "--output-dir=$OUTPUT_DIR/preprocessing" || \
    warning_log "CLAHE preprocessing had issues, but continuing..."

# Run data augmentation
log "Running data augmentation..."
run_script "$PROJECT_ROOT/scripts/augment_dataset.py" "false" "2" "--output-dir=$OUTPUT_DIR/augmented_data" || \
    warning_log "Data augmentation had issues, but continuing..."

# Phase 3: Model Optimization
log "======= PHASE 3: MODEL OPTIMIZATION ======="

# Run the optimization pipeline
log "Running model optimization pipeline..."
"$PROJECT_ROOT/scripts/ci/run_optimization_pipeline.sh" || \
    warning_log "Model optimization had issues, but continuing..."

# Phase 4: Testing
log "======= PHASE 4: TESTING ======="

# Run automated test suite
log "Running automated test suite..."
run_script "$PROJECT_ROOT/scripts/run_pizza_tests.py" "true" "2" "--detailed" || \
    error_log "Automated tests failed. Check logs for details."

# Run verification on the model
log "Verifying model..."
run_script "$PROJECT_ROOT/scripts/models/rp2040_export/verify_model.py" "false" "2" "--model-dir=$PROJECT_ROOT/models" || \
    warning_log "Model verification had issues, but continuing..."

# Run temporal smoothing tests
log "Testing temporal smoothing..."
run_script "$PROJECT_ROOT/scripts/test_temporal_smoothing.py" "false" "2" "--output-dir=$OUTPUT_DIR/temporal_tests" || \
    warning_log "Temporal smoothing tests had issues, but continuing..."

# Phase 5: Integration Validation
log "======= PHASE 5: INTEGRATION VALIDATION ======="

# Run integration validation script
"$SCRIPT_DIR/validate_integration.py" --output "$INTEGRATION_REPORT" --status-file "$SCRIPT_STATUS_JSON" --log-dir "$LOG_DIR"
if [ $? -ne 0 ]; then
    warning_log "Integration validation found issues. Check $INTEGRATION_REPORT for details."
else
    success_log "Integration validation completed successfully."
fi

# Phase 6: Reporting
log "======= PHASE 6: REPORTING ======="

# Generate comprehensive reports
log "Generating final reports..."
"$SCRIPT_DIR/generate_pipeline_report.py" --status-file "$SCRIPT_STATUS_JSON" --validation-report "$VALIDATION_REPORT" \
    --integration-report "$INTEGRATION_REPORT" --output-dir "$REPORTS_DIR"

# Generate performance analysis
log "Analyzing performance..."
run_script "$PROJECT_ROOT/scripts/analyze_performance_logs.py" "false" "2" "--input=$LOG_DIR/pipeline.log" "--output-dir=$REPORTS_DIR" || \
    warning_log "Performance analysis had issues, but continuing..."

# Cleanup temporary files
log "Cleaning up..."
run_script "$PROJECT_ROOT/scripts/cleanup.py" "false" "2" "--temp-only" || \
    warning_log "Cleanup had issues, but continuing..."

# Final status
if [ -s "$FAILED_SCRIPTS_LOG" ]; then
    warning_log "Pipeline completed with some failed scripts. Check $FAILED_SCRIPTS_LOG for details."
else
    success_log "Pipeline completed successfully!"
fi

# Update final status in JSON
jq '.pipeline_end = "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"' "$SCRIPT_STATUS_JSON" > /tmp/tmp_status.json && mv /tmp/tmp_status.json "$SCRIPT_STATUS_JSON"

log "Pipeline results available at: $OUTPUT_DIR"

# Generate symlink to latest run
ln -sf "$OUTPUT_DIR" "$PROJECT_ROOT/output/pipeline_latest"

exit 0
