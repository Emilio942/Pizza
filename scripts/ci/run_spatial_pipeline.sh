#!/bin/bash
# Extended CI/CD Pipeline for Spatial-MLLM Integration
# SPATIAL-4.2 Implementation

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Directories
OUTPUT_DIR="$PROJECT_ROOT/output/spatial_pipeline_runs/$TIMESTAMP"
LOG_DIR="$OUTPUT_DIR/logs"
REPORTS_DIR="$OUTPUT_DIR/reports"
ARTIFACTS_DIR="$OUTPUT_DIR/artifacts"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$REPORTS_DIR"
mkdir -p "$ARTIFACTS_DIR"

# Configuration files
MAIN_LOG="$LOG_DIR/spatial_pipeline.log"
FAILED_TESTS_LOG="$LOG_DIR/failed_spatial_tests.log"
SPATIAL_TEST_REPORT="$REPORTS_DIR/spatial_test_report.json"
DEPLOYMENT_REPORT="$REPORTS_DIR/deployment_report.json"
MODEL_VALIDATION_REPORT="$REPORTS_DIR/model_validation_report.json"

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export SPATIAL_MODEL_PATH="$PROJECT_ROOT/models/spatial_mllm"
export SPATIAL_TEST_DATA="$PROJECT_ROOT/data/test"

# Function to log messages
log() {
    local message="$1"
    local level="${2:-INFO}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] [SPATIAL] $message" | tee -a "$MAIN_LOG"
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

# Function to check CUDA availability
check_cuda() {
    log "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi >> "$LOG_DIR/cuda_info.log" 2>&1
        log "CUDA detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    else
        warning_log "CUDA not available - some tests may be skipped"
        return 1
    fi
    
    python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')" >> "$LOG_DIR/cuda_info.log" 2>&1
    
    return 0
}

# Function to validate Spatial-MLLM environment
validate_spatial_environment() {
    log "Validating Spatial-MLLM environment..."
    
    # Check Python dependencies
    python3 -c "
import sys
required_packages = [
    'torch', 'transformers', 'accelerate', 
    'qwen_vl_utils', 'decord', 'flash_attn'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages available')

# Test enhanced spatial functionality
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'GPU name: {torch.cuda.get_device_properties(0).name}')
    print('✅ Enhanced spatial environment validation passed')
except Exception as e:
    print(f'❌ Enhanced spatial validation failed: {e}')
    sys.exit(1)
    sys.exit(0)
" >> "$LOG_DIR/dependency_check.log" 2>&1

    if [ $? -ne 0 ]; then
        error_log "Spatial-MLLM dependencies not satisfied"
        return 1
    fi
    
    success_log "Spatial-MLLM environment validated"
    return 0
}

# Function to run Spatial-MLLM model validation
validate_spatial_models() {
    log "Validating Spatial-MLLM models..."
    
    python3 "$PROJECT_ROOT/scripts/spatial_model_validation.py" \
        --output-dir "$ARTIFACTS_DIR/model_validation" \
        --report-file "$MODEL_VALIDATION_REPORT" \
        >> "$LOG_DIR/model_validation.log" 2>&1
    
    if [ $? -eq 0 ]; then
        success_log "Spatial model validation completed"
        return 0
    else
        error_log "Spatial model validation failed"
        return 1
    fi
}

# Function to run comprehensive spatial feature tests
run_spatial_feature_tests() {
    log "Running comprehensive spatial feature tests..."
    
    python3 "$PROJECT_ROOT/scripts/spatial_feature_tests.py" \
        --output-dir "$ARTIFACTS_DIR/spatial_tests" \
        --api-url "http://localhost:8001" \
        --verbose \
        >> "$LOG_DIR/spatial_tests.log" 2>&1
    
    local exit_code=$?
    
    # Copy test results to reports
    if [ -f "$ARTIFACTS_DIR/spatial_tests/spatial_tests_"*".json" ]; then
        cp "$ARTIFACTS_DIR/spatial_tests/spatial_tests_"*".json" "$SPATIAL_TEST_REPORT"
    fi
    
    if [ $exit_code -eq 0 ]; then
        success_log "Spatial feature tests completed successfully"
        return 0
    else
        error_log "Spatial feature tests failed"
        echo "spatial_feature_tests.py" >> "$FAILED_TESTS_LOG"
        return 1
    fi
}

# Function to test Docker container builds
test_docker_builds() {
    log "Testing Docker container builds..."
    
    # Build spatial Docker image
    log "Building Spatial-MLLM Docker image..."
    docker build -f "$PROJECT_ROOT/Dockerfile.spatial" -t pizza-detection-spatial:test "$PROJECT_ROOT" \
        >> "$LOG_DIR/docker_build_spatial.log" 2>&1
    
    if [ $? -ne 0 ]; then
        error_log "Spatial Docker build failed"
        return 1
    fi
    
    # Test container startup
    log "Testing spatial container startup..."
    docker run --rm --gpus all -d --name pizza-spatial-test \
        -p 8099:8001 \
        -e MODEL_TYPE=spatial \
        pizza-detection-spatial:test \
        >> "$LOG_DIR/docker_test_spatial.log" 2>&1
    
    # Wait for container to start
    sleep 30
    
    # Test health endpoint
    curl -f "http://localhost:8099/health" >> "$LOG_DIR/docker_test_spatial.log" 2>&1
    local health_status=$?
    
    # Stop test container
    docker stop pizza-spatial-test >> "$LOG_DIR/docker_test_spatial.log" 2>&1
    
    if [ $health_status -eq 0 ]; then
        success_log "Spatial Docker container test passed"
        return 0
    else
        error_log "Spatial Docker container test failed"
        return 1
    fi
}

# Function to test multi-environment deployment
test_multi_environment_deployment() {
    log "Testing multi-environment deployment..."
    
    # Test docker-compose stack
    log "Testing docker-compose deployment..."
    cd "$PROJECT_ROOT"
    
    # Start the stack
    docker-compose -f docker-compose.yml up -d --build \
        >> "$LOG_DIR/docker_compose.log" 2>&1
    
    # Wait for services to be ready
    sleep 60
    
    # Test service endpoints
    local test_passed=true
    
    # Test standard API
    if curl -f "http://localhost:8000/health" >> "$LOG_DIR/docker_compose.log" 2>&1; then
        log "Standard API endpoint healthy"
    else
        warning_log "Standard API endpoint failed"
        test_passed=false
    fi
    
    # Test spatial API
    if curl -f "http://localhost:8001/health" >> "$LOG_DIR/docker_compose.log" 2>&1; then
        log "Spatial API endpoint healthy"
    else
        warning_log "Spatial API endpoint failed"
        test_passed=false
    fi
    
    # Test load balancer
    if curl -f "http://localhost/health" >> "$LOG_DIR/docker_compose.log" 2>&1; then
        log "Load balancer healthy"
    else
        warning_log "Load balancer failed"
        test_passed=false
    fi
    
    # Cleanup
    docker-compose -f docker-compose.yml down >> "$LOG_DIR/docker_compose.log" 2>&1
    
    if [ "$test_passed" = true ]; then
        success_log "Multi-environment deployment test passed"
        return 0
    else
        error_log "Multi-environment deployment test failed"
        return 1
    fi
}

# Function to test model versioning system
test_model_versioning() {
    log "Testing model versioning system..."
    
    python3 "$PROJECT_ROOT/scripts/spatial_model_versioning.py" \
        --test-mode \
        --model-dir "$PROJECT_ROOT/models/spatial_mllm" \
        --output-dir "$ARTIFACTS_DIR/versioning_test" \
        >> "$LOG_DIR/model_versioning.log" 2>&1
    
    if [ $? -eq 0 ]; then
        success_log "Model versioning test completed"
        return 0
    else
        error_log "Model versioning test failed"
        return 1
    fi
}

# Function to run performance benchmarks
run_performance_benchmarks() {
    log "Running performance benchmarks..."
    
    python3 "$PROJECT_ROOT/scripts/spatial_performance_benchmark.py" \
        --output-dir "$ARTIFACTS_DIR/benchmarks" \
        --iterations 10 \
        --include-memory-profiling \
        >> "$LOG_DIR/performance_benchmark.log" 2>&1
    
    if [ $? -eq 0 ]; then
        success_log "Performance benchmarks completed"
        return 0
    else
        warning_log "Performance benchmarks had issues"
        return 1
    fi
}

# Function to generate deployment report
generate_deployment_report() {
    log "Generating deployment report..."
    
    python3 -c "
import json
import os
from datetime import datetime
from pathlib import Path

# Collect all test results and logs
report = {
    'pipeline_timestamp': '$TIMESTAMP',
    'pipeline_date': datetime.now().isoformat(),
    'spatial_pipeline_version': '4.2.0',
    'tests_run': [],
    'artifacts_generated': [],
    'deployment_status': 'success',
    'warnings': [],
    'errors': []
}

# Check for test results
test_files = [
    ('$SPATIAL_TEST_REPORT', 'spatial_feature_tests'),
    ('$MODEL_VALIDATION_REPORT', 'model_validation')
]

for test_file, test_name in test_files:
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            try:
                test_data = json.load(f)
                report['tests_run'].append({
                    'name': test_name,
                    'status': 'completed',
                    'file': test_file,
                    'summary': test_data.get('summary', {})
                })
            except:
                report['tests_run'].append({
                    'name': test_name,
                    'status': 'error',
                    'file': test_file
                })
    else:
        report['tests_run'].append({
            'name': test_name,
            'status': 'missing',
            'file': test_file
        })

# Check for artifacts
artifact_dir = Path('$ARTIFACTS_DIR')
if artifact_dir.exists():
    for item in artifact_dir.rglob('*'):
        if item.is_file():
            report['artifacts_generated'].append(str(item.relative_to(artifact_dir)))

# Check for errors in failed tests log
if os.path.exists('$FAILED_TESTS_LOG'):
    with open('$FAILED_TESTS_LOG', 'r') as f:
        failed_tests = f.read().strip().split('\n')
        if failed_tests and failed_tests[0]:
            report['errors'].extend(failed_tests)
            report['deployment_status'] = 'partial_failure'

# Write report
with open('$DEPLOYMENT_REPORT', 'w') as f:
    json.dump(report, f, indent=2)

print('Deployment report generated successfully')
"
    
    if [ $? -eq 0 ]; then
        success_log "Deployment report generated"
        return 0
    else
        error_log "Failed to generate deployment report"
        return 1
    fi
}

# Main pipeline execution
main() {
    log "======= SPATIAL-4.2 DEPLOYMENT PIPELINE START ======="
    log "Pipeline timestamp: $TIMESTAMP"
    log "Output directory: $OUTPUT_DIR"
    
    local pipeline_status="success"
    local phase_failures=0
    
    # Phase 1: Environment Validation
    log "======= PHASE 1: ENVIRONMENT VALIDATION ======="
    
    if ! check_cuda; then
        warning_log "CUDA check failed - continuing with CPU-only tests"
    fi
    
    if ! validate_spatial_environment; then
        error_log "Environment validation failed - stopping pipeline"
        exit 1
    fi
    
    # Phase 2: Model Validation
    log "======= PHASE 2: MODEL VALIDATION ======="
    
    if ! validate_spatial_models; then
        error_log "Model validation failed"
        pipeline_status="partial_failure"
        ((phase_failures++))
    fi
    
    # Phase 3: Feature Testing
    log "======= PHASE 3: SPATIAL FEATURE TESTING ======="
    
    if ! run_spatial_feature_tests; then
        error_log "Spatial feature tests failed"
        pipeline_status="partial_failure"
        ((phase_failures++))
    fi
    
    # Phase 4: Docker Testing
    log "======= PHASE 4: CONTAINER TESTING ======="
    
    if command -v docker &> /dev/null; then
        if ! test_docker_builds; then
            error_log "Docker build tests failed"
            pipeline_status="partial_failure"
            ((phase_failures++))
        fi
    else
        warning_log "Docker not available - skipping container tests"
    fi
    
    # Phase 5: Multi-Environment Testing
    log "======= PHASE 5: MULTI-ENVIRONMENT TESTING ======="
    
    if command -v docker-compose &> /dev/null; then
        if ! test_multi_environment_deployment; then
            error_log "Multi-environment deployment tests failed"
            pipeline_status="partial_failure"
            ((phase_failures++))
        fi
    else
        warning_log "Docker Compose not available - skipping multi-environment tests"
    fi
    
    # Phase 6: Model Versioning Testing
    log "======= PHASE 6: MODEL VERSIONING TESTING ======="
    
    if ! test_model_versioning; then
        error_log "Model versioning tests failed"
        pipeline_status="partial_failure"
        ((phase_failures++))
    fi
    
    # Phase 7: Performance Benchmarking
    log "======= PHASE 7: PERFORMANCE BENCHMARKING ======="
    
    if ! run_performance_benchmarks; then
        warning_log "Performance benchmarks had issues"
        # Note: Not counted as failure since it's informational
    fi
    
    # Phase 8: Report Generation
    log "======= PHASE 8: REPORT GENERATION ======="
    
    if ! generate_deployment_report; then
        error_log "Report generation failed"
        pipeline_status="partial_failure"
        ((phase_failures++))
    fi
    
    # Final summary
    log "======= SPATIAL-4.2 PIPELINE SUMMARY ======="
    log "Pipeline status: $pipeline_status"
    log "Phase failures: $phase_failures"
    log "Total duration: $(($(date +%s) - $(date -d \"$TIMESTAMP\" +%s 2>/dev/null || echo 0)))s"
    log "Reports available at: $REPORTS_DIR"
    log "Artifacts available at: $ARTIFACTS_DIR"
    
    # Create symlink to latest run
    ln -sf "$OUTPUT_DIR" "$PROJECT_ROOT/output/spatial_pipeline_latest"
    
    if [ "$pipeline_status" = "success" ]; then
        success_log "SPATIAL-4.2 deployment pipeline completed successfully!"
        exit 0
    else
        warning_log "SPATIAL-4.2 deployment pipeline completed with $phase_failures failures"
        exit 1
    fi
}

# Trap to cleanup on exit
cleanup() {
    log "Cleaning up pipeline resources..."
    
    # Stop any running containers
    docker stop pizza-spatial-test 2>/dev/null || true
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down 2>/dev/null || true
    
    # Clear CUDA cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

trap cleanup EXIT

# Run main pipeline
main "$@"
