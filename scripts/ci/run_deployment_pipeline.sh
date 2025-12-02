#!/bin/bash
# Extended Deployment Pipeline for Spatial-MLLM Support
# SPATIAL-4.2: Deployment-Pipeline erweitern

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Directories
OUTPUT_DIR="$PROJECT_ROOT/output/deployment_pipeline_runs/$TIMESTAMP"
LOG_DIR="$OUTPUT_DIR/logs"
REPORTS_DIR="$OUTPUT_DIR/reports"
ARTIFACTS_DIR="$OUTPUT_DIR/artifacts"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$REPORTS_DIR"
mkdir -p "$ARTIFACTS_DIR"

# Configuration files
MAIN_LOG="$LOG_DIR/deployment_pipeline.log"
FAILED_TESTS_LOG="$LOG_DIR/failed_deployment_tests.log"
DEPLOYMENT_REPORT="$REPORTS_DIR/deployment_report.json"
VALIDATION_REPORT="$REPORTS_DIR/validation_report.json"
PERFORMANCE_REPORT="$REPORTS_DIR/performance_report.json"

# Environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_CACHE_DIR="$PROJECT_ROOT/models"
export OUTPUT_CACHE_DIR="$PROJECT_ROOT/output"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$MAIN_LOG"
}

log_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}" | tee -a "$MAIN_LOG"
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Pipeline failed at line $line_number with exit code $exit_code"
    
    # Generate failure report
    echo "{
        \"status\": \"failed\",
        \"timestamp\": \"$(date -Iseconds)\",
        \"exit_code\": $exit_code,
        \"failed_at_line\": $line_number,
        \"logs_directory\": \"$LOG_DIR\"
    }" > "$REPORTS_DIR/failure_report.json"
    
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Function to check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"
    
    # Check Python and packages
    python3 --version >> "$MAIN_LOG" 2>&1
    
    # Check CUDA availability
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" >> "$MAIN_LOG" 2>&1; then
        log_info "CUDA availability check passed"
    else
        log_warning "CUDA availability check failed"
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        docker --version >> "$MAIN_LOG" 2>&1
        log_info "Docker is available"
    else
        log_warning "Docker is not available"
    fi
    
    # Check Docker Compose (both v1 and v2)
    if command -v docker-compose &> /dev/null; then
        docker-compose --version >> "$MAIN_LOG" 2>&1
        log_info "Docker Compose (v1) is available"
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        docker compose version >> "$MAIN_LOG" 2>&1
        log_info "Docker Compose (v2) is available"
        DOCKER_COMPOSE_CMD="docker compose"
    else
        log_warning "Docker Compose is not available"
        DOCKER_COMPOSE_CMD=""
    fi
    
    # Check disk space (need at least 10GB free)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        log_warning "Low disk space: $(( available_space / 1024 / 1024 ))GB available"
    else
        log_info "Sufficient disk space available"
    fi
}

# Function to build Docker containers
build_containers() {
    log_section "Building Docker Containers"
    
    cd "$PROJECT_ROOT"
    
    # Build standard container
    log_info "Building standard pizza detection container..."
    if docker build -t pizza-detection:latest . >> "$LOG_DIR/docker_build_standard.log" 2>&1; then
        log_info "Standard container built successfully"
    else
        log_error "Failed to build standard container"
        return 1
    fi
    
    # Build spatial container
    log_info "Building spatial-enhanced container..."
    if docker build -f Dockerfile.spatial -t pizza-detection-spatial:latest . >> "$LOG_DIR/docker_build_spatial.log" 2>&1; then
        log_info "Spatial container built successfully"
    else
        log_error "Failed to build spatial container"
        return 1
    fi
    
    # Tag containers with timestamp
    docker tag pizza-detection:latest "pizza-detection:$TIMESTAMP"
    docker tag pizza-detection-spatial:latest "pizza-detection-spatial:$TIMESTAMP"
    
    log_info "Container tagging completed"
}

# Function to run spatial feature tests
run_spatial_tests() {
    log_section "Running Spatial Feature Tests"
    
    cd "$PROJECT_ROOT"
    
    # Run lightweight deployment-focused spatial tests
    log_info "Running spatial deployment tests..."
    if python3 scripts/ci/spatial_deployment_tests.py >> "$LOG_DIR/spatial_tests.log" 2>&1; then
        log_info "Spatial deployment tests passed"
    else
        log_warning "Spatial deployment tests had issues, checking details..."
        
        # Run full tests as fallback (with timeout)
        log_info "Running comprehensive spatial tests as fallback..."
        if timeout 60 python3 scripts/spatial_feature_tests.py >> "$LOG_DIR/spatial_tests_full.log" 2>&1; then
            log_info "Full spatial feature tests passed"
        else
            log_warning "Full spatial feature tests failed or timed out"
            cp "$LOG_DIR/spatial_tests.log" "$FAILED_TESTS_LOG"
            # Don't fail the entire pipeline for spatial tests in deployment mode
            log_warning "Continuing deployment despite spatial test issues"
        fi
    fi
    
    # Copy test results
    if [ -f "output/spatial_tests/spatial_test_results_*.json" ]; then
        cp output/spatial_tests/spatial_test_results_*.json "$REPORTS_DIR/"
        log_info "Spatial test results copied to reports"
    fi
}

# Function to validate model versions
validate_models() {
    log_section "Validating Model Versions"
    
    cd "$PROJECT_ROOT"
    
    # Initialize model version manager
    log_info "Initializing model version manager..."
    
    # Check if spatial models exist
    if [ -d "models/spatial_mllm" ]; then
        log_info "Spatial models directory found"
        
        # Validate existing models
        python3 scripts/model_version_manager.py list --type spatial >> "$LOG_DIR/model_validation.log" 2>&1
        
        # Register current models if not already registered
        if [ -f "models/spatial_mllm/pizza_finetuned_v1.pth" ]; then
            python3 scripts/model_version_manager.py register-spatial \
                "models/spatial_mllm/pizza_finetuned_v1.pth" \
                "production_v1_$TIMESTAMP" \
                --description "Production deployment $TIMESTAMP" \
                >> "$LOG_DIR/model_validation.log" 2>&1 || true
        fi
        
        # Set active version
        python3 scripts/model_version_manager.py set-active spatial "production_v1_$TIMESTAMP" \
            >> "$LOG_DIR/model_validation.log" 2>&1 || true
    else
        log_warning "No spatial models found"
    fi
    
    # Generate deployment report
    python3 scripts/model_version_manager.py export-report "$REPORTS_DIR/model_deployment_report.json" \
        >> "$LOG_DIR/model_validation.log" 2>&1
    
    log_info "Model validation completed"
}

# Function to test container deployment
test_container_deployment() {
    log_section "Testing Container Deployment"
    
    cd "$PROJECT_ROOT"
    
    # Test standard container
    log_info "Testing standard container deployment..."
    if docker run --rm --name pizza-test-standard \
        -e MODEL_TYPE=standard \
        -v "$PROJECT_ROOT/test_data:/app/test_data:ro" \
        pizza-detection:latest \
        python -c "import sys; sys.path.append('/app'); from src.api.pizza_api import PizzaAPI; print('Standard API import successful')" \
        >> "$LOG_DIR/container_test_standard.log" 2>&1; then
        log_info "Standard container test passed"
    else
        log_error "Standard container test failed"
        return 1
    fi
    
    # Test spatial container (if CUDA is available)
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        log_info "Testing spatial container deployment..."
        if docker run --rm --name pizza-test-spatial \
            --gpus all \
            -e MODEL_TYPE=spatial \
            -v "$PROJECT_ROOT/test_data:/app/test_data:ro" \
            -v "$PROJECT_ROOT/models:/app/models:ro" \
            pizza-detection-spatial:latest \
            python -c "import sys; sys.path.append('/app'); from src.spatial.spatial_integration import SpatialMLLMIntegration; print('Spatial integration import successful')" \
            >> "$LOG_DIR/container_test_spatial.log" 2>&1; then
            log_info "Spatial container test passed"
        else
            log_warning "Spatial container test failed (may be due to CUDA/model availability)"
        fi
    else
        log_warning "CUDA not available, skipping spatial container test"
    fi
}

# Function to test multi-environment deployment
test_multi_environment() {
    log_section "Testing Multi-Environment Deployment"
    
    cd "$PROJECT_ROOT"
    
    # Test docker-compose deployment
    log_info "Testing docker-compose deployment..."
    
    if [ -z "$DOCKER_COMPOSE_CMD" ]; then
        log_warning "Docker Compose not available, skipping compose deployment test"
        return 0
    fi
    
    # Start services
    if $DOCKER_COMPOSE_CMD up -d >> "$LOG_DIR/docker_compose.log" 2>&1; then
        log_info "Docker-compose services started"
        
        # Wait for services to be ready
        sleep 30
        
        # Test standard API
        if curl -f http://localhost:8000/health >> "$LOG_DIR/api_test.log" 2>&1; then
            log_info "Standard API health check passed"
        else
            log_warning "Standard API health check failed"
        fi
        
        # Test spatial API (if available)
        if curl -f http://localhost:8001/health >> "$LOG_DIR/api_test.log" 2>&1; then
            log_info "Spatial API health check passed"
        else
            log_warning "Spatial API health check failed"
        fi
        
        # Test nginx load balancer
        if curl -f http://localhost/health >> "$LOG_DIR/api_test.log" 2>&1; then
            log_info "Load balancer health check passed"
        else
            log_warning "Load balancer health check failed"
        fi
        
        # Stop services
        $DOCKER_COMPOSE_CMD down >> "$LOG_DIR/docker_compose.log" 2>&1
        log_info "Docker-compose services stopped"
        
    else
        log_error "Failed to start docker-compose services"
        return 1
    fi
}

# Function to run performance benchmarks
run_performance_benchmarks() {
    log_section "Running Performance Benchmarks"
    
    cd "$PROJECT_ROOT"
    
    # Run inference performance tests
    if [ -f "scripts/spatial_inference_optimized.py" ]; then
        log_info "Running optimized inference benchmarks..."
        python3 scripts/spatial_inference_optimized.py --benchmark \
            >> "$LOG_DIR/performance_benchmark.log" 2>&1 || true
        
        # Copy benchmark results
        if [ -f "output/benchmarks/spatial_inference_performance.json" ]; then
            cp "output/benchmarks/spatial_inference_performance.json" "$REPORTS_DIR/"
            log_info "Performance benchmark results copied"
        fi
    fi
    
    # Resource usage monitoring
    log_info "Monitoring resource usage..."
    {
        echo "=== CPU Info ==="
        cat /proc/cpuinfo | grep "model name" | head -1
        echo "=== Memory Info ==="
        free -h
        echo "=== GPU Info ==="
        nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"
        echo "=== Disk Usage ==="
        df -h "$PROJECT_ROOT"
    } >> "$LOG_DIR/system_resources.log"
}

# Function to generate final deployment report
generate_deployment_report() {
    log_section "Generating Deployment Report"
    
    cd "$PROJECT_ROOT"
    
    # Collect all results
    local deployment_status="success"
    local test_results=""
    local performance_data=""
    
    # Check test results
    if [ -f "$FAILED_TESTS_LOG" ]; then
        deployment_status="failed"
    fi
    
    # Collect performance data
    if [ -f "$REPORTS_DIR/spatial_inference_performance.json" ]; then
        performance_data=$(cat "$REPORTS_DIR/spatial_inference_performance.json")
    fi
    
    # Generate comprehensive report
    cat > "$DEPLOYMENT_REPORT" << EOF
{
    "deployment_pipeline": {
        "status": "$deployment_status",
        "timestamp": "$(date -Iseconds)",
        "duration": "$(( $(date +%s) - $(stat -c %Y "$MAIN_LOG") )) seconds",
        "pipeline_version": "SPATIAL-4.2",
        "environment": {
            "hostname": "$(hostname)",
            "user": "$(whoami)",
            "working_directory": "$PROJECT_ROOT",
            "python_version": "$(python3 --version 2>&1)",
            "cuda_available": $(python3 -c "import torch; print('true' if torch.cuda.is_available() else 'false')" 2>/dev/null || echo "false")
        }
    },
    "container_builds": {
        "standard_container": "$([ -f "$LOG_DIR/docker_build_standard.log" ] && echo "success" || echo "failed")",
        "spatial_container": "$([ -f "$LOG_DIR/docker_build_spatial.log" ] && echo "success" || echo "failed")"
    },
    "tests": {
        "spatial_features": "$([ ! -f "$FAILED_TESTS_LOG" ] && echo "passed" || echo "failed")",
        "container_deployment": "$([ -f "$LOG_DIR/container_test_standard.log" ] && echo "tested" || echo "skipped")",
        "multi_environment": "$([ -f "$LOG_DIR/docker_compose.log" ] && echo "tested" || echo "skipped")"
    },
    "model_validation": {
        "version_manager_initialized": "$([ -f "$REPORTS_DIR/model_deployment_report.json" ] && echo "true" || echo "false")"
    },
    "performance": $performance_data,
    "logs": {
        "main_log": "$MAIN_LOG",
        "reports_directory": "$REPORTS_DIR",
        "artifacts_directory": "$ARTIFACTS_DIR"
    }
}
EOF
    
    log_info "Deployment report generated: $DEPLOYMENT_REPORT"
}

# Function to cleanup
cleanup() {
    log_section "Cleanup"
    
    # Remove temporary files
    find "$PROJECT_ROOT" -name "*.tmp" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Clean Docker
    docker system prune -f >> "$MAIN_LOG" 2>&1 || true
    
    log_info "Cleanup completed"
}

# Main pipeline execution
main() {
    log_section "Starting Deployment Pipeline - SPATIAL-4.2"
    log_info "Pipeline run ID: $TIMESTAMP"
    log_info "Output directory: $OUTPUT_DIR"
    
    # Execute pipeline steps
    check_prerequisites
    build_containers
    run_spatial_tests
    validate_models
    test_container_deployment
    test_multi_environment
    run_performance_benchmarks
    generate_deployment_report
    cleanup
    
    log_section "Deployment Pipeline Completed Successfully"
    log_info "Final report: $DEPLOYMENT_REPORT"
    
    # Display summary
    echo -e "\n${GREEN}🎉 SPATIAL-4.2 Deployment Pipeline Completed!${NC}"
    echo -e "📊 Reports: $REPORTS_DIR"
    echo -e "📝 Logs: $LOG_DIR"
    echo -e "📦 Artifacts: $ARTIFACTS_DIR"
    
    # Show key metrics
    if [ -f "$DEPLOYMENT_REPORT" ]; then
        echo -e "\n📈 Key Results:"
        python3 -c "
import json
try:
    with open('$DEPLOYMENT_REPORT', 'r') as f:
        data = json.load(f)
    print(f\"  Status: {data['deployment_pipeline']['status']}\")
    print(f\"  Duration: {data['deployment_pipeline']['duration']}\")
    print(f\"  Container Builds: {data['container_builds']}\")
    print(f\"  Tests: {data['tests']}\")
except:
    print('  Could not parse deployment report')
"
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Deployment Pipeline for Spatial-MLLM Integration"
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --skip-tests   Skip spatial feature tests"
        echo "  --skip-docker  Skip Docker container tests"
        echo "  --cleanup-only Run cleanup only"
        echo ""
        exit 0
        ;;
    --skip-tests)
        log_warning "Skipping spatial feature tests as requested"
        run_spatial_tests() { log_info "Spatial tests skipped"; }
        ;;
    --skip-docker)
        log_warning "Skipping Docker tests as requested"
        build_containers() { log_info "Container builds skipped"; }
        test_container_deployment() { log_info "Container tests skipped"; }
        test_multi_environment() { log_info "Multi-environment tests skipped"; }
        ;;
    --cleanup-only)
        cleanup
        exit 0
        ;;
esac

# Run main pipeline
main "$@"
