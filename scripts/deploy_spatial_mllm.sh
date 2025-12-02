#!/bin/bash
# Complete Spatial-MLLM Deployment Script
# Part of SPATIAL-4.2: Deployment-Pipeline erweitern

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"
SKIP_TESTS="${SKIP_TESTS:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Cleaning up..."
        
        # Stop any running containers
        docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down --remove-orphans || true
    fi
    exit $exit_code
}

trap cleanup EXIT

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check required files
    required_files=(
        "$PROJECT_ROOT/Dockerfile.spatial"
        "$PROJECT_ROOT/docker-compose.yml"
        "$PROJECT_ROOT/spatial_requirements.txt"
        "$PROJECT_ROOT/config/nginx/nginx.conf"
        "$PROJECT_ROOT/config/monitoring/prometheus.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Function to validate model
validate_model() {
    log_info "Validating Spatial-MLLM model..."
    
    local model_path="$PROJECT_ROOT/models/spatial_mllm_model.pth"
    local validation_report="$PROJECT_ROOT/reports/model_validation_$(date +%Y%m%d_%H%M%S).json"
    
    # Create reports directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/reports"
    
    if [[ -f "$model_path" ]]; then
        # Run model validation
        if python3 "$PROJECT_ROOT/scripts/spatial_model_validation.py" \
            --model-path "$model_path" \
            --output-report "$validation_report" \
            --verbose; then
            log_success "Model validation passed"
        else
            log_error "Model validation failed"
            exit 1
        fi
    else
        log_warning "Model file not found, skipping validation"
    fi
}

# Function to build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build Spatial-MLLM image
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        docker build --no-cache -f Dockerfile.spatial -t spatial-mllm-api:latest .
    else
        docker build -f Dockerfile.spatial -t spatial-mllm-api:latest .
    fi
    
    log_success "Docker images built successfully"
}

# Function to deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p logs/{nginx,api,prometheus,grafana}
    mkdir -p data/{prometheus,grafana,redis}
    mkdir -p config/ssl
    
    # Set proper permissions
    chmod -R 755 logs data config
    
    # Deploy with Docker Compose
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    local max_attempts=12
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Checking service health (attempt $attempt/$max_attempts)..."
        
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API service is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "API service failed to start"
            docker-compose logs spatial-api-1
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    log_success "Services deployed successfully"
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (SKIP_TESTS=true)"
        return 0
    fi
    
    log_info "Running integration tests..."
    
    local test_report="$PROJECT_ROOT/reports/integration_test_$(date +%Y%m%d_%H%M%S).json"
    
    # Install test dependencies
    pip3 install aiohttp docker pyyaml psutil
    
    # Run integration tests
    if python3 "$PROJECT_ROOT/scripts/spatial_integration_tests.py" \
        --environment "$ENVIRONMENT" \
        --output-report "$test_report" \
        --verbose; then
        log_success "Integration tests passed"
    else
        log_error "Integration tests failed"
        log_info "Check test report: $test_report"
        exit 1
    fi
}

# Function to run spatial pipeline
run_spatial_pipeline() {
    log_info "Running spatial pipeline validation..."
    
    if "$PROJECT_ROOT/scripts/ci/run_spatial_pipeline.sh" "$ENVIRONMENT"; then
        log_success "Spatial pipeline validation passed"
    else
        log_error "Spatial pipeline validation failed"
        exit 1
    fi
}

# Function to setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Wait for Prometheus to be ready
    local max_attempts=6
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:9090/-/ready &> /dev/null; then
            log_success "Prometheus is ready"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_warning "Prometheus is not responding, monitoring may not be fully functional"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana is ready"
    else
        log_warning "Grafana is not responding"
    fi
    
    log_success "Monitoring setup completed"
}

# Function to perform post-deployment verification
post_deployment_verification() {
    log_info "Performing post-deployment verification..."
    
    # Check all services are running
    local services=("spatial-api-1" "spatial-api-2" "spatial-api-3" "nginx" "prometheus" "grafana" "redis")
    
    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$service"; then
            log_success "Service $service is running"
        else
            log_warning "Service $service is not running"
        fi
    done
    
    # Check load balancer
    if curl -f http://localhost/health &> /dev/null; then
        log_success "Load balancer is working"
    else
        log_warning "Load balancer health check failed"
    fi
    
    # Display service URLs
    log_info "Service URLs:"
    echo "  API (via load balancer): http://localhost/api/"
    echo "  Direct API access: http://localhost:8000/"
    echo "  Prometheus: http://localhost:9090/"
    echo "  Grafana: http://localhost:3000/ (admin/admin)"
    echo "  Nginx status: http://localhost:8080/nginx_status"
    
    log_success "Post-deployment verification completed"
}

# Function to generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="$PROJECT_ROOT/reports/deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "environment": "$ENVIRONMENT",
    "version": "SPATIAL-4.2",
    "status": "SUCCESS",
    "services": {
      "api_instances": 3,
      "load_balancer": "nginx",
      "monitoring": "prometheus + grafana",
      "cache": "redis",
      "model": "spatial-mllm"
    },
    "endpoints": {
      "main_api": "http://localhost/api/",
      "health_check": "http://localhost/health",
      "spatial_features": "http://localhost/api/spatial/",
      "monitoring": "http://localhost:9090/",
      "dashboard": "http://localhost:3000/"
    },
    "features": [
      "Spatial-MLLM integration",
      "Dual-encoder architecture",
      "Load balancing",
      "Auto-scaling",
      "Monitoring & alerting",
      "Performance benchmarking",
      "Model versioning",
      "Multi-environment support"
    ]
  }
}
EOF
    
    log_success "Deployment report saved to: $report_file"
}

# Main deployment function
main() {
    log_info "Starting Spatial-MLLM deployment pipeline for $ENVIRONMENT environment"
    log_info "Project root: $PROJECT_ROOT"
    
    # Execute deployment steps
    check_prerequisites
    validate_model
    build_images
    deploy_services
    setup_monitoring
    run_spatial_pipeline
    run_tests
    post_deployment_verification
    generate_deployment_report
    
    log_success "🎉 Spatial-MLLM deployment completed successfully!"
    log_info "The system is now ready for use."
    
    # Display final summary
    echo ""
    echo "========================================"
    echo "SPATIAL-4.2 DEPLOYMENT SUMMARY"
    echo "========================================"
    echo "Environment: $ENVIRONMENT"
    echo "Status: DEPLOYED"
    echo "API Endpoint: http://localhost/api/"
    echo "Monitoring: http://localhost:9090/"
    echo "Dashboard: http://localhost:3000/"
    echo "========================================"
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <environment> [options]"
    echo ""
    echo "Environments:"
    echo "  development  - Local development environment"
    echo "  staging      - Staging environment"
    echo "  production   - Production environment"
    echo ""
    echo "Environment Variables:"
    echo "  SKIP_TESTS=true     - Skip integration tests"
    echo "  FORCE_REBUILD=true  - Force rebuild of Docker images"
    echo ""
    echo "Example:"
    echo "  $0 development"
    echo "  SKIP_TESTS=true $0 staging"
    echo "  FORCE_REBUILD=true $0 production"
    exit 1
fi

# Run main function
main "$@"
