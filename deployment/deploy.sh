#!/bin/bash
# SPATIAL-6.2: Production Deployment Script
# Deploy hybrid dual-model pizza classification system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/deployment/deploy.log"

# Define Docker Compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

# Info message
info() {
    log "${BLUE}INFO: $1${NC}"
}

# Warning message
warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi
    
    # Check Docker Compose availability (plugin or standalone)
    if ! (command -v docker-compose &> /dev/null || docker compose version &> /dev/null); then
        error_exit "Docker Compose is not installed"
    fi
    
    # Check Docker permissions
    if ! docker ps >/dev/null 2>&1; then
        error_exit "Docker permission denied. Please run with sudo or add user to docker group."
    fi
    
    # Check NVIDIA runtime (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        warning "NVIDIA Docker runtime not available. GPU acceleration disabled."
    else
        success "NVIDIA Docker runtime available"
    fi
    
    # Check required files
    local required_files=(
        "$PROJECT_ROOT/deployment/docker-compose.prod.yml"
        "$PROJECT_ROOT/deployment/production.env"
        "$PROJECT_ROOT/deployment/Dockerfile.prod"
        "$PROJECT_ROOT/src/api/pizza_api.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error_exit "Required file not found: $file"
        fi
    done
    
    success "Prerequisites check passed"
}

# Validate models
validate_models() {
    info "Validating models..."
    
    # Check if models directory exists
    if [ ! -d "$PROJECT_ROOT/models" ]; then
        error_exit "Models directory not found: $PROJECT_ROOT/models"
    fi
    
    # Check for required model files
    if [ ! -d "$PROJECT_ROOT/models/spatial_mllm" ] && [ ! -d "$PROJECT_ROOT/models/standard" ]; then
        error_exit "No model directories found. Please ensure models are available."
    fi
    
    success "Model validation passed"
}

# Build production images
build_images() {
    info "Building production Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build -f deployment/Dockerfile.prod -t pizza-classification:latest . || error_exit "Failed to build application image"
    
    # Tag with version
    local version=$(date +%Y%m%d-%H%M%S)
    docker tag pizza-classification:latest pizza-classification:$version
    
    success "Docker images built successfully"
    info "Image tagged as: pizza-classification:$version"
}

# Create necessary directories
create_directories() {
    info "Creating necessary directories..."
    
    local dirs=(
        "$PROJECT_ROOT/deployment/backups"
        "$PROJECT_ROOT/deployment/logs"
        "$PROJECT_ROOT/deployment/dashboards"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    success "Directories created"
}

# Deploy services
deploy_services() {
    info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing services if running
    if $COMPOSE_CMD -f deployment/docker-compose.prod.yml ps | grep -q "Up"; then
        warning "Stopping existing services..."
        $COMPOSE_CMD -f deployment/docker-compose.prod.yml down
    fi
    
    # Start services
    $COMPOSE_CMD -f deployment/docker-compose.prod.yml up -d
    
    success "Services deployed"
}

# Wait for services to be ready
wait_for_services() {
    info "Waiting for services to be ready..."
    
    local services=("pizza-api:8001" "nginx:80")
    local max_attempts=30
    
    for service in "${services[@]}"; do
        local host_port=(${service//:/ })
        local host=${host_port[0]}
        local port=${host_port[1]}
        local attempt=1
        
        info "Checking $service..."
        
        while [ $attempt -le $max_attempts ]; do
            if $COMPOSE_CMD -f "$PROJECT_ROOT/deployment/docker-compose.prod.yml" exec -T "$host" curl -s http://localhost:$port/health > /dev/null 2>&1; then
                success "$service is ready"
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                error_exit "$service failed to start after $max_attempts attempts"
            fi
            
            sleep 10
            ((attempt++))
        done
    done
}

# Run health checks
run_health_checks() {
    info "Running comprehensive health checks..."
    
    # API health check
    local api_response=$(curl -s http://localhost:8001/health)
    if echo "$api_response" | grep -q "healthy"; then
        success "API health check passed"
    else
        error_exit "API health check failed"
    fi
    
    # Model status check
    local status_response=$(curl -s http://localhost:8001/status)
    if echo "$status_response" | grep -q "models_loaded.*true"; then
        success "Models loaded successfully"
    else
        error_exit "Models not loaded properly"
    fi
    
    # Load balancer check
    local nginx_response=$(curl -s http://localhost:80/health)
    if echo "$nginx_response" | grep -q "healthy"; then
        success "Load balancer health check passed"
    else
        warning "Load balancer health check failed"
    fi
    
    success "Health checks completed"
}

# Display deployment information
show_deployment_info() {
    info "Deployment completed successfully!"
    echo ""
    echo "🍕 Pizza Classification System - Production Deployment"
    echo "================================================"
    echo ""
    echo "📊 Service Endpoints:"
    echo "  • Main API:        http://localhost:80"
    echo "  • Direct API:      http://localhost:8001" 
    echo "  • Health Check:    http://localhost:80/health"
    echo "  • System Status:   http://localhost:80/status"
    echo ""
    echo "📈 Monitoring:"
    echo "  • Prometheus:      http://localhost:9090"
    echo "  • Grafana:         http://localhost:3000 (admin/admin123)"
    echo ""
    echo "🔧 Management:"
    echo "  • View logs:       $COMPOSE_CMD -f deployment/docker-compose.prod.yml logs -f"
    echo "  • Stop services:   $COMPOSE_CMD -f deployment/docker-compose.prod.yml down"
    echo "  • Rollback:        ./deployment/rollback.sh"
    echo ""
    echo "📋 Deployment Details:"
    echo "  • Primary Model:   Spatial-MLLM (Float16)"
    echo "  • Fallback Model:  Standard CNN (INT8)"
    echo "  • Deployment Mode: Production"
    echo "  • Version:         $(date +%Y%m%d-%H%M%S)"
    echo ""
}

# Main deployment function
main() {
    log "Starting SPATIAL-6.2 production deployment..."
    
    check_prerequisites
    validate_models
    create_directories
    build_images
    deploy_services
    wait_for_services
    run_health_checks
    show_deployment_info
    
    success "SPATIAL-6.2 production deployment completed successfully!"
}

# Show usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo "  --skip-build  Skip Docker image building"
    echo "  --skip-health Skip health checks"
    echo ""
    echo "This script deploys the hybrid dual-model pizza classification system"
    echo "to production with full monitoring and load balancing."
}

# Handle command line arguments
case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
