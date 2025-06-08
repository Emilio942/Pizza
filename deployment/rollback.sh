#!/bin/bash
# SPATIAL-6.2: Production Rollback Strategy
# Emergency rollback script for pizza classification system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/deployment/backups"
LOG_FILE="$PROJECT_ROOT/deployment/rollback.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Warning message
warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

# Check if running as root or with docker permissions
check_permissions() {
    if ! docker ps >/dev/null 2>&1; then
        error_exit "Docker permission denied. Please run with sudo or add user to docker group."
    fi
}

# Create backup of current deployment
create_backup() {
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    log "Creating backup: $backup_name"
    mkdir -p "$backup_path"
    
    # Backup current containers
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        docker compose -f "$PROJECT_ROOT/deployment/docker-compose.prod.yml" ps --format json > "$backup_path/containers.json"
    elif command -v docker-compose &> /dev/null; then
        docker-compose -f "$PROJECT_ROOT/deployment/docker-compose.prod.yml" ps --format json > "$backup_path/containers.json"
    else
        log "WARNING: No docker compose command found, skipping container backup"
    fi
    
    # Backup current images
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}" | grep pizza > "$backup_path/images.txt" || true
    
    # Backup configuration
    cp -r "$PROJECT_ROOT/deployment/"*.{yml,env,conf} "$backup_path/" 2>/dev/null || true
    
    echo "$backup_name" > "$BACKUP_DIR/latest_backup.txt"
    success "Backup created: $backup_path"
}

# Stop current services
stop_services() {
    log "Stopping current services..."
    cd "$PROJECT_ROOT"
    
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log "WARNING: No docker compose command found"
        COMPOSE_CMD=""
    fi
    
    if [ -n "$COMPOSE_CMD" ] && $COMPOSE_CMD -f deployment/docker-compose.prod.yml ps | grep -q "Up"; then
        $COMPOSE_CMD -f deployment/docker-compose.prod.yml down
        success "Services stopped"
    else
        warning "No running services found or docker compose unavailable"
    fi
}

# Rollback to previous version
rollback_to_version() {
    local version=${1:-"previous"}
    log "Rolling back to version: $version"
    
    case $version in
        "previous"|"last")
            rollback_to_previous
            ;;
        "stable")
            rollback_to_stable
            ;;
        "emergency")
            rollback_to_emergency
            ;;
        *)
            error_exit "Unknown rollback version: $version"
            ;;
    esac
}

# Rollback to previous backup
rollback_to_previous() {
    if [ ! -f "$BACKUP_DIR/latest_backup.txt" ]; then
        error_exit "No previous backup found"
    fi
    
    local backup_name=$(cat "$BACKUP_DIR/latest_backup.txt")
    local backup_path="$BACKUP_DIR/$backup_name"
    
    if [ ! -d "$backup_path" ]; then
        error_exit "Backup directory not found: $backup_path"
    fi
    
    log "Restoring from backup: $backup_name"
    
    # Restore configuration files
    cp "$backup_path/"*.{yml,env,conf} "$PROJECT_ROOT/deployment/" 2>/dev/null || true
    
    success "Rollback to previous version completed"
}

# Rollback to stable version (standard CNN only)
rollback_to_stable() {
    log "Rolling back to stable configuration (Standard CNN only)"
    
    # Create emergency configuration
    cat > "$PROJECT_ROOT/deployment/emergency.env" << EOF
# Emergency rollback configuration - Standard CNN only
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=2
MAX_CONCURRENT_REQUESTS=50

# Use only standard model
PRIMARY_MODEL=standard-cnn
FALLBACK_MODEL=standard-cnn
MODEL_TIMEOUT=10
INFERENCE_TIMEOUT=5

# Reduced resource usage
GPU_MEMORY_LIMIT=2048
CPU_WORKERS=2
CACHE_SIZE=100

# Monitoring disabled for stability
PROMETHEUS_ENABLED=false
LOG_LEVEL=ERROR
METRICS_COLLECTION=false

# Health checks
HEALTH_CHECK_INTERVAL=60
MODEL_WARMUP_ENABLED=false
STARTUP_TIMEOUT=120

# Deployment
DEPLOYMENT_MODE=emergency
VERSION=emergency-rollback
BUILD_DATE=$(date -Iseconds)
EOF

    success "Emergency configuration created"
}

# Emergency rollback (API server only)
rollback_to_emergency() {
    log "Performing emergency rollback - API server only"
    
    # Stop all services
    stop_services
    
    # Start only the API server with minimal configuration
    cd "$PROJECT_ROOT"
    
    # Use development server as emergency fallback
    log "Starting emergency API server..."
    python3 -m src.api.pizza_api --host 0.0.0.0 --port 8001 &
    local api_pid=$!
    
    sleep 10
    
    # Check if API is responsive
    if curl -s http://localhost:8001/health > /dev/null; then
        success "Emergency API server started (PID: $api_pid)"
        echo $api_pid > "$PROJECT_ROOT/deployment/emergency.pid"
    else
        error_exit "Emergency API server failed to start"
    fi
}

# Health check after rollback
health_check() {
    log "Performing health check..."
    
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8001/health > /dev/null; then
            success "Health check passed"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    error_exit "Health check failed after $max_attempts attempts"
}

# Start services after rollback
start_services() {
    log "Starting services..."
    cd "$PROJECT_ROOT"
    
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log "WARNING: No docker compose command found"
        COMPOSE_CMD=""
    fi
    
    if [ -n "$COMPOSE_CMD" ]; then
        $COMPOSE_CMD -f deployment/docker-compose.prod.yml up -d
        
        # Wait for services to be ready
        sleep 30
        health_check
        
        success "Services started successfully"
    else
        warning "Docker compose not available, starting emergency API server"
        rollback_to_emergency
    fi
}

# Main rollback function
main() {
    local rollback_type=${1:-"previous"}
    
    log "Starting rollback procedure (type: $rollback_type)"
    
    check_permissions
    create_backup
    stop_services
    rollback_to_version "$rollback_type"
    
    if [ "$rollback_type" != "emergency" ]; then
        start_services
    fi
    
    success "Rollback procedure completed successfully"
    log "Check logs at: $LOG_FILE"
}

# Show usage
usage() {
    echo "Usage: $0 [rollback_type]"
    echo ""
    echo "Rollback types:"
    echo "  previous  - Rollback to previous backup (default)"
    echo "  stable    - Rollback to stable configuration (Standard CNN only)"
    echo "  emergency - Emergency rollback (API server only)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Rollback to previous version"
    echo "  $0 stable            # Rollback to stable configuration"
    echo "  $0 emergency         # Emergency rollback"
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
