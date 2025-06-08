#!/bin/bash
"""
Pizza RL Training Launch Script (Aufgabe 4.1)
=============================================

Comprehensive training launcher for the Pizza RL system with monitoring and logging.
Supports different training configurations and scenarios.
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

# Default configuration
CONFIG_FILE="$PROJECT_ROOT/config/pizza_rl_training_config.json"
TIMESTEPS=500000
DEVICE="auto"
OUTPUT_DIR="$PROJECT_ROOT/results/pizza_rl_training_$(date +%Y%m%d_%H%M%S)"
USE_WANDB=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Pizza RL Training Script

OPTIONS:
    -c, --config FILE       Path to training configuration file (default: $CONFIG_FILE)
    -t, --timesteps NUM     Total training timesteps (default: $TIMESTEPS)
    -d, --device DEVICE     Training device: auto, cpu, cuda (default: $DEVICE)
    -o, --output DIR        Output directory (default: auto-generated)
    --no-wandb              Disable Weights & Biases logging
    --quick                 Quick training run (10k timesteps)
    --test                  Test run (1k timesteps)
    --resume CHECKPOINT     Resume from checkpoint
    -h, --help              Show this help message

EXAMPLES:
    # Standard training run
    $0
    
    # Quick test run
    $0 --quick
    
    # Custom configuration
    $0 -c custom_config.json -t 1000000
    
    # Resume from checkpoint
    $0 --resume checkpoint_iter_100.pth
    
    # CPU-only training
    $0 -d cpu --no-wandb

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -t|--timesteps)
                TIMESTEPS="$2"
                shift 2
                ;;
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --no-wandb)
                USE_WANDB=false
                shift
                ;;
            --quick)
                TIMESTEPS=10000
                OUTPUT_DIR="$PROJECT_ROOT/results/pizza_rl_quick_$(date +%Y%m%d_%H%M%S)"
                shift
                ;;
            --test)
                TIMESTEPS=1000
                OUTPUT_DIR="$PROJECT_ROOT/results/pizza_rl_test_$(date +%Y%m%d_%H%M%S)"
                shift
                ;;
            --resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        error "Virtual environment not found at $VENV_PATH"
        error "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    
    # Check if configuration file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        warn "Configuration file not found: $CONFIG_FILE"
        warn "Using default configuration"
        CONFIG_FILE=""
    fi
    
    # Check CUDA availability if requested
    if [[ "$DEVICE" == "cuda" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            warn "CUDA requested but nvidia-smi not found, falling back to auto detection"
            DEVICE="auto"
        fi
    fi
    
    success "System requirements check completed"
}

# Setup environment
setup_environment() {
    log "Setting up training environment..."
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/checkpoints"
    
    # Setup Python path
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Create training configuration if needed
    if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="$OUTPUT_DIR/training_config.json"
        log "Creating default training configuration at $CONFIG_FILE"
        python -c "
import json
config = {
    'total_timesteps': $TIMESTEPS,
    'output_dir': '$OUTPUT_DIR',
    'environment': {
        'max_steps_per_episode': 15,
        'battery_capacity_mah': 2500.0,
        'initial_battery_level': 0.9,
        'enable_logging': True
    },
    'ppo_hyperparams': {
        'learning_rate': 2.5e-4,
        'batch_size': 256,
        'ppo_epochs': 10,
        'rollout_steps': 2048
    }
}
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
"
    fi
    
    success "Environment setup completed"
}

# Monitor training process
monitor_training() {
    local PID=$1
    local LOG_FILE="$OUTPUT_DIR/logs/training.log"
    
    log "Monitoring training process (PID: $PID)"
    log "Log file: $LOG_FILE"
    
    # Monitor GPU usage if CUDA is available
    if command -v nvidia-smi &> /dev/null && [[ "$DEVICE" != "cpu" ]]; then
        log "Starting GPU monitoring..."
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 10 > "$OUTPUT_DIR/logs/gpu_usage.csv" &
        GPU_MONITOR_PID=$!
    fi
    
    # Monitor system resources
    log "Starting system resource monitoring..."
    (
        echo "timestamp,cpu_percent,memory_percent,disk_usage"
        while kill -0 $PID 2>/dev/null; do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
            memory_percent=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
            disk_usage=$(df "$OUTPUT_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
            echo "$timestamp,$cpu_percent,$memory_percent,$disk_usage"
            sleep 30
        done
    ) > "$OUTPUT_DIR/logs/system_usage.csv" &
    SYSTEM_MONITOR_PID=$!
    
    # Wait for training to complete
    wait $PID
    TRAINING_EXIT_CODE=$?
    
    # Stop monitoring processes
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
    if [[ -n "$SYSTEM_MONITOR_PID" ]]; then
        kill $SYSTEM_MONITOR_PID 2>/dev/null || true
    fi
    
    return $TRAINING_EXIT_CODE
}

# Generate training summary
generate_summary() {
    log "Generating training summary..."
    
    python << EOF
import json
import os
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
config_file = '$CONFIG_FILE'

# Load configuration
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = {}

# Create summary
summary = {
    'training_config': config,
    'timesteps': $TIMESTEPS,
    'device': '$DEVICE',
    'output_directory': str(output_dir),
    'wandb_enabled': $USE_WANDB,
    'training_completed': True,
    'timestamp': '$(date -Iseconds)'
}

# Save summary
with open(output_dir / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Training summary saved to {output_dir / 'training_summary.json'}")
EOF
    
    success "Training summary generated"
}

# Main execution
main() {
    log "Starting Pizza RL Training (Aufgabe 4.1)"
    log "=========================================="
    
    # Parse arguments
    parse_args "$@"
    
    # System checks
    check_requirements
    
    # Setup environment
    setup_environment
    
    # Display configuration
    log "Training Configuration:"
    log "  Configuration file: $CONFIG_FILE"
    log "  Timesteps: $TIMESTEPS"
    log "  Device: $DEVICE"
    log "  Output directory: $OUTPUT_DIR"
    log "  Weights & Biases: $USE_WANDB"
    
    # Build training command
    TRAIN_CMD="python $PROJECT_ROOT/scripts/train_pizza_rl.py"
    
    if [[ -n "$CONFIG_FILE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
    fi
    
    TRAIN_CMD="$TRAIN_CMD --device $DEVICE --timesteps $TIMESTEPS --output-dir $OUTPUT_DIR"
    
    if [[ "$USE_WANDB" == "false" ]]; then
        TRAIN_CMD="$TRAIN_CMD --no-wandb"
    fi
    
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CHECKPOINT"
    fi
    
    log "Training command: $TRAIN_CMD"
    
    # Start training
    log "Starting training process..."
    $TRAIN_CMD > "$OUTPUT_DIR/logs/training.log" 2>&1 &
    TRAINING_PID=$!
    
    # Monitor training
    if monitor_training $TRAINING_PID; then
        success "Training completed successfully!"
        generate_summary
        
        log "Training results available in: $OUTPUT_DIR"
        log "View logs: tail -f $OUTPUT_DIR/logs/training.log"
        
        if [[ "$USE_WANDB" == "true" ]]; then
            log "View training metrics on Weights & Biases dashboard"
        fi
        
    else
        error "Training failed with exit code: $TRAINING_EXIT_CODE"
        error "Check logs at: $OUTPUT_DIR/logs/training.log"
        exit $TRAINING_EXIT_CODE
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
