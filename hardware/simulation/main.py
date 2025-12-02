#!/usr/bin/env python3
"""
Main training script for PG-ES Sim-to-Real Hardware Optimization
Run this script to start the hybrid training process
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import MasterConfig, create_debug_config, create_production_config
from src.hybrid_trainer import HybridPGESTrainer

def setup_logging(log_level="INFO", log_dir="./logs"):
    """Setup logging configuration"""
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Log file: {log_file}")
    
    return logger

def main():
    """Main training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PG-ES Sim-to-Real Hardware Optimization")
    parser.add_argument("--config", type=str, default="production", 
                       choices=["debug", "production"], 
                       help="Configuration preset to use")
    parser.add_argument("--iterations", type=int, default=10000,
                       help="Total training iterations")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory for log files")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_dir)
    
    try:
        # Load configuration
        if args.config == "debug":
            config = create_debug_config()
            logger.info("Using debug configuration")
        else:
            config = create_production_config()
            logger.info("Using production configuration")
        
        # Update log directory in config
        config.logging.log_dir = args.log_dir
        
        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            logger.error("Configuration validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return 1
        
        logger.info("Configuration validated successfully")
        
        # Print configuration summary
        logger.info("=== Configuration Summary ===")
        logger.info(f"Simulation duration: {config.simulation.sim_duration}s")
        logger.info(f"PG learning rate: {config.pg.learning_rate}")
        logger.info(f"ES population size: {config.es.population_size}")
        logger.info(f"ES mutation strength: {config.es.sigma}")
        logger.info(f"PG-ES ratio: {config.hybrid.pg_es_ratio}")
        logger.info(f"Safety max temp: {config.safety.max_temperature}°C")
        logger.info(f"Domain randomization: {config.domain_rand.enable_thermal_events}")
        logger.info("=============================")
        
        # Initialize trainer
        logger.info("Initializing hybrid trainer...")
        trainer = HybridPGESTrainer(config)
        
        # Start training
        logger.info(f"Starting training for {args.iterations} iterations...")
        logger.info("Press Ctrl+C to stop training gracefully")
        
        # Run training
        final_metrics = trainer.train(total_iterations=args.iterations)
        
        # Print final results
        logger.info("=== Training Completed ===")
        logger.info(f"Total iterations: {args.iterations}")
        logger.info(f"Wall time: {final_metrics.wall_time:.1f}s")
        logger.info(f"Final combined fitness: {final_metrics.combined_fitness:.6f}")
        logger.info(f"PG contribution: {final_metrics.pg_contribution_ratio:.3f}")
        logger.info(f"ES contribution: {final_metrics.es_contribution_ratio:.3f}")
        logger.info(f"Total supervisor alerts: {final_metrics.supervisor_alerts}")
        logger.info(f"Results saved to: {config.logging.log_dir}")
        logger.info("==========================")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
