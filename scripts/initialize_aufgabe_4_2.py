#!/usr/bin/env python3
"""
Initialize Aufgabe 4.2: Continuous Pizza Verifier Improvement
Integration script that works alongside ongoing RL training

This script:
- Sets up the continuous improvement system
- Integrates with ongoing RL training (Aufgabe 4.1)
- Prepares for automatic improvement when training completes
- Provides real-time monitoring of both systems
"""

import json
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.continuous_improvement.pizza_verifier_improvement import ContinuousPizzaVerifierImprovement

def setup_logging():
    """Setup logging for initialization"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/aufgabe_4_2_init.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_improvement_config(config_path: str) -> dict:
    """Load continuous improvement configuration"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load improvement config: {e}")

def check_rl_training_status(rl_results_dir: str) -> dict:
    """Check status of ongoing RL training"""
    results_path = Path(rl_results_dir)
    
    if not results_path.exists():
        return {'status': 'no_training_found'}
    
    # Find active training directories
    training_dirs = list(results_path.glob("pizza_rl_training_*"))
    
    if not training_dirs:
        return {'status': 'no_training_directories'}
    
    # Get the most recent training
    latest_training = sorted(training_dirs, key=lambda x: x.name)[-1]
    
    # Check if training log exists and is recent
    log_file = latest_training / "logs" / "training.log"
    
    if log_file.exists():
        import os
        log_modified = datetime.fromtimestamp(os.path.getmtime(log_file))
        time_since_update = datetime.now() - log_modified
        
        return {
            'status': 'active' if time_since_update.total_seconds() < 600 else 'possibly_complete',
            'training_directory': str(latest_training),
            'last_update': log_modified.isoformat(),
            'time_since_update_minutes': time_since_update.total_seconds() / 60
        }
    
    return {'status': 'training_found_but_no_logs'}

def initialize_aufgabe_4_2(config_path: str, base_models_dir: str, rl_results_dir: str, device: str):
    """Initialize Aufgabe 4.2 continuous improvement system"""
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("INITIALIZING AUFGABE 4.2: Continuous Pizza Verifier Improvement")
    logger.info("="*60)
    
    # Load configuration
    logger.info("Loading continuous improvement configuration...")
    config = load_improvement_config(config_path)
    logger.info(f"✓ Configuration loaded from {config_path}")
    
    # Check RL training status
    logger.info("Checking RL training status (Aufgabe 4.1)...")
    rl_status = check_rl_training_status(rl_results_dir)
    logger.info(f"RL Training Status: {rl_status['status']}")
    
    if rl_status['status'] == 'active':
        logger.info(f"✓ RL training is active - last update: {rl_status.get('last_update', 'unknown')}")
        logger.info("Continuous improvement will integrate with ongoing training")
    elif rl_status['status'] == 'possibly_complete':
        logger.info("⚠ RL training may be complete - will check for final results")
    else:
        logger.warning(f"⚠ RL training status unclear: {rl_status}")
    
    # Initialize continuous improvement system
    logger.info("Initializing continuous improvement system...")
    
    improvement_config = config.get('improvement_parameters', {})
    improvement_system = ContinuousPizzaVerifierImprovement(
        base_models_dir=base_models_dir,
        rl_training_results_dir=rl_results_dir,
        improvement_config=improvement_config,
        device=device
    )
    
    # Initialize system components
    logger.info("Setting up system components...")
    if improvement_system.initialize_system():
        logger.info("✓ Continuous improvement system initialized successfully")
        
        # Generate initial system report
        logger.info("Generating initial system report...")
        initial_report = improvement_system.generate_improvement_report()
        
        # Log system status
        logger.info(f"Models managed: {initial_report['models_managed']}")
        logger.info(f"RL agent available: {initial_report['rl_agent_available']}")
        
        # Save initial report
        report_file = Path("reports") / f"aufgabe_4_2_initialization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                'initialization_timestamp': datetime.now().isoformat(),
                'aufgabe_4_1_status': rl_status,
                'aufgabe_4_2_status': initial_report,
                'configuration': config
            }, f, indent=2)
        
        logger.info(f"✓ Initial report saved to {report_file}")
        
        return improvement_system, initial_report
        
    else:
        logger.error("✗ Failed to initialize continuous improvement system")
        return None, None

def create_monitoring_script():
    """Create script for integrated monitoring of both Aufgabe 4.1 and 4.2"""
    monitoring_script_content = '''#!/usr/bin/env python3
"""
Integrated Monitoring for Aufgabe 4.1 and 4.2
Real-time monitoring of RL training and continuous improvement
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

def monitor_integrated_systems():
    """Monitor both RL training and continuous improvement"""
    print("="*60)
    print("INTEGRATED MONITORING: Aufgabe 4.1 & 4.2")
    print("="*60)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] System Status Check")
            print("-" * 40)
            
            # Check RL training status
            rl_log = Path("/home/emilio/Documents/ai/pizza/results/pizza_rl_training_20250608_134938/logs/training.log")
            if rl_log.exists():
                with open(rl_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if "Iteration" in last_line:
                            print(f"RL Training (4.1): {last_line.split('INFO:')[1] if 'INFO:' in last_line else last_line}")
                        else:
                            print("RL Training (4.1): Status unclear")
                    else:
                        print("RL Training (4.1): No training data")
            else:
                print("RL Training (4.1): Training log not found")
            
            # Check continuous improvement status
            improvement_log = Path("improvement_workspace/logs/continuous_improvement.log")
            if improvement_log.exists():
                try:
                    with open(improvement_log, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_entry = json.loads(lines[-1])
                            timestamp = last_entry.get('timestamp', 'unknown')
                            performance = last_entry.get('overall_performance_score', 0)
                            print(f"Improvement (4.2): Last update {timestamp}, Performance: {performance:.3f}")
                        else:
                            print("Improvement (4.2): No improvement data")
                except:
                    print("Improvement (4.2): Status monitoring available")
            else:
                print("Improvement (4.2): System ready (no activity yet)")
            
            print("-" * 40)
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\\n\\nMonitoring stopped by user")

if __name__ == "__main__":
    monitor_integrated_systems()
'''
    
    script_path = Path("scripts/monitor_integrated_systems.py")
    with open(script_path, 'w') as f:
        f.write(monitoring_script_content)
    
    # Make executable
    import os
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Initialize Aufgabe 4.2: Continuous Pizza Verifier Improvement")
    parser.add_argument("--config", default="config/continuous_improvement/improvement_config.json", 
                       help="Path to improvement configuration file")
    parser.add_argument("--models-dir", default="models", help="Base models directory")
    parser.add_argument("--rl-results", default="results", help="RL training results directory")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--start-monitoring", action="store_true", help="Start continuous monitoring")
    
    args = parser.parse_args()
    
    # Initialize Aufgabe 4.2
    improvement_system, initial_report = initialize_aufgabe_4_2(
        config_path=args.config,
        base_models_dir=args.models_dir,
        rl_results_dir=args.rl_results,
        device=args.device
    )
    
    if improvement_system is None:
        print("✗ Failed to initialize Aufgabe 4.2")
        sys.exit(1)
    
    # Create integrated monitoring script
    monitoring_script = create_monitoring_script()
    print(f"✓ Integrated monitoring script created: {monitoring_script}")
    
    if args.start_monitoring:
        print("\\nStarting continuous monitoring...")
        improvement_system.start_continuous_monitoring()
        
        try:
            print("Continuous improvement system is now active!")
            print("Use the integrated monitoring script to track both systems:")
            print(f"  python {monitoring_script}")
            print("\\nPress Ctrl+C to stop...")
            
            while True:
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\\nStopping continuous improvement system...")
            improvement_system.stop_continuous_monitoring()
            print("✓ System stopped")
    else:
        print("\\nAufgabe 4.2 initialized successfully!")
        print("To start continuous monitoring, use:")
        print(f"  python {__file__} --start-monitoring")
        print("\\nTo monitor both systems, use:")
        print(f"  python {monitoring_script}")

if __name__ == "__main__":
    main()
