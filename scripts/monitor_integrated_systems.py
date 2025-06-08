#!/usr/bin/env python3
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
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] System Status Check")
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
        print("\n\nMonitoring stopped by user")

if __name__ == "__main__":
    monitor_integrated_systems()
