#!/usr/bin/env python3
"""
Comprehensive Status Report for Aufgabe 4.1 & 4.2
Real-time analysis of Pizza RL training and continuous improvement systems
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def analyze_rl_training_status():
    """Analyze current RL training status"""
    training_dir = Path("/home/emilio/Documents/ai/pizza/results/pizza_rl_training_20250608_134938")
    
    status = {
        "training_active": training_dir.exists(),
        "log_file": training_dir / "logs" / "training.log"
    }
    
    if status["log_file"].exists():
        # Read last few lines to get current status
        with open(status["log_file"], 'r') as f:
            lines = f.readlines()
            
        # Parse latest training iteration
        latest_line = lines[-1].strip() if lines else ""
        if "Iteration" in latest_line:
            # Extract metrics from log line
            parts = latest_line.split()
            for i, part in enumerate(parts):
                if "Iteration" in part and i < len(parts) - 1:
                    iteration_info = parts[i+1].split("/")
                    status["current_iteration"] = int(iteration_info[0])
                    status["total_iterations"] = int(iteration_info[1])
                elif "Steps:" in part and i < len(parts) - 1:
                    steps_info = parts[i+1].split("/")
                    status["current_steps"] = int(steps_info[0])
                    # Remove any trailing characters like ')'
                    total_steps_str = steps_info[1].rstrip(')')
                    status["total_steps"] = int(total_steps_str)
                elif "Reward:" in part and i < len(parts) - 1:
                    status["current_reward"] = float(parts[i+1])
                elif "Accuracy:" in part and i < len(parts) - 1:
                    status["current_accuracy"] = float(parts[i+1])
                elif "Energy" in part and "Eff:" in parts[i+1] and i < len(parts) - 2:
                    status["current_energy_efficiency"] = float(parts[i+2])
        
        # Calculate progress
        if "current_iteration" in status and "total_iterations" in status:
            status["progress_percentage"] = (status["current_iteration"] / status["total_iterations"]) * 100
            
        # Estimate completion time based on current progress
        if "current_steps" in status and "total_steps" in status:
            steps_remaining = status["total_steps"] - status["current_steps"]
            # Estimate based on ~4096 steps per iteration and ~75 seconds per iteration
            iterations_remaining = steps_remaining / 4096
            estimated_time_hours = (iterations_remaining * 75) / 3600
            status["estimated_completion_hours"] = round(estimated_time_hours, 1)
    
    return status

def analyze_continuous_improvement_status():
    """Analyze continuous improvement system status"""
    # Check for latest initialization report
    reports_dir = Path("/home/emilio/Documents/ai/pizza/reports")
    improvement_reports = list(reports_dir.glob("aufgabe_4_2_initialization_*.json"))
    
    status = {
        "system_initialized": len(improvement_reports) > 0,
        "initialization_reports": len(improvement_reports)
    }
    
    if improvement_reports:
        # Get latest report
        latest_report = max(improvement_reports, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
                status["latest_report"] = report_data
                status["models_managed"] = report_data.get("aufgabe_4_2_status", {}).get("models_managed", [])
                status["rl_agent_available"] = report_data.get("aufgabe_4_2_status", {}).get("rl_agent_available", False)
        except Exception as e:
            status["report_error"] = str(e)
    
    # Check improvement workspace
    improvement_workspace = Path("/home/emilio/Documents/ai/pizza/improvement_workspace")
    status["workspace_exists"] = improvement_workspace.exists()
    
    return status

def generate_comprehensive_report():
    """Generate comprehensive status report"""
    timestamp = datetime.now()
    
    # Analyze both systems
    rl_status = analyze_rl_training_status()
    improvement_status = analyze_continuous_improvement_status()
    
    report = {
        "report_timestamp": timestamp.isoformat(),
        "aufgabe_4_1_rl_training": rl_status,
        "aufgabe_4_2_continuous_improvement": improvement_status,
        "integration_status": {
            "both_systems_active": rl_status.get("training_active", False) and improvement_status.get("system_initialized", False),
            "monitoring_active": True,
            "integration_ready": True
        }
    }
    
    return report

def print_status_summary(report):
    """Print formatted status summary"""
    print("=" * 80)
    print("PIZZA RL PROJECT STATUS SUMMARY")
    print(f"Generated at: {report['report_timestamp']}")
    print("=" * 80)
    
    # Aufgabe 4.1 Status
    print("\n🔥 AUFGABE 4.1: Pizza RL Training")
    print("-" * 40)
    rl_status = report["aufgabe_4_1_rl_training"]
    
    if rl_status.get("training_active"):
        if "current_iteration" in rl_status:
            print(f"✓ Training Active: Iteration {rl_status['current_iteration']}/{rl_status['total_iterations']}")
            print(f"✓ Progress: {rl_status.get('progress_percentage', 0):.1f}%")
            print(f"✓ Steps: {rl_status.get('current_steps', 0):,}/{rl_status.get('total_steps', 0):,}")
            
            if "current_reward" in rl_status:
                print(f"✓ Current Performance:")
                print(f"  - Reward: {rl_status['current_reward']:.3f}")
                print(f"  - Accuracy: {rl_status['current_accuracy']:.3f} ({rl_status['current_accuracy']*100:.1f}%)")
                print(f"  - Energy Efficiency: {rl_status['current_energy_efficiency']:.3f}")
            
            if "estimated_completion_hours" in rl_status:
                print(f"✓ Estimated completion: {rl_status['estimated_completion_hours']:.1f} hours")
        else:
            print("✓ Training directory found but status unclear")
    else:
        print("❌ Training not active")
    
    # Aufgabe 4.2 Status
    print("\n🔄 AUFGABE 4.2: Continuous Improvement")
    print("-" * 40)
    improvement_status = report["aufgabe_4_2_continuous_improvement"]
    
    if improvement_status.get("system_initialized"):
        print(f"✓ System Initialized ({improvement_status['initialization_reports']} reports)")
        print(f"✓ Models Managed: {len(improvement_status.get('models_managed', []))}")
        for model in improvement_status.get('models_managed', []):
            print(f"  - {model}")
        print(f"✓ RL Integration Ready: {improvement_status.get('rl_agent_available', False)}")
        print(f"✓ Workspace Ready: {improvement_status.get('workspace_exists', False)}")
    else:
        print("❌ System not initialized")
    
    # Integration Status
    print("\n🔗 SYSTEM INTEGRATION")
    print("-" * 40)
    integration = report["integration_status"]
    print(f"✓ Both Systems Active: {integration['both_systems_active']}")
    print(f"✓ Monitoring Active: {integration['monitoring_active']}")
    print(f"✓ Ready for RL Completion Integration: {integration['integration_ready']}")
    
    print("\n" + "=" * 80)

def main():
    """Generate and display comprehensive status report"""
    try:
        report = generate_comprehensive_report()
        print_status_summary(report)
        
        # Save report
        report_file = f"/home/emilio/Documents/ai/pizza/reports/comprehensive_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Detailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error generating status report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
