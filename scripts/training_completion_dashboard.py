#!/usr/bin/env python3
"""
Real-time Pizza RL Training Completion Dashboard
Monitors training progress and prepares for Aufgabe 4.2 activation
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def parse_training_log():
    """Parse training log to extract progress data"""
    log_file = Path("/home/emilio/Documents/ai/pizza/results/pizza_rl_training_20250608_134938/logs/training.log")
    
    if not log_file.exists():
        return None
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            if "Iteration" in line and "Steps:" in line:
                try:
                    parts = line.strip().split()
                    iteration = None
                    steps = None
                    reward = None
                    accuracy = None
                    energy_eff = None
                    time_taken = None
                    
                    for i, part in enumerate(parts):
                        if "Iteration" in part and i < len(parts) - 1:
                            iteration_info = parts[i+1].split("/")
                            iteration = int(iteration_info[0])
                        elif "Steps:" in part and i < len(parts) - 1:
                            steps_info = parts[i+1].split("/")
                            steps = int(steps_info[0])
                        elif "Reward:" in part and i < len(parts) - 1:
                            reward = float(parts[i+1])
                        elif "Accuracy:" in part and i < len(parts) - 1:
                            accuracy = float(parts[i+1])
                        elif "Energy" in part and "Eff:" in parts[i+1] and i < len(parts) - 2:
                            energy_eff = float(parts[i+2])
                        elif "Time:" in part and i < len(parts) - 1:
                            time_taken = float(parts[i+1].rstrip('s'))
                    
                    if all(x is not None for x in [iteration, steps, reward, accuracy, energy_eff]):
                        data.append({
                            'iteration': iteration,
                            'steps': steps,
                            'reward': reward,
                            'accuracy': accuracy,
                            'energy_efficiency': energy_eff,
                            'time_per_iteration': time_taken or 0
                        })
                except Exception as e:
                    continue
    
    return pd.DataFrame(data) if data else None

def estimate_completion_time(df):
    """Estimate training completion time"""
    if df is None or len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    current_iteration = latest['iteration']
    total_iterations = 122
    remaining_iterations = total_iterations - current_iteration
    
    # Calculate average time per iteration from recent data (last 10 iterations)
    recent_data = df.tail(10)
    avg_time_per_iteration = recent_data['time_per_iteration'].mean()
    
    estimated_remaining_seconds = remaining_iterations * avg_time_per_iteration
    estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
    
    return {
        'remaining_iterations': remaining_iterations,
        'avg_time_per_iteration': avg_time_per_iteration,
        'estimated_remaining_hours': estimated_remaining_seconds / 3600,
        'estimated_completion_time': estimated_completion.strftime('%Y-%m-%d %H:%M:%S')
    }

def check_aufgabe_4_2_readiness():
    """Check if Aufgabe 4.2 is ready for activation"""
    improvement_workspace = Path("/home/emilio/Documents/ai/pizza/improvement_workspace")
    config_file = Path("/home/emilio/Documents/ai/pizza/config/continuous_improvement/improvement_config.json")
    
    return {
        'workspace_ready': improvement_workspace.exists(),
        'config_ready': config_file.exists(),
        'system_initialized': len(list(Path("/home/emilio/Documents/ai/pizza/reports").glob("aufgabe_4_2_initialization_*.json"))) > 0
    }

def generate_completion_dashboard():
    """Generate completion dashboard with progress visualization"""
    print("=" * 80)
    print("🍕 PIZZA RL TRAINING COMPLETION DASHBOARD")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Parse training data
    df = parse_training_log()
    if df is None:
        print("❌ No training data found")
        return
    
    latest = df.iloc[-1]
    print(f"\n📊 CURRENT TRAINING STATUS")
    print(f"🔄 Iteration: {latest['iteration']}/122 ({latest['iteration']/122*100:.1f}%)")
    print(f"👣 Steps: {latest['steps']:,}/500,000 ({latest['steps']/500000*100:.1f}%)")
    print(f"🏆 Reward: {latest['reward']:.3f}")
    print(f"🎯 Accuracy: {latest['accuracy']:.3f} ({latest['accuracy']*100:.1f}%)")
    print(f"⚡ Energy Efficiency: {latest['energy_efficiency']:.3f}")
    
    # Performance trends (last 5 iterations)
    if len(df) >= 5:
        recent = df.tail(5)
        reward_trend = (recent['reward'].iloc[-1] - recent['reward'].iloc[0]) / recent['reward'].iloc[0] * 100
        accuracy_trend = (recent['accuracy'].iloc[-1] - recent['accuracy'].iloc[0]) / recent['accuracy'].iloc[0] * 100
        energy_trend = (recent['energy_efficiency'].iloc[-1] - recent['energy_efficiency'].iloc[0]) / recent['energy_efficiency'].iloc[0] * 100
        
        print(f"\n📈 PERFORMANCE TRENDS (Last 5 iterations)")
        print(f"🏆 Reward: {reward_trend:+.1f}%")
        print(f"🎯 Accuracy: {accuracy_trend:+.1f}%")
        print(f"⚡ Energy Efficiency: {energy_trend:+.1f}%")
    
    # Completion estimate
    completion_info = estimate_completion_time(df)
    if completion_info:
        print(f"\n⏱️  COMPLETION ESTIMATE")
        print(f"🔄 Remaining iterations: {completion_info['remaining_iterations']}")
        print(f"⏰ Avg time per iteration: {completion_info['avg_time_per_iteration']:.1f}s")
        print(f"⏳ Estimated remaining: {completion_info['estimated_remaining_hours']:.1f} hours")
        print(f"🎯 Expected completion: {completion_info['estimated_completion_time']}")
    
    # Aufgabe 4.2 readiness
    readiness = check_aufgabe_4_2_readiness()
    print(f"\n🔄 AUFGABE 4.2 READINESS")
    print(f"📁 Workspace ready: {'✅' if readiness['workspace_ready'] else '❌'}")
    print(f"⚙️  Config ready: {'✅' if readiness['config_ready'] else '❌'}")
    print(f"🚀 System initialized: {'✅' if readiness['system_initialized'] else '❌'}")
    
    # Overall status
    progress_percent = latest['iteration'] / 122 * 100
    if progress_percent < 50:
        status_emoji = "🌱"
        status_text = "Early Training"
    elif progress_percent < 80:
        status_emoji = "🚀"
        status_text = "Active Training"
    else:
        status_emoji = "🏁"
        status_text = "Near Completion"
    
    print(f"\n{status_emoji} OVERALL STATUS: {status_text} ({progress_percent:.1f}% complete)")
    
    if progress_percent > 95:
        print("\n🎉 TRAINING NEARLY COMPLETE! Prepare for Aufgabe 4.2 activation!")
    
    print("=" * 80)

def main():
    """Main dashboard function"""
    try:
        generate_completion_dashboard()
        
        # Save current status
        df = parse_training_log()
        if df is not None:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'current_iteration': int(df.iloc[-1]['iteration']),
                'progress_percentage': float(df.iloc[-1]['iteration'] / 122 * 100),
                'current_performance': {
                    'reward': float(df.iloc[-1]['reward']),
                    'accuracy': float(df.iloc[-1]['accuracy']),
                    'energy_efficiency': float(df.iloc[-1]['energy_efficiency'])
                },
                'completion_estimate': estimate_completion_time(df),
                'aufgabe_4_2_ready': check_aufgabe_4_2_readiness()
            }
            
            status_file = f"/home/emilio/Documents/ai/pizza/reports/training_completion_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
            
            print(f"\n💾 Status saved to: {status_file}")
        
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
