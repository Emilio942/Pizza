#!/usr/bin/env python3
"""
Monitor the progress of DIFFUSION-4.2 A/B testing.
"""

import os
import time
import json
from pathlib import Path

def monitor_ab_testing():
    """Monitor the A/B testing progress."""
    base_dir = Path("/home/emilio/Documents/ai/pizza")
    output_dir = base_dir / "output" / "diffusion_evaluation"
    
    print("=== DIFFUSION-4.2 A/B Testing Progress Monitor ===")
    print(f"Monitoring directory: {output_dir}")
    print()
    
    while True:
        # Check filtering status
        filtering_report = output_dir / "synthetic_filtering_report.json"
        if filtering_report.exists():
            with open(filtering_report) as f:
                filter_data = json.load(f)
            print(f"✓ Filtering completed: {filter_data['kept_count']}/{filter_data['original_count']} images kept")
        else:
            print("• Filtering in progress...")
        
        # Check training experiments
        experiments = ["real_only", "mixed_ratio_0.3", "mixed_ratio_0.5"]
        
        for exp in experiments:
            exp_dir = output_dir / exp
            if exp_dir.exists():
                # Check for results
                results_file = exp_dir / "results.json"
                history_file = exp_dir / "history.json"
                
                if results_file.exists():
                    with open(results_file) as f:
                        results = json.load(f)
                    acc = results.get('best_val_accuracy', 0)
                    print(f"✓ {exp}: Completed (Best Val Acc: {acc:.4f})")
                elif history_file.exists():
                    with open(history_file) as f:
                        history = json.load(f)
                    epochs_done = len(history)
                    latest_acc = history[-1]['val_acc'] if history else 0
                    print(f"• {exp}: Training... (Epoch {epochs_done}/15, Latest Val Acc: {latest_acc:.4f})")
                else:
                    print(f"• {exp}: Starting...")
            else:
                print(f"- {exp}: Pending")
        
        # Check final results
        final_results = output_dir / "synthetic_data_impact.json"
        if final_results.exists():
            with open(final_results) as f:
                results = json.load(f)
            impact = results.get("impact_analysis", {})
            improvement = impact.get("relative_improvement_percent", 0)
            print(f"\n✓ COMPLETED! Synthetic data improvement: {improvement:.2f}%")
            break
        
        print("-" * 60)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_ab_testing()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
