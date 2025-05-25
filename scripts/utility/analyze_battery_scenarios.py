#!/usr/bin/env python3
"""
ENERGIE-3.2: Battery Life Scenario Analysis and Reporting
=========================================================

Generate comprehensive analysis and visualization of battery life simulations
across different usage scenarios and battery types.

Author: AI Assistant  
Date: 2024-12-19
Version: 1.0.0
"""

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Project paths
project_root = Path(__file__).parent.parent
output_dir = project_root / "output" / "energy_analysis"

def load_simulation_data(json_file: Path) -> dict:
    """Load simulation data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_results_table(data: dict) -> pd.DataFrame:
    """Create a structured DataFrame from simulation results."""
    rows = []
    
    for combo_key, result in data["detailed_results"].items():
        battery_key, scenario_key = combo_key.split('_', 1)
        
        row = {
            'Battery_Type': result['battery_type'],
            'Scenario': result['scenario'],
            'Runtime_Days': result['total_runtime_days'],
            'Runtime_Hours': result['total_runtime_hours'],
            'Average_Current_mA': result['average_current_ma'],
            'Temperature_C': result['temperature_c'],
            'Energy_per_Detection_Wh': result['energy_efficiency']['wh_per_detection'],
            'Detections_per_Wh': result['energy_efficiency']['detections_per_wh']
        }
        
        # Add power breakdown
        for state, power in result['power_breakdown'].items():
            row[f'Power_{state.title()}_mA'] = power
            
        rows.append(row)
    
    return pd.DataFrame(rows)

def create_comparison_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table showing runtime by battery and scenario."""
    return df.pivot(index='Battery_Type', columns='Scenario', values='Runtime_Days')

def generate_csv_reports(df: pd.DataFrame, output_dir: Path):
    """Generate CSV reports for easy analysis."""
    # Full results
    full_csv = output_dir / "battery_life_full_results.csv"
    df.to_csv(full_csv, index=False)
    print(f"📊 Full results saved to: {full_csv}")
    
    # Summary matrix
    matrix = create_comparison_matrix(df)
    matrix_csv = output_dir / "battery_life_matrix.csv"
    matrix.to_csv(matrix_csv)
    print(f"📊 Runtime matrix saved to: {matrix_csv}")
    
    # Best combinations
    best_combinations = df.nlargest(10, 'Runtime_Days')[
        ['Battery_Type', 'Scenario', 'Runtime_Days', 'Average_Current_mA']
    ]
    best_csv = output_dir / "best_battery_combinations.csv"
    best_combinations.to_csv(best_csv, index=False)
    print(f"📊 Best combinations saved to: {best_csv}")

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""
    plt.style.use('default')
    
    # 1. Runtime comparison matrix heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Heatmap
    matrix = create_comparison_matrix(df)
    im = ax1.imshow(matrix.values, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(matrix.columns)))
    ax1.set_yticks(range(len(matrix.index)))
    ax1.set_xticklabels(matrix.columns, rotation=45, ha='right')
    ax1.set_yticklabels(matrix.index)
    ax1.set_title('Battery Life (Days) by Combination')
    
    # Add value annotations
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if not pd.isna(value):
                ax1.text(j, i, f'{value:.1f}', ha='center', va='center', 
                        fontweight='bold', fontsize=9)
    
    plt.colorbar(im, ax=ax1, label='Runtime (Days)')
    
    # 2. Bar chart by scenario
    scenarios = df['Scenario'].unique()
    x = np.arange(len(scenarios))
    width = 0.2
    
    batteries = df['Battery_Type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(batteries)))
    
    for i, battery in enumerate(batteries):
        battery_data = df[df['Battery_Type'] == battery]
        runtimes = [battery_data[battery_data['Scenario'] == scenario]['Runtime_Days'].iloc[0] 
                   for scenario in scenarios]
        ax2.bar(x + i * width, runtimes, width, label=battery, color=colors[i])
    
    ax2.set_xlabel('Usage Scenarios')
    ax2.set_ylabel('Runtime (Days)')
    ax2.set_title('Battery Life by Scenario and Battery Type')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy efficiency scatter plot
    ax3.scatter(df['Average_Current_mA'], df['Runtime_Days'], 
               c=df['Energy_per_Detection_Wh'], cmap='viridis', s=100, alpha=0.7)
    ax3.set_xlabel('Average Current (mA)')
    ax3.set_ylabel('Runtime (Days)')
    ax3.set_title('Runtime vs Current Consumption')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for energy per detection
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Energy per Detection (Wh)')
    
    # 4. Power breakdown pie chart for best scenario
    best_combo = df.loc[df['Runtime_Days'].idxmax()]
    power_cols = [col for col in df.columns if col.startswith('Power_') and col.endswith('_mA')]
    power_data = [best_combo[col] for col in power_cols]
    power_labels = [col.replace('Power_', '').replace('_mA', '').replace('_', ' ') 
                   for col in power_cols]
    
    # Filter out zero or very small values
    filtered_data = []
    filtered_labels = []
    for data, label in zip(power_data, power_labels):
        if data > 0.01:  # Only include values > 0.01 mA
            filtered_data.append(data)
            filtered_labels.append(label)
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(filtered_data)))
    ax4.pie(filtered_data, labels=filtered_labels, autopct='%1.1f%%', 
           colors=colors_pie, startangle=90)
    ax4.set_title(f'Power Breakdown\n{best_combo["Battery_Type"]} + {best_combo["Scenario"]}')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = output_dir / "battery_life_analysis_charts.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {viz_file}")
    plt.show()

def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    stats = {
        'total_combinations': len(df),
        'scenarios_analyzed': len(df['Scenario'].unique()),
        'battery_types_analyzed': len(df['Battery_Type'].unique()),
        'best_combination': {
            'combination': f"{df.loc[df['Runtime_Days'].idxmax(), 'Battery_Type']} + {df.loc[df['Runtime_Days'].idxmax(), 'Scenario']}",
            'runtime_days': df['Runtime_Days'].max(),
            'current_ma': df.loc[df['Runtime_Days'].idxmax(), 'Average_Current_mA']
        },
        'worst_combination': {
            'combination': f"{df.loc[df['Runtime_Days'].idxmin(), 'Battery_Type']} + {df.loc[df['Runtime_Days'].idxmin(), 'Scenario']}",
            'runtime_days': df['Runtime_Days'].min(),
            'current_ma': df.loc[df['Runtime_Days'].idxmin(), 'Average_Current_mA']
        },
        'runtime_statistics': {
            'mean_days': df['Runtime_Days'].mean(),
            'median_days': df['Runtime_Days'].median(),
            'std_days': df['Runtime_Days'].std(),
            'min_days': df['Runtime_Days'].min(),
            'max_days': df['Runtime_Days'].max()
        },
        'current_statistics': {
            'mean_ma': df['Average_Current_mA'].mean(),
            'median_ma': df['Average_Current_mA'].median(),
            'std_ma': df['Average_Current_mA'].std(),
            'min_ma': df['Average_Current_mA'].min(),
            'max_ma': df['Average_Current_mA'].max()
        }
    }
    return stats

def main():
    """Main analysis function."""
    print("🔋 ENERGIE-3.2: Battery Life Scenario Analysis")
    print("=" * 60)
    
    # Load simulation data
    sim_file = output_dir / "battery_life_simulations.json"
    if not sim_file.exists():
        print(f"❌ Error: Simulation file not found: {sim_file}")
        return
    
    print(f"📊 Loading simulation data from: {sim_file}")
    data = load_simulation_data(sim_file)
    
    # Create DataFrame
    df = create_results_table(data)
    print(f"📋 Analyzed {len(df)} battery/scenario combinations")
    
    # Generate summary statistics
    stats = generate_summary_stats(df)
    
    # Save summary statistics
    stats_file = output_dir / "battery_analysis_summary.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Summary statistics saved to: {stats_file}")
    
    # Generate CSV reports
    generate_csv_reports(df, output_dir)
    
    # Create visualizations
    try:
        create_visualizations(df, output_dir)
    except Exception as e:
        print(f"⚠️  Visualization failed (running headless?): {e}")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("📈 KEY FINDINGS")
    print("=" * 60)
    
    print(f"🏆 Best Overall: {stats['best_combination']['combination']}")
    print(f"   Runtime: {stats['best_combination']['runtime_days']:.1f} days")
    print(f"   Current: {stats['best_combination']['current_ma']:.2f} mA")
    
    print(f"\n📉 Shortest Runtime: {stats['worst_combination']['combination']}")
    print(f"   Runtime: {stats['worst_combination']['runtime_days']:.1f} days")
    print(f"   Current: {stats['worst_combination']['current_ma']:.2f} mA")
    
    print(f"\n📊 Runtime Statistics:")
    print(f"   Average: {stats['runtime_statistics']['mean_days']:.1f} days")
    print(f"   Median: {stats['runtime_statistics']['median_days']:.1f} days")
    print(f"   Range: {stats['runtime_statistics']['min_days']:.1f} - {stats['runtime_statistics']['max_days']:.1f} days")
    
    print(f"\n⚡ Current Statistics:")
    print(f"   Average: {stats['current_statistics']['mean_ma']:.2f} mA")
    print(f"   Range: {stats['current_statistics']['min_ma']:.2f} - {stats['current_statistics']['max_ma']:.2f} mA")
    
    print(f"\n✅ ENERGIE-3.2 Analysis Complete!")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📊 Generated {len(list(output_dir.glob('*.csv')))} CSV files")
    print(f"   📈 Generated {len(list(output_dir.glob('*.png')))} visualization files")

if __name__ == "__main__":
    main()
