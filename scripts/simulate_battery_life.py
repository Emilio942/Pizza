#!/usr/bin/env python3
"""
ENERGIE-3.1: Advanced Battery Life Simulation Model
==================================================

Comprehensive battery lifetime prediction system for RP2040-based pizza detection device.
Considers measured/simulated power consumption across different operating modes,
battery specifications, discharge curves, and realistic usage profiles.

Author: AI Assistant
Date: 2024-12-19
Version: 1.0.0
"""

import sys
import os
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import existing power management components
try:
    from src.utils.power_manager import PowerUsage, PowerMode as PMPowerMode
    from src.utils.devices import BatteryStatus, PowerMode, DeviceStatus
    from src.utils.constants import RP2040_CLOCK_SPEED_MHZ
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    print("Running in standalone mode with default values.")

# Simulation Configuration
OUTPUT_DIR = Path(project_root) / "output" / "battery_simulations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class UsageProfile(Enum):
    """Predefined usage scenarios for the pizza detection system."""
    CONTINUOUS_MONITORING = "continuous"
    FREQUENT_DETECTION = "frequent"  # 1 detection every 5 minutes
    MODERATE_DETECTION = "moderate"  # 1 detection every 15 minutes
    RARE_DETECTION = "rare"         # 1 detection every hour
    BATTERY_SAVER = "battery_saver"  # 1 detection every 6 hours


@dataclass
class BatteryType:
    """Battery specification with discharge characteristics."""
    name: str
    capacity_mah: float
    nominal_voltage_v: float
    cutoff_voltage_v: float
    weight_g: float
    chemistry: str
    discharge_curve: Optional[Dict[float, float]] = None  # voltage_percent -> capacity_factor
    
    def get_effective_capacity(self, voltage_percent: float = 1.0) -> float:
        """Calculate effective capacity considering discharge curve."""
        if self.discharge_curve and voltage_percent in self.discharge_curve:
            return self.capacity_mah * self.discharge_curve[voltage_percent]
        return self.capacity_mah * voltage_percent


@dataclass
class PowerState:
    """Detailed power consumption state definition."""
    name: str
    current_ma: float
    duration_ms: int = 0
    frequency_per_hour: float = 0.0
    description: str = ""


@dataclass
class UsageScenario:
    """Complete usage profile with timing and frequencies."""
    name: str
    description: str
    sleep_time_percent: float
    idle_time_percent: float
    active_time_percent: float
    camera_time_percent: float
    inference_time_percent: float
    transitions_per_hour: int = 10
    average_detection_time_ms: int = 2000
    inference_cycles_per_hour: int = 12


class BatteryLifeSimulator:
    """Advanced battery life simulation with realistic power modeling."""
    
    # Standard battery configurations
    BATTERY_TYPES = {
        "CR123A": BatteryType(
            name="CR123A Lithium",
            capacity_mah=1500,
            nominal_voltage_v=3.0,
            cutoff_voltage_v=2.0,
            weight_g=17,
            chemistry="Lithium",
            discharge_curve={
                1.0: 1.0, 0.9: 0.98, 0.8: 0.95, 0.7: 0.92,
                0.6: 0.88, 0.5: 0.82, 0.4: 0.75, 0.3: 0.65,
                0.2: 0.5, 0.1: 0.3, 0.0: 0.0
            }
        ),
        "18650": BatteryType(
            name="18650 Li-Ion",
            capacity_mah=3400,
            nominal_voltage_v=3.7,
            cutoff_voltage_v=2.5,
            weight_g=47,
            chemistry="Li-Ion",
            discharge_curve={
                1.0: 1.0, 0.9: 0.99, 0.8: 0.97, 0.7: 0.95,
                0.6: 0.92, 0.5: 0.88, 0.4: 0.83, 0.3: 0.76,
                0.2: 0.65, 0.1: 0.45, 0.0: 0.0
            }
        ),
        "AA_ALKALINE": BatteryType(
            name="AA Alkaline",
            capacity_mah=2500,
            nominal_voltage_v=1.5,
            cutoff_voltage_v=0.9,
            weight_g=23,
            chemistry="Alkaline",
            discharge_curve={
                1.0: 1.0, 0.9: 0.95, 0.8: 0.88, 0.7: 0.82,
                0.6: 0.75, 0.5: 0.68, 0.4: 0.58, 0.3: 0.45,
                0.2: 0.28, 0.1: 0.12, 0.0: 0.0
            }
        ),
        "LIPO_500": BatteryType(
            name="LiPo 500mAh",
            capacity_mah=500,
            nominal_voltage_v=3.7,
            cutoff_voltage_v=3.0,
            weight_g=10,
            chemistry="Li-Polymer",
            discharge_curve={
                1.0: 1.0, 0.9: 0.98, 0.8: 0.96, 0.7: 0.93,
                0.6: 0.89, 0.5: 0.84, 0.4: 0.78, 0.3: 0.70,
                0.2: 0.58, 0.1: 0.40, 0.0: 0.0
            }
        )
    }
    
    def __init__(self):
        """Initialize simulator with power consumption profiles."""
        # Use existing PowerUsage if available, otherwise define defaults
        try:
            self.power_usage = PowerUsage()
        except NameError:
            # Fallback power usage values
            self.power_usage = type('PowerUsage', (), {
                'sleep_mode_ma': 0.5,
                'idle_ma': 10.0,
                'active_ma': 80.0,
                'camera_active_ma': 40.0,
                'inference_ma': 100.0
            })()
        
        # Define power states with transition considerations
        self.power_states = {
            "sleep": PowerState("Sleep Mode", self.power_usage.sleep_mode_ma, 
                              description="Deep sleep with RTC running"),
            "idle": PowerState("Idle", self.power_usage.idle_ma,
                             description="CPU idle, peripherals off"),
            "active": PowerState("Active CPU", self.power_usage.active_ma,
                               description="CPU active, processing"),
            "camera": PowerState("Camera Active", self.power_usage.camera_active_ma,
                               description="Camera capturing image"),
            "inference": PowerState("AI Inference", self.power_usage.inference_ma,
                                  description="Running neural network inference"),
            "transition": PowerState("State Transition", 50.0,
                                   description="Power during state changes")
        }
        
        # Define usage scenarios
        self.usage_scenarios = self._create_usage_scenarios()
    
    def _create_usage_scenarios(self) -> Dict[str, UsageScenario]:
        """Create predefined usage scenarios."""
        return {
            UsageProfile.CONTINUOUS_MONITORING.value: UsageScenario(
                name="Continuous Monitoring",
                description="Always-on detection with minimal sleep",
                sleep_time_percent=10.0,
                idle_time_percent=20.0,
                active_time_percent=30.0,
                camera_time_percent=20.0,
                inference_time_percent=20.0,
                transitions_per_hour=3600,  # Frequent transitions
                average_detection_time_ms=3000,
                inference_cycles_per_hour=1200
            ),
            UsageProfile.FREQUENT_DETECTION.value: UsageScenario(
                name="Frequent Detection",
                description="Detection every 5 minutes",
                sleep_time_percent=85.0,
                idle_time_percent=5.0,
                active_time_percent=5.0,
                camera_time_percent=3.0,
                inference_time_percent=2.0,
                transitions_per_hour=24,  # 12 detection cycles * 2 transitions each
                average_detection_time_ms=2000,
                inference_cycles_per_hour=12
            ),
            UsageProfile.MODERATE_DETECTION.value: UsageScenario(
                name="Moderate Detection",
                description="Detection every 15 minutes",
                sleep_time_percent=92.0,
                idle_time_percent=3.0,
                active_time_percent=3.0,
                camera_time_percent=1.5,
                inference_time_percent=0.5,
                transitions_per_hour=8,  # 4 detection cycles * 2 transitions each
                average_detection_time_ms=2000,
                inference_cycles_per_hour=4
            ),
            UsageProfile.RARE_DETECTION.value: UsageScenario(
                name="Rare Detection",
                description="Detection once per hour",
                sleep_time_percent=98.0,
                idle_time_percent=1.0,
                active_time_percent=0.7,
                camera_time_percent=0.2,
                inference_time_percent=0.1,
                transitions_per_hour=2,  # 1 detection cycle * 2 transitions
                average_detection_time_ms=2000,
                inference_cycles_per_hour=1
            ),
            UsageProfile.BATTERY_SAVER.value: UsageScenario(
                name="Battery Saver",
                description="Detection every 6 hours",
                sleep_time_percent=99.5,
                idle_time_percent=0.3,
                active_time_percent=0.15,
                camera_time_percent=0.04,
                inference_time_percent=0.01,
                transitions_per_hour=0.33,  # 1/6 of a detection cycle per hour
                average_detection_time_ms=2000,
                inference_cycles_per_hour=0.16
            )
        }
    
    def calculate_average_current(self, scenario: UsageScenario, 
                                temperature_c: float = 25.0) -> Tuple[float, Dict[str, float]]:
        """
        Calculate average current consumption for a usage scenario.
        
        Args:
            scenario: Usage scenario definition
            temperature_c: Operating temperature in Celsius
            
        Returns:
            Tuple of (average_current_ma, detailed_breakdown)
        """
        # Temperature scaling factor (2% increase per °C above 25°C)
        temp_factor = 1.0 + max(0, (temperature_c - 25.0) * 0.02)
        
        # Calculate base power consumption for each state
        power_breakdown = {
            "sleep": (scenario.sleep_time_percent / 100.0) * self.power_states["sleep"].current_ma,
            "idle": (scenario.idle_time_percent / 100.0) * self.power_states["idle"].current_ma,
            "active": (scenario.active_time_percent / 100.0) * self.power_states["active"].current_ma,
            "camera": (scenario.camera_time_percent / 100.0) * self.power_states["camera"].current_ma,
            "inference": (scenario.inference_time_percent / 100.0) * self.power_states["inference"].current_ma,
        }
        
        # Add transition power overhead
        transition_power_per_hour = (scenario.transitions_per_hour * 
                                   self.power_states["transition"].current_ma * 
                                   0.1 / 3600)  # 100ms per transition
        power_breakdown["transitions"] = transition_power_per_hour
        
        # Apply temperature scaling
        for key in power_breakdown:
            power_breakdown[key] *= temp_factor
        
        average_current = sum(power_breakdown.values())
        
        return average_current, power_breakdown
    
    def simulate_battery_discharge(self, battery: BatteryType, scenario: UsageScenario,
                                 temperature_c: float = 25.0, 
                                 simulation_hours: int = 8760) -> Dict:
        """
        Simulate detailed battery discharge over time.
        
        Args:
            battery: Battery type specification
            scenario: Usage scenario
            temperature_c: Operating temperature
            simulation_hours: Maximum simulation time (default: 1 year)
            
        Returns:
            Dictionary with simulation results
        """
        average_current, power_breakdown = self.calculate_average_current(scenario, temperature_c)
        
        # Time tracking
        time_hours = []
        voltage_levels = []
        capacity_remaining = []
        runtime_hours = 0.0
        
        # Simulation parameters
        time_step_hours = 1.0  # 1-hour time steps
        remaining_capacity = battery.capacity_mah
        
        while remaining_capacity > 0 and runtime_hours < simulation_hours:
            # Calculate voltage level based on remaining capacity
            capacity_percent = remaining_capacity / battery.capacity_mah
            
            # Apply discharge curve effects
            if battery.discharge_curve:
                # Find closest discharge curve point
                curve_points = sorted(battery.discharge_curve.keys(), reverse=True)
                voltage_factor = 1.0
                for point in curve_points:
                    if capacity_percent >= point:
                        voltage_factor = battery.discharge_curve[point]
                        break
                
                # Adjust current consumption based on voltage
                voltage_adjusted_current = average_current * (1.0 + (1.0 - voltage_factor) * 0.1)
            else:
                voltage_adjusted_current = average_current
            
            # Update capacity
            capacity_consumed = voltage_adjusted_current * time_step_hours
            remaining_capacity -= capacity_consumed
            
            # Record data points
            time_hours.append(runtime_hours)
            voltage_levels.append(capacity_percent * 100)
            capacity_remaining.append(max(0, remaining_capacity))
            
            runtime_hours += time_step_hours
        
        # Calculate final metrics
        total_runtime_hours = runtime_hours
        total_runtime_days = total_runtime_hours / 24.0
        
        return {
            "battery_type": battery.name,
            "scenario": scenario.name,
            "average_current_ma": average_current,
            "power_breakdown": power_breakdown,
            "total_runtime_hours": total_runtime_hours,
            "total_runtime_days": total_runtime_days,
            "temperature_c": temperature_c,
            "time_series": {
                "time_hours": time_hours,
                "voltage_percent": voltage_levels,
                "capacity_mah": capacity_remaining
            },
            "energy_efficiency": {
                "wh_per_detection": (battery.capacity_mah * battery.nominal_voltage_v / 1000) / 
                                  max(1, scenario.inference_cycles_per_hour * total_runtime_hours),
                "detections_per_wh": (scenario.inference_cycles_per_hour * total_runtime_hours) / 
                                   max(0.001, battery.capacity_mah * battery.nominal_voltage_v / 1000)
            }
        }
    
    def run_comprehensive_simulation(self, temperature_c: float = 25.0,
                                   output_file: Optional[str] = None) -> Dict:
        """
        Run comprehensive battery life simulation across all scenarios and battery types.
        
        Args:
            temperature_c: Operating temperature
            output_file: Optional JSON output file path
            
        Returns:
            Complete simulation results
        """
        print(f"🔋 Starting Comprehensive Battery Life Simulation")
        print(f"   Temperature: {temperature_c}°C")
        print(f"   Scenarios: {len(self.usage_scenarios)}")
        print(f"   Battery Types: {len(self.BATTERY_TYPES)}")
        print("=" * 60)
        
        results = {
            "simulation_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature_c": temperature_c,
                "power_states": {name: asdict(state) for name, state in self.power_states.items()},
                "scenarios_count": len(self.usage_scenarios),
                "battery_types_count": len(self.BATTERY_TYPES)
            },
            "detailed_results": {},
            "summary": {
                "best_combinations": [],
                "worst_combinations": [],
                "recommendations": []
            }
        }
        
        all_results = []
        
        # Simulate each combination
        for battery_key, battery in self.BATTERY_TYPES.items():
            for scenario_key, scenario in self.usage_scenarios.items():
                print(f"⚡ Simulating: {battery.name} + {scenario.name}")
                
                sim_result = self.simulate_battery_discharge(
                    battery, scenario, temperature_c
                )
                
                # Store detailed result
                combo_key = f"{battery_key}_{scenario_key}"
                results["detailed_results"][combo_key] = sim_result
                
                # Track for ranking
                all_results.append({
                    "combination": f"{battery.name} + {scenario.name}",
                    "runtime_days": sim_result["total_runtime_days"],
                    "runtime_hours": sim_result["total_runtime_hours"],
                    "average_current_ma": sim_result["average_current_ma"],
                    "battery_weight_g": battery.weight_g,
                    "energy_per_detection_wh": sim_result["energy_efficiency"]["wh_per_detection"],
                    "battery_key": battery_key,
                    "scenario_key": scenario_key
                })
                
                print(f"   Runtime: {sim_result['total_runtime_days']:.1f} days")
                print(f"   Average Current: {sim_result['average_current_ma']:.2f} mA")
        
        # Analyze results
        all_results.sort(key=lambda x: x["runtime_days"], reverse=True)
        
        # Best and worst combinations
        results["summary"]["best_combinations"] = all_results[:3]
        results["summary"]["worst_combinations"] = all_results[-3:]
        
        # Generate recommendations
        results["summary"]["recommendations"] = self._generate_recommendations(all_results)
        
        # Save results if requested
        if output_file:
            output_path = OUTPUT_DIR / output_file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
        
        return results
    
    def _generate_recommendations(self, all_results: List[Dict]) -> List[str]:
        """Generate practical recommendations based on simulation results."""
        recommendations = []
        
        # Find best overall combination
        best = all_results[0]
        recommendations.append(
            f"🏆 Best Overall: {best['combination']} provides {best['runtime_days']:.1f} days runtime"
        )
        
        # Find most energy-efficient combination
        most_efficient = min(all_results, key=lambda x: x["energy_per_detection_wh"])
        recommendations.append(
            f"⚡ Most Efficient: {most_efficient['combination']} uses "
            f"{most_efficient['energy_per_detection_wh']:.4f} Wh per detection"
        )
        
        # Find best weight-to-runtime ratio
        weight_efficiency = [(r["runtime_days"] / r["battery_weight_g"], r) for r in all_results]
        weight_efficiency.sort(reverse=True)
        best_weight = weight_efficiency[0][1]
        recommendations.append(
            f"🪶 Best Weight Efficiency: {best_weight['combination']} provides "
            f"{best_weight['runtime_days']/best_weight['battery_weight_g']:.2f} days per gram"
        )
        
        # Scenario-specific recommendations
        scenarios = set(r["scenario_key"] for r in all_results)
        for scenario in scenarios:
            scenario_results = [r for r in all_results if r["scenario_key"] == scenario]
            best_for_scenario = max(scenario_results, key=lambda x: x["runtime_days"])
            recommendations.append(
                f"📊 For {scenario}: {best_for_scenario['combination'].split(' + ')[0]} "
                f"provides {best_for_scenario['runtime_days']:.1f} days"
            )
        
        return recommendations
    
    def create_visualization(self, results: Dict, output_dir: Optional[Path] = None) -> None:
        """Create comprehensive visualization of simulation results."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        # Set up the plot style
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # Extract data for plotting
        battery_names = []
        scenario_names = []
        runtime_matrix = []
        
        batteries = list(self.BATTERY_TYPES.keys())
        scenarios = list(self.usage_scenarios.keys())
        
        runtime_data = np.zeros((len(batteries), len(scenarios)))
        
        for i, battery_key in enumerate(batteries):
            for j, scenario_key in enumerate(scenarios):
                combo_key = f"{battery_key}_{scenario_key}"
                if combo_key in results["detailed_results"]:
                    runtime_data[i, j] = results["detailed_results"][combo_key]["total_runtime_days"]
        
        # 1. Heatmap of runtime combinations
        ax1 = plt.subplot(2, 3, 1)
        im = ax1.imshow(runtime_data, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_yticks(range(len(batteries)))
        ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
        ax1.set_yticklabels([self.BATTERY_TYPES[b].name for b in batteries])
        ax1.set_title('Battery Life (Days)\nby Combination')
        plt.colorbar(im, ax=ax1, label='Days')
        
        # Add value annotations
        for i in range(len(batteries)):
            for j in range(len(scenarios)):
                ax1.text(j, i, f'{runtime_data[i, j]:.1f}', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 2. Power consumption breakdown for best scenario
        best_combo = results["summary"]["best_combinations"][0]
        best_key = f"{best_combo['battery_key']}_{best_combo['scenario_key']}"
        best_result = results["detailed_results"][best_key]
        
        ax2 = plt.subplot(2, 3, 2)
        power_breakdown = best_result["power_breakdown"]
        labels = list(power_breakdown.keys())
        values = list(power_breakdown.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title(f'Power Breakdown\n{best_combo["combination"]}')
        
        # 3. Runtime comparison by battery type
        ax3 = plt.subplot(2, 3, 3)
        battery_runtimes = {}
        for battery_key in batteries:
            runtimes = []
            for scenario_key in scenarios:
                combo_key = f"{battery_key}_{scenario_key}"
                if combo_key in results["detailed_results"]:
                    runtimes.append(results["detailed_results"][combo_key]["total_runtime_days"])
            battery_runtimes[self.BATTERY_TYPES[battery_key].name] = runtimes
        
        x = np.arange(len(scenarios))
        width = 0.2
        for i, (battery_name, runtimes) in enumerate(battery_runtimes.items()):
            ax3.bar(x + i * width, runtimes, width, label=battery_name)
        
        ax3.set_xlabel('Usage Scenarios')
        ax3.set_ylabel('Runtime (Days)')
        ax3.set_title('Runtime Comparison by Battery Type')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Time series for best combination
        ax4 = plt.subplot(2, 3, 4)
        time_series = best_result["time_series"]
        ax4.plot(np.array(time_series["time_hours"]) / 24, time_series["voltage_percent"], 
                'b-', linewidth=2, label='Battery Level (%)')
        ax4.set_xlabel('Time (Days)')
        ax4.set_ylabel('Battery Level (%)')
        ax4.set_title(f'Discharge Curve\n{best_combo["combination"]}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Energy efficiency comparison
        ax5 = plt.subplot(2, 3, 5)
        efficiencies = []
        combo_labels = []
        for combo_key, result in results["detailed_results"].items():
            efficiencies.append(result["energy_efficiency"]["wh_per_detection"])
            battery_key, scenario_key = combo_key.split('_', 1)
            combo_labels.append(f"{self.BATTERY_TYPES[battery_key].name}\n{scenario_key.replace('_', ' ')}")
        
        y_pos = np.arange(len(efficiencies))
        ax5.barh(y_pos, efficiencies, color=plt.cm.viridis(np.linspace(0, 1, len(efficiencies))))
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(combo_labels, fontsize=8)
        ax5.set_xlabel('Energy per Detection (Wh)')
        ax5.set_title('Energy Efficiency Comparison')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary text
        summary_text = "🔋 SIMULATION SUMMARY\n\n"
        summary_text += f"Temperature: {results['simulation_info']['temperature_c']}°C\n"
        summary_text += f"Scenarios: {results['simulation_info']['scenarios_count']}\n"
        summary_text += f"Battery Types: {results['simulation_info']['battery_types_count']}\n\n"
        
        summary_text += "🏆 TOP RECOMMENDATIONS:\n"
        for i, rec in enumerate(results["summary"]["recommendations"][:4], 1):
            summary_text += f"{i}. {rec}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_dir / f"battery_life_simulation_{int(time.time())}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_file}")
        
        plt.show()


def main():
    """Main simulation entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="ENERGIE-3.1: Advanced Battery Life Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --temperature 25 --output results.json
  %(prog)s --temperature -10 --visualize
  %(prog)s --list-scenarios
        """
    )
    
    parser.add_argument('--temperature', '-t', type=float, default=25.0,
                       help='Operating temperature in Celsius (default: 25.0)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for detailed results')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--list-scenarios', action='store_true',
                       help='List available usage scenarios and exit')
    parser.add_argument('--list-batteries', action='store_true',
                       help='List available battery types and exit')
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = BatteryLifeSimulator()
    
    # Handle info requests
    if args.list_scenarios:
        print("📋 Available Usage Scenarios:")
        for key, scenario in simulator.usage_scenarios.items():
            print(f"  {key}: {scenario.description}")
            print(f"    Sleep: {scenario.sleep_time_percent}%, Active: {scenario.active_time_percent}%")
            print(f"    Inference cycles/hour: {scenario.inference_cycles_per_hour}")
        return
    
    if args.list_batteries:
        print("🔋 Available Battery Types:")
        for key, battery in simulator.BATTERY_TYPES.items():
            print(f"  {key}: {battery.name}")
            print(f"    Capacity: {battery.capacity_mah} mAh, Weight: {battery.weight_g}g")
            print(f"    Chemistry: {battery.chemistry}")
        return
    
    # Run comprehensive simulation
    print(f"🚀 ENERGIE-3.1: Battery Life Simulation Starting...")
    
    output_filename = args.output if args.output else f"battery_simulation_{int(time.time())}.json"
    
    results = simulator.run_comprehensive_simulation(
        temperature_c=args.temperature,
        output_file=output_filename
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n🏆 BEST COMBINATIONS:")
    for i, combo in enumerate(results["summary"]["best_combinations"], 1):
        print(f"  {i}. {combo['combination']}: {combo['runtime_days']:.1f} days")
        print(f"     Current: {combo['average_current_ma']:.2f} mA, Weight: {combo['battery_weight_g']}g")
    
    print("\n🔧 RECOMMENDATIONS:")
    for rec in results["summary"]["recommendations"]:
        print(f"  • {rec}")
    
    # Generate visualization if requested
    if args.visualize:
        print(f"\n📊 Generating visualization...")
        simulator.create_visualization(results)
    
    print(f"\n✅ Simulation completed successfully!")
    print(f"   Results saved to: {OUTPUT_DIR / output_filename}")


if __name__ == "__main__":
    main()
