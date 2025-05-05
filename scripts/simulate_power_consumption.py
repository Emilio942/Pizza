"""
Simulation der Batterielebensdauer unter verschiedenen Betriebsbedingungen für das RP2040-basierte Pizza-Erkennungssystem.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json

# Füge das Stammverzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.constants import (
    RP2040_FLASH_SIZE_KB,
    RP2040_RAM_SIZE_KB,
    RP2040_CLOCK_SPEED_MHZ
)

# Simulationsparameter
OUTPUT_DIR = Path("/home/emilio/Documents/ai/pizza/output/simulations")

# Batteriekonfigurationen
BATTERY_CONFIGS = [
    {"name": "CR123A", "capacity_mah": 1500, "weight_g": 17},
    {"name": "AA Alkaline", "capacity_mah": 2500, "weight_g": 23},
    {"name": "18650 Li-Ion", "capacity_mah": 3400, "weight_g": 47},
    {"name": "LiPo 500mAh", "capacity_mah": 500, "weight_g": 10}
]

# Leistungsprofile
POWER_PROFILES = {
    "Sleep Mode": 0.5,  # mA
    "Idle": 10,         # mA
    "Active (133MHz)": 80,       # mA
    "Active (250MHz)": 150,      # mA
    "Camera Active": 40,         # mA
    "Inference Running": 100     # mA
}

def simulate_battery_life():
    """Simuliert Batterielebensdauer unter verschiedenen Betriebsbedingungen."""
    
    # Ergebnisstruktur
    results = {
        "battery_types": [],
        "continuous_runtime_hours": [],
        "duty_cycle_runtime_hours": [],
        "weight_g": []
    }
    
    # Szenario 1: Kontinuierliche Erkennung
    continuous_scenario = {
        "Sleep Mode": 0.0,       # 0% der Zeit
        "Idle": 0.2,             # 20% der Zeit
        "Active (133MHz)": 0.3,  # 30% der Zeit
        "Camera Active": 0.2,    # 20% der Zeit
        "Inference Running": 0.3 # 30% der Zeit
    }
    
    # Szenario 2: Periodisches Aufwecken (1 Erkennung pro Minute)
    duty_cycle_scenario = {
        "Sleep Mode": 0.9,       # 90% der Zeit
        "Idle": 0.02,            # 2% der Zeit
        "Active (133MHz)": 0.03, # 3% der Zeit
        "Camera Active": 0.02,   # 2% der Zeit
        "Inference Running": 0.03 # 3% der Zeit
    }
    
    # Berechne durchschnittlichen Stromverbrauch für jedes Szenario
    continuous_current = sum(POWER_PROFILES[mode] * ratio for mode, ratio in continuous_scenario.items())
    duty_cycle_current = sum(POWER_PROFILES[mode] * ratio for mode, ratio in duty_cycle_scenario.items())
    
    print(f"Szenario 1 (Kontinuierlich): Durchschnittlicher Stromverbrauch: {continuous_current:.2f} mA")
    print(f"Szenario 2 (Duty Cycle): Durchschnittlicher Stromverbrauch: {duty_cycle_current:.2f} mA")
    
    # Berechne Laufzeit für jede Batterie
    for battery in BATTERY_CONFIGS:
        results["battery_types"].append(battery["name"])
        results["weight_g"].append(battery["weight_g"])
        
        # Laufzeit = Kapazität / Stromverbrauch (in Stunden)
        continuous_runtime = battery["capacity_mah"] / continuous_current
        duty_cycle_runtime = battery["capacity_mah"] / duty_cycle_current
        
        results["continuous_runtime_hours"].append(continuous_runtime)
        results["duty_cycle_runtime_hours"].append(duty_cycle_runtime)
        
        print(f"Batterie: {battery['name']} ({battery['capacity_mah']} mAh)")
        print(f"  Laufzeit bei kontinuierlicher Erkennung: {continuous_runtime:.2f} Stunden ({continuous_runtime/24:.2f} Tage)")
        print(f"  Laufzeit im Duty-Cycle-Modus: {duty_cycle_runtime:.2f} Stunden ({duty_cycle_runtime/24:.2f} Tage)")
    
    return results

def simulate_cpu_clock_impact():
    """Simuliert den Einfluss der CPU-Taktrate auf die Leistung und Batterielebensdauer."""
    
    # CPU-Frequenzen (MHz)
    clock_speeds = [50, 100, 133, 150, 200, 250]
    
    # Geschätzte Inferenzzeit basierend auf Taktrate (ms)
    inference_times = [400, 200, 150, 133, 100, 80]
    
    # Geschätzter Stromverbrauch basierend auf Taktrate (mA)
    power_consumption = [40, 70, 80, 90, 120, 150]
    
    # Batterielebensdauer bei periodischer Erkennung (1x pro Minute)
    # mit einer 1500mAh CR123A-Batterie
    battery_life = []
    
    for i, power in enumerate(power_consumption):
        # Aktive Zeit pro Stunde (s) = 60 * (Inferenzzeit / 1000)
        active_time_per_hour = 60 * (inference_times[i] / 1000)
        
        # Aktiver Anteil
        active_ratio = active_time_per_hour / 3600
        
        # Durchschnittlicher Stromverbrauch (mA)
        avg_current = (power * active_ratio) + (POWER_PROFILES["Sleep Mode"] * (1 - active_ratio))
        
        # Batterielebensdauer (h)
        life = 1500 / avg_current
        battery_life.append(life)
    
    return {
        "clock_speeds": clock_speeds,
        "inference_times": inference_times,
        "power_consumption": power_consumption,
        "battery_life": battery_life
    }

def plot_battery_results(results):
    """Erstellt Diagramme der Batterielebensdauer-Simulationsergebnisse."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Batterielebensdauer
    plt.subplot(2, 2, 1)
    x = np.arange(len(results["battery_types"]))
    width = 0.35
    
    plt.bar(x - width/2, results["continuous_runtime_hours"], width, label='Kontinuierlich')
    plt.bar(x + width/2, results["duty_cycle_runtime_hours"], width, label='Duty Cycle')
    
    plt.xlabel('Batterietyp')
    plt.ylabel('Laufzeit (Stunden)')
    plt.title('Batterielebensdauer nach Betriebsmodus')
    plt.xticks(x, results["battery_types"])
    plt.legend()
    
    # Zeige Lebensdauer in Tagen für Duty Cycle
    for i, v in enumerate(results["duty_cycle_runtime_hours"]):
        plt.text(i + width/2, v + 10, f"{v/24:.1f} Tage", ha='center')
    
    # Plot 2: Energie/Gewicht-Verhältnis
    plt.subplot(2, 2, 2)
    energy_density = [rt / wt for rt, wt in zip(results["duty_cycle_runtime_hours"], results["weight_g"])]
    
    plt.bar(x, energy_density, color='green')
    plt.xlabel('Batterietyp')
    plt.ylabel('Stunden Laufzeit / Gramm')
    plt.title('Energieeffizienz (Duty Cycle)')
    plt.xticks(x, results["battery_types"])
    
    # Plot 3: Stromverbrauch nach Modus
    plt.subplot(2, 2, 3)
    modes = list(POWER_PROFILES.keys())
    currents = list(POWER_PROFILES.values())
    
    plt.bar(modes, currents, color='orange')
    plt.xlabel('Betriebsmodus')
    plt.ylabel('Stromverbrauch (mA)')
    plt.title('Stromverbrauch nach Betriebsmodus')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Speichere die Ergebnisse
    output_path = OUTPUT_DIR / "battery_simulation.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_clock_speed_results(results):
    """Erstellt Diagramme der CPU-Taktrate-Simulationsergebnisse."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Inferenzzeit vs. Taktrate
    plt.subplot(2, 2, 1)
    plt.plot(results["clock_speeds"], results["inference_times"], 'o-', linewidth=2)
    plt.xlabel('CPU-Taktrate (MHz)')
    plt.ylabel('Inferenzzeit (ms)')
    plt.title('Inferenzzeit vs. CPU-Taktrate')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Stromverbrauch vs. Taktrate
    plt.subplot(2, 2, 2)
    plt.plot(results["clock_speeds"], results["power_consumption"], 'o-', linewidth=2, color='red')
    plt.xlabel('CPU-Taktrate (MHz)')
    plt.ylabel('Stromverbrauch (mA)')
    plt.title('Stromverbrauch vs. CPU-Taktrate')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Batterielebensdauer vs. Taktrate
    plt.subplot(2, 2, 3)
    plt.plot(results["clock_speeds"], results["battery_life"], 'o-', linewidth=2, color='green')
    plt.xlabel('CPU-Taktrate (MHz)')
    plt.ylabel('Batterielebensdauer (Stunden)')
    plt.title('Batterielebensdauer vs. CPU-Taktrate\n(1 Inferenz pro Minute)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Effizienz-Metrik: Inferenzen pro Ladung vs. Taktrate
    plt.subplot(2, 2, 4)
    inferences_per_charge = [life * 60 for life in results["battery_life"]]  # Stunden * 60 Inferenzen/Stunde
    
    plt.plot(results["clock_speeds"], inferences_per_charge, 'o-', linewidth=2, color='purple')
    plt.xlabel('CPU-Taktrate (MHz)')
    plt.ylabel('Inferenzen pro Batterieladung')
    plt.title('Anzahl möglicher Inferenzen pro Batterieladung')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichere die Ergebnisse
    output_path = OUTPUT_DIR / "clock_speed_simulation.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    # Stelle sicher, dass der Ausgabeordner existiert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Simuliere Batterielebensdauer unter verschiedenen Betriebsbedingungen...")
    battery_results = simulate_battery_life()
    
    battery_plot_path = plot_battery_results(battery_results)
    print(f"Batterielebensdauer-Simulationsergebnisse gespeichert unter: {battery_plot_path}")
    
    print("\nSimuliere den Einfluss der CPU-Taktrate auf Leistung und Batterielebensdauer...")
    clock_results = simulate_cpu_clock_impact()
    
    clock_plot_path = plot_clock_speed_results(clock_results)
    print(f"CPU-Taktrate-Simulationsergebnisse gespeichert unter: {clock_plot_path}")
    
    # Speichere vollständige Simulationsergebnisse als JSON
    results_data = {
        "battery_simulation": battery_results,
        "clock_speed_simulation": clock_results,
        "power_profiles": POWER_PROFILES,
        "battery_configs": BATTERY_CONFIGS
    }
    
    results_path = OUTPUT_DIR / "power_simulation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Vollständige Simulationsergebnisse gespeichert unter: {results_path}")