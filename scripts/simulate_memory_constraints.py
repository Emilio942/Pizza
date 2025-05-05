"""
Simulation der Speichernutzung für Modellkomplexität und Eingabebildgrößen auf dem RP2040.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

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

# Definition der Speicherbeschränkungen
FLASH_LIMIT_KB = RP2040_FLASH_SIZE_KB
RAM_LIMIT_KB = RP2040_RAM_SIZE_KB

# Typische RP2040 Nutzung (systemrelevant)
SYSTEM_FLASH_KB = 300  # Firmware, Libraries, etc.
SYSTEM_RAM_KB = 60     # OS, Treiber, etc.

# Verfügbarer Speicher für die Anwendung
AVAILABLE_FLASH_KB = FLASH_LIMIT_KB - SYSTEM_FLASH_KB
AVAILABLE_RAM_KB = RAM_LIMIT_KB - SYSTEM_RAM_KB

def estimate_model_memory(params, bit_width=8):
    """Schätzt Speicherbedarf eines Modells basierend auf Parameterzahl und Bitbreite."""
    bytes_per_param = bit_width / 8
    model_size_kb = (params * bytes_per_param) / 1024
    return model_size_kb

def estimate_activation_memory(input_size, channels=3):
    """Schätzt RAM-Bedarf für Aktivierungen während der Inferenz."""
    # Einfache Schätzung basierend auf Eingabegröße
    # Annahme: Aktivierungen benötigen etwa 5x die Größe des Eingabebilds
    input_memory_kb = (input_size[0] * input_size[1] * channels * 4) / 1024  # Float32 = 4 Bytes
    activation_memory_kb = input_memory_kb * 5
    return activation_memory_kb

def simulate_model_complexity():
    """Simuliert verschiedene Modellkomplexitäten und deren Speicherbedarf."""
    
    # Modellparameter (von einfach bis komplex)
    param_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    
    # Bit-Breiten für Quantisierung
    bit_widths = [32, 16, 8]
    
    # Ergebnisstruktur
    results = {
        "param_counts": param_counts,
        "model_sizes": {}
    }
    
    # Berechne Modellgrößen für verschiedene Bit-Breiten
    for bits in bit_widths:
        model_sizes = [estimate_model_memory(params, bits) for params in param_counts]
        results["model_sizes"][f"{bits}bit"] = model_sizes
    
    # Berechne Flash-Auslastung für 8-bit Modelle
    flash_usage_percent = [(size / AVAILABLE_FLASH_KB) * 100 for size in results["model_sizes"]["8bit"]]
    results["flash_usage_percent"] = flash_usage_percent
    
    print(f"Verfügbarer Flash-Speicher: {AVAILABLE_FLASH_KB:.2f} KB von {FLASH_LIMIT_KB} KB")
    print(f"Verfügbarer RAM: {AVAILABLE_RAM_KB:.2f} KB von {RAM_LIMIT_KB} KB")
    print("\nModellgröße nach Parameterzahl und Quantisierung:")
    
    for i, params in enumerate(param_counts):
        print(f"  {params:,} Parameter:")
        for bits in bit_widths:
            print(f"    {bits}-bit: {results['model_sizes'][f'{bits}bit'][i]:.2f} KB " + 
                  f"({(results['model_sizes'][f'{bits}bit'][i] / AVAILABLE_FLASH_KB) * 100:.1f}% des verfügbaren Flash)")
    
    return results

def simulate_input_size_impact():
    """Simuliert den Einfluss der Eingabebildgröße auf den Speicherbedarf."""
    
    # Bildgrößen (quadratisch)
    input_sizes = [(28, 28), (32, 32), (48, 48), (64, 64), (96, 96), (128, 128), (160, 160), (224, 224)]
    
    # Ergebnisstruktur
    results = {
        "input_sizes": [f"{size[0]}x{size[1]}" for size in input_sizes],
        "input_memory_kb": [],
        "activation_memory_kb": [],
        "total_runtime_kb": [],
        "ram_usage_percent": []
    }
    
    # Annahme: Feste Modellgröße (int8-quantisiert)
    model_params = 20000
    model_size_kb = estimate_model_memory(model_params, 8)
    
    print(f"\nSimuliere Auswirkungen verschiedener Eingabebildgrößen auf den Speicherverbrauch")
    print(f"Feste Modellgröße: {model_params:,} Parameter, 8-bit quantisiert = {model_size_kb:.2f} KB")
    
    # Berechne Speicheranforderungen für verschiedene Eingabegrößen
    for size in input_sizes:
        # Eingabespeicher (3 Kanäle, RGB)
        input_memory_kb = (size[0] * size[1] * 3 * 4) / 1024  # Float32
        
        # Schätze Aktivierungsspeicher
        activation_memory_kb = estimate_activation_memory(size)
        
        # Gesamter Laufzeitspeicher (Modell muss auch im RAM sein)
        total_runtime_kb = model_size_kb + activation_memory_kb
        
        # Prozentuale RAM-Nutzung
        ram_usage_percent = (total_runtime_kb / AVAILABLE_RAM_KB) * 100
        
        # Speichere Ergebnisse
        results["input_memory_kb"].append(input_memory_kb)
        results["activation_memory_kb"].append(activation_memory_kb)
        results["total_runtime_kb"].append(total_runtime_kb)
        results["ram_usage_percent"].append(ram_usage_percent)
        
        # Gibt Überlastungswarnungen aus
        status = "OK" if ram_usage_percent < 100 else "ÜBERLAUF!"
        print(f"  {size[0]}x{size[1]}: {total_runtime_kb:.2f} KB RAM ({ram_usage_percent:.1f}% des verfügbaren RAM) - {status}")
    
    return results

def plot_model_complexity_results(results):
    """Erstellt Diagramme der Modellkomplexitäts-Simulationsergebnisse."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Modellgröße vs. Parameterzahl
    plt.subplot(2, 2, 1)
    for bits, sizes in results["model_sizes"].items():
        plt.plot(results["param_counts"], sizes, 'o-', linewidth=2, label=f"{bits}")
    
    plt.axhline(y=AVAILABLE_FLASH_KB, color='r', linestyle='--', label="Verfügbarer Flash")
    plt.xlabel('Anzahl Parameter')
    plt.ylabel('Modellgröße (KB)')
    plt.title('Modellgröße vs. Parameterzahl und Quantisierung')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    # Plot 2: Flash-Auslastung vs. Parameterzahl (8-bit)
    plt.subplot(2, 2, 2)
    safe_zone = plt.axhspan(0, 80, alpha=0.2, color='green', label="Sicher (<80%)")
    warning_zone = plt.axhspan(80, 100, alpha=0.2, color='yellow', label="Warnung (80-100%)")
    danger_zone = plt.axhspan(100, max(results["flash_usage_percent"])*1.1, alpha=0.2, color='red', label="Überlauf (>100%)")
    
    plt.plot(results["param_counts"], results["flash_usage_percent"], 'o-', linewidth=2, color='blue')
    plt.xlabel('Anzahl Parameter')
    plt.ylabel('Flash-Auslastung (%)')
    plt.title('Flash-Speicherauslastung (8-bit Modelle)')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.legend(handles=[safe_zone, warning_zone, danger_zone])
    
    # Markiere die maximale sichere Parameterzahl
    max_safe_params = 0
    for i, usage in enumerate(results["flash_usage_percent"]):
        if usage > 80:
            if i > 0:
                max_safe_params = results["param_counts"][i-1]
            break
    
    if max_safe_params > 0:
        plt.axvline(x=max_safe_params, color='g', linestyle='--')
        plt.text(max_safe_params, 40, f"{max_safe_params:,}\nParameter", 
                 ha='right', va='center', bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 3: Modellgröße vs. Quantisierung für ausgewählte Parameterzahlen
    plt.subplot(2, 2, 3)
    selected_indices = [2, 5, 7]  # Ausgewählte Indizes für Parameterzahlen
    bit_values = [32, 16, 8]
    
    for idx in selected_indices:
        param_count = results["param_counts"][idx]
        sizes = [results["model_sizes"][f"{bits}bit"][idx] for bits in bit_values]
        plt.plot(bit_values, sizes, 'o-', linewidth=2, label=f"{param_count:,} Param.")
    
    plt.axhline(y=AVAILABLE_FLASH_KB, color='r', linestyle='--', label="Verfügbarer Flash")
    plt.xlabel('Bitbreite')
    plt.ylabel('Modellgröße (KB)')
    plt.title('Auswirkung der Quantisierung auf die Modellgröße')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(bit_values)
    
    # Plot 4: Speichereffizienz (Parameter pro KB) vs. Bitbreite
    plt.subplot(2, 2, 4)
    bit_values = [32, 16, 8]
    params_per_kb = [(8/bits) * 1024 / 8 for bits in bit_values]  # Parameter pro KB
    
    plt.bar(bit_values, params_per_kb, color='purple')
    plt.xlabel('Bitbreite')
    plt.ylabel('Parameter pro KB')
    plt.title('Speichereffizienz verschiedener Quantisierungen')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(bit_values)
    
    # Werte über den Balken anzeigen
    for i, v in enumerate(params_per_kb):
        plt.text(bit_values[i], v + 10, f"{int(v)}", ha='center')
    
    plt.tight_layout()
    
    # Speichere die Ergebnisse
    output_path = OUTPUT_DIR / "model_complexity_simulation.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_input_size_results(results):
    """Erstellt Diagramme der Eingabebildgrößen-Simulationsergebnisse."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: RAM-Verbrauch vs. Eingabebildgröße
    plt.subplot(2, 2, 1)
    x = np.arange(len(results["input_sizes"]))
    plt.bar(x, results["total_runtime_kb"], color='blue')
    plt.axhline(y=AVAILABLE_RAM_KB, color='r', linestyle='--', label="Verfügbarer RAM")
    plt.xlabel('Eingabebildgröße')
    plt.ylabel('RAM-Verbrauch (KB)')
    plt.title('Gesamter RAM-Verbrauch während der Inferenz')
    plt.xticks(x, results["input_sizes"], rotation=45)
    plt.legend()
    
    # Plot 2: RAM-Aufteilung für verschiedene Bildgrößen
    plt.subplot(2, 2, 2)
    width = 0.4
    x = np.arange(len(results["input_sizes"]))
    
    # Annahme: Feste Modellgröße (aus dem vorherigen Code)
    model_params = 20000
    model_size_kb = estimate_model_memory(model_params, 8)
    model_sizes = [model_size_kb] * len(results["input_sizes"])
    
    plt.bar(x, model_sizes, width, label='Modell')
    plt.bar(x, results["activation_memory_kb"], width, bottom=model_sizes, label='Aktivierungen')
    
    plt.axhline(y=AVAILABLE_RAM_KB, color='r', linestyle='--', label="Verfügbarer RAM")
    plt.xlabel('Eingabebildgröße')
    plt.ylabel('RAM-Verbrauch (KB)')
    plt.title('Aufteilung des RAM-Verbrauchs')
    plt.xticks(x, results["input_sizes"], rotation=45)
    plt.legend()
    
    # Plot 3: RAM-Auslastung vs. Eingabebildgröße
    plt.subplot(2, 2, 3)
    safe_zone = plt.axhspan(0, 80, alpha=0.2, color='green', label="Sicher (<80%)")
    warning_zone = plt.axhspan(80, 100, alpha=0.2, color='yellow', label="Warnung (80-100%)")
    danger_zone = plt.axhspan(100, max(results["ram_usage_percent"])*1.1, alpha=0.2, color='red', label="Überlauf (>100%)")
    
    plt.plot(results["input_sizes"], results["ram_usage_percent"], 'o-', linewidth=2, color='blue')
    plt.xlabel('Eingabebildgröße')
    plt.ylabel('RAM-Auslastung (%)')
    plt.title('RAM-Auslastung während der Inferenz')
    plt.grid(True, alpha=0.3)
    plt.legend(handles=[safe_zone, warning_zone, danger_zone])
    plt.xticks(rotation=45)
    
    # Markiere die maximale sichere Bildgröße
    max_safe_idx = 0
    for i, usage in enumerate(results["ram_usage_percent"]):
        if usage > 80:
            if i > 0:
                max_safe_idx = i-1
            break
    
    if max_safe_idx > 0:
        safe_size = results["input_sizes"][max_safe_idx]
        plt.plot([safe_size], [results["ram_usage_percent"][max_safe_idx]], 'go', markersize=10)
        plt.text(safe_size, results["ram_usage_percent"][max_safe_idx] + 5, 
                 f"Max. sichere Größe:\n{safe_size}", ha='center', va='bottom', 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 4: Aktivierungsspeicher vs. Pixelanzahl
    plt.subplot(2, 2, 4)
    pixel_counts = [int(size.split('x')[0]) * int(size.split('x')[1]) for size in results["input_sizes"]]
    
    plt.plot(pixel_counts, results["activation_memory_kb"], 'o-', linewidth=2, color='green')
    plt.xlabel('Anzahl Pixel')
    plt.ylabel('Aktivierungsspeicher (KB)')
    plt.title('Aktivierungsspeicherbedarf vs. Pixelanzahl')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Zeigt die Trendlinie
    coeffs = np.polyfit(np.log10(pixel_counts), np.log10(results["activation_memory_kb"]), 1)
    poly = np.poly1d(coeffs)
    trend_label = f"Steigung: {coeffs[0]:.2f} (nahe 1 = linear)"
    log_x = np.log10(pixel_counts)
    plt.plot(pixel_counts, 10**poly(log_x), 'r--', label=trend_label)
    plt.legend()
    
    plt.tight_layout()
    
    # Speichere die Ergebnisse
    output_path = OUTPUT_DIR / "input_size_simulation.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    # Stelle sicher, dass der Ausgabeordner existiert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Simuliere den Einfluss der Modellkomplexität auf den Speicherbedarf...")
    complexity_results = simulate_model_complexity()
    
    complexity_plot_path = plot_model_complexity_results(complexity_results)
    print(f"Modellkomplexitäts-Simulationsergebnisse gespeichert unter: {complexity_plot_path}")
    
    print("\nSimuliere den Einfluss der Eingabebildgröße auf den Speicherbedarf...")
    input_size_results = simulate_input_size_impact()
    
    input_size_plot_path = plot_input_size_results(input_size_results)
    print(f"Eingabebildgrößen-Simulationsergebnisse gespeichert unter: {input_size_plot_path}")
    
    # Speichere vollständige Simulationsergebnisse als JSON
    results_data = {
        "complexity_simulation": complexity_results,
        "input_size_simulation": input_size_results,
        "memory_limits": {
            "flash_total_kb": FLASH_LIMIT_KB,
            "ram_total_kb": RAM_LIMIT_KB,
            "flash_available_kb": AVAILABLE_FLASH_KB,
            "ram_available_kb": AVAILABLE_RAM_KB
        }
    }
    
    results_path = OUTPUT_DIR / "memory_simulation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Vollständige Simulationsergebnisse gespeichert unter: {results_path}")