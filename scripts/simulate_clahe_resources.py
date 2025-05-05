#!/usr/bin/env python3
"""
Simuliert und analysiert die Ressourcenanforderungen der CLAHE-Bildvorverarbeitung für RP2040.

Dieses Skript simuliert die Ausführung der CLAHE-Implementierung auf dem RP2040-Mikrocontroller
und generiert einen Bericht über Speichernutzung, Verarbeitungszeit und Temperatur.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Füge das Projekt-Root zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import INPUT_SIZE

# Konstanten für RP2040
RP2040_CLOCK_SPEED = 133_000_000  # 133 MHz
RP2040_FLASH_TOTAL = 2048 * 1024  # 2048 KB
RP2040_RAM_TOTAL = 264 * 1024     # 264 KB
RP2040_RAM_AVAILABLE = 204 * 1024 # 204 KB (nach Berücksichtigung von System-Overhead)

class CLAHEResourceSimulator:
    """Simuliert die Ressourcenanforderungen der CLAHE-Implementierung."""
    
    def __init__(self, image_size=(48, 48), channels=3):
        """Initialisiert den Simulator mit der angegebenen Bildgröße."""
        self.image_width, self.image_height = image_size
        self.channels = channels
        self.pixels = self.image_width * self.image_height
        
    def simulate_ram_usage(self, grid_size=8):
        """Simuliert die RAM-Nutzung der CLAHE-Implementierung."""
        # Histogramm (uint32_t[256])
        histogram_size = 256 * 4  # 256 Einträge x 4 Bytes
        
        # Lookup-Tabelle (uint8_t[256])
        lut_size = 256  # 256 Einträge x 1 Byte
        
        # Y-Kanal (uint8_t[width*height]) für RGB-Modus
        y_channel_size = self.pixels
        
        # Zusätzliche Variablen (Schätzung)
        misc_size = 512
        
        # Gesamte RAM-Nutzung
        total_ram = histogram_size + lut_size + y_channel_size + misc_size
        
        # Zusätzlicher RAM für I/O und Zwischenpuffer
        # (konservative Schätzung)
        io_buffer = self.pixels * self.channels  # Eingabebild
        
        return {
            "core": total_ram,
            "with_io": total_ram + io_buffer,
            "histogram_bytes": histogram_size,
            "lut_bytes": lut_size,
            "y_channel_bytes": y_channel_size,
            "misc_bytes": misc_size,
            "io_buffer_bytes": io_buffer
        }
    
    def simulate_flash_usage(self):
        """Schätzt die Flash-Nutzung des CLAHE-Codes."""
        # Geschätzter Codeumfang in Bytes (basierend auf der Implementierung)
        code_size = 2048  # ~2KB Code
        return code_size
    
    def simulate_processing_time(self, grid_size=8, clip_limit=4):
        """Simuliert die Verarbeitungszeit der CLAHE-Implementierung."""
        # Geschätzte Zyklen pro Operation
        create_histogram_cycles = 5  # Zyklen pro Pixel für Histogrammerstellung
        clip_limit_cycles = 2        # Zyklen pro Histogrammeintrag
        create_lut_cycles = 3        # Zyklen pro Histogrammeintrag
        apply_lut_cycles = 4         # Zyklen pro Pixel für LUT-Anwendung
        
        # Anzahl der Gitter
        grid_count = grid_size * grid_size
        
        # Pixels pro Gitter (durchschnittlich)
        pixels_per_grid = self.pixels / grid_count
        
        # Berechne Zyklen für jeden Schritt
        histogram_cycles = grid_count * pixels_per_grid * create_histogram_cycles
        clip_cycles = grid_count * 256 * clip_limit_cycles
        lut_cycles = grid_count * 256 * create_lut_cycles
        apply_cycles = self.pixels * apply_lut_cycles
        
        # Gesamtzahl der Zyklen
        total_cycles = histogram_cycles + clip_cycles + lut_cycles + apply_cycles
        
        # Overhead-Faktor (Funktionsaufrufe, Schleifenverwaltung, etc.)
        overhead_factor = 1.2
        total_cycles *= overhead_factor
        
        # Konvertiere zu Zeit
        seconds = total_cycles / RP2040_CLOCK_SPEED
        
        return {
            "total_cycles": int(total_cycles),
            "seconds": seconds,
            "histogram_cycles": int(histogram_cycles),
            "clip_cycles": int(clip_cycles),
            "lut_cycles": int(lut_cycles),
            "apply_cycles": int(apply_cycles)
        }
    
    def simulate_temperature_impact(self, processing_time_seconds):
        """Schätzt den Temperaturanstieg während der CLAHE-Verarbeitung."""
        # Geschätzte Parameter für Temperaturmodellierung
        # (basierend auf empirischen Daten für den RP2040)
        base_temp_rise = 0.1  # °C/s bei Dauerlast
        power_factor = 0.8    # Wie viel Leistung im Vergleich zur Maximalleistung
        
        # Temperaturanstieg während der Verarbeitung
        temp_rise = base_temp_rise * power_factor * processing_time_seconds
        
        return {
            "temp_rise_celsius": temp_rise
        }
    
    def simulate_all(self, grid_size=8, clip_limit=4):
        """Führt alle Simulationen durch und gibt ein umfassendes Ergebnis zurück."""
        ram_usage = self.simulate_ram_usage(grid_size)
        flash_usage = self.simulate_flash_usage()
        processing_time = self.simulate_processing_time(grid_size, clip_limit)
        temp_impact = self.simulate_temperature_impact(processing_time["seconds"])
        
        return {
            "ram_usage": ram_usage,
            "flash_usage": flash_usage,
            "processing_time": processing_time,
            "temperature_impact": temp_impact,
            "parameters": {
                "image_width": self.image_width,
                "image_height": self.image_height,
                "channels": self.channels,
                "grid_size": grid_size,
                "clip_limit": clip_limit
            }
        }
    
    def generate_report(self, results, output_file):
        """Generiert einen Bericht basierend auf den Simulationsergebnissen."""
        with open(output_file, 'w') as f:
            f.write("=== CLAHE-Ressourcensimulationsbericht ===\n\n")
            
            # Parameter
            params = results["parameters"]
            f.write(f"Bildgröße: {params['image_width']}x{params['image_height']} Pixel, {params['channels']} Kanäle\n")
            f.write(f"Grid-Größe: {params['grid_size']}x{params['grid_size']}\n")
            f.write(f"Clip-Limit: {params['clip_limit']}\n\n")
            
            # RAM-Nutzung
            ram = results["ram_usage"]
            f.write("RAM-Nutzung:\n")
            f.write(f"  Kern-Funktionalität: {ram['core']} Bytes ({ram['core']/1024:.2f} KB)\n")
            f.write(f"  Mit I/O-Puffer: {ram['with_io']} Bytes ({ram['with_io']/1024:.2f} KB)\n")
            ram_percentage = (ram['with_io'] / RP2040_RAM_AVAILABLE) * 100
            f.write(f"  Prozentsatz des verfügbaren RAM: {ram_percentage:.2f}%\n\n")
            
            # Flash-Nutzung
            f.write("Flash-Nutzung:\n")
            f.write(f"  Code-Größe: {results['flash_usage']} Bytes ({results['flash_usage']/1024:.2f} KB)\n")
            flash_percentage = (results['flash_usage'] / RP2040_FLASH_TOTAL) * 100
            f.write(f"  Prozentsatz des verfügbaren Flash: {flash_percentage:.2f}%\n\n")
            
            # Verarbeitungszeit
            time = results["processing_time"]
            f.write("Verarbeitungszeit:\n")
            f.write(f"  Gesamtzyklen: {time['total_cycles']} Zyklen\n")
            f.write(f"  Zeit bei {RP2040_CLOCK_SPEED/1000000} MHz: {time['seconds']*1000:.2f} ms\n\n")
            
            # Temperatur
            temp = results["temperature_impact"]
            f.write("Thermische Auswirkungen:\n")
            f.write(f"  Geschätzter Temperaturanstieg: {temp['temp_rise_celsius']:.3f} °C\n\n")
            
            # Empfehlungen
            f.write("Empfehlungen:\n")
            
            if ram_percentage > 80:
                f.write("  [!] WARNUNG: Hohe RAM-Nutzung. Erwägen Sie eine Reduzierung der Grid-Größe\n")
                f.write("      oder verwenden Sie die In-Place-Verarbeitung.\n")
            else:
                f.write("  [✓] RAM-Nutzung ist akzeptabel.\n")
                
            if time['seconds'] > 0.1:
                f.write("  [!] WARNUNG: Verarbeitungszeit könnte die Echtzeit-Anforderungen beeinträchtigen.\n")
                f.write("      Erwägen Sie eine Reduzierung der Grid-Größe oder die Verwendung\n")
                f.write("      einer optimierten Implementierung.\n")
            else:
                f.write("  [✓] Verarbeitungszeit ist akzeptabel.\n")
                
            f.write("\n=== Ende des Berichts ===\n")
    
    def generate_visualization(self, results, output_file):
        """Erzeugt eine Visualisierung der Ressourcensimulation."""
        plt.figure(figsize=(10, 8))
        
        # RAM-Nutzung (als Balken)
        plt.subplot(2, 2, 1)
        ram_sizes = [results["ram_usage"]["histogram_bytes"], 
                     results["ram_usage"]["lut_bytes"],
                     results["ram_usage"]["y_channel_bytes"],
                     results["ram_usage"]["misc_bytes"]]
        ram_labels = ['Histogramm', 'LUT', 'Y-Kanal', 'Sonstiges']
        plt.bar(ram_labels, ram_sizes)
        plt.title('RAM-Nutzung (Bytes)')
        plt.ylabel('Bytes')
        plt.xticks(rotation=45)
        
        # Zyklen pro Schritt (als Kreisdiagramm)
        plt.subplot(2, 2, 2)
        cycle_sizes = [results["processing_time"]["histogram_cycles"],
                        results["processing_time"]["clip_cycles"],
                        results["processing_time"]["lut_cycles"],
                        results["processing_time"]["apply_cycles"]]
        cycle_labels = ['Histogramm', 'Clip', 'LUT', 'Anwenden']
        plt.pie(cycle_sizes, labels=cycle_labels, autopct='%1.1f%%')
        plt.title('CPU-Zyklen pro Schritt')
        
        # Gesamte RAM-Nutzung vs. Verfügbarer RAM (als gefüllter Bereich)
        plt.subplot(2, 2, 3)
        labels = ['Verwendet', 'Verfügbar']
        sizes = [results["ram_usage"]["with_io"], 
                 RP2040_RAM_AVAILABLE - results["ram_usage"]["with_io"]]
        colors = ['#ff9999','#66b3ff']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('RAM-Nutzung vs. Verfügbar')
        
        # Verarbeitungszeit bei verschiedenen Grid-Größen (als Linie)
        plt.subplot(2, 2, 4)
        grid_sizes = [2, 4, 6, 8, 10, 12]
        times = []
        for gs in grid_sizes:
            times.append(self.simulate_processing_time(grid_size=gs)["seconds"] * 1000)  # ms
        plt.plot(grid_sizes, times, marker='o')
        plt.title('Verarbeitungszeit vs. Grid-Größe')
        plt.xlabel('Grid-Größe')
        plt.ylabel('Zeit (ms)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Simuliert die Ressourcenanforderungen der CLAHE-Implementierung')
    parser.add_argument('--width', type=int, default=INPUT_SIZE, help='Bildbreite in Pixeln')
    parser.add_argument('--height', type=int, default=INPUT_SIZE, help='Bildhöhe in Pixeln')
    parser.add_argument('--channels', type=int, default=3, help='Anzahl der Bildkanäle (1 für Graustufen, 3 für RGB)')
    parser.add_argument('--grid-size', type=int, default=8, help='Größe des CLAHE-Gitters')
    parser.add_argument('--clip-limit', type=int, default=4, help='CLAHE Clip-Limit')
    parser.add_argument('--output', default='output/clahe_resources', help='Ausgabeverzeichnis')
    args = parser.parse_args()
    
    # Erstelle Ausgabeverzeichnis
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Erstelle Simulator
    simulator = CLAHEResourceSimulator(
        image_size=(args.width, args.height),
        channels=args.channels
    )
    
    print(f"Simuliere CLAHE-Ressourcen für {args.width}x{args.height} Bild...")
    
    # Führe Simulation durch
    results = simulator.simulate_all(
        grid_size=args.grid_size,
        clip_limit=args.clip_limit
    )
    
    # Speichere Ergebnisse als JSON
    with open(output_dir / 'clahe_simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generiere Bericht
    simulator.generate_report(results, output_dir / 'clahe_simulation_report.txt')
    
    # Generiere Visualisierung
    simulator.generate_visualization(results, output_dir / 'clahe_simulation_visualization.png')
    
    print(f"Simulation abgeschlossen. Ergebnisse gespeichert in {output_dir}")
    
    # Gib zusammenfassende Informationen aus
    ram_percentage = (results["ram_usage"]["with_io"] / RP2040_RAM_AVAILABLE) * 100
    processing_time_ms = results["processing_time"]["seconds"] * 1000
    
    print(f"\nZusammenfassung:")
    print(f"  RAM-Nutzung: {results['ram_usage']['with_io']/1024:.2f} KB ({ram_percentage:.2f}% des verfügbaren RAM)")
    print(f"  Flash-Nutzung: {results['flash_usage']/1024:.2f} KB")
    print(f"  Verarbeitungszeit: {processing_time_ms:.2f} ms")
    print(f"  Temperaturanstieg: {results['temperature_impact']['temp_rise_celsius']:.3f} °C\n")
    
    if ram_percentage > 80:
        print("WARNUNG: Hohe RAM-Nutzung könnte problematisch sein!")
    
    if processing_time_ms > 100:
        print("WARNUNG: Hohe Verarbeitungszeit könnte Echtzeitanforderungen beeinträchtigen!")

if __name__ == "__main__":
    main()