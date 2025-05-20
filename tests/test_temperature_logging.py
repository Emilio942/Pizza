"""
Testskript für die Temperaturmessung und das Logging im RP2040-Emulator.
Dieses Skript demonstriert die Verwendung des emulierten Temperatursensors und
die Protokollierung der Temperaturwerte über UART und ins Dateisystem.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path

print("Starting test script...")

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Füge den src-Pfad dem Python-Path hinzu, damit die Module gefunden werden
import sys
src_path = str(Path(__file__).parent.parent)
print(f"Adding source path: {src_path}")
if src_path not in sys.path:
    sys.path.append(src_path)

# Importiere die benötigten Module aus dem Emulator
from src.emulation.emulator import RP2040Emulator
from src.emulation.temperature_sensor import SensorType
from src.emulation.simple_power_manager import AdaptiveMode

def main():
    """Hauptfunktion des Testskripts."""
    print("Starte Temperaturmessung und Logging-Test für RP2040-Emulator")
    
    # Erstelle Emulator mit Standardparametern
    emulator = RP2040Emulator(
        battery_capacity_mah=1500.0,
        adaptive_mode=AdaptiveMode.BALANCED
    )
    
    # Setze das Temperatur-Logging-Intervall auf 5 Sekunden für den Test
    emulator.set_temperature_log_interval(5.0)
    
    print(f"Emulator gestartet. Temperaturmessung aktiv.")
    print(f"Logs werden in 'output/emulator_logs/' gespeichert.")
    
    try:
        # Simuliere verschiedene Szenarien
        
        # 1. Normale Temperatur für 10 Sekunden
        print("\n1. Normale Betriebstemperatur (10s)")
        for _ in range(5):
            temp = emulator.read_temperature()
            print(f"Temperatur: {temp:.2f}°C")
            # Simuliere leichte Last
            emulator.execute_operation(50 * 1024, 100)  # 50KB, 100ms
            time.sleep(2)
        
        # 2. Simuliere hohe Last (z.B. Inferenz) für 20 Sekunden
        print("\n2. Hohe Last - Erwärmung (20s)")
        # Erstelle ein Dummy-Bild für die Inferenz
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Lade eine simulierte Firmware
        dummy_firmware = {
            'name': 'test_firmware',
            'version': '1.0.0',
            'total_size_bytes': 512 * 1024,  # 512KB
            'ram_usage_bytes': 100 * 1024,   # 100KB
        }
        emulator.load_firmware(dummy_firmware)
        
        # Führe einige Inferenzen durch
        for i in range(10):
            print(f"Inferenz {i+1}/10...")
            result = emulator.simulate_inference(dummy_image)
            temp = emulator.read_temperature()
            print(f"  Temperatur: {temp:.2f}°C, Inferenzzeit: {result['inference_time']*1000:.2f}ms")
            time.sleep(1)
        
        # 3. Simuliere einen Temperaturspike
        print("\n3. Temperatur-Spike (+5°C für 15s)")
        emulator.inject_temperature_spike(5.0, 15.0)
        
        for _ in range(5):
            temp = emulator.read_temperature()
            print(f"Temperatur nach Spike: {temp:.2f}°C")
            time.sleep(3)
        
        # 4. Sleep-Modus und Abkühlung
        print("\n4. Sleep-Modus und Abkühlung (15s)")
        emulator.enter_sleep_mode()
        
        for _ in range(5):
            temp = emulator.read_temperature()
            print(f"Temperatur im Sleep-Modus: {temp:.2f}°C")
            time.sleep(3)
        
        # 5. Aufwachen und letzte Messung
        print("\n5. Aufwachen und finale Messungen")
        emulator.wake_up()
        
        for _ in range(3):
            temp = emulator.read_temperature()
            print(f"Temperatur nach Wake-up: {temp:.2f}°C")
            time.sleep(2)
        
        # Zeige Systemstatistiken an
        stats = emulator.get_system_stats()
        print("\nSystemstatistiken:")
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"  RAM-Nutzung: {stats['ram_used_kb']:.1f}KB / {stats['ram_used_kb'] + stats['ram_free_kb']:.1f}KB")
        print(f"  Aktuelle Temperatur: {stats['current_temperature_c']:.2f}°C")
        print(f"  Minimale Temperatur: {stats['temperature_min_c']:.2f}°C")
        print(f"  Maximale Temperatur: {stats['temperature_max_c']:.2f}°C")
        print(f"  Durchschnittliche Temperatur: {stats['temperature_avg_c']:.2f}°C")
        print(f"  Anzahl Temperaturmessungen: {stats['temperature_readings_count']}")
        
    except Exception as e:
        print(f"Fehler während des Tests: {e}")
    
    finally:
        # Schließe den Emulator am Ende
        print("\nTest abgeschlossen. Schließe Emulator...")
        emulator.close()
        print("Emulator geschlossen. Überprüfe die Logs in 'output/emulator_logs/'.")
        
        # Hinweis zur Visualisierung
        print("\nTipp: Visualisiere die Temperaturlogs mit:")
        print("python scripts/visualize_temperature.py")

if __name__ == "__main__":
    main()
