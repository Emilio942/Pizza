"""
Demo-Skript für das erweiterte adaptive Energiemanagement im RP2040 Emulator.
Demonstriert temperaturbasierte und kontextabhängige Optimierungen sowie verbesserte Statistiken.
"""

import os
import sys
import time
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Füge das Stammverzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.emulator import RP2040Emulator
from src.power_manager import PowerManager, PowerUsage, AdaptiveMode


def simulate_extended_operation(emulator, duration_seconds=60, interval_seconds=5, plot_stats=False):
    """
    Simuliert einen längeren Betrieb des Emulators mit regelmäßigen Inferenzen.
    
    Args:
        emulator: Der RP2040Emulator
        duration_seconds: Gesamtdauer der Simulation in Sekunden
        interval_seconds: Intervall zwischen Inferenzen in Sekunden
        plot_stats: Wenn True, werden die Statistiken als Grafik angezeigt
    """
    print(f"\n=== Simuliere Betrieb über {duration_seconds} Sekunden ===")
    print(f"Abtastintervall: {interval_seconds} Sekunden")
    print(f"Energiemanagement-Modus: {emulator.power_manager.mode.value}")
    
    # Erstelle ein simuliertes Bild
    test_image = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
    
    start_time = time.time()
    last_inference_time = 0
    inference_count = 0
    
    # Simuliere Sequenz von Klassen für ein realistisches Szenario
    class_sequence = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 0, 0]  # Simuliert Pizza-Backvorgang
    
    stats_interval = 10  # Statistik alle 10 Sekunden ausgeben
    last_stats_time = 0
    
    # Für die Statistikaufzeichnung
    timestamps = []
    temperature_values = []
    energy_consumption = []
    active_states = []
    sampling_intervals = []
    
    # Simuliere Raumtemperatur (leicht schwankend zwischen 25-30°C)
    base_temperature = 25.0
    
    try:
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Simuliere Temperaturänderungen (realistischer Verlauf)
            # Bei längerer Aktivität steigt die Temperatur langsam an
            if not emulator.sleep_mode:
                temperature_drift = min(5.0, elapsed / 60.0 * 5.0)  # Max +5°C nach 60 Sekunden aktiv
            else:
                temperature_drift = max(0.0, (temperature_drift if 'temperature_drift' in locals() else 0) - 0.5)  # Abkühlung im Sleep-Modus
            
            # Simuliere kleine zufällige Schwankungen
            temperature_noise = random.uniform(-0.5, 0.5)
            current_temperature = base_temperature + temperature_drift + temperature_noise
            
            # Aktualisiere die Temperatur im PowerManager
            emulator.power_manager.update_temperature(current_temperature)
            
            # Zeit für eine neue Inferenz?
            if current_time - last_inference_time >= interval_seconds:
                # Simuliere Veränderung des erkannten Zustands
                current_class = class_sequence[inference_count % len(class_sequence)]
                
                # Manipuliere die Klassenerkennung für eine realistischere Simulation
                emulator.last_detection_class = current_class
                
                # Führe die Inferenz durch
                result = emulator.simulate_inference(test_image)
                inference_count += 1
                last_inference_time = current_time
                
                # Aktualisiere den Kontext im PowerManager mit der erkannten Klasse
                emulator.power_manager.update_detection_class(current_class)
                
                # Zeige Ergebnis an
                print(f"[{elapsed:.1f}s] Inferenz #{inference_count}: Klasse {result['class_id']} "
                      f"(Konf: {result['confidence']:.2f}, Zeit: {result['inference_time']*1000:.1f}ms, "
                      f"Temp: {current_temperature:.1f}°C)")
                
                # Bei kritischen Klassen (1 oder 2), empfehle eventuell einen anderen Energiemodus
                if current_class in [1, 2] and emulator.power_manager.mode not in [AdaptiveMode.CONTEXT_AWARE, AdaptiveMode.PERFORMANCE]:
                    recommended_mode = emulator.power_manager.recommend_power_mode()
                    if recommended_mode != emulator.power_manager.mode:
                        print(f"[{elapsed:.1f}s] Kritische Pizzaklasse {current_class} erkannt - "
                              f"Wechsel zu {recommended_mode.value}-Modus empfohlen")
            
            # Zeichne Statistiken auf
            if plot_stats:
                timestamps.append(elapsed)
                temperature_values.append(current_temperature)
                power_stats = emulator.power_manager.get_power_statistics()
                energy_consumption.append(power_stats['energy_consumed_mah'])
                active_states.append(0 if emulator.sleep_mode else 1)
                sampling_intervals.append(power_stats['sampling_interval_s'])
            
            # Zeige regelmäßig Statistiken an
            if current_time - last_stats_time >= stats_interval:
                print_system_stats(emulator)
                last_stats_time = current_time
            
            # Prüfe, ob wir aufwachen sollten (falls im Sleep-Modus)
            if emulator.sleep_mode and emulator.power_manager.should_wake_up():
                emulator.power_manager.wake_up()
                print(f"[{elapsed:.1f}s] System aufgeweckt nach "
                      f"{elapsed - (emulator.power_manager.last_sleep_time - start_time):.1f}s Schlaf")
            
            # Prüfe, ob wir in den Sleep-Modus wechseln sollten
            elif not emulator.sleep_mode and emulator.power_manager.should_enter_sleep():
                emulator.power_manager.enter_sleep_mode()
                print(f"[{elapsed:.1f}s] System geht in Sleep-Modus (Intervall: "
                      f"{emulator.power_manager.get_next_sampling_interval():.1f}s)")
            
            # Kurze Pause zur CPU-Entlastung
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nSimulation durch Benutzer abgebrochen.")
    
    # Zeige abschließende Statistiken
    print("\n=== Simulationsergebnisse ===")
    print(f"Dauer: {time.time() - start_time:.1f} Sekunden")
    print(f"Durchgeführte Inferenzen: {inference_count}")
    print_system_stats(emulator)
    
    # Optional: Plotte die aufgezeichneten Statistiken
    if plot_stats and len(timestamps) > 0:
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Temperatur
        plt.subplot(4, 1, 1)
        plt.plot(timestamps, temperature_values, 'r-')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Temperatur (°C)')
        plt.title('Temperaturverlauf')
        plt.grid(True)
        
        # Plot 2: Energieverbrauch
        plt.subplot(4, 1, 2)
        plt.plot(timestamps, energy_consumption, 'b-')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Verbrauch (mAh)')
        plt.title('Energieverbrauch')
        plt.grid(True)
        
        # Plot 3: Aktivitätszustand
        plt.subplot(4, 1, 3)
        plt.step(timestamps, active_states, 'g-', where='post')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Zustand')
        plt.yticks([0, 1], ['Sleep', 'Aktiv'])
        plt.title('Systemzustand')
        plt.grid(True)
        
        # Plot 4: Abtastintervalle
        plt.subplot(4, 1, 4)
        plt.plot(timestamps, sampling_intervals, 'm-')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Intervall (s)')
        plt.title('Abtastintervalle')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Speichere die Grafik
        output_dir = os.path.join(project_root, "output", "simulations")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"power_simulation_{emulator.power_manager.mode.value}.png")
        plt.savefig(plot_path)
        print(f"\nStatistik-Grafik gespeichert unter: {plot_path}")
        
        plt.close()


def print_system_stats(emulator):
    """Zeigt detaillierte Systemstatistiken an."""
    stats = emulator.get_system_stats()
    power_stats = emulator.power_manager.get_power_statistics()
    
    print("\n----- Systemstatus -----")
    print(f"Betriebszeit: {stats['uptime_seconds']:.1f}s")
    print(f"Aktueller Modus: {'Sleep' if stats['sleep_mode'] else 'Aktiv'}")
    print(f"RAM verwendet: {stats['ram_used_kb']:.1f}KB von {(stats['ram_used_kb'] + stats['ram_free_kb']):.1f}KB")
    print(f"Energiemanagement: {power_stats['mode']}")
    print(f"Temperatur: {power_stats.get('current_temp_c', 25.0):.1f}°C")
    print(f"Aktivitätslevel: {power_stats['activity_level']:.2f}")
    
    # Zeige Pizza-Klasse an, wenn verfügbar
    if 'dominant_pizza_class' in power_stats:
        pizza_class_names = ["Normal", "Fast Fertig", "Fast Verbrannt", "Verbrannt"]
        dominant_class = power_stats['dominant_pizza_class']
        print(f"Vorherrschende Pizza-Klasse: {dominant_class} ({pizza_class_names[dominant_class]})")
    
    print(f"Abtastintervall: {power_stats['sampling_interval_s']:.1f}s")
    print(f"Verbrauchte Energie: {power_stats['energy_consumed_mah']:.2f}mAh")
    print(f"Geschätzte verbleibende Laufzeit: {power_stats['estimated_runtime_hours']:.1f}h "
          f"({power_stats['estimated_runtime_hours']/24:.1f} Tage)")
    print(f"Gesamte Schlafzeit: {stats['total_sleep_time']:.1f}s")
    print(f"Duty-Cycle: {power_stats['duty_cycle']*100:.1f}%")
    print("-----------------------")


def test_various_power_modes():
    """Testet das Energiemanagement in verschiedenen Modi."""
    test_firmware = {
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    
    # Test 1: Performance-Modus (höchste Leistung, höchster Stromverbrauch)
    print("\n\n=== TEST 1: PERFORMANCE-MODUS ===")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.PERFORMANCE)
    emulator.load_firmware(test_firmware)
    simulate_extended_operation(emulator, duration_seconds=30, interval_seconds=2, plot_stats=True)
    
    # Test 2: Energiesparmodus (längere Batterielebensdauer)
    print("\n\n=== TEST 2: ENERGIESPARMODUS ===")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.POWER_SAVE)
    emulator.load_firmware(test_firmware)
    simulate_extended_operation(emulator, duration_seconds=30, interval_seconds=2, plot_stats=True)
    
    # Test 3: Adaptiver Modus (passt sich an Aktivitätsmuster an)
    print("\n\n=== TEST 3: ADAPTIVER MODUS ===")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.ADAPTIVE)
    emulator.load_firmware(test_firmware)
    simulate_extended_operation(emulator, duration_seconds=60, interval_seconds=2, plot_stats=True)
    
    # Test 4: Kontextbasierter Modus (basiert auf Pizza-Klasse)
    print("\n\n=== TEST 4: KONTEXTBASIERTER MODUS ===")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.CONTEXT_AWARE)
    emulator.load_firmware(test_firmware)
    simulate_extended_operation(emulator, duration_seconds=60, interval_seconds=2, plot_stats=True)


def test_temperature_effects():
    """Testet die Auswirkungen verschiedener Temperaturen auf das Energiemanagement."""
    test_firmware = {
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    
    print("\n\n=== TEST: TEMPERATUREFFEKTE ===")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.BALANCED)
    emulator.load_firmware(test_firmware)
    
    # Setze initiale Temperatur
    emulator.power_manager.update_temperature(35.0)
    print("Temperatur auf 35°C gesetzt (simuliert höhere Umgebungstemperatur)")
    
    simulate_extended_operation(emulator, duration_seconds=45, interval_seconds=2, plot_stats=True)


def test_battery_scenarios():
    """Testet das Verhalten bei unterschiedlichen Batteriekapazitäten und -zuständen."""
    test_firmware = {
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    
    # Verschiedene Batteriekapazitäten
    print("\n\n=== TEST: VERSCHIEDENE BATTERIETYPEN ===")
    
    # CR123A (1500 mAh)
    print("\n--- Batterie: CR123A (1500 mAh) ---")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.BALANCED, battery_capacity_mah=1500.0)
    emulator.load_firmware(test_firmware)
    simulate_extended_operation(emulator, duration_seconds=30, interval_seconds=2)
    
    # 18650 Li-Ion (3400 mAh)
    print("\n--- Batterie: 18650 Li-Ion (3400 mAh) ---")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.BALANCED, battery_capacity_mah=3400.0)
    emulator.load_firmware(test_firmware)
    simulate_extended_operation(emulator, duration_seconds=30, interval_seconds=2)
    
    # Batterie mit niedrigem Ladestand (200 mAh übrig)
    print("\n--- Batterie: Niedriger Ladestand (200 mAh übrig) ---")
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.BALANCED, battery_capacity_mah=200.0)
    emulator.load_firmware(test_firmware)
    
    # Führe fünf Inferenzen durch, um zu sehen, wie der PowerManager reagiert
    for i in range(5):
        emulator.simulate_inference(np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8))
        
        # Nach der dritten Inferenz automatisch Modus empfehlen
        if i == 2:
            recommended_mode = emulator.power_manager.recommend_power_mode()
            if recommended_mode != emulator.power_manager.mode:
                print(f"\nAutomatische Modusempfehlung: Wechsel zu {recommended_mode.value}")
                emulator.power_manager.set_mode(recommended_mode)
    
    # Zeige finale Statistiken
    print_system_stats(emulator)


if __name__ == "__main__":
    print("Wählen Sie den Test-Modus:")
    print("1: Verschiedene Power-Modi (Standard)")
    print("2: Temperatureffekte")
    print("3: Batterieszenarien")
    
    try:
        choice = int(input("Auswahl (1-3): ").strip())
    except ValueError:
        choice = 1
    
    if choice == 2:
        test_temperature_effects()
    elif choice == 3:
        test_battery_scenarios()
    else:
        test_various_power_modes()