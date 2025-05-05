"""
Test für den PowerManager und die energieoptimierte Emulation.
"""

import os
import sys
import time
import unittest
import pytest
import numpy as np
from pathlib import Path

# Füge das Stammverzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.emulator import RP2040Emulator
from src.power_manager import PowerManager, PowerUsage, AdaptiveMode


class TestPowerManager(unittest.TestCase):
    """Testet die PowerManager-Klasse und die energieoptimierte Emulation."""
    
    def setUp(self):
        """Initialisiert die Testumgebung."""
        # Erstelle einen Emulator im Balanced-Modus für Tests
        self.emulator = RP2040Emulator(
            battery_capacity_mah=1500.0,  # CR123A
            adaptive_mode=AdaptiveMode.BALANCED
        )
        
        # Lade eine Test-Firmware
        self.test_firmware = {
            'path': 'test.bin',
            'total_size_bytes': 100 * 1024,
            'model_size_bytes': 50 * 1024,
            'ram_usage_bytes': 40 * 1024,
            'model_input_size': (48, 48)
        }
        self.emulator.load_firmware(self.test_firmware)
    
    def test_power_manager_initialization(self):
        """Testet die Initialisierung des PowerManagers."""
        # Überprüfe, ob der PowerManager korrekt initialisiert wurde
        self.assertIsNotNone(self.emulator.power_manager)
        self.assertEqual(self.emulator.power_manager.mode, AdaptiveMode.BALANCED)
        self.assertEqual(self.emulator.power_manager.battery_capacity_mah, 1500.0)
        
        # Überprüfe die Standardwerte
        self.assertFalse(self.emulator.sleep_mode)
        self.assertEqual(self.emulator.power_manager.energy_consumed_mah, 0.0)
        self.assertGreater(self.emulator.power_manager.estimated_runtime_hours, 0)
    
    def test_adaptive_interval_calculation(self):
        """Testet die Berechnung der adaptiven Abtastintervalle."""
        # Initialer Zustand: keine Aktivität
        self.assertEqual(len(self.emulator.power_manager.activity_history), 0)
        initial_interval = self.emulator.power_manager.get_next_sampling_interval()
        self.assertEqual(initial_interval, 30.0)  # Standardwert für Balanced-Modus
        
        # Wechsle in den adaptiven Modus
        self.emulator.power_manager.set_mode(AdaptiveMode.ADAPTIVE)
        
        # Simuliere keine Aktivität
        for _ in range(5):
            self.emulator.power_manager.update_activity(False)
        
        # Bei niedriger Aktivität sollte das Intervall länger sein
        inactive_interval = self.emulator.power_manager._calculate_adaptive_interval()
        self.assertGreater(inactive_interval, 80.0)  # Länger als 80 Sekunden
        
        # Simuliere hohe Aktivität
        for _ in range(8):
            self.emulator.power_manager.update_activity(True)
        
        # Bei hoher Aktivität sollte das Intervall kürzer sein
        active_interval = self.emulator.power_manager._calculate_adaptive_interval()
        self.assertLess(active_interval, 40.0)  # Kürzer als 40 Sekunden
    
    def test_energy_consumption_tracking(self):
        """Testet die Verfolgung des Energieverbrauchs."""
        # Initialer Verbrauch sollte 0 sein
        self.assertEqual(self.emulator.power_manager.energy_consumed_mah, 0.0)
        
        # Simuliere aktive Phase
        active_duration = 10.0  # 10 Sekunden
        self.emulator.power_manager.update_energy_consumption(active_duration, True)
        
        # Verbrauch sollte jetzt größer als 0 sein
        self.assertGreater(self.emulator.power_manager.energy_consumed_mah, 0.0)
        
        # Berechne erwarteten Verbrauch
        expected_consumption = (self.emulator.power_usage.get_total_active_current() * active_duration) / 3600.0
        self.assertAlmostEqual(
            self.emulator.power_manager.energy_consumed_mah,
            expected_consumption,
            delta=0.01
        )
        
        # Simuliere Sleep-Phase
        sleep_duration = 60.0  # 60 Sekunden
        previous_consumption = self.emulator.power_manager.energy_consumed_mah
        self.emulator.power_manager.update_energy_consumption(sleep_duration, False)
        
        # Verbrauch im Sleep-Modus sollte deutlich niedriger sein
        sleep_consumption = self.emulator.power_manager.energy_consumed_mah - previous_consumption
        self.assertLess(sleep_consumption, expected_consumption * 0.1)  # Weniger als 10% des aktiven Verbrauchs
    
    def test_sleep_wake_cycle(self):
        """Testet den Sleep-Wake-Zyklus."""
        # System sollte initial im aktiven Zustand sein
        self.assertFalse(self.emulator.sleep_mode)
        
        # Speichere den ursprünglichen RAM-Wert (ohne System-Overhead)
        initial_ram_total = self.emulator.get_ram_usage()
        initial_ram = initial_ram_total - self.emulator.system_ram_overhead
        
        # Gehe in den Sleep-Modus
        self.emulator.power_manager.enter_sleep_mode()
        self.assertTrue(self.emulator.sleep_mode)
        
        # RAM-Nutzung sollte im Sleep-Modus reduziert sein
        sleep_ram = self.emulator.get_ram_usage() - self.emulator.system_ram_overhead
        self.assertLess(sleep_ram, initial_ram)
        
        # Zeitversatz, um eine Schlafperiode zu simulieren
        time.sleep(0.1)
        
        # Wecke das System auf
        self.emulator.power_manager.wake_up()
        self.assertFalse(self.emulator.sleep_mode)
        
        # RAM-Nutzung sollte wieder auf dem ursprünglichen Niveau sein
        wake_ram_total = self.emulator.get_ram_usage()
        wake_ram = wake_ram_total - self.emulator.system_ram_overhead
        self.assertEqual(wake_ram, initial_ram)
        
        # Schlafzeit sollte aufgezeichnet worden sein
        self.assertGreater(self.emulator.power_manager.total_sleep_time, 0)
    
    def test_inference_with_power_management(self):
        """Testet die Inferenz mit aktiviertem Energiemanagement."""
        # Erstelle ein simuliertes Bild
        test_image = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        
        # Führe eine Inferenz aus
        result = self.emulator.simulate_inference(test_image)
        
        # Überprüfe die Ergebnisse
        self.assertTrue(result['success'])
        self.assertGreater(result['inference_time'], 0)
        
        # Überprüfe, ob PowerManager-Statistiken vorhanden sind
        self.assertIn('power_stats', result)
        self.assertIn('mode', result['power_stats'])
        self.assertIn('estimated_runtime_hours', result['power_stats'])
        
        # Überprüfe, ob die Inferenzzeit aufgezeichnet wurde
        self.assertEqual(len(self.emulator.power_manager.inference_times_ms), 1)
        
        # Simuliere mehrere Inferenzen mit unterschiedlichen Ergebnissen
        for _ in range(5):
            test_image = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
            result = self.emulator.simulate_inference(test_image)
        
        # Überprüfe, ob die Aktivitätshistorie aufgebaut wurde
        self.assertGreater(len(self.emulator.power_manager.activity_history), 0)
        
        # Prüfe die Systemstatistiken
        stats = self.emulator.get_system_stats()
        self.assertIn('energy_management_mode', stats)
        self.assertIn('estimated_runtime_hours', stats)
        self.assertIn('duty_cycle', stats)
    
    def test_different_power_modes(self):
        """Testet verschiedene Energiemodi."""
        modes = [
            AdaptiveMode.PERFORMANCE,
            AdaptiveMode.BALANCED,
            AdaptiveMode.POWER_SAVE,
            AdaptiveMode.ULTRA_LOW_POWER,
            AdaptiveMode.ADAPTIVE
        ]
        
        for mode in modes:
            # Setze den Modus
            self.emulator.power_manager.set_mode(mode)
            self.assertEqual(self.emulator.power_manager.mode, mode)
            
            # Überprüfe die Abtastintervalle
            interval = self.emulator.power_manager.get_next_sampling_interval()
            
            if mode == AdaptiveMode.PERFORMANCE:
                self.assertLess(interval, 10.0)  # Kurze Intervalle im Performance-Modus
            elif mode == AdaptiveMode.ULTRA_LOW_POWER:
                self.assertGreater(interval, 150.0)  # Lange Intervalle im Ultra-Low-Power-Modus
            
            # Überprüfe die geschätzte Laufzeit
            runtime = self.emulator.power_manager.estimated_runtime_hours
            self.assertGreater(runtime, 0)
            
            # Im Ultra-Low-Power-Modus sollte die Laufzeit am längsten sein
            if mode == AdaptiveMode.ULTRA_LOW_POWER:
                ultra_low_runtime = runtime
            elif mode == AdaptiveMode.PERFORMANCE:
                performance_runtime = runtime
            
        # Ultra-Low-Power sollte längere Laufzeit haben als Performance
        self.assertGreater(ultra_low_runtime, performance_runtime)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])