"""
Vereinfachte PowerManager-Implementierung für die Temperatur-Messungs-Tests.
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class AdaptiveMode(Enum):
    """Betriebsmodi für das adaptive Energiemanagement."""
    PERFORMANCE = "performance"     # Maximale Leistung, höchster Energieverbrauch
    BALANCED = "balanced"           # Ausgewogenes Verhältnis zwischen Leistung und Batterielebensdauer
    POWER_SAVE = "power_save"       # Maximale Batterielebensdauer, reduzierte Leistung
    ULTRA_LOW_POWER = "ultra_low"   # Extrem niedriger Energieverbrauch für kritische Batteriebedingungen
    ADAPTIVE = "adaptive"           # Passt sich automatisch an Erkennungsmuster an
    CONTEXT_AWARE = "context_aware" # Passt sich basierend auf erkannten Pizzatypen an

@dataclass
class PowerUsage:
    """Energieverbrauchsdaten für verschiedene Systemzustände."""
    sleep_mode_ma: float = 0.5       # Strom im Sleep-Modus (mA)
    idle_ma: float = 10.0            # Strom im Leerlauf (mA)
    active_ma: float = 80.0          # Strom bei aktiver CPU (mA)
    camera_active_ma: float = 40.0   # Strom bei aktiver Kamera (mA)
    inference_ma: float = 100.0      # Strom während Inferenz (mA)

class PowerManager:
    """
    Vereinfachter PowerManager für Temperatur-Tests.
    Diese Version implementiert nur die minimalen Funktionen,
    die für die Temperaturmessung benötigt werden.
    """
    
    def __init__(self, 
                 emulator,
                 power_usage: Optional[PowerUsage] = None,
                 battery_capacity_mah: float = 1500.0,
                 adaptive_mode: AdaptiveMode = AdaptiveMode.BALANCED):
        """
        Initialisiert den PowerManager.
        
        Args:
            emulator: Referenz auf den RP2040Emulator
            power_usage: Energieverbrauchsdaten für verschiedene Systemzustände
            battery_capacity_mah: Batteriekapazität in mAh
            adaptive_mode: Betriebsmodus für das Energiemanagement
        """
        self.emulator = emulator
        self.power_usage = power_usage or PowerUsage()
        self.battery_capacity_mah = battery_capacity_mah
        self.mode = adaptive_mode
        
        # Initialisiere Basiszustände
        self.current_temperature_c = 25.0
        self.energy_consumed_mah = 0.0
        self.estimated_runtime_hours = 0.0
        self.last_activity_time = time.time()
        self.sleep_start_time = 0
        self.last_wakeup_time = 0
        self.total_sleep_time = 0
        self.activity_level = 0.5  # Mittlere Aktivität
        
        # Inferenz-Zeiten
        self.inference_times_ms = []
        self.avg_inference_time_ms = 0.0
        
        # Berechne geschätzte Laufzeit
        self._calculate_estimated_runtime()
    
    def update_temperature(self, temperature_c: float) -> None:
        """
        Aktualisiert die gemessene Temperatur.
        
        Args:
            temperature_c: Aktuelle Temperatur in Grad Celsius
        """
        self.current_temperature_c = temperature_c
    
    def update_activity(self, activity_changed: bool) -> None:
        """
        Aktualisiert das Aktivitätsniveau basierend auf Änderungen.
        
        Args:
            activity_changed: True wenn sich die Aktivität geändert hat
        """
        if activity_changed:
            # Erhöhe Aktivitätsniveau bei Änderungen
            self.activity_level = min(1.0, self.activity_level + 0.1)
        else:
            # Verringere Aktivitätsniveau bei Konstanz
            self.activity_level = max(0.1, self.activity_level - 0.05)
        
        self.last_activity_time = time.time()
    
    def update_energy_consumption(self, duration_s: float, active: bool) -> None:
        """
        Aktualisiert den Energieverbrauch basierend auf Dauer und Aktivität.
        
        Args:
            duration_s: Dauer in Sekunden
            active: True wenn aktiver Modus, False wenn Sleep-Modus
        """
        if active:
            current_ma = self.power_usage.active_ma
        else:
            current_ma = self.power_usage.sleep_mode_ma
        
        # Berechne verbrauchte Energie in mAh
        energy_mah = current_ma * (duration_s / 3600)
        self.energy_consumed_mah += energy_mah
        
        # Aktualisiere geschätzte Laufzeit
        self._calculate_estimated_runtime()
    
    def add_inference_time(self, time_ms: float) -> None:
        """
        Fügt eine neue Inferenzzeit hinzu und aktualisiert den Durchschnitt.
        
        Args:
            time_ms: Inferenzzeit in Millisekunden
        """
        self.inference_times_ms.append(time_ms)
        
        # Begrenze die Anzahl gespeicherter Werte
        if len(self.inference_times_ms) > 20:
            self.inference_times_ms.pop(0)
        
        # Aktualisiere Durchschnitt
        self.avg_inference_time_ms = sum(self.inference_times_ms) / len(self.inference_times_ms)
    
    def enter_sleep_mode(self) -> None:
        """Setzt den PowerManager in den Sleep-Modus."""
        self.sleep_start_time = time.time()
        # Call the emulator's enter_sleep_mode method to actually enter sleep
        self.emulator.enter_sleep_mode()
    
    def should_enter_sleep(self) -> bool:
        """
        Entscheidet, ob in den Sleep-Modus gewechselt werden soll.
        
        Returns:
            True wenn Sleep-Modus empfohlen wird
        """
        # Bei niedriger Aktivität empfehle Sleep-Modus
        return self.activity_level < 0.3
    
    def wake_up(self) -> None:
        """Weckt das System aus dem Sleep-Modus auf."""
        if hasattr(self.emulator, 'sleep_mode') and self.emulator.sleep_mode:
            sleep_duration = time.time() - self.sleep_start_time
            self.total_sleep_time += sleep_duration
            self.last_wakeup_time = time.time()
            
            # Aktualisiere Energieverbrauch für Sleep-Phase
            self.update_energy_consumption(sleep_duration, False)
            
            # Rufe die Emulator wake_up Methode auf
            self.emulator.wake_up()
    
    def should_wake_up(self) -> bool:
        """
        Entscheidet, ob das System aufgeweckt werden sollte.
        
        Returns:
            True, wenn das System aufgeweckt werden sollte
        """
        if not hasattr(self.emulator, 'sleep_mode') or not self.emulator.sleep_mode:
            return False
            
        # Einfache Logik: wecke nach einer bestimmten Zeit auf
        if hasattr(self, 'sleep_start_time'):
            sleep_duration = time.time() - self.sleep_start_time
            return sleep_duration >= 1.0  # Wake up after 1 second
        
        return False

    def _calculate_estimated_runtime(self) -> float:
        """
        Berechnet die geschätzte Batterielebensdauer.
        
        Returns:
            Geschätzte Laufzeit in Stunden
        """
        # Sehr vereinfachte Berechnung für Test-Zwecke
        if self.energy_consumed_mah > 0:
            avg_current = self.energy_consumed_mah / ((time.time() - self.last_activity_time + 1) / 3600)
            self.estimated_runtime_hours = (self.battery_capacity_mah - self.energy_consumed_mah) / max(1.0, avg_current)
        else:
            # Grobe Schätzung basierend auf Aktivitätsniveau
            avg_current = self.power_usage.sleep_mode_ma * 0.7 + self.power_usage.active_ma * 0.3
            self.estimated_runtime_hours = self.battery_capacity_mah / avg_current
        
        return self.estimated_runtime_hours
    
    def get_power_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken zum Energieverbrauch zurück.
        
        Returns:
            Dictionary mit Power-Statistiken
        """
        return {
            'mode': self.mode.value,
            'battery_capacity_mah': self.battery_capacity_mah,
            'energy_consumed_mah': self.energy_consumed_mah,
            'estimated_runtime_hours': self.estimated_runtime_hours,
            'avg_inference_time_ms': self.avg_inference_time_ms if self.inference_times_ms else 0,
            'activity_level': self.activity_level,
            'duty_cycle': 0.3,  # Fester Wert für Tests
            'sampling_interval_s': 30.0  # Fester Wert für Tests
        }
    
    def get_battery_voltage_mv(self) -> int:
        """
        Gibt die simulierte Batteriespannung in Millivolt zurück.
        Die Spannung sinkt linear mit dem Energieverbrauch von 3.0V (voll) bis 2.0V (leer).
        """
        FULL_MV = 3000
        CRITICAL_MV = 2000
        # Linearer Abfall je nach verbrauchter Kapazität
        used_fraction = min(1.0, self.energy_consumed_mah / self.battery_capacity_mah)
        voltage_mv = int(FULL_MV - (FULL_MV - CRITICAL_MV) * used_fraction)
        return voltage_mv
    
    def set_adaptive_mode(self, mode: AdaptiveMode) -> None:
        """
        Set the adaptive power management mode.
        
        Args:
            mode: New adaptive mode
        """
        self.mode = mode
        # Adjust parameters based on mode
        if mode == AdaptiveMode.PERFORMANCE:
            self.activity_level = 1.0
        elif mode == AdaptiveMode.POWER_SAVE:
            self.activity_level = 0.3
        elif mode == AdaptiveMode.ULTRA_LOW_POWER:
            self.activity_level = 0.1
        else:  # BALANCED, ADAPTIVE, CONTEXT_AWARE
            self.activity_level = 0.6
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive power management statistics.
        
        Returns:
            Dictionary with power statistics including current consumption
        """
        # Calculate current power consumption based on mode
        base_power = {
            AdaptiveMode.PERFORMANCE: 150.0,
            AdaptiveMode.BALANCED: 100.0, 
            AdaptiveMode.POWER_SAVE: 50.0,
            AdaptiveMode.ULTRA_LOW_POWER: 20.0,
            AdaptiveMode.ADAPTIVE: 80.0,
            AdaptiveMode.CONTEXT_AWARE: 90.0
        }.get(self.mode, 100.0)
        
        current_power_mw = base_power * (0.5 + 0.5 * self.activity_level)
        
        stats = self.get_power_statistics()
        stats.update({
            'current_power_mw': current_power_mw,
            'current_temperature_c': self.current_temperature_c,
            'last_activity_time': self.last_activity_time,
            'adaptive_mode': self.mode.value
        })
        
        return stats
