"""
Intelligentes Energiemanagement für das RP2040-basierte Pizza-Erkennungssystem.
Implementiert adaptive Abtastraten und optimierte Duty-Cycles für verbesserte Batterielebensdauer.
"""

import time
import logging
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, replace
import numpy as np
from collections import deque

from .devices import PowerMode, BatteryStatus
from .constants import (
    MAX_POWER_CONSUMPTION_MW,
    RP2040_CLOCK_SPEED_MHZ
)
from .metrics import PowerProfile

logger = logging.getLogger(__name__)


@dataclass
class PowerUsage:
    """Energieverbrauchsdaten für verschiedene Systemzustände."""
    sleep_mode_ma: float = 0.5       # Strom im Sleep-Modus (mA)
    idle_ma: float = 10.0            # Strom im Leerlauf (mA)
    active_ma: float = 80.0          # Strom bei aktiver CPU (mA)
    camera_active_ma: float = 40.0   # Strom bei aktiver Kamera (mA)
    inference_ma: float = 100.0      # Strom während Inferenz (mA)
    
    def get_total_active_current(self) -> float:
        """Berechnet den Gesamtstrom im aktiven Zustand mit Kamera und Inferenz."""
        return self.active_ma + self.camera_active_ma + self.inference_ma
    
    def scale_for_temperature(self, temperature_c: float) -> 'PowerUsage':
        """
        Passt den Stromverbrauch basierend auf der Temperatur an.
        Höhere Temperaturen führen zu erhöhtem Leckstrom.
        
        Args:
            temperature_c: Temperatur in Grad Celsius
        
        Returns:
            Angepasstes PowerUsage-Objekt
        """
        # Referenztemperatur ist 25°C
        # Der Leckstrom steigt etwa 2% pro °C über der Referenztemperatur
        if temperature_c <= 25.0:
            return self
        
        temp_delta = temperature_c - 25.0
        scaling_factor = 1.0 + (temp_delta * 0.02)
        
        return replace(
            self,
            sleep_mode_ma=self.sleep_mode_ma * scaling_factor,
            idle_ma=self.idle_ma * scaling_factor,
            active_ma=self.active_ma * scaling_factor
        )


class AdaptiveMode(Enum):
    """Betriebsmodi für das adaptive Energiemanagement."""
    PERFORMANCE = "performance"     # Maximale Leistung, höchster Energieverbrauch
    BALANCED = "balanced"           # Ausgewogenes Verhältnis zwischen Leistung und Batterielebensdauer
    POWER_SAVE = "power_save"       # Maximale Batterielebensdauer, reduzierte Leistung
    ULTRA_LOW_POWER = "ultra_low"   # Extrem niedriger Energieverbrauch für kritische Batteriebedingungen
    ADAPTIVE = "adaptive"           # Passt sich automatisch an Erkennungsmuster an
    CONTEXT_AWARE = "context_aware" # Passt sich basierend auf erkannten Pizzatypen an


class PowerManager:
    """
    Intelligentes Energiemanagement für das RP2040-System.
    
    Implementiert:
    - Adaptive Abtastraten basierend auf Erkennungsaktivität
    - Automatische Anpassung des Duty-Cycles für optimale Batterielebensdauer
    - Präzise Batterieüberwachung und -prognose
    - Verschiedene Energiesparmodi für unterschiedliche Anwendungsszenarien
    - Temperaturbasierte Leistungsanpassung
    - Kontext-basiertes Energiemanagement basierend auf erkannten Pizzatypen
    """
    
    def __init__(self, 
                 emulator,  # Referenz auf den RP2040Emulator
                 power_usage: Optional[PowerUsage] = None,
                 battery_capacity_mah: float = 1500.0,  # CR123A Standardkapazität
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
        
        # Initialisiere alle Temperatur- und kontextbezogenen Attribute zuerst
        # um Attributfehler bei der Initialisierung zu vermeiden
        self.current_temperature_c = 25.0  # Startwert: Raumtemperatur
        self.temperature_history = deque(maxlen=60)  # Speichert Temperaturwerte der letzten Stunde
        self.temperature_history.append(self.current_temperature_c)
        
        # Kontextbasierte Parameter
        self.detection_class_history = deque(maxlen=10)  # Speichert die letzten 10 erkannten Klassen
        
        # Erweitertes Tracking für Energieverbrauch
        self.power_consumption_history = deque(maxlen=24)  # Speichert Verbrauch der letzten 24 Stunden
        self.hourly_consumption_mah = 0.0  # Verbrauch in der aktuellen Stunde
        self.last_hour = time.localtime().tm_hour
        
        # Abtastintervalle für verschiedene Modi (in Sekunden)
        self.sampling_intervals = {
            AdaptiveMode.PERFORMANCE: 5.0,       # Alle 5 Sekunden
            AdaptiveMode.BALANCED: 30.0,         # Alle 30 Sekunden
            AdaptiveMode.POWER_SAVE: 60.0,       # Alle 60 Sekunden
            AdaptiveMode.ULTRA_LOW_POWER: 180.0, # Alle 3 Minuten
            AdaptiveMode.ADAPTIVE: 30.0,         # Startet mit 30 Sekunden, passt sich dann an
            AdaptiveMode.CONTEXT_AWARE: 30.0     # Basisintervall, wird je nach Kontext angepasst
        }
        
        # Klassenspezifische Abtastintervalle
        self.class_sampling_intervals = {
            0: 30.0,  # normale Pizza: Standard-Abtastintervall
            1: 15.0,  # Pizza kurz vor fertig: häufigere Überprüfungen
            2: 10.0,  # kritischer Zustand (fast verbrannt): sehr häufige Überprüfungen
            3: 60.0   # verbrannte Pizza: seltenere Überprüfungen
        }
        
        # Aktivitätserkennung für den adaptiven Modus
        self.activity_history = deque(maxlen=10)  # Speichert die letzten 10 Erkennungsergebnisse
        self.activity_threshold = 0.3             # Schwellwert für Aktivitätserkennung (30% Änderung)
        
        # Zeitvariablen
        self.last_wakeup_time = time.time()
        self.last_sleep_time = 0
        self.total_active_time = 0
        self.total_sleep_time = 0
        self.sleep_start_time = 0
        
        # Energiestatistiken
        self.energy_consumed_mah = 0.0
        self.estimated_runtime_hours = 0.0  # Wird später berechnet
        
        # Inferenzmessung
        self.inference_times_ms = deque(maxlen=20)  # Speichert die letzten 20 Inferenzzeiten
        
        # Jetzt berechne die geschätzte Laufzeit
        self.estimated_runtime_hours = self._calculate_estimated_runtime()
        
        logger.info(f"PowerManager initialisiert im {self.mode.value}-Modus")
        logger.info(f"Geschätzte Batterielebensdauer: {self.estimated_runtime_hours:.1f} Stunden")
    
    def set_mode(self, mode: AdaptiveMode) -> None:
        """
        Ändert den Betriebsmodus des Energiemanagements.
        
        Args:
            mode: Neuer Betriebsmodus
        """
        self.mode = mode
        logger.info(f"Energiemanagement-Modus geändert auf: {mode.value}")
        self.estimated_runtime_hours = self._calculate_estimated_runtime()
        logger.info(f"Neue geschätzte Batterielebensdauer: {self.estimated_runtime_hours:.1f} Stunden")
    
    def update_temperature(self, temperature_c: float) -> None:
        """
        Aktualisiert die gemessene Temperatur und passt Energieberechnungen an.
        
        Args:
            temperature_c: Aktuelle Temperatur in Grad Celsius
        """
        self.current_temperature_c = temperature_c
        self.temperature_history.append(temperature_c)
        logger.debug(f"Temperatur aktualisiert: {temperature_c:.1f}°C")
    
    def update_detection_class(self, class_id: int) -> None:
        """
        Aktualisiert die erkannte Pizzaklasse für kontextbasiertes Energiemanagement.
        
        Args:
            class_id: ID der erkannten Pizzaklasse (0-3)
        """
        if class_id not in range(4):
            logger.warning(f"Ungültige Klassen-ID: {class_id}")
            return
        
        self.detection_class_history.append(class_id)
        logger.debug(f"Erkannte Pizzaklasse: {class_id}")
        
        # Bei Wechsel zu kritischen Klassen (1-2) sofort aufwecken und Abtastintervall anpassen
        if class_id in [1, 2] and self.emulator.sleep_mode:
            logger.info(f"Kritische Pizza-Klasse {class_id} erkannt - aufwecken")
            self.wake_up()
    
    def update_activity(self, detection_changed: bool) -> None:
        """
        Aktualisiert die Aktivitätshistorie für den adaptiven Modus.
        
        Args:
            detection_changed: True, wenn sich das Erkennungsergebnis geändert hat
        """
        self.activity_history.append(1 if detection_changed else 0)
    
    def _get_activity_level(self) -> float:
        """
        Berechnet das aktuelle Aktivitätsniveau basierend auf der Aktivitätshistorie.
        
        Returns:
            Aktivitätsniveau zwischen 0.0 und 1.0
        """
        if not self.activity_history:
            return 0.0
        
        # Neuere Aktivitäten werden stärker gewichtet
        weights = np.linspace(0.5, 1.0, len(self.activity_history))
        weighted_activity = np.array(self.activity_history) * weights
        return float(np.sum(weighted_activity) / np.sum(weights))
    
    def _get_dominant_class(self) -> int:
        """
        Ermittelt die vorherrschende Pizzaklasse aus der Historie.
        
        Returns:
            Vorherrschende Klassen-ID (0-3) oder 0, wenn keine Historie existiert
        """
        if not self.detection_class_history:
            return 0
        
        # Zähle die Häufigkeit jeder Klasse
        class_counts = {i: 0 for i in range(4)}
        for cls in self.detection_class_history:
            class_counts[cls] += 1
        
        # Finde die häufigste Klasse
        return max(class_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_adaptive_interval(self) -> float:
        """
        Berechnet das optimale Abtastintervall basierend auf dem Aktivitätsniveau.
        
        Returns:
            Angepasstes Abtastintervall in Sekunden
        """
        activity_level = self._get_activity_level()
        
        # Bei hoher Aktivität: häufigere Abtastung (min 10s)
        # Bei niedriger Aktivität: seltenere Abtastung (max 120s)
        min_interval = 10.0
        max_interval = 120.0
        
        # Exponentielles Abklingverhalten: Je weniger Aktivität, desto länger das Intervall
        # activity_level=1.0 -> min_interval, activity_level=0.0 -> max_interval
        interval = min_interval + (max_interval - min_interval) * (1.0 - activity_level) ** 2
        
        # Zusätzliche Anpassung basierend auf der Temperatur
        avg_temp = sum(self.temperature_history) / max(1, len(self.temperature_history))
        if avg_temp > 30.0:
            # Bei höheren Temperaturen häufiger abtasten (10% reduziert pro 5°C über 30°C)
            temp_factor = max(0.7, 1.0 - (avg_temp - 30.0) / 5.0 * 0.1)
            interval *= temp_factor
        
        return interval
    
    def _calculate_context_aware_interval(self) -> float:
        """
        Berechnet das Abtastintervall basierend auf dem erkannten Pizzatyp.
        
        Returns:
            Kontextbasiertes Abtastintervall in Sekunden
        """
        dominant_class = self._get_dominant_class()
        base_interval = self.class_sampling_intervals[dominant_class]
        
        # Aktivitätsbasierte Anpassung
        activity_factor = 1.0 - (self._get_activity_level() * 0.3)  # Reduziert Intervall bei hoher Aktivität
        
        # Temperaturbasierte Anpassung (wie bei _calculate_adaptive_interval)
        avg_temp = sum(self.temperature_history) / max(1, len(self.temperature_history))
        temp_factor = 1.0
        if avg_temp > 30.0:
            temp_factor = max(0.7, 1.0 - (avg_temp - 30.0) / 5.0 * 0.1)
        
        return base_interval * activity_factor * temp_factor
    
    def get_next_sampling_interval(self) -> float:
        """
        Ermittelt das nächste Abtastintervall basierend auf dem aktuellen Modus.
        
        Returns:
            Abtastintervall in Sekunden
        """
        if self.mode == AdaptiveMode.ADAPTIVE:
            return self._calculate_adaptive_interval()
        elif self.mode == AdaptiveMode.CONTEXT_AWARE:
            return self._calculate_context_aware_interval()
        return self.sampling_intervals[self.mode]
    
    def should_enter_sleep(self) -> bool:
        """
        Entscheidet, ob das System in den Sleep-Modus wechseln sollte.
        
        Returns:
            True, wenn das System in den Sleep-Modus wechseln sollte
        """
        current_time = time.time()
        time_since_last_wakeup = current_time - self.last_wakeup_time
        
        # Prüfe, ob die minimale Aktivzeit erreicht wurde (mindestens 1 Sekunde aktiv)
        if time_since_last_wakeup < 1.0:
            return False
        
        # Im Context-Aware-Modus nicht schlafen, wenn kritische Pizza-Klassen erkannt wurden
        if self.mode == AdaptiveMode.CONTEXT_AWARE and self.detection_class_history:
            dominant_class = self._get_dominant_class()
            if dominant_class in [1, 2]:  # Kritische Zustände
                return False
        
        # Bei hoher Temperatur nicht zu lange schlafen
        if self.current_temperature_c > 35.0 and self.mode in [AdaptiveMode.POWER_SAVE, AdaptiveMode.ULTRA_LOW_POWER]:
            # Prüfe zusätzlich alle 30 Sekunden unabhängig vom Modus
            time_since_last_sleep = current_time - self.last_sleep_time
            if time_since_last_sleep > 30.0:
                logger.debug(f"Verhindere längeren Sleep-Modus bei hoher Temperatur ({self.current_temperature_c:.1f}°C)")
                return False
        
        return True
    
    def should_wake_up(self) -> bool:
        """
        Entscheidet, ob das System aufgeweckt werden sollte.
        
        Returns:
            True, wenn das System aufgeweckt werden sollte
        """
        if not self.emulator.sleep_mode:
            return False
            
        current_time = time.time()
        sleep_duration = current_time - self.sleep_start_time
        sampling_interval = self.get_next_sampling_interval()
        
        # Aufwecken wenn das Abtastintervall erreicht wurde
        if sleep_duration >= sampling_interval:
            return True
        
        # Bei hohen Temperaturen früher aufwecken
        if self.current_temperature_c > 35.0 and sleep_duration >= sampling_interval * 0.7:
            logger.debug(f"Früheres Aufwecken bei hoher Temperatur ({self.current_temperature_c:.1f}°C)")
            return True
        
        return False
    
    def enter_sleep_mode(self) -> None:
        """
        Versetzt das System in den Sleep-Modus.
        """
        if not self.emulator.sleep_mode:
            logger.debug("System geht in Sleep-Modus")
            self.sleep_start_time = time.time()
            self.last_sleep_time = self.sleep_start_time
            self.emulator.enter_sleep_mode()
    
    def wake_up(self) -> None:
        """
        Weckt das System aus dem Sleep-Modus auf.
        """
        if self.emulator.sleep_mode:
            current_time = time.time()
            sleep_duration = current_time - self.sleep_start_time
            
            logger.debug(f"System wird aufgeweckt nach {sleep_duration:.2f}s Schlaf")
            self.emulator.wake_up()
            self.last_wakeup_time = current_time
            self.total_sleep_time += sleep_duration
    
    def update_energy_consumption(self, duration_seconds: float, is_active: bool) -> None:
        """
        Aktualisiert die Energieverbrauchsstatistiken.
        
        Args:
            duration_seconds: Zeitdauer in Sekunden
            is_active: True, wenn das System aktiv war, False wenn im Sleep-Modus
        """
        # Berechne den verbrauchten Strom basierend auf dem Zustand und der Temperatur
        adjusted_power_usage = self.power_usage.scale_for_temperature(self.current_temperature_c)
        current_ma = (adjusted_power_usage.get_total_active_current() if is_active 
                     else adjusted_power_usage.sleep_mode_ma)
        
        # Berechne den Energieverbrauch in mAh
        consumed_mah = (current_ma * duration_seconds) / 3600.0
        self.energy_consumed_mah += consumed_mah
        
        # Aktualisiere stündliche Statistiken
        self.hourly_consumption_mah += consumed_mah
        current_hour = time.localtime().tm_hour
        if current_hour != self.last_hour:
            self.power_consumption_history.append(self.hourly_consumption_mah)
            self.hourly_consumption_mah = 0.0
            self.last_hour = current_hour
        
        # Aktualisiere die verbleibende Laufzeit
        self.estimated_runtime_hours = self._calculate_estimated_runtime()
    
    def _calculate_estimated_runtime(self) -> float:
        """
        Berechnet die geschätzte verbleibende Laufzeit basierend auf dem aktuellen Energieverbrauchsmuster.
        
        Returns:
            Geschätzte Laufzeit in Stunden
        """
        # Bestimme das Duty-Cycle-Verhältnis basierend auf dem Modus
        duty_cycles = {
            AdaptiveMode.PERFORMANCE: 0.5,      # 50% aktiv
            AdaptiveMode.BALANCED: 0.1,         # 10% aktiv
            AdaptiveMode.POWER_SAVE: 0.05,      # 5% aktiv
            AdaptiveMode.ULTRA_LOW_POWER: 0.02, # 2% aktiv
            AdaptiveMode.ADAPTIVE: 0.1,         # Startwert, wird angepasst
            AdaptiveMode.CONTEXT_AWARE: 0.1     # Startwert, wird angepasst
        }
        
        # Berechne tatsächlichen Duty-Cycle aus der Nutzungshistorie
        if self.mode in [AdaptiveMode.ADAPTIVE, AdaptiveMode.CONTEXT_AWARE]:
            total_time = self.total_active_time + self.total_sleep_time
            if total_time > 0:
                duty_cycle = self.total_active_time / total_time
            else:
                duty_cycle = duty_cycles[self.mode]
                
            # Bei adaptiven Modi sorgen wir für eine realistischere Berechnung
            # Begrenze den Duty-Cycle auf sinnvolle Werte
            duty_cycle = max(0.01, min(duty_cycle, 0.5))
        else:
            duty_cycle = duty_cycles[self.mode]
        
        # Berechne durchschnittlichen Stromverbrauch basierend auf Duty-Cycle und Temperatur
        adjusted_power_usage = self.power_usage.scale_for_temperature(self.current_temperature_c)
        avg_current_ma = (
            duty_cycle * adjusted_power_usage.get_total_active_current() + 
            (1 - duty_cycle) * adjusted_power_usage.sleep_mode_ma
        )
        
        # Berechne verbleibende Kapazität
        remaining_capacity = self.battery_capacity_mah - self.energy_consumed_mah
        
        # Berechne geschätzte Laufzeit
        if avg_current_ma > 0:
            # Realistischere Berechnung mit Sicherheitsmargin (90%)
            return (remaining_capacity / avg_current_ma) * 0.9
        
        return float('inf')  # Unendlich bei 0 Stromverbrauch (theoretisch)
    
    def add_inference_time(self, inference_time_ms: float) -> None:
        """
        Fügt eine neue Inferenzzeit zur Historie hinzu.
        
        Args:
            inference_time_ms: Inferenzzeit in Millisekunden
        """
        self.inference_times_ms.append(inference_time_ms)
    
    def get_average_inference_time(self) -> float:
        """
        Berechnet die durchschnittliche Inferenzzeit.
        
        Returns:
            Durchschnittliche Inferenzzeit in Millisekunden
        """
        if not self.inference_times_ms:
            return 0.0
        return sum(self.inference_times_ms) / len(self.inference_times_ms)
    
    def get_optimal_duty_cycle(self) -> float:
        """
        Berechnet den optimalen Duty-Cycle basierend auf Inferenzzeit und Ziellaufzeit.
        
        Returns:
            Optimaler Duty-Cycle (0.0 bis 1.0)
        """
        avg_inference_ms = self.get_average_inference_time()
        if avg_inference_ms <= 0:
            return 0.05  # Standardwert 5%
        
        # Berechne, wie viel Zeit pro Stunde für Inferenzen benötigt wird
        # basierend auf dem aktuellen Sampling-Intervall
        interval_s = self.get_next_sampling_interval()
        inferences_per_hour = 3600.0 / interval_s
        active_time_per_hour_ms = inferences_per_hour * avg_inference_ms
        
        # Umrechnung in Duty-Cycle (aktive Zeit / Gesamtzeit)
        duty_cycle = active_time_per_hour_ms / (3600.0 * 1000.0)
        
        # Begrenze auf sinnvollen Bereich
        return max(0.01, min(duty_cycle, 0.5))  # 1% bis 50%
    
    def get_temperature_statistics(self) -> Dict[str, float]:
        """
        Liefert Statistiken über die Temperaturhistorie.
        
        Returns:
            Dictionary mit Temperaturstatistiken
        """
        if not self.temperature_history:
            return {
                'current_temp_c': self.current_temperature_c,
                'avg_temp_c': self.current_temperature_c,
                'min_temp_c': self.current_temperature_c,
                'max_temp_c': self.current_temperature_c
            }
        
        temp_array = np.array(list(self.temperature_history))
        return {
            'current_temp_c': self.current_temperature_c,
            'avg_temp_c': float(np.mean(temp_array)),
            'min_temp_c': float(np.min(temp_array)),
            'max_temp_c': float(np.max(temp_array))
        }
    
    def get_power_statistics(self) -> Dict[str, Any]:
        """
        Liefert detaillierte Statistiken über den Energieverbrauch.
        
        Returns:
            Dictionary mit Energiestatistiken
        """
        stats = {
            'mode': self.mode.value,
            'energy_consumed_mah': self.energy_consumed_mah,
            'battery_capacity_mah': self.battery_capacity_mah,
            'estimated_runtime_hours': self.estimated_runtime_hours,
            'duty_cycle': self.get_optimal_duty_cycle(),
            'sampling_interval_s': self.get_next_sampling_interval(),
            'total_active_time_s': self.total_active_time,
            'total_sleep_time_s': self.total_sleep_time,
            'activity_level': self._get_activity_level(),
            'avg_inference_time_ms': self.get_average_inference_time()
        }
        
        # Füge Temperaturstatistiken hinzu
        stats.update(self.get_temperature_statistics())
        
        # Füge kontextbasierte Statistiken hinzu, wenn verfügbar
        if self.detection_class_history:
            stats['dominant_pizza_class'] = self._get_dominant_class()
        
        return stats
    
    def recommend_power_mode(self) -> AdaptiveMode:
        """
        Empfiehlt einen optimalen Energiemodus basierend auf aktuellen Systemdaten.
        
        Returns:
            Empfohlener AdaptiveMode
        """
        # Bei kritischem Batteriestand immer ULTRA_LOW_POWER verwenden
        remaining_capacity_percent = (self.battery_capacity_mah - self.energy_consumed_mah) / self.battery_capacity_mah * 100
        if remaining_capacity_percent < 10:
            return AdaptiveMode.ULTRA_LOW_POWER
        
        # Bei hoher Temperatur reduziere Leistung
        if self.current_temperature_c > 35:
            return AdaptiveMode.POWER_SAVE
        
        # Bei stark variierenden Pizza-Klassen verwende kontextbasiertes Energiemanagement
        if len(self.detection_class_history) >= 5 and len(set(self.detection_class_history)) >= 3:
            return AdaptiveMode.CONTEXT_AWARE
        
        # Bei hoher Aktivität wähle zwischen Performance und Adaptive
        activity_level = self._get_activity_level()
        if activity_level > 0.7:
            return AdaptiveMode.ADAPTIVE
        
        # Standardfall: ausgeglichener Modus
        return AdaptiveMode.BALANCED