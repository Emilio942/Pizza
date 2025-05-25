"""
Emulation der Spannungsmessung für den RP2040.
Unterstützt sowohl die interne VDD-Spannungsmessung des RP2040 als auch
externe Spannungsmessungen über GPIO-Pins für Batteriespannungen.
"""

import time
import random
import logging
import math
from enum import Enum
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class VoltageSensorType(Enum):
    """Typen von Spannungssensoren"""
    INTERNAL = "internal"     # Interner ADC des RP2040 für VDD-Messung
    BATTERY = "battery"       # Externer Sensor für Batteriespannung
    EXTERNAL = "external"     # Generischer externer Spannungssensor

class VoltageSensor:
    """
    Emuliert einen Spannungssensor, entweder den internen VDD-Sensor des RP2040
    oder einen externen Sensor für Batterie- oder andere Spannungen.
    """
    
    def __init__(self, sensor_type: VoltageSensorType = VoltageSensorType.INTERNAL, 
                 accuracy_mv: float = 10.0, update_interval: float = 1.0,
                 noise_level_mv: float = 2.0):
        """
        Initialisiert den emulierten Spannungssensor.
        
        Args:
            sensor_type: Typ des zu emulierenden Sensors
            accuracy_mv: Genauigkeit des Sensors in mV
            update_interval: Minimales Intervall zwischen Messungen in Sekunden
            noise_level_mv: Stärke des zufälligen Rauschens in den Messwerten in mV
        """
        self.sensor_type = sensor_type
        self.accuracy_mv = accuracy_mv
        self.update_interval = update_interval
        self.noise_level_mv = noise_level_mv
        
        # Basisspannung und Variablen für die Simulation
        if sensor_type == VoltageSensorType.INTERNAL:
            self.base_voltage_mv = 3300.0  # 3.3V für VDD
        elif sensor_type == VoltageSensorType.BATTERY:
            self.base_voltage_mv = 3700.0  # 3.7V für eine typische LiPo-Batterie
        else:  # VoltageSensorType.EXTERNAL
            self.base_voltage_mv = 5000.0  # 5V für externe Versorgung
        
        self.current_voltage_mv = self.base_voltage_mv
        self.last_update_time = time.time()
        
        # Emulierte Sensorstatus-Variablen
        self.initialized = False
        self.last_read_time = 0
        self.readings_count = 0
        self.reading_history = []  # Speichert bis zu 100 Messungen
        
        # Simuliere Batterieverhalten für den Batteriesensor
        if sensor_type == VoltageSensorType.BATTERY:
            # Simuliere eine Entladekurve für die Batterie
            self.discharge_rate_mv_per_hour = 5.0  # 5mV pro Stunde Spannungsabfall
            self.battery_capacity_percent = 100.0
        
        logger.info(
            f"Spannungssensor ({sensor_type.value}) emuliert: "
            f"Genauigkeit ±{accuracy_mv:.1f}mV, Update-Intervall {update_interval:.1f}s"
        )
    
    def initialize(self) -> bool:
        """
        Initialisiert den Sensor.
        
        Returns:
            True wenn erfolgreich initialisiert, sonst False
        """
        # Simuliere Initialisierungszeit
        if self.sensor_type == VoltageSensorType.INTERNAL:
            init_time = 0.01  # 10ms für internen Sensor
        else:
            init_time = 0.05  # 50ms für externe Sensoren
        
        time.sleep(init_time)
        
        self.initialized = True
        logger.info(f"Spannungssensor ({self.sensor_type.value}) initialisiert")
        return True
    
    def read_voltage(self) -> float:
        """
        Liest die aktuelle Spannung vom Sensor.
        
        Returns:
            Spannung in Millivolt
        
        Raises:
            RuntimeError: Wenn der Sensor nicht initialisiert wurde
        """
        if not self.initialized:
            error_msg = f"Spannungssensor ({self.sensor_type.value}) nicht initialisiert"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Überprüfe, ob genügend Zeit seit der letzten Messung vergangen ist
        current_time = time.time()
        time_since_last_read = current_time - self.last_read_time
        
        if time_since_last_read < self.update_interval:
            # Gib den letzten Messwert zurück, wenn nicht genug Zeit vergangen ist
            return self.current_voltage_mv
        
        # Aktualisiere die Batteriespannung basierend auf der vergangenen Zeit
        if self.sensor_type == VoltageSensorType.BATTERY:
            time_since_last_update = current_time - self.last_update_time
            hours_passed = time_since_last_update / 3600.0
            
            # Simuliere Batterieverbrauch
            voltage_drop = hours_passed * self.discharge_rate_mv_per_hour
            self.base_voltage_mv = max(3000.0, self.base_voltage_mv - voltage_drop)  # Nicht unter 3.0V fallen
            
            # Aktualisiere den Batterieladezustand basierend auf der Spannung
            # Einfache lineare Mapping zwischen 3.0V (0%) und 4.2V (100%)
            self.battery_capacity_percent = min(100.0, max(0.0, 
                                              (self.base_voltage_mv - 3000.0) / (4200.0 - 3000.0) * 100.0))
        
        # Berechne die aktuelle Spannung mit realistischem Rauschen
        noise = random.uniform(-self.noise_level_mv, self.noise_level_mv)
        measured_voltage_mv = self.base_voltage_mv + noise
        
        # Runde auf die Genauigkeit des Sensors
        measured_voltage_mv = round(measured_voltage_mv / self.accuracy_mv) * self.accuracy_mv
        
        # Füge Wert zur Historie hinzu und begrenze die Größe auf 100 Einträge
        self.reading_history.append(measured_voltage_mv)
        if len(self.reading_history) > 100:
            self.reading_history.pop(0)
        
        # Aktualisiere Statusinformationen
        self.last_read_time = current_time
        self.last_update_time = current_time
        self.readings_count += 1
        self.current_voltage_mv = measured_voltage_mv
        
        return measured_voltage_mv
    
    def get_battery_percentage(self) -> float:
        """
        Gibt den aktuellen Batterieladezustand in Prozent zurück (nur für Batteriesensor).
        
        Returns:
            Batterieladezustand in Prozent oder -1, wenn kein Batteriesensor
        """
        if self.sensor_type != VoltageSensorType.BATTERY:
            return -1.0
        
        # Aktualisiere den Wert durch eine Messung
        self.read_voltage()
        return self.battery_capacity_percent
    
    def get_status(self) -> Dict:
        """
        Gibt Statusinformationen zum Sensor zurück.
        
        Returns:
            Dictionary mit Statusinformationen
        """
        return {
            "type": self.sensor_type.value,
            "initialized": self.initialized,
            "readings_count": self.readings_count,
            "current_value_mv": self.current_voltage_mv,
            "battery_percentage": self.get_battery_percentage() if self.sensor_type == VoltageSensorType.BATTERY else None,
            "last_update": self.last_update_time,
            "accuracy_mv": self.accuracy_mv
        }