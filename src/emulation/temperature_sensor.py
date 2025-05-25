"""
Emulation der Temperaturmessung für den RP2040.
Unterstützt sowohl den internen ADC-Temperatursensor des RP2040 als auch
externe I2C oder SPI Temperatursensoren.
"""

import time
import random
import logging
import math
from enum import Enum
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Typen von Temperatursensoren"""
    INTERNAL = "internal"  # Interner ADC-Temperatursensor des RP2040
    EXTERNAL_I2C = "i2c"   # Externer Sensor über I2C
    EXTERNAL_SPI = "spi"   # Externer Sensor über SPI

class TemperatureSensor:
    """
    Emuliert einen Temperatursensor, entweder den internen Sensor des RP2040
    oder einen externen Sensor über I2C/SPI.
    """
    
    def __init__(self, sensor_type: SensorType = SensorType.INTERNAL, 
                 accuracy: float = 0.5, update_interval: float = 1.0,
                 noise_level: float = 0.1):
        """
        Initialisiert den emulierten Temperatursensor.
        
        Args:
            sensor_type: Typ des zu emulierenden Sensors
            accuracy: Genauigkeit des Sensors in °C
            update_interval: Minimales Intervall zwischen Messungen in Sekunden
            noise_level: Stärke des zufälligen Rauschens in den Messwerten
        """
        self.sensor_type = sensor_type
        self.accuracy = accuracy
        self.update_interval = update_interval
        self.noise_level = noise_level
        
        # Basistemperatur und Variablen für die Simulation
        self.base_temperature = 25.0  # Startwert: Raumtemperatur
        self.current_temperature = self.base_temperature
        self.last_update_time = time.time()
        
        # Emulierte Sensorstatus-Variablen
        self.initialized = False
        self.last_read_time = 0
        self.readings_count = 0
        self.reading_history = []  # Speichert bis zu 100 Messungen
        
        # Sensorspezifische Attribute
        if sensor_type == SensorType.INTERNAL:
            # Interner Sensor hat etwas schlechtere Genauigkeit und kann nur langsamer lesen
            self.accuracy = max(1.0, self.accuracy)
            self.update_interval = max(2.0, self.update_interval)
            self.address = None
        elif sensor_type == SensorType.EXTERNAL_I2C:
            # Emuliere I2C-Adresse eines typischen Temperatursensors (z.B. LM75)
            self.address = 0x48
        elif sensor_type == SensorType.EXTERNAL_SPI:
            # SPI benötigt keine Adresse, aber Channel Nummer
            self.channel = 0
        
        logger.info(
            f"Temperatursensor ({sensor_type.value}) emuliert: "
            f"Genauigkeit ±{accuracy:.1f}°C, Update-Intervall {update_interval:.1f}s"
        )
    
    def initialize(self) -> bool:
        """
        Initialisiert den Sensor.
        
        Returns:
            True wenn erfolgreich initialisiert, sonst False
        """
        # Simuliere Initialisierungszeit
        if self.sensor_type == SensorType.INTERNAL:
            init_time = 0.01  # 10ms für internen Sensor
        else:
            init_time = 0.05  # 50ms für externe Sensoren
        
        time.sleep(init_time)
        
        self.initialized = True
        logger.info(f"Temperatursensor ({self.sensor_type.value}) initialisiert")
        return True
    
    def read_temperature(self) -> float:
        """
        Liest die aktuelle Temperatur vom Sensor.
        
        Returns:
            Temperatur in Grad Celsius
        
        Raises:
            RuntimeError: Wenn der Sensor nicht initialisiert wurde
        """
        if not self.initialized:
            raise RuntimeError("Temperatursensor nicht initialisiert")
        
        # Prüfe, ob genügend Zeit seit der letzten Messung vergangen ist
        current_time = time.time()
        time_since_last_read = current_time - self.last_read_time
        
        if time_since_last_read < self.update_interval:
            # Nicht genug Zeit vergangen - bei echtem Sensor würde nochmal der gleiche Wert gelesen
            return self.current_temperature
        
        # Aktualisiere die simulierte Temperatur basierend auf verstrichener Zeit
        self._update_simulated_temperature()
        
        # Füge Sensor-Ungenauigkeit und Rauschen hinzu
        accuracy_error = random.uniform(-self.accuracy, self.accuracy)
        noise = random.uniform(-self.noise_level, self.noise_level)
        measured_temp = self.current_temperature + accuracy_error + noise
        
        # Runde auf die nächste halbe Grad (typisch für viele Temperatursensoren)
        if self.sensor_type == SensorType.INTERNAL:
            # Interner Sensor hat schlechtere Auflösung
            measured_temp = round(measured_temp)
        else:
            # Externe Sensoren haben oft 0.5°C oder 0.125°C Auflösung
            measured_temp = round(measured_temp * 8) / 8
        
        # Aktualisiere Statusvariablen
        self.last_read_time = current_time
        self.readings_count += 1
        
        # Speichere die Messung im Verlauf (maximal 100 Einträge)
        if len(self.reading_history) >= 100:
            self.reading_history.pop(0)
        self.reading_history.append((current_time, measured_temp))
        
        # Simuliere Lesezeit
        if self.sensor_type == SensorType.INTERNAL:
            read_time = 0.001  # 1ms für internen Sensor
        elif self.sensor_type == SensorType.EXTERNAL_I2C:
            read_time = 0.005  # 5ms für I2C Kommunikation
        else:  # SPI
            read_time = 0.002  # 2ms für SPI Kommunikation
        
        time.sleep(read_time)
        
        return measured_temp
    
    def _update_simulated_temperature(self) -> None:
        """
        Aktualisiert die simulierte Umgebungstemperatur basierend auf der
        verstrichenen Zeit und simulierten externen Einflüssen.
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        
        if elapsed_time < 0.1:
            # Zu kurze Zeit für signifikante Änderungen
            return
        
        # Simuliere langsame Änderungen der Basistemperatur
        # (z.B. durch Tageszeit, Heizung, etc.)
        time_factor = elapsed_time / 3600  # Stunden
        temp_oscillation = 0.5 * math.sin(current_time / 900)  # ~15 min Periode
        
        # Kleine zufällige Änderung
        random_drift = random.uniform(-0.2, 0.2) * time_factor
        
        # Aktualisiere Basistemperatur (max. 2°C pro Stunde)
        max_change = 2.0 * time_factor
        self.base_temperature += min(max_change, max(-max_change, random_drift + temp_oscillation))
        
        # Aktuelle Temperatur ist Basistemperatur plus kurzfristige Schwankungen
        self.current_temperature = self.base_temperature + 0.2 * temp_oscillation
        
        # Begrenze Temperatur auf plausiblen Bereich (erweitert für Testszenarien)
        self.current_temperature = max(10.0, min(100.0, self.current_temperature))
        
        self.last_update_time = current_time
    
    def inject_temperature_spike(self, delta: float, duration: float = 60.0) -> None:
        """
        Injiziert einen künstlichen Temperaturanstieg für Testzwecke.
        
        Args:
            delta: Temperaturanstieg in °C
            duration: Ungefähre Dauer des Anstiegs in Sekunden
        """
        self.base_temperature += delta
        self.current_temperature += delta  # Direkte Aktualisierung für sofortige Wirkung
        logger.info(f"Temperatursensor: Künstlicher Anstieg um {delta:.1f}°C für ~{duration:.0f}s")
        
        # Nach 'duration' Sekunden wird die Basistemperatur langsam zurückgehen
        # durch die natürliche Drift in _update_simulated_temperature()
    
    def set_temperature_for_testing(self, temperature: float) -> None:
        """
        Setzt die Temperatur direkt für Testzwecke.
        Dies überschreibt die normale Temperatur-Simulation.
        
        Args:
            temperature: Gewünschte Temperatur in °C
        """
        self.base_temperature = temperature
        self.current_temperature = temperature
        logger.debug(f"Temperatursensor: Temperatur für Tests auf {temperature:.1f}°C gesetzt")
    
    def get_stats(self) -> Dict:
        """
        Gibt Statistiken zum Temperatursensor zurück.
        
        Returns:
            Dictionary mit Sensorstatistiken
        """
        history_temps = [temp for _, temp in self.reading_history]
        
        return {
            "sensor_type": self.sensor_type.value,
            "initialized": self.initialized,
            "readings_count": self.readings_count,
            "last_temperature": self.reading_history[-1][1] if self.reading_history else None,
            "min_temperature": min(history_temps) if history_temps else None,
            "max_temperature": max(history_temps) if history_temps else None,
            "avg_temperature": sum(history_temps) / len(history_temps) if history_temps else None
        }
