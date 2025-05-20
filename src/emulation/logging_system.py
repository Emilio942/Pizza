"""
Logging-Framework für das RP2040 Emulationssystem.
Ermöglicht das strukturierte Logging von Temperatur-, Performance- und Diagnosedaten.
"""

import time
import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from .uart_emulator import UARTEmulator

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log-Level für das Logging-System."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogType(Enum):
    """Typen von Log-Einträgen."""
    TEMPERATURE = "TEMP"
    PERFORMANCE = "PERF"
    SYSTEM = "SYS"
    DIAGNOSTIC = "DIAG"
    TEST = "TEST"

class LoggingSystem:
    """
    Zentrales Logging-System für den RP2040-Emulator.
    Unterstützt Logging über UART und/oder in Dateien.
    """
    
    def __init__(self, 
                 uart: Optional[UARTEmulator] = None,
                 log_to_file: bool = True,
                 log_dir: Optional[str] = None,
                 default_log_level: LogLevel = LogLevel.INFO):
        """
        Initialisiert das Logging-System.
        
        Args:
            uart: UARTEmulator-Instanz für UART-Logging
            log_to_file: Wenn True, werden Logs auch in Dateien geschrieben
            log_dir: Verzeichnis für Log-Dateien
            default_log_level: Standard-Log-Level
        """
        self.uart = uart
        self.log_to_file = log_to_file
        self.default_log_level = default_log_level
        
        # Erstelle Log-Verzeichnis falls notwendig
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path("output") / "emulator_logs"
        
        if log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Erstelle separate Log-Dateien für verschiedene Log-Typen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.temperature_log_file = self.log_dir / f"temperature_log_{timestamp}.csv"
            self.performance_log_file = self.log_dir / f"performance_log_{timestamp}.csv"
            self.system_log_file = self.log_dir / f"system_log_{timestamp}.log"
            self.test_log_file = self.log_dir / f"test_log_{timestamp}.log"
            
            # Schreibe Header für CSV-Dateien
            with open(self.temperature_log_file, 'w') as f:
                f.write("timestamp,temperature_c,sensor_type,status\n")
            
            with open(self.performance_log_file, 'w') as f:
                f.write("timestamp,cpu_usage_percent,ram_used_kb,flash_used_kb,inference_time_ms,temperature_c\n")
            
            logger.info(f"Logging-System initialisiert mit Logfiles in {self.log_dir}")
        else:
            logger.info("Logging-System initialisiert ohne Datei-Logging")
    
    def log(self, message: str, level: LogLevel = None, log_type: LogType = LogType.SYSTEM) -> None:
        """
        Loggt eine einfache Textnachricht.
        
        Args:
            message: Die zu loggende Nachricht
            level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_type: Typ des Log-Eintrags
        """
        if level is None:
            level = self.default_log_level
        
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        log_entry = f"[{timestamp}] [{level.value}] [{log_type.value}] {message}"
        
        # Sende über UART, falls konfiguriert
        if self.uart:
            self.uart.write_line(log_entry)
        
        # Logge in entsprechende Datei, falls aktiviert
        if self.log_to_file:
            if log_type == LogType.SYSTEM or log_type == LogType.DIAGNOSTIC:
                with open(self.system_log_file, 'a') as f:
                    f.write(log_entry + '\n')
            elif log_type == LogType.TEST:
                with open(self.test_log_file, 'a') as f:
                    f.write(log_entry + '\n')
        
        # Gib Log auch im Emulator-Logger aus
        py_level = getattr(logging, level.value)
        logger.log(py_level, f"[{log_type.value}] {message}")
    
    def log_temperature(self, temperature_c: float, sensor_type: str, status: str = "OK") -> None:
        """
        Loggt einen Temperaturmesswert.
        
        Args:
            temperature_c: Gemessene Temperatur in °C
            sensor_type: Typ des Temperatursensors (z.B. 'internal', 'i2c')
            status: Status des Sensors/der Messung
        """
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        
        # Erstelle Log-Eintrag für UART/Datei
        message = f"Temperature: {temperature_c:.2f}°C (sensor: {sensor_type}, status: {status})"
        self.log(message, LogLevel.INFO, LogType.TEMPERATURE)
        
        # Schreibe strukturierte Daten in CSV, falls aktiviert
        if self.log_to_file:
            with open(self.temperature_log_file, 'a') as f:
                f.write(f"{timestamp},{temperature_c:.2f},{sensor_type},{status}\n")
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Loggt Performance-Daten.
        
        Args:
            metrics: Dictionary mit Performance-Metriken
        """
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        
        # Extrahiere wichtige Metriken mit Standardwerten
        cpu_usage = metrics.get('cpu_usage_percent', 0)
        ram_used = metrics.get('ram_used_kb', 0)
        flash_used = metrics.get('flash_used_kb', 0)
        inference_time = metrics.get('inference_time_ms', 0)
        temperature = metrics.get('temperature_c', 0)
        
        # Erstelle menschenlesbare Log-Nachricht
        message = (
            f"Performance: CPU {cpu_usage:.1f}%, RAM {ram_used:.1f}KB, "
            f"Inference {inference_time:.2f}ms, Temp {temperature:.1f}°C"
        )
        self.log(message, LogLevel.INFO, LogType.PERFORMANCE)
        
        # Schreibe strukturierte Daten in CSV, falls aktiviert
        if self.log_to_file:
            with open(self.performance_log_file, 'a') as f:
                f.write(f"{timestamp},{cpu_usage:.1f},{ram_used:.1f},{flash_used:.1f},{inference_time:.2f},{temperature:.2f}\n")
    
    def log_json(self, data: Dict[str, Any], log_type: LogType = LogType.SYSTEM) -> None:
        """
        Loggt strukturierte Daten im JSON-Format.
        
        Args:
            data: Dictionary mit zu loggenden Daten
            log_type: Typ des Log-Eintrags
        """
        json_str = json.dumps(data)
        self.log(json_str, LogLevel.INFO, log_type)
    
    def close(self) -> None:
        """Schließt das Logging-System."""
        self.log("Logging system shutting down", LogLevel.INFO, LogType.SYSTEM)
