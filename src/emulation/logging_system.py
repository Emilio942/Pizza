"""
Logging-Framework für das RP2040 Emulationssystem.
Ermöglicht das strukturierte Logging von Temperatur-, Performance- und Diagnosedaten.
Unterstützt Logging über UART, Dateien und simulierte SD-Karte.
"""

import time
import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from .uart_emulator import UARTEmulator
from .sd_card_emulator import SDCardEmulator

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
    Unterstützt Logging über UART, Dateien und/oder simulierte SD-Karte.
    """
    
    def __init__(self, 
                 uart: Optional[UARTEmulator] = None,
                 log_to_file: bool = True,
                 log_to_sd: bool = False,
                 sd_card: Optional[SDCardEmulator] = None,
                 log_dir: Optional[str] = None,
                 default_log_level: LogLevel = LogLevel.INFO):
        """
        Initialisiert das Logging-System.
        
        Args:
            uart: UARTEmulator-Instanz für UART-Logging
            log_to_file: Wenn True, werden Logs auch in Dateien geschrieben
            log_to_sd: Wenn True, werden Logs auch auf die simulierte SD-Karte geschrieben
            sd_card: SDCardEmulator-Instanz für SD-Karten-Logging
            log_dir: Verzeichnis für Log-Dateien
            default_log_level: Standard-Log-Level
        """
        self.uart = uart
        self.log_to_file = log_to_file
        self.log_to_sd = log_to_sd
        self.sd_card = sd_card
        self.default_log_level = default_log_level
        
        # SD-Karten-Dateihandles
        self.sd_temperature_handle = None
        self.sd_performance_handle = None
        self.sd_system_handle = None
        self.sd_test_handle = None
        
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
                f.write("timestamp,cpu_usage_percent,ram_used_kb,flash_used_kb,inference_time_ms,temperature_c,vdd_voltage_mv,battery_voltage_mv\n")
            
            logger.info(f"Logging-System initialisiert mit Logfiles in {self.log_dir}")
        else:
            logger.info("Logging-System initialisiert ohne Datei-Logging")
        
        # Initialisiere SD-Karten-Logging
        if log_to_sd and self.sd_card:
            if not self.sd_card.mounted:
                # Initialisiere SD-Karte, falls nicht bereits geschehen
                if self.sd_card.status.value == "not_initialized":
                    self.sd_card.initialize()
                
                # Versuche SD-Karte zu mounten
                if not self.sd_card.mount():
                    logger.error("SD-Karte konnte nicht gemountet werden, SD-Logging deaktiviert")
                    self.log_to_sd = False
                    return
            
            # Erstelle Verzeichnisstruktur auf SD-Karte (emuliert)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Öffne Log-Dateien auf SD-Karte
            self.sd_temperature_handle = self.sd_card.open_file(f"logs/temperature_log_{timestamp}.csv", "w")
            self.sd_performance_handle = self.sd_card.open_file(f"logs/performance_log_{timestamp}.csv", "w")
            self.sd_system_handle = self.sd_card.open_file(f"logs/system_log_{timestamp}.log", "w")
            self.sd_test_handle = self.sd_card.open_file(f"logs/test_log_{timestamp}.log", "w")
            
            # Schreibe Header für CSV-Dateien
            if self.sd_temperature_handle:
                self.sd_card.write_file(self.sd_temperature_handle, "timestamp,temperature_c,sensor_type,status\n")
            
            if self.sd_performance_handle:
                self.sd_card.write_file(self.sd_performance_handle, 
                                        "timestamp,cpu_usage_percent,ram_used_kb,flash_used_kb,inference_time_ms,temperature_c,vdd_voltage_mv,battery_voltage_mv\n")
            
            logger.info("SD-Karten-Logging initialisiert")
        elif log_to_sd:
            logger.warning("SD-Karten-Logging aktiviert, aber keine SD-Karte konfiguriert")
            self.log_to_sd = False
    
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
        
        # Logge auf SD-Karte, falls aktiviert
        if self.log_to_sd and self.sd_card and self.sd_card.mounted:
            if log_type == LogType.SYSTEM or log_type == LogType.DIAGNOSTIC:
                if self.sd_system_handle:
                    self.sd_card.write_file(self.sd_system_handle, log_entry + '\n')
            elif log_type == LogType.TEST:
                if self.sd_test_handle:
                    self.sd_card.write_file(self.sd_test_handle, log_entry + '\n')
        
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
        
        # Schreibe auf SD-Karte, falls aktiviert
        if self.log_to_sd and self.sd_card and self.sd_card.mounted and self.sd_temperature_handle:
            self.sd_card.write_file(self.sd_temperature_handle, 
                                  f"{timestamp},{temperature_c:.2f},{sensor_type},{status}\n")
    
    def log_voltage(self, voltage_mv: float, sensor_type: str, status: str = "OK") -> None:
        """
        Loggt einen Spannungsmesswert.
        
        Args:
            voltage_mv: Gemessene Spannung in mV
            sensor_type: Typ des Spannungssensors (z.B. 'internal', 'battery')
            status: Status des Sensors/der Messung
        """
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        
        # Erstelle Log-Eintrag für UART/Datei
        message = f"Voltage: {voltage_mv/1000:.3f}V (sensor: {sensor_type}, status: {status})"
        self.log(message, LogLevel.INFO, LogType.DIAGNOSTIC)
        
        # Spannung wird primär im Performance-Log mit erfasst
    
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
        vdd_voltage = metrics.get('vdd_voltage_mv', 0)
        battery_voltage = metrics.get('battery_voltage_mv', 0)
        
        # Erstelle menschenlesbare Log-Nachricht
        message = (
            f"Performance: CPU {cpu_usage:.1f}%, RAM {ram_used:.1f}KB, "
            f"Inference {inference_time:.2f}ms, Temp {temperature:.1f}°C, "
            f"VDD {vdd_voltage/1000:.2f}V, Batt {battery_voltage/1000:.2f}V"
        )
        self.log(message, LogLevel.INFO, LogType.PERFORMANCE)
        
        # Schreibe strukturierte Daten in CSV, falls aktiviert
        if self.log_to_file:
            with open(self.performance_log_file, 'a') as f:
                f.write(f"{timestamp},{cpu_usage:.1f},{ram_used:.1f},{flash_used:.1f},{inference_time:.2f},{temperature:.2f},{vdd_voltage:.1f},{battery_voltage:.1f}\n")
        
        # Schreibe auf SD-Karte, falls aktiviert
        if self.log_to_sd and self.sd_card and self.sd_card.mounted and self.sd_performance_handle:
            self.sd_card.write_file(self.sd_performance_handle, 
                                  f"{timestamp},{cpu_usage:.1f},{ram_used:.1f},{flash_used:.1f},{inference_time:.2f},{temperature:.2f},{vdd_voltage:.1f},{battery_voltage:.1f}\n")
    
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
        
        # Schließe SD-Karten-Dateien
        if self.log_to_sd and self.sd_card and self.sd_card.mounted:
            if self.sd_temperature_handle:
                self.sd_card.close_file(self.sd_temperature_handle)
                self.sd_temperature_handle = None
            
            if self.sd_performance_handle:
                self.sd_card.close_file(self.sd_performance_handle)
                self.sd_performance_handle = None
            
            if self.sd_system_handle:
                self.sd_card.close_file(self.sd_system_handle)
                self.sd_system_handle = None
            
            if self.sd_test_handle:
                self.sd_card.close_file(self.sd_test_handle)
                self.sd_test_handle = None
            
            # Unmounte SD-Karte
            self.sd_card.unmount()
            
            logger.info("SD-Karten-Logging beendet")
