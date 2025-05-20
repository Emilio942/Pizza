"""
Emulation der UART-Schnittstelle für den RP2040.
Ermöglicht das Senden von Datenpaketen über UART, die im Emulator aufgefangen
und in eine Datei oder auf die Konsole ausgegeben werden.
"""

import time
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

class UARTEmulator:
    """Emuliert die UART-Schnittstelle des RP2040."""
    
    def __init__(self, log_to_file: bool = True, log_dir: Optional[str] = None):
        """
        Initialisiert den UART-Emulator.
        
        Args:
            log_to_file: Wenn True, werden UART-Ausgaben in eine Datei geschrieben
            log_dir: Verzeichnis für Log-Dateien, falls log_to_file=True
        """
        self.initialized = False
        self.baudrate = 115200  # Standard-Baudrate
        self.log_to_file = log_to_file
        self.log_file = None
        self.buffer = []
        
        if log_to_file:
            # Erstelle Log-Verzeichnis falls nicht vorhanden
            if log_dir:
                self.log_dir = Path(log_dir)
            else:
                self.log_dir = Path("output") / "emulator_logs"
            
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Erstelle Log-Datei mit Zeitstempel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"uart_log_{timestamp}.txt"
            
            logger.info(f"UART-Emulator initialisiert. Logfile: {self.log_file}")
        else:
            logger.info("UART-Emulator initialisiert. Logging auf Konsole.")
    
    def initialize(self, baudrate: int = 115200) -> bool:
        """
        Initialisiert die UART-Schnittstelle mit der angegebenen Baudrate.
        
        Args:
            baudrate: Die zu verwendende Baudrate (z.B. 9600, 115200)
            
        Returns:
            True wenn erfolgreich, sonst False
        """
        self.baudrate = baudrate
        self.initialized = True
        
        # Schreibe Initialisierungsinformation ins Log
        self._log_message(f"UART initialized with {baudrate} baud")
        
        logger.info(f"UART initialisiert mit {baudrate} Baud")
        return True
    
    def write(self, data: Union[str, bytes]) -> int:
        """
        Sendet Daten über UART und gibt die Anzahl gesendeter Bytes zurück.
        
        Args:
            data: Zu sendende Daten (String oder Bytes)
            
        Returns:
            Anzahl gesendeter Bytes
        """
        if not self.initialized:
            logger.warning("UART nicht initialisiert")
            return 0
        
        # Wandle String in Bytes um, falls notwendig
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Simuliere Übertragungszeit basierend auf Baudrate und Datenmenge
        # (10 Bits pro Byte: 8 Datenbits + Start + Stop)
        transmission_time = len(data_bytes) * 10 / self.baudrate
        
        # Logge die Daten
        if isinstance(data, str):
            self._log_message(data)
        else:
            self._log_message(data.decode('utf-8', errors='replace'))
        
        # Simuliere Übertragungszeit
        time.sleep(transmission_time)
        
        return len(data_bytes)
    
    def write_line(self, line: str) -> int:
        """
        Sendet eine Textzeile über UART (fügt automatisch Zeilenumbruch hinzu).
        
        Args:
            line: Zu sendende Zeile
            
        Returns:
            Anzahl gesendeter Bytes
        """
        if not line.endswith('\n'):
            line += '\n'
        
        return self.write(line)
    
    def _log_message(self, message: str) -> None:
        """
        Protokolliert eine UART-Nachricht in Datei oder Konsole.
        
        Args:
            message: Die zu protokollierende Nachricht
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted_message = f"[{timestamp}] {message}"
        
        # Speichere im Puffer
        self.buffer.append(formatted_message)
        
        if self.log_to_file and self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message)
                if not formatted_message.endswith('\n'):
                    f.write('\n')
        else:
            # Ausgabe auf Konsole
            print(f"UART: {formatted_message}")
    
    def get_buffer(self) -> List[str]:
        """
        Gibt den aktuellen UART-Ausgabepuffer zurück.
        
        Returns:
            Liste von UART-Ausgabezeilen
        """
        return self.buffer.copy()
    
    def clear_buffer(self) -> None:
        """Leert den UART-Ausgabepuffer."""
        self.buffer.clear()
    
    def close(self) -> None:
        """Schließt den UART-Emulator."""
        if self.initialized:
            self._log_message("UART connection closed")
            self.initialized = False
