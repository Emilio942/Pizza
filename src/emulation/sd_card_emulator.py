"""
SD Card Emulator for RP2040 Emulation System.

Implements a simple emulation of an SD card for logging purposes,
providing file operations similar to real hardware.
"""

import os
import time
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set

logger = logging.getLogger(__name__)

class SDCardStatus(Enum):
    """Status des SD-Karten-Emulators."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZED = "initialized"
    MOUNTED = "mounted"
    ERROR = "error"

class SDCardEmulator:
    """
    Emuliert eine SD-Karte für das RP2040-System.
    Ermöglicht das Erstellen, Lesen und Schreiben von Dateien in einem
    simulierten Dateisystem.
    """
    
    def __init__(self, 
                 root_dir: str = "output/sd_card",
                 capacity_mb: int = 1024,
                 fail_probability: float = 0.0):
        """
        Initialisiert den SD-Karten-Emulator.
        
        Args:
            root_dir: Verzeichnis, das als SD-Karte emuliert wird
            capacity_mb: Kapazität der emulierten SD-Karte in MB
            fail_probability: Wahrscheinlichkeit für simulierte Fehler (0.0-1.0)
        """
        self.root_dir = Path(root_dir)
        self.capacity_mb = capacity_mb
        self.capacity_bytes = capacity_mb * 1024 * 1024
        self.fail_probability = fail_probability
        self.status = SDCardStatus.NOT_INITIALIZED
        self.mounted = False
        self.open_files = set()  # Set zum Speichern offener Dateien
        
        logger.info(f"SD-Karten-Emulator erstellt: {capacity_mb}MB, Root: {root_dir}")
    
    def initialize(self) -> bool:
        """
        Initialisiert die SD-Karte (simuliert Hardware-Initialisierung).
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if self.status != SDCardStatus.NOT_INITIALIZED:
            logger.warning("SD-Karte bereits initialisiert")
            return True
        
        try:
            # Erstelle Root-Verzeichnis, wenn es nicht existiert
            self.root_dir.mkdir(parents=True, exist_ok=True)
            
            self.status = SDCardStatus.INITIALIZED
            logger.info(f"SD-Karte initialisiert: {self.capacity_mb}MB")
            return True
        except Exception as e:
            logger.error(f"Fehler bei SD-Karten-Initialisierung: {e}")
            self.status = SDCardStatus.ERROR
            return False
    
    def mount(self) -> bool:
        """
        Mountet das Dateisystem der SD-Karte (simuliert FatFS-Mounting).
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if self.status == SDCardStatus.NOT_INITIALIZED:
            logger.error("SD-Karte nicht initialisiert")
            return False
        
        if self.status == SDCardStatus.ERROR:
            logger.error("SD-Karte im Fehlerzustand")
            return False
        
        if self.mounted:
            logger.warning("SD-Karte bereits gemountet")
            return True
        
        try:
            # Prüfe, ob Verzeichnis existiert und beschreibbar ist
            if not self.root_dir.exists() or not os.access(self.root_dir, os.W_OK):
                logger.error(f"SD-Karten-Verzeichnis nicht beschreibbar: {self.root_dir}")
                self.status = SDCardStatus.ERROR
                return False
            
            self.mounted = True
            self.status = SDCardStatus.MOUNTED
            logger.info(f"SD-Karte gemountet: {self.root_dir}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Mounten der SD-Karte: {e}")
            self.status = SDCardStatus.ERROR
            return False
    
    def unmount(self) -> bool:
        """
        Unmountet das Dateisystem der SD-Karte.
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not self.mounted:
            logger.warning("SD-Karte ist nicht gemountet")
            return True
        
        # Schließe alle offenen Dateien
        self._close_all_files()
        
        self.mounted = False
        self.status = SDCardStatus.INITIALIZED
        logger.info("SD-Karte unmounted")
        return True
    
    def open_file(self, filename: str, mode: str) -> Optional[int]:
        """
        Öffnet eine Datei auf der SD-Karte.
        
        Args:
            filename: Name der Datei (relativ zum Root-Verzeichnis)
            mode: Öffnungsmodus ('r', 'w', 'a', 'r+', ...)
        
        Returns:
            Optional[int]: File-Handle (ID) oder None bei Fehler
        """
        if not self.mounted:
            logger.error("SD-Karte nicht gemountet")
            return None
        
        try:
            # Erstelle vollständigen Pfad
            file_path = self.root_dir / filename
            
            # Erstelle übergeordnete Verzeichnisse, falls nötig
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Öffne Datei und speichere in open_files mit einer eindeutigen ID
            file_handle = len(self.open_files) + 1
            self.open_files.add((file_handle, file_path, mode))
            
            logger.debug(f"Datei geöffnet: {filename}, Modus: {mode}, Handle: {file_handle}")
            return file_handle
        except Exception as e:
            logger.error(f"Fehler beim Öffnen der Datei {filename}: {e}")
            return None
    
    def write_file(self, file_handle: int, data: str) -> bool:
        """
        Schreibt Daten in eine geöffnete Datei.
        
        Args:
            file_handle: File-Handle von open_file()
            data: Zu schreibende Daten (String)
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not self.mounted:
            logger.error("SD-Karte nicht gemountet")
            return False
        
        # Suche nach dem passenden file_handle in open_files
        for handle, path, mode in self.open_files:
            if handle == file_handle:
                if 'r' in mode and '+' not in mode:
                    logger.error(f"Datei mit Handle {file_handle} ist nur zum Lesen geöffnet")
                    return False
                
                try:
                    # Schreibe in die Datei, je nach Modus
                    if 'a' in mode:
                        with open(path, 'a') as f:
                            f.write(data)
                    elif 'w' in mode:
                        with open(path, 'w') as f:
                            f.write(data)
                    else:  # 'r+' oder ähnlich
                        with open(path, 'r+') as f:
                            f.write(data)
                    
                    logger.debug(f"In Datei geschrieben: {path.name}, {len(data)} Bytes")
                    return True
                except Exception as e:
                    logger.error(f"Fehler beim Schreiben in Datei: {e}")
                    return False
        
        logger.error(f"Ungültiger File-Handle: {file_handle}")
        return False
    
    def read_file(self, file_handle: int, size: int = -1) -> Optional[str]:
        """
        Liest Daten aus einer geöffneten Datei.
        
        Args:
            file_handle: File-Handle von open_file()
            size: Anzahl zu lesender Bytes (-1 für alles)
        
        Returns:
            Optional[str]: Gelesene Daten oder None bei Fehler
        """
        if not self.mounted:
            logger.error("SD-Karte nicht gemountet")
            return None
        
        # Suche nach dem passenden file_handle in open_files
        for handle, path, mode in self.open_files:
            if handle == file_handle:
                if 'w' in mode and '+' not in mode:
                    logger.error(f"Datei mit Handle {file_handle} ist nur zum Schreiben geöffnet")
                    return None
                
                try:
                    # Lese aus der Datei
                    with open(path, 'r') as f:
                        if size < 0:
                            data = f.read()
                        else:
                            data = f.read(size)
                    
                    logger.debug(f"Aus Datei gelesen: {path.name}, {len(data)} Bytes")
                    return data
                except Exception as e:
                    logger.error(f"Fehler beim Lesen aus Datei: {e}")
                    return None
        
        logger.error(f"Ungültiger File-Handle: {file_handle}")
        return None
    
    def close_file(self, file_handle: int) -> bool:
        """
        Schließt eine geöffnete Datei.
        
        Args:
            file_handle: File-Handle von open_file()
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not self.mounted:
            logger.warning("SD-Karte nicht gemountet")
            return False
        
        for file_info in list(self.open_files):
            handle, _, _ = file_info
            if handle == file_handle:
                self.open_files.remove(file_info)
                logger.debug(f"Datei geschlossen: Handle {file_handle}")
                return True
        
        logger.error(f"Ungültiger File-Handle: {file_handle}")
        return False
    
    def list_directory(self, directory: str = "") -> Optional[List[str]]:
        """
        Listet den Inhalt eines Verzeichnisses auf.
        
        Args:
            directory: Pfad zum Verzeichnis (relativ zum Root-Verzeichnis)
        
        Returns:
            Optional[List[str]]: Liste der Dateien und Unterverzeichnisse oder None bei Fehler
        """
        if not self.mounted:
            logger.error("SD-Karte nicht gemountet")
            return None
        
        try:
            dir_path = self.root_dir / directory
            if not dir_path.exists() or not dir_path.is_dir():
                logger.error(f"Verzeichnis existiert nicht: {directory}")
                return None
            
            # Liste Dateien und Verzeichnisse auf
            items = [item.name for item in dir_path.iterdir()]
            return items
        except Exception as e:
            logger.error(f"Fehler beim Auflisten des Verzeichnisses {directory}: {e}")
            return None
    
    def file_exists(self, filename: str) -> bool:
        """
        Prüft, ob eine Datei existiert.
        
        Args:
            filename: Name der Datei (relativ zum Root-Verzeichnis)
        
        Returns:
            bool: True, wenn die Datei existiert, sonst False
        """
        if not self.mounted:
            logger.error("SD-Karte nicht gemountet")
            return False
        
        try:
            file_path = self.root_dir / filename
            return file_path.exists() and file_path.is_file()
        except Exception as e:
            logger.error(f"Fehler beim Prüfen auf Dateiexistenz: {e}")
            return False
    
    def get_file_size(self, filename: str) -> Optional[int]:
        """
        Ermittelt die Größe einer Datei.
        
        Args:
            filename: Name der Datei (relativ zum Root-Verzeichnis)
        
        Returns:
            Optional[int]: Größe der Datei in Bytes oder None bei Fehler
        """
        if not self.mounted:
            logger.error("SD-Karte nicht gemountet")
            return None
        
        try:
            file_path = self.root_dir / filename
            if not file_path.exists() or not file_path.is_file():
                logger.error(f"Datei existiert nicht: {filename}")
                return None
            
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"Fehler beim Ermitteln der Dateigröße: {e}")
            return None
    
    def get_free_space(self) -> int:
        """
        Ermittelt den freien Speicherplatz auf der SD-Karte.
        
        Returns:
            int: Freier Speicherplatz in Bytes
        """
        try:
            # Berechne tatsächliche Nutzung
            total_size = sum(f.stat().st_size for f in self.root_dir.glob('**/*') if f.is_file())
            free_space = max(0, self.capacity_bytes - total_size)
            return free_space
        except Exception as e:
            logger.error(f"Fehler beim Ermitteln des freien Speicherplatzes: {e}")
            return 0
    
    def get_status(self) -> Dict[str, Any]:
        """
        Liefert Statusinformationen zur SD-Karte.
        
        Returns:
            Dict[str, Any]: Statusinformationen
        """
        return {
            'status': self.status.value,
            'mounted': self.mounted,
            'capacity_mb': self.capacity_mb,
            'free_space_mb': self.get_free_space() / (1024 * 1024),
            'root_dir': str(self.root_dir),
            'open_files': len(self.open_files)
        }
    
    def _close_all_files(self) -> None:
        """Schließt alle offenen Dateien."""
        self.open_files.clear()
        logger.debug("Alle Dateien geschlossen")
