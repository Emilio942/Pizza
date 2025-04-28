"""
Hardware-Emulator für RP2040 und OV2640 Kamera.
"""

import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from .constants import (
    RP2040_FLASH_SIZE_KB,
    RP2040_RAM_SIZE_KB,
    RP2040_CLOCK_SPEED_MHZ,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    INPUT_SIZE
)
from .exceptions import HardwareError, ResourceError
from .types import HardwareSpecs

logger = logging.getLogger(__name__)

class CameraEmulator:
    """Emuliert OV2640 Kamera."""
    
    def __init__(self):
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.rgb = True
        self.initialized = False
        self.frames_captured = 0
        self.startup_time = 0.1  # 100ms Startup-Zeit
        self.frame_time = 1.0 / 7  # ~7 FPS
        self.last_capture = 0
    
    def initialize(self) -> bool:
        """Emuliert Kamera-Initialisierung."""
        if not self.initialized:
            time.sleep(self.startup_time)
            self.initialized = True
            self.last_capture = time.time()
        return True
    
    def capture_frame(self) -> np.ndarray:
        """Emuliert Bildaufnahme."""
        if not self.initialized:
            raise HardwareError("Kamera nicht initialisiert")
        
        # Simuliere Framerate-Begrenzung
        elapsed = time.time() - self.last_capture
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        
        # Generiere simuliertes Bild
        channels = 3 if self.rgb else 1
        frame = np.random.randint(0, 256, (self.height, self.width, channels), dtype=np.uint8)
        
        self.frames_captured += 1
        self.last_capture = time.time()
        return frame
    
    def set_format(self, width: int, height: int, rgb: bool = True) -> None:
        """Konfiguriert Bildformat."""
        self.width = width
        self.height = height
        self.rgb = rgb

class RP2040Emulator:
    """Emuliert RP2040 Mikrocontroller."""
    
    def __init__(self):
        self.flash_size_bytes = RP2040_FLASH_SIZE_KB * 1024
        self.ram_size_bytes = RP2040_RAM_SIZE_KB * 1024
        self.cpu_speed_mhz = RP2040_CLOCK_SPEED_MHZ
        self.cores = 2
        
        self.ram_used = 0
        self.flash_used = 0
        self.system_ram_overhead = 40 * 1024  # 40KB System-Overhead
        
        self.camera = CameraEmulator()
        
        self.start_time = time.time()
        self.firmware = None
        self.firmware_loaded = False
        self.inference_time = 0
        
        logger.info(f"RP2040 Emulator gestartet")
        logger.info(f"Flash: {self.flash_size_bytes/1024:.0f}KB")
        logger.info(f"RAM: {self.ram_size_bytes/1024:.0f}KB")
        logger.info(f"CPU: {self.cores} Cores @ {self.cpu_speed_mhz}MHz")
    
    def load_firmware(self, firmware_path: Union[str, Path]) -> None:
        """Lädt simulierte Firmware."""
        logger.info(f"Lade Firmware: {firmware_path}")
        time.sleep(0.5)  # Simuliere Ladezeit
        
        self.firmware = {
            'path': str(firmware_path),
            'size_bytes': random.randint(50*1024, 150*1024),  # 50-150KB
            'loaded_at': time.time()
        }
        self.firmware_loaded = True
    
    def simulate_inference(self, image: np.ndarray) -> Dict:
        """Simuliert Modellinferenz."""
        if not self.firmware_loaded:
            raise HardwareError("Keine Firmware geladen")
        
        # Simuliere Verarbeitungszeit basierend auf Bildgröße und CPU-Geschwindigkeit
        ops_per_pixel = 100  # Geschätzte Operationen pro Pixel
        total_ops = image.size * ops_per_pixel
        
        # Grobe Schätzung der Inferenzzeit
        base_time = total_ops / (self.cpu_speed_mhz * 1e6)  # MHz zu Hz
        variance = random.uniform(0.9, 1.1)  # ±10% Varianz
        
        inference_time = base_time * variance
        time.sleep(inference_time)  # Simuliere Verarbeitung
        
        self.inference_time = inference_time
        
        return {
            'success': True,
            'inference_time': inference_time,
            'ram_used': self.get_ram_usage(),
            'class_id': random.randint(0, 5),  # Simulierte Klassifikation
            'confidence': random.uniform(0.6, 0.99)
        }
    
    def get_ram_usage(self) -> int:
        """Liefert simulierte RAM-Nutzung."""
        if not self.firmware_loaded:
            return self.system_ram_overhead
        
        # Simuliere RAM-Nutzung basierend auf Firmware-Größe
        firmware_ram = self.firmware['size_bytes'] * 0.2  # ~20% der Firmware-Größe
        return int(self.system_ram_overhead + firmware_ram)
    
    def get_flash_usage(self) -> int:
        """Liefert simulierte Flash-Nutzung."""
        if not self.firmware_loaded:
            return 0
        return self.firmware['size_bytes']
    
    def validate_resources(self) -> None:
        """Prüft Ressourcenbeschränkungen."""
        ram_usage = self.get_ram_usage()
        flash_usage = self.get_flash_usage()
        
        if ram_usage > self.ram_size_bytes:
            raise ResourceError(
                f"RAM-Überlauf: {ram_usage/1024:.1f}KB > {self.ram_size_bytes/1024:.1f}KB"
            )
        
        if flash_usage > self.flash_size_bytes:
            raise ResourceError(
                f"Flash-Überlauf: {flash_usage/1024:.1f}KB > {self.flash_size_bytes/1024:.1f}KB"
            )
    
    def get_system_stats(self) -> Dict:
        """Liefert Systemstatistiken."""
        ram_usage = self.get_ram_usage()
        flash_usage = self.get_flash_usage()
        
        return {
            'uptime_seconds': time.time() - self.start_time,
            'ram_used_kb': ram_usage / 1024,
            'ram_free_kb': (self.ram_size_bytes - ram_usage) / 1024,
            'flash_used_kb': flash_usage / 1024,
            'flash_free_kb': (self.flash_size_bytes - flash_usage) / 1024,
            'firmware_loaded': self.firmware_loaded,
            'last_inference_ms': self.inference_time * 1000,
            'camera_frames': self.camera.frames_captured
        }
    
    def reset(self) -> None:
        """Setzt Emulator zurück."""
        self.ram_used = 0
        self.flash_used = 0
        self.firmware = None
        self.firmware_loaded = False
        self.inference_time = 0
        self.camera = CameraEmulator()
        self.start_time = time.time()