"""
Hardware-Management für RP2040 und OV2640 Kamera.
"""

import time
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
import numpy as np

from .exceptions import HardwareError
from .constants import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    RP2040_FLASH_SIZE_KB, RP2040_RAM_SIZE_KB
)

logger = logging.getLogger(__name__)

class PowerMode(Enum):
    """Energiesparmodi des Systems."""
    ACTIVE = "active"
    SLEEP = "sleep"
    DEEP_SLEEP = "deep_sleep"

class DeviceStatus(Enum):
    """Gerätestatus."""
    OK = "ok"
    ERROR = "error"
    BUSY = "busy"
    LOW_BATTERY = "low_battery"
    CRITICAL_BATTERY = "critical_battery"

class BatteryStatus:
    """Batteriestatusüberwachung."""
    
    # Batteriespannungsgrenzen (mV)
    FULL_MV = 3000
    LOW_MV = 2200
    CRITICAL_MV = 2000
    
    def __init__(self):
        self.voltage_mv = self.FULL_MV
        self.capacity_percent = 100.0
        self.last_check = time.time()
    
    def update(self, voltage_mv: float) -> None:
        """Aktualisiert Batteriestatus."""
        self.voltage_mv = voltage_mv
        
        # Lineare Approximation der Kapazität
        voltage_range = self.FULL_MV - self.CRITICAL_MV
        voltage_above_critical = max(0, voltage_mv - self.CRITICAL_MV)
        self.capacity_percent = (voltage_above_critical / voltage_range) * 100
        
        self.last_check = time.time()
    
    def is_low(self) -> bool:
        """Prüft ob Batterie schwach ist."""
        return self.voltage_mv <= self.LOW_MV
    
    def is_critical(self) -> bool:
        """Prüft ob Batterie kritisch ist."""
        return self.voltage_mv <= self.CRITICAL_MV

class Camera:
    """OV2640 Kamera-Management."""
    
    def __init__(self):
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.fps = CAMERA_FPS
        self.is_initialized = False
        self.power_mode = PowerMode.SLEEP
        self.frame_count = 0
        self.error_count = 0
    
    def initialize(self) -> bool:
        """Initialisiert die Kamera."""
        try:
            # Simuliere Kamerainitialisierung
            time.sleep(0.1)
            self.is_initialized = True
            self.power_mode = PowerMode.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Kamerainitialisierung fehlgeschlagen: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Nimmt ein Bild auf."""
        if not self.is_initialized:
            raise HardwareError("Kamera nicht initialisiert")
        
        try:
            # Simuliere Bildaufnahme (grayscale)
            frame = np.random.randint(0, 256, (self.height, self.width), dtype=np.uint8)
            self.frame_count += 1
            return frame
        except Exception as e:
            self.error_count += 1
            logger.error(f"Bildaufnahme fehlgeschlagen: {e}")
            return None
    
    def set_power_mode(self, mode: PowerMode) -> None:
        """Setzt Energiesparmodus."""
        if mode == PowerMode.DEEP_SLEEP and self.power_mode != PowerMode.DEEP_SLEEP:
            self.is_initialized = False
        self.power_mode = mode

class SystemController:
    """RP2040 System-Controller."""
    
    def __init__(self):
        self.camera = Camera()
        self.battery = BatteryStatus()
        self.status = DeviceStatus.OK
        self.power_mode = PowerMode.ACTIVE
        self.start_time = time.time()
        self.last_inference_time = 0.0
    
    def initialize(self) -> bool:
        """Initialisiert das System."""
        try:
            success = self.camera.initialize()
            if not success:
                self.status = DeviceStatus.ERROR
                return False
            
            self.status = DeviceStatus.OK
            return True
        except Exception as e:
            logger.error(f"Systeminitialisierung fehlgeschlagen: {e}")
            self.status = DeviceStatus.ERROR
            return False
    
    def get_system_stats(self) -> Dict:
        """Liefert Systemstatistiken."""
        uptime = time.time() - self.start_time
        
        return {
            'status': self.status.value,
            'power_mode': self.power_mode.value,
            'uptime_seconds': uptime,
            'battery_percent': self.battery.capacity_percent,
            'battery_voltage_mv': self.battery.voltage_mv,
            'camera_frames': self.camera.frame_count,
            'camera_errors': self.camera.error_count,
            'last_inference_ms': self.last_inference_time * 1000
        }
    
    def update_power_state(self) -> None:
        """Aktualisiert Energiezustand basierend auf Batterie."""
        if self.battery.is_critical():
            self.status = DeviceStatus.CRITICAL_BATTERY
            self.set_power_mode(PowerMode.DEEP_SLEEP)
        elif self.battery.is_low():
            self.status = DeviceStatus.LOW_BATTERY
    
    def set_power_mode(self, mode: PowerMode) -> None:
        """Setzt Systemenergiemodus."""
        self.power_mode = mode
        self.camera.set_power_mode(mode)
        
        if mode == PowerMode.DEEP_SLEEP:
            logger.warning("System geht in Deep-Sleep")
        elif mode == PowerMode.SLEEP:
            logger.info("System geht in Sleep-Mode")
        else:
            logger.info("System ist aktiv")