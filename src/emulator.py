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
from .power_manager import PowerManager, PowerUsage, AdaptiveMode

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
        if (elapsed < self.frame_time):
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
    
    def __init__(self, battery_capacity_mah: float = 1500.0, adaptive_mode: AdaptiveMode = AdaptiveMode.BALANCED):
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
        
        # Energiemanagement-Zustände
        self.sleep_mode = False
        self.sleep_ram_reduction = 0.6  # 60% RAM-Reduktion im Sleep-Mode
        self.original_ram_used = 0
        self.sleep_start_time = 0
        self.total_sleep_time = 0
        
        # Simulation der Systemtemperatur
        self.current_temperature_c = 25.0  # Startwert: Raumtemperatur
        
        # Inizialisierung des Power Managers
        self.power_usage = PowerUsage(
            sleep_mode_ma=0.5,
            idle_ma=10.0,
            active_ma=80.0,
            camera_active_ma=40.0,
            inference_ma=100.0
        )
        self.battery_capacity_mah = battery_capacity_mah
        self.power_manager = PowerManager(
            emulator=self,
            power_usage=self.power_usage,
            battery_capacity_mah=battery_capacity_mah,
            adaptive_mode=adaptive_mode
        )
        
        # Initialisiere die Temperatur im PowerManager
        self.power_manager.update_temperature(self.current_temperature_c)
        
        # Tracking für Erkennungsänderungen
        self.last_detection_class = None
        self.last_detection_time = 0
        
        logger.info(f"RP2040 Emulator gestartet")
        logger.info(f"Flash: {self.flash_size_bytes/1024:.0f}KB")
        logger.info(f"RAM: {self.ram_size_bytes/1024:.0f}KB")
        logger.info(f"CPU: {self.cores} Cores @ {self.cpu_speed_mhz}MHz")
        logger.info(f"Energiemanagement-Modus: {adaptive_mode.value}")
        logger.info(f"Geschätzte Batterielebensdauer: {self.power_manager.estimated_runtime_hours:.1f} Stunden")
    
    def load_firmware(self, firmware: Dict) -> bool:
        """Lädt simulierte Firmware."""
        logger.info(f"Lade Firmware: {firmware}")
        
        # Prüfe, ob die Firmware ins Flash passt
        if firmware['total_size_bytes'] > self.flash_size_bytes:
            logger.error(f"Flash-Überlauf: {firmware['total_size_bytes']/1024:.1f}KB > {self.flash_size_bytes/1024:.1f}KB")
            raise ResourceError(
                f"Flash-Überlauf: {firmware['total_size_bytes']/1024:.1f}KB > {self.flash_size_bytes/1024:.1f}KB"
            )
        
        # Prüfe, ob der RAM-Bedarf erfüllt werden kann (inklusive System-Overhead)
        total_ram_needed = firmware['ram_usage_bytes'] + self.system_ram_overhead
        if total_ram_needed > self.ram_size_bytes:
            logger.error(f"RAM-Überlauf: {total_ram_needed/1024:.1f}KB > {self.ram_size_bytes/1024:.1f}KB")
            raise ResourceError(
                f"RAM-Überlauf: {total_ram_needed/1024:.1f}KB > {self.ram_size_bytes/1024:.1f}KB"
            )
        
        # Firmware laden, wenn sie in den Speicher passt
        time.sleep(0.1)  # Simuliere Ladezeit
        
        self.firmware = firmware
        self.firmware_loaded = True
        self.flash_used = firmware['total_size_bytes']
        self.ram_used = firmware['ram_usage_bytes']
        
        return True
    
    def simulate_inference(self, image: np.ndarray) -> Dict:
        """Simuliert Modellinferenz."""
        # Verlasse automatisch den Sleep-Mode, wenn eine Inferenz angefordert wird
        if self.sleep_mode:
            self.wake_up()
            
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
        
        # Generiere simulierte Klassifikation
        current_class_id = random.randint(0, 3)  # 4 Klassen: basic, burnt, undercooked, perfect
        confidence = random.uniform(0.6, 0.99)
        
        # Aktualisiere Energiemanagement und Aktivitätsverfolgung
        self.power_manager.add_inference_time(inference_time * 1000)  # ms
        
        current_time = time.time()
        # Aktualisiere den Energieverbrauch für die aktive Phase
        active_duration = current_time - self.last_detection_time if self.last_detection_time > 0 else 0
        if active_duration > 0:
            self.power_manager.update_energy_consumption(active_duration, True)
            
        # Prüfe, ob sich die Erkennung geändert hat
        detection_changed = (self.last_detection_class != current_class_id)
        self.power_manager.update_activity(detection_changed)
        
        # Aktualisiere Tracking für die nächste Inferenz
        self.last_detection_class = current_class_id
        self.last_detection_time = current_time
        
        # Überprüfe, ob nach der Inferenz in den Sleep-Modus gewechselt werden sollte
        if self.power_manager.should_enter_sleep():
            self.power_manager.enter_sleep_mode()
        
        return {
            'success': True,
            'inference_time': inference_time,
            'ram_used': self.get_ram_usage(),
            'class_id': current_class_id,
            'confidence': confidence,
            'power_stats': self.power_manager.get_power_statistics()
        }
    
    def get_ram_usage(self) -> int:
        """Liefert simulierte RAM-Nutzung."""
        if not self.firmware_loaded:
            return self.system_ram_overhead
        
        # Verwende die Werte, die in load_firmware gesetzt wurden
        base_ram = self.ram_used + self.system_ram_overhead
        
        # Wenn im Sleep-Mode, reduziere RAM-Nutzung
        if self.sleep_mode:
            return int(base_ram * (1 - self.sleep_ram_reduction))
        return base_ram
    
    def get_flash_usage(self) -> int:
        """Liefert simulierte Flash-Nutzung."""
        if not self.firmware_loaded:
            return 0
        return self.flash_used
    
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
        
        # Basisstatistiken
        stats = {
            'uptime_seconds': time.time() - self.start_time,
            'ram_used_kb': ram_usage / 1024,
            'ram_free_kb': (self.ram_size_bytes - ram_usage) / 1024,
            'flash_used_kb': flash_usage / 1024,
            'flash_free_kb': (self.flash_size_bytes - flash_usage) / 1024,
            'firmware_loaded': self.firmware_loaded,
            'last_inference_ms': self.inference_time * 1000,
            'camera_frames': self.camera.frames_captured,
            'sleep_mode': self.sleep_mode,
            'total_sleep_time': self.total_sleep_time
        }
        
        # Füge Energiemanagement-Statistiken hinzu
        power_stats = self.power_manager.get_power_statistics()
        stats.update({
            'energy_management_mode': power_stats['mode'],
            'battery_capacity_mah': power_stats['battery_capacity_mah'],
            'energy_consumed_mah': power_stats['energy_consumed_mah'],
            'estimated_runtime_hours': power_stats['estimated_runtime_hours'],
            'estimated_runtime_days': power_stats['estimated_runtime_hours'] / 24,
            'duty_cycle': power_stats['duty_cycle'],
            'sampling_interval_s': power_stats['sampling_interval_s'],
            'activity_level': power_stats['activity_level']
        })
        
        return stats
    
    def reset(self) -> None:
        """Setzt Emulator zurück."""
        self.ram_used = 0
        self.flash_used = 0
        self.firmware = None
        self.firmware_loaded = False
        self.inference_time = 0
        self.camera = CameraEmulator()
        self.start_time = time.time()
    
    def enter_sleep_mode(self) -> None:
        """Versetzt den Emulator in den Sleep-Mode für Energieeinsparung."""
        if not self.sleep_mode:
            logger.info("Emulator geht in Sleep-Mode")
            self.sleep_mode = True
            self.sleep_start_time = time.time()
            
            # Speichere aktuellen RAM-Verbrauch VOR Anwendung der sleep_mode Variable
            self.original_ram_used = self.ram_used
            
            # Reduziere RAM-Verbrauch im Sleep-Mode (deaktivierte Komponenten)
            self.ram_used = int(self.original_ram_used * (1 - self.sleep_ram_reduction))
            
            # Aktualisiere den PowerManager
            active_duration = time.time() - self.last_detection_time if self.last_detection_time > 0 else 0
            if active_duration > 0:
                self.power_manager.update_energy_consumption(active_duration, True)
                self.power_manager.sleep_start_time = time.time()
    
    def wake_up(self) -> None:
        """Weckt den Emulator aus dem Sleep-Mode auf."""
        if self.sleep_mode:
            # Berechne Schlafzeit für Energieberechnung
            sleep_duration = time.time() - self.sleep_start_time
            self.total_sleep_time += sleep_duration
            
            # Stelle originalen RAM-Verbrauch wieder her
            self.ram_used = self.original_ram_used
            
            # Aktualisiere den PowerManager mit der Schlafzeit
            self.power_manager.update_energy_consumption(sleep_duration, False)
            self.power_manager.total_sleep_time += sleep_duration
            self.power_manager.last_wakeup_time = time.time()
            
            logger.info(f"Emulator aufgeweckt (war {sleep_duration:.2f}s im Sleep-Mode)")
            self.sleep_mode = False
    
    def execute_operation(self, memory_usage_bytes, operation_time_ms):
        """
        Simuliert die Ausführung einer Operation und berechnet die Temperaturentwicklung.
        
        Args:
            memory_usage_bytes: Speicherverbrauch der Operation in Bytes
            operation_time_ms: Ausführungszeit der Operation in Millisekunden
        """
        # Speichernutzung temporär erhöhen
        original_ram_used = self.ram_used
        self.ram_used += memory_usage_bytes
        
        # Prüfe, ob genug Speicher vorhanden ist
        if self.get_ram_usage() > self.ram_size_bytes:
            self.ram_used = original_ram_used  # Setze zurück
            raise ResourceError(f"Nicht genug RAM für diese Operation! Benötigt: {memory_usage_bytes/1024:.1f}KB")
        
        # Simuliere Temperaturentwicklung
        # Temperaturanstieg ist abhängig von:
        # 1. CPU-Auslastung (höhere Auslastung = mehr Wärme)
        # 2. Ausführungszeit (längere Zeit = mehr Wärme)
        # 3. Speichernutzung (mehr Speicher = mehr Wärme, aber weniger Einfluss)
        
        # Basis-Temperaturanstieg durch CPU-Aktivität
        cpu_temp_rise = 0.02 * operation_time_ms / 100  # 0.02°C pro 100ms bei Vollauslastung
        
        # Zusätzlicher Anstieg durch Speichernutzung (geringerer Einfluss)
        memory_temp_rise = 0.005 * memory_usage_bytes / (1024 * 100)  # 0.005°C pro 100KB Speicher
        
        # Temperatur steigt an
        self.current_temperature_c += cpu_temp_rise + memory_temp_rise
        
        # Simuliere Wärmeabgabe an die Umgebung (abhängig von Temperaturdifferenz)
        ambient_temp = 25.0  # Angenommene Raumtemperatur
        temp_diff = self.current_temperature_c - ambient_temp
        cooling_rate = 0.001 * temp_diff  # Wärmeabgabe proportional zur Temperaturdifferenz
        
        # Temperatur sinkt durch Wärmeabgabe (aber nicht unter Raumtemperatur)
        self.current_temperature_c = max(ambient_temp, self.current_temperature_c - cooling_rate)
        
        # Simuliere Verarbeitungszeit
        time.sleep(operation_time_ms / 1000.0)
        
        # Zurück zum ursprünglichen Speicherzustand
        self.ram_used = original_ram_used
        
        # Temperatur für die Simulation zurückgeben
        return self.current_temperature_c
    
    @property
    def temperature(self):
        """Liefert die aktuelle simulierte Temperatur in Grad Celsius."""
        return self.current_temperature_c