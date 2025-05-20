"""
Hardware-Emulator für RP2040 und OV2040 Kamera.
"""

import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from enum import Enum

# Import from utils
from src.utils.constants import (
    RP2040_FLASH_SIZE_KB,
    RP2040_RAM_SIZE_KB,
    RP2040_CLOCK_SPEED_MHZ,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    INPUT_SIZE
)
# Use our simplified PowerManager instead of the original implementation
# from src.utils.power_manager import PowerManager, PowerUsage, AdaptiveMode
from .simple_power_manager import PowerManager, PowerUsage, AdaptiveMode

# Local imports
from .frame_buffer import FrameBuffer, PixelFormat
from .temperature_sensor import TemperatureSensor, SensorType
from .uart_emulator import UARTEmulator
from .logging_system import LoggingSystem, LogLevel, LogType

logger = logging.getLogger(__name__)

# Define these classes here since we can't import them
class ResourceError(Exception):
    """Fehler bei Ressourcenüberschreitung (RAM, Flash)."""
    pass

class HardwareError(Exception):
    """Fehler bei der Hardware-Emulation."""
    pass

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
        
        # Erstelle einen Framebuffer mit dem richtigen Format
        pixel_format = PixelFormat.RGB888 if self.rgb else PixelFormat.GRAYSCALE
        self.frame_buffer = FrameBuffer(self.width, self.height, pixel_format)
        
        logger.info(
            f"Kamera-Emulator initialisiert: {self.width}x{self.height}, "
            f"{'RGB' if self.rgb else 'Grayscale'}, {1000/self.frame_time:.1f} ms/Frame, "
            f"Framebuffer-Größe: {self.frame_buffer.total_size_bytes/1024:.1f} KB"
        )
    
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
        
        # Schreibe das Bild in den Framebuffer
        self.frame_buffer.begin_frame_write()
        self.frame_buffer.write_pixel_data(frame)
        self.frame_buffer.end_frame_write()
        
        self.frames_captured += 1
        self.last_capture = time.time()
        
        # Lese das Bild aus dem Framebuffer zurück
        return self.frame_buffer.get_frame_as_numpy()
    
    def set_format(self, width: int, height: int, rgb: bool = True) -> None:
        """Konfiguriert Bildformat."""
        self.width = width
        self.height = height
        self.rgb = rgb
        
        # Aktualisiere den Framebuffer mit dem neuen Format
        pixel_format = PixelFormat.RGB888 if rgb else PixelFormat.GRAYSCALE
        self.frame_buffer = FrameBuffer(width, height, pixel_format)
        
        logger.info(
            f"Kameraformat geändert: {width}x{height}, {'RGB' if rgb else 'Grayscale'}, "
            f"Framebuffer-Größe: {self.frame_buffer.total_size_bytes/1024:.1f} KB"
        )
    
    def get_frame_buffer_size_bytes(self) -> int:
        """Liefert die Größe des Framebuffers in Bytes."""
        return self.frame_buffer.total_size_bytes
    
    def get_frame_buffer_stats(self) -> Dict:
        """Liefert Statistiken über den Framebuffer."""
        return self.frame_buffer.get_statistics()

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
        
        # Erstelle die Kamera mit Framebuffer
        self.camera = CameraEmulator()
        
        # Frame Buffer RAM wird jetzt explizit verfolgt
        self.framebuffer_ram_bytes = self.camera.get_frame_buffer_size_bytes()
        
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
        
        # Initialisiere UART für Logging
        self.uart = UARTEmulator(log_to_file=True, log_dir="output/emulator_logs")
        self.uart.initialize(baudrate=115200)
        
        # Initialisiere Temperatursensor
        self.temperature_sensor = TemperatureSensor(
            sensor_type=SensorType.INTERNAL,  # Internen Sensor des RP2040 verwenden
            accuracy=0.5,                    # Genauigkeit in °C
            update_interval=1.0,             # Minimales Intervall in Sekunden
            noise_level=0.1                  # Rauschen in den Messwerten
        )
        self.temperature_sensor.initialize()
        self.current_temperature_c = self.temperature_sensor.read_temperature()
        
        # Initialisiere Logging-System
        self.logging_system = LoggingSystem(
            uart=self.uart, 
            log_to_file=True,
            log_dir="output/emulator_logs"
        )
        self.logging_system.log("RP2040 Emulator initialized", LogLevel.INFO, LogType.SYSTEM)
        
        # Aktiviere periodisches Temperaturlogging
        self.last_temp_log_time = time.time()
        self.temp_log_interval = 60.0  # Log temperature every 60 seconds
        
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
        logger.info(f"Framebuffer-Größe: {self.framebuffer_ram_bytes/1024:.1f}KB")
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
        
        # Prüfe, ob der RAM-Bedarf erfüllt werden kann (inklusive System-Overhead und Framebuffer)
        total_ram_needed = firmware['ram_usage_bytes'] + self.system_ram_overhead + self.framebuffer_ram_bytes
        if total_ram_needed > self.ram_size_bytes:
            logger.error(
                f"RAM-Überlauf: {total_ram_needed/1024:.1f}KB > {self.ram_size_bytes/1024:.1f}KB "
                f"(Modell: {firmware['ram_usage_bytes']/1024:.1f}KB, "
                f"System: {self.system_ram_overhead/1024:.1f}KB, "
                f"Framebuffer: {self.framebuffer_ram_bytes/1024:.1f}KB)"
            )
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
            # Nur System-Overhead und Framebuffer
            return self.system_ram_overhead + self.framebuffer_ram_bytes
        
        # Verwende die Werte, die in load_firmware gesetzt wurden, plus Framebuffer
        base_ram = self.ram_used + self.system_ram_overhead + self.framebuffer_ram_bytes
        
        # Wenn im Sleep-Mode, reduziere RAM-Nutzung (außer Framebuffer)
        if self.sleep_mode:
            # Framebuffer-Größe bleibt, andere RAM-Nutzung wird reduziert
            reduced_ram = (self.ram_used + self.system_ram_overhead) * (1 - self.sleep_ram_reduction)
            return int(reduced_ram + self.framebuffer_ram_bytes)
        
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
            # Detaillierte RAM-Aufschlüsselung für Debugging
            model_ram = self.ram_used if self.firmware_loaded else 0
            framebuffer_kb = self.framebuffer_ram_bytes / 1024
            system_overhead_kb = self.system_ram_overhead / 1024
            model_ram_kb = model_ram / 1024
            total_kb = ram_usage / 1024
            
            raise ResourceError(
                f"RAM-Überlauf: {total_kb:.1f}KB > {self.ram_size_bytes/1024:.1f}KB\n"
                f"Aufschlüsselung: Modell {model_ram_kb:.1f}KB + "
                f"System {system_overhead_kb:.1f}KB + "
                f"Framebuffer {framebuffer_kb:.1f}KB"
            )
        
        if flash_usage > self.flash_size_bytes:
            raise ResourceError(
                f"Flash-Überlauf: {flash_usage/1024:.1f}KB > {self.flash_size_bytes/1024:.1f}KB"
            )
    
    def get_system_stats(self) -> Dict:
        """Liefert Systemstatistiken."""
        ram_usage = self.get_ram_usage()
        flash_usage = self.get_flash_usage()
        
        # Berechne RAM-Aufschlüsselung
        model_ram = self.ram_used if self.firmware_loaded else 0
        system_ram = self.system_ram_overhead
        framebuffer_ram = self.framebuffer_ram_bytes
        
        # Aktualisiere Temperatur
        current_temperature = self.read_temperature()
        
        # Basisstatistiken
        stats = {
            'uptime_seconds': time.time() - self.start_time,
            'ram_used_kb': ram_usage / 1024,
            'ram_free_kb': (self.ram_size_bytes - ram_usage) / 1024,
            'model_ram_kb': model_ram / 1024,
            'system_ram_kb': system_ram / 1024,
            'framebuffer_ram_kb': framebuffer_ram / 1024,
            'flash_used_kb': flash_usage / 1024,
            'flash_free_kb': (self.flash_size_bytes - flash_usage) / 1024,
            'firmware_loaded': self.firmware_loaded,
            'last_inference_ms': self.inference_time * 1000,
            'camera_frames': self.camera.frames_captured,
            'sleep_mode': self.sleep_mode,
            'total_sleep_time': self.total_sleep_time,
            'current_temperature_c': current_temperature
        }
        
        # Füge Kamera-Framebuffer-Statistiken hinzu
        framebuffer_stats = self.camera.get_frame_buffer_stats()
        stats.update({
            'camera_width': self.camera.width,
            'camera_height': self.camera.height,
            'camera_pixel_format': framebuffer_stats['pixel_format'],
            'framebuffer_total_size_kb': framebuffer_stats['total_size_kb'],
            'framebuffer_frames_processed': framebuffer_stats['frames_processed'],
            'framebuffer_frames_dropped': framebuffer_stats['frames_dropped']
        })
        
        # Füge Temperatur-Sensorstatistiken hinzu
        temp_stats = self.temperature_sensor.get_stats()
        stats.update({
            'temperature_sensor_type': temp_stats['sensor_type'],
            'temperature_readings_count': temp_stats['readings_count'],
            'temperature_min_c': temp_stats['min_temperature'],
            'temperature_max_c': temp_stats['max_temperature'],
            'temperature_avg_c': temp_stats['avg_temperature']
        })
        
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
    
    def set_camera_format(self, width: int, height: int, pixel_format: PixelFormat = PixelFormat.RGB888) -> None:
        """
        Konfiguriert das Kameraformat und aktualisiert den Framebuffer.
        
        Args:
            width: Kamerabreite in Pixeln
            height: Kamerahöhe in Pixeln
            pixel_format: Pixelformat (RGB888, RGB565, GRAYSCALE, YUV422)
        """
        # Aktualisiere Kamera und Framebuffer
        is_rgb = pixel_format in (PixelFormat.RGB888, PixelFormat.RGB565)
        self.camera.set_format(width, height, is_rgb)
        
        # Aktualisiere Framebuffer-Größe im Speichermanagement
        old_framebuffer_size = self.framebuffer_ram_bytes
        self.framebuffer_ram_bytes = self.camera.get_frame_buffer_size_bytes()
        
        logger.info(
            f"Kameraformat geändert auf {width}x{height} ({pixel_format.name}). "
            f"Framebuffer-Größe: {self.framebuffer_ram_bytes/1024:.1f}KB "
            f"(vorher: {old_framebuffer_size/1024:.1f}KB)"
        )
        
        # Prüfe, ob die neue Framebuffer-Größe zu einem RAM-Überlauf führen würde
        if self.firmware_loaded:
            total_ram_needed = self.ram_used + self.system_ram_overhead + self.framebuffer_ram_bytes
            if total_ram_needed > self.ram_size_bytes:
                logger.warning(
                    f"WARNUNG: Neues Kameraformat könnte RAM-Überlauf verursachen: "
                    f"{total_ram_needed/1024:.1f}KB > {self.ram_size_bytes/1024:.1f}KB"
                )
    
    def set_camera_pixel_format(self, pixel_format: PixelFormat) -> None:
        """
        Ändert das Pixelformat des Kamera-Framebuffers.
        
        Args:
            pixel_format: Neues Pixelformat (RGB888, RGB565, GRAYSCALE, YUV422)
        """
        # Speichere aktuelle Dimensionen
        width = self.camera.width
        height = self.camera.height
        
        # Setze Kameraformat mit neuem Pixelformat
        is_rgb = pixel_format in (PixelFormat.RGB888, PixelFormat.RGB565)
        
        # Erstelle einen neuen Framebuffer mit dem neuen Format
        old_buffer = self.camera.frame_buffer
        self.camera.frame_buffer = FrameBuffer(width, height, pixel_format)
        
        # Aktualisiere RGB-Flag für die Kamera basierend auf dem Format
        self.camera.rgb = is_rgb
        
        # Aktualisiere die Framebuffer-Größe im Speichermanagement
        old_framebuffer_size = self.framebuffer_ram_bytes
        self.framebuffer_ram_bytes = self.camera.get_frame_buffer_size_bytes()
        
        logger.info(
            f"Kamera-Pixelformat geändert auf {pixel_format.name}. "
            f"Framebuffer-Größe: {self.framebuffer_ram_bytes/1024:.1f}KB "
            f"(vorher: {old_framebuffer_size/1024:.1f}KB)"
        )
        
        # Überprüfe Speichernutzung nach Formatänderung
        self.validate_resources()
        
        # Gib Statistiken über den alten und neuen Framebuffer aus
        old_stats = old_buffer.get_memory_layout()
        new_stats = self.camera.frame_buffer.get_memory_layout()
        
        logger.debug(
            f"Speicherlayout-Änderung:\n"
            f"Alt: {old_stats['total_size_bytes']} Bytes, {old_stats['aligned_row_bytes']} Bytes/Zeile\n"
            f"Neu: {new_stats['total_size_bytes']} Bytes, {new_stats['aligned_row_bytes']} Bytes/Zeile"
        )
    
    def read_temperature(self) -> float:
        """
        Liest die aktuelle Temperatur vom Sensor.
        Diese Methode wird von der Firmware aufgerufen, um die Temperatur auszulesen.
        
        Returns:
            Die aktuelle Temperatur in Grad Celsius
        """
        try:
            # Lese den Temperatursensor aus
            temperature = self.temperature_sensor.read_temperature()
            
            # Aktualisiere die interne Temperatur
            self.current_temperature_c = temperature
            
            # Aktualisiere PowerManager
            self.power_manager.update_temperature(temperature)
            
            # Überprüfe, ob ein Temperaturlogging fällig ist
            current_time = time.time()
            if current_time - self.last_temp_log_time >= self.temp_log_interval:
                self.log_temperature()
                self.last_temp_log_time = current_time
            
            return temperature
        
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Temperatur: {e}")
            # Bei Fehler, gib den letzten bekannten Wert zurück
            return self.current_temperature_c
    
    def log_temperature(self) -> None:
        """
        Loggt die aktuelle Temperatur sowohl über das Logging-System als auch über UART.
        """
        temperature = self.current_temperature_c
        sensor_type = self.temperature_sensor.sensor_type.value
        
        # Logge Temperatur über das Logging-System
        self.logging_system.log_temperature(temperature, sensor_type)
        
        # Logge auch Gesamtstatistik für Temperatursensor
        sensor_stats = self.temperature_sensor.get_stats()
        if sensor_stats["readings_count"] > 0:
            stats_msg = (
                f"Temperatur-Statistik: "
                f"Min {sensor_stats['min_temperature']:.1f}°C, "
                f"Max {sensor_stats['max_temperature']:.1f}°C, "
                f"Avg {sensor_stats['avg_temperature']:.1f}°C, "
                f"Messungen: {sensor_stats['readings_count']}"
            )
            self.logging_system.log(stats_msg, LogLevel.DEBUG, LogType.TEMPERATURE)
    
    def set_temperature_log_interval(self, interval_seconds: float) -> None:
        """
        Setzt das Intervall für periodisches Temperaturlogging.
        
        Args:
            interval_seconds: Intervall in Sekunden (0 = deaktiviert)
        """
        self.temp_log_interval = max(0.0, interval_seconds)
        logger.info(f"Temperatur-Log-Intervall auf {self.temp_log_interval:.1f}s gesetzt")
        
        if self.temp_log_interval > 0:
            self.logging_system.log(
                f"Temperatur-Logging aktiviert (Intervall: {self.temp_log_interval:.1f}s)",
                LogLevel.INFO,
                LogType.SYSTEM
            )
        else:
            self.logging_system.log("Periodisches Temperatur-Logging deaktiviert", LogLevel.INFO, LogType.SYSTEM)
    
    def inject_temperature_spike(self, delta: float, duration: float = 60.0) -> None:
        """
        Injiziert einen künstlichen Temperaturanstieg für Testzwecke.
        
        Args:
            delta: Temperaturanstieg in °C
            duration: Ungefähre Dauer des Anstiegs in Sekunden
        """
        self.temperature_sensor.inject_temperature_spike(delta, duration)
        self.logging_system.log(
            f"Temperatur-Spike: +{delta:.1f}°C für ~{duration:.0f}s injiziert",
            LogLevel.WARNING,
            LogType.SYSTEM
        )
    
    def close(self) -> None:
        """
        Schließt den Emulator und gibt alle Ressourcen frei.
        Diese Methode sollte aufgerufen werden, wenn der Emulator nicht mehr benötigt wird.
        """
        if hasattr(self, 'uart') and self.uart:
            self.uart.close()
        
        if hasattr(self, 'logging_system') and self.logging_system:
            self.logging_system.close()
        
        # Logge Abschlussinformationen
        logger.info(f"RP2040 Emulator wird beendet. Laufzeit: {(time.time() - self.start_time):.1f}s")
        
        if self.sleep_mode:
            # Wenn im Sleep-Modus, wecke zuerst auf, um korrekte Statistiken zu bekommen
            self.wake_up()
        
        # Logge abschließende Statistiken
        stats = self.get_system_stats()
        
        logger.info(f"Abschlussstatistiken:")
        logger.info(f"  Temperatur: {stats['current_temperature_c']:.1f}°C")
        logger.info(f"  RAM-Nutzung: {stats['ram_used_kb']:.1f}KB / {self.ram_size_bytes/1024:.1f}KB")
        logger.info(f"  Flash-Nutzung: {stats['flash_used_kb']:.1f}KB / {self.flash_size_bytes/1024:.1f}KB")
        logger.info(f"  Verarbeitete Frames: {stats['camera_frames']}")
        logger.info(f"  Energieverbrauch: {stats['energy_consumed_mah']:.2f}mAh")
        logger.info(f"  Schlafzeit: {self.total_sleep_time:.1f}s")
        
        # Speichere Temperaturverlauf in separatem Log
        if hasattr(self, 'temperature_sensor') and self.temperature_sensor:
            temp_history = [(t, temp) for t, temp in self.temperature_sensor.reading_history]
            if temp_history:
                logger.info(f"Temperaturverlauf: {len(temp_history)} Messwerte erfasst")