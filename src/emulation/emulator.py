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
from datetime import datetime

# Import from utils
from ..constants import (
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
from .voltage_sensor import VoltageSensor, VoltageSensorType
from .uart_emulator import UARTEmulator
from .sd_card_emulator import SDCardEmulator
from .logging_system import LoggingSystem, LogLevel, LogType
from .ov2640_timing_emulator import OV2640TimingEmulator
from .rp2040_dma_emulator import RP2040DMAEmulator, DVPInterfaceEmulator

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
    """Emuliert OV2640 Kamera mit detaillierter Timing-Simulation."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.rgb = True
        self.initialized = False
        self.frames_captured = 0
        self.startup_time = 0.1  # 100ms Startup-Zeit
        self.frame_time = 1.0 / 7  # ~7 FPS
        self.last_capture = 0
        
        # Pizza detection specific settings
        self.pizza_detection_width = 48
        self.pizza_detection_height = 48
        
        # Initialize OV2640 timing emulator
        self.ov2640_emulator = OV2640TimingEmulator(log_dir)
        
        # Erstelle einen Framebuffer mit dem richtigen Format
        pixel_format = PixelFormat.RGB888 if self.rgb else PixelFormat.GRAYSCALE
        self.frame_buffer = FrameBuffer(self.width, self.height, pixel_format)
        
        logger.info(
            f"Kamera-Emulator initialisiert: {self.width}x{self.height}, "
            f"{'RGB' if self.rgb else 'Grayscale'}, {1000/self.frame_time:.1f} ms/Frame, "
            f"Framebuffer-Größe: {self.frame_buffer.total_size_bytes/1024:.1f} KB"
        )
    
    def initialize(self) -> bool:
        """Emuliert vollständige OV2640-Initialisierung mit Timing."""
        if not self.initialized:
            logger.info("Starte OV2640 Initialisierung mit Timing-Emulation...")
            
            # Run the detailed initialization sequence with timing
            success = self.ov2640_emulator.emulate_camera_init_sequence()
            
            if success:
                self.initialized = True
                self.last_capture = time.time()
                
                # Configure for pizza detection (48x48 RGB565)
                self.configure_for_pizza_detection()
                
                logger.info("OV2640 Initialisierung erfolgreich abgeschlossen")
            else:
                logger.error("OV2640 Initialisierung fehlgeschlagen")
            
            return success
        return True
    
    def configure_for_pizza_detection(self):
        """Konfiguriert die Kamera speziell für Pizza-Erkennung."""
        logger.info("Konfiguriere Kamera für Pizza-Erkennung (48x48 RGB565)...")
        
        # Set format to RGB565 for pizza detection
        self.width = self.pizza_detection_width
        self.height = self.pizza_detection_height
        self.rgb = True
        
        # Update framebuffer for new resolution
        pixel_format = PixelFormat.RGB565  # Use RGB565 for efficiency
        self.frame_buffer = FrameBuffer(self.width, self.height, pixel_format)
        
        # Log the configuration change
        self.ov2640_emulator.log_timing_event("CONFIG_PIZZA", 
            f"Camera configured for pizza detection: {self.width}x{self.height} RGB565")
    
    def capture_frame(self) -> np.ndarray:
        """Emuliert Bildaufnahme mit OV2640 Timing."""
        if not self.initialized:
            raise HardwareError("Kamera nicht initialisiert")
        
        # Simuliere Framerate-Begrenzung
        elapsed = time.time() - self.last_capture
        if (elapsed < self.frame_time):
            time.sleep(self.frame_time - elapsed)
        
        # Emulate the actual frame capture with timing
        format_name = "RGB565" if self.rgb else "GRAYSCALE"
        success = self.ov2640_emulator.emulate_frame_capture(
            self.width, self.height, format_name
        )
        
        if not success:
            raise HardwareError("Frame capture failed")
        
        # Generiere simuliertes Bild (für Pizza-Detektion optimiert)
        if self.width == 48 and self.height == 48:
            # Generate pizza-like patterns for testing
            frame = self._generate_pizza_test_image()
        else:
            # Standard random image
            channels = 3 if self.rgb else 1
            frame = np.random.randint(0, 256, (self.height, self.width, channels), dtype=np.uint8)
        
        # Use DMA transfer if DMA emulator is available, otherwise use CPU-based transfer
        if hasattr(self, 'dma_emulator') and self.dma_emulator:
            success = self._capture_frame_with_dma(frame)
            if not success:
                logger.warning("DMA transfer failed, falling back to CPU-based transfer")
                self._capture_frame_cpu_based(frame)
        else:
            self._capture_frame_cpu_based(frame)
        
        self.frames_captured += 1
        self.last_capture = time.time()
        
        # Lese das Bild aus dem Framebuffer zurück
        return self.frame_buffer.get_frame_as_numpy()
    
    def _generate_pizza_test_image(self) -> np.ndarray:
        """Generiert ein Test-Bild das einer Pizza ähnelt."""
        # Create a simple pizza-like test pattern
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Pizza base (brownish)
        frame[:, :, 0] = 139  # Red
        frame[:, :, 1] = 69   # Green
        frame[:, :, 2] = 19   # Blue
        
        # Add some random "toppings" (red and green spots)
        num_toppings = np.random.randint(3, 8)
        for _ in range(num_toppings):
            x = np.random.randint(5, self.width - 5)
            y = np.random.randint(5, self.height - 5)
            color = [255, 0, 0] if np.random.rand() > 0.5 else [0, 255, 0]  # Red or green
            frame[y-2:y+3, x-2:x+3] = color
        
        return frame
    
    def get_timing_summary(self) -> Dict:
        """Liefert eine Zusammenfassung der Timing-Messungen."""
        return self.ov2640_emulator.get_timing_summary()
    
    def save_timing_logs(self):
        """Speichert detaillierte Timing-Logs."""
        self.ov2640_emulator.save_detailed_log()
    
    def get_frame_buffer_size_bytes(self) -> int:
        """Liefert die Größe des Framebuffers in Bytes."""
        return self.frame_buffer.total_size_bytes
    
    def get_frame_buffer_stats(self) -> Dict:
        """Liefert Statistiken über den Framebuffer."""
        return self.frame_buffer.get_statistics()

    def set_format(self, width: int, height: int, rgb: bool = True) -> None:
        """Ändert das Kameraformat."""
        self.width = width
        self.height = height
        self.rgb = rgb
        
        # Erstelle neuen Framebuffer mit neuem Format
        pixel_format = PixelFormat.RGB565 if rgb else PixelFormat.GRAYSCALE
        self.frame_buffer = FrameBuffer(width, height, pixel_format)
        
        logger.info(f"Kameraformat geändert auf {width}x{height}, {'RGB565' if rgb else 'GRAYSCALE'}")
    
    def enable_dma_mode(self, dma_emulator, dvp_interface):
        """Enables DMA mode for camera data transfers."""
        self.dma_emulator = dma_emulator
        self.dvp_interface = dvp_interface
        logger.info("DMA mode enabled for camera captures")
    
    def get_dma_statistics(self) -> Dict:
        """Gets DMA transfer statistics if DMA mode is enabled."""
        if hasattr(self, 'dma_emulator') and self.dma_emulator:
            return self.dma_emulator.get_statistics()
        return {"dma_enabled": False}
    

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
        
        # Initialize DMA controller emulator (12-channel DMA controller of RP2040)
        self.dma_emulator = RP2040DMAEmulator()
        
        # Initialize DVP interface emulator for camera data capture
        self.dvp_interface = DVPInterfaceEmulator()
        
        # Connect camera to DMA/DVP interface
        self.camera.dma_emulator = self.dma_emulator
        self.camera.dvp_interface = self.dvp_interface
        
        logger.info("DMA controller and DVP interface initialized")
        logger.info(f"DMA channels available: {len(self.dma_emulator.channels)}")
        
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
        
        # Initialisiere SD-Karte
        self.sd_card = SDCardEmulator(root_dir="output/sd_card", capacity_mb=1024)
        self.sd_card.initialize()
        self.sd_card.mount()
        logger.info(f"SD-Karte initialisiert und gemountet: {self.sd_card.get_status()}")
        
        # Initialisiere Temperatursensor
        self.temperature_sensor = TemperatureSensor(
            sensor_type=SensorType.INTERNAL,  # Internen Sensor des RP2040 verwenden
            accuracy=0.5,                    # Genauigkeit in °C
            update_interval=1.0,             # Minimales Intervall in Sekunden
            noise_level=0.1                  # Rauschen in den Messwerten
        )
        self.temperature_sensor.initialize()
        self.current_temperature_c = self.temperature_sensor.read_temperature()
        
        # Initialisiere Spannungssensoren
        self.voltage_sensor_vdd = VoltageSensor(
            sensor_type=VoltageSensorType.INTERNAL,  # Interne VDD-Spannung des RP2040
            accuracy_mv=10.0,                        # Genauigkeit in mV
            update_interval=1.0,                     # Minimales Intervall in Sekunden
            noise_level_mv=2.0                       # Rauschen in den Messwerten
        )
        self.voltage_sensor_vdd.initialize()
        
        self.voltage_sensor_battery = VoltageSensor(
            sensor_type=VoltageSensorType.BATTERY,   # Batteriespannung
            accuracy_mv=15.0,                        # Genauigkeit in mV
            update_interval=5.0,                     # Minimales Intervall in Sekunden
            noise_level_mv=5.0                       # Rauschen in den Messwerten
        )
        self.voltage_sensor_battery.initialize()
        
        self.current_vdd_voltage_mv = self.voltage_sensor_vdd.read_voltage()
        self.current_battery_voltage_mv = self.voltage_sensor_battery.read_voltage()
        
        # Initialisiere Logging-System
        self.logging_system = LoggingSystem(
            uart=self.uart, 
            log_to_file=True,
            log_to_sd=True,
            sd_card=self.sd_card,
            log_dir="output/emulator_logs"
        )
        self.logging_system.log("RP2040 Emulator initialized", LogLevel.INFO, LogType.SYSTEM)
        
        # Aktiviere periodisches Temperaturlogging
        self.last_temp_log_time = time.time()
        self.temp_log_interval = 60.0  # Log temperature every 60 seconds
        
        # Initialisiere Performance-Metrics-Datei auf SD-Karte
        if self.sd_card and self.sd_card.mounted:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.sd_performance_metrics_file = f"logs/performance_metrics_{timestamp}.csv"
            self.sd_performance_metrics_handle = self.sd_card.open_file(self.sd_performance_metrics_file, "w")
            
            if self.sd_performance_metrics_handle:
                # Schreibe CSV-Header
                header = "Timestamp,InferenceTime,PeakRamUsage,CpuLoad,Temperature,VddVoltage,BatteryVoltage,Prediction,Confidence\n"
                self.sd_card.write_file(self.sd_performance_metrics_handle, header)
                logger.info(f"Performance-Metrics-Logging auf SD-Karte aktiviert: {self.sd_performance_metrics_file}")
            else:
                logger.warning("Konnte Performance-Metrics-Datei auf SD-Karte nicht öffnen")
        
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
        
        # Initialize new trigger systems for ENERGIE-2.2 adaptive duty-cycle logic
        from .motion_sensor import MotionSensorController
        from .rtc_scheduler import ScheduleManager
        from .interrupt_emulator import InterruptController
        from .adaptive_state_machine import DutyCycleStateMachine
        
        # Initialize trigger controllers
        self.motion_controller = MotionSensorController()
        self.schedule_manager = ScheduleManager()
        self.interrupt_controller = InterruptController()
        
        # Initialize enhanced adaptive state machine
        self.adaptive_state_machine = DutyCycleStateMachine(
            emulator=self,
            power_manager=self.power_manager,
            motion_controller=self.motion_controller,
            schedule_manager=self.schedule_manager,
            interrupt_controller=self.interrupt_controller
        )
        
        # Start trigger systems
        self.motion_controller.start()
        self.schedule_manager.start()
        
        # Setup common interrupts
        self.interrupt_controls = self.interrupt_controller.setup_common_interrupts()
        
        # Start adaptive state machine
        self.adaptive_state_machine.start()
        
        # Initialisiere die Temperatur im PowerManager
        self.power_manager.update_temperature(self.current_temperature_c)
        
        # Initialize adaptive clock frequency management (HWEMU-2.2)
        self.adaptive_clock_enabled = True
        self.adaptive_clock_config = {
            'enabled': True,
            'update_interval_ms': 1000,  # Update every second
            'verbose_logging': False
        }
        
        # Temperature thresholds for clock adjustment (°C)
        self.temp_thresholds = {
            'low': 40.0,        # Below this: maximum performance (133 MHz)
            'medium': 60.0,     # Above this: balanced mode (100 MHz)
            'high': 75.0,       # Above this: conservative mode (75 MHz)
            'critical': 85.0    # Above this: emergency mode (48 MHz)
        }
        
        # Clock frequencies (MHz)
        self.clock_frequencies = {
            'max': 133,         # Maximum performance
            'balanced': 100,    # Balanced performance/thermal
            'conservative': 75, # Conservative mode
            'emergency': 48     # Emergency thermal protection
        }
        
        # Adaptive clock state
        self.current_frequency_mhz = self.cpu_speed_mhz  # Track current frequency
        self.target_frequency_mhz = self.cpu_speed_mhz   # Target frequency
        self.last_clock_adjustment_time = time.time()
        self.last_adjustment_direction = None  # 'up', 'down', or None
        self.thermal_protection_active = False
        self.total_clock_adjustments = 0
        self.emergency_mode_activations = 0
        self.temp_hysteresis = 2.0  # 2°C hysteresis to prevent oscillation
        
        logger.info(f"Adaptive clock frequency management initialized")
        logger.info(f"  Temperature thresholds: {self.temp_thresholds}")
        logger.info(f"  Clock frequencies: {self.clock_frequencies}")
        
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
        logger.info("Adaptive duty-cycle trigger systems initialized")
    
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
        
        # Aktualisiere die Temperatur, um realistische Werte zu simulieren
        # Nach einer Inferenz steigt die Temperatur leicht an
        self.current_temperature_c = self.temperature_sensor.read_temperature()
        
        # Logge Performance-Metriken auf SD-Karte
        self._log_performance_metrics_to_sd(
            inference_time_us=int(inference_time * 1000000),
            ram_usage_bytes=self.get_ram_usage(),
            cpu_load_percent=int(self.power_manager.get_current_cpu_load() * 100),
            temperature_c=int(self.current_temperature_c * 100),
            prediction=current_class_id,
            confidence=int(confidence * 100)
        )
        
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
        """Versetzt den Emulator in den optimierten Sleep-Mode für maximale Energieeinsparung."""
        if not self.sleep_mode:
            # Performance measurement - start timing
            sleep_transition_start = time.perf_counter()
            
            logger.info("Emulator geht in optimierten Sleep-Mode")
            self.sleep_mode = True
            self.sleep_start_time = time.time()
            
            # Speichere aktuellen Zustand für schnelle Wiederherstellung
            self.original_ram_used = self.ram_used
            self._save_peripheral_states()
            
            # Systematisches Herunterfahren aller Peripheriegeräte für maximale Energieeinsparung
            self._shutdown_peripherals()
            
            # Reduziere RAM-Verbrauch im Sleep-Mode (deaktivierte Komponenten)
            # Erhöhe Reduktion für tieferen Sleep-Zustand
            enhanced_sleep_reduction = min(0.8, self.sleep_ram_reduction + 0.2)  # 80% max reduction
            self.ram_used = int(self.original_ram_used * (1 - enhanced_sleep_reduction))
            
            # Aktualisiere den PowerManager
            active_duration = time.time() - self.last_detection_time if self.last_detection_time > 0 else 0
            if active_duration > 0:
                self.power_manager.update_energy_consumption(active_duration, True)
                self.power_manager.sleep_start_time = time.time()
            
            # Performance measurement - end timing
            sleep_transition_time = (time.perf_counter() - sleep_transition_start) * 1000
            logger.debug(f"Sleep transition completed in {sleep_transition_time:.3f}ms")
            
            # Store transition time for performance tracking
            if not hasattr(self, 'sleep_transition_times'):
                self.sleep_transition_times = []
            self.sleep_transition_times.append(sleep_transition_time)
    
    def wake_up(self) -> None:
        """Weckt den Emulator aus dem optimierten Sleep-Mode auf mit schneller Wiederherstellung."""
        if self.sleep_mode:
            # Performance measurement - start timing
            wake_transition_start = time.perf_counter()
            
            # Berechne Schlafzeit für Energieberechnung
            sleep_duration = time.time() - self.sleep_start_time
            self.total_sleep_time += sleep_duration
            
            # Schnelle Wiederherstellung des RAM-Verbrauchs
            self.ram_used = self.original_ram_used
            
            # Schnelle Wiederherstellung der Peripheriegeräte
            self._restore_peripherals()
            
            # Aktualisiere den PowerManager mit der Schlafzeit
            self.power_manager.update_energy_consumption(sleep_duration, False)
            self.power_manager.total_sleep_time += sleep_duration
            self.power_manager.last_wakeup_time = time.time()
            
            # Performance measurement - end timing
            wake_transition_time = (time.perf_counter() - wake_transition_start) * 1000
            logger.info(f"Emulator aufgeweckt (war {sleep_duration:.2f}s im Sleep-Mode, Wake-up: {wake_transition_time:.3f}ms)")
            
            # Store transition time for performance tracking
            if not hasattr(self, 'wake_transition_times'):
                self.wake_transition_times = []
            self.wake_transition_times.append(wake_transition_time)
            
            # Clear sleep mode flag first
            self.sleep_mode = False
            
            # Verifiziere erfolgreiche Wiederherstellung
            self._verify_wake_up_restoration()
    
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
            
            # Aktualisiere adaptive Taktfrequenz basierend auf Temperatur (HWEMU-2.2)
            self.update_adaptive_clock_frequency()
            
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
    
    def set_temperature_for_testing(self, temperature: float) -> None:
        """
        Setzt die Temperatur direkt für Testzwecke.
        Dies überschreibt die normale Temperatur-Simulation und 
        löst sofort ein Update der adaptiven Taktfrequenz aus.
        
        Args:
            temperature: Gewünschte Temperatur in °C
        """
        # Setze die Temperatur im Sensor
        self.temperature_sensor.set_temperature_for_testing(temperature)
        
        # Aktualisiere die interne Temperatur
        self.current_temperature_c = temperature
        
        # Aktualisiere PowerManager
        self.power_manager.update_temperature(temperature)
        
        # Triggere sofort ein Update der adaptiven Taktfrequenz
        self.update_adaptive_clock_frequency()
        
        logger.debug(f"Emulator-Temperatur für Tests auf {temperature:.1f}°C gesetzt")
    
    def _determine_target_frequency(self, temperature: float) -> int:
        """
        Bestimmt die Ziel-Taktfrequenz basierend auf der Temperatur.
        
        Args:
            temperature: Aktuelle Temperatur in °C
            
        Returns:
            Ziel-Taktfrequenz in MHz
        """
        # Wende Hysterese an, um Oszillation zu verhindern
        temp_adjusted = temperature
        if self.last_adjustment_direction == 'up':
            temp_adjusted -= self.temp_hysteresis
        elif self.last_adjustment_direction == 'down':
            temp_adjusted += self.temp_hysteresis
        
        # Bestimme Frequenz basierend auf Temperaturschwellen
        if temp_adjusted >= self.temp_thresholds['critical']:
            self.thermal_protection_active = True
            return self.clock_frequencies['emergency']
        elif temp_adjusted >= self.temp_thresholds['high']:
            self.thermal_protection_active = True
            return self.clock_frequencies['emergency']
        elif temp_adjusted >= self.temp_thresholds['medium']:
            self.thermal_protection_active = False
            return self.clock_frequencies['conservative']
        elif temp_adjusted >= self.temp_thresholds['low']:
            self.thermal_protection_active = False
            return self.clock_frequencies['balanced']
        else:
            self.thermal_protection_active = False
            return self.clock_frequencies['max']
    
    def _apply_clock_frequency_change(self, target_freq: int) -> bool:
        """
        Wendet eine Taktfrequenz-Änderung an.
        
        Args:
            target_freq: Ziel-Taktfrequenz in MHz
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if target_freq == self.current_frequency_mhz:
            return True  # Bereits auf Zielfrequenz
        
        # Logge die Frequenzänderung
        logger.info(f"[ADAPTIVE_CLOCK] Changing system clock: {self.current_frequency_mhz} MHz -> {target_freq} MHz")
        
        # Simuliere das Ändern der Systemtaktfrequenz
        # In einer echten Implementierung würde hier die RP2040 SDK verwendet
        old_freq = self.current_frequency_mhz
        self.current_frequency_mhz = target_freq
        self.cpu_speed_mhz = target_freq  # Aktualisiere auch die CPU-Geschwindigkeit für Inferenz-Simulation
        
        # Bestimme Anpassungsrichtung
        if target_freq > old_freq:
            self.last_adjustment_direction = 'up'
        else:
            self.last_adjustment_direction = 'down'
        
        # Aktualisiere Statistiken
        self.total_clock_adjustments += 1
        
        if target_freq == self.clock_frequencies['emergency']:
            self.emergency_mode_activations += 1
            logger.warning(f"[ADAPTIVE_CLOCK] EMERGENCY: Thermal protection activated at {self.current_temperature_c:.1f}°C")
        
        # Logge die erfolgreiche Änderung
        if self.adaptive_clock_config['verbose_logging']:
            logger.info(f"[ADAPTIVE_CLOCK] Frequency adjusted: {target_freq} MHz (temp: {self.current_temperature_c:.1f}°C)")
        
        # Logge über das Logging-System
        self.logging_system.log(
            f"Clock frequency changed: {old_freq} MHz -> {target_freq} MHz (temp: {self.current_temperature_c:.1f}°C)",
            LogLevel.INFO,
            LogType.SYSTEM
        )
        
        return True
    
    def update_adaptive_clock_frequency(self) -> bool:
        """
        Aktualisiert die adaptive Taktfrequenz basierend auf der aktuellen Temperatur.
        
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if not self.adaptive_clock_enabled or not self.adaptive_clock_config['enabled']:
            return True
        
        current_time = time.time()
        
        # Prüfe, ob genug Zeit seit der letzten Anpassung vergangen ist
        time_since_last = (current_time - self.last_clock_adjustment_time) * 1000  # ms
        if time_since_last < self.adaptive_clock_config['update_interval_ms']:
            return True  # Noch nicht Zeit für Update
        
        # Lese aktuelle Temperatur
        temperature = self.current_temperature_c
        
        # Bestimme Zielfrequenz
        target_freq = self._determine_target_frequency(temperature)
        self.target_frequency_mhz = target_freq
        
        # Prüfe, ob Frequenzanpassung nötig ist
        if target_freq != self.current_frequency_mhz:
            if self._apply_clock_frequency_change(target_freq):
                # Erfolgreiche Anpassung
                pass
            else:
                logger.error(f"[ADAPTIVE_CLOCK] ERROR: Failed to set frequency to {target_freq} MHz")
                return False
        elif self.adaptive_clock_config['verbose_logging'] and int(current_time) % 10 == 0:
            # Logge Status alle 10 Sekunden wenn verbose
            logger.debug(f"[ADAPTIVE_CLOCK] Status: {self.current_frequency_mhz} MHz, {temperature:.1f}°C")
        
        self.last_clock_adjustment_time = current_time
        return True
    
    def get_adaptive_clock_state(self) -> Dict:
        """
        Liefert den aktuellen Zustand des adaptiven Taktfrequenz-Systems.
        
        Returns:
            Dictionary mit dem aktuellen Zustand
        """
        return {
            'current_frequency_mhz': self.current_frequency_mhz,
            'target_frequency_mhz': self.target_frequency_mhz,
            'last_temperature': self.current_temperature_c,
            'thermal_protection_active': self.thermal_protection_active,
            'total_adjustments': self.total_clock_adjustments,
            'emergency_activations': self.emergency_mode_activations,
            'enabled': self.adaptive_clock_enabled and self.adaptive_clock_config['enabled']
        }
    
    def get_adaptive_clock_stats(self) -> Dict:
        """
        Liefert Statistiken des adaptiven Taktfrequenz-Systems.
        
        Returns:
            Dictionary mit Statistiken
        """
        return {
            'total_adjustments': self.total_clock_adjustments,
            'emergency_activations': self.emergency_mode_activations,
            'current_frequency_mhz': self.current_frequency_mhz,
            'current_temperature': self.current_temperature_c,
            'thermal_protection_active': self.thermal_protection_active
        }
    
    def force_clock_frequency(self, freq_mhz: int) -> bool:
        """
        Erzwingt eine bestimmte Taktfrequenz (für Tests).
        
        Args:
            freq_mhz: Ziel-Taktfrequenz in MHz
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        logger.info(f"[ADAPTIVE_CLOCK] Forcing frequency to {freq_mhz} MHz")
        
        if self._apply_clock_frequency_change(freq_mhz):
            self.target_frequency_mhz = freq_mhz
            return True
        
        return False
    
    def set_adaptive_clock_enabled(self, enabled: bool) -> None:
        """
        Aktiviert oder deaktiviert das adaptive Taktfrequenz-Management.
        
        Args:
            enabled: True zum Aktivieren, False zum Deaktivieren
        """
        self.adaptive_clock_enabled = enabled
        logger.info(f"[ADAPTIVE_CLOCK] Adaptive clock management {'enabled' if enabled else 'disabled'}")
        
        self.logging_system.log(
            f"Adaptive clock management {'enabled' if enabled else 'disabled'}",
            LogLevel.INFO,
            LogType.SYSTEM
        )
    
    def is_thermal_protection_active(self) -> bool:
        """
        Prüft, ob der thermische Schutz aktiv ist.
        
        Returns:
            True wenn thermischer Schutz aktiv, False sonst
        """
        return self.thermal_protection_active
    
    def close(self) -> None:
        """
        Schließt den Emulator und gibt alle Ressourcen frei.
        Diese Methode sollte aufgerufen werden, wenn der Emulator nicht mehr benötigt wird.
        """
        # Stop adaptive trigger systems first
        if hasattr(self, 'adaptive_state_machine') and self.adaptive_state_machine:
            self.adaptive_state_machine.stop()
            logger.info("Adaptive state machine stopped")
        
        if hasattr(self, 'motion_controller') and self.motion_controller:
            self.motion_controller.stop()
            logger.info("Motion controller stopped")
        
        if hasattr(self, 'schedule_manager') and self.schedule_manager:
            self.schedule_manager.stop()
            logger.info("Schedule manager stopped")
        
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
    
    def _log_performance_metrics_to_sd(self, 
                              inference_time_us: int, 
                              ram_usage_bytes: int, 
                              cpu_load_percent: int, 
                              temperature_c: int, 
                              prediction: int, 
                              confidence: int) -> bool:
        """
        Loggt Performance-Metriken auf die SD-Karte im CSV-Format.
        
        Args:
            inference_time_us: Inferenzzeit in Mikrosekunden
            ram_usage_bytes: RAM-Nutzung in Bytes
            cpu_load_percent: CPU-Last in Prozent (0-100)
            temperature_c: Temperatur in 1/100 Grad Celsius
            prediction: Klassifikationsergebnis (Klassen-ID)
            confidence: Konfidenz in Prozent (0-100)
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not hasattr(self, 'sd_performance_metrics_handle') or not self.sd_performance_metrics_handle:
            return False
            
        if not self.sd_card or not self.sd_card.mounted:
            return False
            
        try:
            # Lese aktuelle Spannungswerte
            vdd_voltage_mv = self.voltage_sensor_vdd.read_voltage()
            battery_voltage_mv = self.voltage_sensor_battery.read_voltage()
            
            # Erzeuge CSV-Zeile: Timestamp,InferenceTime,PeakRamUsage,CpuLoad,Temperature,VddVoltage,BatteryVoltage,Prediction,Confidence
            timestamp = int(time.time() * 1000)  # Millisekunden seit Epoch
            csv_line = f"{timestamp},{inference_time_us},{ram_usage_bytes},{cpu_load_percent},{temperature_c},{int(vdd_voltage_mv)},{int(battery_voltage_mv)},{prediction},{confidence}\n"
            
            # Schreibe in SD-Karten-Datei
            self.sd_card.write_file(self.sd_performance_metrics_handle, csv_line)
            
            return True
        except Exception as e:
            logger.error(f"Fehler beim Loggen von Performance-Metriken auf SD-Karte: {e}")
            return False
    
    def log_performance_metrics(self, inference_time_ms: float, peak_ram_kb: float, 
                               cpu_load: float, prediction: float, confidence: float) -> None:
        """
        Logs performance metrics to both logging system and SD card.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            peak_ram_kb: Peak RAM usage in kilobytes
            cpu_load: CPU load in percentage
            prediction: Prediction result (0 or 1 for pizza/not pizza)
            confidence: Confidence score (0.0-1.0)
        """
        # Get current temperature
        temperature = self.read_temperature()
        
        # Create metrics dictionary for logging system
        metrics = {
            'inference_time_ms': inference_time_ms,
            'peak_ram_kb': peak_ram_kb,
            'cpu_usage_percent': cpu_load,
            'temperature_c': temperature,
            'prediction': prediction,
            'confidence': confidence
        }
        
        # Log to the logging system
        self.logging_system.log_performance(metrics)
        
        # Log to SD card performance metrics file if available
        if self.sd_card and self.sd_card.mounted and hasattr(self, 'sd_performance_metrics_handle'):
            timestamp = datetime.now().isoformat(timespec='milliseconds')
            log_entry = f"{timestamp},{inference_time_ms:.2f},{peak_ram_kb:.1f},{cpu_load:.1f},{temperature:.2f},{prediction:.0f},{confidence:.4f}\n"
            self.sd_card.write_file(self.sd_performance_metrics_handle, log_entry)
            
            logger.debug(f"Performance metrics logged to SD card: {log_entry.strip()}")
    
    def _save_peripheral_states(self) -> None:
        """Speichert den Zustand aller Peripheriegeräte für schnelle Wiederherstellung."""
        if not hasattr(self, 'peripheral_states'):
            self.peripheral_states = {}
        
        # Speichere Kamera-Zustand
        self.peripheral_states['camera_initialized'] = self.camera.initialized
        self.peripheral_states['camera_fps'] = getattr(self.camera, 'max_fps', 10)
        
        # Speichere Sensor-Zustände
        self.peripheral_states['temp_sensor_active'] = self.temperature_sensor.is_initialized if hasattr(self.temperature_sensor, 'is_initialized') else True
        self.peripheral_states['voltage_sensors_active'] = True
        
        # Speichere UART-Zustand (minimal für Logging)
        self.peripheral_states['uart_active'] = getattr(self.uart, 'initialized', True)
        
        # Speichere SD-Karte-Zustand
        self.peripheral_states['sd_mounted'] = self.sd_card.mounted if hasattr(self.sd_card, 'mounted') else False
        
        logger.debug("Peripheral states saved for sleep mode")
    
    def _shutdown_peripherals(self) -> None:
        """Fährt alle nicht-essentiellen Peripheriegeräte für maximale Energieeinsparung herunter."""
        # Kamera deaktivieren (größter Energieverbraucher)
        if hasattr(self.camera, 'initialized') and self.camera.initialized:
            self.camera.initialized = False
            logger.debug("Camera shut down for sleep mode")
        
        # Reduziere Sensor-Aktivität auf Minimum
        # Temperatursensor bleibt minimal aktiv für Sicherheit
        if hasattr(self.temperature_sensor, 'set_low_power_mode'):
            self.temperature_sensor.set_low_power_mode(True)
        
        # Spannungssensoren reduzieren
        # Batteriespannung minimal überwachen für kritische Zustände
        if hasattr(self.voltage_sensor_vdd, 'set_low_power_mode'):
            self.voltage_sensor_vdd.set_low_power_mode(True)
        if hasattr(self.voltage_sensor_battery, 'set_low_power_mode'):
            self.voltage_sensor_battery.set_low_power_mode(True)
        
        # UART bleibt minimal aktiv für kritische Logs
        # SD-Karte kann in Standby gehen
        if hasattr(self.sd_card, 'enter_standby'):
            self.sd_card.enter_standby()
        
        logger.debug("Peripherals shut down for deep sleep mode")
    
    def _restore_peripherals(self) -> None:
        """Stellt alle Peripheriegeräte nach dem Aufwachen schnell wieder her."""
        if not hasattr(self, 'peripheral_states'):
            logger.warning("No peripheral states saved, using defaults")
            self.peripheral_states = {}
        
        # Kamera schnell reaktivieren
        if self.peripheral_states.get('camera_initialized', True):
            self.camera.initialized = True
            logger.debug("Camera restored from sleep mode")
        
        # Sensoren reaktivieren
        if hasattr(self.temperature_sensor, 'set_low_power_mode'):
            self.temperature_sensor.set_low_power_mode(False)
        
        if hasattr(self.voltage_sensor_vdd, 'set_low_power_mode'):
            self.voltage_sensor_vdd.set_low_power_mode(False)
        if hasattr(self.voltage_sensor_battery, 'set_low_power_mode'):
            self.voltage_sensor_battery.set_low_power_mode(False)
        
        # SD-Karte reaktivieren
        if hasattr(self.sd_card, 'exit_standby'):
            self.sd_card.exit_standby()
        
        logger.debug("Peripherals restored from sleep mode")
    
    def _verify_wake_up_restoration(self) -> bool:
        """Verifiziert, dass alle Systemkomponenten nach dem Aufwachen korrekt funktionieren."""
        verification_success = True
        
        # Überprüfe Kamera-Zustand - sollte dem ursprünglich gespeicherten Zustand entsprechen
        if hasattr(self, 'peripheral_states') and 'camera_initialized' in self.peripheral_states:
            expected_camera_state = self.peripheral_states['camera_initialized']
            actual_camera_state = getattr(self.camera, 'initialized', False)
            if actual_camera_state != expected_camera_state:
                logger.warning(f"Camera restoration verification failed: expected {expected_camera_state}, got {actual_camera_state}")
                verification_success = False
        
        # Überprüfe RAM-Wiederherstellung
        ram_restored = abs(self.ram_used - self.original_ram_used) < 1024  # 1KB Toleranz
        if not ram_restored:
            logger.warning(f"RAM restoration verification failed: expected {self.original_ram_used}, got {self.ram_used}")
            verification_success = False
        
        # Überprüfe Sleep-Mode-Flag (sollte bereits False sein zu diesem Zeitpunkt)
        if self.sleep_mode:
            logger.warning("Sleep mode flag still active after wake-up")
            verification_success = False
        
        # Überprüfe Temperatursensor
        try:
            temp = self.temperature_sensor.read_temperature()
            if temp <= 0:
                logger.warning("Temperature sensor verification failed")
                verification_success = False
        except Exception as e:
            logger.warning(f"Temperature sensor verification error: {e}")
            verification_success = False
        
        if verification_success:
            logger.debug("Wake-up restoration verification successful")
        else:
            logger.error("Wake-up restoration verification failed")
        
        return verification_success
    
    def get_sleep_performance_metrics(self) -> Dict:
        """Liefert Performance-Metriken für Sleep-Wake-Zyklen."""
        metrics = {
            'sleep_transition_times': getattr(self, 'sleep_transition_times', []),
            'wake_transition_times': getattr(self, 'wake_transition_times', []),
            'total_sleep_cycles': len(getattr(self, 'sleep_transition_times', [])),
            'total_sleep_time': self.total_sleep_time
        }
        
        if metrics['sleep_transition_times']:
            metrics['avg_sleep_transition_ms'] = sum(metrics['sleep_transition_times']) / len(metrics['sleep_transition_times'])
            metrics['max_sleep_transition_ms'] = max(metrics['sleep_transition_times'])
        
        if metrics['wake_transition_times']:
            metrics['avg_wake_transition_ms'] = sum(metrics['wake_transition_times']) / len(metrics['wake_transition_times'])
            metrics['max_wake_transition_ms'] = max(metrics['wake_transition_times'])
            metrics['wake_time_under_10ms'] = all(t < 10.0 for t in metrics['wake_transition_times'])
        
        return metrics
    
    def _capture_frame_with_dma(self, frame: np.ndarray) -> bool:
        """Captures camera frame using DMA transfer (emulated)."""
        try:
            # Configure DMA channel for camera data transfer
            channel_id = 0  # Use DMA channel 0 for camera data
            
            # Calculate buffer size based on frame format
            bytes_per_pixel = 2 if self.rgb else 1  # RGB565 = 2, GRAYSCALE = 1
            buffer_size = self.width * self.height * bytes_per_pixel
            
            # Generate DVP data (simulate camera data from DVP interface)
            dvp_data = self.dvp_interface.generate_camera_data(
                width=self.width, 
                height=self.height,
                pixel_format="RGB565" if self.rgb else "GRAYSCALE",
                test_pattern="pizza" if (self.width == 48 and self.height == 48) else "random"
            )
            
            # Configure DMA channel for camera data transfer
            from .rp2040_dma_emulator import DMAChannelConfig, DMATransferSize, DMARequest
            
            config = DMAChannelConfig(
                read_addr=0x50000000,  # DVP FIFO register address (simulated)
                write_addr=0x20000000,  # RAM framebuffer address (simulated)
                trans_count=buffer_size // 4,  # Transfer in 32-bit words
                data_size=DMATransferSize.SIZE_32,
                treq_sel=DMARequest.TREQ_DVP_FIFO,
                chain_to=0,
                incr_read=False,   # DVP FIFO is at fixed address
                incr_write=True,   # Increment RAM address
                enable=True
            )
            
            # Configure and start DMA transfer
            success = self.dma_emulator.configure_channel(channel_id, config)
            if not success:
                return False
            
            # Start DMA transfer
            transfer_id = self.dma_emulator.start_transfer(
                channel_id=channel_id,
                source_data=dvp_data,
                description=f"Camera frame capture {self.width}x{self.height}"
            )
            
            if transfer_id is None:
                return False
            
            # Wait for DMA transfer completion (simulate)
            success = self.dma_emulator.wait_for_completion(transfer_id, timeout_ms=100)
            
            if success:
                # Verify data integrity
                integrity_ok = self.dma_emulator.verify_transfer_integrity(transfer_id)
                if integrity_ok:
                    logger.debug(f"DMA transfer completed successfully for frame {self.frames_captured}")
                    
                    # Copy the DMA transferred data to the framebuffer
                    transfer_data = self.dma_emulator.get_transfer_data(transfer_id)
                    if transfer_data:
                        self.frame_buffer.begin_frame_write()
                        self.frame_buffer.write_pixel_data(transfer_data)
                        self.frame_buffer.end_frame_write()
                    
                    return True
                else:
                    logger.error("DMA transfer data integrity check failed")
                    return False
            else:
                logger.error("DMA transfer timeout")
                return False
                
        except Exception as e:
            logger.error(f"DMA capture failed: {e}")
            return False
    
    def _capture_frame_cpu_based(self, frame: np.ndarray):
        """Fallback CPU-based frame capture (original method)."""
        # Schreibe das Bild in den Framebuffer
        self.frame_buffer.begin_frame_write()
        self.frame_buffer.write_pixel_data(frame)
        self.frame_buffer.end_frame_write()