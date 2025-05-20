"""
Tests für die EMU-01 Framebuilder-Korrektur.
Überprüft die korrekte Berücksichtigung des Kamera-Framebuffers im Speichermanagement.
"""

import unittest
import numpy as np
from enum import Enum
from typing import Dict, Optional

# Füge Importpfad für die Emulator-Module hinzu
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from emulation.frame_buffer import FrameBuffer, PixelFormat
    from emulation.emulator import RP2040Emulator, CameraEmulator
except ImportError:
    # Mock-Implementierungen für Tests, wenn die tatsächlichen Module nicht verfügbar sind
    class PixelFormat(Enum):
        RGB888 = 1
        RGB565 = 2
        GRAYSCALE = 3
        YUV422 = 4
    
    class FrameBuffer:
        def __init__(self, width, height, pixel_format=PixelFormat.RGB888):
            self.width = width
            self.height = height
            self.pixel_format = pixel_format
            self.total_size_bytes = width * height * (3 if pixel_format == PixelFormat.RGB888 else 
                                                    2 if pixel_format in (PixelFormat.RGB565, PixelFormat.YUV422) else 1)
            
        def get_statistics(self):
            return {'pixel_format': self.pixel_format.name, 'total_size_kb': self.total_size_bytes / 1024}
    
    class CameraEmulator:
        def __init__(self):
            self.width = 320
            self.height = 240
            self.rgb = True
            self.frame_buffer = FrameBuffer(self.width, self.height, PixelFormat.RGB888)
            self.frames_captured = 0
            
        def get_frame_buffer_size_bytes(self):
            return self.frame_buffer.total_size_bytes
            
        def get_frame_buffer_stats(self):
            return self.frame_buffer.get_statistics()
            
        def set_format(self, width, height, rgb=True):
            self.width = width
            self.height = height
            self.rgb = rgb
            pixel_format = PixelFormat.RGB888 if rgb else PixelFormat.GRAYSCALE
            self.frame_buffer = FrameBuffer(width, height, pixel_format)
    
    class PowerUsage:
        def __init__(self, sleep_mode_ma, idle_ma, active_ma, camera_active_ma, inference_ma):
            pass
    
    class AdaptiveMode(Enum):
        BALANCED = "balanced"
    
    class PowerManager:
        def __init__(self, emulator, power_usage, battery_capacity_mah, adaptive_mode):
            self.estimated_runtime_hours = 24.0
            
        def get_power_statistics(self):
            return {'mode': 'balanced', 'battery_capacity_mah': 1500, 'energy_consumed_mah': 0, 
                    'estimated_runtime_hours': 24.0, 'duty_cycle': 0.1, 'sampling_interval_s': 60, 
                    'activity_level': 0.5}
            
        def update_temperature(self, temp):
            pass
    
    class RP2040Emulator:
        def __init__(self, battery_capacity_mah=1500.0, adaptive_mode=AdaptiveMode.BALANCED):
            self.ram_size_bytes = 264 * 1024
            self.flash_size_bytes = 2 * 1024 * 1024
            self.system_ram_overhead = 40 * 1024
            self.camera = CameraEmulator()
            self.framebuffer_ram_bytes = self.camera.get_frame_buffer_size_bytes()
            self.firmware_loaded = False
            self.ram_used = 0
            self.power_manager = PowerManager(None, None, battery_capacity_mah, adaptive_mode)
            
        def load_firmware(self, firmware):
            total_ram_needed = firmware['ram_usage_bytes'] + self.system_ram_overhead + self.framebuffer_ram_bytes
            if total_ram_needed > self.ram_size_bytes:
                raise Exception(f"RAM-Überlauf: {total_ram_needed/1024:.1f}KB > {self.ram_size_bytes/1024:.1f}KB")
            self.firmware_loaded = True
            self.ram_used = firmware['ram_usage_bytes']
            return True
            
        def get_ram_usage(self):
            if not self.firmware_loaded:
                return self.system_ram_overhead + self.framebuffer_ram_bytes
            return self.ram_used + self.system_ram_overhead + self.framebuffer_ram_bytes
            
        def set_camera_format(self, width, height, pixel_format=PixelFormat.RGB888):
            is_rgb = pixel_format in (PixelFormat.RGB888, PixelFormat.RGB565)
            self.camera.set_format(width, height, is_rgb)
            self.framebuffer_ram_bytes = self.camera.get_frame_buffer_size_bytes()
            
        def set_camera_pixel_format(self, pixel_format):
            width = self.camera.width
            height = self.camera.height
            is_rgb = pixel_format in (PixelFormat.RGB888, PixelFormat.RGB565)
            self.camera.set_format(width, height, is_rgb)
            self.framebuffer_ram_bytes = self.camera.get_frame_buffer_size_bytes()

class TestFrameBufferRAM(unittest.TestCase):
    """Tests für die korrekte Berücksichtigung des Kamera-Framebuffers im Speichermanagement."""
    
    def test_initial_framebuffer_accounting(self):
        """Testet, ob der Framebuffer initial korrekt berücksichtigt wird."""
        emulator = RP2040Emulator()
        
        # Überprüfe, ob der Framebuffer in der Speicherberechnung berücksichtigt wird
        initial_ram_usage = emulator.get_ram_usage()
        self.assertGreater(initial_ram_usage, emulator.system_ram_overhead)
        
        # Berechne erwartete Framebuffer-Größe für RGB888 (320x240x3 = 230400 Bytes)
        expected_framebuffer_size = 320 * 240 * 3
        # Berücksichtige mögliche Paddings durch Speicherausrichtung
        self.assertGreaterEqual(emulator.framebuffer_ram_bytes, expected_framebuffer_size)
    
    def test_pixel_format_changes(self):
        """Testet, ob Änderungen des Pixelformats korrekt im RAM-Verbrauch reflektiert werden."""
        emulator = RP2040Emulator()
        
        # Initialer RAM-Verbrauch (sollte RGB888 sein)
        initial_ram = emulator.get_ram_usage()
        
        # Wechsel zu RGB565 (2 Bytes pro Pixel statt 3)
        emulator.set_camera_pixel_format(PixelFormat.RGB565)
        rgb565_ram = emulator.get_ram_usage()
        self.assertLess(rgb565_ram, initial_ram)  # Sollte weniger RAM benötigen
        
        # Prüfe ungefähre Größendifferenz (von 3 auf 2 Bytes pro Pixel)
        expected_reduction = 320 * 240  # 1 Byte Reduktion pro Pixel
        actual_reduction = initial_ram - rgb565_ram
        # Berücksichtige mögliche Paddings
        self.assertGreaterEqual(actual_reduction, expected_reduction * 0.9)
        
        # Wechsel zu Graustufen (1 Byte pro Pixel)
        emulator.set_camera_pixel_format(PixelFormat.GRAYSCALE)
        gray_ram = emulator.get_ram_usage()
        self.assertLess(gray_ram, rgb565_ram)  # Sollte noch weniger RAM benötigen
        
        # Wechsel zurück zu RGB888
        emulator.set_camera_pixel_format(PixelFormat.RGB888)
        final_ram = emulator.get_ram_usage()
        self.assertAlmostEqual(final_ram, initial_ram, delta=100)  # Sollte fast gleich sein
    
    def test_resolution_changes(self):
        """Testet, ob Änderungen der Auflösung korrekt im RAM-Verbrauch reflektiert werden."""
        emulator = RP2040Emulator()
        
        # Initialer RAM-Verbrauch (320x240 RGB888)
        initial_ram = emulator.get_ram_usage()
        
        # Verdoppele die Auflösung (640x480 RGB888)
        emulator.set_camera_format(640, 480, PixelFormat.RGB888)
        high_res_ram = emulator.get_ram_usage()
        
        # Sollte ca. 4x mehr Framebuffer-RAM benötigen
        expected_increase = 3 * (640 * 480 - 320 * 240)  # 3 Bytes pro zusätzlichem Pixel
        actual_increase = high_res_ram - initial_ram
        # Berücksichtige mögliche Paddings
        self.assertGreaterEqual(actual_increase, expected_increase * 0.9)
        
        # Reduziere auf niedrige Auflösung (160x120 RGB888)
        emulator.set_camera_format(160, 120, PixelFormat.RGB888)
        low_res_ram = emulator.get_ram_usage()
        
        # Sollte deutlich weniger als initial sein
        self.assertLess(low_res_ram, initial_ram)
    
    def test_firmware_loading_with_framebuffer(self):
        """Testet, ob der Framebuffer bei der Firmware-Validierung korrekt berücksichtigt wird."""
        emulator = RP2040Emulator()
        
        # Berechne verfügbaren RAM (abzüglich System-Overhead und Framebuffer)
        available_ram = emulator.ram_size_bytes - emulator.system_ram_overhead - emulator.framebuffer_ram_bytes
        
        # Firmware, die fast den gesamten verfügbaren RAM nutzt
        large_firmware = {
            'ram_usage_bytes': int(available_ram * 0.95),
            'total_size_bytes': 100 * 1024  # 100KB Flash-Verbrauch
        }
        
        # Diese Firmware sollte geladen werden können
        emulator.load_firmware(large_firmware)
        
        # Firmware, die zu viel RAM benötigt
        oversized_firmware = {
            'ram_usage_bytes': available_ram + 10 * 1024,  # 10KB zu viel
            'total_size_bytes': 100 * 1024
        }
        
        # Diese Firmware sollte einen Fehler verursachen
        with self.assertRaises(Exception):
            emulator.load_firmware(oversized_firmware)
    
    def test_framebuffer_formats_edge_cases(self):
        """Testet Grenzfälle der Framebuffer-Formate."""
        emulator = RP2040Emulator()
        
        # Test mit minimaler Auflösung
        emulator.set_camera_format(1, 1, PixelFormat.RGB888)
        min_res_ram = emulator.get_ram_usage()
        
        # Test mit Grenzwerten
        emulator.set_camera_format(1, 1, PixelFormat.GRAYSCALE)
        min_gray_ram = emulator.get_ram_usage()
        self.assertLess(min_gray_ram, min_res_ram)
        
        # Test mit nicht standard Dimensionen, die Padding erfordern
        emulator.set_camera_format(321, 241, PixelFormat.RGB888)  # 321x241 ist nicht durch 4 teilbar
        odd_ram = emulator.get_ram_usage()
        
        # Berechne erwartete Größe ohne Padding
        expected_odd_size = 321 * 241 * 3
        # Der tatsächliche Framebuffer sollte größer sein wegen Padding
        self.assertGreater(emulator.framebuffer_ram_bytes, expected_odd_size)
        
        # Teste Format mit Zeilenlänge, die ein Vielfaches von 4 ist
        emulator.set_camera_format(320, 240, PixelFormat.RGB565)  # 320*2 = 640 Bytes pro Zeile, durch 4 teilbar
        aligned_ram = emulator.get_ram_usage()
        
        # Berechne erwartete Größe
        expected_aligned_size = 320 * 240 * 2
        # Tatsächliche und berechnete Größe sollten übereinstimmen, da kein Padding notwendig ist
        self.assertEqual(emulator.framebuffer_ram_bytes, expected_aligned_size)
    
    def test_real_world_scenario(self):
        """Testet ein realistisches Szenario mit simulierter Firmware."""
        emulator = RP2040Emulator()
        
        # Setze realistisches Kameraformat (RGB565 ist üblich für embedded Systeme)
        emulator.set_camera_pixel_format(PixelFormat.RGB565)
        
        # Simuliere eine typische Firmware für RP2040 + OV2640
        typical_firmware = {
            'ram_usage_bytes': 90 * 1024,  # 90KB für Tensor Arena und andere Daten
            'total_size_bytes': 180 * 1024  # 180KB Flash-Verbrauch
        }
        
        # Lade die Firmware
        emulator.load_firmware(typical_firmware)
        
        # Berechne erwarteten Gesamt-RAM-Verbrauch
        expected_ram = (
            typical_firmware['ram_usage_bytes'] +  # Tensor Arena
            emulator.system_ram_overhead +         # System-Overhead
            emulator.framebuffer_ram_bytes         # Framebuffer
        )
        actual_ram = emulator.get_ram_usage()
        self.assertEqual(actual_ram, expected_ram)
        
        # Überprüfe, ob der RAM-Verbrauch unter dem Limit liegt
        self.assertLess(actual_ram, emulator.ram_size_bytes)
        
        # Berechne Auslastung in Prozent
        ram_usage_percent = (actual_ram / emulator.ram_size_bytes) * 100
        # Realistische Auslastung sollte zwischen 60% und 95% liegen
        self.assertGreater(ram_usage_percent, 60)
        self.assertLess(ram_usage_percent, 95)

if __name__ == '__main__':
    unittest.main()
