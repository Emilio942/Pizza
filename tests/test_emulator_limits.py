"""
Tests für Grenzfälle und Ressourcenlimits des Hardware-Emulators.
"""

import pytest
import time
import numpy as np
from pathlib import Path

from src.emulator import CameraEmulator, RP2040Emulator
from src.exceptions import HardwareError, ResourceError
from src.constants import (
    RP2040_FLASH_SIZE_KB,
    RP2040_RAM_SIZE_KB,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)

def test_memory_limits():
    """Testet Speicherlimits."""
    emulator = RP2040Emulator()
    
    # Test: Zu große Firmware
    large_firmware = {
        'path': 'large.bin',
        'total_size_bytes': RP2040_FLASH_SIZE_KB * 1024 * 2,  # 2x Flash
        'model_size_bytes': RP2040_FLASH_SIZE_KB * 1024,
        'ram_usage_bytes': 50 * 1024,  # 50KB
        'model_input_size': (48, 48)
    }
    
    # Sollte fehlschlagen
    with pytest.raises(ResourceError):
        emulator.load_firmware(large_firmware)
    
    # Test: Zu hoher RAM-Bedarf
    high_ram_firmware = {
        'path': 'ram_heavy.bin',
        'total_size_bytes': 100 * 1024,  # 100KB
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': RP2040_RAM_SIZE_KB * 1024,  # Kompletter RAM
        'model_input_size': (48, 48)
    }
    
    # Sollte fehlschlagen
    with pytest.raises(ResourceError):
        emulator.load_firmware(high_ram_firmware)

def test_camera_edge_cases():
    """Testet Kamera-Grenzfälle."""
    camera = CameraEmulator()
    
    # Kamera initialisieren, bevor sie verwendet wird
    camera.initialize()
    
    # Test: Extremes Bildformat
    camera.set_format(1, 1, rgb=True)
    frame = camera.capture_frame()
    assert frame.shape == (1, 1, 3)
    
    camera.set_format(1024, 1024, rgb=False)
    frame = camera.capture_frame()
    assert frame.shape == (1024, 1024, 1)
    
    # Test: Schnelle Bildaufnahmen
    start = time.time()
    frames = []
    for _ in range(10):
        frames.append(camera.capture_frame())
    elapsed = time.time() - start
    
    # Prüfe Framerate-Begrenzung
    assert elapsed >= (10 * camera.frame_time)
    assert len(frames) == 10

def test_power_management():
    """Testet Energiemanagement."""
    emulator = RP2040Emulator()
    
    # Lade eine Test-Firmware
    test_firmware = {
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    emulator.load_firmware(test_firmware)
    
    # Test: Deep Sleep
    initial_stats = emulator.get_system_stats()
    emulator.enter_sleep_mode()
    sleep_stats = emulator.get_system_stats()
    
    assert sleep_stats['ram_used_kb'] < initial_stats['ram_used_kb']
    
    # Test: Aufwecken
    emulator.wake_up()
    wake_stats = emulator.get_system_stats()
    assert wake_stats['ram_used_kb'] == initial_stats['ram_used_kb']

def test_long_running_operation():
    """Testet Langzeitbetrieb."""
    emulator = RP2040Emulator()
    
    # Lade Test-Firmware
    test_firmware = {
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    emulator.load_firmware(test_firmware)
    
    # Simuliere lange Laufzeit
    test_image = np.zeros((32, 32, 3), dtype=np.uint8)
    start_stats = emulator.get_system_stats()
    
    for _ in range(100):
        emulator.simulate_inference(test_image)
        time.sleep(0.01)  # Kleine Pause
    
    end_stats = emulator.get_system_stats()
    
    # Prüfe Ressourcenstabilität
    assert abs(end_stats['ram_used_kb'] - start_stats['ram_used_kb']) < 1.0
    assert end_stats['flash_used_kb'] == start_stats['flash_used_kb']

def test_concurrent_operations():
    """Testet gleichzeitige Operationen."""
    emulator = RP2040Emulator()
    camera = emulator.camera
    
    # Lade Test-Firmware
    test_firmware = {
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    emulator.load_firmware(test_firmware)
    
    # Simuliere gleichzeitige Bildaufnahme und Inferenz
    camera.initialize()
    test_image = camera.capture_frame()
    
    # Dies sollte funktionieren, da RP2040 dual-core ist
    result = emulator.simulate_inference(test_image)
    assert result['success']
    
    # Prüfe, ob Kamera weiterhin funktioniert
    next_frame = camera.capture_frame()
    assert isinstance(next_frame, np.ndarray)

def test_error_recovery():
    """Testet Fehlerbehandlung und Wiederherstellung."""
    emulator = RP2040Emulator()
    
    # Test: Wiederherstellung nach Firmware-Fehler
    with pytest.raises(ResourceError):
        emulator.load_firmware({
            'path': 'invalid.bin',
            'total_size_bytes': RP2040_FLASH_SIZE_KB * 1024 * 2,
            'model_size_bytes': RP2040_FLASH_SIZE_KB * 1024,
            'ram_usage_bytes': 40 * 1024,
            'model_input_size': (48, 48)
        })
    
    # System sollte sich erholen
    valid_firmware = {
        'path': 'valid.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    }
    assert emulator.load_firmware(valid_firmware)
    
    # Test: Wiederherstellung nach Kamera-Fehler
    emulator.camera.initialized = False
    with pytest.raises(HardwareError):
        emulator.camera.capture_frame()
    
    # Kamera sollte sich nach Neuinitialisierung erholen
    assert emulator.camera.initialize()
    frame = emulator.camera.capture_frame()
    assert isinstance(frame, np.ndarray)