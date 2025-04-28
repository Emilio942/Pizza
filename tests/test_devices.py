"""
Unit-Tests für das Hardware-Management-Modul.
"""

import pytest
import time
import numpy as np

from src.devices import (
    PowerMode,
    DeviceStatus,
    BatteryStatus,
    Camera,
    SystemController,
    HardwareError
)

def test_battery_status():
    """Testet die Batteriestatusüberwachung."""
    battery = BatteryStatus()
    
    # Test Initialisierung
    assert battery.voltage_mv == battery.FULL_MV
    assert battery.capacity_percent == 100.0
    
    # Test Spannungsaktualisierung
    battery.update(2500)  # 50% zwischen CRITICAL und FULL
    assert 45 <= battery.capacity_percent <= 55  # Ungefähre Überprüfung
    assert not battery.is_critical()
    assert not battery.is_low()
    
    # Test niedrige Batterie
    battery.update(battery.LOW_MV)
    assert battery.is_low()
    assert not battery.is_critical()
    
    # Test kritische Batterie
    battery.update(battery.CRITICAL_MV)
    assert battery.is_low()
    assert battery.is_critical()
    assert battery.capacity_percent == 0.0

def test_camera():
    """Testet die Kamerasteuerung."""
    camera = Camera()
    
    # Test Initialisierung
    assert not camera.is_initialized
    assert camera.power_mode == PowerMode.SLEEP
    
    # Test Kamerainitialisierung
    assert camera.initialize()
    assert camera.is_initialized
    assert camera.power_mode == PowerMode.ACTIVE
    
    # Test Bildaufnahme
    frame = camera.capture_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (camera.height, camera.width)
    assert camera.frame_count == 1
    assert camera.error_count == 0
    
    # Test Deep Sleep
    camera.set_power_mode(PowerMode.DEEP_SLEEP)
    assert not camera.is_initialized
    with pytest.raises(HardwareError):
        camera.capture_frame()

def test_system_controller():
    """Testet den System-Controller."""
    system = SystemController()
    
    # Test Initialisierung
    assert system.initialize()
    assert system.status == DeviceStatus.OK
    assert system.power_mode == PowerMode.ACTIVE
    
    # Test Systemstatistiken
    stats = system.get_system_stats()
    assert isinstance(stats, dict)
    assert all(key in stats for key in [
        'status', 'power_mode', 'uptime_seconds',
        'battery_percent', 'battery_voltage_mv',
        'camera_frames', 'camera_errors',
        'last_inference_ms'
    ])
    
    # Test Energieverwaltung
    system.battery.update(system.battery.LOW_MV)
    system.update_power_state()
    assert system.status == DeviceStatus.LOW_BATTERY
    
    system.battery.update(system.battery.CRITICAL_MV)
    system.update_power_state()
    assert system.status == DeviceStatus.CRITICAL_BATTERY
    assert system.power_mode == PowerMode.DEEP_SLEEP

@pytest.mark.parametrize('mode', [
    PowerMode.ACTIVE,
    PowerMode.SLEEP,
    PowerMode.DEEP_SLEEP
])
def test_power_modes(mode):
    """Testet verschiedene Energiesparmodi."""
    system = SystemController()
    system.set_power_mode(mode)
    assert system.power_mode == mode
    assert system.camera.power_mode == mode

def test_error_handling():
    """Testet Fehlerbehandlung."""
    camera = Camera()
    
    # Test Bildaufnahme ohne Initialisierung
    with pytest.raises(HardwareError):
        camera.capture_frame()
    
    # Test Systeminitialisierung mit defekter Kamera
    def mock_failed_init(self):
        return False
    
    original_init = Camera.initialize
    Camera.initialize = mock_failed_init
    
    system = SystemController()
    assert not system.initialize()
    assert system.status == DeviceStatus.ERROR
    
    # Cleanup
    Camera.initialize = original_init

def test_system_timing():
    """Testet Systemzeitmessungen."""
    system = SystemController()
    system.initialize()
    
    # Test Uptime
    time.sleep(0.1)
    stats = system.get_system_stats()
    assert stats['uptime_seconds'] >= 0.1
    
    # Test Inference-Zeit
    system.last_inference_time = 0.123
    stats = system.get_system_stats()
    assert stats['last_inference_ms'] == 123.0  # 0.123s -> 123ms