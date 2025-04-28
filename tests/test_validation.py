"""
Unit-Tests für das Validierungsmodul.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from src.validation import (
    validate_image,
    validate_model_config,
    validate_camera_config,
    validate_firmware,
    validate_resource_usage,
    validate_hardware_status,
    validate_dataset_path,
    validate_config_file,
    validate_output_path,
    validate_model_input,
    check_system_compatibility
)
from src.types import (
    CameraConfig,
    FirmwareInfo,
    ResourceUsage,
    HardwareStatus
)
from src.exceptions import (
    InvalidInputError,
    ConfigError,
    ResourceError,
    HardwareError
)
from src.constants import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CLASS_NAMES
)

def test_validate_image():
    """Testet Bildvalidierung."""
    # Test: Gültiges RGB-Bild
    valid_image = np.zeros((64, 64, 3), dtype=np.uint8)
    validate_image(valid_image)  # Sollte keine Exception werfen
    
    # Test: Gültiges Graustufenbild
    valid_gray = np.zeros((64, 64), dtype=np.uint8)
    validate_image(valid_gray)
    
    # Test: Ungültige Eingabe
    with pytest.raises(InvalidInputError):
        validate_image([1, 2, 3])  # Keine numpy array
    
    # Test: Falsche Dimensionen
    invalid_dims = np.zeros((64, 64, 5))  # 5 Kanäle
    with pytest.raises(InvalidInputError):
        validate_image(invalid_dims)
    
    # Test: Zu kleine Bildgröße
    small_image = np.zeros((16, 16, 3))
    with pytest.raises(InvalidInputError):
        validate_image(small_image, min_size=32)
    
    # Test: Zu große Bildgröße
    large_image = np.zeros((2048, 2048, 3))
    with pytest.raises(InvalidInputError):
        validate_image(large_image, max_size=1024)

def test_validate_model_config():
    """Testet Modellkonfigurationsvalidierung."""
    # Test: Gültige Konfiguration
    valid_config = {
        'input_size': 48,
        'num_classes': len(CLASS_NAMES),
        'conv_channels': [16, 32, 64],
        'fc_features': [256, 128]
    }
    validate_model_config(valid_config)
    
    # Test: Fehlende Schlüssel
    invalid_config = {
        'input_size': 48,
        'num_classes': len(CLASS_NAMES)
    }
    with pytest.raises(ConfigError):
        validate_model_config(invalid_config)
    
    # Test: Ungültige Eingabegröße
    invalid_size = valid_config.copy()
    invalid_size['input_size'] = 0
    with pytest.raises(ConfigError):
        validate_model_config(invalid_size)
    
    # Test: Falsche Klassenanzahl
    invalid_classes = valid_config.copy()
    invalid_classes['num_classes'] = len(CLASS_NAMES) + 1
    with pytest.raises(ConfigError):
        validate_model_config(invalid_classes)

def test_validate_camera_config():
    """Testet Kamerakonfigurationsvalidierung."""
    # Test: Gültige Konfiguration
    valid_config = CameraConfig(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        format='RGB565',
        exposure=0.5,
        gain=1.0
    )
    validate_camera_config(valid_config)
    
    # Test: Ungültige Auflösung
    invalid_res = valid_config._replace(width=320, height=240)
    with pytest.raises(ConfigError):
        validate_camera_config(invalid_res)
    
    # Test: Zu hohe FPS
    invalid_fps = valid_config._replace(fps=CAMERA_FPS + 1)
    with pytest.raises(ConfigError):
        validate_camera_config(invalid_fps)
    
    # Test: Ungültiges Format
    invalid_format = valid_config._replace(format='RGB888')
    with pytest.raises(ConfigError):
        validate_camera_config(invalid_format)

def test_validate_firmware():
    """Testet Firmware-Validierung."""
    # Test: Gültige Firmware
    valid_firmware = FirmwareInfo(
        version='1.0.0',
        build_date='2025-04-10',
        size_bytes=100 * 1024,  # 100KB
        entry_point=0x10000000,
        model_size_bytes=150 * 1024,  # 150KB
        crc32=0x12345678
    )
    validate_firmware(valid_firmware)
    
    # Test: Zu große Firmware
    invalid_size = valid_firmware._replace(
        size_bytes=3 * 1024 * 1024  # 3MB
    )
    with pytest.raises(ResourceError):
        validate_firmware(invalid_size)
    
    # Test: Zu großes Modell
    invalid_model = valid_firmware._replace(
        model_size_bytes=200 * 1024  # 200KB
    )
    with pytest.raises(ResourceError):
        validate_firmware(invalid_model)

def test_validate_resource_usage():
    """Testet Ressourcennutzungsvalidierung."""
    # Test: Gültige Nutzung
    valid_usage = ResourceUsage(
        ram_used_kb=50.0,
        flash_used_kb=1024.0,
        cpu_usage_percent=50.0,
        power_mw=100.0
    )
    validate_resource_usage(valid_usage)
    
    # Test: RAM-Überlauf
    invalid_ram = valid_usage._replace(ram_used_kb=150.0)
    with pytest.raises(ResourceError):
        validate_resource_usage(invalid_ram)
    
    # Test: Flash-Überlauf
    invalid_flash = valid_usage._replace(flash_used_kb=3072.0)
    with pytest.raises(ResourceError):
        validate_resource_usage(invalid_flash)

def test_validate_hardware_status():
    """Testet Hardware-Statusvalidierung."""
    # Test: Gültiger Status
    valid_status = HardwareStatus(
        temperature_c=25.0,
        voltage_mv=3300,
        current_ma=50.0,
        power_mode='normal',
        error_count=0
    )
    validate_hardware_status(valid_status)
    
    # Test: Zu hohe Temperatur
    invalid_temp = valid_status._replace(temperature_c=90.0)
    with pytest.raises(HardwareError):
        validate_hardware_status(invalid_temp)
    
    # Test: Zu niedrige Spannung
    invalid_voltage = valid_status._replace(voltage_mv=2800)
    with pytest.raises(HardwareError):
        validate_hardware_status(invalid_voltage)
    
    # Test: Zu viele Fehler
    invalid_errors = valid_status._replace(error_count=11)
    with pytest.raises(HardwareError):
        validate_hardware_status(invalid_errors)

@pytest.fixture
def temp_dataset():
    """Temporäres Datensatzverzeichnis."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Erstelle Testbilder
    (temp_dir / 'test.jpg').touch()
    (temp_dir / 'test.png').touch()
    
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_validate_dataset_path(temp_dataset):
    """Testet Datensatzpfadvalidierung."""
    # Test: Gültiger Pfad
    validate_dataset_path(temp_dataset)
    
    # Test: Nicht existierender Pfad
    with pytest.raises(InvalidInputError):
        validate_dataset_path(temp_dataset / 'nonexistent')
    
    # Test: Datei statt Verzeichnis
    with pytest.raises(InvalidInputError):
        validate_dataset_path(temp_dataset / 'test.jpg')
    
    # Test: Leeres Verzeichnis
    empty_dir = temp_dataset / 'empty'
    empty_dir.mkdir()
    with pytest.raises(InvalidInputError):
        validate_dataset_path(empty_dir)

@pytest.fixture
def temp_config():
    """Temporäre Konfigurationsdatei."""
    temp_dir = Path(tempfile.mkdtemp())
    config_file = temp_dir / 'config.json'
    
    valid_config = {
        'model': {
            'input_size': 48,
            'num_classes': len(CLASS_NAMES)
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(valid_config, f)
    
    yield config_file
    shutil.rmtree(temp_dir)

def test_validate_config_file(temp_config):
    """Testet Konfigurationsdateivalidierung."""
    # Test: Gültige Konfiguration
    config = validate_config_file(temp_config)
    assert isinstance(config, dict)
    assert 'model' in config
    
    # Test: Nicht existierende Datei
    with pytest.raises(ConfigError):
        validate_config_file(temp_config.parent / 'nonexistent.json')
    
    # Test: Ungültiges JSON
    with open(temp_config, 'w') as f:
        f.write('invalid json')
    with pytest.raises(ConfigError):
        validate_config_file(temp_config)
    
    # Test: Ungültiges Format (Array statt Dict)
    with open(temp_config, 'w') as f:
        json.dump([1, 2, 3], f)
    with pytest.raises(ConfigError):
        validate_config_file(temp_config)

def test_validate_output_path(tmp_path):
    """Testet Ausgabepfadvalidierung."""
    # Test: Neuer Pfad
    new_path = tmp_path / 'output' / 'test.txt'
    validate_output_path(new_path)
    assert new_path.parent.exists()
    
    # Test: Existierender Pfad
    existing_path = tmp_path / 'existing.txt'
    existing_path.touch()
    validate_output_path(existing_path)  # Sollte nur warnen

def test_validate_model_input():
    """Testet Modelleingabevalidierung."""
    # Test: Gültige Eingabe
    valid_input = np.zeros((1, 48, 48, 3))
    validate_model_input(valid_input, (48, 48, 3))
    
    # Test: Ungültiger Typ
    with pytest.raises(InvalidInputError):
        validate_model_input([1, 2, 3], (48, 48, 3))
    
    # Test: Falsche Form
    invalid_shape = np.zeros((1, 32, 32, 3))
    with pytest.raises(InvalidInputError):
        validate_model_input(invalid_shape, (48, 48, 3))
    
    # Test: Ungültige Werte
    invalid_values = np.full((1, 48, 48, 3), np.inf)
    with pytest.raises(InvalidInputError):
        validate_model_input(invalid_values, (48, 48, 3))

def test_system_compatibility():
    """Testet Systemkompatibilitätsprüfung."""
    # Diese Funktion sollte keine Fehler werfen
    check_system_compatibility()