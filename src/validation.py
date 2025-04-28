"""
Validierungsfunktionen für Eingaben, Konfigurationen und Systemstatus.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np

from .types import (
    ModelConfig, CameraConfig, FirmwareInfo,
    ResourceUsage, HardwareStatus
)
from .constants import (
    RP2040_FLASH_SIZE_KB,
    RP2040_RAM_SIZE_KB,
    MAX_MODEL_SIZE_KB,
    MAX_RUNTIME_RAM_KB,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    IMAGE_EXTENSIONS,
    CLASS_NAMES
)
from .exceptions import (
    InvalidInputError,
    ConfigError,
    ResourceError,
    HardwareError
)

logger = logging.getLogger(__name__)

def validate_image(
    image: np.ndarray,
    min_size: int = 32,
    max_size: int = 1024
) -> None:
    """Validiert ein Eingabebild."""
    if not isinstance(image, np.ndarray):
        raise InvalidInputError("Eingabe ist kein numpy array")
    
    if image.ndim not in [2, 3]:
        raise InvalidInputError(
            f"Ungültige Bilddimensionen: {image.ndim}, erwartet 2 oder 3"
        )
    
    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        raise InvalidInputError(
            f"Ungültige Anzahl Farbkanäle: {image.shape[2]}, erwartet 1, 3 oder 4"
        )
    
    height, width = image.shape[:2]
    if (width < min_size or height < min_size or
        width > max_size or height > max_size):
        raise InvalidInputError(
            f"Ungültige Bildgröße: {width}x{height}, "
            f"erwartet zwischen {min_size} und {max_size} Pixel"
        )

def validate_model_config(config: ModelConfig) -> None:
    """Validiert eine Modellkonfiguration."""
    required_keys = {
        'input_size',
        'num_classes',
        'conv_channels',
        'fc_features'
    }
    
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ConfigError(
            f"Fehlende Konfigurationsschlüssel: {missing_keys}"
        )
    
    if config['input_size'] <= 0:
        raise ConfigError(f"Ungültige Eingabegröße: {config['input_size']}")
    
    if config['num_classes'] != len(CLASS_NAMES):
        raise ConfigError(
            f"Ungültige Anzahl Klassen: {config['num_classes']}, "
            f"erwartet {len(CLASS_NAMES)}"
        )

def validate_camera_config(config: CameraConfig) -> None:
    """Validiert eine Kamerakonfiguration."""
    if config.width != CAMERA_WIDTH or config.height != CAMERA_HEIGHT:
        raise ConfigError(
            f"Ungültige Kameraauflösung: {config.width}x{config.height}, "
            f"erwartet {CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )
    
    if config.fps > CAMERA_FPS:
        raise ConfigError(
            f"Ungültige FPS: {config.fps}, Maximum ist {CAMERA_FPS}"
        )
    
    if config.format not in ['RGB565', 'YUV422', 'GRAYSCALE']:
        raise ConfigError(f"Ungültiges Bildformat: {config.format}")

def validate_firmware(firmware: FirmwareInfo) -> None:
    """Validiert Firmware-Informationen."""
    if firmware.size_bytes > RP2040_FLASH_SIZE_KB * 1024:
        raise ResourceError(
            f"Firmware zu groß: {firmware.size_bytes} Bytes, "
            f"Maximum ist {RP2040_FLASH_SIZE_KB * 1024} Bytes"
        )
    
    if firmware.model_size_bytes > MAX_MODEL_SIZE_KB * 1024:
        raise ResourceError(
            f"Modell zu groß: {firmware.model_size_bytes} Bytes, "
            f"Maximum ist {MAX_MODEL_SIZE_KB * 1024} Bytes"
        )

def validate_resource_usage(usage: ResourceUsage) -> None:
    """Validiert Ressourcennutzung."""
    if usage.ram_used_kb > MAX_RUNTIME_RAM_KB:
        raise ResourceError(
            f"RAM-Überlauf: {usage.ram_used_kb}KB verwendet, "
            f"Maximum ist {MAX_RUNTIME_RAM_KB}KB"
        )
    
    if usage.flash_used_kb > RP2040_FLASH_SIZE_KB:
        raise ResourceError(
            f"Flash-Überlauf: {usage.flash_used_kb}KB verwendet, "
            f"Maximum ist {RP2040_FLASH_SIZE_KB}KB"
        )

def validate_hardware_status(status: HardwareStatus) -> None:
    """Validiert Hardware-Status."""
    # Temperaturgrenzwerte für RP2040
    MAX_TEMP_C = 85
    MIN_TEMP_C = -40
    
    if not MIN_TEMP_C <= status.temperature_c <= MAX_TEMP_C:
        raise HardwareError(
            f"Temperatur außerhalb des zulässigen Bereichs: "
            f"{status.temperature_c}°C"
        )
    
    # Spannungsgrenzwerte für 3.3V Betrieb
    MIN_VOLTAGE_MV = 3000  # 3.0V
    MAX_VOLTAGE_MV = 3600  # 3.6V
    
    if not MIN_VOLTAGE_MV <= status.voltage_mv <= MAX_VOLTAGE_MV:
        raise HardwareError(
            f"Spannung außerhalb des zulässigen Bereichs: "
            f"{status.voltage_mv}mV"
        )
    
    if status.error_count > 10:
        raise HardwareError(
            f"Zu viele Hardware-Fehler: {status.error_count}"
        )

def validate_dataset_path(path: Union[str, Path]) -> None:
    """Validiert einen Datensatzpfad."""
    path = Path(path)
    
    if not path.exists():
        raise InvalidInputError(f"Datensatzpfad existiert nicht: {path}")
    
    if not path.is_dir():
        raise InvalidInputError(f"Datensatzpfad ist kein Verzeichnis: {path}")
    
    # Prüfe ob Bilder vorhanden sind
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(path.glob(f"**/*{ext}"))
    
    if not image_files:
        raise InvalidInputError(
            f"Keine Bilder mit Endungen {IMAGE_EXTENSIONS} in {path} gefunden"
        )

def validate_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Validiert eine JSON-Konfigurationsdatei."""
    path = Path(path)
    
    if not path.exists():
        raise ConfigError(f"Konfigurationsdatei existiert nicht: {path}")
    
    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Ungültiges JSON in {path}: {str(e)}")
    
    if not isinstance(config, dict):
        raise ConfigError(
            f"Ungültiges Konfigurationsformat in {path}, "
            "erwartet ein Dictionary"
        )
    
    return config

def validate_output_path(path: Union[str, Path]) -> None:
    """Validiert und erstellt einen Ausgabepfad."""
    path = Path(path)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise InvalidInputError(
            f"Konnte Ausgabeverzeichnis nicht erstellen: {str(e)}"
        )
    
    if path.exists() and path.is_file():
        logger.warning(f"Überschreibe existierende Datei: {path}")

def validate_model_input(
    batch: np.ndarray,
    expected_shape: tuple
) -> None:
    """Validiert Modelleingaben."""
    if not isinstance(batch, np.ndarray):
        raise InvalidInputError("Eingabe ist kein numpy array")
    
    if batch.shape[1:] != expected_shape:
        raise InvalidInputError(
            f"Ungültige Eingabeform: {batch.shape[1:]}, "
            f"erwartet {expected_shape}"
        )
    
    if not np.isfinite(batch).all():
        raise InvalidInputError("Eingabe enthält ungültige Werte (inf/nan)")

def check_system_compatibility() -> None:
    """Prüft Systemkompatibilität."""
    # Prüfe Python-Version
    import sys
    if sys.version_info < (3, 7):
        raise ConfigError("Python 3.7 oder höher erforderlich")
    
    # Prüfe verfügbaren RAM
    try:
        import psutil
        available_ram = psutil.virtual_memory().available / (1024 * 1024)  # MB
        if available_ram < 50:  # Mindestens 50MB erforderlich
            raise ResourceError(
                f"Zu wenig RAM verfügbar: {available_ram:.1f}MB"
            )
    except ImportError:
        logger.warning("psutil nicht verfügbar, überspringe RAM-Prüfung")
    
    # Prüfe GPU-Verfügbarkeit (optional)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU verfügbar: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Keine GPU verfügbar, verwende CPU")
    except ImportError:
        logger.warning("PyTorch nicht verfügbar, überspringe GPU-Prüfung")