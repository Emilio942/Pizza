#!/usr/bin/env python3
"""Constants for the Pizza Detection System.

Centralizes configuration values and derives class metadata from the
authoritative JSON definition located at ``data/class_definitions.json``.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Image processing constants
INPUT_SIZE = 48  # Size of input images (48x48 pixels)
IMAGE_MEAN = [0.47935871, 0.39572979, 0.32422196]  # Mean for normalization
IMAGE_STD = [0.23475593, 0.25177728, 0.26392367]  # Std for normalization
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}  # Supported image formats

# Paths
PROJECT_ROOT = Path(__file__).parents[1]
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "augmented")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CLASSIFIED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "classified")
SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Hardware constants
RP2040_CLOCK_SPEED = 133  # MHz
# Alias for modules expecting explicit unit suffix
RP2040_CLOCK_SPEED_MHZ = RP2040_CLOCK_SPEED
RP2040_RAM = 264  # KB
RP2040_FLASH = 2048  # KB
RP2040_FLASH_SIZE_KB = 2048  # Duplicate for compatibility
RP2040_RAM_SIZE_KB = 264  # Duplicate for compatibility

# Camera parameters
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 7

# Power management
BATTERY_CAPACITY = 1500  # mAh
ACTIVE_POWER_CONSUMPTION = 180  # mA
STANDBY_POWER_CONSUMPTION = 0.5  # mA
MAX_POWER_CONSUMPTION_MW = 450.0  # Upper bound for acceptable power draw
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence before triggering warnings
MAX_INFERENCE_TIME_MS = 150.0  # Target maximum inference latency

# Model constraints
MAX_MODEL_SIZE_KB = 180
MAX_RUNTIME_RAM_KB = 100


def _hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
    """Convert a hex color string (e.g. "#FFAACC") to an RGB tuple."""

    color_hex = color_hex.strip().lstrip("#")
    if len(color_hex) != 6:
        raise ValueError(f"Invalid color hex string: {color_hex}")
    return tuple(int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))


def _load_class_definitions() -> Tuple[List[str], Dict[str, Tuple[int, int, int]]]:
    """Load class names and colors from ``data/class_definitions.json``.

    Falls back to the legacy hard-coded definitions if the file is missing or
    malformed so that existing code paths continue to work.
    """

    fallback_colors: Dict[str, Tuple[int, int, int]] = {
        "basic": (0, 255, 0),
        "burnt": (255, 0, 0),
        "combined": (0, 0, 255),
        "mixed": (255, 255, 0),
        "progression": (255, 0, 255),
        "segment": (0, 255, 255),
    }
    fallback_names: List[str] = list(fallback_colors.keys())

    class_def_path = PROJECT_ROOT / "data" / "class_definitions.json"

    try:
        class_data = json.loads(class_def_path.read_text(encoding="utf-8"))
        if not isinstance(class_data, dict) or not class_data:
            raise ValueError("class_definitions.json must contain a non-empty object")

        class_names: List[str] = list(class_data.keys())
        class_colors: Dict[str, Tuple[int, int, int]] = {}

        for name in class_names:
            entry = class_data.get(name, {})
            color_hex = entry.get("color")
            if isinstance(color_hex, str):
                try:
                    class_colors[name] = _hex_to_rgb(color_hex)
                except ValueError:
                    class_colors[name] = fallback_colors.get(name, (255, 255, 255))
            else:
                class_colors[name] = fallback_colors.get(name, (255, 255, 255))

        return class_names, class_colors

    except FileNotFoundError:
        # File missing: use legacy fallbacks
        return fallback_names, fallback_colors
    except (json.JSONDecodeError, ValueError):
        # Malformed JSON or invalid structure: fall back as well
        return fallback_names, fallback_colors


CLASS_NAMES, CLASS_COLORS = _load_class_definitions()
DEFAULT_CLASSES = list(CLASS_NAMES)
NUM_CLASSES = len(CLASS_NAMES)
