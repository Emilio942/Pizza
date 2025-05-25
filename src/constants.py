#!/usr/bin/env python3
"""
Constants for the Pizza Detection System

This module defines constants used throughout the Pizza Detection System.
These include image dimensions, paths, and configuration parameters.
"""

import os
from pathlib import Path

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

# Model constraints
MAX_MODEL_SIZE_KB = 180
MAX_RUNTIME_RAM_KB = 100

# Classification classes
CLASS_NAMES = ["basic", "burnt", "combined", "mixed", "progression", "segment"]
DEFAULT_CLASSES = ["basic", "burnt", "combined", "mixed", "progression", "segment"]

# Class colors for visualization
CLASS_COLORS = {
    "basic": (0, 255, 0),       # Green
    "burnt": (255, 0, 0),       # Red
    "combined": (0, 0, 255),    # Blue
    "mixed": (255, 255, 0),     # Yellow
    "progression": (255, 0, 255), # Magenta
    "segment": (0, 255, 255)    # Cyan
}
