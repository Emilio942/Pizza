"""
Konstanten für das Pizzaerkennungssystem.
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple

# Hardware-Spezifikationen für RP2040
RP2040_FLASH_SIZE_KB = 2048      # 2MB Flash
RP2040_RAM_SIZE_KB = 264         # 264KB SRAM
RP2040_CLOCK_SPEED_MHZ = 133     # 133MHz
MAX_MODEL_SIZE_KB = 180          # Maximale Modellgröße
MAX_RUNTIME_RAM_KB = 100         # Maximaler RAM-Bedarf während Inferenz

# Kamerakonfiguration OV2640
CAMERA_WIDTH = 160               # VGA-Auflösung
CAMERA_HEIGHT = 120
CAMERA_FPS = 7                   # Typische FPS für OV2640
CAMERA_FORMAT = 'RGB565'         # Standardformat
MIN_CAPTURE_INTERVAL_MS = 100    # Minimales Intervall zwischen Aufnahmen

# Bildverarbeitung
INPUT_SIZE = (64, 64)            # Bildgröße für Modelleingang
CHANNELS = 3                     # RGB-Bilder
IMAGE_MEAN = (0.485, 0.456, 0.406)  # ImageNet-Normalisierung
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp'}

# Modellparameter
NUM_CLASSES = 6                  # Anzahl Klassen
CLASS_NAMES: List[str] = [
    "basic",
    "burnt",
    "combined",
    "mixed",
    "progression",
    "segment"
]
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "basic": (0, 255, 0),       # Green
    "burnt": (255, 0, 0),       # Red
    "combined": (0, 0, 255),    # Blue
    "mixed": (255, 255, 0),     # Yellow
    "progression": (255, 0, 255), # Magenta
    "segment": (0, 255, 255)    # Cyan
}

# Modellarchitektur
CONV_CHANNELS = [16, 32, 64, 128]  # Kanäle der Conv-Layer
FC_FEATURES = [256, 128]           # Neuronen der FC-Layer
DROPOUT_RATE = 0.5                 # Dropout-Wahrscheinlichkeit
BATCH_NORM = True                  # Batch-Normalisierung verwenden

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8                  # 80% Training, 20% Validierung
RANDOM_SEED = 42
AUGMENTATION_FACTOR = 10           # Datenerweiterungsfaktor

# Quantisierung
QUANTIZE_DTYPE = 'int8'            # Quantisierungstyp
QUANTIZE_MODE = 'per_tensor'       # Quantisierungsmodus
CALIBRATION_STEPS = 100            # Schritte für Quantisierungskalibrierung

# Hardware-Limits
MAX_INFERENCE_TIME_MS = 100        # Maximale Inferenzzeit
MIN_CONFIDENCE_THRESHOLD = 0.7     # Minimaler Konfidenzschwellwert
MAX_POWER_CONSUMPTION_MW = 100     # Maximaler Stromverbrauch

# I/O und Dateisystem
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'
LOGS_DIR = PROJECT_ROOT / 'logs'

RAW_DATA_DIR = DATA_DIR / 'raw'
AUGMENTED_DATA_DIR = DATA_DIR / 'augmented'
MODEL_CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
MODEL_EXPORTS_DIR = MODELS_DIR / 'exports'
EVALUATION_DIR = OUTPUT_DIR / 'evaluation'

# Logging
LOG_FILE = LOGS_DIR / 'pizza_detector.log'
ERROR_LOG_FILE = LOGS_DIR / 'error.log'
LOG_ROTATION_SIZE = 5 * 1024 * 1024  # 5MB
LOG_BACKUP_COUNT = 5

# Systemkonstanten
VERSION = '1.0.0'
DEBUG = False                       # Debug-Modus aktivieren
SYSTEM_NAME = 'PizzaDetector'
BUILD_DATE = '2025-04-10'          # Aktuelles Build-Datum

# Visualisierung
PLOT_DPI = 100                      # DPI für Plot-Exports
FIGURE_SIZE = (10, 6)               # Standard-Plotgröße
COLOR_PALETTE = {                   # Farben für verschiedene Visualisierungen
    'basic': '#1f77b4',
    'burnt': '#d62728',
    'undercooked': '#2ca02c',
    'perfect': '#ff7f0e',
    'background': '#333333',
    'grid': '#cccccc',
    'text': '#000000'
}