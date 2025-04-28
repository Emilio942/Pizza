"""
Zentrale Konfigurationsdatei für das Pizza-Erkennungssystem
"""

class BaseConfig:
    # Verzeichnisstruktur
    DATA_DIR = 'data'
    OUTPUT_DIR = 'output'
    MODEL_DIR = 'models'
    TEMP_DIR = 'output/temp'
    LOG_DIR = 'output/logs'
    
    # Trainingsparameter
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.002
    EARLY_STOPPING_PATIENCE = 10
    
    # Augmentierungsparameter
    AUGMENTATION_FACTOR = 10
    USE_ADVANCED_AUGMENTATION = True

class RP2040Config(BaseConfig):
    """Konfiguration für RP2040-basierte Bildklassifikation"""
    # RP2040 Hardware-Spezifikationen
    RP2040_FLASH_SIZE_KB = 2048  # 2MB Flash
    RP2040_RAM_SIZE_KB = 264     # 264KB RAM
    RP2040_CLOCK_SPEED_MHZ = 133 # 133MHz Dual-Core Arm Cortex M0+
    
    # OV2640 Kamera-Spezifikationen
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    CAMERA_FPS = 7
    
    # Modellparameter
    IMG_SIZE = 48
    MAX_MODEL_SIZE_KB = 180
    MAX_RUNTIME_RAM_KB = 100
    QUANTIZATION_BITS = 8
    
    # Batterieparameter (CR123A)
    BATTERY_CAPACITY_MAH = 1500
    ACTIVE_CURRENT_MA = 180
    SLEEP_CURRENT_MA = 0.5

# Standard-Konfiguration
DefaultConfig = RP2040Config