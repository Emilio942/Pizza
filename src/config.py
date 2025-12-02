import os
import time
import json
import torch
import logging

logger = logging.getLogger(__name__)

class RP2040Config:
    """Ausführliche Konfiguration für RP2040-basierte Bildklassifikation mit Speicher- und Leistungsanalyse"""
    # RP2040 Hardware-Spezifikationen (Defaults)
    RP2040_FLASH_SIZE_KB = 2048  # 2MB Flash
    RP2040_RAM_SIZE_KB = 264     # 264KB RAM
    RP2040_CLOCK_SPEED_MHZ = 133 # 133MHz Dual-Core Arm Cortex M0+
    
    # OV2640 Kamera-Spezifikationen (Defaults)
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    CAMERA_FPS = 7  # Durchschnittliche FPS für Batteriebetrieb
    
    # Batterieparameter (CR123A) (Defaults)
    BATTERY_CAPACITY_MAH = 1500   # Typische CR123A Kapazität
    ACTIVE_CURRENT_MA = 180       # Durchschnittlicher Stromverbrauch im aktiven Zustand
    SLEEP_CURRENT_MA = 0.5        # Stromverbrauch im Schlafmodus
    
    # Datensatz-Konfiguration (Defaults)
    DATA_DIR = 'augmented_pizza'
    MODEL_DIR = 'models_optimized'
    TEMP_DIR = 'temp_preprocessing'
    
    # Modellparameter (Defaults)
    IMG_SIZE = 48       # Kleine Bildgröße für Mikrocontroller
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.002
    EARLY_STOPPING_PATIENCE = 10
    
    # Speicheroptimierungen (Defaults)
    MAX_MODEL_SIZE_KB = 180       # Maximale Modellgröße (Flash)
    MAX_RUNTIME_RAM_KB = 100      # Maximaler RAM-Verbrauch während Inferenz
    QUANTIZATION_BITS = 8         # Int8-Quantisierung
    
    # Trainingsgerät
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, config_path=None, data_dir=None):
        self.start_time = time.time()
        
        # Determine default config path relative to this file
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, 'config', 'hardware.json')
        
        # Load from JSON if exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    self._apply_config(config_data)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Override with environment variables
        self._apply_env_vars()
        
        # Manual overrides
        if data_dir:
            self.DATA_DIR = data_dir

    def _apply_config(self, data):
        """Apply configuration from dictionary"""
        if 'hardware' in data:
            self.RP2040_FLASH_SIZE_KB = data['hardware'].get('flash_size_kb', self.RP2040_FLASH_SIZE_KB)
            self.RP2040_RAM_SIZE_KB = data['hardware'].get('ram_size_kb', self.RP2040_RAM_SIZE_KB)
            self.RP2040_CLOCK_SPEED_MHZ = data['hardware'].get('clock_speed_mhz', self.RP2040_CLOCK_SPEED_MHZ)
            
        if 'camera' in data:
            self.CAMERA_WIDTH = data['camera'].get('width', self.CAMERA_WIDTH)
            self.CAMERA_HEIGHT = data['camera'].get('height', self.CAMERA_HEIGHT)
            self.CAMERA_FPS = data['camera'].get('fps', self.CAMERA_FPS)
            
        if 'battery' in data:
            self.BATTERY_CAPACITY_MAH = data['battery'].get('capacity_mah', self.BATTERY_CAPACITY_MAH)
            self.ACTIVE_CURRENT_MA = data['battery'].get('active_current_ma', self.ACTIVE_CURRENT_MA)
            self.SLEEP_CURRENT_MA = data['battery'].get('sleep_current_ma', self.SLEEP_CURRENT_MA)
            
        if 'paths' in data:
            self.DATA_DIR = data['paths'].get('data_dir', self.DATA_DIR)
            self.MODEL_DIR = data['paths'].get('model_dir', self.MODEL_DIR)
            self.TEMP_DIR = data['paths'].get('temp_dir', self.TEMP_DIR)
            
        if 'model' in data:
            self.IMG_SIZE = data['model'].get('img_size', self.IMG_SIZE)
            self.BATCH_SIZE = data['model'].get('batch_size', self.BATCH_SIZE)
            self.EPOCHS = data['model'].get('epochs', self.EPOCHS)
            self.LEARNING_RATE = data['model'].get('learning_rate', self.LEARNING_RATE)
            self.EARLY_STOPPING_PATIENCE = data['model'].get('early_stopping_patience', self.EARLY_STOPPING_PATIENCE)
            
        if 'constraints' in data:
            self.MAX_MODEL_SIZE_KB = data['constraints'].get('max_model_size_kb', self.MAX_MODEL_SIZE_KB)
            self.MAX_RUNTIME_RAM_KB = data['constraints'].get('max_runtime_ram_kb', self.MAX_RUNTIME_RAM_KB)
            self.QUANTIZATION_BITS = data['constraints'].get('quantization_bits', self.QUANTIZATION_BITS)

    def _apply_env_vars(self):
        """Override configuration with environment variables"""
        # Hardware
        if os.getenv('RP2040_FLASH_SIZE_KB'): self.RP2040_FLASH_SIZE_KB = int(os.getenv('RP2040_FLASH_SIZE_KB'))
        if os.getenv('RP2040_RAM_SIZE_KB'): self.RP2040_RAM_SIZE_KB = int(os.getenv('RP2040_RAM_SIZE_KB'))
        
        # Battery
        if os.getenv('BATTERY_CAPACITY_MAH'): self.BATTERY_CAPACITY_MAH = int(os.getenv('BATTERY_CAPACITY_MAH'))
        
        # Paths
        if os.getenv('DATA_DIR'): self.DATA_DIR = os.getenv('DATA_DIR')
        if os.getenv('MODEL_DIR'): self.MODEL_DIR = os.getenv('MODEL_DIR')
        
        # Model
        if os.getenv('IMG_SIZE'): self.IMG_SIZE = int(os.getenv('IMG_SIZE'))
        if os.getenv('BATCH_SIZE'): self.BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
        if os.getenv('EPOCHS'): self.EPOCHS = int(os.getenv('EPOCHS'))
        if os.getenv('LEARNING_RATE'): self.LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
            
        # Modellverzeichnis erstellen
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        
        # Batterielebensdauer-Berechnungen
        active_time_hours = self.BATTERY_CAPACITY_MAH / self.ACTIVE_CURRENT_MA
        standby_time_hours = self.BATTERY_CAPACITY_MAH / self.SLEEP_CURRENT_MA
        
        logger.info("=" * 80)
        logger.info("RP2040 PIZZA-ERKENNUNGSSYSTEM - DETAILLIERTE KONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Hardware: RP2040 - {self.RP2040_CLOCK_SPEED_MHZ}MHz, {self.RP2040_RAM_SIZE_KB}KB RAM, {self.RP2040_FLASH_SIZE_KB}KB Flash")
        logger.info(f"Kamera: OV2640 - {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT}, {self.CAMERA_FPS} FPS")
        logger.info(f"Stromversorgung: CR123A - {self.BATTERY_CAPACITY_MAH}mAh, {active_time_hours:.2f}h aktiv, {standby_time_hours:.2f}h standby")
        logger.info(f"Modellparameter: {self.IMG_SIZE}x{self.IMG_SIZE} Eingabegröße, {self.QUANTIZATION_BITS}-Bit Quantisierung")
        logger.info(f"Speicherbeschränkungen: Max. {self.MAX_MODEL_SIZE_KB}KB Modellgröße, {self.MAX_RUNTIME_RAM_KB}KB Laufzeit-RAM")
        logger.info("=" * 80)
        
    def get_runtime_stats(self):
        """Gibt Laufzeitstatistiken zurück"""
        elapsed_time = time.time() - self.start_time
        return {
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_formatted': f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
        }
