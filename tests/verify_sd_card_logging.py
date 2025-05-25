#!/usr/bin/env python3
"""
Comprehensive test for SD card logging in the RP2040 pizza detection system.
This script simulates a complete pizza detection workflow and verifies that
all relevant performance metrics are logged to the SD card.
"""

import os
import time
import logging
import sys
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to Python path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.emulation.emulator import RP2040Emulator
from src.emulation.sd_card_emulator import SDCardEmulator
from src.emulation.logging_system import LogLevel, LogType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sd_card_verification")

def verify_sd_card_logging():
    """Comprehensive test for SD card logging functionality."""
    logger.info("Starting comprehensive SD card logging verification")
    
    # Create output directory if it doesn't exist
    output_dir = Path("output/sd_card_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure SD card directory exists
    sd_card_dir = Path("output/sd_card")
    sd_card_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the RP2040 emulator
    emulator = RP2040Emulator(battery_capacity_mah=1500.0)
    
    # Load test firmware
    test_firmware = {
        "name": "Pizza Detection Test Firmware",
        "version": "1.0.0",
        "total_size_bytes": 256 * 1024,  # 256KB
        "ram_usage_bytes": 100 * 1024,   # 100KB
        "model_input_size": (96, 96)
    }
    emulator.load_firmware(test_firmware)
    
    logger.info("Emulator initialized and firmware loaded")
    
    # Create a simulated test image
    test_image = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    
    # Perform a series of pizza detection operations
    logger.info("Running simulated pizza detection workflow")
    
    # 1. Perform multiple inference operations with varying conditions
    for i in range(10):
        logger.info(f"Running inference operation {i+1}/10")
        
        # Simulate different lighting conditions by adjusting pixel values
        brightness = 0.7 + (i * 0.03)  # Gradually increase brightness
        adjusted_image = np.clip(test_image * brightness, 0, 255).astype(np.uint8)
        
        # Run inference
        result = emulator.simulate_inference(adjusted_image)
        
        logger.info(f"  Inference time: {result['inference_time']*1000:.2f}ms, " 
                   f"Class: {result['class_id']}, Confidence: {result['confidence']:.4f}")
        
        # Short delay between operations
        time.sleep(0.2)
    
    # 2. Perform temperature logging
    logger.info("Testing temperature logging")
    for i in range(3):
        emulator.log_temperature()
        time.sleep(0.1)
    
    # 3. Inject a temperature spike to test extreme conditions
    logger.info("Injecting temperature spike")
    emulator.inject_temperature_spike(delta=10.0, duration=2.0)
    time.sleep(0.5)
    emulator.log_temperature()
    
    # 4. Test sleep mode (energy efficiency)
    logger.info("Testing sleep mode")
    emulator.enter_sleep_mode()
    time.sleep(1.0)
    emulator.wake_up()
    
    # 5. Log final system stats
    stats = emulator.get_system_stats()
    logger.info(f"Final system statistics:")
    logger.info(f"  RAM usage: {stats['ram_used_kb']:.1f}KB / {stats['ram_free_kb'] + stats['ram_used_kb']:.1f}KB")
    logger.info(f"  Temperature: {stats['current_temperature_c']:.1f}°C")
    logger.info(f"  Battery consumption: {stats['energy_consumed_mah']:.2f}mAh")
    logger.info(f"  Estimated runtime: {stats['estimated_runtime_hours']:.1f} hours")
    
    # Close the emulator and finalize logs
    emulator.close()
    logger.info("Emulator closed")
    
    # Verify that SD card log files exist and contain data
    sd_card_dir = Path("output/sd_card/logs")
    performance_logs = list(sd_card_dir.glob("performance*.csv"))
    temperature_logs = list(sd_card_dir.glob("temperature*.csv"))
    
    if not performance_logs or not temperature_logs:
        logger.error("Expected log files not found on SD card")
        return False
    
    # Check log file contents
    logger.info(f"Found {len(performance_logs)} performance log files:")
    has_performance_data = False
    
    for log_file in performance_logs:
        file_size = log_file.stat().st_size
        logger.info(f"  - {log_file.name} ({file_size} bytes)")
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            logger.info(f"    Contains {len(lines)} lines")
            
            # Ensure we have actual data (not just headers)
            if len(lines) > 1:
                has_performance_data = True
                sample_line = lines[-1]  # Get the last line with data
                logger.info(f"    Sample data: {sample_line}")
    
    logger.info(f"Found {len(temperature_logs)} temperature log files:")
    has_temperature_data = False
    
    for log_file in temperature_logs:
        file_size = log_file.stat().st_size
        logger.info(f"  - {log_file.name} ({file_size} bytes)")
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            logger.info(f"    Contains {len(lines)} lines")
            
            # Ensure we have actual data (not just headers)
            if len(lines) > 1:
                has_temperature_data = True
                sample_line = lines[-1]  # Get the last line with data
                logger.info(f"    Sample data: {sample_line}")
    
    # Check if verification was successful
    if has_performance_data and has_temperature_data:
        logger.info("SD card logging verification completed successfully")
        return True
    else:
        logger.error("SD card logging verification failed: Missing data in log files")
        return False

if __name__ == "__main__":
    success = verify_sd_card_logging()
    sys.exit(0 if success else 1)
