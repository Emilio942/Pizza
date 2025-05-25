#!/usr/bin/env python3
"""
Test script for verifying SD card logging functionality in the RP2040 emulator.
This script simulates a sequence of operations that generate performance metrics
and verifies that they are correctly logged to the SD card.
"""

import os
import time
import logging
import sys
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
logger = logging.getLogger("sd_card_test")

def test_sd_card_logging():
    """Test SD card logging functionality in the RP2040 emulator."""
    logger.info("Starting SD card logging test")
    
    # Create output directory if it doesn't exist
    output_dir = Path("output/test_sd_logging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure SD card directory exists
    sd_card_dir = Path("output/sd_card")
    sd_card_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the RP2040 emulator with SD card logging enabled
    emulator = RP2040Emulator(battery_capacity_mah=1500.0)
    
    # Load test firmware
    test_firmware = {
        "name": "Pizza Detection Firmware",
        "version": "1.0.0",
        "total_size_bytes": 256 * 1024,  # 256KB
        "ram_usage_bytes": 100 * 1024,   # 100KB
        "model_input_size": (96, 96)
    }
    emulator.load_firmware(test_firmware)
    
    logger.info("Emulator initialized and firmware loaded")
    
    # Perform a series of operations that generate performance metrics
    logger.info("Performing operations to generate performance metrics")
    for i in range(5):
        # Execute operation with varying memory usage and execution time
        memory_usage = (5 + i * 2) * 1024  # 5KB to 13KB
        operation_time = (50 + i * 10)      # 50ms to 90ms
        
        logger.info(f"Operation {i+1}: Memory usage: {memory_usage/1024:.1f}KB, Time: {operation_time}ms")
        emulator.execute_operation(memory_usage, operation_time)
        
        # Log custom performance metrics
        prediction = 1 if i % 2 == 0 else 0  # Alternate between pizza/not pizza
        confidence = 0.7 + (i * 0.05)  # Increasing confidence
        
        # Log performance metrics
        emulator.log_performance_metrics(
            inference_time_ms=operation_time,
            peak_ram_kb=memory_usage / 1024,
            cpu_load=70 + (i * 5),  # Increasing CPU load
            prediction=prediction,
            confidence=confidence
        )
        
        # Take temperature reading and log it
        emulator.log_temperature()
        
        # Log overall system stats
        stats = emulator.get_system_stats()
        logger.info(f"System stats: CPU: {stats.get('cpu_usage_percent', 0):.1f}%, "
                   f"RAM: {stats.get('ram_used_kb', 0):.1f}KB, "
                   f"Temperature: {stats.get('current_temperature_c', 0):.1f}°C")
        
        # Short delay to simulate real-world operation
        time.sleep(0.5)
    
    # Close the emulator to ensure all log files are properly closed
    emulator.close()
    logger.info("Emulator closed")
    
    # Verify that SD card log files were created
    sd_card_dir = Path("output/sd_card/logs")
    log_files = list(sd_card_dir.glob("*.csv"))
    
    if not log_files:
        logger.error("No log files found on SD card")
        return False
    
    logger.info(f"Found {len(log_files)} log files on SD card:")
    for log_file in log_files:
        file_size = log_file.stat().st_size
        logger.info(f"  - {log_file.name} ({file_size} bytes)")
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            line_count = len(content.strip().split('\n'))
            logger.info(f"    Contains {line_count} lines")
    
    logger.info("SD card logging test completed successfully")
    return True

if __name__ == "__main__":
    success = test_sd_card_logging()
    sys.exit(0 if success else 1)
