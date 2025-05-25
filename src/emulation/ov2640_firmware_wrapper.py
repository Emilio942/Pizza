#!/usr/bin/env python3
"""
OV2640 Firmware Wrapper for Emulation
Bridges the firmware driver logic with the Python emulator
"""

import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from .ov2640_timing_emulator import OV2640TimingEmulator

logger = logging.getLogger(__name__)

class OV2640FirmwareWrapper:
    """
    Wrapper that emulates the firmware driver behavior from ov2640_driver.c
    This bridges the C firmware logic with the Python emulator
    """
    
    def __init__(self, emulator: Optional[OV2640TimingEmulator] = None):
        """Initialize the firmware wrapper"""
        self.emulator = emulator or OV2640TimingEmulator()
        self.initialized = False
        
        # Configuration state (mirrors ov2640_config_t)
        self.current_config = {
            'format': 'RGB565',
            'size': '48x48',
            'quality': 12,
            'brightness': 0,
            'contrast': 0,
            'saturation': 0,
            'flip_horizontal': False,
            'flip_vertical': False
        }
        
        # GPIO state tracking
        self.gpio_states = {
            'pwdn': True,    # Start powered down
            'reset': True,   # Start in reset
            'xclk': False    # Clock disabled
        }
        
        logger.info("OV2640 Firmware Wrapper initialized")
    
    def ov2640_gpio_init(self):
        """Initialize GPIO pins for camera control (mirrors firmware function)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", "ov2640_gpio_init() called")
        
        # Initialize control pins
        self.emulator.emulate_gpio_write(15, True, "PWDN")   # Power down
        self.emulator.emulate_gpio_write(14, False, "RESET") # Reset active
        
        # PWM/Clock initialization would be here in real firmware
        self.emulator.log_timing_event("PWM_INIT", "XCLK PWM configured for 20MHz")
        
        # I2C initialization
        self.emulator.log_timing_event("I2C_INIT", "I2C initialized at 100kHz (SCCB compatible)")
        
        # Parallel data pins initialization
        self.emulator.log_timing_event("PIO_INIT", "Parallel data pins (D0-D7, PCLK, VSYNC, HSYNC) configured")
        
        return True
    
    def ov2640_set_pwdn(self, state: bool):
        """Set power down state (mirrors firmware function)"""
        self.emulator.emulate_gpio_write(15, state, "PWDN")
        self.gpio_states['pwdn'] = state
        
        if not state:
            # Power-up delay as per firmware
            time.sleep(0.001)  # 1ms = 1000µs
    
    def ov2640_set_reset(self, state: bool):
        """Set reset state (mirrors firmware function)"""
        # Reset is active low in firmware
        gpio_state = not state
        self.emulator.emulate_gpio_write(14, gpio_state, "RESET")
        self.gpio_states['reset'] = state
        
        if not state:
            # Reset delay as per firmware
            time.sleep(0.001)  # 1ms = 1000µs
    
    def ov2640_set_xclk(self, enable: bool):
        """Enable/disable external clock (mirrors firmware function)"""
        self.emulator.emulate_gpio_write(13, enable, "XCLK")
        self.gpio_states['xclk'] = enable
        
        if enable:
            # Clock stabilization delay as per firmware
            time.sleep(0.0001)  # 100µs
    
    def ov2640_write_reg(self, reg: int, value: int) -> bool:
        """Write to OV2640 register via I2C/SCCB (mirrors firmware function)"""
        success = self.emulator.emulate_i2c_write(0x30, reg, value)
        
        if success:
            # Register write delay as per firmware
            time.sleep(0.00001)  # 10µs
        
        return success
    
    def ov2640_read_reg(self, reg: int) -> Tuple[bool, int]:
        """Read from OV2640 register via I2C/SCCB (mirrors firmware function)"""
        return self.emulator.emulate_i2c_read(0x30, reg)
    
    def ov2640_select_bank(self, bank: int) -> bool:
        """Select register bank (mirrors firmware function)"""
        return self.ov2640_write_reg(0xFF, bank)
    
    def ov2640_check_id(self) -> bool:
        """Check device ID to verify communication (mirrors firmware function)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", "ov2640_check_id() called")
        
        # Select sensor bank
        if not self.ov2640_select_bank(0x00):
            return False
        
        # Read product and manufacturer IDs
        success1, pidh = self.ov2640_read_reg(0x0A)
        success2, pidl = self.ov2640_read_reg(0x0B) 
        success3, midh = self.ov2640_read_reg(0x1C)
        success4, midl = self.ov2640_read_reg(0x1D)
        
        if not (success1 and success2 and success3 and success4):
            return False
        
        # Verify IDs match expected values (from ov2640_driver.h)
        id_match = (pidh == 0x26 and pidl == 0x42 and midh == 0x7F and midl == 0xA2)
        
        if id_match:
            self.emulator.log_timing_event("ID_VERIFY", "Device ID verification PASSED")
        else:
            self.emulator.log_timing_event("ID_VERIFY", f"Device ID verification FAILED: PID={pidh:02X}{pidl:02X}, MID={midh:02X}{midl:02X}")
        
        return id_match
    
    def ov2640_reset(self) -> bool:
        """Perform software reset (mirrors firmware function)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", "ov2640_reset() called")
        
        # Hardware reset sequence
        self.ov2640_set_reset(True)
        time.sleep(0.01)  # 10ms
        self.ov2640_set_reset(False)
        time.sleep(0.01)  # 10ms
        
        # Software reset via register
        if not self.ov2640_select_bank(0x01):  # DSP bank
            return False
        
        if not self.ov2640_write_reg(0x80, 0x80):
            return False
        
        time.sleep(0.05)  # 50ms wait for reset to complete
        self.emulator.log_timing_event("SW_RESET", "Software reset completed")
        
        return True
    
    def ov2640_set_power_down(self, enable: bool) -> bool:
        """Set power down mode (mirrors firmware function)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", f"ov2640_set_power_down({enable}) called")
        self.ov2640_set_pwdn(enable)
        return True
    
    def load_48x48_configuration(self) -> bool:
        """Load 48x48 pixel configuration (mirrors firmware logic)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", "Loading 48x48 configuration")
        
        # Configuration sequence from ov2640_driver.c
        config_sequence = [
            # Bank selection and reset
            (0xFF, 0x01),  # DSP bank
            (0x2c, 0xff),
            (0x2e, 0xdf),
            
            # Bank 0 (Sensor) registers
            (0xFF, 0x00),  # Sensor bank
            (0x32, 0x36),
            (0x0c, 0x36),
            (0x0d, 0x34),
            (0x0e, 0x05),
            (0x0f, 0xc5),
            (0x10, 0x20),
            (0x11, 0x01),
            (0x12, 0x02),
            (0x13, 0x28),
            (0x14, 0x32),
            (0x15, 0x30),
            
            # Bank 1 (DSP) registers for 48x48 resolution
            (0xFF, 0x01),  # DSP bank
            (0xc0, 0x64),
            (0xc1, 0x4b),
            (0x86, 0x35),
            (0x50, 0x92),
            (0x51, 0x01),
            (0x52, 0x01),
            (0x53, 0x00),
            (0x54, 0x00),
            (0x55, 0x88),
            (0x57, 0x00),
            (0x5a, 0x30),  # Output width 48 pixels
            (0x5b, 0x30),  # Output height 48 pixels
            (0x5c, 0x00),
            (0xd3, 0x04),
            
            # Color matrix and format settings for RGB565
            (0x7f, 0x00),
            (0xe0, 0x00),
            (0xe1, 0x00),
            (0xe5, 0x00),
            (0xd7, 0x00),
            (0xda, 0x09),  # RGB565 format
        ]
        
        for reg, value in config_sequence:
            if not self.ov2640_write_reg(reg, value):
                self.emulator.log_timing_event("CONFIG_ERROR", f"Failed to write reg 0x{reg:02X}")
                return False
        
        self.emulator.log_timing_event("CONFIG_COMPLETE", "48x48 RGB565 configuration loaded successfully")
        return True
    
    def ov2640_init(self) -> bool:
        """Initialize OV2640 camera (mirrors firmware function exactly)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", "ov2640_init() called")
        
        # Initialize GPIO and I2C
        self.ov2640_gpio_init()
        
        # Power up sequence (exact timing from firmware)
        self.ov2640_set_pwdn(True)   # Power down first
        self.ov2640_set_reset(True)  # Reset
        self.ov2640_set_xclk(True)   # Enable clock
        
        time.sleep(0.01)  # 10ms
        
        self.ov2640_set_pwdn(False)  # Release power down
        time.sleep(0.01)  # 10ms
        
        self.ov2640_set_reset(False) # Release reset
        time.sleep(0.05)  # 50ms
        
        # Check if device is responding
        if not self.ov2640_check_id():
            self.emulator.log_timing_event("INIT_ERROR", "Device ID check failed")
            return False
        
        # Perform software reset
        if not self.ov2640_reset():
            self.emulator.log_timing_event("INIT_ERROR", "Software reset failed")
            return False
        
        # Load 48x48 pixel configuration
        if not self.load_48x48_configuration():
            self.emulator.log_timing_event("INIT_ERROR", "Configuration load failed")
            return False
        
        # Set default configuration
        self.current_config = {
            'format': 'RGB565',
            'size': '48x48',
            'quality': 12,
            'brightness': 0,
            'contrast': 0,
            'saturation': 0,
            'flip_horizontal': False,
            'flip_vertical': False
        }
        
        self.initialized = True
        self.emulator.log_timing_event("INIT_SUCCESS", "OV2640 initialization completed successfully")
        
        return True
    
    def ov2640_start_capture(self) -> bool:
        """Start frame capture (mirrors firmware function)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", "ov2640_start_capture() called")
        
        if not self.initialized:
            self.emulator.log_timing_event("CAPTURE_ERROR", "Camera not initialized")
            return False
        
        # Enable capture by clearing any previous capture state
        if not self.ov2640_select_bank(0x01):  # DSP bank
            return False
        
        # Trigger capture start (exact register from firmware)
        success = self.ov2640_write_reg(0x3c, 0x46)
        
        if success:
            self.emulator.log_timing_event("CAPTURE_TRIGGER", "Frame capture triggered via register 0x3c")
        
        return success
    
    def ov2640_capture_to_buffer(self, buffer_size: int = 4608) -> bool:
        """Capture image to buffer (mirrors firmware function)"""
        self.emulator.log_timing_event("FIRMWARE_CALL", f"ov2640_capture_to_buffer(size={buffer_size}) called")
        
        if not self.initialized:
            return False
        
        # Start capture
        if not self.ov2640_start_capture():
            return False
        
        # Simulate the actual capture timing
        # In real firmware this would:
        # 1. Wait for VSYNC
        # 2. Capture pixel data via PIO DMA  
        # 3. Process data according to format
        
        # Use the emulator's frame capture simulation
        width = 48
        height = 48
        format_name = self.current_config['format']
        
        success = self.emulator.emulate_frame_capture(width, height, format_name)
        
        if success:
            self.emulator.log_timing_event("BUFFER_CAPTURE", f"Frame data captured to buffer ({buffer_size} bytes)")
        
        return success
    
    def get_firmware_state(self) -> Dict:
        """Get current firmware state for debugging"""
        return {
            'initialized': self.initialized,
            'config': self.current_config.copy(),
            'gpio_states': self.gpio_states.copy(),
            'emulator_state': self.emulator.get_timing_summary()
        }
