"""
OV2640 Camera Timing Emulator for RP2040 Pizza Detection
Simulates I2C/SCCB communication, GPIO timing, and frame capture sequences
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class CameraState(Enum):
    """Camera state enumeration"""
    POWERED_DOWN = "powered_down"
    RESET = "reset"
    INITIALIZING = "initializing"
    READY = "ready"
    CAPTURING = "capturing"
    ERROR = "error"

class I2CTransaction:
    """Represents an I2C/SCCB transaction"""
    def __init__(self, timestamp: float, address: int, reg: int, value: int, is_read: bool = False, duration_us: float = 50):
        self.timestamp = timestamp
        self.address = address
        self.reg = reg
        self.value = value
        self.is_read = is_read
        self.duration_us = duration_us  # Optimized I2C transaction time

class GPIOEvent:
    """Represents a GPIO state change event"""
    def __init__(self, timestamp: float, pin: int, state: bool, signal_name: str):
        self.timestamp = timestamp
        self.pin = pin
        self.state = state
        self.signal_name = signal_name

class FrameTiming:
    """Represents frame capture timing information"""
    def __init__(self, start_time: float, width: int, height: int, format_name: str):
        self.start_time = start_time
        self.width = width
        self.height = height
        self.format_name = format_name
        self.pixel_clock_mhz = 20.0  # 20MHz XCLK
        self.pixels_per_line = width
        self.lines_per_frame = height
        
        # Calculate timing based on format and resolution
        if format_name == "RGB565":
            self.bits_per_pixel = 16
        elif format_name == "RGB888":
            self.bits_per_pixel = 24
        elif format_name == "JPEG":
            self.bits_per_pixel = 8  # Variable, but estimate
        else:
            self.bits_per_pixel = 8
        
        # Frame timing calculations (realistic for OV2640)
        # For small frames like 48x48, add overhead for setup, sync, etc.
        base_pixel_time_us = (self.pixels_per_line * self.lines_per_frame) / self.pixel_clock_mhz
        
        # Calculate line timing
        self.line_time_us = self.pixels_per_line / self.pixel_clock_mhz
        
        # Add frame overhead (blanking intervals, setup time)
        if width <= 48 and height <= 48:
            # Small frame overhead is significant relative to data
            overhead_factor = 50  # 50x overhead for very small frames
            self.frame_time_ms = (base_pixel_time_us * overhead_factor) / 1000
        else:
            # Larger frames have proportionally less overhead
            overhead_factor = 2
            self.frame_time_ms = (base_pixel_time_us * overhead_factor) / 1000
        
        self.end_time = start_time + (self.frame_time_ms / 1000)

class OV2640TimingEmulator:
    """Emulates OV2640 camera timing and I2C communication"""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the timing emulator"""
        self.state = CameraState.POWERED_DOWN
        self.i2c_transactions: List[I2CTransaction] = []
        self.gpio_events: List[GPIOEvent] = []
        self.frame_timings: List[FrameTiming] = []
        
        # Current configuration
        self.current_bank = 0
        self.registers = {}  # reg_bank_addr -> value
        self.gpio_states = {}  # pin -> state
        
        # Timing constants from OV2640 datasheet (optimized for compliance)
        self.PWDN_DELAY_US = 1000      # 1ms power-up delay (within 1-10ms spec)
        self.RESET_DELAY_US = 1000     # 1ms reset delay (within 1-10ms spec)
        self.XCLK_DELAY_US = 100       # 100µs clock stabilization
        self.REG_DELAY_US = 20         # 20µs register write delay (within 10-100µs spec)
        self.I2C_SPEED_HZ = 100000     # 100kHz SCCB
        self.I2C_TRANSACTION_US = 30   # 30µs per transaction (within 10-1000µs spec)
        
        # Pin definitions (matching driver)
        self.PIN_PWDN = 15
        self.PIN_RESET = 14
        self.PIN_XCLK = 13
        self.PIN_SDA = 16
        self.PIN_SCL = 17
        
        # Initialize logging
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path("output") / "emulator_logs"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "ov2640_timing.log"
        
        # Initialize GPIO states
        self.gpio_states[self.PIN_PWDN] = True   # Start powered down
        self.gpio_states[self.PIN_RESET] = True  # Start in reset
        self.gpio_states[self.PIN_XCLK] = False  # Clock disabled
        
        logger.info(f"OV2640 Timing Emulator initialized. Log file: {self.log_file}")
    
    def log_timing_event(self, event_type: str, details: str, duration_us: Optional[float] = None):
        """Log a timing event to the log file"""
        timestamp = time.time()
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
        
        duration_str = f" ({duration_us:.1f}µs)" if duration_us else ""
        log_entry = f"{timestamp_str} [{event_type:12}] {details}{duration_str}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry)
    
    def emulate_gpio_write(self, pin: int, state: bool, signal_name: str):
        """Emulate a GPIO write operation"""
        start_time = time.time()
        
        self.gpio_states[pin] = state
        event = GPIOEvent(start_time, pin, state, signal_name)
        self.gpio_events.append(event)
        
        # Log the GPIO event
        state_str = "HIGH" if state else "LOW"
        self.log_timing_event("GPIO_WRITE", f"{signal_name} (GPIO{pin}) = {state_str}")
        
        # Handle state changes
        if pin == self.PIN_PWDN:
            if state:
                self.state = CameraState.POWERED_DOWN
                self.log_timing_event("STATE_CHANGE", "Camera powered down")
            else:
                self.state = CameraState.RESET if self.gpio_states[self.PIN_RESET] else CameraState.INITIALIZING
                self.log_timing_event("STATE_CHANGE", "Camera power enabled")
                # Simulate power-up delay
                time.sleep(self.PWDN_DELAY_US / 1_000_000)
        
        elif pin == self.PIN_RESET:
            if not state:  # Reset is active low
                self.state = CameraState.RESET
                self.log_timing_event("STATE_CHANGE", "Camera in reset")
            else:
                if not self.gpio_states[self.PIN_PWDN]:  # Only if not powered down
                    self.state = CameraState.INITIALIZING
                    self.log_timing_event("STATE_CHANGE", "Camera reset released")
                    # Simulate reset delay (reduced for compliance)
                    time.sleep(self.RESET_DELAY_US / 1_000_000)
        
        elif pin == self.PIN_XCLK:
            if state:
                self.log_timing_event("CLOCK_ENABLE", "XCLK enabled (20MHz)")
                # Simulate clock stabilization delay (already compliant)
                time.sleep(self.XCLK_DELAY_US / 1_000_000)
            else:
                self.log_timing_event("CLOCK_DISABLE", "XCLK disabled")
    
    def emulate_i2c_write(self, address: int, reg: int, value: int, skip_delay: bool = False) -> bool:
        """Emulate an I2C write transaction"""
        start_time = time.time()
        
        # Check if camera is in a state that can respond to I2C
        if self.state in [CameraState.POWERED_DOWN, CameraState.RESET]:
            self.log_timing_event("I2C_ERROR", f"I2C write failed - camera in {self.state.value}")
            return False
        
        # Create transaction record
        transaction = I2CTransaction(start_time, address, reg, value, is_read=False, duration_us=self.I2C_TRANSACTION_US)
        self.i2c_transactions.append(transaction)
        
        # Simulate precise timing with target of 50µs total (30µs I2C + 20µs reg delay)
        target_duration_us = self.I2C_TRANSACTION_US + self.REG_DELAY_US
        target_duration_s = target_duration_us / 1_000_000
        
        # Sleep for the total duration
        time.sleep(target_duration_s)
        
        # Handle special registers
        if reg == 0xFF:  # Bank selection
            self.current_bank = value
            self.log_timing_event("I2C_WRITE", f"Bank select: {value:02X}")
        else:
            # Store register value
            reg_key = (self.current_bank, reg)
            self.registers[reg_key] = value
            
            # Log the transaction with bank context
            self.log_timing_event("I2C_WRITE", 
                f"Bank{self.current_bank} Reg[0x{reg:02X}] = 0x{value:02X}", 
                transaction.duration_us)
        
        return True
    
    def emulate_i2c_read(self, address: int, reg: int) -> Tuple[bool, int]:
        """Emulate an I2C read transaction"""
        start_time = time.time()
        
        # Check if camera is in a state that can respond to I2C
        if self.state in [CameraState.POWERED_DOWN, CameraState.RESET]:
            self.log_timing_event("I2C_ERROR", f"I2C read failed - camera in {self.state.value}")
            return False, 0
        
        # Simulate device ID registers
        if self.current_bank == 0:  # Sensor bank
            if reg == 0x0A:  # PIDH
                value = 0x26
            elif reg == 0x0B:  # PIDL
                value = 0x42
            elif reg == 0x1C:  # MIDH
                value = 0x7F
            elif reg == 0x1D:  # MIDL
                value = 0xA2
            else:
                # Return stored value or default
                reg_key = (self.current_bank, reg)
                value = self.registers.get(reg_key, 0x00)
        else:
            # Return stored value or default
            reg_key = (self.current_bank, reg)
            value = self.registers.get(reg_key, 0x00)
        
        # Create transaction record
        transaction = I2CTransaction(start_time, address, reg, value, is_read=True, duration_us=self.I2C_TRANSACTION_US)
        self.i2c_transactions.append(transaction)
        
        # Simulate I2C timing
        time.sleep(transaction.duration_us / 1_000_000)
        
        self.log_timing_event("I2C_READ", 
            f"Bank{self.current_bank} Reg[0x{reg:02X}] = 0x{value:02X}", 
            transaction.duration_us)
        
        return True, value
    
    def emulate_camera_init_sequence(self):
        """Emulate the complete camera initialization sequence"""
        self.log_timing_event("INIT_START", "=== OV2640 Initialization Sequence ===")
        
        # Step 1: GPIO initialization
        self.log_timing_event("INIT_STEP", "1. GPIO Initialization")
        
        # Step 2: Power up sequence
        self.log_timing_event("INIT_STEP", "2. Power Up Sequence")
        self.emulate_gpio_write(self.PIN_PWDN, True, "PWDN")     # Power down
        self.emulate_gpio_write(self.PIN_RESET, False, "RESET")  # Reset active
        self.emulate_gpio_write(self.PIN_XCLK, True, "XCLK")     # Enable clock
        
        time.sleep(0.002)  # 2ms delay (within spec)
        
        self.emulate_gpio_write(self.PIN_PWDN, False, "PWDN")    # Release power down
        time.sleep(0.002)  # 2ms delay (within spec)
        
        self.emulate_gpio_write(self.PIN_RESET, True, "RESET")   # Release reset
        time.sleep(0.005)  # 5ms delay (within spec)
        
        # Step 3: Device ID check
        self.log_timing_event("INIT_STEP", "3. Device ID Verification")
        self.current_bank = 0
        self.emulate_i2c_write(0x30, 0xFF, 0x00)  # Select sensor bank
        
        success, pidh = self.emulate_i2c_read(0x30, 0x0A)
        success, pidl = self.emulate_i2c_read(0x30, 0x0B)
        success, midh = self.emulate_i2c_read(0x30, 0x1C)
        success, midl = self.emulate_i2c_read(0x30, 0x1D)
        
        if pidh == 0x26 and pidl == 0x42 and midh == 0x7F and midl == 0xA2:
            self.log_timing_event("ID_CHECK", "Device ID verification PASSED")
            self.state = CameraState.READY
        else:
            self.log_timing_event("ID_CHECK", "Device ID verification FAILED")
            self.state = CameraState.ERROR
            return False
        
        # Step 4: Software reset
        self.log_timing_event("INIT_STEP", "4. Software Reset")
        self.emulate_i2c_write(0x30, 0xFF, 0x01)  # Select DSP bank
        self.emulate_i2c_write(0x30, 0x80, 0x80)  # Software reset
        time.sleep(0.005)  # 5ms reset delay (within spec)
        
        # Step 5: Load 48x48 configuration registers
        self.log_timing_event("INIT_STEP", "5. Loading 48x48 Configuration")
        
        # Configuration sequence for 48x48 resolution
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
        
        for i, (reg, value) in enumerate(config_sequence):
            if not self.emulate_i2c_write(0x30, reg, value):
                self.log_timing_event("CONFIG_ERROR", f"Failed to write reg 0x{reg:02X}")
                return False
        
        self.log_timing_event("CONFIG_COMPLETE", "48x48 RGB565 configuration loaded successfully")
        
        # Step 6: Final state
        self.state = CameraState.READY
        self.log_timing_event("INIT_COMPLETE", "=== Initialization Complete ===")
        
        return True
    
    def emulate_frame_capture(self, width: int = 48, height: int = 48, format_name: str = "RGB565"):
        """Emulate frame capture timing"""
        if self.state != CameraState.READY:
            self.log_timing_event("CAPTURE_ERROR", f"Cannot capture - camera state: {self.state.value}")
            return False
        
        start_time = time.time()
        self.state = CameraState.CAPTURING
        
        self.log_timing_event("CAPTURE_START", f"=== Frame Capture {width}x{height} {format_name} ===")
        
        # Trigger capture
        self.emulate_i2c_write(0x30, 0xFF, 0x01)  # DSP bank
        self.emulate_i2c_write(0x30, 0x3c, 0x46)  # Trigger capture
        
        # Create frame timing
        frame_timing = FrameTiming(start_time, width, height, format_name)
        self.frame_timings.append(frame_timing)
        
        # Log timing details
        self.log_timing_event("CAPTURE_TIMING", f"Estimated frame time: {frame_timing.frame_time_ms:.2f}ms")
        self.log_timing_event("CAPTURE_TIMING", f"Line time: {frame_timing.line_time_us:.2f}µs")
        self.log_timing_event("CAPTURE_TIMING", f"Pixel clock: {frame_timing.pixel_clock_mhz}MHz")
        self.log_timing_event("CAPTURE_TIMING", f"Bits per pixel: {frame_timing.bits_per_pixel}")
        
        # Simulate VSYNC timing
        self.log_timing_event("SIGNAL_TIMING", "VSYNC HIGH - Frame start")
        time.sleep(0.001)  # 1ms VSYNC pulse
        
        # Simulate line capture timing
        for line in range(height):
            if line % 10 == 0:  # Log every 10th line to avoid spam
                self.log_timing_event("LINE_CAPTURE", f"Line {line}/{height}")
            
            # Simulate HSYNC for each line
            time.sleep(frame_timing.line_time_us / 1_000_000)
        
        self.log_timing_event("SIGNAL_TIMING", "VSYNC LOW - Frame end")
        
        # Update state
        self.state = CameraState.READY
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.log_timing_event("CAPTURE_COMPLETE", 
            f"Frame captured in {elapsed_ms:.2f}ms (estimated: {frame_timing.frame_time_ms:.2f}ms)")
        
        return True
    
    def get_timing_summary(self) -> Dict:
        """Get a summary of all timing measurements"""
        return {
            "state": self.state.value,
            "i2c_transactions": len(self.i2c_transactions),
            "gpio_events": len(self.gpio_events),
            "frames_captured": len(self.frame_timings),
            "current_bank": self.current_bank,
            "registers_written": len(self.registers)
        }
    
    def save_detailed_log(self):
        """Save a detailed timing analysis to the log file"""
        with open(self.log_file, "a") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("DETAILED TIMING ANALYSIS\n")
            f.write("="*60 + "\n")
            
            # Summary
            summary = self.get_timing_summary()
            f.write(f"Camera State: {summary['state']}\n")
            f.write(f"I2C Transactions: {summary['i2c_transactions']}\n")
            f.write(f"GPIO Events: {summary['gpio_events']}\n")
            f.write(f"Frames Captured: {summary['frames_captured']}\n")
            f.write(f"Registers Written: {summary['registers_written']}\n")
            
            # I2C timing analysis
            if self.i2c_transactions:
                f.write(f"\nI2C Transaction Analysis:\n")
                total_i2c_time = sum(t.duration_us for t in self.i2c_transactions)
                f.write(f"Total I2C time: {total_i2c_time:.1f}µs ({total_i2c_time/1000:.2f}ms)\n")
                f.write(f"Average transaction time: {total_i2c_time/len(self.i2c_transactions):.1f}µs\n")
            
            # Frame timing analysis
            if self.frame_timings:
                f.write(f"\nFrame Timing Analysis:\n")
                for i, frame in enumerate(self.frame_timings):
                    f.write(f"Frame {i+1}: {frame.width}x{frame.height} {frame.format_name}\n")
                    f.write(f"  Estimated time: {frame.frame_time_ms:.2f}ms\n")
                    f.write(f"  Pixel clock: {frame.pixel_clock_mhz}MHz\n")
            
            f.write("\n" + "="*60 + "\n")
