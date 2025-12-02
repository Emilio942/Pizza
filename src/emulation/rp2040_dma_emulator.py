"""
RP2040 DMA Controller Emulator
Emulates the 12-channel DMA controller in the RP2040 microcontroller
with detailed timing simulation and DVP interface support.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DMATransferSize(Enum):
    """DMA transfer size options."""
    SIZE_8 = 0   # 8-bit transfers (1 byte)
    SIZE_16 = 1  # 16-bit transfers (2 bytes)
    SIZE_32 = 2  # 32-bit transfers (4 bytes)

class DMARequest(Enum):
    """DMA request sources (DREQ)."""
    TREQ_PIO0_TX0 = 0
    TREQ_PIO0_TX1 = 1
    TREQ_PIO0_RX0 = 4
    TREQ_PIO0_RX1 = 5
    TREQ_SPI0_TX = 16
    TREQ_SPI0_RX = 17
    TREQ_UART0_TX = 20
    TREQ_UART0_RX = 21
    TREQ_ADC = 36
    TREQ_XOR_IRQ = 37
    TREQ_DVP_FIFO = 38  # Custom DVP (camera) FIFO request
    TREQ_TIMER0 = 59
    TREQ_PERMANENT = 63  # Always active

class DMAChannelState(Enum):
    """DMA channel states."""
    IDLE = 0
    ACTIVE = 1
    PAUSED = 2
    ERROR = 3

@dataclass
class DMAChannelConfig:
    """DMA channel configuration."""
    # Control register fields
    enable: bool = False
    high_priority: bool = False
    data_size: DMATransferSize = DMATransferSize.SIZE_8
    incr_read: bool = True
    incr_write: bool = True
    ring_size: int = 0
    ring_sel: bool = False  # False=read, True=write
    chain_to: int = 0
    treq_sel: DMARequest = DMARequest.TREQ_PERMANENT
    irq_quiet: bool = False
    bswap: bool = False
    sniff_en: bool = False
    
    # Address and count registers
    read_addr: int = 0
    write_addr: int = 0
    trans_count: int = 0
    
    # Trigger register
    trigger: bool = False

@dataclass
class DMATransferEvent:
    """DMA transfer event for logging."""
    timestamp: float
    channel: int
    event_type: str
    source_addr: int
    dest_addr: int
    bytes_transferred: int
    trigger_source: str
    duration_us: float

class DVPInterfaceEmulator:
    """Emulates the Digital Video Port (DVP) interface for camera data."""
    
    def __init__(self, width: int = 48, height: int = 48, pixel_format: str = "RGB565"):
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.bytes_per_pixel = 2 if pixel_format == "RGB565" else 3 if pixel_format == "RGB888" else 1
        self.frame_size_bytes = width * height * self.bytes_per_pixel
        
        # DVP state
        self.enabled = False
        self.frame_ready = False
        self.current_frame_data = bytearray()
        self.fifo_buffer = bytearray()
        self.fifo_threshold = 32  # 32-byte FIFO threshold
        
        # Timing parameters
        self.pixel_clock_mhz = 12  # 12 MHz pixel clock
        self.line_blanking_cycles = 100
        self.frame_blanking_cycles = 1000
        
        logger.info(f"DVP Interface initialized: {width}x{height} {pixel_format}, {self.frame_size_bytes} bytes/frame")
    
    def enable(self):
        """Enable DVP interface."""
        self.enabled = True
        logger.debug("DVP interface enabled")
    
    def disable(self):
        """Disable DVP interface."""
        self.enabled = False
        logger.debug("DVP interface disabled")
    
    def start_frame_capture(self) -> bool:
        """Start capturing a new frame."""
        if not self.enabled:
            return False
        
        # Generate simulated camera data
        self.current_frame_data = self._generate_frame_data()
        self.frame_ready = True
        
        logger.debug(f"DVP frame capture started: {len(self.current_frame_data)} bytes")
        return True
    
    def _generate_frame_data(self) -> bytearray:
        """Generate simulated camera frame data."""
        if self.pixel_format == "RGB565":
            # Generate RGB565 data (pizza-like patterns for testing)
            data = bytearray()
            for y in range(self.height):
                for x in range(self.width):
                    # Create circular pizza-like pattern
                    center_x, center_y = self.width // 2, self.height // 2
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    
                    if dist < self.width // 3:
                        # Pizza center (yellow-orange)
                        r, g, b = 255, 200, 100
                    elif dist < self.width // 2:
                        # Pizza crust (brown)
                        r, g, b = 150, 100, 50
                    else:
                        # Background (black)
                        r, g, b = 0, 0, 0
                    
                    # Convert to RGB565
                    r5 = (r >> 3) & 0x1F
                    g6 = (g >> 2) & 0x3F
                    b5 = (b >> 3) & 0x1F
                    rgb565 = (r5 << 11) | (g6 << 5) | b5
                    
                    data.extend(rgb565.to_bytes(2, 'little'))
            
        elif self.pixel_format == "RGB888":
            # Generate RGB888 data
            data = bytearray()
            for y in range(self.height):
                for x in range(self.width):
                    # Similar pattern as above but with full 8-bit color
                    center_x, center_y = self.width // 2, self.height // 2
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    
                    if dist < self.width // 3:
                        r, g, b = 255, 200, 100
                    elif dist < self.width // 2:
                        r, g, b = 150, 100, 50
                    else:
                        r, g, b = 0, 0, 0
                    
                    data.extend([r, g, b])
        
        else:  # GRAYSCALE
            data = bytearray()
            for y in range(self.height):
                for x in range(self.width):
                    center_x, center_y = self.width // 2, self.height // 2
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    
                    if dist < self.width // 3:
                        gray = 200
                    elif dist < self.width // 2:
                        gray = 100
                    else:
                        gray = 0
                    
                    data.append(gray)
        
        return data
    
    def get_fifo_data(self, max_bytes: int) -> bytes:
        """Get data from DVP FIFO."""
        if not self.frame_ready or not self.current_frame_data:
            return b''
        
        # Simulate FIFO operation
        available_bytes = min(max_bytes, len(self.current_frame_data))
        data = bytes(self.current_frame_data[:available_bytes])
        self.current_frame_data = self.current_frame_data[available_bytes:]
        
        if not self.current_frame_data:
            self.frame_ready = False
        
        return data
    
    def fifo_has_data(self) -> bool:
        """Check if FIFO has data available."""
        return self.frame_ready and len(self.current_frame_data) > 0
    
    def generate_camera_data(self, width: int, height: int, pixel_format: str, test_pattern: str = "pizza") -> bytes:
        """Generate camera data with specified parameters."""
        # Update DVP configuration
        old_width, old_height, old_format = self.width, self.height, self.pixel_format
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.bytes_per_pixel = 2 if pixel_format == "RGB565" else 3 if pixel_format == "RGB888" else 1
        self.frame_size_bytes = width * height * self.bytes_per_pixel
        
        # Generate frame data
        data = self._generate_frame_data()
        
        # Restore original configuration
        self.width, self.height, self.pixel_format = old_width, old_height, old_format
        self.bytes_per_pixel = 2 if old_format == "RGB565" else 3 if old_format == "RGB888" else 1
        self.frame_size_bytes = old_width * old_height * self.bytes_per_pixel
        
        return bytes(data)

    def get_frame_timing_us(self) -> float:
        """Calculate frame capture timing in microseconds."""
        pixels_per_frame = self.width * self.height
        cycles_per_frame = pixels_per_frame + (self.height * self.line_blanking_cycles) + self.frame_blanking_cycles
        return (cycles_per_frame / self.pixel_clock_mhz)

class RP2040DMAEmulator:
    """Emulates the RP2040 DMA controller with 12 channels."""
    
    def __init__(self, memory_size: int = 264 * 1024):  # 264KB RAM
        self.num_channels = 12
        self.channels = [DMAChannelConfig() for _ in range(self.num_channels)]
        self.channel_states = [DMAChannelState.IDLE for _ in range(self.num_channels)]
        
        # Memory simulation
        self.memory_size = memory_size
        self.memory = bytearray(memory_size)
        
        # DVP interface
        self.dvp = DVPInterfaceEmulator()
        
        # Transfer tracking
        self.transfer_events: List[DMATransferEvent] = []
        self.active_transfers: Dict[int, Dict[str, Any]] = {}
        self.transfer_logs: List[Dict[str, Any]] = []
        self.transfer_counter: int = 0
        self.transfer_map: Dict[int, int] = {}
        self.completed_transfers: Dict[int, Dict[str, Any]] = {}
        self._stats_accumulator: Dict[str, Any] = {
            'total_transfers': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'total_duration_ms': 0.0,
        }
        
        # IRQ handling
        self.irq_callbacks: Dict[int, Callable] = {}
        
        logger.info(f"RP2040 DMA emulator initialized with {self.num_channels} channels, {memory_size/1024:.0f}KB memory")
    
    def configure_channel(self, channel: int, config: DMAChannelConfig) -> bool:
        """Configure a DMA channel."""
        if channel >= self.num_channels:
            logger.error(f"Invalid DMA channel: {channel}")
            return False
        
        self.channels[channel] = config
        logger.debug(f"DMA channel {channel} configured: {config.data_size.name} transfers, "
                    f"TREQ={config.treq_sel.name}")
        return True

    def _get_transfer_unit_size(self, data_size: DMATransferSize) -> int:
        """Return the transfer unit size in bytes for a given DMA transfer size."""
        return 1 << data_size.value

    def _resolve_channel(self, identifier: int) -> Optional[int]:
        """Resolve a transfer or channel identifier to a channel index."""
        if identifier in self.transfer_map:
            return self.transfer_map[identifier]
        if 0 <= identifier < self.num_channels:
            return identifier
        return None

    def _normalize_address(self, addr: int) -> Optional[int]:
        """Normalize RP2040-style absolute addresses to emulator memory offsets."""
        if addr < 0:
            return None
        if addr < self.memory_size:
            return addr
        # Map RP2040 SRAM window (0x2000_0000) into local memory
        sram_base = 0x20000000
        if sram_base <= addr < sram_base + self.memory_size:
            return addr - sram_base
        return None
    
    def configure_camera_dma(self, channel: int, dest_buffer_addr: int, 
                           width: int = 48, height: int = 48, 
                           pixel_format: str = "RGB565") -> bool:
        """Configure DMA for camera data transfer."""
        
        # Calculate transfer parameters
        bytes_per_pixel = 2 if pixel_format == "RGB565" else 3 if pixel_format == "RGB888" else 1
        transfer_count = width * height * bytes_per_pixel
        
        # Validate destination buffer
        normalized_dest = self._normalize_address(dest_buffer_addr)
        if normalized_dest is None:
            logger.error(f"Invalid DMA destination address for camera: 0x{dest_buffer_addr:08X}")
            return False

        if normalized_dest + transfer_count > self.memory_size:
            logger.error(
                "DMA destination buffer overflow: 0x%08X + %d > %d",
                dest_buffer_addr,
                transfer_count,
                self.memory_size,
            )
            return False
        
        # Configure DVP interface
        self.dvp = DVPInterfaceEmulator(width, height, pixel_format)
        
        # Configure DMA channel
        config = DMAChannelConfig(
            enable=True,
            high_priority=True,  # Camera data needs high priority
            data_size=DMATransferSize.SIZE_8,  # Byte transfers for camera data
            incr_read=False,  # DVP FIFO address doesn't increment
            incr_write=True,  # Destination buffer increments
            ring_size=0,  # No ring buffer
            treq_sel=DMARequest.TREQ_DVP_FIFO,  # DVP FIFO request
            read_addr=0x50000000,  # Simulated DVP FIFO register address
            write_addr=dest_buffer_addr,
            trans_count=transfer_count,
            trigger=False
        )
        
        success = self.configure_channel(channel, config)
        if success:
            logger.info(f"Camera DMA configured: channel {channel}, {width}x{height} {pixel_format}, "
                       f"{transfer_count} bytes -> 0x{dest_buffer_addr:08X}")
        
        return success
    
    def start_transfer(self, channel_id: int, source_data: Union[bytes, bytearray], description: str = "") -> Optional[int]:
        """Start a DMA transfer and return a transfer identifier."""
        channel = self._resolve_channel(channel_id) if channel_id >= self.num_channels else channel_id
        if channel is None or channel >= self.num_channels:
            logger.error(f"Invalid DMA channel for transfer start: {channel_id}")
            return None

        config = self.channels[channel]
        if not config.enable:
            logger.warning(f"DMA channel {channel} not enabled for transfer start")
            return None

        unit_size = self._get_transfer_unit_size(config.data_size)
        target_bytes = config.trans_count * unit_size
        if target_bytes <= 0:
            logger.error("Cannot start DMA transfer with zero length")
            return None

        # Prepare transfer data
        data = bytes(source_data)
        if len(data) < target_bytes:
            logger.warning(
                "Source data shorter than transfer size: %s < %s. Padding with zeros.",
                len(data),
                target_bytes,
            )
            data = data + bytes(target_bytes - len(data))
        elif len(data) > target_bytes:
            data = data[:target_bytes]

        normalized_addr = self._normalize_address(config.write_addr)
        if normalized_addr is None:
            logger.error(f"DMA transfer has invalid destination address: 0x{config.write_addr:08X}")
            return None

        if normalized_addr + target_bytes > self.memory_size:
            logger.error(
                "DMA transfer would overflow memory: 0x%08X + %d > %d",
                config.write_addr,
                target_bytes,
                self.memory_size,
            )
            return None

        self.transfer_counter += 1
        transfer_id = self.transfer_counter
        self.transfer_map[transfer_id] = channel

        transfer_info = {
            'start_time': time.time(),
            'bytes_transferred': 0,
            'target_bytes': target_bytes,
            'source_addr': config.read_addr,
            'dest_addr': config.write_addr,
            'description': description or f"{config.treq_sel.name} transfer",
            'transfer_id': transfer_id,
        }
        self.active_transfers[channel] = transfer_info
        self.channel_states[channel] = DMAChannelState.ACTIVE

        # Perform the transfer immediately for the emulator
        if not self._write_memory(config.write_addr, data):
            logger.error("DMA transfer failed while writing to memory")
            del self.active_transfers[channel]
            self.channel_states[channel] = DMAChannelState.ERROR
            self.transfer_map.pop(transfer_id, None)
            return None

        transfer_info['bytes_transferred'] = target_bytes

        # Mark transfer as complete and log
        self._complete_transfer(channel, transfer_info)

        end_time = time.time()
        integrity_verified = False
        if self.transfer_logs:
            integrity_verified = self.transfer_logs[-1].get('integrity_verified', False)
        self.completed_transfers[transfer_id] = {
            'channel': channel,
            'description': transfer_info['description'],
            'start_time': transfer_info['start_time'],
            'end_time': end_time,
            'duration_ms': (end_time - transfer_info['start_time']) * 1000,
            'bytes_transferred': target_bytes,
            'data': data,
            'integrity_verified': integrity_verified,
        }

        return transfer_id

    def wait_for_completion(self, transfer_id: int, timeout_ms: int = 100) -> bool:
        """Wait for a transfer to complete (synchronous in emulator)."""
        if transfer_id not in self.transfer_map:
            logger.error(f"Unknown transfer id: {transfer_id}")
            return False
        return transfer_id in self.completed_transfers

    def get_transfer_data(self, transfer_id: int) -> Optional[bytes]:
        """Return the transferred data for a completed transfer."""
        transfer = self.completed_transfers.get(transfer_id)
        if not transfer:
            logger.error(f"No completed transfer found for id: {transfer_id}")
            return None
        return transfer.get('data')

    def trigger_transfer(self, channel: int) -> bool:
        """Trigger a DMA transfer."""
        if channel >= self.num_channels:
            return False
        
        config = self.channels[channel]
        if not config.enable:
            logger.warning(f"DMA channel {channel} not enabled")
            return False
        
        if self.channel_states[channel] != DMAChannelState.IDLE:
            logger.warning(f"DMA channel {channel} not idle: {self.channel_states[channel]}")
            return False
        
        # Start transfer
        self.channel_states[channel] = DMAChannelState.ACTIVE
        
        # For camera transfers, start frame capture
        if config.treq_sel == DMARequest.TREQ_DVP_FIFO:
            self.dvp.enable()
            self.dvp.start_frame_capture()
        
        # Record transfer start
        transfer_info = {
            'start_time': time.time(),
            'bytes_transferred': 0,
            'target_bytes': config.trans_count * self._get_transfer_unit_size(config.data_size),
            'source_addr': config.read_addr,
            'dest_addr': config.write_addr
        }
        self.active_transfers[channel] = transfer_info
        
        logger.debug(f"DMA transfer triggered on channel {channel}: {config.trans_count} bytes")
        return True
    
    def process_transfers(self) -> List[int]:
        """Process active DMA transfers. Returns list of completed channels."""
        completed_channels = []
        
        for channel, transfer_info in list(self.active_transfers.items()):
            config = self.channels[channel]
            
            if self.channel_states[channel] != DMAChannelState.ACTIVE:
                continue
            
            # Simulate transfer progress
            if config.treq_sel == DMARequest.TREQ_DVP_FIFO:
                completed = self._process_camera_transfer(channel, transfer_info)
            else:
                completed = self._process_generic_transfer(channel, transfer_info)
            
            if completed:
                completed_channels.append(channel)
                
        return completed_channels
    
    def _process_camera_transfer(self, channel: int, transfer_info: Dict) -> bool:
        """Process camera-specific DMA transfer."""
        config = self.channels[channel]
        
        # Check if DVP has data available
        if not self.dvp.fifo_has_data():
            return False
        
        # Calculate transfer rate (simulate hardware limitations)
        remaining_bytes = transfer_info['target_bytes'] - transfer_info['bytes_transferred']
        if remaining_bytes <= 0:
            transfer_info['bytes_transferred'] = transfer_info['target_bytes']
            self._complete_transfer(channel, transfer_info)
            return True

        bytes_per_transfer = min(32, remaining_bytes)
        
        # Get data from DVP FIFO
        data = self.dvp.get_fifo_data(bytes_per_transfer)
        
        if data:
            # Write to memory
            dest_addr = transfer_info['dest_addr'] + transfer_info['bytes_transferred']
            self._write_memory(dest_addr, data)
            
            transfer_info['bytes_transferred'] += len(data)
            
            # Check if transfer is complete
            if transfer_info['bytes_transferred'] >= transfer_info['target_bytes']:
                self._complete_transfer(channel, transfer_info)
                return True
        
        return False
    
    def _process_generic_transfer(self, channel: int, transfer_info: Dict) -> bool:
        """Process generic DMA transfer."""
        # Simulate generic transfer (placeholder)
        transfer_info['bytes_transferred'] = transfer_info['target_bytes']
        self._complete_transfer(channel, transfer_info)
        return True
    
    def _complete_transfer(self, channel: int, transfer_info: Dict):
        """Complete a DMA transfer."""
        config = self.channels[channel]
        duration = time.time() - transfer_info['start_time']
        transfer_id = transfer_info.get('transfer_id')
        description = transfer_info.get('description', f"{config.treq_sel.name} transfer")
        
        # Record transfer event
        event = DMATransferEvent(
            timestamp=time.time(),
            channel=channel,
            event_type="TRANSFER_COMPLETE",
            source_addr=transfer_info['source_addr'],
            dest_addr=transfer_info['dest_addr'],
            bytes_transferred=transfer_info['bytes_transferred'],
            trigger_source=config.treq_sel.name,
            duration_us=duration * 1_000_000
        )
        self.transfer_events.append(event)

        integrity_verified = False
        try:
            integrity_verified = self.verify_transfer_integrity(channel)
        except Exception as exc:
            logger.error(f"Integrity verification failed for channel {channel}: {exc}")
            integrity_verified = False

        self._stats_accumulator['total_transfers'] += 1
        if integrity_verified:
            self._stats_accumulator['successful_transfers'] += 1
        else:
            self._stats_accumulator['failed_transfers'] += 1
        self._stats_accumulator['total_duration_ms'] += duration * 1000

        log_transfer_id = transfer_id if transfer_id is not None else len(self.transfer_logs) + 1
        if transfer_id is None and log_transfer_id not in self.transfer_map:
            self.transfer_map[log_transfer_id] = channel

        log_entry = {
            'transfer_id': log_transfer_id,
            'timestamp': event.timestamp,
            'channel_id': channel,
            'description': description,
            'duration_ms': duration * 1000,
            'bytes_transferred': transfer_info.get('bytes_transferred', 0),
            'integrity_verified': integrity_verified,
        }
        self.transfer_logs.append(log_entry)
        
        # Update channel state
        self.channel_states[channel] = DMAChannelState.IDLE
        
        # Trigger IRQ if not quiet
        if not config.irq_quiet:
            self._trigger_irq(channel)
        
        # Chain to next channel if configured
        if config.chain_to != channel:
            self.trigger_transfer(config.chain_to)
        
        # Remove from active transfers
        del self.active_transfers[channel]
        
        logger.info(f"DMA transfer completed: channel {channel}, {transfer_info['bytes_transferred']} bytes, "
                   f"{duration*1000:.2f}ms")
    
    def _write_memory(self, addr: int, data: bytes):
        """Write data to emulated memory."""
        normalized = self._normalize_address(addr)
        if normalized is None:
            logger.error(f"Invalid memory write address: 0x{addr:08X}")
            return False

        if normalized + len(data) > self.memory_size:
            logger.error(f"Memory write overflow: {normalized + len(data)} > {self.memory_size}")
            return False
        
        self.memory[normalized:normalized + len(data)] = data
        return True
    
    def read_memory(self, addr: int, length: int) -> bytes:
        """Read data from emulated memory."""
        normalized = self._normalize_address(addr)
        if normalized is None:
            logger.error(f"Invalid memory read address: 0x{addr:08X}")
            return b''

        if normalized + length > self.memory_size:
            logger.error(f"Memory read overflow: {normalized + length} > {self.memory_size}")
            return b''
        
        return bytes(self.memory[normalized:normalized + length])
    
    def _trigger_irq(self, channel: int):
        """Trigger DMA IRQ."""
        if channel in self.irq_callbacks:
            self.irq_callbacks[channel]()
        
        logger.debug(f"DMA IRQ triggered for channel {channel}")
    
    def register_irq_callback(self, channel: int, callback: Callable):
        """Register IRQ callback for a channel."""
        self.irq_callbacks[channel] = callback
    
    def get_transfer_stats(self) -> Dict:
        """Get DMA transfer statistics."""
        total_transfers = len(self.transfer_events)
        total_bytes = sum(event.bytes_transferred for event in self.transfer_events)
        
        if self.transfer_events:
            avg_duration = sum(event.duration_us for event in self.transfer_events) / total_transfers
            throughput_mbps = (total_bytes * 8) / (sum(event.duration_us for event in self.transfer_events) / 1_000_000) / 1_000_000
        else:
            avg_duration = 0
            throughput_mbps = 0
        
        return {
            'total_transfers': total_transfers,
            'total_bytes_transferred': total_bytes,
            'average_duration_us': avg_duration,
            'throughput_mbps': throughput_mbps,
            'active_channels': len(self.active_transfers)
        }

    def get_statistics(self) -> Dict:
        """Return high-level DMA statistics for camera integration."""
        totals = dict(self._stats_accumulator)
        average_duration = 0.0
        if totals['total_transfers']:
            average_duration = totals['total_duration_ms'] / totals['total_transfers']

        base_stats = self.get_transfer_stats()
        last_transfer = self.transfer_logs[-1] if self.transfer_logs else None

        return {
            'dma_enabled': True,
            'total_transfers': totals['total_transfers'],
            'successful_transfers': totals['successful_transfers'],
            'failed_transfers': totals['failed_transfers'],
            'average_transfer_time_ms': average_duration,
            'throughput_mbps': base_stats['throughput_mbps'],
            'active_channels': base_stats['active_channels'],
            'last_transfer': last_transfer,
        }

    def get_transfer_logs(self) -> List[Dict]:
        """Return a copy of detailed transfer logs."""
        return list(self.transfer_logs)
    
    def verify_transfer_integrity(self, identifier: int, expected_pattern: Optional[bytes] = None) -> bool:
        """Verify the integrity of transferred data."""
        channel = self._resolve_channel(identifier)
        if channel is None:
            logger.error(f"Cannot verify integrity for unknown identifier: {identifier}")
            return False
        
        config = self.channels[channel]
        transfer_length = config.trans_count * self._get_transfer_unit_size(config.data_size)
        if transfer_length <= 0:
            logger.error("Transfer length is zero; integrity check not possible")
            return False
        
        transferred_data = self.read_memory(config.write_addr, transfer_length)
        
        if expected_pattern:
            if len(transferred_data) != len(expected_pattern):
                logger.error(f"Data length mismatch: {len(transferred_data)} != {len(expected_pattern)}")
                return False
            mismatches = sum(1 for a, b in zip(transferred_data, expected_pattern) if a != b)
            if mismatches > 0:
                logger.error(f"Data integrity check failed: {mismatches} mismatches")
                return False
        
        if len(transferred_data) != transfer_length:
            logger.error(
                "Transfer count mismatch: %s bytes read != %s expected",
                len(transferred_data),
                transfer_length,
            )
            return False
        
        if not any(b != 0 for b in transferred_data):
            logger.warning("Transferred data is all zeros - may indicate transfer issue")
            return False
        
        logger.info(f"DMA transfer integrity verified: channel {channel}, {len(transferred_data)} bytes")
        return True
    
    def get_channel_status(self, channel: int) -> Dict:
        """Get status of a specific DMA channel."""
        if channel >= self.num_channels:
            return {}
        
        config = self.channels[channel]
        state = self.channel_states[channel]
        
        status = {
            'channel': channel,
            'state': state.name,
            'enabled': config.enable,
            'source_addr': f"0x{config.read_addr:08X}",
            'dest_addr': f"0x{config.write_addr:08X}",
            'transfer_count': config.trans_count,
            'trigger_source': config.treq_sel.name,
            'data_size': config.data_size.name
        }
        
        if channel in self.active_transfers:
            transfer_info = self.active_transfers[channel]
            status['bytes_transferred'] = transfer_info['bytes_transferred']
            status['progress_percent'] = (transfer_info['bytes_transferred'] / transfer_info['target_bytes']) * 100
        
        return status

    def get_channel_state(self, channel: int) -> Optional[Dict]:
        """Return simplified channel state information for tests."""
        if not (0 <= channel < self.num_channels):
            logger.error(f"Invalid DMA channel query: {channel}")
            return None

        config = self.channels[channel]
        state = self.channel_states[channel]

        if state == DMAChannelState.ACTIVE:
            status_value = 'active'
        elif state == DMAChannelState.ERROR:
            status_value = 'error'
        elif config.enable:
            status_value = 'configured'
        else:
            status_value = 'disabled'

        info: Dict[str, Union[int, str, bool]] = {
            'channel': channel,
            'status': status_value,
            'state': state.name,
            'enabled': config.enable,
            'read_addr': config.read_addr,
            'write_addr': config.write_addr,
            'transfer_count': config.trans_count,
            'data_size': config.data_size.name,
            'trigger_source': config.treq_sel.name,
            'chain_to': config.chain_to,
        }

        completed_transfer = None
        for transfer in reversed(list(self.completed_transfers.values())):
            if transfer['channel'] == channel:
                completed_transfer = transfer
                break
        if completed_transfer:
            info['last_transfer_bytes'] = completed_transfer.get('bytes_transferred', 0)
            info['integrity_verified'] = completed_transfer.get('integrity_verified', False)

        return info
