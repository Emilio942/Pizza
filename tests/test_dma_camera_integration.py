#!/usr/bin/env python3
"""
Tests for DMA camera integration in RP2040 emulator.
Verifies that the DMA transfer functionality for camera data is working correctly.
"""

import unittest
import time
import sys
import os
import logging

# Add import path for emulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.emulation.emulator import RP2040Emulator, CameraEmulator
    from src.emulation.rp2040_dma_emulator import RP2040DMAEmulator, DVPInterfaceEmulator, DMAChannelConfig, DMATransferSize, DMARequest
    from src.emulation.frame_buffer import PixelFormat
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this test from the correct directory.")
    sys.exit(1)

class TestDMACameraIntegration(unittest.TestCase):
    """Tests for DMA camera integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging to capture test output
        logging.basicConfig(level=logging.DEBUG)
        
        # Create emulator instance
        self.emulator = RP2040Emulator()
        
        # Initialize camera
        self.camera = self.emulator.camera
        self.camera.initialize()
        
    def test_dma_emulator_initialization(self):
        """Test that DMA emulator is properly initialized in RP2040Emulator."""
        self.assertIsNotNone(self.emulator.dma_emulator)
        self.assertIsNotNone(self.emulator.dvp_interface)
        self.assertIsInstance(self.emulator.dma_emulator, RP2040DMAEmulator)
        self.assertIsInstance(self.emulator.dvp_interface, DVPInterfaceEmulator)
        
    def test_camera_dma_mode_enabled(self):
        """Test that camera has DMA mode enabled."""
        self.assertTrue(hasattr(self.camera, 'dma_emulator'))
        self.assertTrue(hasattr(self.camera, 'dvp_interface'))
        self.assertIsNotNone(self.camera.dma_emulator)
        self.assertIsNotNone(self.camera.dvp_interface)
        
    def test_dma_channel_configuration(self):
        """Test DMA channel configuration for camera data."""
        # Configure DMA channel for a standard capture
        channel_id = 0
        
        config = DMAChannelConfig(
            read_addr=0x50000000,  # DVP FIFO
            write_addr=0x20000000,  # RAM
            trans_count=100,
            data_size=DMATransferSize.SIZE_32,
            treq_sel=DMARequest.TREQ_DVP_FIFO,
            chain_to=0,
            incr_read=False,
            incr_write=True,
            enable=True
        )
        
        success = self.emulator.dma_emulator.configure_channel(channel_id, config)
        self.assertTrue(success)
        
        # Check that configuration was applied
        channel_state = self.emulator.dma_emulator.get_channel_state(channel_id)
        self.assertEqual(channel_state['read_addr'], 0x50000000)
        self.assertEqual(channel_state['write_addr'], 0x20000000)
        
    def test_dvp_data_generation(self):
        """Test DVP interface data generation."""
        # Test different formats and resolutions
        test_cases = [
            (48, 48, "RGB565", "pizza"),
            (320, 240, "RGB565", "random"),
            (160, 120, "GRAYSCALE", "random")
        ]
        
        for width, height, pixel_format, pattern in test_cases:
            with self.subTest(width=width, height=height, format=pixel_format):
                # Set DVP parameters temporarily
                old_width, old_height, old_format = self.emulator.dvp_interface.width, self.emulator.dvp_interface.height, self.emulator.dvp_interface.pixel_format
                self.emulator.dvp_interface.width = width
                self.emulator.dvp_interface.height = height
                self.emulator.dvp_interface.pixel_format = pixel_format
                
                try:
                    data = self.emulator.dvp_interface._generate_frame_data()
                    
                    # Verify data properties
                    expected_bytes_per_pixel = 2 if pixel_format == "RGB565" else 1
                    expected_size = width * height * expected_bytes_per_pixel
                    
                    self.assertIsNotNone(data)
                    self.assertEqual(len(data), expected_size)
                    self.assertIsInstance(data, bytearray)
                finally:
                    # Restore original parameters
                    self.emulator.dvp_interface.width = old_width
                    self.emulator.dvp_interface.height = old_height
                    self.emulator.dvp_interface.pixel_format = old_format
                
    def test_dma_camera_capture_pizza_detection_format(self):
        """Test DMA camera capture with pizza detection format (48x48 RGB565)."""
        # Configure camera for pizza detection
        self.camera.configure_for_pizza_detection()
        
        # Capture frame with DMA
        frame = self.camera.capture_frame()
        
        # Verify frame properties
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (48, 48, 3))  # RGB format for numpy array
        self.assertEqual(frame.dtype.name, 'uint8')
        
        # Get DMA statistics
        dma_stats = self.camera.get_dma_statistics()
        self.assertTrue(dma_stats["dma_enabled"])
        self.assertGreater(dma_stats["total_transfers"], 0)
        
    def test_dma_camera_capture_different_resolutions(self):
        """Test DMA camera capture with different resolutions."""
        test_resolutions = [
            (160, 120),
            (320, 240),
            (48, 48)
        ]
        
        for width, height in test_resolutions:
            with self.subTest(width=width, height=height):
                # Set camera format
                self.camera.set_format(width, height, rgb=True)
                
                # Capture frame
                frame = self.camera.capture_frame()
                
                # Verify frame properties
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape, (height, width, 3))
                
                # Check DMA statistics
                dma_stats = self.camera.get_dma_statistics()
                self.assertTrue(dma_stats["dma_enabled"])
                
    def test_dma_transfer_integrity(self):
        """Test DMA transfer data integrity verification."""
        # Configure for a small frame to make verification easier
        self.camera.set_format(32, 32, rgb=False)  # Grayscale
        
        # Capture frame
        frame = self.camera.capture_frame()
        
        # Get detailed DMA statistics
        dma_stats = self.camera.get_dma_statistics()
        
        # Verify integrity checks passed
        self.assertTrue(dma_stats["dma_enabled"])
        self.assertGreater(dma_stats["successful_transfers"], 0)
        self.assertEqual(dma_stats["failed_transfers"], 0)
        
        # Check transfer logs for integrity verification
        transfer_logs = self.emulator.dma_emulator.get_transfer_logs()
        self.assertGreater(len(transfer_logs), 0)
        
        # Verify last transfer had integrity check
        last_transfer = transfer_logs[-1]
        self.assertIn("integrity_verified", last_transfer)
        self.assertTrue(last_transfer["integrity_verified"])
        
    def test_dma_transfer_performance(self):
        """Test DMA transfer performance and timing."""
        # Configure for high-resolution capture
        self.camera.set_format(320, 240, rgb=True)
        
        # Measure capture time
        start_time = time.time()
        frame = self.camera.capture_frame()
        end_time = time.time()
        
        capture_time = end_time - start_time
        
        # Verify frame was captured
        self.assertIsNotNone(frame)
        
        # Get DMA performance statistics
        dma_stats = self.camera.get_dma_statistics()
        
        # Check transfer timing
        self.assertIn("average_transfer_time_ms", dma_stats)
        self.assertGreater(dma_stats["average_transfer_time_ms"], 0)
        
        # DMA should be faster than a reasonable threshold (e.g., 50ms for 320x240)
        self.assertLess(capture_time, 0.1)  # Should complete within 100ms
        
    def test_cpu_fallback_functionality(self):
        """Test CPU fallback when DMA is not available."""
        # Temporarily disable DMA by removing the attribute
        original_dma = self.camera.dma_emulator
        delattr(self.camera, 'dma_emulator')
        
        try:
            # Capture should still work using CPU-based method
            frame = self.camera.capture_frame()
            
            # Verify frame was captured
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (48, 48, 3))  # Current format
            
        finally:
            # Restore DMA emulator
            self.camera.dma_emulator = original_dma
            
    def test_dma_error_handling(self):
        """Test DMA error handling and recovery."""
        # Force a DMA configuration error by using invalid parameters
        channel_id = 15  # Invalid channel (only 0-11 are valid)
        
        config = DMAChannelConfig(
            read_addr=0x50000000,
            write_addr=0x20000000,
            trans_count=100,
            data_size=DMATransferSize.SIZE_32,
            treq_sel=DMARequest.TREQ_DVP_FIFO,
            chain_to=0,
            incr_read=False,
            incr_write=True,
            enable=True
        )
        
        # This should fail
        success = self.emulator.dma_emulator.configure_channel(channel_id, config)
        self.assertFalse(success)
        
        # Camera capture should still work (fallback to CPU)
        frame = self.camera.capture_frame()
        self.assertIsNotNone(frame)
        
    def test_concurrent_dma_channels(self):
        """Test handling of multiple DMA channels."""
        # Configure multiple channels
        for channel_id in range(3):
            config = DMAChannelConfig(
                read_addr=0x50000000 + channel_id * 0x1000,
                write_addr=0x20000000 + channel_id * 0x10000,
                trans_count=50,
                data_size=DMATransferSize.SIZE_32,
                treq_sel=DMARequest.TREQ_DVP_FIFO,
                chain_to=0,
                incr_read=False,
                incr_write=True,
                enable=True
            )
            
            success = self.emulator.dma_emulator.configure_channel(channel_id, config)
            self.assertTrue(success, f"Failed to configure channel {channel_id}")
            
        # Verify all channels are configured
        for channel_id in range(3):
            state = self.emulator.dma_emulator.get_channel_state(channel_id)
            self.assertIsNotNone(state)
            self.assertEqual(state['status'], 'configured')
            
    def test_emulator_integration_with_firmware_loading(self):
        """Test DMA functionality with firmware loading."""
        # Load a realistic firmware
        firmware = {
            'ram_usage_bytes': 80 * 1024,  # 80KB
            'total_size_bytes': 150 * 1024  # 150KB
        }
        
        # Should load successfully (DMA doesn't affect RAM calculation)
        self.emulator.load_firmware(firmware)
        self.assertTrue(self.emulator.firmware_loaded)
        
        # Camera capture should still work with firmware loaded
        frame = self.camera.capture_frame()
        self.assertIsNotNone(frame)
        
        # Check that DMA is still functional
        dma_stats = self.camera.get_dma_statistics()
        self.assertTrue(dma_stats["dma_enabled"])
        
    def test_detailed_logging_output(self):
        """Test that DMA operations produce detailed logging."""
        # Enable debug logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # Configure for pizza detection and capture frame
        self.camera.configure_for_pizza_detection()
        frame = self.camera.capture_frame()
        
        # Get transfer logs
        transfer_logs = self.emulator.dma_emulator.get_transfer_logs()
        
        # Verify we have detailed logs
        self.assertGreater(len(transfer_logs), 0)
        
        last_log = transfer_logs[-1]
        required_fields = [
            'timestamp', 'channel_id', 'transfer_id', 'description',
            'duration_ms', 'bytes_transferred', 'integrity_verified'
        ]
        
        for field in required_fields:
            self.assertIn(field, last_log, f"Missing field: {field}")


def run_dma_demo():
    """Run a demonstration of DMA functionality."""
    print("=" * 60)
    print("DMA Camera Integration Demo")
    print("=" * 60)
    
    # Create emulator
    emulator = RP2040Emulator()
    camera = emulator.camera
    
    # Initialize camera
    print("Initializing camera...")
    camera.initialize()
    
    # Configure for pizza detection
    print("Configuring camera for pizza detection (48x48 RGB565)...")
    camera.configure_for_pizza_detection()
    
    # Capture several frames to show DMA in action
    print("\nCapturing frames with DMA transfers:")
    for i in range(5):
        print(f"\nFrame {i+1}:")
        start_time = time.time()
        frame = camera.capture_frame()
        end_time = time.time()
        
        print(f"  - Frame shape: {frame.shape}")
        print(f"  - Capture time: {(end_time - start_time)*1000:.2f} ms")
        
        # Show frame statistics
        print(f"  - Min pixel value: {frame.min()}")
        print(f"  - Max pixel value: {frame.max()}")
        print(f"  - Mean pixel value: {frame.mean():.1f}")
        
    # Show DMA statistics
    print("\nDMA Transfer Statistics:")
    dma_stats = camera.get_dma_statistics()
    for key, value in dma_stats.items():
        print(f"  - {key}: {value}")
    
    # Show detailed transfer logs
    print("\nRecent DMA Transfer Logs:")
    transfer_logs = emulator.dma_emulator.get_transfer_logs()
    for log in transfer_logs[-3:]:  # Show last 3 transfers
        print(f"  - Transfer {log['transfer_id']}: "
              f"{log['bytes_transferred']} bytes in {log['duration_ms']:.2f} ms "
              f"(Integrity: {'✓' if log['integrity_verified'] else '✗'})")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    # Ask user what to run
    import argparse
    
    parser = argparse.ArgumentParser(description="DMA Camera Integration Test")
    parser.add_argument("--demo", action="store_true", help="Run DMA demo instead of tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.demo:
        run_dma_demo()
    else:
        # Configure logging for tests
        log_level = logging.DEBUG if args.verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        
        # Run tests
        unittest.main(verbosity=2 if args.verbose else 1)
