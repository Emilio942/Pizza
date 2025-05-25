#!/usr/bin/env python3
"""
Simple Sleep Mode Performance Test
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.emulation.emulator import RP2040Emulator
from src.emulation.simple_power_manager import AdaptiveMode

def test_sleep_wake_performance():
    """Test sleep-wake performance with timing measurements."""
    print("Testing sleep-wake performance...")
    
    emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.POWER_SAVE)
    
    # Load test firmware
    test_firmware = {
        'path': 'test_sleep_optimization.bin',
        'total_size_bytes': 80 * 1024,
        'model_size_bytes': 40 * 1024,
        'ram_usage_bytes': 30 * 1024,
        'model_input_size': (48, 48)
    }
    emulator.load_firmware(test_firmware)
    
    print(f"Initial state - Sleep mode: {emulator.sleep_mode}")
    print(f"Initial RAM usage: {emulator.get_ram_usage() / 1024:.1f}KB")
    
    sleep_times = []
    wake_times = []
    
    for i in range(10):
        print(f"\nCycle {i+1}/10:")
        
        # Measure sleep transition
        start_time = time.perf_counter()
        emulator.power_manager.enter_sleep_mode()
        sleep_time = (time.perf_counter() - start_time) * 1000
        sleep_times.append(sleep_time)
        
        print(f"  Sleep transition: {sleep_time:.3f}ms")
        print(f"  Sleep mode active: {emulator.sleep_mode}")
        print(f"  RAM in sleep: {emulator.get_ram_usage() / 1024:.1f}KB")
        
        # Short sleep period
        time.sleep(0.05)
        
        # Measure wake transition
        start_time = time.perf_counter()
        emulator.power_manager.wake_up()
        wake_time = (time.perf_counter() - start_time) * 1000
        wake_times.append(wake_time)
        
        print(f"  Wake transition: {wake_time:.3f}ms")
        print(f"  Sleep mode cleared: {not emulator.sleep_mode}")
        print(f"  RAM after wake: {emulator.get_ram_usage() / 1024:.1f}KB")
    
    # Calculate statistics
    if sleep_times and wake_times:
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Sleep transitions:")
        print(f"  Mean: {sum(sleep_times)/len(sleep_times):.3f}ms")
        print(f"  Max: {max(sleep_times):.3f}ms")
        print(f"  Min: {min(sleep_times):.3f}ms")
        
        print(f"Wake transitions:")
        print(f"  Mean: {sum(wake_times)/len(wake_times):.3f}ms")
        print(f"  Max: {max(wake_times):.3f}ms")
        print(f"  Min: {min(wake_times):.3f}ms")
        
        # Check 10ms requirement
        max_wake_time = max(wake_times)
        print(f"\nRequirement check:")
        print(f"  Wake-up time < 10ms: {'✓ PASS' if max_wake_time < 10.0 else '✗ FAIL'}")
        print(f"  Max wake time: {max_wake_time:.3f}ms")
    
    emulator.close()
    return sleep_times, wake_times

if __name__ == "__main__":
    try:
        sleep_times, wake_times = test_sleep_wake_performance()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
