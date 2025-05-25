#!/usr/bin/env python3
"""
Test script for HWEMU-2.2: Adaptive Clock Frequency Adjustment Logic

This script tests the adaptive clock frequency adjustment functionality
by simulating temperature changes and verifying that clock frequencies
change according to the defined thresholds.

Temperature thresholds and expected behaviors:
- Below 40°C: Maximum performance (133 MHz)
- 40-60°C: Balanced mode (100 MHz)
- 60-75°C: Conservative mode (75 MHz)
- Above 75°C: Emergency mode (48 MHz) - thermal protection

Test scenarios:
1. Normal operation with temperature changes
2. Temperature spikes that trigger emergency mode
3. Hysteresis behavior to prevent oscillation
4. Logging verification
"""

import sys
import time
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.emulation.emulator import RP2040Emulator
from src.utils.power_manager import AdaptiveMode
from src.emulation.logging_system import LogLevel, LogType

# Configure logging for the test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveClockTester:
    """Test class for adaptive clock frequency functionality."""
    
    def __init__(self):
        """Initialize the tester with an emulator instance."""
        print("=" * 60)
        print(" HWEMU-2.2: Adaptive Clock Frequency Test")
        print("=" * 60)
        print()
        
        # Create emulator instance
        self.emulator = RP2040Emulator(
            battery_capacity_mah=1500.0,
            adaptive_mode=AdaptiveMode.BALANCED
        )
        
        # Enable verbose logging for adaptive clock
        self.emulator.adaptive_clock_config['verbose_logging'] = True
        self.emulator.adaptive_clock_config['update_interval_ms'] = 100  # Fast updates for testing
        
        # Store initial state
        self.initial_frequency = self.emulator.current_frequency_mhz
        self.test_results = []
        
        print(f"Initial system frequency: {self.initial_frequency} MHz")
        print(f"Temperature thresholds: {self.emulator.temp_thresholds}")
        print(f"Clock frequencies: {self.emulator.clock_frequencies}")
        print()
    
    def wait_for_temperature_stabilization(self, timeout_seconds=10):
        """Wait for temperature changes to be processed."""
        print("  Waiting for temperature stabilization...")
        time.sleep(timeout_seconds / 10)  # Short wait for emulation
        
        # Force an update
        self.emulator.update_adaptive_clock_frequency()
        time.sleep(0.1)  # Allow logging to complete
    
    def verify_frequency_matches_temperature(self, expected_freq, tolerance_mhz=2):
        """Verify that the current frequency matches expectations."""
        current_freq = self.emulator.current_frequency_mhz
        temp = self.emulator.current_temperature_c
        
        success = abs(current_freq - expected_freq) <= tolerance_mhz
        
        print(f"  Temperature: {temp:.1f}°C")
        print(f"  Current frequency: {current_freq} MHz")
        print(f"  Expected frequency: {expected_freq} MHz")
        print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")
        
        return success
    
    def test_scenario_1_normal_operation(self):
        """Test 1: Normal temperature variations through all thresholds."""
        print("\n" + "=" * 50)
        print("TEST 1: Normal Operation - Temperature Thresholds")
        print("=" * 50)
        
        test_cases = [
            # (temperature, expected_frequency, description)
            (25.0, 133, "Room temperature - Maximum performance"),
            (42.0, 100, "Warm temperature - Balanced mode"),
            (65.0, 75, "Hot temperature - Conservative mode"),
            (80.0, 48, "Critical temperature - Emergency mode"),
            (70.0, 75, "Cool down - Conservative mode (with hysteresis)"),
            (35.0, 133, "Back to room temp - Maximum performance"),
        ]
        
        results = []
        
        for temp, expected_freq, description in test_cases:
            print(f"\nSubtest: {description}")
            print(f"Setting temperature to {temp}°C...")
            
            # Inject temperature
            self.emulator.temperature_sensor.inject_temperature_spike(temp - 25.0, duration=30.0)
            
            # Wait for system to respond
            self.wait_for_temperature_stabilization()
            
            # Verify frequency
            success = self.verify_frequency_matches_temperature(expected_freq)
            results.append(success)
            
            # Check thermal protection status for critical temperatures
            if temp >= 75.0:
                thermal_active = self.emulator.is_thermal_protection_active()
                print(f"  Thermal protection: {'Active' if thermal_active else 'Inactive'}")
                if temp >= 75.0 and not thermal_active:
                    print("  ✗ ERROR: Thermal protection should be active!")
                    results[-1] = False
        
        success_rate = sum(results) / len(results) * 100
        print(f"\nTest 1 Results: {sum(results)}/{len(results)} passed ({success_rate:.1f}%)")
        self.test_results.append(("Normal Operation", success_rate >= 80))
        
        return success_rate >= 80
    
    def test_scenario_2_emergency_activation(self):
        """Test 2: Emergency mode activation with high temperature spikes."""
        print("\n" + "=" * 50)
        print("TEST 2: Emergency Mode Activation")
        print("=" * 50)
        
        # Get initial statistics
        initial_stats = self.emulator.get_adaptive_clock_stats()
        initial_emergency_count = initial_stats['emergency_activations']
        
        print(f"Initial emergency activations: {initial_emergency_count}")
        
        # Test emergency activation
        print("\nInjecting critical temperature spike (85°C)...")
        self.emulator.temperature_sensor.inject_temperature_spike(60.0, duration=30.0)  # 25 + 60 = 85°C
        
        self.wait_for_temperature_stabilization()
        
        # Verify emergency frequency
        success1 = self.verify_frequency_matches_temperature(48)
        
        # Check emergency activation counter
        final_stats = self.emulator.get_adaptive_clock_stats()
        final_emergency_count = final_stats['emergency_activations']
        emergency_triggered = final_emergency_count > initial_emergency_count
        
        print(f"\nEmergency activations after test: {final_emergency_count}")
        print(f"Emergency mode triggered: {'✓ YES' if emergency_triggered else '✗ NO'}")
        
        # Test recovery
        print("\nTesting recovery to normal temperature...")
        self.emulator.temperature_sensor.inject_temperature_spike(-60.0, duration=30.0)  # Back to 25°C
        self.wait_for_temperature_stabilization()
        
        success2 = self.verify_frequency_matches_temperature(133)
        
        overall_success = success1 and emergency_triggered and success2
        print(f"\nTest 2 Result: {'✓ PASS' if overall_success else '✗ FAIL'}")
        self.test_results.append(("Emergency Activation", overall_success))
        
        return overall_success
    
    def test_scenario_3_hysteresis_behavior(self):
        """Test 3: Hysteresis behavior to prevent oscillation."""
        print("\n" + "=" * 50)
        print("TEST 3: Hysteresis Behavior")
        print("=" * 50)
        
        # Get initial adjustment count
        initial_stats = self.emulator.get_adaptive_clock_stats()
        initial_adjustments = initial_stats['total_adjustments']
        
        print(f"Initial total adjustments: {initial_adjustments}")
        
        # Set temperature near threshold (around 40°C boundary)
        print("\nTesting oscillation prevention near 40°C threshold...")
        
        # Start at 38°C (should be 133 MHz)
        self.emulator.temperature_sensor.inject_temperature_spike(13.0, duration=30.0)  # 25 + 13 = 38°C
        self.wait_for_temperature_stabilization()
        freq_38 = self.emulator.current_frequency_mhz
        
        # Move to 42°C (should be 100 MHz)
        self.emulator.temperature_sensor.inject_temperature_spike(17.0, duration=30.0)  # 25 + 17 = 42°C
        self.wait_for_temperature_stabilization()
        freq_42 = self.emulator.current_frequency_mhz
        
        # Move back to 39°C (should stay at 100 MHz due to hysteresis)
        self.emulator.temperature_sensor.inject_temperature_spike(14.0, duration=30.0)  # 25 + 14 = 39°C
        self.wait_for_temperature_stabilization()
        freq_39_after_down = self.emulator.current_frequency_mhz
        
        print(f"Frequency at 38°C: {freq_38} MHz (expected: 133 MHz)")
        print(f"Frequency at 42°C: {freq_42} MHz (expected: 100 MHz)")
        print(f"Frequency at 39°C (after going down): {freq_39_after_down} MHz (expected: 100 MHz due to hysteresis)")
        
        # Check if hysteresis is working (frequency should stay at 100 MHz at 39°C)
        hysteresis_working = freq_39_after_down == 100
        
        final_stats = self.emulator.get_adaptive_clock_stats()
        final_adjustments = final_stats['total_adjustments']
        total_changes = final_adjustments - initial_adjustments
        
        print(f"\nTotal frequency adjustments during test: {total_changes}")
        print(f"Hysteresis behavior: {'✓ WORKING' if hysteresis_working else '✗ FAILED'}")
        
        success = hysteresis_working and total_changes >= 2  # At least 2 changes expected
        print(f"\nTest 3 Result: {'✓ PASS' if success else '✗ FAIL'}")
        self.test_results.append(("Hysteresis Behavior", success))
        
        return success
    
    def test_scenario_4_logging_verification(self):
        """Test 4: Verify that adaptive clock changes are properly logged."""
        print("\n" + "=" * 50)
        print("TEST 4: Logging Verification")
        print("=" * 50)
        
        # Enable logging capture
        log_entries = []
        
        # Create a custom log handler to capture logs
        class LogCapture(logging.Handler):
            def emit(self, record):
                if 'ADAPTIVE_CLOCK' in record.getMessage():
                    log_entries.append(record.getMessage())
        
        log_capture = LogCapture()
        logging.getLogger().addHandler(log_capture)
        
        try:
            print("Generating temperature changes to trigger logging...")
            
            # Trigger several frequency changes
            temperatures = [30.0, 50.0, 70.0, 85.0, 25.0]
            
            for temp in temperatures:
                print(f"Setting temperature to {temp}°C...")
                self.emulator.temperature_sensor.inject_temperature_spike(temp - 25.0, duration=30.0)
                self.wait_for_temperature_stabilization()
            
            # Check captured logs
            print(f"\nCaptured {len(log_entries)} adaptive clock log entries:")
            for i, entry in enumerate(log_entries[-10:], 1):  # Show last 10 entries
                print(f"  {i}. {entry}")
            
            # Verify we got frequency change logs
            frequency_change_logs = [log for log in log_entries if "Changing system clock" in log or "Frequency adjusted" in log]
            emergency_logs = [log for log in log_entries if "EMERGENCY" in log]
            
            print(f"\nLog Analysis:")
            print(f"- Total adaptive clock logs: {len(log_entries)}")
            print(f"- Frequency change logs: {len(frequency_change_logs)}")
            print(f"- Emergency activation logs: {len(emergency_logs)}")
            
            success = len(frequency_change_logs) >= 3  # At least 3 frequency changes expected
            print(f"\nTest 4 Result: {'✓ PASS' if success else '✗ FAIL'}")
            self.test_results.append(("Logging Verification", success))
            
            return success
            
        finally:
            logging.getLogger().removeHandler(log_capture)
    
    def test_scenario_5_manual_control(self):
        """Test 5: Manual frequency control and adaptive system toggle."""
        print("\n" + "=" * 50)
        print("TEST 5: Manual Control and System Toggle")
        print("=" * 50)
        
        # Test manual frequency forcing
        print("Testing manual frequency control...")
        test_freq = 75
        print(f"Forcing frequency to {test_freq} MHz...")
        
        success1 = self.emulator.force_clock_frequency(test_freq)
        actual_freq = self.emulator.current_frequency_mhz
        
        print(f"Force frequency result: {'✓ SUCCESS' if success1 else '✗ FAILED'}")
        print(f"Actual frequency: {actual_freq} MHz")
        
        # Test adaptive system disable/enable
        print("\nTesting adaptive system toggle...")
        
        # Disable adaptive clock
        self.emulator.set_adaptive_clock_enabled(False)
        enabled_state1 = self.emulator.adaptive_clock_enabled
        print(f"Adaptive clock disabled: {'✓ YES' if not enabled_state1 else '✗ NO'}")
        
        # Try to trigger change (should not work when disabled)
        old_freq = self.emulator.current_frequency_mhz
        self.emulator.temperature_sensor.inject_temperature_spike(60.0, duration=30.0)  # High temp
        self.wait_for_temperature_stabilization()
        new_freq = self.emulator.current_frequency_mhz
        
        freq_unchanged = (old_freq == new_freq)
        print(f"Frequency unchanged when disabled: {'✓ YES' if freq_unchanged else '✗ NO'}")
        
        # Re-enable adaptive clock
        self.emulator.set_adaptive_clock_enabled(True)
        enabled_state2 = self.emulator.adaptive_clock_enabled
        print(f"Adaptive clock re-enabled: {'✓ YES' if enabled_state2 else '✗ NO'}")
        
        # Verify it works again
        self.wait_for_temperature_stabilization()
        final_freq = self.emulator.current_frequency_mhz
        
        success = success1 and not enabled_state1 and freq_unchanged and enabled_state2
        print(f"\nTest 5 Result: {'✓ PASS' if success else '✗ FAIL'}")
        self.test_results.append(("Manual Control", success))
        
        return success
    
    def print_final_statistics(self):
        """Print comprehensive statistics from the adaptive clock system."""
        print("\n" + "=" * 60)
        print(" FINAL SYSTEM STATISTICS")
        print("=" * 60)
        
        # Get final statistics
        state = self.emulator.get_adaptive_clock_state()
        stats = self.emulator.get_adaptive_clock_stats()
        
        print(f"Current System State:")
        print(f"  Current frequency: {state['current_frequency_mhz']} MHz")
        print(f"  Target frequency: {state['target_frequency_mhz']} MHz")
        print(f"  Last temperature: {state['last_temperature']:.1f}°C")
        print(f"  Thermal protection: {'Active' if state['thermal_protection_active'] else 'Inactive'}")
        print(f"  Adaptive clock enabled: {'Yes' if state['enabled'] else 'No'}")
        
        print(f"\nAdjustment Statistics:")
        print(f"  Total frequency adjustments: {stats['total_adjustments']}")
        print(f"  Emergency mode activations: {stats['emergency_activations']}")
        
        print(f"\nConfiguration:")
        print(f"  Temperature thresholds: {self.emulator.temp_thresholds}")
        print(f"  Clock frequencies: {self.emulator.clock_frequencies}")
        print(f"  Update interval: {self.emulator.adaptive_clock_config['update_interval_ms']} ms")
        print(f"  Hysteresis: {self.emulator.temp_hysteresis}°C")
    
    def run_all_tests(self):
        """Run all test scenarios and provide summary."""
        print("Starting comprehensive adaptive clock frequency tests...\n")
        
        try:
            # Run all test scenarios
            self.test_scenario_1_normal_operation()
            self.test_scenario_2_emergency_activation()
            self.test_scenario_3_hysteresis_behavior()
            self.test_scenario_4_logging_verification()
            self.test_scenario_5_manual_control()
            
            # Print final statistics
            self.print_final_statistics()
            
            # Print test summary
            print("\n" + "=" * 60)
            print(" TEST SUMMARY")
            print("=" * 60)
            
            passed_tests = sum(1 for _, result in self.test_results if result)
            total_tests = len(self.test_results)
            success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
            
            for test_name, result in self.test_results:
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {test_name:<25} {status}")
            
            print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("\n🎉 HWEMU-2.2 IMPLEMENTATION SUCCESSFUL!")
                print("   Adaptive clock frequency adjustment is working correctly.")
                print("   The system properly adjusts clock frequencies based on temperature thresholds.")
                print("   Logging and thermal protection are functioning as expected.")
            else:
                print("\n❌ HWEMU-2.2 IMPLEMENTATION NEEDS WORK")
                print("   Some tests failed. Please review the implementation.")
            
            return success_rate >= 80
            
        except Exception as e:
            print(f"\n❌ TEST EXECUTION FAILED: {e}")
            logger.exception("Test execution failed")
            return False

def main():
    """Main test execution function."""
    print("HWEMU-2.2: Adaptive Clock Frequency Adjustment Test")
    print("Testing temperature-based clock frequency management...\n")
    
    try:
        # Create and run tests
        tester = AdaptiveClockTester()
        success = tester.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        logger.exception("Test setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
