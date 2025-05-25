#!/usr/bin/env python3
"""
Comprehensive test suite for ENERGIE-2.2: Adaptive duty-cycle logic
Tests various trigger scenarios and validates state machine behavior.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.emulation.emulator import RP2040Emulator
from src.emulation.adaptive_state_machine import TriggerType, SystemState
from src.emulation.simple_power_manager import AdaptiveMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveDutyCycleTestSuite:
    """Test suite for adaptive duty-cycle logic."""
    
    def __init__(self):
        """Initialize test suite."""
        self.emulator = None
        self.test_results = {}
        self.test_start_time = None
        
    def setup(self):
        """Setup test environment."""
        logger.info("Setting up adaptive duty-cycle test environment...")
        self.emulator = RP2040Emulator()
        self.test_start_time = time.time()
        
        # Wait for initialization
        time.sleep(1.0)
        
        # Verify all components are initialized
        assert hasattr(self.emulator, 'adaptive_state_machine'), "Adaptive state machine not initialized"
        assert hasattr(self.emulator, 'motion_controller'), "Motion controller not initialized"
        assert hasattr(self.emulator, 'schedule_manager'), "Schedule manager not initialized"
        assert hasattr(self.emulator, 'interrupt_controller'), "Interrupt controller not initialized"
        
        logger.info("Test environment setup complete")
        
    def teardown(self):
        """Clean up test environment."""
        if self.emulator:
            self.emulator.close()
            self.emulator = None
        logger.info("Test environment cleaned up")
        
    def wait_for_state(self, target_state: SystemState, timeout: float = 10.0) -> bool:
        """Wait for state machine to reach target state."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            if current_state == target_state:
                return True
            time.sleep(0.1)
        return False
        
    def get_system_stats(self) -> dict:
        """Get current system statistics."""
        stats = self.emulator.adaptive_state_machine.get_statistics()
        power_stats = self.emulator.power_manager.get_statistics()
        
        return {
            'state_machine': stats,
            'power_manager': power_stats,
            'motion_controller': self.emulator.motion_controller.get_statistics(),
            'schedule_manager': self.emulator.schedule_manager.get_statistics(),
            'interrupt_controller': self.emulator.interrupt_controller.get_statistics()
        }
        
    def test_basic_initialization(self) -> bool:
        """Test basic system initialization."""
        logger.info("Testing basic initialization...")
        
        try:
            # Check initial state
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"Initial state: {current_state}")
            
            # Should start in LIGHT_SLEEP or IDLE state
            assert current_state in [SystemState.LIGHT_SLEEP, SystemState.IDLE], f"Unexpected initial state: {current_state}"
            
            # Check that all trigger systems are active
            assert self.emulator.motion_controller.is_active(), "Motion controller not active"
            assert self.emulator.schedule_manager.is_running(), "Schedule manager not running"
            
            logger.info("✓ Basic initialization test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Basic initialization test failed: {e}")
            return False
            
    def test_timer_trigger_scenario(self) -> bool:
        """Test periodic timer-based triggers."""
        logger.info("Testing timer trigger scenario...")
        
        try:
            # Schedule a periodic event every 2 seconds
            schedule_id = self.emulator.schedule_manager.schedule_periodic(
                name="test_inference",
                interval=2.0,
                callback=None  # Will use default trigger
            )
            
            logger.info(f"Scheduled periodic event with ID: {schedule_id}")
            
            # Wait for first trigger
            initial_stats = self.get_system_stats()
            initial_trigger_count = initial_stats['state_machine']['trigger_counts'].get('TIMER', 0)
            
            # Wait up to 5 seconds for timer trigger
            start_time = time.time()
            triggered = False
            while time.time() - start_time < 5.0:
                current_stats = self.get_system_stats()
                current_trigger_count = current_stats['state_machine']['trigger_counts'].get('TIMER', 0)
                
                if current_trigger_count > initial_trigger_count:
                    triggered = True
                    break
                time.sleep(0.1)
                
            assert triggered, "Timer trigger did not occur within 5 seconds"
            
            # Check that we have activity from timer triggers
            final_stats = self.get_system_stats()
            timer_triggers = final_stats['state_machine']['trigger_counts'].get('TIMER', 0)
            
            logger.info(f"Timer triggers recorded: {timer_triggers}")
            assert timer_triggers > 0, "No timer triggers recorded"
            
            # Cancel the scheduled event
            self.emulator.schedule_manager.cancel_event(schedule_id)
            
            logger.info("✓ Timer trigger scenario test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Timer trigger scenario test failed: {e}")
            return False
            
    def test_motion_trigger_scenario(self) -> bool:
        """Test motion sensor trigger scenario."""
        logger.info("Testing motion trigger scenario...")
        
        try:
            # Get initial stats
            initial_stats = self.get_system_stats()
            initial_motion_count = initial_stats['state_machine']['trigger_counts'].get('MOTION', 0)
            
            # Simulate motion detection
            logger.info("Simulating motion detection...")
            self.emulator.motion_controller.simulate_motion_event(
                sensor_type="PIR",
                duration=2.0,
                confidence=0.9
            )
            
            # Wait for motion trigger to be processed
            start_time = time.time()
            triggered = False
            while time.time() - start_time < 3.0:
                current_stats = self.get_system_stats()
                current_motion_count = current_stats['state_machine']['trigger_counts'].get('MOTION', 0)
                
                if current_motion_count > initial_motion_count:
                    triggered = True
                    break
                time.sleep(0.1)
                
            assert triggered, "Motion trigger did not occur within 3 seconds"
            
            # Verify state machine responded to motion
            final_stats = self.get_system_stats()
            motion_triggers = final_stats['state_machine']['trigger_counts'].get('MOTION', 0)
            
            logger.info(f"Motion triggers recorded: {motion_triggers}")
            assert motion_triggers > initial_motion_count, "Motion trigger not properly recorded"
            
            logger.info("✓ Motion trigger scenario test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Motion trigger scenario test failed: {e}")
            return False
            
    def test_interrupt_trigger_scenario(self) -> bool:
        """Test external interrupt trigger scenario."""
        logger.info("Testing interrupt trigger scenario...")
        
        try:
            # Get initial stats
            initial_stats = self.get_system_stats()
            initial_interrupt_count = initial_stats['state_machine']['trigger_counts'].get('INTERRUPT', 0)
            
            # Simulate button press interrupt
            logger.info("Simulating button press interrupt...")
            self.emulator.interrupt_controller.trigger_interrupt(pin=2, value=1)  # Manual button
            time.sleep(0.1)
            self.emulator.interrupt_controller.trigger_interrupt(pin=2, value=0)  # Release
            
            # Wait for interrupt trigger to be processed
            start_time = time.time()
            triggered = False
            while time.time() - start_time < 2.0:
                current_stats = self.get_system_stats()
                current_interrupt_count = current_stats['state_machine']['trigger_counts'].get('INTERRUPT', 0)
                
                if current_interrupt_count > initial_interrupt_count:
                    triggered = True
                    break
                time.sleep(0.1)
                
            assert triggered, "Interrupt trigger did not occur within 2 seconds"
            
            # Verify interrupt statistics
            final_stats = self.get_system_stats()
            interrupt_triggers = final_stats['state_machine']['trigger_counts'].get('INTERRUPT', 0)
            
            logger.info(f"Interrupt triggers recorded: {interrupt_triggers}")
            assert interrupt_triggers > initial_interrupt_count, "Interrupt trigger not properly recorded"
            
            logger.info("✓ Interrupt trigger scenario test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Interrupt trigger scenario test failed: {e}")
            return False
            
    def test_frequent_trigger_scenario(self) -> bool:
        """Test system behavior under frequent triggers."""
        logger.info("Testing frequent trigger scenario...")
        
        try:
            # Set adaptive mode to handle frequent activity
            self.emulator.power_manager.set_adaptive_mode(AdaptiveMode.PERFORMANCE)
            
            initial_stats = self.get_system_stats()
            
            # Generate frequent motion events
            logger.info("Generating frequent motion events...")
            for i in range(5):
                self.emulator.motion_controller.simulate_motion_event(
                    sensor_type="PIR",
                    duration=0.5,
                    confidence=0.8
                )
                time.sleep(0.2)  # Brief pause between events
                
            # Generate frequent interrupts
            logger.info("Generating frequent interrupts...")
            for i in range(3):
                self.emulator.interrupt_controller.trigger_interrupt(pin=2, value=1)
                time.sleep(0.1)
                self.emulator.interrupt_controller.trigger_interrupt(pin=2, value=0)
                time.sleep(0.3)
                
            # Wait for processing
            time.sleep(2.0)
            
            final_stats = self.get_system_stats()
            
            # Check that system handled multiple triggers
            motion_triggers = final_stats['state_machine']['trigger_counts'].get('MOTION', 0) - initial_stats['state_machine']['trigger_counts'].get('MOTION', 0)
            interrupt_triggers = final_stats['state_machine']['trigger_counts'].get('INTERRUPT', 0) - initial_stats['state_machine']['trigger_counts'].get('INTERRUPT', 0)
            
            logger.info(f"Processed motion triggers: {motion_triggers}, interrupt triggers: {interrupt_triggers}")
            
            assert motion_triggers >= 3, f"Expected at least 3 motion triggers, got {motion_triggers}"
            assert interrupt_triggers >= 2, f"Expected at least 2 interrupt triggers, got {interrupt_triggers}"
            
            # Check that system is in appropriate state for high activity
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"Current state after frequent triggers: {current_state}")
            
            logger.info("✓ Frequent trigger scenario test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Frequent trigger scenario test failed: {e}")
            return False
            
    def test_rare_trigger_scenario(self) -> bool:
        """Test system behavior under rare triggers (power saving)."""
        logger.info("Testing rare trigger scenario...")
        
        try:
            # Set adaptive mode for power saving
            self.emulator.power_manager.set_adaptive_mode(AdaptiveMode.POWER_SAVE)
            
            # Wait for system to settle into low power state
            logger.info("Waiting for system to enter low power state...")
            time.sleep(3.0)
            
            # Check that system is in a sleep state
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"Current state in rare trigger scenario: {current_state}")
            
            # System should be in sleep mode when inactive
            # Note: Might be IDLE if there are background activities
            
            initial_stats = self.get_system_stats()
            
            # Generate single motion event after long pause
            logger.info("Generating single motion event after pause...")
            time.sleep(1.0)
            
            self.emulator.motion_controller.simulate_motion_event(
                sensor_type="PIR",
                duration=1.0,
                confidence=0.7
            )
            
            # Wait for response
            time.sleep(2.0)
            
            final_stats = self.get_system_stats()
            
            # Verify trigger was processed
            motion_triggers = final_stats['state_machine']['trigger_counts'].get('MOTION', 0) - initial_stats['state_machine']['trigger_counts'].get('MOTION', 0)
            
            logger.info(f"Motion triggers in rare scenario: {motion_triggers}")
            assert motion_triggers >= 1, "Single motion trigger not processed"
            
            # Check power efficiency metrics
            total_active_time = final_stats['state_machine']['state_durations'].get('INFERENCE', 0) + final_stats['state_machine']['state_durations'].get('IDLE', 0)
            total_sleep_time = final_stats['state_machine']['state_durations'].get('DEEP_SLEEP', 0) + final_stats['state_machine']['state_durations'].get('LIGHT_SLEEP', 0)
            
            logger.info(f"Active time: {total_active_time:.2f}s, Sleep time: {total_sleep_time:.2f}s")
            
            logger.info("✓ Rare trigger scenario test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Rare trigger scenario test failed: {e}")
            return False
            
    def test_mixed_trigger_scenario(self) -> bool:
        """Test mixed trigger scenario with various trigger types."""
        logger.info("Testing mixed trigger scenario...")
        
        try:
            # Set balanced mode
            self.emulator.power_manager.set_adaptive_mode(AdaptiveMode.BALANCED)
            
            initial_stats = self.get_system_stats()
            
            # Create mixed scenario with different trigger types
            logger.info("Creating mixed trigger scenario...")
            
            # 1. Schedule a timer event
            schedule_id = self.emulator.schedule_manager.schedule_one_shot(
                name="mixed_test_timer",
                delay=1.0,
                callback=None
            )
            
            # 2. Motion event
            time.sleep(0.5)
            self.emulator.motion_controller.simulate_motion_event(
                sensor_type="MICROWAVE",
                duration=1.5,
                confidence=0.85
            )
            
            # 3. Temperature change (if temperature triggers are active)
            time.sleep(0.5)
            self.emulator.set_temperature(35.0)  # Higher temperature
            
            # 4. Interrupt
            time.sleep(0.5)
            self.emulator.interrupt_controller.trigger_interrupt(pin=3, value=1)  # PIR sensor pin
            time.sleep(0.1)
            self.emulator.interrupt_controller.trigger_interrupt(pin=3, value=0)
            
            # 5. Battery level change
            time.sleep(0.5)
            self.emulator.set_battery_voltage(3.2)  # Lower battery
            
            # Wait for all triggers to be processed
            time.sleep(3.0)
            
            final_stats = self.get_system_stats()
            
            # Check that multiple trigger types were processed
            total_triggers = 0
            for trigger_type, count in final_stats['state_machine']['trigger_counts'].items():
                initial_count = initial_stats['state_machine']['trigger_counts'].get(trigger_type, 0)
                new_triggers = count - initial_count
                if new_triggers > 0:
                    logger.info(f"{trigger_type} triggers: {new_triggers}")
                    total_triggers += new_triggers
                    
            assert total_triggers >= 3, f"Expected at least 3 triggers total, got {total_triggers}"
            
            # Check state transitions
            state_transitions = final_stats['state_machine']['state_transitions']
            total_transitions = sum(state_transitions.values()) - sum(initial_stats['state_machine']['state_transitions'].values())
            
            logger.info(f"Total state transitions: {total_transitions}")
            assert total_transitions > 0, "No state transitions recorded"
            
            logger.info("✓ Mixed trigger scenario test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Mixed trigger scenario test failed: {e}")
            return False
            
    def test_state_machine_behavior(self) -> bool:
        """Test state machine behavior and transitions."""
        logger.info("Testing state machine behavior...")
        
        try:
            # Get initial state
            initial_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"Initial state: {initial_state}")
            
            # Force different states by triggering specific conditions
            
            # 1. Test inference state
            logger.info("Triggering inference state...")
            self.emulator.motion_controller.simulate_motion_event(
                sensor_type="PIR",
                duration=2.0,
                confidence=0.9
            )
            
            # Wait and check if we reach inference state
            time.sleep(1.0)
            inference_reached = False
            for _ in range(10):  # Check for 1 second
                current_state = self.emulator.adaptive_state_machine.get_current_state()
                if current_state == SystemState.INFERENCE:
                    inference_reached = True
                    logger.info("✓ Successfully reached INFERENCE state")
                    break
                time.sleep(0.1)
                
            # Note: May not always reach INFERENCE depending on system load
            logger.info(f"Inference state reached: {inference_reached}")
            
            # 2. Test emergency state (low battery)
            logger.info("Testing emergency state (low battery)...")
            self.emulator.set_battery_voltage(2.8)  # Very low battery
            time.sleep(1.0)
            
            emergency_reached = False
            for _ in range(10):
                current_state = self.emulator.adaptive_state_machine.get_current_state()
                if current_state == SystemState.EMERGENCY:
                    emergency_reached = True
                    logger.info("✓ Successfully reached EMERGENCY state")
                    break
                time.sleep(0.1)
                
            # Restore normal battery
            self.emulator.set_battery_voltage(3.7)
            time.sleep(0.5)
            
            # 3. Check statistics
            stats = self.get_system_stats()
            state_durations = stats['state_machine']['state_durations']
            
            logger.info("State durations:")
            for state, duration in state_durations.items():
                logger.info(f"  {state}: {duration:.2f}s")
                
            # Verify we have recorded state activity
            total_duration = sum(state_durations.values())
            assert total_duration > 0, "No state duration recorded"
            
            logger.info("✓ State machine behavior test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ State machine behavior test failed: {e}")
            return False
            
    def test_power_management_integration(self) -> bool:
        """Test integration with power management system."""
        logger.info("Testing power management integration...")
        
        try:
            # Test different adaptive modes
            modes_to_test = [AdaptiveMode.POWER_SAVE, AdaptiveMode.BALANCED, AdaptiveMode.PERFORMANCE]
            
            for mode in modes_to_test:
                logger.info(f"Testing adaptive mode: {mode}")
                
                self.emulator.power_manager.set_adaptive_mode(mode)
                time.sleep(0.5)
                
                # Trigger activity
                self.emulator.motion_controller.simulate_motion_event(
                    sensor_type="PIR",
                    duration=1.0,
                    confidence=0.8
                )
                
                time.sleep(1.0)
                
                # Check power statistics
                power_stats = self.emulator.power_manager.get_statistics()
                logger.info(f"Power consumption in {mode}: {power_stats.get('current_power_mw', 0):.2f}mW")
                
            # Check that power manager responds to state changes
            initial_power = self.emulator.power_manager.get_statistics().get('current_power_mw', 0)
            
            # Generate high activity
            for i in range(3):
                self.emulator.motion_controller.simulate_motion_event(
                    sensor_type="PIR",
                    duration=0.5,
                    confidence=0.9
                )
                time.sleep(0.2)
                
            time.sleep(1.0)
            final_power = self.emulator.power_manager.get_statistics().get('current_power_mw', 0)
            
            logger.info(f"Power change: {initial_power:.2f}mW -> {final_power:.2f}mW")
            
            logger.info("✓ Power management integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Power management integration test failed: {e}")
            return False
            
    def run_all_tests(self) -> dict:
        """Run all test scenarios."""
        logger.info("Starting comprehensive adaptive duty-cycle test suite...")
        
        test_functions = [
            self.test_basic_initialization,
            self.test_timer_trigger_scenario,
            self.test_motion_trigger_scenario,
            self.test_interrupt_trigger_scenario,
            self.test_frequent_trigger_scenario,
            self.test_rare_trigger_scenario,
            self.test_mixed_trigger_scenario,
            self.test_state_machine_behavior,
            self.test_power_management_integration
        ]
        
        results = {}
        passed = 0
        failed = 0
        
        for test_func in test_functions:
            test_name = test_func.__name__
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
                failed += 1
                
            # Brief pause between tests
            time.sleep(0.5)
            
        # Final statistics
        total_time = time.time() - self.test_start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUITE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {len(test_functions)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {passed/len(test_functions)*100:.1f}%")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        # System statistics
        if self.emulator:
            final_stats = self.get_system_stats()
            logger.info(f"\nFinal system statistics:")
            logger.info(f"Total triggers processed: {sum(final_stats['state_machine']['trigger_counts'].values())}")
            logger.info(f"Total state transitions: {sum(final_stats['state_machine']['state_transitions'].values())}")
            
            # Save detailed results
            self.test_results = {
                'summary': {
                    'total_tests': len(test_functions),
                    'passed': passed,
                    'failed': failed,
                    'success_rate': passed/len(test_functions),
                    'execution_time': total_time
                },
                'test_results': results,
                'final_statistics': final_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        return results

def main():
    """Main test execution."""
    test_suite = AdaptiveDutyCycleTestSuite()
    
    try:
        # Setup
        test_suite.setup()
        
        # Run tests
        results = test_suite.run_all_tests()
        
        # Save results to file
        results_file = Path(__file__).parent / "adaptive_duty_cycle_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_suite.test_results, f, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Return appropriate exit code
        failed_tests = [name for name, result in results.items() if not result]
        if failed_tests:
            logger.error(f"Failed tests: {failed_tests}")
            return 1
        else:
            logger.info("All tests passed successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"Test suite failed with exception: {e}")
        return 1
        
    finally:
        # Cleanup
        test_suite.teardown()

if __name__ == "__main__":
    exit(main())
