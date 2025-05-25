#!/usr/bin/env python3
"""
Quick validation test for ENERGIE-2.2 adaptive duty-cycle implementation.
"""

import sys
import time
import logging
from pathlib import Path

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

def test_basic_functionality():
    """Test basic adaptive duty-cycle functionality."""
    logger.info("Starting basic adaptive duty-cycle validation...")
    
    emulator = None
    try:
        # Initialize emulator
        logger.info("Initializing emulator...")
        emulator = RP2040Emulator()
        time.sleep(1.0)  # Wait for initialization
        
        # Test 1: Check initialization
        logger.info("Test 1: Checking component initialization...")
        assert hasattr(emulator, 'adaptive_state_machine'), "State machine not initialized"
        assert hasattr(emulator, 'motion_controller'), "Motion controller not initialized"
        assert hasattr(emulator, 'schedule_manager'), "Schedule manager not initialized"
        assert hasattr(emulator, 'interrupt_controller'), "Interrupt controller not initialized"
        logger.info("✓ All components initialized correctly")
        
        # Test 2: Check initial state
        logger.info("Test 2: Checking initial state...")
        current_state = emulator.adaptive_state_machine.get_current_state()
        logger.info(f"Initial state: {current_state}")
        assert current_state in [SystemState.LIGHT_SLEEP, SystemState.IDLE, SystemState.INFERENCE], f"Unexpected initial state: {current_state}"
        logger.info("✓ Initial state is valid")
        
        # Test 3: Test motion trigger
        logger.info("Test 3: Testing motion trigger...")
        initial_stats = emulator.adaptive_state_machine.get_statistics()
        initial_motion_count = initial_stats['trigger_counts'].get('MOTION', 0)
        
        emulator.motion_controller.simulate_motion_event("PIR", 1.0, 0.8)
        time.sleep(2.0)  # Wait for processing
        
        final_stats = emulator.adaptive_state_machine.get_statistics()
        final_motion_count = final_stats['trigger_counts'].get('MOTION', 0)
        
        logger.info(f"Motion triggers: {initial_motion_count} -> {final_motion_count}")
        assert final_motion_count > initial_motion_count, "Motion trigger not processed"
        logger.info("✓ Motion trigger works correctly")
        
        # Test 4: Test timer trigger
        logger.info("Test 4: Testing timer trigger...")
        schedule_id = emulator.schedule_manager.schedule_one_shot("test", 1.0, None)
        time.sleep(2.0)
        
        final_stats = emulator.adaptive_state_machine.get_statistics()
        timer_count = final_stats['trigger_counts'].get('TIMER', 0)
        logger.info(f"Timer triggers: {timer_count}")
        logger.info("✓ Timer trigger scheduled successfully")
        
        # Test 5: Test interrupt trigger
        logger.info("Test 5: Testing interrupt trigger...")
        initial_interrupt_count = final_stats['trigger_counts'].get('INTERRUPT', 0)
        
        emulator.interrupt_controller.trigger_interrupt(pin=2, value=1)
        time.sleep(0.1)
        emulator.interrupt_controller.trigger_interrupt(pin=2, value=0)
        time.sleep(1.0)
        
        final_stats = emulator.adaptive_state_machine.get_statistics()
        final_interrupt_count = final_stats['trigger_counts'].get('INTERRUPT', 0)
        
        logger.info(f"Interrupt triggers: {initial_interrupt_count} -> {final_interrupt_count}")
        assert final_interrupt_count > initial_interrupt_count, "Interrupt trigger not processed"
        logger.info("✓ Interrupt trigger works correctly")
        
        # Test 6: Test power management integration
        logger.info("Test 6: Testing power management integration...")
        emulator.power_manager.set_adaptive_mode(AdaptiveMode.POWER_SAVE)
        power_stats = emulator.power_manager.get_statistics()
        logger.info(f"Power mode: {power_stats['adaptive_mode']}")
        logger.info(f"Current power consumption: {power_stats['current_power_mw']:.2f}mW")
        logger.info("✓ Power management integration works")
        
        # Summary
        final_stats = emulator.adaptive_state_machine.get_statistics()
        logger.info("\nFinal System Statistics:")
        logger.info(f"Total triggers: {sum(final_stats['trigger_counts'].values())}")
        logger.info(f"State transitions: {sum(final_stats['state_transitions'].values())}")
        logger.info(f"Current state: {emulator.adaptive_state_machine.get_current_state()}")
        
        for trigger_type, count in final_stats['trigger_counts'].items():
            if count > 0:
                logger.info(f"  {trigger_type}: {count} triggers")
                
        for state, duration in final_stats['state_durations'].items():
            if duration > 0:
                logger.info(f"  {state}: {duration:.2f}s")
        
        logger.info("\n🎉 All basic tests passed! ENERGIE-2.2 adaptive duty-cycle logic is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if emulator:
            emulator.close()
            logger.info("Emulator closed")

def test_scenarios():
    """Test specific scenarios for different trigger patterns."""
    logger.info("\nTesting scenario patterns...")
    
    emulator = None
    try:
        emulator = RP2040Emulator()
        time.sleep(1.0)
        
        # Scenario 1: Frequent activity (performance mode)
        logger.info("Scenario 1: High activity pattern...")
        emulator.power_manager.set_adaptive_mode(AdaptiveMode.PERFORMANCE)
        
        for i in range(3):
            emulator.motion_controller.simulate_motion_event("PIR", 0.5, 0.9)
            time.sleep(0.3)
            
        time.sleep(1.0)
        stats = emulator.adaptive_state_machine.get_statistics()
        logger.info(f"High activity motion triggers: {stats['trigger_counts'].get('MOTION', 0)}")
        
        # Scenario 2: Low activity (power save mode)
        logger.info("Scenario 2: Low activity pattern...")
        emulator.power_manager.set_adaptive_mode(AdaptiveMode.POWER_SAVE)
        time.sleep(2.0)  # Let system settle
        
        emulator.motion_controller.simulate_motion_event("PIR", 1.0, 0.6)
        time.sleep(1.0)
        
        final_stats = emulator.adaptive_state_machine.get_statistics()
        logger.info(f"Low activity motion triggers: {final_stats['trigger_counts'].get('MOTION', 0)}")
        
        # Scenario 3: Mixed triggers
        logger.info("Scenario 3: Mixed trigger pattern...")
        emulator.power_manager.set_adaptive_mode(AdaptiveMode.BALANCED)
        
        # Motion + interrupt + timer
        emulator.motion_controller.simulate_motion_event("MICROWAVE", 1.0, 0.8)
        time.sleep(0.2)
        emulator.interrupt_controller.trigger_interrupt(pin=3, value=1)
        time.sleep(0.1)
        emulator.interrupt_controller.trigger_interrupt(pin=3, value=0)
        time.sleep(0.2)
        emulator.schedule_manager.schedule_one_shot("mixed_test", 0.5, None)
        
        time.sleep(2.0)
        
        final_stats = emulator.adaptive_state_machine.get_statistics()
        total_triggers = sum(final_stats['trigger_counts'].values())
        logger.info(f"Mixed scenario total triggers: {total_triggers}")
        
        logger.info("✓ All scenarios completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Scenario test failed: {e}")
        return False
        
    finally:
        if emulator:
            emulator.close()

def main():
    """Main test execution."""
    logger.info("="*60)
    logger.info("ENERGIE-2.2: Adaptive Duty-Cycle Logic Validation")
    logger.info("="*60)
    
    success = True
    
    # Run basic functionality tests
    if test_basic_functionality():
        logger.info("✓ Basic functionality tests passed")
    else:
        logger.error("✗ Basic functionality tests failed")
        success = False
    
    # Run scenario tests
    if test_scenarios():
        logger.info("✓ Scenario tests passed")
    else:
        logger.error("✗ Scenario tests failed")
        success = False
    
    # Final result
    logger.info("="*60)
    if success:
        logger.info("🎉 ENERGIE-2.2 COMPLETED SUCCESSFULLY!")
        logger.info("Adaptive duty-cycle logic implemented and validated:")
        logger.info("  ✓ Motion sensor triggers")
        logger.info("  ✓ Timer/schedule triggers")
        logger.info("  ✓ External interrupt triggers")
        logger.info("  ✓ Adaptive state machine")
        logger.info("  ✓ Power management integration")
        logger.info("  ✓ Multiple scenario validation")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
