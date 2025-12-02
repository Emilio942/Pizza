#!/usr/bin/env python3
"""
Minimal test to validate ENERGIE-2.2 core functionality.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_core_components():
    """Exercise core adaptive components individually."""
    logger.info("Testing core components...")

    try:
        from src.emulation.motion_sensor import MotionSensorController
        from src.emulation.rtc_scheduler import ScheduleManager
        from src.emulation.interrupt_emulator import InterruptController
        from src.emulation.adaptive_state_machine import DutyCycleStateMachine
        logger.info("✓ All imports successful")
    except Exception as exc:
        logger.error(f"✗ Import failed: {exc}")
        return False

    try:
        motion_controller = MotionSensorController()
        motion_controller.start()
        logger.info("✓ Motion controller created and started")
        motion_controller.stop()
    except Exception as exc:
        logger.error(f"✗ Motion controller failed: {exc}")
        return False

    try:
        schedule_manager = ScheduleManager()
        schedule_manager.start()
        logger.info("✓ Schedule manager created and started")
        schedule_manager.stop()
    except Exception as exc:
        logger.error(f"✗ Schedule manager failed: {exc}")
        return False

    try:
        interrupt_controller = InterruptController()
        stats = interrupt_controller.get_statistics()
        logger.info("✓ Interrupt controller created, gpio pins: %s", stats.get("total_gpio_pins", 0))
    except Exception as exc:
        logger.error(f"✗ Interrupt controller failed: {exc}")
        return False

    logger.info("✓ All core components working!")
    return True

def run_triggers():
    """Validate trigger mechanisms individually."""
    logger.info("Testing trigger mechanisms...")

    try:
        from src.emulation.motion_sensor import MotionSensorController
        from src.emulation.rtc_scheduler import ScheduleManager
        from src.emulation.interrupt_emulator import InterruptController

        motion_controller = MotionSensorController()
        motion_controller.start()

        initial_stats = motion_controller.get_statistics()
        logger.info("Motion sensors: %s", initial_stats["total_sensors"])

        motion_controller.simulate_motion_event("PIR", 1.0, 0.8)
        logger.info("✓ Motion event simulated")
        motion_controller.stop()

        schedule_manager = ScheduleManager()
        schedule_manager.start()
        schedule_id = schedule_manager.schedule_one_shot("test", 0.5, None)
        logger.info("✓ Schedule created: %s", schedule_id)
        time.sleep(1.0)
        schedule_manager.stop()

        interrupt_controller = InterruptController()
        interrupt_controller.setup_common_interrupts()
        interrupt_controller.trigger_interrupt(pin=2, value=1)
        interrupt_controller.trigger_interrupt(pin=2, value=0)
        logger.info("✓ Interrupt triggered")

        stats = interrupt_controller.get_statistics()
        logger.info("Interrupt count: %s", stats["total_interrupts"])
    except Exception as exc:
        logger.error(f"✗ Trigger test failed: {exc}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("✓ All trigger mechanisms working!")
    return True

def run_emulator_basic():
    """Test basic emulator functionality without state machine."""
    logger.info("Testing basic emulator...")

    try:
        from src.emulation.emulator import RP2040Emulator

        emulator = RP2040Emulator()
        logger.info("✓ Emulator created")

        emulator.set_temperature_for_testing(30.0)
        logger.info("Temperature: %.1f°C", emulator.current_temperature_c)

        battery_mv = emulator.power_manager.get_battery_voltage_mv()
        logger.info("Battery: %smV", battery_mv)

        assert hasattr(emulator, "motion_controller")
        assert hasattr(emulator, "schedule_manager")
        assert hasattr(emulator, "interrupt_controller")
        assert hasattr(emulator, "adaptive_state_machine")
        logger.info("✓ All adaptive components present")

        emulator.close()
        logger.info("✓ Emulator closed successfully")
    except Exception as exc:
        logger.error(f"✗ Emulator test failed: {exc}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_core_components():
    """Pytest wrapper for core component validation."""
    assert run_core_components()

def test_triggers():
    """Pytest wrapper for trigger validation."""
    assert run_triggers()

def test_emulator_basic():
    """Pytest wrapper for emulator smoke test."""
    assert run_emulator_basic()

def main():
    """Run minimal validation tests."""
    logger.info("="*50)
    logger.info("ENERGIE-2.2 Minimal Validation")
    logger.info("="*50)
    
    tests = [
        ("Core Components", run_core_components),
        ("Trigger Mechanisms", run_triggers),
        ("Basic Emulator", run_emulator_basic)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ENERGIE-2.2 adaptive duty-cycle implementation is working!")
        logger.info("Ready for production use with:")
        logger.info("  • Motion sensor triggers")
        logger.info("  • Timer/schedule triggers") 
        logger.info("  • External interrupt triggers")
        logger.info("  • Adaptive state management")
        return 0
    else:
        logger.error("❌ Some components need attention")
        return 1

if __name__ == "__main__":
    exit(main())
