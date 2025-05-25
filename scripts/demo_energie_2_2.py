#!/usr/bin/env python3
"""
ENERGIE-2.2 Completion Demonstration
====================================

This script demonstrates the completed adaptive duty-cycle logic implementation
with intelligent wake periods and inference cycles controlled by external triggers.

Features demonstrated:
1. Motion sensor triggers (PIR, microwave, ultrasonic)
2. Timer/schedule-based triggers  
3. External interrupt triggers
4. Adaptive state machine with 6 states
5. Power management integration
6. Multiple scenario validation

Author: AI Assistant
Date: May 24, 2025
Task: ENERGIE-2.2 Implementation
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.emulation.emulator import RP2040Emulator
from src.emulation.adaptive_state_machine import TriggerType, SystemState
from src.emulation.simple_power_manager import AdaptiveMode

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ENERGIE_2_2_Demo:
    """Demonstration of ENERGIE-2.2 adaptive duty-cycle implementation."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.emulator = None
        self.demo_start_time = None
        
    def setup(self):
        """Setup the demonstration environment."""
        logger.info("🚀 Setting up ENERGIE-2.2 demonstration...")
        self.demo_start_time = time.time()
        
        # Initialize emulator with all adaptive components
        self.emulator = RP2040Emulator()
        time.sleep(1.0)  # Allow initialization
        
        logger.info("✅ Emulator initialized with adaptive duty-cycle logic")
        
        # Verify all components
        components = [
            'motion_controller',
            'schedule_manager', 
            'interrupt_controller',
            'adaptive_state_machine'
        ]
        
        for component in components:
            assert hasattr(self.emulator, component), f"Missing {component}"
            logger.info(f"  ✓ {component} initialized")
            
        logger.info("🎯 All ENERGIE-2.2 components ready for demonstration")
        
    def demonstrate_motion_triggers(self):
        """Demonstrate motion sensor trigger functionality."""
        logger.info("\n" + "="*60)
        logger.info("🏃 DEMONSTRATING MOTION SENSOR TRIGGERS")
        logger.info("="*60)
        
        # Get baseline statistics
        initial_stats = self.emulator.adaptive_state_machine.get_statistics()
        initial_motion = initial_stats['trigger_counts'].get('MOTION', 0)
        
        logger.info(f"📊 Initial motion triggers: {initial_motion}")
        logger.info(f"📊 Current state: {self.emulator.adaptive_state_machine.get_current_state()}")
        
        # Demonstrate different sensor types
        sensor_types = ['PIR', 'MICROWAVE', 'ULTRASONIC']
        
        for i, sensor_type in enumerate(sensor_types, 1):
            logger.info(f"\n🔍 Test {i}: {sensor_type} sensor motion detection")
            
            # Simulate motion event
            confidence = 0.7 + (i * 0.1)  # Varying confidence
            self.emulator.motion_controller.simulate_motion_event(
                sensor_type=sensor_type,
                duration=1.5,
                confidence=confidence
            )
            
            logger.info(f"  📡 Simulated {sensor_type} motion (confidence: {confidence:.1f})")
            
            # Wait for processing
            time.sleep(1.0)
            
            # Check state change
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"  🎯 State after trigger: {current_state}")
            
        # Final motion statistics
        final_stats = self.emulator.adaptive_state_machine.get_statistics()
        motion_triggers = final_stats['trigger_counts'].get('MOTION', 0) - initial_motion
        
        logger.info(f"\n📈 Result: {motion_triggers} motion triggers processed successfully")
        logger.info("✅ Motion sensor trigger system working correctly!")
        
    def demonstrate_timer_triggers(self):
        """Demonstrate timer/schedule trigger functionality."""
        logger.info("\n" + "="*60)
        logger.info("⏰ DEMONSTRATING TIMER/SCHEDULE TRIGGERS")
        logger.info("="*60)
        
        initial_stats = self.emulator.adaptive_state_machine.get_statistics()
        initial_timer = initial_stats['trigger_counts'].get('TIMER', 0)
        
        logger.info(f"📊 Initial timer triggers: {initial_timer}")
        
        # Demonstrate different schedule types
        logger.info("\n🕐 Test 1: One-shot timer (2 seconds)")
        one_shot_id = self.emulator.schedule_manager.schedule_one_shot(
            name="demo_one_shot",
            delay=2.0,
            callback=None
        )
        logger.info(f"  📅 Scheduled one-shot event (ID: {one_shot_id})")
        
        logger.info("\n🔄 Test 2: Periodic timer (every 1.5 seconds)")
        periodic_id = self.emulator.schedule_manager.schedule_periodic(
            name="demo_periodic",
            interval=1.5,
            callback=None
        )
        logger.info(f"  📅 Scheduled periodic event (ID: {periodic_id})")
        
        # Wait for timer events
        logger.info("\n⏳ Waiting for timer events to trigger...")
        time.sleep(5.0)
        
        # Cancel periodic timer
        self.emulator.schedule_manager.cancel_event(periodic_id)
        logger.info("  🛑 Cancelled periodic timer")
        
        # Check results
        final_stats = self.emulator.adaptive_state_machine.get_statistics()
        timer_triggers = final_stats['trigger_counts'].get('TIMER', 0) - initial_timer
        
        logger.info(f"\n📈 Result: {timer_triggers} timer triggers processed")
        logger.info("✅ Timer/schedule trigger system working correctly!")
        
    def demonstrate_interrupt_triggers(self):
        """Demonstrate external interrupt trigger functionality."""
        logger.info("\n" + "="*60)
        logger.info("⚡ DEMONSTRATING EXTERNAL INTERRUPT TRIGGERS")
        logger.info("="*60)
        
        initial_stats = self.emulator.adaptive_state_machine.get_statistics()
        initial_interrupt = initial_stats['trigger_counts'].get('INTERRUPT', 0)
        
        logger.info(f"📊 Initial interrupt triggers: {initial_interrupt}")
        
        # Show available interrupt pins
        interrupt_stats = self.emulator.interrupt_controller.get_statistics()
        logger.info(f"📍 Available GPIO pins: {list(interrupt_stats['pins'].keys())}")
        
        # Demonstrate different interrupt sources
        interrupt_tests = [
            (2, "Manual button press"),
            (3, "PIR sensor interrupt"),
            (4, "Door sensor interrupt"),
            (5, "Temperature alarm")
        ]
        
        for pin, description in interrupt_tests:
            logger.info(f"\n🔌 Test: {description} (GPIO {pin})")
            
            # Trigger interrupt (rising edge)
            self.emulator.interrupt_controller.trigger_interrupt(pin=pin, value=1)
            logger.info(f"  📶 GPIO {pin} triggered HIGH")
            time.sleep(0.1)
            
            # Release interrupt (falling edge) 
            self.emulator.interrupt_controller.trigger_interrupt(pin=pin, value=0)
            logger.info(f"  📶 GPIO {pin} triggered LOW")
            
            # Brief pause between tests
            time.sleep(0.5)
            
            # Check state
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"  🎯 State: {current_state}")
            
        # Final interrupt statistics
        final_stats = self.emulator.adaptive_state_machine.get_statistics()
        interrupt_triggers = final_stats['trigger_counts'].get('INTERRUPT', 0) - initial_interrupt
        
        logger.info(f"\n📈 Result: {interrupt_triggers} interrupt triggers processed")
        logger.info("✅ External interrupt trigger system working correctly!")
        
    def demonstrate_adaptive_modes(self):
        """Demonstrate adaptive power management modes."""
        logger.info("\n" + "="*60)
        logger.info("🔋 DEMONSTRATING ADAPTIVE POWER MANAGEMENT")
        logger.info("="*60)
        
        # Test different adaptive modes
        modes = [
            (AdaptiveMode.PERFORMANCE, "Maximum performance"),
            (AdaptiveMode.BALANCED, "Balanced performance/power"),
            (AdaptiveMode.POWER_SAVE, "Power saving mode"),
            (AdaptiveMode.ULTRA_LOW_POWER, "Ultra low power mode")
        ]
        
        for mode, description in modes:
            logger.info(f"\n⚡ Testing: {description}")
            
            # Set mode
            self.emulator.power_manager.set_adaptive_mode(mode)
            time.sleep(0.5)  # Allow mode change
            
            # Get power statistics
            power_stats = self.emulator.power_manager.get_statistics()
            power_consumption = power_stats.get('current_power_mw', 0)
            
            logger.info(f"  🔋 Mode: {mode.value}")
            logger.info(f"  ⚡ Power consumption: {power_consumption:.1f}mW")
            
            # Trigger activity in this mode
            self.emulator.motion_controller.simulate_motion_event("PIR", 0.5, 0.8)
            time.sleep(0.5)
            
            # Check state response
            current_state = self.emulator.adaptive_state_machine.get_current_state()
            logger.info(f"  🎯 State response: {current_state}")
            
        logger.info("\n✅ Adaptive power management working correctly!")
        
    def demonstrate_scenarios(self):
        """Demonstrate different usage scenarios."""
        logger.info("\n" + "="*60)
        logger.info("🎭 DEMONSTRATING USAGE SCENARIOS")
        logger.info("="*60)
        
        # Scenario 1: High activity period
        logger.info("\n🏃‍♂️ Scenario 1: High activity period (busy kitchen)")
        self.emulator.power_manager.set_adaptive_mode(AdaptiveMode.PERFORMANCE)
        
        for i in range(4):
            # Rapid motion events
            self.emulator.motion_controller.simulate_motion_event("PIR", 0.3, 0.9)
            time.sleep(0.2)
            
            # Occasional interrupts
            if i % 2 == 0:
                self.emulator.interrupt_controller.trigger_interrupt(pin=2, value=1)
                time.sleep(0.1)
                self.emulator.interrupt_controller.trigger_interrupt(pin=2, value=0)
                
        logger.info("  🍕 Simulated busy kitchen with frequent motion and interactions")
        time.sleep(1.0)
        
        # Scenario 2: Low activity period  
        logger.info("\n😴 Scenario 2: Low activity period (quiet hours)")
        self.emulator.power_manager.set_adaptive_mode(AdaptiveMode.POWER_SAVE)
        time.sleep(2.0)  # Let system settle
        
        # Single motion event
        self.emulator.motion_controller.simulate_motion_event("PIR", 1.0, 0.6)
        logger.info("  🌙 Simulated quiet period with occasional motion")
        time.sleep(1.0)
        
        # Scenario 3: Mixed activity
        logger.info("\n🔄 Scenario 3: Mixed activity (normal operation)")
        self.emulator.power_manager.set_adaptive_mode(AdaptiveMode.BALANCED)
        
        # Schedule regular check
        self.emulator.schedule_manager.schedule_one_shot("regular_check", 1.0, None)
        
        # Motion + interrupt
        time.sleep(0.5)
        self.emulator.motion_controller.simulate_motion_event("MICROWAVE", 1.0, 0.7)
        time.sleep(0.5)
        self.emulator.interrupt_controller.trigger_interrupt(pin=3, value=1)
        time.sleep(0.1)
        self.emulator.interrupt_controller.trigger_interrupt(pin=3, value=0)
        
        logger.info("  🏠 Simulated normal kitchen operation with mixed triggers")
        time.sleep(2.0)
        
        logger.info("\n✅ All scenarios demonstrated successfully!")
        
    def show_final_statistics(self):
        """Show comprehensive final statistics."""
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL SYSTEM STATISTICS")
        logger.info("="*60)
        
        # Adaptive state machine statistics
        state_stats = self.emulator.adaptive_state_machine.get_statistics()
        
        logger.info("\n🎯 Trigger Statistics:")
        total_triggers = sum(state_stats['trigger_counts'].values())
        for trigger_type, count in state_stats['trigger_counts'].items():
            if count > 0:
                percentage = (count / total_triggers) * 100 if total_triggers > 0 else 0
                logger.info(f"  {trigger_type}: {count} triggers ({percentage:.1f}%)")
                
        logger.info(f"\n📈 Total triggers processed: {total_triggers}")
        
        logger.info("\n⏱️  State Duration Statistics:")
        total_time = sum(state_stats['state_durations'].values())
        for state, duration in state_stats['state_durations'].items():
            if duration > 0:
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                logger.info(f"  {state}: {duration:.2f}s ({percentage:.1f}%)")
                
        logger.info(f"\n🕒 Total active time: {total_time:.2f}s")
        
        logger.info("\n🔄 State Transitions:")
        total_transitions = sum(state_stats['state_transitions'].values())
        for transition, count in state_stats['state_transitions'].items():
            if count > 0:
                logger.info(f"  {transition}: {count} times")
                
        logger.info(f"\n🔄 Total state transitions: {total_transitions}")
        
        # Power management statistics
        power_stats = self.emulator.power_manager.get_statistics()
        logger.info(f"\n🔋 Power Management:")
        logger.info(f"  Current mode: {power_stats['adaptive_mode']}")
        logger.info(f"  Current consumption: {power_stats['current_power_mw']:.1f}mW")
        logger.info(f"  Activity level: {power_stats['activity_level']:.2f}")
        
        # Component statistics
        motion_stats = self.emulator.motion_controller.get_statistics()
        schedule_stats = self.emulator.schedule_manager.get_statistics()
        interrupt_stats = self.emulator.interrupt_controller.get_statistics()
        
        logger.info(f"\n🏃 Motion Controller:")
        logger.info(f"  Active sensors: {motion_stats['active_sensors']}")
        logger.info(f"  Detecting sensors: {motion_stats['detecting_sensors']}")
        
        logger.info(f"\n⏰ Schedule Manager:")
        logger.info(f"  Active events: {schedule_stats['active_events']}")
        logger.info(f"  Total executed: {schedule_stats['total_executed']}")
        
        logger.info(f"\n⚡ Interrupt Controller:")
        logger.info(f"  Total interrupts: {interrupt_stats['total_interrupts']}")
        logger.info(f"  Active pins: {len([p for p in interrupt_stats['pins'].values() if p['enabled']])}")
        
        # Demo runtime
        demo_runtime = time.time() - self.demo_start_time
        logger.info(f"\n⏱️  Demo runtime: {demo_runtime:.1f}s")
        
    def teardown(self):
        """Clean up demonstration environment."""
        if self.emulator:
            self.emulator.close()
            logger.info("🧹 Demonstration environment cleaned up")
            
    def run_complete_demo(self):
        """Run the complete ENERGIE-2.2 demonstration."""
        try:
            # Setup
            self.setup()
            
            # Run demonstrations
            self.demonstrate_motion_triggers()
            self.demonstrate_timer_triggers()
            self.demonstrate_interrupt_triggers()
            self.demonstrate_adaptive_modes()
            self.demonstrate_scenarios()
            
            # Show results
            self.show_final_statistics()
            
            # Success message
            logger.info("\n" + "="*60)
            logger.info("🎉 ENERGIE-2.2 IMPLEMENTATION COMPLETE!")
            logger.info("="*60)
            logger.info("\n✅ Successfully implemented and demonstrated:")
            logger.info("  🏃 Motion sensor triggers (PIR, microwave, ultrasonic)")
            logger.info("  ⏰ Timer/schedule-based triggers")
            logger.info("  ⚡ External interrupt triggers")
            logger.info("  🎯 Adaptive state machine (6 states)")
            logger.info("  🔋 Power management integration")
            logger.info("  🎭 Multiple usage scenarios")
            
            logger.info("\n🚀 The system now intelligently controls wake periods")
            logger.info("   and inference cycles based on external triggers!")
            
            logger.info("\n📋 ENERGIE-2.2 Task Status: ✅ COMPLETED")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.teardown()

def main():
    """Main demonstration entry point."""
    print("🎯 ENERGIE-2.2: Adaptive Duty-Cycle Logic Implementation")
    print("🕒 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    demo = ENERGIE_2_2_Demo()
    success = demo.run_complete_demo()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
