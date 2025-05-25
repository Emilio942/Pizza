"""
External Interrupt Emulation for RP2040-based Pizza Detection System.
Implements GPIO interrupt simulation for adaptive duty-cycle triggering.
"""

import time
import random
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import threading

logger = logging.getLogger(__name__)


class InterruptTrigger(Enum):
    """Types of interrupt triggers"""
    RISING_EDGE = "rising"      # Trigger on rising edge (0 to 1)
    FALLING_EDGE = "falling"    # Trigger on falling edge (1 to 0)
    BOTH_EDGES = "both"         # Trigger on both edges
    LOW_LEVEL = "low"           # Trigger while signal is low
    HIGH_LEVEL = "high"         # Trigger while signal is high


class InterruptPin(Enum):
    """GPIO pins that can generate interrupts"""
    GPIO_0 = 0
    GPIO_1 = 1
    GPIO_2 = 2
    GPIO_3 = 3
    GPIO_4 = 4
    GPIO_5 = 5
    GPIO_6 = 6
    GPIO_7 = 7
    GPIO_8 = 8
    GPIO_9 = 9
    GPIO_10 = 10
    GPIO_11 = 11
    GPIO_12 = 12
    GPIO_13 = 13
    GPIO_14 = 14
    GPIO_15 = 15
    # Additional pins...
    GPIO_20 = 20
    GPIO_21 = 21
    GPIO_22 = 22
    GPIO_26 = 26  # ADC0
    GPIO_27 = 27  # ADC1
    GPIO_28 = 28  # ADC2


@dataclass
class InterruptEvent:
    """Represents an interrupt event"""
    timestamp: float
    pin: InterruptPin
    trigger: InterruptTrigger
    previous_state: bool
    current_state: bool
    trigger_source: str = "gpio_interrupt"
    metadata: Optional[Dict] = None


class GPIOEmulator:
    """
    Emulates a single GPIO pin with interrupt capabilities.
    """
    
    def __init__(self, 
                 pin: InterruptPin,
                 initial_state: bool = False,
                 pull_up: bool = False,
                 pull_down: bool = False):
        """
        Initialize GPIO pin emulator.
        
        Args:
            pin: GPIO pin number
            initial_state: Initial pin state (True=HIGH, False=LOW)
            pull_up: Enable internal pull-up resistor
            pull_down: Enable internal pull-down resistor
        """
        self.pin = pin
        self.state = initial_state
        self.pull_up = pull_up
        self.pull_down = pull_down
        
        # Interrupt configuration
        self.interrupt_enabled = False
        self.interrupt_trigger = InterruptTrigger.RISING_EDGE
        self.interrupt_callback: Optional[Callable[[InterruptEvent], None]] = None
        
        # State tracking
        self.last_change_time = time.time()
        self.state_history = deque(maxlen=100)
        self.interrupt_count = 0
        
        # Debouncing
        self.debounce_time_ms = 0
        self.last_interrupt_time = 0
        
        # Simulation parameters
        self.noise_level = 0.0  # 0.0 to 1.0
        self.bounce_probability = 0.0  # Mechanical bounce simulation
        
        logger.debug(f"GPIO {pin.value} initialized (state={initial_state}, pull_up={pull_up}, pull_down={pull_down})")
    
    def set_state(self, state: bool, external: bool = True) -> None:
        """
        Set GPIO pin state.
        
        Args:
            state: New pin state
            external: Whether this is an external signal change
        """
        previous_state = self.state
        current_time = time.time()
        
        # Apply pull resistors if no external signal
        if not external:
            if self.pull_up and not state:
                state = True
            elif self.pull_down and state:
                state = False
        
        # Apply noise if configured
        if external and self.noise_level > 0:
            if random.random() < self.noise_level:
                state = not state  # Flip state due to noise
        
        # Check for state change
        if state != previous_state:
            self.state = state
            self.last_change_time = current_time
            
            # Store in history
            self.state_history.append({
                'timestamp': current_time,
                'state': state,
                'external': external
            })
            
            # Handle mechanical bounce if configured
            if external and self.bounce_probability > 0:
                if random.random() < self.bounce_probability:
                    # Simulate bounce - rapid state changes
                    bounce_count = random.randint(1, 3)
                    bounce_delay = 0.001  # 1ms bounce delay
                    
                    for i in range(bounce_count):
                        time.sleep(bounce_delay)
                        self.state = not self.state
                        time.sleep(bounce_delay)
                        self.state = state  # Return to desired state
            
            # Check for interrupt trigger
            if self.interrupt_enabled and self._should_trigger_interrupt(previous_state, state):
                self._trigger_interrupt(previous_state, state, current_time)
    
    def get_state(self) -> bool:
        """Get current GPIO pin state"""
        return self.state
    
    def enable_interrupt(self, 
                        trigger: InterruptTrigger,
                        callback: Callable[[InterruptEvent], None],
                        debounce_ms: int = 0) -> None:
        """
        Enable interrupt on this GPIO pin.
        
        Args:
            trigger: Interrupt trigger type
            callback: Function to call when interrupt occurs
            debounce_ms: Debounce time in milliseconds
        """
        self.interrupt_enabled = True
        self.interrupt_trigger = trigger
        self.interrupt_callback = callback
        self.debounce_time_ms = debounce_ms
        
        logger.info(f"GPIO {self.pin.value} interrupt enabled ({trigger.value}, debounce={debounce_ms}ms)")
    
    def disable_interrupt(self) -> None:
        """Disable interrupt on this GPIO pin"""
        self.interrupt_enabled = False
        self.interrupt_callback = None
        logger.debug(f"GPIO {self.pin.value} interrupt disabled")
    
    def set_noise_level(self, level: float) -> None:
        """Set noise level for signal simulation (0.0 to 1.0)"""
        self.noise_level = max(0.0, min(1.0, level))
    
    def set_bounce_probability(self, probability: float) -> None:
        """Set mechanical bounce probability (0.0 to 1.0)"""
        self.bounce_probability = max(0.0, min(1.0, probability))
    
    def _should_trigger_interrupt(self, previous_state: bool, current_state: bool) -> bool:
        """Check if interrupt should be triggered based on state change"""
        current_time = time.time()
        
        # Check debounce time
        if self.debounce_time_ms > 0:
            time_since_last = (current_time - self.last_interrupt_time) * 1000
            if time_since_last < self.debounce_time_ms:
                return False
        
        # Check trigger condition
        if self.interrupt_trigger == InterruptTrigger.RISING_EDGE:
            return not previous_state and current_state
        elif self.interrupt_trigger == InterruptTrigger.FALLING_EDGE:
            return previous_state and not current_state
        elif self.interrupt_trigger == InterruptTrigger.BOTH_EDGES:
            return previous_state != current_state
        elif self.interrupt_trigger == InterruptTrigger.LOW_LEVEL:
            return not current_state
        elif self.interrupt_trigger == InterruptTrigger.HIGH_LEVEL:
            return current_state
        
        return False
    
    def _trigger_interrupt(self, previous_state: bool, current_state: bool, timestamp: float) -> None:
        """Trigger the interrupt callback"""
        if not self.interrupt_callback:
            return
        
        self.interrupt_count += 1
        self.last_interrupt_time = timestamp
        
        event = InterruptEvent(
            timestamp=timestamp,
            pin=self.pin,
            trigger=self.interrupt_trigger,
            previous_state=previous_state,
            current_state=current_state,
            metadata={
                'interrupt_count': self.interrupt_count,
                'time_since_last_change': timestamp - self.last_change_time
            }
        )
        
        try:
            self.interrupt_callback(event)
            logger.debug(f"GPIO {self.pin.value} interrupt triggered ({self.interrupt_trigger.value})")
        except Exception as e:
            logger.error(f"GPIO {self.pin.value} interrupt callback error: {e}")
    
    def get_stats(self) -> Dict:
        """Get GPIO pin statistics"""
        current_time = time.time()
        
        return {
            'pin': self.pin.value,
            'current_state': self.state,
            'interrupt_enabled': self.interrupt_enabled,
            'interrupt_trigger': self.interrupt_trigger.value if self.interrupt_enabled else None,
            'interrupt_count': self.interrupt_count,
            'last_change_time': self.last_change_time,
            'time_since_last_change': current_time - self.last_change_time,
            'debounce_time_ms': self.debounce_time_ms,
            'noise_level': self.noise_level,
            'bounce_probability': self.bounce_probability,
            'pull_up': self.pull_up,
            'pull_down': self.pull_down,
            'state_changes_recent': len([h for h in self.state_history 
                                       if current_time - h['timestamp'] <= 60])
        }


class InterruptController:
    """
    Controller for managing multiple GPIO interrupts.
    Coordinates interrupt handling and provides system-wide interrupt management.
    """
    
    def __init__(self):
        self.gpio_pins: Dict[int, GPIOEmulator] = {}
        self.global_interrupt_enabled = True
        self.interrupt_callbacks: List[Callable[[InterruptEvent], None]] = []
        
        # Interrupt priority system
        self.interrupt_priorities: Dict[int, int] = {}  # pin -> priority (lower = higher)
        self.interrupt_queue = deque(maxlen=100)
        
        # Statistics
        self.total_interrupts = 0
        self.interrupts_by_pin: Dict[int, int] = {}
        
        # Background processing
        self._processing_thread: Optional[threading.Thread] = None
        self._processing_enabled = False
        
        logger.info("Interrupt controller initialized")
    
    def add_gpio_pin(self, 
                    pin: InterruptPin,
                    initial_state: bool = False,
                    pull_up: bool = False,
                    pull_down: bool = False) -> GPIOEmulator:
        """
        Add a GPIO pin to the controller.
        
        Args:
            pin: GPIO pin to add
            initial_state: Initial pin state
            pull_up: Enable pull-up resistor
            pull_down: Enable pull-down resistor
            
        Returns:
            GPIOEmulator instance
        """
        gpio = GPIOEmulator(pin, initial_state, pull_up, pull_down)
        self.gpio_pins[pin.value] = gpio
        self.interrupts_by_pin[pin.value] = 0
        
        logger.info(f"Added GPIO {pin.value} to interrupt controller")
        return gpio
    
    def remove_gpio_pin(self, pin: InterruptPin) -> bool:
        """Remove a GPIO pin from the controller"""
        if pin.value in self.gpio_pins:
            self.gpio_pins[pin.value].disable_interrupt()
            del self.gpio_pins[pin.value]
            del self.interrupts_by_pin[pin.value]
            if pin.value in self.interrupt_priorities:
                del self.interrupt_priorities[pin.value]
            logger.info(f"Removed GPIO {pin.value} from interrupt controller")
            return True
        return False
    
    def get_gpio(self, pin: InterruptPin) -> Optional[GPIOEmulator]:
        """Get GPIO emulator for a specific pin"""
        return self.gpio_pins.get(pin.value)
    
    def enable_global_interrupts(self) -> None:
        """Enable global interrupt processing"""
        self.global_interrupt_enabled = True
        logger.info("Global interrupts enabled")
    
    def disable_global_interrupts(self) -> None:
        """Disable global interrupt processing"""
        self.global_interrupt_enabled = False
        logger.info("Global interrupts disabled")
    
    def set_interrupt_priority(self, pin: InterruptPin, priority: int) -> None:
        """
        Set interrupt priority for a pin.
        
        Args:
            pin: GPIO pin
            priority: Priority level (lower number = higher priority)
        """
        self.interrupt_priorities[pin.value] = priority
        logger.debug(f"Set GPIO {pin.value} interrupt priority to {priority}")
    
    def add_global_interrupt_callback(self, callback: Callable[[InterruptEvent], None]) -> None:
        """Add a global interrupt callback"""
        self.interrupt_callbacks.append(callback)
    
    def simulate_external_signal(self, 
                                pin: InterruptPin,
                                pattern: str,
                                duration_s: float = 1.0,
                                frequency_hz: float = 1.0) -> None:
        """
        Simulate an external signal on a GPIO pin.
        
        Args:
            pin: GPIO pin to simulate signal on
            pattern: Signal pattern ('pulse', 'toggle', 'high', 'low', 'random')
            duration_s: Duration of simulation
            frequency_hz: Frequency for periodic patterns
        """
        gpio = self.gpio_pins.get(pin.value)
        if not gpio:
            logger.warning(f"GPIO {pin.value} not found for signal simulation")
            return
        
        def signal_thread():
            start_time = time.time()
            period = 1.0 / frequency_hz if frequency_hz > 0 else 1.0
            
            while time.time() - start_time < duration_s:
                if pattern == 'pulse':
                    gpio.set_state(True)
                    time.sleep(period * 0.1)  # 10% duty cycle
                    gpio.set_state(False)
                    time.sleep(period * 0.9)
                elif pattern == 'toggle':
                    gpio.set_state(not gpio.get_state())
                    time.sleep(period / 2)
                elif pattern == 'high':
                    gpio.set_state(True)
                    time.sleep(duration_s)
                    break
                elif pattern == 'low':
                    gpio.set_state(False)
                    time.sleep(duration_s)
                    break
                elif pattern == 'random':
                    gpio.set_state(random.choice([True, False]))
                    time.sleep(random.uniform(0.1, period))
                else:
                    break
        
        thread = threading.Thread(target=signal_thread, daemon=True)
        thread.start()
        logger.info(f"Started signal simulation on GPIO {pin.value} ({pattern}, {duration_s}s, {frequency_hz}Hz)")
    
    def create_button_simulation(self, pin: InterruptPin, press_duration_ms: int = 100) -> Callable[[], None]:
        """
        Create a button press simulation function.
        
        Args:
            pin: GPIO pin connected to button
            press_duration_ms: Duration of button press
            
        Returns:
            Function to call to simulate button press
        """
        def press_button():
            gpio = self.gpio_pins.get(pin.value)
            if gpio:
                # Assume button pulls pin low when pressed (with pull-up)
                gpio.set_state(False)  # Press
                time.sleep(press_duration_ms / 1000.0)
                gpio.set_state(True)   # Release
                logger.debug(f"Simulated button press on GPIO {pin.value}")
        
        return press_button
    
    def handle_interrupt_event(self, event: InterruptEvent) -> None:
        """Handle an interrupt event from a GPIO pin"""
        if not self.global_interrupt_enabled:
            return
        
        self.total_interrupts += 1
        self.interrupts_by_pin[event.pin.value] += 1
        
        # Add to interrupt queue with priority
        priority = self.interrupt_priorities.get(event.pin.value, 100)
        self.interrupt_queue.append((priority, event))
        
        # Sort by priority (lower number = higher priority)
        sorted_queue = sorted(self.interrupt_queue, key=lambda x: x[0])
        self.interrupt_queue.clear()
        self.interrupt_queue.extend(sorted_queue)
        
        # Call global callbacks
        for callback in self.interrupt_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Global interrupt callback error: {e}")
    
    def setup_common_interrupts(self) -> Dict[str, Callable]:
        """
        Set up common interrupt sources for pizza detection system.
        
        Returns:
            Dictionary of simulation functions
        """
        # Button for manual trigger
        manual_button_gpio = self.add_gpio_pin(InterruptPin.GPIO_2, initial_state=True, pull_up=True)
        manual_button_gpio.enable_interrupt(
            InterruptTrigger.FALLING_EDGE,
            self.handle_interrupt_event,
            debounce_ms=50
        )
        
        # PIR sensor output
        pir_sensor_gpio = self.add_gpio_pin(InterruptPin.GPIO_3, initial_state=False)
        pir_sensor_gpio.enable_interrupt(
            InterruptTrigger.RISING_EDGE,
            self.handle_interrupt_event,
            debounce_ms=100
        )
        
        # Door sensor (magnetic switch)
        door_sensor_gpio = self.add_gpio_pin(InterruptPin.GPIO_4, initial_state=True, pull_up=True)
        door_sensor_gpio.enable_interrupt(
            InterruptTrigger.BOTH_EDGES,
            self.handle_interrupt_event,
            debounce_ms=200
        )
        
        # Temperature alarm (over-temperature pin)
        temp_alarm_gpio = self.add_gpio_pin(InterruptPin.GPIO_5, initial_state=False)
        temp_alarm_gpio.enable_interrupt(
            InterruptTrigger.RISING_EDGE,
            self.handle_interrupt_event
        )
        
        # Set priorities
        self.set_interrupt_priority(InterruptPin.GPIO_5, 1)  # Temperature alarm (highest)
        self.set_interrupt_priority(InterruptPin.GPIO_2, 2)  # Manual button
        self.set_interrupt_priority(InterruptPin.GPIO_3, 3)  # PIR sensor
        self.set_interrupt_priority(InterruptPin.GPIO_4, 4)  # Door sensor
        
        # Return simulation functions
        return {
            'press_manual_button': self.create_button_simulation(InterruptPin.GPIO_2),
            'trigger_pir_sensor': lambda: pir_sensor_gpio.set_state(True),
            'open_door': lambda: door_sensor_gpio.set_state(False),
            'close_door': lambda: door_sensor_gpio.set_state(True),
            'trigger_temp_alarm': lambda: temp_alarm_gpio.set_state(True),
            'clear_temp_alarm': lambda: temp_alarm_gpio.set_state(False)
        }
    
    def get_controller_stats(self) -> Dict:
        """Get comprehensive controller statistics"""
        gpio_stats = {pin: gpio.get_stats() for pin, gpio in self.gpio_pins.items()}
        
        active_interrupts = sum(1 for gpio in self.gpio_pins.values() if gpio.interrupt_enabled)
        
        return {
            'total_gpio_pins': len(self.gpio_pins),
            'active_interrupts': active_interrupts,
            'global_interrupts_enabled': self.global_interrupt_enabled,
            'total_interrupts': self.total_interrupts,
            'interrupts_by_pin': self.interrupts_by_pin.copy(),
            'interrupt_queue_size': len(self.interrupt_queue),
            'gpio_details': gpio_stats
        }
    
    def get_statistics(self) -> Dict:
        """Get interrupt controller statistics"""
        return self.get_controller_stats()
    
    def trigger_interrupt(self, pin: int, value: bool) -> None:
        """
        Manually trigger an interrupt for testing.
        
        Args:
            pin: GPIO pin number
            value: New pin value
        """
        gpio = self.gpio_pins.get(pin)
        if gpio:
            gpio.set_state(value, external=True)
        else:
            logger.warning(f"Cannot trigger interrupt on non-existent GPIO {pin}")
