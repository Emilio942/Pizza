"""
Enhanced Adaptive Duty-Cycle State Machine for RP2040-based Pizza Detection System.
Implements intelligent wake periods and inference cycles based on multiple trigger sources.
"""

import time
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import deque
import threading

from .motion_sensor import MotionSensor, MotionEvent, MotionSensorController
from .rtc_scheduler import RTCEmulator, ScheduleManager, TimerEvent
from .interrupt_emulator import InterruptController, InterruptEvent, InterruptPin

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System power states"""
    DEEP_SLEEP = auto()      # Lowest power, minimal peripherals active
    LIGHT_SLEEP = auto()     # Medium power, some peripherals active  
    IDLE = auto()            # Active but not processing
    INFERENCE = auto()       # Actively running inference
    MAINTENANCE = auto()     # System maintenance tasks
    EMERGENCY = auto()       # Emergency/critical state


class TriggerType(Enum):
    """Types of wake-up triggers"""
    TIMER = auto()           # RTC timer/scheduled event
    MOTION = auto()          # Motion sensor detection
    INTERRUPT = auto()       # External GPIO interrupt
    TEMPERATURE = auto()     # Temperature threshold
    BATTERY = auto()         # Battery level change
    MANUAL = auto()          # Manual wake-up
    ACTIVITY = auto()        # Detection activity level
    CONTEXT = auto()         # Context-aware triggers


class TriggerPriority(Enum):
    """Priority levels for triggers"""
    CRITICAL = 1    # Emergency situations (temperature, battery critical)
    HIGH = 2        # Important triggers (manual, interrupts)
    MEDIUM = 3      # Normal operation (motion, scheduled)
    LOW = 4         # Background activities


@dataclass
class WakeupTrigger:
    """Represents a wake-up trigger event"""
    trigger_type: TriggerType
    priority: TriggerPriority
    timestamp: float
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    handled: bool = False


@dataclass
class StateTransition:
    """Represents a state transition"""
    from_state: SystemState
    to_state: SystemState
    trigger: WakeupTrigger
    timestamp: float
    duration_s: float = 0.0


class DutyCycleStateMachine:
    """
    Enhanced state machine for adaptive duty-cycle control.
    Intelligently manages system power states based on multiple trigger sources.
    """
    
    def __init__(self, 
                 emulator,
                 power_manager,
                 motion_controller: Optional[MotionSensorController] = None,
                 schedule_manager: Optional[ScheduleManager] = None,
                 interrupt_controller: Optional[InterruptController] = None):
        """
        Initialize adaptive duty-cycle state machine.
        
        Args:
            emulator: RP2040 emulator instance
            power_manager: Power manager instance
            motion_controller: Motion sensor controller
            schedule_manager: Schedule manager for timer events
            interrupt_controller: Interrupt controller for GPIO events
        """
        self.emulator = emulator
        self.power_manager = power_manager
        
        # Trigger controllers
        self.motion_controller = motion_controller or MotionSensorController()
        self.schedule_manager = schedule_manager or ScheduleManager()
        self.interrupt_controller = interrupt_controller or InterruptController()
        
        # State machine
        self.current_state = SystemState.IDLE
        self.previous_state = SystemState.IDLE
        self.state_start_time = time.time()
        
        # Trigger management
        self.trigger_queue = deque(maxlen=100)
        self.trigger_handlers: Dict[TriggerType, List[Callable[[WakeupTrigger], None]]] = {
            trigger_type: [] for trigger_type in TriggerType
        }
        
        # State transition history
        self.transition_history = deque(maxlen=500)
        
        # Configuration
        self.config = {
            # Sleep timeouts (seconds)
            'light_sleep_timeout': 30.0,
            'deep_sleep_timeout': 300.0,
            
            # Inference intervals based on triggers
            'motion_inference_interval': 5.0,
            'scheduled_inference_interval': 30.0,
            'interrupt_inference_interval': 2.0,
            'emergency_inference_interval': 1.0,
            
            # Trigger sensitivity
            'motion_sensitivity': 0.7,
            'temperature_threshold_high': 40.0,
            'temperature_threshold_critical': 50.0,
            'battery_low_threshold': 20.0,
            'battery_critical_threshold': 10.0,
            
            # State machine parameters
            'max_inference_duration': 10.0,
            'maintenance_interval': 3600.0,  # 1 hour
            'emergency_override_enabled': True
        }
        
        # Statistics
        self.stats = {
            'state_changes': 0,
            'triggers_processed': 0,
            'triggers_by_type': {t: 0 for t in TriggerType},
            'time_in_states': {s: 0.0 for s in SystemState},
            'last_reset_time': time.time()
        }
        
        # Runtime state
        self.running = False
        self.main_thread: Optional[threading.Thread] = None
        self.last_maintenance_time = time.time()
        self.inference_start_time = 0.0
        
        # Set up trigger callbacks
        self._setup_trigger_callbacks()
        
        logger.info("Adaptive duty-cycle state machine initialized")
    
    def start(self) -> None:
        """Start the state machine"""
        if not self.running:
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            # Start subsystems
            self.schedule_manager.start()
            
            logger.info("Adaptive duty-cycle state machine started")
    
    def stop(self) -> None:
        """Stop the state machine"""
        if self.running:
            self.running = False
            if self.main_thread:
                self.main_thread.join(timeout=2.0)
            
            # Stop subsystems
            self.schedule_manager.stop()
            
            logger.info("Adaptive duty-cycle state machine stopped")
    
    def _setup_trigger_callbacks(self) -> None:
        """Set up callbacks for various trigger sources"""
        # Motion sensor callbacks
        if self.motion_controller:
            self.motion_controller.add_motion_callback(self._handle_motion_event)
        
        # Timer/schedule callbacks
        if self.schedule_manager:
            self.schedule_manager.add_timer_callback(self._handle_timer_event)
        
        # Interrupt callbacks
        if self.interrupt_controller:
            self.interrupt_controller.add_global_interrupt_callback(self._handle_interrupt_event)
    
    def _handle_motion_event(self, event: MotionEvent) -> None:
        """Handle motion sensor event"""
        trigger = WakeupTrigger(
            trigger_type=TriggerType.MOTION,
            priority=TriggerPriority.MEDIUM,
            timestamp=event.timestamp,
            source_id=event.sensor_id,
            metadata={
                'confidence': event.confidence,
                'sensor_type': 'motion'
            }
        )
        self._add_trigger(trigger)
    
    def _handle_timer_event(self, event: TimerEvent) -> None:
        """Handle timer/scheduled event"""
        trigger = WakeupTrigger(
            trigger_type=TriggerType.TIMER,
            priority=TriggerPriority.MEDIUM,
            timestamp=event.timestamp,
            source_id=event.event_id,
            metadata=event.metadata or {}
        )
        self._add_trigger(trigger)
    
    def _handle_interrupt_event(self, event: InterruptEvent) -> None:
        """Handle GPIO interrupt event"""
        # Determine priority based on pin
        priority = TriggerPriority.HIGH
        if event.pin.value == 2:  # Manual button
            priority = TriggerPriority.HIGH
        elif event.pin.value == 5:  # Temperature alarm
            priority = TriggerPriority.CRITICAL
        
        trigger = WakeupTrigger(
            trigger_type=TriggerType.INTERRUPT,
            priority=priority,
            timestamp=event.timestamp,
            source_id=f"gpio_{event.pin.value}",
            metadata={
                'pin': event.pin.value,
                'trigger': event.trigger.value,
                'previous_state': event.previous_state,
                'current_state': event.current_state
            }
        )
        self._add_trigger(trigger)
    
    def _add_trigger(self, trigger: WakeupTrigger) -> None:
        """Add a trigger to the queue"""
        self.trigger_queue.append(trigger)
        self.stats['triggers_by_type'][trigger.trigger_type] += 1
        
        # Sort queue by priority (critical first)
        sorted_queue = sorted(self.trigger_queue, key=lambda t: t.priority.value)
        self.trigger_queue.clear()
        self.trigger_queue.extend(sorted_queue)
        
        logger.debug(f"Added trigger: {trigger.trigger_type.name} from {trigger.source_id} (priority: {trigger.priority.name})")
    
    def add_manual_trigger(self, source_id: str = "manual", metadata: Optional[Dict] = None) -> None:
        """Add a manual wake-up trigger"""
        trigger = WakeupTrigger(
            trigger_type=TriggerType.MANUAL,
            priority=TriggerPriority.HIGH,
            timestamp=time.time(),
            source_id=source_id,
            metadata=metadata or {}
        )
        self._add_trigger(trigger)
    
    def add_temperature_trigger(self, temperature: float, critical: bool = False) -> None:
        """Add a temperature-based trigger"""
        priority = TriggerPriority.CRITICAL if critical else TriggerPriority.HIGH
        
        trigger = WakeupTrigger(
            trigger_type=TriggerType.TEMPERATURE,
            priority=priority,
            timestamp=time.time(),
            source_id="temperature_sensor",
            metadata={
                'temperature': temperature,
                'critical': critical
            }
        )
        self._add_trigger(trigger)
    
    def add_battery_trigger(self, battery_percent: float, critical: bool = False) -> None:
        """Add a battery level trigger"""
        priority = TriggerPriority.CRITICAL if critical else TriggerPriority.HIGH
        
        trigger = WakeupTrigger(
            trigger_type=TriggerType.BATTERY,
            priority=priority,
            timestamp=time.time(),
            source_id="battery_monitor",
            metadata={
                'battery_percent': battery_percent,
                'critical': critical
            }
        )
        self._add_trigger(trigger)
    
    def add_activity_trigger(self, activity_level: float, detection_changed: bool = True) -> None:
        """Add an activity-based trigger"""
        trigger = WakeupTrigger(
            trigger_type=TriggerType.ACTIVITY,
            priority=TriggerPriority.MEDIUM,
            timestamp=time.time(),
            source_id="activity_monitor",
            metadata={
                'activity_level': activity_level,
                'detection_changed': detection_changed
            }
        )
        self._add_trigger(trigger)
    
    def add_context_trigger(self, context: str, pizza_class: Optional[int] = None) -> None:
        """Add a context-aware trigger"""
        priority = TriggerPriority.HIGH if pizza_class in [1, 2] else TriggerPriority.MEDIUM
        
        trigger = WakeupTrigger(
            trigger_type=TriggerType.CONTEXT,
            priority=priority,
            timestamp=time.time(),
            source_id="context_manager",
            metadata={
                'context': context,
                'pizza_class': pizza_class
            }
        )
        self._add_trigger(trigger)
    
    def _main_loop(self) -> None:
        """Main state machine loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Process triggers
                self._process_triggers()
                
                # Update state based on current conditions
                self._update_state_machine()
                
                # Handle current state
                self._handle_current_state()
                
                # Update statistics
                self._update_statistics(current_time)
                
                # Check for maintenance
                if current_time - self.last_maintenance_time > self.config['maintenance_interval']:
                    self._schedule_maintenance()
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"State machine loop error: {e}")
                time.sleep(1.0)
    
    def _process_triggers(self) -> None:
        """Process pending triggers"""
        while self.trigger_queue:
            trigger = self.trigger_queue.popleft()
            
            if trigger.handled:
                continue
            
            logger.info(f"Processing trigger: {trigger.trigger_type.name} from {trigger.source_id}")
            
            # Handle based on trigger type and current state
            self._handle_trigger(trigger)
            
            trigger.handled = True
            self.stats['triggers_processed'] += 1
    
    def _handle_trigger(self, trigger: WakeupTrigger) -> None:
        """Handle a specific trigger"""
        current_time = time.time()
        
        # Emergency triggers always wake up system
        if trigger.priority == TriggerPriority.CRITICAL:
            if self.current_state in [SystemState.DEEP_SLEEP, SystemState.LIGHT_SLEEP]:
                self._transition_to_state(SystemState.EMERGENCY, trigger)
            return
        
        # Handle based on current state
        if self.current_state == SystemState.DEEP_SLEEP:
            # Wake up from deep sleep for high priority triggers
            if trigger.priority in [TriggerPriority.HIGH, TriggerPriority.MEDIUM]:
                self._transition_to_state(SystemState.IDLE, trigger)
        
        elif self.current_state == SystemState.LIGHT_SLEEP:
            # Wake up from light sleep for any trigger
            self._transition_to_state(SystemState.IDLE, trigger)
        
        elif self.current_state == SystemState.IDLE:
            # Start inference if appropriate trigger
            if self._should_start_inference(trigger):
                self._transition_to_state(SystemState.INFERENCE, trigger)
        
        elif self.current_state == SystemState.INFERENCE:
            # Handle triggers during inference (may extend or interrupt)
            self._handle_inference_trigger(trigger)
        
        # Call trigger-specific handlers
        for handler in self.trigger_handlers.get(trigger.trigger_type, []):
            try:
                handler(trigger)
            except Exception as e:
                logger.error(f"Trigger handler error: {e}")
    
    def _should_start_inference(self, trigger: WakeupTrigger) -> bool:
        """Determine if inference should start based on trigger"""
        if trigger.trigger_type == TriggerType.MOTION:
            return trigger.metadata.get('confidence', 0) > self.config['motion_sensitivity']
        
        elif trigger.trigger_type == TriggerType.TIMER:
            return True  # Scheduled inferences always run
        
        elif trigger.trigger_type == TriggerType.INTERRUPT:
            # Check which pin triggered
            pin = trigger.metadata.get('pin')
            return pin in [2, 3, 4]  # Manual button, PIR, door sensor
        
        elif trigger.trigger_type == TriggerType.MANUAL:
            return True
        
        elif trigger.trigger_type == TriggerType.ACTIVITY:
            return trigger.metadata.get('detection_changed', False)
        
        elif trigger.trigger_type == TriggerType.CONTEXT:
            pizza_class = trigger.metadata.get('pizza_class')
            return pizza_class in [1, 2]  # Critical pizza states
        
        return False
    
    def _handle_inference_trigger(self, trigger: WakeupTrigger) -> None:
        """Handle triggers during inference state"""
        current_time = time.time()
        inference_duration = current_time - self.inference_start_time
        
        # Critical triggers interrupt inference
        if trigger.priority == TriggerPriority.CRITICAL:
            self._transition_to_state(SystemState.EMERGENCY, trigger)
            return
        
        # High priority triggers may extend inference
        if trigger.priority == TriggerPriority.HIGH:
            # Reset inference start time to extend duration
            self.inference_start_time = current_time
            logger.debug(f"Extended inference duration due to {trigger.trigger_type.name}")
    
    def _update_state_machine(self) -> None:
        """Update state machine based on timeouts and conditions"""
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        if self.current_state == SystemState.IDLE:
            # Transition to light sleep after timeout
            if state_duration > self.config['light_sleep_timeout']:
                self._transition_to_state(SystemState.LIGHT_SLEEP)
        
        elif self.current_state == SystemState.LIGHT_SLEEP:
            # Transition to deep sleep after timeout
            if state_duration > self.config['deep_sleep_timeout']:
                self._transition_to_state(SystemState.DEEP_SLEEP)
        
        elif self.current_state == SystemState.INFERENCE:
            # Check for inference timeout
            inference_duration = current_time - self.inference_start_time
            if inference_duration > self.config['max_inference_duration']:
                self._transition_to_state(SystemState.IDLE)
                logger.warning("Inference timed out, returning to idle")
        
        elif self.current_state == SystemState.EMERGENCY:
            # Return to normal operation after handling emergency
            if not self._has_critical_conditions():
                self._transition_to_state(SystemState.IDLE)
    
    def _has_critical_conditions(self) -> bool:
        """Check if critical conditions still exist"""
        # Check temperature
        current_temp = self.power_manager.current_temperature_c
        if current_temp > self.config['temperature_threshold_critical']:
            return True
        
        # Check battery
        battery_stats = self.power_manager.get_power_statistics()
        battery_percent = battery_stats.get('battery_percent', 100)
        if battery_percent < self.config['battery_critical_threshold']:
            return True
        
        return False
    
    def _transition_to_state(self, new_state: SystemState, trigger: Optional[WakeupTrigger] = None) -> None:
        """Transition to a new system state"""
        if new_state == self.current_state:
            return
        
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        # Create transition record
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            timestamp=current_time,
            duration_s=state_duration
        )
        
        # Update statistics
        self.stats['time_in_states'][self.current_state] += state_duration
        self.stats['state_changes'] += 1
        
        # Perform state exit actions
        self._exit_state(self.current_state)
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = current_time
        
        # Perform state entry actions
        self._enter_state(new_state)
        
        # Store transition
        self.transition_history.append(transition)
        
        trigger_info = f" (trigger: {trigger.trigger_type.name})" if trigger else ""
        logger.info(f"State transition: {self.previous_state.name} -> {new_state.name}{trigger_info}")
    
    def _enter_state(self, state: SystemState) -> None:
        """Perform actions when entering a state"""
        if state == SystemState.DEEP_SLEEP:
            # Enter deep sleep mode
            self.power_manager.enter_sleep_mode()
            logger.debug("Entered deep sleep mode")
        
        elif state == SystemState.LIGHT_SLEEP:
            # Enter light sleep mode (some peripherals remain active)
            self.power_manager.enter_sleep_mode()
            logger.debug("Entered light sleep mode")
        
        elif state == SystemState.INFERENCE:
            # Start inference
            self.inference_start_time = time.time()
            if hasattr(self.power_manager, 'wake_up'):
                self.power_manager.wake_up()
            logger.debug("Started inference")
        
        elif state == SystemState.EMERGENCY:
            # Wake up immediately and prepare for emergency handling
            if hasattr(self.power_manager, 'wake_up'):
                self.power_manager.wake_up()
            logger.warning("Entered emergency state")
        
        elif state == SystemState.IDLE:
            # Ensure system is awake
            if hasattr(self.power_manager, 'wake_up'):
                self.power_manager.wake_up()
            logger.debug("Entered idle state")
    
    def _exit_state(self, state: SystemState) -> None:
        """Perform actions when exiting a state"""
        if state in [SystemState.DEEP_SLEEP, SystemState.LIGHT_SLEEP]:
            # Wake up from sleep
            if hasattr(self.power_manager, 'wake_up'):
                self.power_manager.wake_up()
    
    def _handle_current_state(self) -> None:
        """Handle ongoing actions for current state"""
        current_time = time.time()
        
        if self.current_state == SystemState.INFERENCE:
            # Simulate inference processing
            # In real implementation, this would trigger actual pizza detection
            pass
        
        elif self.current_state == SystemState.MAINTENANCE:
            # Perform maintenance tasks
            self._perform_maintenance()
        
        elif self.current_state == SystemState.EMERGENCY:
            # Handle emergency conditions
            self._handle_emergency()
    
    def _perform_maintenance(self) -> None:
        """Perform system maintenance tasks"""
        logger.info("Performing system maintenance")
        
        # Clean up old history data
        cutoff_time = time.time() - 86400  # 24 hours
        self.transition_history = deque(
            [t for t in self.transition_history if t.timestamp > cutoff_time],
            maxlen=500
        )
        
        # Update maintenance time
        self.last_maintenance_time = time.time()
        
        # Return to idle state
        self._transition_to_state(SystemState.IDLE)
    
    def _schedule_maintenance(self) -> None:
        """Schedule maintenance if not already in progress"""
        if self.current_state != SystemState.MAINTENANCE:
            self._transition_to_state(SystemState.MAINTENANCE)
    
    def _handle_emergency(self) -> None:
        """Handle emergency state"""
        current_temp = self.power_manager.current_temperature_c
        
        # Handle high temperature
        if current_temp > self.config['temperature_threshold_critical']:
            logger.critical(f"Critical temperature: {current_temp:.1f}°C")
            # In real implementation: reduce clock speed, increase cooling, etc.
        
        # Handle low battery
        battery_stats = self.power_manager.get_power_statistics()
        battery_percent = battery_stats.get('battery_percent', 100)
        if battery_percent < self.config['battery_critical_threshold']:
            logger.critical(f"Critical battery level: {battery_percent:.1f}%")
            # In real implementation: reduce power consumption, save state, etc.
    
    def _update_statistics(self, current_time: float) -> None:
        """Update runtime statistics"""
        # Update current state time
        state_duration = current_time - self.state_start_time
        # Note: This is ongoing time, not added to total until state transition
    
    def add_trigger_handler(self, trigger_type: TriggerType, handler: Callable[[WakeupTrigger], None]) -> None:
        """Add a custom trigger handler"""
        self.trigger_handlers[trigger_type].append(handler)
        logger.debug(f"Added handler for {trigger_type.name} triggers")
    
    def get_current_state(self) -> SystemState:
        """Get current system state"""
        return self.current_state
    
    def get_state_duration(self) -> float:
        """Get time spent in current state"""
        return time.time() - self.state_start_time
    
    def get_recent_transitions(self, count: int = 10) -> List[StateTransition]:
        """Get recent state transitions"""
        return list(self.transition_history)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive state machine statistics"""
        current_time = time.time()
        uptime = current_time - self.stats['last_reset_time']
        
        # Calculate current state time
        current_state_time = current_time - self.state_start_time
        total_time_in_states = self.stats['time_in_states'].copy()
        total_time_in_states[self.current_state] += current_state_time
        
        # Calculate state percentages
        state_percentages = {
            state.name: (time_spent / uptime * 100) if uptime > 0 else 0
            for state, time_spent in total_time_in_states.items()
        }
        
        return {
            'current_state': self.current_state.name,
            'current_state_duration_s': current_state_time,
            'uptime_s': uptime,
            'total_state_changes': self.stats['state_changes'],
            'total_triggers_processed': self.stats['triggers_processed'],
            'triggers_by_type': {t.name: count for t, count in self.stats['triggers_by_type'].items()},
            'time_in_states_s': {s.name: t for s, t in total_time_in_states.items()},
            'state_percentages': state_percentages,
            'pending_triggers': len(self.trigger_queue),
            'recent_transitions': len(self.transition_history)
        }
