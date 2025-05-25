"""
Real-Time Clock (RTC) and Scheduling Emulation for RP2040-based Pizza Detection System.
Implements timer-based triggering for adaptive duty-cycle control.
"""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import deque

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduled events"""
    PERIODIC = "periodic"          # Recurring at fixed intervals
    CRON_LIKE = "cron_like"       # Cron-style scheduling
    ONE_SHOT = "one_shot"         # Single execution
    CONDITIONAL = "conditional"    # Based on conditions


@dataclass
class ScheduledEvent:
    """Represents a scheduled event"""
    event_id: str
    schedule_type: ScheduleType
    next_execution: float
    callback: Callable
    enabled: bool = True
    
    # For periodic events
    interval_s: Optional[float] = None
    
    # For cron-like events
    hour: Optional[int] = None
    minute: Optional[int] = None
    weekday: Optional[int] = None  # 0=Monday, 6=Sunday
    
    # For conditional events
    condition_func: Optional[Callable[[], bool]] = None
    
    # Event metadata
    execution_count: int = 0
    last_execution: Optional[float] = None
    max_executions: Optional[int] = None


@dataclass
class TimerEvent:
    """Represents a timer event for triggering"""
    timestamp: float
    event_id: str
    trigger_source: str = "timer"
    metadata: Optional[Dict] = None


class RTCEmulator:
    """
    Emulates a Real-Time Clock for RP2040-based systems.
    Provides accurate timing and scheduling capabilities.
    """
    
    def __init__(self, sync_with_system: bool = True):
        """
        Initialize RTC emulator.
        
        Args:
            sync_with_system: Whether to sync with system time
        """
        self.sync_with_system = sync_with_system
        self.base_time = time.time() if sync_with_system else 0
        self.offset = 0.0
        self.drift_rate = 0.0  # PPM drift simulation
        self.initialized = False
        
        # RTC state
        self.last_sync_time = self.base_time
        self.total_drift = 0.0
        
        logger.info(f"RTC emulator initialized (sync_with_system={sync_with_system})")
    
    def initialize(self) -> bool:
        """Initialize the RTC"""
        try:
            self.initialized = True
            self.last_sync_time = time.time()
            logger.info("RTC emulator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"RTC initialization failed: {e}")
            return False
    
    def get_time(self) -> float:
        """
        Get current RTC time.
        
        Returns:
            Current time as Unix timestamp
        """
        if not self.initialized:
            return time.time()
        
        if self.sync_with_system:
            current_system_time = time.time()
            # Simulate clock drift
            elapsed = current_system_time - self.last_sync_time
            drift = elapsed * self.drift_rate / 1_000_000  # Convert PPM to seconds
            self.total_drift += drift
            self.last_sync_time = current_system_time
            
            return current_system_time + self.offset + self.total_drift
        else:
            return self.base_time + self.offset
    
    def set_time(self, timestamp: float) -> None:
        """Set RTC time"""
        if self.sync_with_system:
            self.offset = timestamp - time.time()
        else:
            self.base_time = timestamp
        logger.debug(f"RTC time set to {datetime.fromtimestamp(timestamp)}")
    
    def set_drift_rate(self, ppm: float) -> None:
        """
        Set clock drift rate in parts per million.
        
        Args:
            ppm: Drift rate in PPM (positive = fast, negative = slow)
        """
        self.drift_rate = ppm
        logger.debug(f"RTC drift rate set to {ppm} PPM")
    
    def get_datetime(self) -> datetime:
        """Get current time as datetime object"""
        return datetime.fromtimestamp(self.get_time())


class ScheduleManager:
    """
    Manages scheduled events and timer-based triggers.
    Provides comprehensive scheduling capabilities for adaptive duty-cycle control.
    """
    
    def __init__(self, rtc: Optional[RTCEmulator] = None):
        """
        Initialize schedule manager.
        
        Args:
            rtc: RTC emulator instance (creates new one if None)
        """
        self.rtc = rtc or RTCEmulator()
        self.events: Dict[str, ScheduledEvent] = {}
        self.event_history = deque(maxlen=1000)
        self.timer_callbacks: List[Callable[[TimerEvent], None]] = []
        
        # Background execution
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._check_interval = 1.0  # Check every second
        
        # Statistics
        self.total_executions = 0
        self.missed_executions = 0
        
        logger.info("Schedule manager initialized")
    
    def start(self) -> None:
        """Start the schedule manager background thread"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._thread.start()
            logger.info("Schedule manager started")
    
    def stop(self) -> None:
        """Stop the schedule manager background thread"""
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join(timeout=2.0)
            logger.info("Schedule manager stopped")
    
    def is_running(self) -> bool:
        """Check if schedule manager is running"""
        return self._running
    
    def add_periodic_event(self, 
                          event_id: str, 
                          interval_s: float, 
                          callback: Callable,
                          start_delay_s: float = 0.0,
                          max_executions: Optional[int] = None) -> bool:
        """
        Add a periodic event.
        
        Args:
            event_id: Unique identifier for the event
            interval_s: Interval between executions in seconds
            callback: Function to call when event triggers
            start_delay_s: Delay before first execution
            max_executions: Maximum number of executions (None for unlimited)
            
        Returns:
            True if event was added successfully
        """
        if event_id in self.events:
            logger.warning(f"Event {event_id} already exists")
            return False
        
        next_exec = self.rtc.get_time() + start_delay_s
        
        event = ScheduledEvent(
            event_id=event_id,
            schedule_type=ScheduleType.PERIODIC,
            next_execution=next_exec,
            callback=callback,
            interval_s=interval_s,
            max_executions=max_executions
        )
        
        self.events[event_id] = event
        logger.info(f"Added periodic event {event_id} (interval: {interval_s}s)")
        return True
    
    def add_cron_event(self, 
                      event_id: str,
                      callback: Callable,
                      hour: Optional[int] = None,
                      minute: Optional[int] = None,
                      weekday: Optional[int] = None) -> bool:
        """
        Add a cron-style scheduled event.
        
        Args:
            event_id: Unique identifier for the event
            callback: Function to call when event triggers
            hour: Hour to trigger (0-23, None for any)
            minute: Minute to trigger (0-59, None for any)
            weekday: Day of week (0=Monday, 6=Sunday, None for any)
            
        Returns:
            True if event was added successfully
        """
        if event_id in self.events:
            logger.warning(f"Event {event_id} already exists")
            return False
        
        next_exec = self._calculate_next_cron_execution(hour, minute, weekday)
        
        event = ScheduledEvent(
            event_id=event_id,
            schedule_type=ScheduleType.CRON_LIKE,
            next_execution=next_exec,
            callback=callback,
            hour=hour,
            minute=minute,
            weekday=weekday
        )
        
        self.events[event_id] = event
        logger.info(f"Added cron event {event_id} (H:{hour} M:{minute} WD:{weekday})")
        return True
    
    def add_one_shot_event(self, 
                          event_id: str,
                          delay_s: float,
                          callback: Callable) -> bool:
        """
        Add a one-shot event.
        
        Args:
            event_id: Unique identifier for the event
            delay_s: Delay before execution
            callback: Function to call when event triggers
            
        Returns:
            True if event was added successfully
        """
        if event_id in self.events:
            logger.warning(f"Event {event_id} already exists")
            return False
        
        next_exec = self.rtc.get_time() + delay_s
        
        event = ScheduledEvent(
            event_id=event_id,
            schedule_type=ScheduleType.ONE_SHOT,
            next_execution=next_exec,
            callback=callback,
            max_executions=1
        )
        
        self.events[event_id] = event
        logger.info(f"Added one-shot event {event_id} (delay: {delay_s}s)")
        return True
    
    def add_conditional_event(self,
                             event_id: str,
                             condition_func: Callable[[], bool],
                             callback: Callable,
                             check_interval_s: float = 10.0) -> bool:
        """
        Add a conditional event that triggers when a condition is met.
        
        Args:
            event_id: Unique identifier for the event
            condition_func: Function that returns True when event should trigger
            callback: Function to call when event triggers
            check_interval_s: How often to check the condition
            
        Returns:
            True if event was added successfully
        """
        if event_id in self.events:
            logger.warning(f"Event {event_id} already exists")
            return False
        
        next_exec = self.rtc.get_time() + check_interval_s
        
        event = ScheduledEvent(
            event_id=event_id,
            schedule_type=ScheduleType.CONDITIONAL,
            next_execution=next_exec,
            callback=callback,
            condition_func=condition_func,
            interval_s=check_interval_s
        )
        
        self.events[event_id] = event
        logger.info(f"Added conditional event {event_id} (check interval: {check_interval_s}s)")
        return True
    
    def remove_event(self, event_id: str) -> bool:
        """Remove a scheduled event"""
        if event_id in self.events:
            del self.events[event_id]
            logger.info(f"Removed event {event_id}")
            return True
        return False
    
    def enable_event(self, event_id: str) -> bool:
        """Enable a scheduled event"""
        if event_id in self.events:
            self.events[event_id].enabled = True
            logger.debug(f"Enabled event {event_id}")
            return True
        return False
    
    def disable_event(self, event_id: str) -> bool:
        """Disable a scheduled event"""
        if event_id in self.events:
            self.events[event_id].enabled = False
            logger.debug(f"Disabled event {event_id}")
            return True
        return False
    
    def add_timer_callback(self, callback: Callable[[TimerEvent], None]) -> None:
        """Add a callback to be called when any timer event triggers"""
        self.timer_callbacks.append(callback)
    
    def schedule_periodic(self, name: str, interval: float, callback: Optional[Callable] = None) -> str:
        """
        Convenience method for scheduling periodic events.
        
        Args:
            name: Name of the event
            interval: Interval in seconds
            callback: Optional callback (uses timer callbacks if None)
            
        Returns:
            Event ID
        """
        event_id = f"periodic_{name}_{int(time.time())}"
        self.add_periodic_event(
            event_id=event_id,
            interval_s=interval,
            callback=callback or (lambda: self._trigger_timer_callbacks(event_id))
        )
        return event_id
    
    def schedule_one_shot(self, name: str, delay: float, callback: Optional[Callable] = None) -> str:
        """
        Convenience method for scheduling one-shot events.
        
        Args:
            name: Name of the event
            delay: Delay in seconds
            callback: Optional callback (uses timer callbacks if None)
            
        Returns:
            Event ID
        """
        event_id = f"oneshot_{name}_{int(time.time())}"
        self.add_one_shot_event(
            event_id=event_id,
            delay_s=delay,
            callback=callback or (lambda: self._trigger_timer_callbacks(event_id))
        )
        return event_id
    
    def cancel_event(self, event_id: str) -> bool:
        """
        Cancel a scheduled event.
        
        Args:
            event_id: ID of event to cancel
            
        Returns:
            True if event was found and cancelled
        """
        return self.remove_event(event_id)
    
    def _trigger_timer_callbacks(self, event_id: str) -> None:
        """Trigger timer callbacks for an event"""
        event = TimerEvent(
            timestamp=time.time(),
            event_id=event_id,
            trigger_source="scheduled_timer"
        )
        
        for callback in self.timer_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Timer callback error: {e}")
    
    def get_statistics(self) -> Dict:
        """Get scheduler statistics"""
        return self.get_scheduler_stats()
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop running in background thread"""
        while self._running:
            try:
                current_time = self.rtc.get_time()
                events_to_execute = []
                
                # Check all events
                for event_id, event in self.events.items():
                    if not event.enabled:
                        continue
                    
                    if self._should_execute_event(event, current_time):
                        events_to_execute.append(event)
                
                # Execute events
                for event in events_to_execute:
                    self._execute_event(event, current_time)
                
                # Sleep until next check
                time.sleep(self._check_interval)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(self._check_interval)
    
    def _should_execute_event(self, event: ScheduledEvent, current_time: float) -> bool:
        """Check if an event should be executed"""
        if current_time < event.next_execution:
            return False
        
        if event.max_executions and event.execution_count >= event.max_executions:
            return False
        
        if event.schedule_type == ScheduleType.CONDITIONAL:
            try:
                return event.condition_func() if event.condition_func else False
            except Exception as e:
                logger.error(f"Condition function error for event {event.event_id}: {e}")
                return False
        
        return True
    
    def _execute_event(self, event: ScheduledEvent, current_time: float) -> None:
        """Execute a scheduled event"""
        try:
            # Create timer event
            timer_event = TimerEvent(
                timestamp=current_time,
                event_id=event.event_id,
                metadata={
                    'schedule_type': event.schedule_type.value,
                    'execution_count': event.execution_count
                }
            )
            
            # Execute callback
            event.callback()
            
            # Update event state
            event.execution_count += 1
            event.last_execution = current_time
            
            # Schedule next execution
            self._schedule_next_execution(event, current_time)
            
            # Store in history
            self.event_history.append(timer_event)
            self.total_executions += 1
            
            # Notify timer callbacks
            for callback in self.timer_callbacks:
                try:
                    callback(timer_event)
                except Exception as e:
                    logger.error(f"Timer callback error: {e}")
            
            logger.debug(f"Executed event {event.event_id} (count: {event.execution_count})")
            
        except Exception as e:
            logger.error(f"Event execution error for {event.event_id}: {e}")
            self.missed_executions += 1
    
    def _schedule_next_execution(self, event: ScheduledEvent, current_time: float) -> None:
        """Schedule the next execution for an event"""
        if event.schedule_type == ScheduleType.PERIODIC:
            if event.interval_s:
                event.next_execution = current_time + event.interval_s
        elif event.schedule_type == ScheduleType.CRON_LIKE:
            event.next_execution = self._calculate_next_cron_execution(
                event.hour, event.minute, event.weekday)
        elif event.schedule_type == ScheduleType.CONDITIONAL:
            if event.interval_s:
                event.next_execution = current_time + event.interval_s
        # ONE_SHOT events are not rescheduled
    
    def _calculate_next_cron_execution(self, 
                                     hour: Optional[int], 
                                     minute: Optional[int], 
                                     weekday: Optional[int]) -> float:
        """Calculate next execution time for cron-style event"""
        now = self.rtc.get_datetime()
        
        # Start from next minute
        next_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Apply minute constraint
        if minute is not None:
            if next_time.minute > minute:
                next_time = next_time.replace(hour=next_time.hour + 1, minute=minute)
            elif next_time.minute < minute:
                next_time = next_time.replace(minute=minute)
        
        # Apply hour constraint
        if hour is not None:
            if next_time.hour > hour:
                next_time = next_time.replace(day=next_time.day + 1, hour=hour, minute=minute or 0)
            elif next_time.hour < hour:
                next_time = next_time.replace(hour=hour, minute=minute or 0)
        
        # Apply weekday constraint
        if weekday is not None:
            days_ahead = weekday - next_time.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            next_time = next_time + timedelta(days=days_ahead)
        
        return next_time.timestamp()
    
    def get_next_events(self, count: int = 5) -> List[Dict]:
        """Get the next scheduled events"""
        enabled_events = [e for e in self.events.values() if e.enabled]
        enabled_events.sort(key=lambda x: x.next_execution)
        
        result = []
        for event in enabled_events[:count]:
            next_dt = datetime.fromtimestamp(event.next_execution)
            result.append({
                'event_id': event.event_id,
                'schedule_type': event.schedule_type.value,
                'next_execution': event.next_execution,
                'next_execution_str': next_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'time_until_s': event.next_execution - self.rtc.get_time(),
                'execution_count': event.execution_count
            })
        
        return result
    
    def get_schedule_stats(self) -> Dict:
        """Get comprehensive scheduling statistics"""
        current_time = self.rtc.get_time()
        enabled_events = sum(1 for e in self.events.values() if e.enabled)
        
        # Recent executions (last hour)
        recent_executions = [
            e for e in self.event_history 
            if current_time - e.timestamp <= 3600
        ]
        
        return {
            'total_events': len(self.events),
            'enabled_events': enabled_events,
            'disabled_events': len(self.events) - enabled_events,
            'total_executions': self.total_executions,
            'missed_executions': self.missed_executions,
            'recent_executions_1h': len(recent_executions),
            'running': self._running,
            'check_interval_s': self._check_interval,
            'rtc_time': current_time,
            'rtc_datetime': self.rtc.get_datetime().isoformat()
        }
