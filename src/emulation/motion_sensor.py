"""
Motion/PIR Sensor Emulation for RP2040-based Pizza Detection System.
Implements simulated motion detection for adaptive duty-cycle triggering.
"""

import time
import random
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


class MotionSensorType(Enum):
    """Types of motion sensors"""
    PIR = "pir"           # Passive Infrared sensor
    MICROWAVE = "microwave"  # Microwave radar sensor
    ULTRASONIC = "ultrasonic"  # Ultrasonic distance sensor


@dataclass
class MotionEvent:
    """Represents a motion detection event"""
    timestamp: float
    sensor_id: str
    motion_detected: bool
    confidence: float  # 0.0 to 1.0
    trigger_source: str = "motion"


class MotionSensor:
    """
    Emulates a motion/PIR sensor for triggering wake-up events.
    Simulates realistic motion detection patterns with configurable sensitivity.
    """
    
    def __init__(self, 
                 sensor_type: MotionSensorType = MotionSensorType.PIR,
                 sensor_id: str = "motion_01",
                 sensitivity: float = 0.7,
                 detection_range_m: float = 3.0,
                 false_positive_rate: float = 0.05):
        """
        Initialize motion sensor emulation.
        
        Args:
            sensor_type: Type of motion sensor to emulate
            sensor_id: Unique identifier for this sensor
            sensitivity: Detection sensitivity (0.0 to 1.0)
            detection_range_m: Detection range in meters
            false_positive_rate: Rate of false positive detections (0.0 to 1.0)
        """
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.detection_range_m = detection_range_m
        self.false_positive_rate = max(0.0, min(1.0, false_positive_rate))
        
        # Sensor state
        self.initialized = False
        self.enabled = True
        self.motion_detected = False
        self.last_detection_time = 0.0
        self.detection_count = 0
        
        # Detection history for pattern analysis
        self.detection_history = deque(maxlen=100)
        self.event_history = deque(maxlen=50)
        
        # Sensor-specific parameters
        if sensor_type == MotionSensorType.PIR:
            self.response_time_ms = 50  # PIR response time
            self.reset_time_s = 2.0     # Time to reset after detection
            self.detection_angle = 110  # Detection angle in degrees
        elif sensor_type == MotionSensorType.MICROWAVE:
            self.response_time_ms = 10  # Faster microwave response
            self.reset_time_s = 0.5     # Faster reset
            self.detection_angle = 360  # 360-degree detection
        else:  # ULTRASONIC
            self.response_time_ms = 20  # Ultrasonic response time
            self.reset_time_s = 1.0     # Medium reset time
            self.detection_angle = 15   # Narrow beam
        
        # Simulated environment
        self.ambient_activity_level = 0.1  # Background activity level
        self.simulation_patterns = []
        
        logger.info(f"Motion sensor {sensor_id} ({sensor_type.value}) initialized: "
                   f"sensitivity={sensitivity:.1f}, range={detection_range_m:.1f}m")
    
    def initialize(self) -> bool:
        """
        Initialize the motion sensor.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Simulate initialization time
            time.sleep(self.response_time_ms / 1000.0)
            self.initialized = True
            self.last_detection_time = time.time()
            
            logger.info(f"Motion sensor {self.sensor_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Motion sensor {self.sensor_id} initialization failed: {e}")
            return False
    
    def enable(self) -> None:
        """Enable motion detection"""
        self.enabled = True
        logger.debug(f"Motion sensor {self.sensor_id} enabled")
    
    def disable(self) -> None:
        """Disable motion detection"""
        self.enabled = False
        self.motion_detected = False
        logger.debug(f"Motion sensor {self.sensor_id} disabled")
    
    def check_motion(self) -> bool:
        """
        Check for motion detection.
        
        Returns:
            True if motion is detected, False otherwise
        """
        if not self.initialized or not self.enabled:
            return False
        
        current_time = time.time()
        
        # Check if we're still in reset period after last detection
        if (self.motion_detected and 
            current_time - self.last_detection_time < self.reset_time_s):
            return self.motion_detected
        
        # Simulate motion detection based on various factors
        motion_probability = self._calculate_motion_probability()
        detected = random.random() < motion_probability
        
        if detected:
            self._trigger_detection()
            
        # Add random false positives
        if not detected and random.random() < self.false_positive_rate:
            detected = True
            self._trigger_detection(confidence=0.3)  # Low confidence for false positive
            logger.debug(f"Motion sensor {self.sensor_id}: False positive triggered")
        
        return detected
    
    def _calculate_motion_probability(self) -> float:
        """
        Calculate the probability of motion detection based on simulation patterns.
        
        Returns:
            Probability of motion detection (0.0 to 1.0)
        """
        base_probability = self.ambient_activity_level
        
        # Add time-based patterns (more activity during day)
        current_hour = time.localtime().tm_hour
        if 6 <= current_hour <= 22:  # Daytime
            time_factor = 1.5
        else:  # Nighttime
            time_factor = 0.3
        
        # Add periodic patterns from simulation
        pattern_factor = 1.0
        current_time = time.time()
        for pattern in self.simulation_patterns:
            pattern_factor *= pattern(current_time)
        
        # Calculate final probability
        probability = base_probability * time_factor * pattern_factor * self.sensitivity
        
        return min(1.0, probability)
    
    def _trigger_detection(self, confidence: float = 0.8) -> None:
        """
        Trigger a motion detection event.
        
        Args:
            confidence: Confidence level of the detection
        """
        current_time = time.time()
        self.motion_detected = True
        self.last_detection_time = current_time
        self.detection_count += 1
        
        # Create motion event
        event = MotionEvent(
            timestamp=current_time,
            sensor_id=self.sensor_id,
            motion_detected=True,
            confidence=confidence
        )
        
        # Store in history
        self.detection_history.append(current_time)
        self.event_history.append(event)
        
        logger.info(f"Motion detected by {self.sensor_id} (confidence: {confidence:.2f})")
    
    def add_simulation_pattern(self, pattern_func: Callable[[float], float]) -> None:
        """
        Add a simulation pattern function.
        
        Args:
            pattern_func: Function that takes current time and returns a multiplier
        """
        self.simulation_patterns.append(pattern_func)
        logger.debug(f"Added simulation pattern to sensor {self.sensor_id}")
    
    def set_ambient_activity(self, level: float) -> None:
        """
        Set the ambient activity level.
        
        Args:
            level: Activity level (0.0 to 1.0)
        """
        self.ambient_activity_level = max(0.0, min(1.0, level))
        logger.debug(f"Sensor {self.sensor_id} ambient activity set to {level:.2f}")
    
    def get_recent_detections(self, time_window_s: float = 60.0) -> List[MotionEvent]:
        """
        Get motion detection events within a time window.
        
        Args:
            time_window_s: Time window in seconds
            
        Returns:
            List of recent motion events
        """
        current_time = time.time()
        cutoff_time = current_time - time_window_s
        
        return [event for event in self.event_history 
                if event.timestamp >= cutoff_time]
    
    def get_detection_rate(self, time_window_s: float = 300.0) -> float:
        """
        Calculate detection rate over a time window.
        
        Args:
            time_window_s: Time window in seconds
            
        Returns:
            Detections per minute
        """
        recent_detections = self.get_recent_detections(time_window_s)
        if not recent_detections:
            return 0.0
        
        return len(recent_detections) / (time_window_s / 60.0)
    
    def reset_detection_state(self) -> None:
        """Reset the current detection state"""
        self.motion_detected = False
        logger.debug(f"Motion sensor {self.sensor_id} detection state reset")
    
    def get_sensor_stats(self) -> Dict:
        """
        Get comprehensive sensor statistics.
        
        Returns:
            Dictionary with sensor statistics
        """
        current_time = time.time()
        recent_detections = self.get_recent_detections(300.0)  # Last 5 minutes
        
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'initialized': self.initialized,
            'enabled': self.enabled,
            'motion_detected': self.motion_detected,
            'total_detections': self.detection_count,
            'recent_detections_5min': len(recent_detections),
            'detection_rate_per_min': self.get_detection_rate(),
            'last_detection_time': self.last_detection_time,
            'time_since_last_detection': current_time - self.last_detection_time,
            'sensitivity': self.sensitivity,
            'detection_range_m': self.detection_range_m,
            'ambient_activity_level': self.ambient_activity_level,
            'false_positive_rate': self.false_positive_rate
        }


class MotionSensorController:
    """
    Controller for multiple motion sensors.
    Manages sensor coordination and fusion of detection events.
    """
    
    def __init__(self):
        """Initialize motion sensor controller"""
        self.sensors: Dict[str, MotionSensor] = {}
        self.motion_callbacks: List[Callable[[MotionEvent], None]] = []
        self.fusion_enabled = True
        self.detection_threshold = 1  # Minimum sensors needed for detection
        self.active = False
        
        # Initialize default sensors for emulation
        self._initialize_default_sensors()
        
    def _initialize_default_sensors(self) -> None:
        """Initialize default set of motion sensors for emulation"""
        # Create default PIR sensor
        pir_sensor = MotionSensor(
            sensor_id="pir_main",
            sensor_type=MotionSensorType.PIR,
            detection_range_m=5.0,
            sensitivity=0.7
        )
        
        # Create default microwave sensor
        microwave_sensor = MotionSensor(
            sensor_id="microwave_main", 
            sensor_type=MotionSensorType.MICROWAVE,
            detection_range_m=8.0,
            sensitivity=0.6
        )
        
        # Add sensors
        self.add_sensor(pir_sensor)
        self.add_sensor(microwave_sensor)
        
        logger.info("Default motion sensors initialized")
        
    def start(self) -> None:
        """Start the motion sensor controller"""
        self.active = True
        
        # Initialize all sensors
        for sensor in self.sensors.values():
            sensor.initialize()
            sensor.enable()
            
        logger.info("Motion sensor controller started")
        
    def stop(self) -> None:
        """Stop the motion sensor controller"""
        self.active = False
        
        # Disable all sensors
        for sensor in self.sensors.values():
            sensor.disable()
            
        logger.info("Motion sensor controller stopped")
        
    def is_active(self) -> bool:
        """Check if motion sensor controller is active"""
        return self.active
        
    def simulate_motion_event(self, sensor_type: str = "PIR", duration: float = 1.0, confidence: float = 0.8) -> None:
        """
        Simulate a motion detection event for testing.
        
        Args:
            sensor_type: Type of sensor to simulate ("PIR", "MICROWAVE", "ULTRASONIC")
            duration: Duration of motion event in seconds
            confidence: Confidence level of detection
        """
        # Find a sensor of the specified type
        target_sensor = None
        for sensor in self.sensors.values():
            if sensor.sensor_type.value.upper() == sensor_type.upper():
                target_sensor = sensor
                break
                
        if not target_sensor:
            # Create temporary sensor if none exists
            temp_sensor = MotionSensor(
                sensor_id=f"temp_{sensor_type.lower()}",
                sensor_type=MotionSensorType(sensor_type.lower()),
                detection_range_m=5.0,
                sensitivity=0.8
            )
            temp_sensor.initialize()
            temp_sensor.enable()
            self.add_sensor(temp_sensor)
            target_sensor = temp_sensor
            
        # Trigger detection
        target_sensor._trigger_detection(confidence=confidence)
        
        # Create motion event and notify callbacks
        event = MotionEvent(
            timestamp=time.time(),
            sensor_id=target_sensor.sensor_id,
            motion_detected=True,
            confidence=confidence,
            trigger_source=f"simulated_{sensor_type.lower()}"
        )
        
        for callback in self.motion_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Motion callback error during simulation: {e}")
                
        logger.info(f"Simulated {sensor_type} motion event (duration: {duration}s, confidence: {confidence})")
        
    def get_statistics(self) -> Dict:
        """Get motion controller statistics"""
        return self.get_controller_stats()
        """Initialize motion sensor controller"""
        self.sensors: Dict[str, MotionSensor] = {}
        self.motion_callbacks: List[Callable[[MotionEvent], None]] = []
        self.fusion_enabled = True
        self.detection_threshold = 1  # Minimum sensors needed for detection
        self.active = False
        
        # Initialize default sensors for emulation
        self._initialize_default_sensors()
        
    def _initialize_default_sensors(self) -> None:
        """Initialize default set of motion sensors for emulation"""
        # Create default PIR sensor
        pir_sensor = MotionSensor(
            sensor_id="pir_main",
            sensor_type=MotionSensorType.PIR,
            detection_range_m=5.0,
            sensitivity=0.7
        )
        
        # Create default microwave sensor
        microwave_sensor = MotionSensor(
            sensor_id="microwave_main", 
            sensor_type=MotionSensorType.MICROWAVE,
            detection_range_m=8.0,
            sensitivity=0.6
        )
        
        # Add sensors
        self.add_sensor(pir_sensor)
        self.add_sensor(microwave_sensor)
        
        logger.info("Default motion sensors initialized")
        
    def start(self) -> None:
        """Start the motion sensor controller"""
        self.active = True
        
        # Initialize all sensors
        for sensor in self.sensors.values():
            sensor.initialize()
            sensor.enable()
            
        logger.info("Motion sensor controller started")
        
    def stop(self) -> None:
        """Stop the motion sensor controller"""
        self.active = False
        
        # Disable all sensors
        for sensor in self.sensors.values():
            sensor.disable()
            
        logger.info("Motion sensor controller stopped")
        
    def is_active(self) -> bool:
        """Check if motion sensor controller is active"""
        return self.active
        
    def add_sensor(self, sensor: MotionSensor) -> None:
        """Add a motion sensor to the controller"""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Added motion sensor {sensor.sensor_id} to controller")
    
    def remove_sensor(self, sensor_id: str) -> bool:
        """Remove a motion sensor from the controller"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            logger.info(f"Removed motion sensor {sensor_id} from controller")
            return True
        return False
    
    def add_motion_callback(self, callback: Callable[[MotionEvent], None]) -> None:
        """Add a callback function to be called when motion is detected"""
        self.motion_callbacks.append(callback)
    
    def check_motion_all_sensors(self) -> bool:
        """
        Check motion on all sensors and apply fusion logic.
        
        Returns:
            True if motion is detected based on fusion logic
        """
        detected_sensors = []
        
        for sensor in self.sensors.values():
            if sensor.check_motion():
                detected_sensors.append(sensor)
        
        # Apply fusion logic
        motion_detected = len(detected_sensors) >= self.detection_threshold
        
        if motion_detected and detected_sensors:
            # Create fused motion event
            event = MotionEvent(
                timestamp=time.time(),
                sensor_id="fused",
                motion_detected=True,
                confidence=min(1.0, len(detected_sensors) / len(self.sensors)),
                trigger_source="motion_fusion"
            )
            
            # Notify callbacks
            for callback in self.motion_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Motion callback error: {e}")
        
        return motion_detected
    
    def set_detection_threshold(self, threshold: int) -> None:
        """Set the minimum number of sensors needed for detection"""
        self.detection_threshold = max(1, min(threshold, len(self.sensors)))
        logger.info(f"Motion detection threshold set to {self.detection_threshold}")
    
    def enable_all_sensors(self) -> None:
        """Enable all motion sensors"""
        for sensor in self.sensors.values():
            sensor.enable()
    
    def disable_all_sensors(self) -> None:
        """Disable all motion sensors"""
        for sensor in self.sensors.values():
            sensor.disable()
    
    def get_controller_stats(self) -> Dict:
        """Get comprehensive controller statistics"""
        sensor_stats = {sid: sensor.get_sensor_stats() 
                       for sid, sensor in self.sensors.items()}
        
        active_sensors = sum(1 for s in self.sensors.values() if s.enabled)
        detecting_sensors = sum(1 for s in self.sensors.values() if s.motion_detected)
        
        return {
            'total_sensors': len(self.sensors),
            'active_sensors': active_sensors,
            'detecting_sensors': detecting_sensors,
            'detection_threshold': self.detection_threshold,
            'fusion_enabled': self.fusion_enabled,
            'sensor_details': sensor_stats
        }
    
    def simulate_motion_event(self, sensor_type: str = "PIR", duration: float = 1.0, confidence: float = 0.8) -> None:
        """
        Simulate a motion detection event for testing.
        
        Args:
            sensor_type: Type of sensor to simulate ("PIR", "MICROWAVE", "ULTRASONIC")
            duration: Duration of motion event in seconds
            confidence: Confidence level of detection
        """
        # Find a sensor of the specified type
        target_sensor = None
        for sensor in self.sensors.values():
            if sensor.sensor_type.value.upper() == sensor_type.upper():
                target_sensor = sensor
                break
                
        if not target_sensor:
            # Create temporary sensor if none exists
            temp_sensor = MotionSensor(
                sensor_id=f"temp_{sensor_type.lower()}",
                sensor_type=MotionSensorType(sensor_type.lower()),
                detection_range_m=5.0,
                sensitivity=0.8
            )
            temp_sensor.initialize()
            temp_sensor.enable()
            self.add_sensor(temp_sensor)
            target_sensor = temp_sensor
            
        # Trigger detection
        target_sensor._trigger_detection(confidence=confidence)
        
        # Create motion event and notify callbacks
        event = MotionEvent(
            timestamp=time.time(),
            sensor_id=target_sensor.sensor_id,
            motion_detected=True,
            confidence=confidence,
            trigger_source=f"simulated_{sensor_type.lower()}"
        )
        
        for callback in self.motion_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Motion callback error during simulation: {e}")
                
        logger.info(f"Simulated {sensor_type} motion event (duration: {duration}s, confidence: {confidence})")
        
    def get_statistics(self) -> Dict:
        """Get motion controller statistics"""
        return self.get_controller_stats()
