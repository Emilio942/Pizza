"""
Enhanced Safety Monitor for 99% Safety Compliance
Implements comprehensive safety checks, predictive alerts, and emergency protocols
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SafetyViolation:
    """Detailed safety violation record"""
    timestamp: float
    violation_type: str
    severity: SafetyLevel
    parameter: str
    current_value: float
    limit_value: float
    description: str
    recommended_action: str

@dataclass
class SafetyLimits:
    """Comprehensive safety limits"""
    # Temperature limits
    temp_max_critical: float = 85.0  # °C - immediate shutdown
    temp_max_warning: float = 75.0   # °C - warning level
    temp_min_critical: float = -10.0 # °C - minimum operating temp
    temp_gradient_max: float = 10.0  # °C/s - maximum temperature change rate
    
    # Electrical limits
    voltage_max: float = 3.6         # V - maximum safe voltage
    voltage_min: float = 3.0         # V - minimum operating voltage
    current_max: float = 0.1         # A - maximum safe current
    power_max: float = 0.5           # W - maximum power dissipation
    
    # Environmental limits
    humidity_max: float = 85.0       # % - maximum humidity
    humidity_condensation: float = 95.0  # % - condensation risk
    
    # Operational limits
    degradation_min: float = 0.8     # Minimum acceptable degradation factor
    fatigue_cycles_max: int = 100000 # Maximum fatigue cycles
    thermal_cycles_max: int = 10000  # Maximum thermal cycles
    
    # Rate limits (per second)
    voltage_rate_max: float = 1.0    # V/s
    current_rate_max: float = 0.05   # A/s
    power_rate_max: float = 0.2      # W/s

class EnhancedSafetyMonitor:
    """
    Enhanced safety monitoring system for 99% compliance
    - Predictive safety alerts
    - Multi-level safety checks
    - Emergency response protocols
    - Comprehensive violation tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Safety limits (configurable)
        self.limits = SafetyLimits()
        self._update_limits_from_config(config)
        
        # Safety state tracking
        self.violations_history: List[SafetyViolation] = []
        self.safety_metrics: Dict[str, float] = {}
        self.emergency_stops: int = 0
        self.warning_count: int = 0
        self.critical_count: int = 0
        
        # Predictive monitoring
        self.parameter_history: Dict[str, List[float]] = {
            'temperature': [],
            'voltage': [],
            'current': [],
            'humidity': [],
            'power': [],
            'degradation': []
        }
        
        self.prediction_window = config.get('safety_prediction_window', 10)
        self.history_limit = config.get('safety_history_limit', 1000)
        
        # Safety compliance tracking
        self.total_checks = 0
        self.passed_checks = 0
        self.compliance_history: List[float] = []
        
        # Emergency protocols
        self.emergency_cooldown = 0
        self.emergency_threshold = config.get('emergency_threshold', 5)  # seconds
        
        self.logger.info("Enhanced Safety Monitor initialized with 99% compliance target")
    
    def _update_limits_from_config(self, config: Dict):
        """Update safety limits from configuration"""
        
        safety_config = config.get('safety_limits', {})
        
        for limit_name, default_value in vars(self.limits).items():
            if limit_name in safety_config:
                setattr(self.limits, limit_name, safety_config[limit_name])
    
    def check_safety(self, simulation_state: Dict, control_inputs: Dict) -> Tuple[bool, List[SafetyViolation]]:
        """
        Comprehensive safety check with predictive monitoring
        Returns: (is_safe, violations_list)
        """
        
        self.total_checks += 1
        violations = []
        current_time = time.time()
        
        # Update parameter history
        self._update_parameter_history(simulation_state)
        
        # Core safety checks
        violations.extend(self._check_temperature_safety(simulation_state, current_time))
        violations.extend(self._check_electrical_safety(simulation_state, current_time))
        violations.extend(self._check_environmental_safety(simulation_state, current_time))
        violations.extend(self._check_degradation_safety(simulation_state, current_time))
        violations.extend(self._check_rate_limits(simulation_state, control_inputs, current_time))
        
        # Predictive safety checks
        violations.extend(self._predictive_safety_checks(current_time))
        
        # Cross-parameter correlations
        violations.extend(self._check_parameter_correlations(simulation_state, current_time))
        
        # Emergency conditions
        violations.extend(self._check_emergency_conditions(simulation_state, current_time))
        
        # Update violation history
        self.violations_history.extend(violations)
        self._cleanup_violation_history()
        
        # Determine overall safety status
        is_safe = not any(v.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY] for v in violations)
        
        # Update compliance metrics
        if is_safe:
            self.passed_checks += 1
        
        self._update_safety_metrics(violations)
        
        # Log violations
        for violation in violations:
            if violation.severity == SafetyLevel.EMERGENCY:
                self.logger.critical(f"EMERGENCY: {violation.description}")
                self.emergency_stops += 1
            elif violation.severity == SafetyLevel.CRITICAL:
                self.logger.error(f"CRITICAL: {violation.description}")
                self.critical_count += 1
            elif violation.severity == SafetyLevel.WARNING:
                self.logger.warning(f"WARNING: {violation.description}")
                self.warning_count += 1
        
        return is_safe, violations
    
    def _update_parameter_history(self, simulation_state: Dict):
        """Update parameter history for predictive monitoring"""
        
        parameters = {
            'temperature': simulation_state.get('temperature', 0.0),
            'voltage': simulation_state.get('voltage', 0.0),
            'current': simulation_state.get('current', 0.0),
            'humidity': simulation_state.get('humidity', 0.0),
            'power': simulation_state.get('power_dissipation', 0.0),
            'degradation': simulation_state.get('degradation_factor', 1.0)
        }
        
        for param, value in parameters.items():
            if param in self.parameter_history:
                self.parameter_history[param].append(value)
                
                # Keep history manageable
                if len(self.parameter_history[param]) > self.history_limit:
                    self.parameter_history[param].pop(0)
    
    def _check_temperature_safety(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Enhanced temperature safety checks"""
        
        violations = []
        temp = state.get('temperature', 0.0)
        temp_gradient = state.get('thermal_gradient', 0.0)
        
        # Critical temperature limits
        if temp > self.limits.temp_max_critical:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="temperature_critical",
                severity=SafetyLevel.EMERGENCY,
                parameter="temperature",
                current_value=temp,
                limit_value=self.limits.temp_max_critical,
                description=f"Temperature {temp:.1f}°C exceeds critical limit {self.limits.temp_max_critical}°C",
                recommended_action="IMMEDIATE SHUTDOWN AND COOLING"
            ))
        elif temp > self.limits.temp_max_warning:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="temperature_warning",
                severity=SafetyLevel.WARNING,
                parameter="temperature",
                current_value=temp,
                limit_value=self.limits.temp_max_warning,
                description=f"Temperature {temp:.1f}°C exceeds warning limit {self.limits.temp_max_warning}°C",
                recommended_action="Increase cooling, reduce power"
            ))
        
        # Minimum temperature
        if temp < self.limits.temp_min_critical:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="temperature_too_low",
                severity=SafetyLevel.CRITICAL,
                parameter="temperature",
                current_value=temp,
                limit_value=self.limits.temp_min_critical,
                description=f"Temperature {temp:.1f}°C below minimum {self.limits.temp_min_critical}°C",
                recommended_action="Apply heating, check thermal insulation"
            ))
        
        # Temperature gradient check
        if abs(temp_gradient) > self.limits.temp_gradient_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="temperature_gradient",
                severity=SafetyLevel.CRITICAL,
                parameter="thermal_gradient",
                current_value=abs(temp_gradient),
                limit_value=self.limits.temp_gradient_max,
                description=f"Temperature gradient {temp_gradient:.2f}°C/s exceeds limit {self.limits.temp_gradient_max}°C/s",
                recommended_action="Reduce heating/cooling rate, check thermal control"
            ))
        
        return violations
    
    def _check_electrical_safety(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Enhanced electrical safety checks"""
        
        violations = []
        voltage = state.get('voltage', 0.0)
        current = state.get('current', 0.0)
        power = state.get('power_dissipation', 0.0)
        
        # Voltage limits
        if voltage > self.limits.voltage_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="voltage_overvoltage",
                severity=SafetyLevel.CRITICAL,
                parameter="voltage",
                current_value=voltage,
                limit_value=self.limits.voltage_max,
                description=f"Voltage {voltage:.2f}V exceeds maximum {self.limits.voltage_max}V",
                recommended_action="Reduce supply voltage, check voltage regulation"
            ))
        elif voltage < self.limits.voltage_min:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="voltage_undervoltage",
                severity=SafetyLevel.WARNING,
                parameter="voltage",
                current_value=voltage,
                limit_value=self.limits.voltage_min,
                description=f"Voltage {voltage:.2f}V below minimum {self.limits.voltage_min}V",
                recommended_action="Check power supply, increase voltage"
            ))
        
        # Current limits
        if current > self.limits.current_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="current_overcurrent",
                severity=SafetyLevel.CRITICAL,
                parameter="current",
                current_value=current,
                limit_value=self.limits.current_max,
                description=f"Current {current:.3f}A exceeds maximum {self.limits.current_max}A",
                recommended_action="Reduce load, check for short circuits"
            ))
        
        # Power limits
        if power > self.limits.power_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="power_overpower",
                severity=SafetyLevel.CRITICAL,
                parameter="power",
                current_value=power,
                limit_value=self.limits.power_max,
                description=f"Power {power:.3f}W exceeds maximum {self.limits.power_max}W",
                recommended_action="Reduce voltage and current, increase cooling"
            ))
        
        return violations
    
    def _check_environmental_safety(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Environmental safety checks"""
        
        violations = []
        humidity = state.get('humidity', 0.0)
        
        # Humidity limits
        if humidity > self.limits.humidity_condensation:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="humidity_condensation",
                severity=SafetyLevel.EMERGENCY,
                parameter="humidity",
                current_value=humidity,
                limit_value=self.limits.humidity_condensation,
                description=f"Humidity {humidity:.1f}% - condensation risk!",
                recommended_action="IMMEDIATE DEHUMIDIFICATION, CHECK FOR WATER DAMAGE"
            ))
        elif humidity > self.limits.humidity_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="humidity_high",
                severity=SafetyLevel.WARNING,
                parameter="humidity",
                current_value=humidity,
                limit_value=self.limits.humidity_max,
                description=f"Humidity {humidity:.1f}% exceeds safe limit {self.limits.humidity_max}%",
                recommended_action="Increase ventilation, reduce humidity"
            ))
        
        return violations
    
    def _check_degradation_safety(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Component degradation safety checks"""
        
        violations = []
        degradation = state.get('degradation_factor', 1.0)
        thermal_cycles = state.get('thermal_cycles', 0)
        fatigue_cycles = state.get('fatigue_cycles', 0)
        
        # Degradation factor
        if degradation < self.limits.degradation_min:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="component_degradation",
                severity=SafetyLevel.CRITICAL,
                parameter="degradation_factor",
                current_value=degradation,
                limit_value=self.limits.degradation_min,
                description=f"Component degradation {degradation:.3f} below safe limit {self.limits.degradation_min}",
                recommended_action="Component replacement required"
            ))
        
        # Thermal cycles
        if thermal_cycles > self.limits.thermal_cycles_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="thermal_cycles_exceeded",
                severity=SafetyLevel.WARNING,
                parameter="thermal_cycles",
                current_value=thermal_cycles,
                limit_value=self.limits.thermal_cycles_max,
                description=f"Thermal cycles {thermal_cycles} exceed recommended limit {self.limits.thermal_cycles_max}",
                recommended_action="Schedule maintenance, monitor degradation"
            ))
        
        # Fatigue cycles
        if fatigue_cycles > self.limits.fatigue_cycles_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="fatigue_cycles_exceeded",
                severity=SafetyLevel.CRITICAL,
                parameter="fatigue_cycles",
                current_value=fatigue_cycles,
                limit_value=self.limits.fatigue_cycles_max,
                description=f"Fatigue cycles {fatigue_cycles} exceed safe limit {self.limits.fatigue_cycles_max}",
                recommended_action="COMPONENT REPLACEMENT REQUIRED"
            ))
        
        return violations
    
    def _check_rate_limits(self, state: Dict, control_inputs: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check parameter change rates"""
        
        violations = []
        
        if len(self.parameter_history['voltage']) < 2:
            return violations
        
        # Calculate rates of change
        dt = 1.0  # Assuming 1 second intervals
        
        voltage_rate = abs(self.parameter_history['voltage'][-1] - self.parameter_history['voltage'][-2]) / dt
        current_rate = abs(self.parameter_history['current'][-1] - self.parameter_history['current'][-2]) / dt
        power_rate = abs(self.parameter_history['power'][-1] - self.parameter_history['power'][-2]) / dt
        
        # Check voltage rate
        if voltage_rate > self.limits.voltage_rate_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="voltage_rate_limit",
                severity=SafetyLevel.WARNING,
                parameter="voltage_rate",
                current_value=voltage_rate,
                limit_value=self.limits.voltage_rate_max,
                description=f"Voltage change rate {voltage_rate:.2f}V/s exceeds limit {self.limits.voltage_rate_max}V/s",
                recommended_action="Reduce voltage change rate"
            ))
        
        # Check current rate
        if current_rate > self.limits.current_rate_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="current_rate_limit",
                severity=SafetyLevel.WARNING,
                parameter="current_rate",
                current_value=current_rate,
                limit_value=self.limits.current_rate_max,
                description=f"Current change rate {current_rate:.3f}A/s exceeds limit {self.limits.current_rate_max}A/s",
                recommended_action="Reduce current change rate"
            ))
        
        # Check power rate
        if power_rate > self.limits.power_rate_max:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="power_rate_limit",
                severity=SafetyLevel.WARNING,
                parameter="power_rate",
                current_value=power_rate,
                limit_value=self.limits.power_rate_max,
                description=f"Power change rate {power_rate:.3f}W/s exceeds limit {self.limits.power_rate_max}W/s",
                recommended_action="Reduce power change rate"
            ))
        
        return violations
    
    def _predictive_safety_checks(self, timestamp: float) -> List[SafetyViolation]:
        """Predictive safety monitoring using parameter trends"""
        
        violations = []
        
        if len(self.parameter_history['temperature']) < self.prediction_window:
            return violations
        
        # Predict temperature trend
        recent_temps = self.parameter_history['temperature'][-self.prediction_window:]
        if len(recent_temps) >= 3:
            # Simple linear prediction
            x = np.arange(len(recent_temps))
            try:
                coeffs = np.polyfit(x, recent_temps, 1)
                predicted_temp_in_10s = recent_temps[-1] + coeffs[0] * 10
                
                if predicted_temp_in_10s > self.limits.temp_max_warning:
                    violations.append(SafetyViolation(
                        timestamp=timestamp,
                        violation_type="temperature_prediction",
                        severity=SafetyLevel.WARNING,
                        parameter="temperature_trend",
                        current_value=predicted_temp_in_10s,
                        limit_value=self.limits.temp_max_warning,
                        description=f"Predicted temperature {predicted_temp_in_10s:.1f}°C will exceed warning limit",
                        recommended_action="Preemptive cooling increase"
                    ))
            except:
                pass  # Ignore prediction errors
        
        return violations
    
    def _check_parameter_correlations(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check dangerous parameter combinations"""
        
        violations = []
        temp = state.get('temperature', 0.0)
        humidity = state.get('humidity', 0.0)
        power = state.get('power_dissipation', 0.0)
        
        # High temperature + high humidity = corrosion risk
        if temp > 60.0 and humidity > 75.0:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="temp_humidity_correlation",
                severity=SafetyLevel.WARNING,
                parameter="temperature_humidity",
                current_value=temp * humidity / 100.0,
                limit_value=45.0,  # 60°C * 75%
                description=f"High temperature ({temp:.1f}°C) + high humidity ({humidity:.1f}%) increases corrosion risk",
                recommended_action="Reduce temperature or humidity"
            ))
        
        # High power + high temperature = thermal runaway risk
        if power > 0.3 and temp > 70.0:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="power_temp_correlation",
                severity=SafetyLevel.CRITICAL,
                parameter="power_temperature",
                current_value=power * temp,
                limit_value=21.0,  # 0.3W * 70°C
                description=f"High power ({power:.2f}W) + high temperature ({temp:.1f}°C) - thermal runaway risk",
                recommended_action="REDUCE POWER IMMEDIATELY, INCREASE COOLING"
            ))
        
        return violations
    
    def _check_emergency_conditions(self, state: Dict, timestamp: float) -> List[SafetyViolation]:
        """Check for emergency shutdown conditions"""
        
        violations = []
        
        # Multiple critical violations in short time
        recent_critical = [v for v in self.violations_history[-10:] 
                          if v.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]]
        
        if len(recent_critical) >= 3:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                violation_type="multiple_critical_violations",
                severity=SafetyLevel.EMERGENCY,
                parameter="system_stability",
                current_value=len(recent_critical),
                limit_value=2.0,
                description=f"{len(recent_critical)} critical violations in short time - system instability",
                recommended_action="EMERGENCY SHUTDOWN AND INSPECTION"
            ))
        
        return violations
    
    def _update_safety_metrics(self, violations: List[SafetyViolation]):
        """Update safety performance metrics"""
        
        # Current compliance rate
        current_compliance = self.passed_checks / max(1, self.total_checks) * 100.0
        self.compliance_history.append(current_compliance)
        
        # Keep compliance history manageable
        if len(self.compliance_history) > 1000:
            self.compliance_history.pop(0)
        
        # Update metrics
        self.safety_metrics = {
            'compliance_rate': current_compliance,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.total_checks - self.passed_checks,
            'warning_count': self.warning_count,
            'critical_count': self.critical_count,
            'emergency_stops': self.emergency_stops,
            'recent_compliance': np.mean(self.compliance_history[-100:]) if len(self.compliance_history) >= 100 else current_compliance,
            'compliance_trend': np.mean(self.compliance_history[-10:]) - np.mean(self.compliance_history[-50:-40]) if len(self.compliance_history) >= 50 else 0.0,
            'violations_per_hour': len([v for v in self.violations_history if time.time() - v.timestamp < 3600]),
            'safety_level': self._get_current_safety_level()
        }
    
    def _get_current_safety_level(self) -> str:
        """Determine current overall safety level"""
        
        recent_violations = [v for v in self.violations_history[-20:]]
        
        if any(v.severity == SafetyLevel.EMERGENCY for v in recent_violations):
            return "EMERGENCY"
        elif any(v.severity == SafetyLevel.CRITICAL for v in recent_violations):
            return "CRITICAL"
        elif any(v.severity == SafetyLevel.WARNING for v in recent_violations):
            return "WARNING"
        else:
            return "SAFE"
    
    def _cleanup_violation_history(self):
        """Remove old violations to manage memory"""
        
        if len(self.violations_history) > 10000:
            # Keep last 5000 violations
            self.violations_history = self.violations_history[-5000:]
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get comprehensive safety metrics"""
        
        return self.safety_metrics.copy()
    
    def get_compliance_percentage(self) -> float:
        """Get current safety compliance percentage"""
        
        return self.safety_metrics.get('compliance_rate', 0.0)
    
    def reset(self):
        """Reset safety monitor state"""
        
        self.violations_history.clear()
        self.parameter_history = {key: [] for key in self.parameter_history.keys()}
        self.total_checks = 0
        self.passed_checks = 0
        self.emergency_stops = 0
        self.warning_count = 0
        self.critical_count = 0
        self.compliance_history.clear()
        self.safety_metrics.clear()
        
        self.logger.info("Safety monitor reset")
