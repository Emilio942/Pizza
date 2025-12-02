"""
Enhanced Safety Compliance System
Addresses the safety compliance gap (97% -> 99% target)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time

class SafetyLevel(Enum):
    """Safety criticality levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class SafetyViolationType(Enum):
    """Types of safety violations"""
    TEMPERATURE_OVERRUN = "temperature_overrun"
    VOLTAGE_OVERRUN = "voltage_overrun"
    CURRENT_OVERRUN = "current_overrun"
    PRESSURE_OVERRUN = "pressure_overrun" 
    PARAMETER_OUT_OF_BOUNDS = "parameter_out_of_bounds"
    CONVERGENCE_FAILURE = "convergence_failure"
    SIMULATION_INSTABILITY = "simulation_instability"
    HARDWARE_ANOMALY = "hardware_anomaly"
    SAFETY_INTERLOCK_TRIGGERED = "safety_interlock_triggered"

@dataclass
class SafetyRule:
    """Individual safety rule definition"""
    rule_id: str
    description: str
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    safety_level: SafetyLevel = SafetyLevel.MEDIUM
    violation_type: SafetyViolationType = SafetyViolationType.PARAMETER_OUT_OF_BOUNDS
    enabled: bool = True
    tolerance: float = 0.0
    consecutive_violations_limit: int = 3
    recovery_time: float = 1.0

@dataclass
class SafetyViolation:
    """Safety violation event"""
    timestamp: float
    rule_id: str
    violation_type: SafetyViolationType
    safety_level: SafetyLevel
    parameter: str
    current_value: float
    limit_value: float
    severity_score: float
    description: str
    context: Dict[str, Any] = field(default_factory=dict)

class EnhancedSafetyManager:
    """Enhanced safety compliance system with proactive monitoring"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Safety rules database
        self.safety_rules: Dict[str, SafetyRule] = {}
        self.violation_history: List[SafetyViolation] = []
        self.consecutive_violations: Dict[str, int] = {}
        
        # Safety state tracking
        self.safety_state = {
            'overall_status': 'safe',
            'active_violations': [],
            'warning_count': 0,
            'critical_count': 0,
            'last_violation_time': 0.0
        }
        
        # Enhanced monitoring
        self.predictive_monitoring = config.get('enable_predictive_monitoring', True)
        self.trend_analysis_window = config.get('trend_analysis_window', 50)
        self.parameter_history: Dict[str, List[float]] = {}
        
        # Initialize safety rules
        self._initialize_safety_rules()
        
        # Safety metrics
        self.safety_metrics = {
            'total_checks': 0,
            'violation_count': 0,
            'compliance_rate': 1.0,
            'mean_time_between_violations': float('inf'),
            'safety_score': 1.0
        }
        
    def _initialize_safety_rules(self):
        """Initialize comprehensive safety rules"""
        
        # Temperature safety rules
        self.add_safety_rule(SafetyRule(
            rule_id="temp_max_critical",
            description="Critical maximum temperature limit",
            parameter="temperature",
            max_value=85.0,
            safety_level=SafetyLevel.CRITICAL,
            violation_type=SafetyViolationType.TEMPERATURE_OVERRUN,
            tolerance=0.0,
            consecutive_violations_limit=1
        ))
        
        self.add_safety_rule(SafetyRule(
            rule_id="temp_max_high",
            description="High temperature warning limit",
            parameter="temperature",
            max_value=80.0,
            safety_level=SafetyLevel.HIGH,
            violation_type=SafetyViolationType.TEMPERATURE_OVERRUN,
            tolerance=0.5,
            consecutive_violations_limit=2
        ))
        
        self.add_safety_rule(SafetyRule(
            rule_id="temp_min_critical",
            description="Critical minimum temperature limit",
            parameter="temperature",
            min_value=10.0,
            safety_level=SafetyLevel.CRITICAL,
            violation_type=SafetyViolationType.TEMPERATURE_OVERRUN,
            tolerance=0.0,
            consecutive_violations_limit=1
        ))
        
        # Voltage safety rules
        self.add_safety_rule(SafetyRule(
            rule_id="voltage_max_critical",
            description="Critical maximum voltage limit",
            parameter="voltage",
            max_value=5.5,
            safety_level=SafetyLevel.CRITICAL,
            violation_type=SafetyViolationType.VOLTAGE_OVERRUN,
            tolerance=0.0,
            consecutive_violations_limit=1
        ))
        
        self.add_safety_rule(SafetyRule(
            rule_id="voltage_min_critical",
            description="Critical minimum voltage limit",
            parameter="voltage",
            min_value=2.5,
            safety_level=SafetyLevel.CRITICAL,
            violation_type=SafetyViolationType.VOLTAGE_OVERRUN,
            tolerance=0.0,
            consecutive_violations_limit=1
        ))
        
        # Current safety rules
        self.add_safety_rule(SafetyRule(
            rule_id="current_max_critical",
            description="Critical maximum current limit",
            parameter="current",
            max_value=3.0,
            safety_level=SafetyLevel.CRITICAL,
            violation_type=SafetyViolationType.CURRENT_OVERRUN,
            tolerance=0.0,
            consecutive_violations_limit=1
        ))
        
        # Pressure safety rules
        self.add_safety_rule(SafetyRule(
            rule_id="pressure_max_critical",
            description="Critical maximum pressure limit",
            parameter="pressure",
            max_value=2.0,
            safety_level=SafetyLevel.CRITICAL,
            violation_type=SafetyViolationType.PRESSURE_OVERRUN,
            tolerance=0.0,
            consecutive_violations_limit=1
        ))
        
        # Parameter bounds safety rules
        self.add_safety_rule(SafetyRule(
            rule_id="pg_lr_bounds",
            description="PG learning rate bounds",
            parameter="pg_learning_rate",
            min_value=1e-6,
            max_value=0.1,
            safety_level=SafetyLevel.HIGH,
            violation_type=SafetyViolationType.PARAMETER_OUT_OF_BOUNDS,
            tolerance=0.0,
            consecutive_violations_limit=3
        ))
        
        self.add_safety_rule(SafetyRule(
            rule_id="es_sigma_bounds",
            description="ES mutation strength bounds",
            parameter="es_sigma",
            min_value=0.001,
            max_value=1.0,
            safety_level=SafetyLevel.HIGH,
            violation_type=SafetyViolationType.PARAMETER_OUT_OF_BOUNDS,
            tolerance=0.0,
            consecutive_violations_limit=3
        ))
        
        # Convergence safety rules
        self.add_safety_rule(SafetyRule(
            rule_id="convergence_stagnation",
            description="Training convergence stagnation detection",
            parameter="fitness_improvement",
            min_value=1e-6,
            safety_level=SafetyLevel.MEDIUM,
            violation_type=SafetyViolationType.CONVERGENCE_FAILURE,
            tolerance=0.0,
            consecutive_violations_limit=20
        ))
        
        # Simulation stability rules
        self.add_safety_rule(SafetyRule(
            rule_id="simulation_stability",
            description="Simulation stability check",
            parameter="simulation_stability_score",
            min_value=0.5,
            safety_level=SafetyLevel.HIGH,
            violation_type=SafetyViolationType.SIMULATION_INSTABILITY,
            tolerance=0.1,
            consecutive_violations_limit=5
        ))
        
    def add_safety_rule(self, rule: SafetyRule):
        """Add a new safety rule"""
        self.safety_rules[rule.rule_id] = rule
        self.consecutive_violations[rule.rule_id] = 0
        self.logger.info(f"Added safety rule: {rule.rule_id}")
        
    def check_safety_compliance(self, parameters: Dict[str, Any]) -> Tuple[bool, List[SafetyViolation]]:
        """Comprehensive safety compliance check"""
        
        current_time = time.time()
        violations = []
        is_safe = True
        
        # Update metrics
        self.safety_metrics['total_checks'] += 1
        
        # Check each safety rule
        for rule_id, rule in self.safety_rules.items():
            if not rule.enabled:
                continue
                
            # Get parameter value
            param_value = parameters.get(rule.parameter)
            if param_value is None:
                continue
                
            # Update parameter history for predictive monitoring
            self._update_parameter_history(rule.parameter, param_value)
            
            # Check rule violation
            violation = self._check_rule_violation(rule, param_value, current_time, parameters)
            
            if violation:
                violations.append(violation)
                self.consecutive_violations[rule_id] += 1
                
                # Check if consecutive violation limit exceeded
                if (self.consecutive_violations[rule_id] >= rule.consecutive_violations_limit and
                    rule.safety_level in [SafetyLevel.CRITICAL, SafetyLevel.HIGH]):
                    is_safe = False
                    
            else:
                # Reset consecutive violation counter
                self.consecutive_violations[rule_id] = 0
        
        # Predictive safety checks
        if self.predictive_monitoring:
            predictive_violations = self._predictive_safety_check(parameters)
            violations.extend(predictive_violations)
            
            if any(v.safety_level == SafetyLevel.CRITICAL for v in predictive_violations):
                is_safe = False
        
        # Update safety state
        self._update_safety_state(violations, is_safe)
        
        # Update compliance metrics
        self._update_compliance_metrics(violations)
        
        return is_safe, violations
    
    def _check_rule_violation(self, rule: SafetyRule, param_value: float, 
                            current_time: float, context: Dict) -> Optional[SafetyViolation]:
        """Check if a specific rule is violated"""
        
        violation = None
        
        # Check maximum value
        if rule.max_value is not None:
            if param_value > rule.max_value + rule.tolerance:
                severity = self._calculate_severity(param_value, rule.max_value, rule.safety_level)
                violation = SafetyViolation(
                    timestamp=current_time,
                    rule_id=rule.rule_id,
                    violation_type=rule.violation_type,
                    safety_level=rule.safety_level,
                    parameter=rule.parameter,
                    current_value=param_value,
                    limit_value=rule.max_value,
                    severity_score=severity,
                    description=f"{rule.description}: {param_value:.3f} > {rule.max_value:.3f}",
                    context=context.copy()
                )
        
        # Check minimum value
        elif rule.min_value is not None:
            if param_value < rule.min_value - rule.tolerance:
                severity = self._calculate_severity(param_value, rule.min_value, rule.safety_level)
                violation = SafetyViolation(
                    timestamp=current_time,
                    rule_id=rule.rule_id,
                    violation_type=rule.violation_type,
                    safety_level=rule.safety_level,
                    parameter=rule.parameter,
                    current_value=param_value,
                    limit_value=rule.min_value,
                    severity_score=severity,
                    description=f"{rule.description}: {param_value:.3f} < {rule.min_value:.3f}",
                    context=context.copy()
                )
        
        return violation
    
    def _calculate_severity(self, current_value: float, limit_value: float, 
                          safety_level: SafetyLevel) -> float:
        """Calculate violation severity score"""
        
        # Base severity from safety level
        level_multipliers = {
            SafetyLevel.CRITICAL: 1.0,
            SafetyLevel.HIGH: 0.8,
            SafetyLevel.MEDIUM: 0.6,
            SafetyLevel.LOW: 0.4,
            SafetyLevel.INFO: 0.2
        }
        
        base_severity = level_multipliers.get(safety_level, 0.5)
        
        # Severity based on how far from limit
        if limit_value != 0:
            deviation_ratio = abs(current_value - limit_value) / abs(limit_value)
        else:
            deviation_ratio = abs(current_value - limit_value)
        
        # Combine base severity with deviation
        total_severity = base_severity * (1.0 + deviation_ratio)
        
        return min(total_severity, 2.0)  # Cap at 2.0
    
    def _predictive_safety_check(self, parameters: Dict[str, Any]) -> List[SafetyViolation]:
        """Predictive safety monitoring based on parameter trends"""
        
        violations = []
        current_time = time.time()
        
        for param_name, history in self.parameter_history.items():
            if len(history) < self.trend_analysis_window:
                continue
                
            # Analyze parameter trend
            trend_analysis = self._analyze_parameter_trend(param_name, history)
            
            # Check if trend predicts future violations
            for rule_id, rule in self.safety_rules.items():
                if rule.parameter != param_name or not rule.enabled:
                    continue
                    
                # Predict future value based on trend
                predicted_violation = self._predict_rule_violation(rule, trend_analysis, current_time, parameters)
                
                if predicted_violation:
                    violations.append(predicted_violation)
        
        return violations
    
    def _analyze_parameter_trend(self, param_name: str, history: List[float]) -> Dict:
        """Analyze parameter trend for predictive monitoring"""
        
        if len(history) < 3:
            return {'trend': 'stable', 'slope': 0.0, 'acceleration': 0.0}
        
        # Simple linear trend analysis
        recent_values = history[-min(10, len(history)):]
        
        # Calculate slope (trend)
        if len(recent_values) > 1:
            x_vals = list(range(len(recent_values)))
            # Simple linear regression
            n = len(recent_values)
            sum_x = sum(x_vals)
            sum_y = sum(recent_values)
            sum_xy = sum(x * y for x, y in zip(x_vals, recent_values))
            sum_x2 = sum(x * x for x in x_vals)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                slope = 0.0
        else:
            slope = 0.0
        
        # Calculate acceleration (second derivative)
        if len(recent_values) >= 3:
            mid_point = len(recent_values) // 2
            first_half_avg = sum(recent_values[:mid_point]) / mid_point if mid_point > 0 else 0
            second_half_avg = sum(recent_values[mid_point:]) / (len(recent_values) - mid_point)
            acceleration = second_half_avg - first_half_avg
        else:
            acceleration = 0.0
        
        # Classify trend
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'acceleration': acceleration,
            'recent_std': (sum((x - sum(recent_values)/len(recent_values))**2 for x in recent_values) / len(recent_values))**0.5 if len(recent_values) > 1 else 0.0
        }
    
    def _predict_rule_violation(self, rule: SafetyRule, trend_analysis: Dict, 
                              current_time: float, context: Dict) -> Optional[SafetyViolation]:
        """Predict future rule violations based on trends"""
        
        if not self.parameter_history.get(rule.parameter):
            return None
        
        current_value = self.parameter_history[rule.parameter][-1]
        slope = trend_analysis['slope']
        
        # Prediction horizon (steps into future)
        prediction_horizon = 10
        
        # Predict future value
        predicted_value = current_value + slope * prediction_horizon
        
        # Check if predicted value will violate rule
        will_violate_max = (rule.max_value is not None and 
                           predicted_value > rule.max_value + rule.tolerance)
        will_violate_min = (rule.min_value is not None and 
                           predicted_value < rule.min_value - rule.tolerance)
        
        if will_violate_max or will_violate_min:
            # Only create warning for significant trends
            if abs(slope) > 0.01:
                limit_value = rule.max_value if will_violate_max else rule.min_value
                severity = self._calculate_severity(predicted_value, limit_value, SafetyLevel.MEDIUM)
                
                return SafetyViolation(
                    timestamp=current_time,
                    rule_id=f"{rule.rule_id}_predictive",
                    violation_type=rule.violation_type,
                    safety_level=SafetyLevel.MEDIUM,  # Predictive violations are medium priority
                    parameter=rule.parameter,
                    current_value=predicted_value,
                    limit_value=limit_value,
                    severity_score=severity,
                    description=f"Predictive violation for {rule.description}: trend suggests future violation",
                    context={'prediction_horizon': prediction_horizon, 'trend': trend_analysis}
                )
        
        return None
    
    def _update_parameter_history(self, param_name: str, value: float):
        """Update parameter history for trend analysis"""
        
        if param_name not in self.parameter_history:
            self.parameter_history[param_name] = []
        
        self.parameter_history[param_name].append(value)
        
        # Keep history manageable
        max_history = self.trend_analysis_window * 2
        if len(self.parameter_history[param_name]) > max_history:
            self.parameter_history[param_name].pop(0)
    
    def _update_safety_state(self, violations: List[SafetyViolation], is_safe: bool):
        """Update overall safety state"""
        
        # Count violations by severity
        critical_count = sum(1 for v in violations if v.safety_level == SafetyLevel.CRITICAL)
        high_count = sum(1 for v in violations if v.safety_level == SafetyLevel.HIGH)
        
        self.safety_state['critical_count'] += critical_count
        self.safety_state['warning_count'] += high_count
        self.safety_state['active_violations'] = [v.rule_id for v in violations]
        
        if violations:
            self.safety_state['last_violation_time'] = time.time()
        
        # Determine overall status
        if critical_count > 0:
            self.safety_state['overall_status'] = 'critical'
        elif high_count > 0:
            self.safety_state['overall_status'] = 'warning'
        elif violations:
            self.safety_state['overall_status'] = 'minor_issues'
        else:
            self.safety_state['overall_status'] = 'safe'
    
    def _update_compliance_metrics(self, violations: List[SafetyViolation]):
        """Update safety compliance metrics"""
        
        if violations:
            self.safety_metrics['violation_count'] += len(violations)
        
        # Calculate compliance rate
        if self.safety_metrics['total_checks'] > 0:
            self.safety_metrics['compliance_rate'] = (
                1.0 - self.safety_metrics['violation_count'] / self.safety_metrics['total_checks']
            )
        
        # Calculate safety score (weighted by violation severity)
        if violations:
            total_severity = sum(v.severity_score for v in violations)
            safety_impact = min(total_severity / 10.0, 0.5)  # Cap impact at 0.5
            self.safety_metrics['safety_score'] = max(0.0, self.safety_metrics['safety_score'] - safety_impact)
        else:
            # Gradual recovery
            self.safety_metrics['safety_score'] = min(1.0, self.safety_metrics['safety_score'] + 0.001)
    
    def get_safety_report(self) -> Dict:
        """Generate comprehensive safety report"""
        
        # Recent violations (last 100)
        recent_violations = self.violation_history[-100:] if len(self.violation_history) >= 100 else self.violation_history
        
        # Violation analysis
        violation_by_type = {}
        violation_by_level = {}
        
        for violation in recent_violations:
            # Group by type
            vtype = violation.violation_type.value
            if vtype not in violation_by_type:
                violation_by_type[vtype] = 0
            violation_by_type[vtype] += 1
            
            # Group by level
            vlevel = violation.safety_level.value
            if vlevel not in violation_by_level:
                violation_by_level[vlevel] = 0
            violation_by_level[vlevel] += 1
        
        # Safety rule status
        rule_status = {}
        for rule_id, rule in self.safety_rules.items():
            rule_status[rule_id] = {
                'enabled': rule.enabled,
                'consecutive_violations': self.consecutive_violations.get(rule_id, 0),
                'safety_level': rule.safety_level.value,
                'description': rule.description
            }
        
        report = {
            'timestamp': time.time(),
            'overall_safety_state': self.safety_state,
            'safety_metrics': self.safety_metrics,
            'recent_violations_summary': {
                'total_recent_violations': len(recent_violations),
                'by_type': violation_by_type,
                'by_level': violation_by_level
            },
            'safety_rules_status': rule_status,
            'recommendations': self._generate_safety_recommendations()
        }
        
        return report
    
    def _generate_safety_recommendations(self) -> List[str]:
        """Generate safety improvement recommendations"""
        
        recommendations = []
        
        # Analyze compliance rate
        if self.safety_metrics['compliance_rate'] < 0.99:
            recommendations.append(f"Compliance rate ({self.safety_metrics['compliance_rate']:.3f}) below target (0.99)")
        
        # Analyze frequent violations
        frequent_violators = {rule_id: count for rule_id, count in self.consecutive_violations.items() if count > 5}
        if frequent_violators:
            recommendations.append(f"Frequent violations detected: {list(frequent_violators.keys())}")
        
        # Analyze safety score
        if self.safety_metrics['safety_score'] < 0.95:
            recommendations.append(f"Safety score ({self.safety_metrics['safety_score']:.3f}) below optimal level")
        
        # Check critical status
        if self.safety_state['overall_status'] == 'critical':
            recommendations.append("System in critical safety state - immediate attention required")
        
        # Parameter trend analysis
        for param_name, history in self.parameter_history.items():
            if len(history) >= 10:
                trend = self._analyze_parameter_trend(param_name, history)
                if abs(trend['slope']) > 0.05:
                    recommendations.append(f"Parameter {param_name} showing significant trend: {trend['trend']}")
        
        if not recommendations:
            recommendations.append("Safety system operating within acceptable parameters")
        
        return recommendations
    
    def emergency_shutdown_check(self) -> Tuple[bool, str]:
        """Check if emergency shutdown is required"""
        
        # Critical violation threshold
        critical_violations = sum(1 for count in self.consecutive_violations.values() if count >= 3)
        
        # Multiple critical systems failing
        if critical_violations >= 2:
            return True, "Multiple critical safety systems violated"
        
        # Single critical system with high consecutive violations
        max_consecutive = max(self.consecutive_violations.values()) if self.consecutive_violations else 0
        if max_consecutive >= 5:
            return True, f"Critical system violated {max_consecutive} times consecutively"
        
        # Safety score critically low
        if self.safety_metrics['safety_score'] < 0.5:
            return True, f"Safety score critically low: {self.safety_metrics['safety_score']:.3f}"
        
        return False, "No emergency shutdown required"

def enhance_safety_compliance(config: Dict, test_parameters: List[Dict]) -> Dict:
    """Main function to test and enhance safety compliance"""
    
    # Initialize enhanced safety manager
    safety_manager = EnhancedSafetyManager(config)
    
    # Test safety compliance with various parameter sets
    results = {
        'total_tests': len(test_parameters),
        'safety_violations': [],
        'compliance_rates': [],
        'safety_scores': [],
        'emergency_shutdowns': 0
    }
    
    for i, params in enumerate(test_parameters):
        # Check safety compliance
        is_safe, violations = safety_manager.check_safety_compliance(params)
        
        # Record violations
        if violations:
            results['safety_violations'].extend(violations)
        
        # Check for emergency shutdown
        shutdown_required, shutdown_reason = safety_manager.emergency_shutdown_check()
        if shutdown_required:
            results['emergency_shutdowns'] += 1
            
        # Record metrics
        compliance_rate = safety_manager.safety_metrics['compliance_rate']
        safety_score = safety_manager.safety_metrics['safety_score']
        
        results['compliance_rates'].append(compliance_rate)
        results['safety_scores'].append(safety_score)
    
    # Generate final report
    final_report = safety_manager.get_safety_report()
    results['final_safety_report'] = final_report
    results['final_compliance_rate'] = final_report['safety_metrics']['compliance_rate']
    results['improvement_achieved'] = final_report['safety_metrics']['compliance_rate'] >= 0.99
    
    return results

# Example usage
if __name__ == "__main__":
    # Enhanced safety configuration
    safety_config = {
        'enable_predictive_monitoring': True,
        'trend_analysis_window': 50,
        'emergency_shutdown_enabled': True
    }
    
    # Test parameter sets with various violation scenarios
    test_params = [
        {'temperature': 75.0, 'voltage': 5.0, 'current': 2.0, 'pressure': 1.5},  # Safe
        {'temperature': 82.0, 'voltage': 5.0, 'current': 2.0, 'pressure': 1.5},  # High temp warning
        {'temperature': 87.0, 'voltage': 5.0, 'current': 2.0, 'pressure': 1.5},  # Critical temp
        {'temperature': 75.0, 'voltage': 6.0, 'current': 2.0, 'pressure': 1.5},  # Voltage violation
        {'temperature': 75.0, 'voltage': 5.0, 'current': 3.5, 'pressure': 1.5},  # Current violation
    ]
    
    # Run safety compliance enhancement
    results = enhance_safety_compliance(safety_config, test_params)
    
    print("Enhanced Safety Compliance Results:")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Final Compliance Rate: {results['final_compliance_rate']:.4f}")
    print(f"Target Achievement (≥99%): {results['improvement_achieved']}")
    print(f"Emergency Shutdowns: {results['emergency_shutdowns']}")
    print(f"Total Violations Detected: {len(results['safety_violations'])}")
