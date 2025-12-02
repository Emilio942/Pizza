# Physikalische Simulationsparameter für PCV-Hardware

## Hardware-Kontext: PizzaBoard-RP2040
Basierend auf dem KiCad-Design im Projekt:
- Mikrocontroller: RP2040 
- Betriebsspannung: 3.3V (vermutlich)
- Temperaturbereich: -40°C bis +85°C (typisch für embedded systems)

## 1. Temperatur-Parameter

### Betriebstemperatur
- **Normal Range:** 20°C - 60°C
- **Stress Range:** -10°C - 80°C  
- **Critical Limit:** 85°C (Hardware-Damage-Threshold)
- **Thermal Time Constant:** τ = 30s (geschätzt für PCB-Masse)

### Temperatur-Dynamik
```python
# Thermal Model Parameters
THERMAL_MASS = 0.05  # kg equivalent
SPECIFIC_HEAT = 900  # J/(kg*K) für PCB-Material
HEAT_TRANSFER_COEFF = 10  # W/(m²*K) natural convection
SURFACE_AREA = 0.01  # m² PCB surface
```

### ES-Optimierbare Thermal-Parameter
- Kühlstrategie-Parameter (PWM für Lüfter, falls vorhanden)
- Thermal throttling thresholds
- Duty cycle modulation bei Überhitzung

## 2. Feuchtigkeits-Parameter

### Relative Luftfeuchtigkeit
- **Normal Range:** 30% - 70% RH
- **Stress Range:** 10% - 90% RH
- **Critical Limit:** 95% RH (Kondensationsrisiko)

### Korrosions-Modell
```python
# Humidity Degradation Model
CORROSION_RATE_BASE = 1e-9  # mm/year at 50% RH
HUMIDITY_EXPONENTIAL = 2.5  # exponential factor
SALT_CONTAMINATION = 0.1   # relative contamination level
```

### ES-Optimierbare Humidity-Parameter
- Conformal coating effectiveness
- Drainage/ventilation strategies
- Sensor calibration drift compensation

## 3. Materialermüdung-Parameter

### Solder Joint Fatigue
- **Fatigue Cycles:** N = 10^6 cycles bei ±20°C
- **Crack Propagation Rate:** da/dN = C(ΔK)^m
- **Paris Law Parameters:** C = 1e-12, m = 3.0

### Component Stress
```python
# Mechanical Stress Model
YOUNGS_MODULUS = 20e9  # Pa for PCB substrate
THERMAL_EXPANSION = 15e-6  # /K CTE mismatch
FATIGUE_LIMIT = 1e6  # cycles
STRESS_CONCENTRATION = 2.5  # at solder joints
```

### ES-Optimierbare Fatigue-Parameter  
- Power cycling frequency optimization
- Thermal gradient minimization
- Component placement stress reduction

## 4. Elektrische Parameter

### Power Supply Variations
- **Nominal:** 3.3V ± 1%
- **Stress Range:** 3.0V - 3.6V
- **Ripple:** < 50mV pp
- **Transient Response:** < 100μs settling

### Current Consumption
```python
# Power Model Parameters
IDLE_CURRENT = 20e-3  # A
ACTIVE_CURRENT = 200e-3  # A  
PEAK_CURRENT = 500e-3  # A (short bursts)
EFFICIENCY = 0.85  # power conversion efficiency
```

### ES-Optimierbare Power-Parameter
- Dynamic voltage scaling
- Sleep mode optimization
- Load balancing strategies

## 5. Safety-Bereiche (Hard Constraints)

### Critical Limits - Never Exceed
```python
SAFETY_LIMITS = {
    'max_temperature': 85.0,  # °C
    'min_temperature': -40.0,  # °C
    'max_voltage': 3.6,       # V
    'min_voltage': 2.7,       # V
    'max_current': 600e-3,    # A
    'max_humidity': 95.0,     # % RH
    'max_thermal_gradient': 30.0,  # °C/cm
    'max_power_dissipation': 2.0,  # W
}
```

## 6. PG vs ES Parameter-Zuordnung

### PG-Optimierbare Parameter (Gradient-based)
- [ ] Control loop gains (PID parameters)
- [ ] Neural network weights für adaptive control
- [ ] Sensor fusion coefficients
- [ ] Predictive model parameters

### ES-Optimierbare Parameter (Black-box)
- [ ] Hardware configuration switches
- [ ] Discrete operating modes
- [ ] Safety threshold values  
- [ ] Noise parameters für Domain Randomization
- [ ] Meta-parameters (learning rates, exploration noise)

## 7. Domain Randomization Setup

### Environmental Variations
```python
DOMAIN_RANDOMIZATION = {
    'temperature_noise': (0.0, 2.0),      # °C std dev
    'humidity_noise': (0.0, 5.0),         # % RH std dev  
    'voltage_noise': (0.0, 0.05),         # V std dev
    'component_tolerance': (0.95, 1.05),  # multiplicative factor
    'aging_factor': (1.0, 1.2),           # degradation multiplier
    'environmental_disturbance': True,     # random heat/cool events
}
```

### Manufacturing Variations
- Component value tolerances (1%, 5%)
- PCB thickness variations
- Solder joint quality variations
- Assembly alignment tolerances

## Implementation Status
- [x] Grundparameter definiert
- [x] Safety limits festgelegt
- [x] PG/ES Zuordnung begonnen
- [x] Domain Randomization konzipiert
- [ ] Parameter validation implementiert
- [ ] Physics engine integration
- [ ] Sensor noise models
- [ ] Aging/degradation models

## Next Steps
1. Implementierung der Physics simulation
2. Validierung gegen bekannte Hardware-Daten
3. Kalibrierung der Noise-Parameter
4. Integration mit PG/ES algorithms
