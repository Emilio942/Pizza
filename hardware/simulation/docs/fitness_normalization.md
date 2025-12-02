# Fitness-Normalisierung Dokumentation

## Übersicht

Das PG-ES System verwendet verschiedene Fitness-Normalisierungsstrategien, um die Optimierung zu stabilisieren und faire Vergleiche zwischen PG und ES zu ermöglichen.

## 1. Fitness-Rohdaten

### Reward Function (für PG)
```python
# Implementiert in hardware_simulator.py::_calculate_reward()

# Positive Komponenten:
operating_range_bonus = 1.0 if in_normal_range else 0.0
efficiency_bonus = (voltage * current) / power_dissipation * 2.0
longevity_bonus = degradation_factor * 2.0

# Negative Komponenten:
safety_penalty = -100.0 * len(safety_violations)
thermal_stress_penalty = -0.5 * max(0, temperature - 70.0)
gradient_penalty = -0.1 * max(0, thermal_gradient - 10.0)
degradation_penalty = -(1.0 - degradation_factor) * 10.0

# Finale Reward:
reward = sum(positive_components) - sum(negative_components)
# Typischer Bereich: [-50, +8]
```

### Fitness Function (für ES)
```python
# ES verwendet die gleiche Reward-Funktion, aber über ganze Episoden aggregiert
fitness = sum(episode_rewards) / episode_length
# Typischer Bereich: [-20, +5] (gemittelt über Episode)
```

## 2. Normalisierung-Strategien

### 2.1 Standard-Normalisierung (Z-Score)
```python
def standard_normalize(values: List[float]) -> List[float]:
    """Standard-Normalisierung: (x - μ) / σ"""
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    
    if std < 1e-8:  # Avoid division by zero
        return [0.0] * len(values)
    
    return [(x - mean) / std for x in values]

# Anwendung:
# - PG Advantages: Standardmäßig normalisiert
# - ES Fitness bei hoher Varianz
```

### 2.2 Rank-Based Normalisierung (für ES)
```python
def rank_normalize(fitness_values: List[float]) -> List[float]:
    """
    Rank-basierte Normalisierung wie in OpenAI ES
    Reduziert Sensitivität auf Fitness-Outliers
    """
    n = len(fitness_values)
    if n <= 1:
        return [0.0] * n
    
    # Sortiere Indices nach Fitness
    sorted_indices = sorted(range(n), key=lambda i: fitness_values[i])
    
    # Erstelle Rank-basierte Werte
    ranks = [0.0] * n
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank / (n - 1) - 0.5  # [-0.5, +0.5]
    
    return ranks

# Vorteile:
# - Robust gegen Outliers
# - Gleiche Skalierung unabhängig von Fitness-Range
# - Bevorzugt relative Ordnung über absolute Werte
```

### 2.3 Min-Max Normalisierung
```python
def minmax_normalize(values: List[float], target_range=(0, 1)) -> List[float]:
    """Min-Max Normalisierung auf Zielbereich"""
    if len(values) <= 1:
        return values
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val - min_val < 1e-8:
        return [target_range[0]] * len(values)
    
    range_size = target_range[1] - target_range[0]
    return [target_range[0] + (x - min_val) / (max_val - min_val) * range_size 
            for x in values]

# Anwendung:
# - Supervisor-Dashboard Visualisierungen
# - Vergleiche zwischen PG und ES Performance
```

### 2.4 Adaptive Normalisierung
```python
def adaptive_normalize(values: List[float], history: List[List[float]]) -> List[float]:
    """
    Adaptive Normalisierung basierend auf historischen Daten
    Verhindert plötzliche Skalierungsänderungen
    """
    # Kombiniere aktuelle Werte mit historischen für stabilere Statistiken
    all_values = []
    for hist_batch in history[-10:]:  # Letzten 10 Batches
        all_values.extend(hist_batch)
    all_values.extend(values)
    
    if len(all_values) < 2:
        return values
    
    # Berechne robuste Statistiken
    mean = sum(all_values) / len(all_values)
    std = (sum((x - mean) ** 2 for x in all_values) / len(all_values)) ** 0.5
    
    # Normalisiere nur aktuelle Werte
    if std < 1e-8:
        return [0.0] * len(values)
    
    return [(x - mean) / std for x in values]
```

## 3. Implementierung im System

### 3.1 PG-Fitness-Normalisierung
```python
# In pg_optimizer.py::update_policy()

def _calculate_advantages(self, rewards, values, dones):
    """Berechne normalisierte Advantages"""
    
    # 1. Berechne TD-Errors
    td_errors = self._compute_td_errors(rewards, values, dones)
    
    # 2. Standard-Normalisierung
    advantages = standard_normalize(td_errors)
    
    # 3. Clipping für Stabilität
    advantages = [max(-10.0, min(10.0, adv)) for adv in advantages]
    
    return advantages
```

### 3.2 ES-Fitness-Normalisierung
```python
# In es_optimizer.py::update_population()

def _normalize_fitness(self, fitness_values):
    """Normalisiere Fitness-Werte für ES-Update"""
    
    if self.config.fitness_shaping:
        # Rank-basierte Normalisierung (Standard für ES)
        return rank_normalize(fitness_values)
    else:
        # Standard-Normalisierung
        return standard_normalize(fitness_values)
```

### 3.3 Supervisor-Normalisierung
```python
# In supervisor.py für Dashboard-Anzeige

def _normalize_for_display(self, pg_metrics, es_metrics):
    """Normalisiere Metriken für Supervisor-Dashboard"""
    
    # PG-Metriken auf [0, 1] skalieren
    pg_normalized = {
        'loss': minmax_normalize([pg_metrics['loss']], (0, 1))[0],
        'reward': minmax_normalize([pg_metrics.get('reward', 0)], (-1, 1))[0]
    }
    
    # ES-Metriken auf [0, 1] skalieren
    es_normalized = {
        'fitness': minmax_normalize([es_metrics['fitness']], (-1, 1))[0],
        'diversity': minmax_normalize([es_metrics['population_diversity']], (0, 1))[0]
    }
    
    return pg_normalized, es_normalized
```

## 4. Konfiguration und Parameter

### 4.1 Normalisierung-Konfiguration
```python
@dataclass
class NormalizationConfig:
    """Konfiguration für Fitness-Normalisierung"""
    
    # PG-Normalisierung
    pg_advantage_normalization: str = "standard"  # standard, adaptive, none
    pg_advantage_clipping: float = 10.0
    
    # ES-Normalisierung  
    es_fitness_shaping: bool = True  # Rank-based wenn True
    es_fitness_clipping: float = 100.0
    
    # Supervisor-Normalisierung
    supervisor_display_range: Tuple[float, float] = (0.0, 1.0)
    supervisor_history_window: int = 1000
    
    # Adaptive Parameter
    adaptive_history_length: int = 10
    adaptive_min_samples: int = 50
```

## 5. Monitoring und Validierung

### 5.1 Normalisierung-Qualität prüfen
```python
def validate_normalization(original_values, normalized_values):
    """Validiere Normalisierung-Qualität"""
    
    checks = {
        'no_nans': not any(x != x for x in normalized_values),  # NaN check
        'no_infs': not any(abs(x) == float('inf') for x in normalized_values),
        'reasonable_range': all(-100 < x < 100 for x in normalized_values),
        'preserved_order': True  # TODO: Implementiere Order-Preservation-Check
    }
    
    return checks
```

### 5.2 Supervisor-Monitoring
```python
# Normalisierung-Alerts im Supervisor-System

def _check_normalization_health(self, pg_advantages, es_fitness):
    """Überwache Normalisierung-Gesundheit"""
    
    alerts = []
    
    # Prüfe PG-Advantages
    if pg_advantages:
        adv_std = np.std(pg_advantages)
        if adv_std < 0.01:
            alerts.append("PG advantages haben sehr niedrige Varianz")
        elif adv_std > 10.0:
            alerts.append("PG advantages haben sehr hohe Varianz")
    
    # Prüfe ES-Fitness
    if es_fitness:
        if any(abs(f) > 1000 for f in es_fitness):
            alerts.append("ES fitness enthält extreme Werte")
    
    return alerts
```

## 6. Best Practices

### 6.1 Wann welche Normalisierung?

1. **Standard-Normalisierung:**
   - PG Advantages (Standard)
   - Kleine Batch-Größen
   - Stabile Fitness-Landschaft

2. **Rank-basierte Normalisierung:**
   - ES Fitness (Standard) 
   - Große Population
   - Noisy/unstabile Fitness

3. **Min-Max Normalisierung:**
   - Supervisor-Dashboards
   - Visualisierungen
   - Vergleiche zwischen Algorithmen

4. **Adaptive Normalisierung:**
   - Lange Training-Läufe
   - Sich ändernde Fitness-Landscape
   - Online-Learning Szenarien

### 6.2 Häufige Probleme

1. **Division durch Null:** Immer auf σ < ε prüfen
2. **Extreme Outliers:** Clipping verwenden
3. **Instabile Normalisierung:** Adaptive Methoden mit History
4. **Skalierungs-Sprünge:** Glättung über mehrere Batches

### 6.3 Debugging

```python
# Normalisierung-Debug-Ausgaben
def debug_normalization(original, normalized, method="unknown"):
    """Debug-Informationen für Normalisierung"""
    
    print(f"=== Normalization Debug ({method}) ===")
    print(f"Original: min={min(original):.4f}, max={max(original):.4f}, mean={np.mean(original):.4f}")
    print(f"Normalized: min={min(normalized):.4f}, max={max(normalized):.4f}, mean={np.mean(normalized):.4f}")
    print(f"NaN count: {sum(1 for x in normalized if x != x)}")
    print(f"Inf count: {sum(1 for x in normalized if abs(x) == float('inf'))}")
```

## 7. Integration mit Logging-System

Alle Normalisierungs-Schritte werden automatisch im Logging-System erfasst:

- **Pre-/Post-Normalisierung Werte** für Debugging
- **Normalisierung-Parameter** (μ, σ, min, max)
- **Qualitäts-Metriken** der Normalisierung
- **Supervisor-Alerts** bei problematischer Normalisierung

Dies ermöglicht vollständige Nachverfolgbarkeit und Debugging der Normalisierung-Pipeline.
