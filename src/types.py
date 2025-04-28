"""
Typdefinitionen für das Pizzaerkennungssystem.
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Callable, NamedTuple
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass

# Hardware-Spezifikationen
HardwareSpecs = Dict[str, Union[int, float]]

# Modellkonfiguration
ModelConfig = Dict[str, Union[int, float]]

# Inferenz-Ergebnis
@dataclass
class InferenceResult:
    """Ergebnis einer einzelnen Inferenz."""
    class_name: str
    confidence: float
    probabilities: Dict[str, float]

# Metriken
@dataclass
class ModelMetrics:
    """Modellleistungsmetriken."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None

# Trainingsfortschritt
class TrainingProgress(NamedTuple):
    """Fortschritt während des Trainings."""
    epoch: int
    batch: int
    loss: float
    metrics: Dict[str, float]
    learning_rate: float

# Bildvorverarbeitung
ImageTransform = Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]

# Ressourcennutzung
@dataclass
class ResourceUsage:
    """Momentane Ressourcennutzung."""
    ram_used_kb: float
    flash_used_kb: float
    cpu_usage_percent: float
    power_mw: float

# Kamerakonfiguration
class CameraConfig(NamedTuple):
    """Kamerakonfiguration."""
    width: int
    height: int
    fps: int
    format: str
    exposure: float
    gain: float

# Modellquantisierung
class QuantizationParams(NamedTuple):
    """Parameter für Modellquantisierung."""
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    weight_scale: float
    activation_scale: float

# Fehleranalyse
class ErrorAnalysis(NamedTuple):
    """Detaillierte Fehleranalyse einer Klasse."""
    true_positives: int
    false_positives: int
    false_negatives: int
    error_rate: float

# Hardware-Status
class HardwareStatus(NamedTuple):
    """Aktueller Hardware-Status."""
    temperature_c: float
    voltage_mv: float
    current_ma: float
    power_mode: str
    error_count: int

# Speicherregion
class MemoryRegion(NamedTuple):
    """Beschreibt eine Speicherregion."""
    start_address: int
    size_bytes: int
    used_bytes: int
    type: str  # 'flash' oder 'ram'
    description: str

# Firmware-Information
class FirmwareInfo(NamedTuple):
    """Information über geladene Firmware."""
    version: str
    build_date: str
    size_bytes: int
    entry_point: int
    model_size_bytes: int
    crc32: int

# Debug-Information
class DebugInfo(NamedTuple):
    """Debug-Information für Entwicklung."""
    timestamp: float
    module: str
    function: str
    message: str
    stack_trace: Optional[str]

# Energieprofil
@dataclass
class PowerProfile:
    """Energieprofil einer Operation."""
    average_current_ma: float
    peak_current_ma: float
    total_energy_mj: float
    duration_ms: float

# Modellexport
class ExportConfig(NamedTuple):
    """Konfiguration für Modellexport."""
    target_platform: str
    quantize: bool
    optimize: bool
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

# Modelloptimierung
class OptimizationResult(NamedTuple):
    """Ergebnis einer Modelloptimierung."""
    original_size_kb: float
    optimized_size_kb: float
    speed_improvement: float
    accuracy_change: float
    energy_savings: float

# Modellarchitektur
class ArchitectureConfig(NamedTuple):
    """Konfiguration der Modellarchitektur."""
    num_layers: int
    layer_sizes: List[int]
    activation_functions: List[str]
    dropout_rates: List[float]
    use_batch_norm: bool