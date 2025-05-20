"""
Type definitions for the pizza detection system.
"""

# Use individual imports instead of bulk imports to avoid circular import issues
from typing import Dict as Dict
from typing import List as List
from typing import Tuple as Tuple
from typing import Optional as Optional
from typing import Union as Union
from typing import NamedTuple as NamedTuple
from typing import Callable as Callable
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# Avoid importing torch here as it causes circular import issues
# through other modules
# Define type aliases for hardware specs and configs first

# Hardware specifications
HardwareSpecs = Dict[str, Union[int, float]]

# Model configuration
ModelConfig = Dict[str, Union[int, float]]

# Inference result
@dataclass
class InferenceResult:
    """Result of a single inference."""
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    prediction: int  # Class ID as integer

# Metrics
@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[np.ndarray] = None  # Changed to NumPy array

# Training progress
class TrainingProgress(NamedTuple):
    """Progress during training."""
    epoch: int
    batch: int
    loss: float
    metrics: Dict[str, float]
    learning_rate: float

# Image preprocessing
# Use np.ndarray instead of torch.Tensor to avoid circular imports
ImageTransform = Callable[[np.ndarray], np.ndarray]

# Resource usage
@dataclass
class ResourceUsage:
    """Current resource usage."""
    ram_used_kb: float
    flash_used_kb: float
    cpu_usage_percent: float
    power_mw: float
    
    def _replace(self, **kwargs):
        """Erstellt eine neue Instanz mit aktualisierten Werten (NamedTuple-Kompatibilität)."""
        # Erstelle ein Dictionary mit den aktuellen Werten
        current_values = {
            'ram_used_kb': self.ram_used_kb,
            'flash_used_kb': self.flash_used_kb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'power_mw': self.power_mw
        }
        
        # Aktualisiere mit den neuen Werten
        current_values.update(kwargs)
        
        # Erstelle eine neue Instanz mit den aktualisierten Werten
        return ResourceUsage(**current_values)

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