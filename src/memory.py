"""
Speichermanagement und Ressourcenschätzung für RP2040-Deployment.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from .constants import (
    RP2040_FLASH_SIZE_KB,
    RP2040_RAM_SIZE_KB,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    INPUT_SIZE
)
from .exceptions import ResourceError, HardwareError
from .types import HardwareSpecs

class MemoryTracker:
    """Verfolgt Speichernutzung während der Inferenz."""
    
    def __init__(self):
        self.peak_ram_usage = 0
        self.current_ram_usage = 0
        self._stack = []
    
    def allocate(self, size_bytes: int, description: str) -> None:
        """Simuliert Speicherallokation."""
        # Überprüfe auf negative Größe
        if size_bytes < 0:
            raise ValueError(f"Negative Speicherallokation nicht erlaubt: {size_bytes} Bytes")
            
        self.current_ram_usage += size_bytes
        self.peak_ram_usage = max(self.peak_ram_usage, self.current_ram_usage)
        self._stack.append((size_bytes, description))
    
    def free(self) -> None:
        """Simuliert Speicherfreigabe."""
        if self._stack:
            size, _ = self._stack.pop()
            self.current_ram_usage -= size
    
    def get_report(self) -> Dict[str, int]:
        """Erstellt Speichernutzungsbericht."""
        return {
            'peak_ram_kb': self.peak_ram_usage // 1024,
            'current_ram_kb': self.current_ram_usage // 1024,
            'allocations': len(self._stack)
        }
    
    def reset(self) -> None:
        """Setzt Tracker zurück."""
        self.peak_ram_usage = 0
        self.current_ram_usage = 0
        self._stack.clear()

class MemoryEstimator:
    """Schätzt Speicherbedarf für RP2040-Deployment."""
    
    SYSTEM_OVERHEAD_KB = 40  # Geschätzter System-Overhead
    INFERENCE_BUFFER_KB = 20  # Zusätzlicher Puffer für Inferenz
    
    @staticmethod
    def estimate_model_size(model: torch.nn.Module, bits: int = 8) -> float:
        """Schätzt Modellgröße in KB."""
        total_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = bits / 8
        
        # Füge 10% Overhead für Modellstruktur hinzu
        total_bytes = total_params * bytes_per_param * 1.1
        return total_bytes / 1024  # Konvertiere zu KB
    
    @staticmethod
    def estimate_runtime_ram(
        model_size_kb: float,
        input_size: Tuple[int, int] = INPUT_SIZE,  # Geändert: INPUT_SIZE direkt verwenden
        rgb: bool = True
    ) -> float:
        """Schätzt RAM-Bedarf während der Inferenz."""
        channels = 3 if rgb else 1
        
        # Kamera-Framebuffer
        frame_buffer_kb = (CAMERA_WIDTH * CAMERA_HEIGHT * channels) / 1024
        
        # Vorverarbeitungs-Buffer
        preprocess_buffer_kb = (input_size[0] * input_size[1] * channels) / 1024
        
        # Aktivierungen (geschätzt als 2x Modellgröße)
        activations_kb = model_size_kb * 2
        
        # Gesamter RAM-Bedarf
        total_ram_kb = (
            frame_buffer_kb +
            preprocess_buffer_kb +
            activations_kb +
            MemoryEstimator.SYSTEM_OVERHEAD_KB +
            MemoryEstimator.INFERENCE_BUFFER_KB
        )
        
        return total_ram_kb
    
    @staticmethod
    def validate_memory_requirements(
        model: torch.nn.Module,
        hardware_specs: Optional[HardwareSpecs] = None
    ) -> Dict[str, float]:
        """Validiert Speicheranforderungen gegen Hardware-Limits."""
        if hardware_specs is None:
            hardware_specs = {
                'flash_size_kb': RP2040_FLASH_SIZE_KB,
                'ram_size_kb': RP2040_RAM_SIZE_KB,
                'clock_speed_mhz': 133,
                'max_model_size_kb': 180,
                'max_runtime_ram_kb': 100
            }
        
        model_size_kb = MemoryEstimator.estimate_model_size(model)
        runtime_ram_kb = MemoryEstimator.estimate_runtime_ram(model_size_kb)
        
        if model_size_kb > hardware_specs['max_model_size_kb']:
            raise HardwareError(
                f"Modell zu groß: {model_size_kb:.1f}KB > "
                f"{hardware_specs['max_model_size_kb']}KB"
            )
        
        if runtime_ram_kb > hardware_specs['max_runtime_ram_kb']:
            raise HardwareError(
                f"RAM-Bedarf zu hoch: {runtime_ram_kb:.1f}KB > "
                f"{hardware_specs['max_runtime_ram_kb']}KB"
            )
        
        return {
            'model_size_kb': model_size_kb,
            'runtime_ram_kb': runtime_ram_kb,
            'flash_usage_percent': (model_size_kb / hardware_specs['flash_size_kb']) * 100,
            'ram_usage_percent': (runtime_ram_kb / hardware_specs['ram_size_kb']) * 100
        }

def analyze_memory_usage(
    model: torch.nn.Module,
    sample_input: torch.Tensor
) -> Dict[str, float]:
    """Analysiert detaillierte Speichernutzung während der Inferenz."""
    tracker = MemoryTracker()
    
    # Verfolge Eingabe
    tracker.allocate(sample_input.nelement() * 4, "input_tensor")
    
    # Führe Inferenz durch und verfolge Aktivierungen
    with torch.no_grad():
        # Aktiviere Hook für Zwischenschichten
        activation_sizes = []
        hooks = []
        
        def hook(module, input, output):
            size = output.nelement() * 4  # float32 = 4 bytes
            activation_sizes.append(size)
            tracker.allocate(size, f"activation_{len(activation_sizes)}")
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook))
        
        # Führe Modellinferenz durch
        output = model(sample_input)
        
        # Entferne Hooks
        for h in hooks:
            h.remove()
    
    # Erstelle detaillierten Bericht
    report = tracker.get_report()
    report['activation_sizes_kb'] = [size/1024 for size in activation_sizes]
    report['total_activations_kb'] = sum(report['activation_sizes_kb'])
    
    return report