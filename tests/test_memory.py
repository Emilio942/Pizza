"""
Unit-Tests für das Speichermanagement-Modul.
"""

import pytest
import torch
import numpy as np
from typing import Dict

from src.memory import MemoryTracker, MemoryEstimator, analyze_memory_usage
from src.exceptions import HardwareError
from src.types import HardwareSpecs

class SimpleModel(torch.nn.Module):
    """Einfaches Testmodell."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.fc = torch.nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def test_memory_tracker():
    """Testet den MemoryTracker."""
    tracker = MemoryTracker()
    
    # Test Allokation
    tracker.allocate(1024, "buffer1")
    assert tracker.current_ram_usage == 1024
    assert tracker.peak_ram_usage == 1024
    
    # Test mehrfache Allokationen
    tracker.allocate(2048, "buffer2")
    assert tracker.current_ram_usage == 3072
    assert tracker.peak_ram_usage == 3072
    
    # Test Freigabe
    tracker.free()
    assert tracker.current_ram_usage == 1024
    assert tracker.peak_ram_usage == 3072  # Peak bleibt unverändert
    
    # Test Reset
    tracker.reset()
    assert tracker.current_ram_usage == 0
    assert tracker.peak_ram_usage == 0
    assert len(tracker._stack) == 0

def test_memory_estimator():
    """Testet den MemoryEstimator."""
    model = SimpleModel()
    
    # Test Modellgrößenschätzung
    model_size = MemoryEstimator.estimate_model_size(model)
    assert model_size > 0
    
    # Test RAM-Schätzung
    runtime_ram = MemoryEstimator.estimate_runtime_ram(model_size)
    assert runtime_ram > model_size  # RAM sollte größer sein als Modellgröße
    
    # Test mit verschiedenen Eingabegrößen
    ram_small = MemoryEstimator.estimate_runtime_ram(model_size, (32, 32))
    ram_large = MemoryEstimator.estimate_runtime_ram(model_size, (64, 64))
    assert ram_large > ram_small  # Größere Eingabe sollte mehr RAM benötigen

def test_memory_validation():
    """Testet die Speichervalidierung."""
    model = SimpleModel()
    
    # Test mit Standard-Limits
    try:
        results = MemoryEstimator.validate_memory_requirements(model)
        assert isinstance(results, dict)
        assert all(key in results for key in [
            'model_size_kb', 'runtime_ram_kb',
            'flash_usage_percent', 'ram_usage_percent'
        ])
    except HardwareError:
        pass  # Es ist ok wenn das Modell zu groß ist
    
    # Test mit benutzerdefinierten Limits
    custom_specs: HardwareSpecs = {
        'flash_size_kb': 1024,
        'ram_size_kb': 512,
        'clock_speed_mhz': 100,
        'max_model_size_kb': 100,
        'max_runtime_ram_kb': 50
    }
    
    # Dies sollte eine HardwareError auslösen
    with pytest.raises(HardwareError):
        MemoryEstimator.validate_memory_requirements(model, custom_specs)

def test_memory_analysis():
    """Testet die Speichernutzungsanalyse."""
    model = SimpleModel()
    sample_input = torch.randn(1, 3, 32, 32)
    
    report = analyze_memory_usage(model, sample_input)
    
    # Überprüfe Berichtsstruktur
    assert isinstance(report, dict)
    assert 'peak_ram_kb' in report
    assert 'activation_sizes_kb' in report
    assert 'total_activations_kb' in report
    
    # Überprüfe Werte
    assert report['peak_ram_kb'] > 0
    assert len(report['activation_sizes_kb']) > 0
    assert report['total_activations_kb'] > 0
    assert sum(report['activation_sizes_kb']) == report['total_activations_kb']

def test_edge_cases():
    """Testet Grenzfälle und Fehlerbedingungen."""
    tracker = MemoryTracker()
    
    # Test negative Allokation
    with pytest.raises(ValueError):
        tracker.allocate(-1024, "negative")
    
    # Test Freigabe ohne Allokation
    tracker.free()  # Sollte keine Exception werfen
    
    # Test Überlauf
    large_alloc = 2**31  # 2GB
    tracker.allocate(large_alloc, "huge")
    assert tracker.peak_ram_usage == large_alloc