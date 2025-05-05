"""
Gemeinsame Test-Fixtures und -Utilities für das Pizza-Erkennungssystem.
"""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Generator, Dict

from src.types import ModelConfig
from src.constants import (
    INPUT_SIZE, NUM_CLASSES, CLASS_NAMES,
    DATA_DIR, MODELS_DIR, OUTPUT_DIR
)

@pytest.fixture
def test_config() -> ModelConfig:
    """Liefert eine Test-Modellkonfiguration."""
    return {
        'input_size': INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 2
    }

@pytest.fixture
def test_image() -> torch.Tensor:
    """Erzeugt ein Test-Bildtensor."""
    return torch.randn(3, INPUT_SIZE, INPUT_SIZE)

@pytest.fixture
def test_batch() -> torch.Tensor:
    """Erzeugt einen Test-Bildbatch."""
    return torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)

@pytest.fixture
def test_labels() -> torch.Tensor:
    """Erzeugt Test-Labels."""
    return torch.randint(0, NUM_CLASSES, (4,))

@pytest.fixture
def temp_output_dir(tmpdir) -> Generator[Path, None, None]:
    """Erstellt ein temporäres Ausgabeverzeichnis."""
    temp_dir = Path(tmpdir) / "test_output"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir

@pytest.fixture
def mock_metrics() -> Dict:
    """Erzeugt Mock-Metriken für Tests."""
    return {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.82,
        'f1_score': 0.825,
        'confusion_matrix': np.random.randint(0, 10, size=(NUM_CLASSES, NUM_CLASSES))
    }

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, temp_output_dir):
    """Konfiguriert die Testumgebung."""
    # Setze temporäre Ausgabeverzeichnisse
    monkeypatch.setattr('src.constants.OUTPUT_DIR', temp_output_dir)
    
    # Stelle sicher, dass CUDA nicht verwendet wird
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    
    # Setze einen festen Random-Seed für Reproduzierbarkeit
    torch.manual_seed(42)
    np.random.seed(42)

def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5) -> None:
    """Vergleicht zwei Tensoren auf Gleichheit."""
    if not torch.allclose(a, b, rtol=rtol):
        raise AssertionError(f"Tensoren sind nicht gleich:\na = {a}\nb = {b}")

def assert_metrics_valid(metrics: Dict) -> None:
    """Überprüft, ob Metriken gültig sind."""
    required_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
    for key in required_keys:
        assert key in metrics, f"Metrik {key} fehlt"
        
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        value = metrics[key]
        assert 0 <= value <= 1, f"{key} außerhalb des gültigen Bereichs: {value}"
    
    conf_matrix = metrics['confusion_matrix']
    assert conf_matrix.shape == (NUM_CLASSES, NUM_CLASSES), \
        f"Ungültige Form der Konfusionsmatrix: {conf_matrix.shape}"