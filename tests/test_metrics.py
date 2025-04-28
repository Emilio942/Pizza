"""
Unit-Tests für das Metriken-Modul.
"""

import numpy as np
import pytest
from pathlib import Path

from src.metrics import (
    calculate_metrics,
    save_metrics,
    load_metrics,
    format_inference_result,
    get_error_analysis
)
from src.types import InferenceResult, Metrics
from src.constants import CLASS_NAMES
from tests.conftest import assert_metrics_valid

def test_calculate_metrics():
    """Testet die Metrikberechnung."""
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 2])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert_metrics_valid(metrics)
    
    # Überprüfe spezifische Werte
    assert metrics['accuracy'] == 0.6  # 3/5 korrekt
    assert metrics['confusion_matrix'].sum() == len(y_true)

def test_save_load_metrics(temp_output_dir, mock_metrics):
    """Testet das Speichern und Laden von Metriken."""
    model_name = "test_model"
    
    # Speichere Metriken
    save_path = save_metrics(mock_metrics, model_name)
    assert save_path.exists()
    
    # Lade Metriken
    loaded_metrics = load_metrics(model_name)
    
    # Vergleiche Original- mit geladenen Metriken
    for key in mock_metrics:
        if key == 'confusion_matrix':
            np.testing.assert_array_equal(
                mock_metrics[key],
                loaded_metrics[key]
            )
        else:
            assert mock_metrics[key] == loaded_metrics[key]

def test_format_inference_result():
    """Testet die Formatierung von Inferenzergebnissen."""
    # Erstelle Dummy-Logits
    logits = torch.tensor([1.0, -1.0, 0.5])
    result = format_inference_result(logits)
    
    assert isinstance(result, InferenceResult)
    assert 0 <= result.confidence <= 1
    assert result.prediction in range(len(result.probabilities))
    assert sum(result.probabilities.values()) == pytest.approx(1.0)

def test_get_error_analysis(mock_metrics):
    """Testet die Fehleranalyse."""
    error_analysis = get_error_analysis(mock_metrics)
    
    for class_name, stats in error_analysis.items():
        # Prüfe Struktur
        assert all(key in stats for key in 
                  ['true_positives', 'false_positives', 'false_negatives', 'error_rate'])
        
        # Prüfe Wertebereich
        assert stats['error_rate'] >= 0 and stats['error_rate'] <= 1
        assert all(stats[k] >= 0 for k in ['true_positives', 'false_positives', 'false_negatives'])

def test_error_cases():
    """Testet Fehlerfälle."""
    # Test mit leeren Arrays
    with pytest.raises(ValueError):
        calculate_metrics(np.array([]), np.array([]))
    
    # Test mit unterschiedlichen Array-Längen
    with pytest.raises(ValueError):
        calculate_metrics(np.array([0, 1]), np.array([0]))
    
    # Test mit ungültigen Klassenlabels
    with pytest.raises(ValueError):
        calculate_metrics(np.array([0, 1]), np.array([0, 10]))  # 10 ist keine gültige Klasse