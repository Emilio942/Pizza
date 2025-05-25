"""
Unit-Tests für das Visualisierungsmodul.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict

from src.visualization.visualization import (
    plot_inference_result,
    plot_confusion_matrix,
    plot_training_progress,
    plot_resource_usage,
    plot_power_profile,
    visualize_model_architecture,
    create_report,
    annotate_image
)
from src.utils.types import (
    InferenceResult,
    ResourceUsage,
    PowerProfile
)
from src.constants import DEFAULT_CLASSES as CLASS_NAMES

@pytest.fixture
def sample_image():
    """Beispielbild für Tests."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

@pytest.fixture
def sample_inference_result():
    """Beispiel-Inferenzergebnis."""
    return InferenceResult(
        class_name=CLASS_NAMES[0],
        confidence=0.95,
        probabilities={name: 0.1 for name in CLASS_NAMES},
        prediction=0  # Hinzugefügt: Klassen-ID als Integer
    )

@pytest.fixture
def temp_output_dir():
    """Temporäres Ausgabeverzeichnis."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_plot_inference_result(
    sample_image,
    sample_inference_result,
    temp_output_dir
):
    """Testet Inferenzvisualisierung."""
    output_path = temp_output_dir / 'inference.png'
    
    # Test: Speichern in Datei
    plot_inference_result(
        sample_image,
        sample_inference_result,
        output_path
    )
    assert output_path.exists()
    
    # Test: Anzeige (sollte keine Fehler werfen)
    plt.switch_backend('Agg')  # Nicht-interaktiver Backend für Tests
    plot_inference_result(sample_image, sample_inference_result)

def test_plot_confusion_matrix(temp_output_dir):
    """Testet Konfusionsmatrix-Visualisierung."""
    confusion_matrix = np.random.randint(0, 10, (len(CLASS_NAMES), len(CLASS_NAMES)))
    output_path = temp_output_dir / 'confusion.png'
    
    # Test: Speichern in Datei
    plot_confusion_matrix(confusion_matrix, output_path)
    assert output_path.exists()
    
    # Test: Anzeige
    plt.switch_backend('Agg')
    plot_confusion_matrix(confusion_matrix)

def test_plot_training_progress(temp_output_dir):
    """Testet Trainingsfortschritt-Visualisierung."""
    epochs = list(range(10))
    losses = [float(x) for x in np.random.rand(10)]
    metrics: Dict[str, list] = {
        'accuracy': [float(x) for x in np.random.rand(10)],
        'precision': [float(x) for x in np.random.rand(10)]
    }
    output_path = temp_output_dir / 'training.png'
    
    # Test: Speichern in Datei
    plot_training_progress(epochs, losses, metrics, output_path)
    assert output_path.exists()
    
    # Test: Anzeige
    plt.switch_backend('Agg')
    plot_training_progress(epochs, losses, metrics)

def test_plot_resource_usage(temp_output_dir):
    """Testet Ressourcennutzungs-Visualisierung."""
    resource_history = [
        ResourceUsage(
            ram_used_kb=float(r),
            flash_used_kb=1024.0,
            cpu_usage_percent=float(c),
            power_mw=float(p)
        )
        for r, c, p in zip(
            np.random.rand(10) * 100,
            np.random.rand(10) * 100,
            np.random.rand(10) * 200
        )
    ]
    output_path = temp_output_dir / 'resources.png'
    
    # Test: Speichern in Datei
    plot_resource_usage(resource_history, output_path)
    assert output_path.exists()
    
    # Test: Anzeige
    plt.switch_backend('Agg')
    plot_resource_usage(resource_history)

def test_plot_power_profile(temp_output_dir):
    """Testet Energieprofil-Visualisierung."""
    profile = PowerProfile(
        average_current_ma=50.0,
        peak_current_ma=100.0,
        total_energy_mj=1000.0,
        duration_ms=100.0
    )
    output_path = temp_output_dir / 'power.png'
    
    # Test: Speichern in Datei
    plot_power_profile(profile, output_path)
    assert output_path.exists()
    
    # Test: Anzeige
    plt.switch_backend('Agg')
    plot_power_profile(profile)

def test_visualize_model_architecture(temp_output_dir):
    """Testet Modellarchitektur-Visualisierung."""
    # Einfaches Testmodell
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 31 * 31, 10)
        
        def forward(self, x):
            x = self.pool(self.conv(x))
            x = x.view(-1, 16 * 31 * 31)
            return self.fc(x)
    
    model = SimpleModel()
    input_shape = (3, 64, 64)
    output_path = temp_output_dir / 'model'
    
    # Test: Speichern in Datei - überspringe Assertion wenn Graphviz nicht verfügbar ist
    try:
        visualize_model_architecture(model, input_shape, output_path)
        if not (output_path.parent / (output_path.name + '.png')).exists():
            pytest.skip("Graphviz (dot) nicht auf dem Pfad, überspringe Test")
    except ImportError:
        pytest.skip("torchviz nicht installiert")

def test_create_report(temp_output_dir):
    """Testet Berichterstellung."""
    # Beispieldaten
    from src.types import ModelMetrics
    
    # Erstelle ModelMetrics-Instanz statt eines Dictionaries
    model_metrics = ModelMetrics(
        accuracy=0.95,
        precision=0.93,
        recall=0.94,
        f1_score=0.935,
        confusion_matrix=np.random.randint(0, 10, (len(CLASS_NAMES), len(CLASS_NAMES)))
    )
    
    resource_usage = [
        ResourceUsage(
            ram_used_kb=float(r),
            flash_used_kb=1024.0,
            cpu_usage_percent=float(c),
            power_mw=float(p)
        )
        for r, c, p in zip(
            np.random.rand(10) * 100,
            np.random.rand(10) * 100,
            np.random.rand(10) * 200
        )
    ]
    
    power_profile = PowerProfile(
        average_current_ma=50.0,
        peak_current_ma=100.0,
        total_energy_mj=1000.0,
        duration_ms=100.0
    )
    
    # Test: Berichterstellung
    create_report(model_metrics, resource_usage, power_profile, temp_output_dir)
    
    # Prüfe ob alle erwarteten Dateien existieren
    report_dirs = list(temp_output_dir.glob('report_*'))
    assert len(report_dirs) == 1
    report_dir = report_dirs[0]
    
    assert (report_dir / 'confusion_matrix.png').exists()
    assert (report_dir / 'resource_usage.png').exists()
    assert (report_dir / 'power_profile.png').exists()
    assert (report_dir / 'report.html').exists()

def test_annotate_image(sample_image, sample_inference_result):
    """Testet Bildannotation."""
    # Test: Mit Konfidenz
    annotated = annotate_image(
        sample_image,
        sample_inference_result,
        draw_confidence=True
    )
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == sample_image.shape
    
    # Test: Ohne Konfidenz
    annotated = annotate_image(
        sample_image,
        sample_inference_result,
        draw_confidence=False
    )
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == sample_image.shape
    
    # Test: Verschiedene Bildgrößen
    for size in [(32, 32), (128, 128), (224, 224)]:
        test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        annotated = annotate_image(test_image, sample_inference_result)
        assert annotated.shape == test_image.shape