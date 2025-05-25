"""
Visualisierungsfunktionen für das Pizzaerkennungssystem.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import torch
import cv2
from datetime import datetime

from src.utils.types import (
    InferenceResult,
    ModelMetrics,
    ResourceUsage,
    PowerProfile
)
from src.constants import (
    DEFAULT_CLASSES as CLASS_NAMES,
    OUTPUT_DIR
)

# Define additional constants for visualization
COLOR_PALETTE = ["#FF5733", "#4CAF50", "#3498DB", "#9B59B6", "#F1C40F", "#E74C3C"]
CLASS_COLORS = {class_name: COLOR_PALETTE[i % len(COLOR_PALETTE)] 
                for i, class_name in enumerate(CLASS_NAMES)}
PLOT_DPI = 100
FIGURE_SIZE = (10, 6)

logger = logging.getLogger(__name__)

def plot_inference_result(
    image: np.ndarray,
    result: InferenceResult,
    output_path: Optional[Path] = None
) -> None:
    """Visualisiert ein Inferenzergebnis mit Wahrscheinlichkeiten."""
    plt.figure(figsize=(12, 6))
    
    # Bild anzeigen
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Erkannt: {result.class_name}\nKonfidenz: {result.confidence:.2%}')
    plt.axis('off')
    
    # Balkendiagramm der Wahrscheinlichkeiten
    plt.subplot(1, 2, 2)
    classes = list(result.probabilities.keys())
    probs = list(result.probabilities.values())
    colors = [CLASS_COLORS[c] for c in classes]
    
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, probs, color=[np.array(c)/255 for c in colors])
    plt.yticks(y_pos, classes)
    plt.xlabel('Wahrscheinlichkeit')
    plt.title('Klassenwahrscheinlichkeiten')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Visualisierung gespeichert unter {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Erstellt eine Konfusionsmatrix-Visualisierung.
    
    Parameters:
        confusion_matrix: NumPy-Array der Konfusionsmatrix
        output_path: Optionaler Pfad zum Speichern der Ausgabe
    """
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap='Blues'
    )
    plt.title('Konfusionsmatrix')
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Wahre Klasse')
    
    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_training_progress(
    epochs: List[int],
    losses: List[float],
    metrics: Dict[str, List[float]],
    output_path: Optional[Path] = None
) -> None:
    """Visualisiert den Trainingsverlauf."""
    plt.figure(figsize=(12, 6))
    
    # Verlust
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-', label='Trainingsverlust')
    plt.xlabel('Epoch')
    plt.ylabel('Verlust')
    plt.title('Trainingsverlauf')
    plt.legend()
    plt.grid(True)
    
    # Metriken
    plt.subplot(1, 2, 2)
    for name, values in metrics.items():
        plt.plot(epochs, values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Wert')
    plt.title('Metriken')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Trainingsverlauf gespeichert unter {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_resource_usage(
    usage_history: List[ResourceUsage],
    output_path: Optional[Path] = None
) -> None:
    """
    Visualisiert den Ressourcenverbrauch über Zeit.
    """
    timestamps = range(len(usage_history))
    ram = [u.ram_used_kb for u in usage_history]
    flash = [u.flash_used_kb for u in usage_history]
    cpu = [u.cpu_usage_percent for u in usage_history]
    power = [u.power_mw for u in usage_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(timestamps, ram, label='RAM (KB)', color=COLOR_PALETTE['basic'])
    ax1.plot(timestamps, flash, label='Flash (KB)', color=COLOR_PALETTE['burnt'])
    ax1.set_ylabel('Speichernutzung (KB)')
    ax1.legend()
    
    ax2.plot(timestamps, cpu, label='CPU (%)', color=COLOR_PALETTE['perfect'])
    ax2.plot(timestamps, power, label='Leistung (mW)', color=COLOR_PALETTE['undercooked'])
    ax2.set_xlabel('Zeit')
    ax2.set_ylabel('CPU & Leistung')
    ax2.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_power_profile(
    profile: PowerProfile,
    output_path: Optional[Path] = None
) -> None:
    """Visualisiert ein Energieprofil."""
    plt.figure(figsize=(10, 6))
    
    time_ms = np.linspace(0, profile.duration_ms, num=100)
    current_ma = np.ones_like(time_ms) * profile.average_current_ma
    current_ma[0] = profile.peak_current_ma  # Spitzenstrom am Anfang
    
    plt.plot(time_ms, current_ma, 'r-')
    plt.axhline(y=profile.average_current_ma, color='b', linestyle='--',
                label=f'Durchschnitt: {profile.average_current_ma:.1f}mA')
    plt.axhline(y=profile.peak_current_ma, color='g', linestyle='--',
                label=f'Maximum: {profile.peak_current_ma:.1f}mA')
    
    plt.xlabel('Zeit (ms)')
    plt.ylabel('Strom (mA)')
    plt.title(f'Energieprofil\nGesamtenergie: {profile.total_energy_mj:.2f}mJ')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Energieprofil gespeichert unter {output_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_model_architecture(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Optional[Path] = None
) -> None:
    """Visualisiert die Modellarchitektur."""
    try:
        from torchviz import make_dot
        
        # Erstelle einen Beispiel-Input
        x = torch.randn(1, *input_shape)
        y = model(x)
        
        # Erstelle den Berechnungsgraphen
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        if output_path:
            dot.render(str(output_path), format='png')
            logger.info(f"Modellarchitektur gespeichert unter {output_path}")
        else:
            dot.view()
            
    except ImportError:
        logger.error("torchviz nicht installiert. Bitte installieren Sie graphviz und torchviz.")
    except Exception as e:
        logger.error(f"Fehler bei der Visualisierung der Modellarchitektur: {str(e)}")

def create_report(
    model_metrics: ModelMetrics,
    resource_usage: List[ResourceUsage],
    power_profile: PowerProfile,
    output_dir: Optional[Path] = None
) -> None:
    """Erstellt einen zusammenfassenden Bericht mit allen Visualisierungen."""
    output_dir = output_dir or OUTPUT_DIR / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = output_dir / f'report_{timestamp}'
    report_dir.mkdir()
    
    # Konfusionsmatrix
    plot_confusion_matrix(
        model_metrics.confusion_matrix,  # Direkter Zugriff auf das Attribut
        report_dir / 'confusion_matrix.png'
    )
    
    # Ressourcennutzung
    plot_resource_usage(
        resource_usage,
        report_dir / 'resource_usage.png'
    )
    
    # Energieprofil
    plot_power_profile(
        power_profile,
        report_dir / 'power_profile.png'
    )
    
    # Erstelle HTML-Bericht
    html_content = f"""
    <html>
    <head>
        <title>Leistungsbericht {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ margin: 10px 0; }}
            .plot {{ margin: 20px 0; }}
            img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Leistungsbericht {timestamp}</h1>
        
        <h2>Modellmetriken</h2>
        <div class="metric">
            <p>Genauigkeit: {model_metrics.accuracy:.2%}</p>
            <p>Precision: {model_metrics.precision:.2%}</p>
            <p>Recall: {model_metrics.recall:.2%}</p>
            <p>F1-Score: {model_metrics.f1_score:.2%}</p>
        </div>
        
        <h2>Konfusionsmatrix</h2>
        <div class="plot">
            <img src="confusion_matrix.png" alt="Konfusionsmatrix">
        </div>
        
        <h2>Ressourcennutzung</h2>
        <div class="plot">
            <img src="resource_usage.png" alt="Ressourcennutzung">
        </div>
        
        <h2>Energieprofil</h2>
        <div class="plot">
            <img src="power_profile.png" alt="Energieprofil">
        </div>
        
        <h2>Details</h2>
        <div class="metric">
            <p>Durchschnittlicher Stromverbrauch: {power_profile.average_current_ma:.1f}mA</p>
            <p>Maximaler Stromverbrauch: {power_profile.peak_current_ma:.1f}mA</p>
            <p>Gesamtenergie: {power_profile.total_energy_mj:.2f}mJ</p>
            <p>Dauer: {power_profile.duration_ms:.1f}ms</p>
        </div>
    </body>
    </html>
    """
    
    with open(report_dir / 'report.html', 'w') as f:
        f.write(html_content)
    
    logger.info(f"Bericht erstellt unter {report_dir}")

def annotate_image(
    image: np.ndarray,
    result: InferenceResult,
    draw_confidence: bool = True
) -> np.ndarray:
    """Fügt Erkennungsergebnisse zu einem Bild hinzu."""
    annotated = image.copy()
    
    # Rahmenfarbe basierend auf Klasse
    color = CLASS_COLORS[result.class_name]
    
    # Zeichne Rahmen
    height, width = image.shape[:2]
    thickness = max(2, int(min(height, width) / 200))
    cv2.rectangle(
        annotated,
        (0, 0),
        (width-1, height-1),
        color,
        thickness
    )
    
    # Textgröße basierend auf Bildgröße
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(font_scale * 2))
    
    # Klassenname
    text = result.class_name
    if draw_confidence:
        text += f' ({result.confidence:.1%})'
    
    # Texthintergrund
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (text_width + 10, text_height + 10),
        color,
        -1  # Gefüllt
    )
    
    # Text
    cv2.putText(
        annotated,
        text,
        (5, text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # Weiß
        font_thickness
    )
    
    return annotated

def plot_inference_results(
    results: List[InferenceResult],
    output_path: Optional[Path] = None
) -> None:
    """
    Visualisiert Inferenzergebnisse als Balkendiagramm.
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    for i, result in enumerate(results):
        plt.bar(
            range(len(result.probabilities)),
            list(result.probabilities.values()),
            alpha=0.5,
            label=f'Inferenz {i+1}'
        )
    
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.ylabel('Wahrscheinlichkeit')
    plt.title('Inferenzergebnisse')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(
    metrics_list: List[Tuple[str, ModelMetrics]],
    output_path: Optional[Path] = None
) -> None:
    """
    Vergleicht Metriken verschiedener Modelle.
    """
    models = [m[0] for m in metrics_list]
    accuracy = [m[1].accuracy for m in metrics_list]
    precision = [m[1].precision for m in metrics_list]
    recall = [m[1].recall for m in metrics_list]
    f1 = [m[1].f1_score for m in metrics_list]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(x - width*1.5, accuracy, width, label='Accuracy', color=COLOR_PALETTE['basic'])
    plt.bar(x - width/2, precision, width, label='Precision', color=COLOR_PALETTE['burnt'])
    plt.bar(x + width/2, recall, width, label='Recall', color=COLOR_PALETTE['undercooked'])
    plt.bar(x + width*1.5, f1, width, label='F1', color=COLOR_PALETTE['perfect'])

    plt.xlabel('Modell')
    plt.ylabel('Metrik-Wert')
    plt.title('Modellvergleich')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()