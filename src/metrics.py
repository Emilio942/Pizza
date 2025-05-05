"""
Leistungsmetriken und Evaluierungswerkzeuge.
"""

import json
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .types import (
    ModelMetrics,
    InferenceResult,
    ResourceUsage,
    PowerProfile,
    ErrorAnalysis
)
from .constants import (
    CLASS_NAMES,
    OUTPUT_DIR,
    MAX_INFERENCE_TIME_MS,
    MIN_CONFIDENCE_THRESHOLD,
    MAX_POWER_CONSUMPTION_MW
)
from .exceptions import DataError

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """Berechnet Leistungsmetriken für ein Modell."""
    # Fehlerbehandlung für leere Arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Eingabe-Arrays dürfen nicht leer sein")
    
    # Fehlerbehandlung für unterschiedliche Array-Längen
    if len(y_true) != len(y_pred):
        raise ValueError(f"Eingabe-Arrays müssen gleiche Länge haben: {len(y_true)} != {len(y_pred)}")
    
    # Überprüfe ungültige Klassenlabels
    from .constants import NUM_CLASSES
    if np.max(y_true) >= NUM_CLASSES or np.max(y_pred) >= NUM_CLASSES or np.min(y_true) < 0 or np.min(y_pred) < 0:
        raise ValueError(f"Ungültige Klassenlabels: Muss zwischen 0 und {NUM_CLASSES-1} liegen")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Erstelle eine Konfusionsmatrix mit fester Größe NUM_CLASSES x NUM_CLASSES
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    accuracy = (y_true == y_pred).mean()
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix
    }

def save_metrics(metrics: ModelMetrics, model_name: str) -> Path:
    """Speichert Metriken in einer JSON-Datei."""
    save_path = OUTPUT_DIR / 'evaluation' / f'{model_name}_metrics.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Konvertiere numpy arrays zu Listen für JSON-Serialisierung
    metrics_json = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return save_path

def load_metrics(model_name: str) -> ModelMetrics:
    """Lädt Metriken aus einer JSON-Datei."""
    load_path = OUTPUT_DIR / 'evaluation' / f'{model_name}_metrics.json'
    
    if not load_path.exists():
        raise DataError(f"Keine Metriken gefunden für: {model_name}")
    
    with open(load_path) as f:
        metrics_json = json.load(f)
    
    # Konvertiere Listen zurück zu numpy arrays
    metrics_json['confusion_matrix'] = np.array(metrics_json['confusion_matrix'])
    
    return metrics_json

def format_inference_result(logits: torch.Tensor) -> InferenceResult:
    """Formatiert Modellergebnis in strukturiertes Format."""
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    prediction = int(torch.argmax(logits).item())
    confidence = float(probs[prediction])
    
    probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(CLASS_NAMES, probs)
    }
    
    return InferenceResult(
        prediction=prediction,
        confidence=confidence,
        class_name=CLASS_NAMES[prediction],
        probabilities=probabilities
    )

def track_training_progress(epoch: int, loss: float, metrics: ModelMetrics) -> None:
    """Verfolgt Trainingfortschritt in einer JSON-Datei."""
    progress_path = OUTPUT_DIR / 'logs' / 'training_progress.json'
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    
    progress = {
        'epoch': epoch,
        'loss': float(loss),
        'metrics': {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        }
    }
    
    # Lade bestehenden Fortschritt oder erstelle neuen
    if progress_path.exists():
        with open(progress_path) as f:
            history = json.load(f)
    else:
        history = {'progress': []}
    
    history['progress'].append(progress)
    
    with open(progress_path, 'w') as f:
        json.dump(history, f, indent=2)

def get_error_analysis(metrics: ModelMetrics) -> Dict[str, Dict[str, float]]:
    """Führt detaillierte Fehleranalyse durch."""
    conf_matrix = metrics['confusion_matrix']
    error_analysis = {}
    
    for i, class_name in enumerate(CLASS_NAMES):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        total = fp + fn + tp
        if total > 0:
            error_rate = (fp + fn) / total
        else:
            error_rate = 0
        
        error_analysis[class_name] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'error_rate': float(error_rate)
        }
    
    return error_analysis

@dataclass
class PerformanceMetrics:
    """Sammelt Leistungsmetriken während der Ausführung."""
    
    # Zeitmessungen
    inference_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    processing_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    startup_times_ms: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Genauigkeitsmetriken
    correct_predictions: int = 0
    total_predictions: int = 0
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Ressourcennutzung
    ram_usage_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    power_samples_mw: deque = field(default_factory=lambda: deque(maxlen=100))
    temperature_samples_c: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Fehlertracking
    hardware_errors: int = 0
    software_errors: int = 0
    camera_errors: int = 0
    memory_errors: int = 0
    
    def add_inference_time(self, time_ms: float) -> None:
        """Fügt eine neue Inferenzzeit hinzu."""
        self.inference_times_ms.append(time_ms)
        if time_ms > MAX_INFERENCE_TIME_MS:
            logger.warning(
                f"Hohe Inferenzzeit: {time_ms:.1f}ms > {MAX_INFERENCE_TIME_MS}ms"
            )
    
    def add_prediction_result(
        self, 
        correct: bool, 
        confidence: float
    ) -> None:
        """Fügt ein Vorhersageergebnis hinzu."""
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1
        self.confidence_scores.append(confidence)
        
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Niedrige Konfidenz: {confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD}"
            )
    
    def add_resource_usage(
        self, 
        usage: ResourceUsage
    ) -> None:
        """Fügt Ressourcennutzungsdaten hinzu."""
        self.ram_usage_samples.append(usage.ram_used_kb)
        self.power_samples_mw.append(usage.power_mw)
        
        if usage.power_mw > MAX_POWER_CONSUMPTION_MW:
            logger.warning(
                f"Hoher Stromverbrauch: {usage.power_mw:.1f}mW > "
                f"{MAX_POWER_CONSUMPTION_MW}mW"
            )
    
    def add_error(self, error_type: str) -> None:
        """Registriert einen neuen Fehler."""
        if error_type == 'hardware':
            self.hardware_errors += 1
        elif error_type == 'software':
            self.software_errors += 1
        elif error_type == 'camera':
            self.camera_errors += 1
        elif error_type == 'memory':
            self.memory_errors += 1
        else:
            logger.error(f"Unbekannter Fehlertyp: {error_type}")
    
    def get_accuracy(self) -> float:
        """Berechnet die Gesamtgenauigkeit."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def get_average_inference_time(self) -> float:
        """Berechnet die durchschnittliche Inferenzzeit."""
        if not self.inference_times_ms:
            return 0.0
        return np.mean(self.inference_times_ms)
    
    def get_power_statistics(self) -> Dict[str, float]:
        """Berechnet Stromverbrauchsstatistiken."""
        if not self.power_samples_mw:
            return {
                'average_power_mw': 0.0,
                'peak_power_mw': 0.0,
                'min_power_mw': 0.0
            }
        
        return {
            'average_power_mw': np.mean(self.power_samples_mw),
            'peak_power_mw': np.max(self.power_samples_mw),
            'min_power_mw': np.min(self.power_samples_mw)
        }
    
    def get_ram_statistics(self) -> Dict[str, float]:
        """Berechnet RAM-Nutzungsstatistiken."""
        if not self.ram_usage_samples:
            return {
                'average_ram_kb': 0.0,
                'peak_ram_kb': 0.0,
                'min_ram_kb': 0.0
            }
        
        return {
            'average_ram_kb': np.mean(self.ram_usage_samples),
            'peak_ram_kb': np.max(self.ram_usage_samples),
            'min_ram_kb': np.min(self.ram_usage_samples)
        }
    
    def get_error_summary(self) -> Dict[str, int]:
        """Erstellt eine Zusammenfassung aller Fehler."""
        return {
            'hardware_errors': self.hardware_errors,
            'software_errors': self.software_errors,
            'camera_errors': self.camera_errors,
            'memory_errors': self.memory_errors,
            'total_errors': (
                self.hardware_errors + self.software_errors +
                self.camera_errors + self.memory_errors
            )
        }
    
    def get_performance_report(self) -> Dict:
        """Erstellt einen umfassenden Leistungsbericht."""
        return {
            'accuracy': self.get_accuracy(),
            'total_predictions': self.total_predictions,
            'average_confidence': (
                np.mean(self.confidence_scores) if self.confidence_scores else 0.0
            ),
            'average_inference_time_ms': self.get_average_inference_time(),
            'power_stats': self.get_power_statistics(),
            'ram_stats': self.get_ram_statistics(),
            'error_summary': self.get_error_summary(),
            'startup_time_ms': (
                np.mean(self.startup_times_ms) if self.startup_times_ms else 0.0
            )
        }
    
    def reset(self) -> None:
        """Setzt alle Metriken zurück."""
        self.inference_times_ms.clear()
        self.processing_times_ms.clear()
        self.startup_times_ms.clear()
        self.confidence_scores.clear()
        self.ram_usage_samples.clear()
        self.power_samples_mw.clear()
        self.temperature_samples_c.clear()
        
        self.correct_predictions = 0
        self.total_predictions = 0
        self.hardware_errors = 0
        self.software_errors = 0
        self.camera_errors = 0
        self.memory_errors = 0

class ConfusionMatrix:
    """Verwaltet eine Konfusionsmatrix für Klassifikationsergebnisse."""
    
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.class_names = CLASS_NAMES
    
    def update(self, true_label: int, pred_label: int) -> None:
        """Aktualisiert die Konfusionsmatrix."""
        if 0 <= true_label < self.num_classes and 0 <= pred_label < self.num_classes:
            self.matrix[true_label][pred_label] += 1
        else:
            logger.error(
                f"Ungültige Label: true={true_label}, pred={pred_label}, "
                f"max={self.num_classes-1}"
            )
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Berechnet Metriken pro Klasse."""
        metrics = {}
        
        for i in range(self.num_classes):
            true_pos = self.matrix[i][i]
            false_pos = np.sum(self.matrix[:, i]) - true_pos
            false_neg = np.sum(self.matrix[i, :]) - true_pos
            true_neg = np.sum(self.matrix) - true_pos - false_pos - false_neg
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[self.class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(self.matrix[i, :])
            }
        
        return metrics
    
    def get_overall_accuracy(self) -> float:
        """Berechnet die Gesamtgenauigkeit."""
        return np.trace(self.matrix) / np.sum(self.matrix)
    
    def reset(self) -> None:
        """Setzt die Konfusionsmatrix zurück."""
        self.matrix.fill(0)

class PowerMonitor:
    """Überwacht den Stromverbrauch des Systems."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.power_history = deque(maxlen=window_size)
        self.start_time = None
        self.current_profile = None
    
    def start_monitoring(self) -> None:
        """Startet die Stromverbrauchsüberwachung."""
        self.start_time = time.time()
        self.current_profile = []
    
    def add_measurement(self, power_mw: float) -> None:
        """Fügt eine neue Leistungsmessung hinzu."""
        if self.start_time is None:
            logger.warning("PowerMonitor nicht gestartet")
            return
            
        self.power_history.append(power_mw)
        if self.current_profile is not None:
            self.current_profile.append((time.time() - self.start_time, power_mw))
    
    def stop_monitoring(self) -> PowerProfile:
        """Beendet die Überwachung und erstellt ein Energieprofil."""
        if self.start_time is None or self.current_profile is None:
            logger.error("PowerMonitor nicht gestartet")
            return None
            
        duration_ms = (time.time() - self.start_time) * 1000
        profile_array = np.array(self.current_profile)
        
        if len(profile_array) == 0:
            logger.warning("Keine Messungen im Profil")
            return None
            
        average_power = np.mean(profile_array[:, 1])
        peak_power = np.max(profile_array[:, 1])
        total_energy = np.trapz(profile_array[:, 1], profile_array[:, 0])
        
        self.start_time = None
        self.current_profile = None
        
        return PowerProfile(
            duration_ms=duration_ms,
            average_current_ma=average_power / 3.3,  # Annahme: 3.3V
            peak_current_ma=peak_power / 3.3,
            total_energy_mj=total_energy
        )
    
    def get_average_power(self) -> float:
        """Berechnet die durchschnittliche Leistungsaufnahme."""
        return np.mean(self.power_history) if self.power_history else 0.0
    
    def get_power_statistics(self) -> Dict[str, float]:
        """Berechnet detaillierte Stromverbrauchsstatistiken."""
        if not self.power_history:
            return {
                'average_power_mw': 0.0,
                'peak_power_mw': 0.0,
                'min_power_mw': 0.0,
                'std_dev_power_mw': 0.0
            }
        
        power_array = np.array(self.power_history)
        return {
            'average_power_mw': np.mean(power_array),
            'peak_power_mw': np.max(power_array),
            'min_power_mw': np.min(power_array),
            'std_dev_power_mw': np.std(power_array)
        }