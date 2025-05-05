#!/usr/bin/env python3
"""
Test-Skript für Temporal Smoothing bei der Pizza-Erkennung.
Demonstriert die Verbesserung der Stabilitität bei verrauschten Inferenz-Ergebnissen.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import time
from enum import Enum
from collections import deque

# Füge das Projekt-Root zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Nur für das Beispiel definierte Klassen
PIZZA_CLASSES = [
    "basic",
    "burnt",
    "combined",
    "mixed",
    "progression",
    "segment"
]

# Temporal Smoothing Strategien, entsprechend den C-Implementierungen
class SmoothingStrategy(Enum):
    MAJORITY_VOTE = 0
    MOVING_AVERAGE = 1
    EXPONENTIAL_MA = 2
    CONFIDENCE_WEIGHTED = 3


class TemporalSmoother:
    """Python-Implementierung der C-Funktionalität zum Testen"""
    
    def __init__(self, window_size=5, strategy=SmoothingStrategy.MAJORITY_VOTE, decay_factor=0.7):
        self.window_size = window_size
        self.strategy = strategy
        self.decay_factor = decay_factor
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
    
    def add_prediction(self, class_idx, confidence, probabilities):
        self.predictions.append(class_idx)
        self.confidences.append(confidence)
        self.probabilities.append(probabilities.copy())
    
    def get_smoothed_prediction(self):
        if not self.predictions:
            return 0, 0.0, np.zeros(len(PIZZA_CLASSES))
            
        if self.strategy == SmoothingStrategy.MAJORITY_VOTE:
            return self._apply_majority_vote()
        elif self.strategy == SmoothingStrategy.MOVING_AVERAGE:
            return self._apply_moving_average()
        elif self.strategy == SmoothingStrategy.EXPONENTIAL_MA:
            return self._apply_exponential_ma()
        elif self.strategy == SmoothingStrategy.CONFIDENCE_WEIGHTED:
            return self._apply_confidence_weighted()
        else:
            # Fallback: Verwende neueste Vorhersage
            idx = len(self.predictions) - 1
            return (self.predictions[idx], self.confidences[idx], self.probabilities[idx])
    
    def _apply_majority_vote(self):
        # Zähle Häufigkeiten
        counts = {}
        for pred in self.predictions:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Finde häufigste Klasse
        most_common_class = max(counts, key=counts.get)
        
        # Berechne durchschnittliche Konfidenz für diese Klasse
        confidence_sum = 0.0
        confidence_count = 0
        
        for i, pred in enumerate(self.predictions):
            if pred == most_common_class:
                confidence_sum += self.confidences[i]
                confidence_count += 1
        
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        # Berechne durchschnittliche Wahrscheinlichkeiten
        avg_probs = np.zeros(len(PIZZA_CLASSES))
        for prob in self.probabilities:
            avg_probs += prob
        avg_probs /= len(self.probabilities)
        
        return most_common_class, avg_confidence, avg_probs
    
    def _apply_moving_average(self):
        # Berechne durchschnittliche Wahrscheinlichkeiten
        avg_probs = np.zeros(len(PIZZA_CLASSES))
        for prob in self.probabilities:
            avg_probs += prob
        avg_probs /= len(self.probabilities)
        
        # Finde Klasse mit höchster durchschnittlicher Wahrscheinlichkeit
        max_class = np.argmax(avg_probs)
        max_prob = avg_probs[max_class]
        
        return max_class, max_prob, avg_probs
    
    def _apply_exponential_ma(self):
        # Berechne Gewichte: neuere Einträge haben höheres Gewicht
        weights = np.array([self.decay_factor ** i for i in range(len(self.probabilities) - 1, -1, -1)])
        weights = weights / weights.sum()  # Normalisieren
        
        # Berechne gewichtete Wahrscheinlichkeiten
        ema_probs = np.zeros(len(PIZZA_CLASSES))
        for i, prob in enumerate(self.probabilities):
            ema_probs += weights[i] * prob
        
        # Finde Klasse mit höchster gewichteter Wahrscheinlichkeit
        max_class = np.argmax(ema_probs)
        max_prob = ema_probs[max_class]
        
        return max_class, max_prob, ema_probs
    
    def _apply_confidence_weighted(self):
        # Berechne gewichtete Wahrscheinlichkeiten basierend auf Konfidenz
        total_confidence = sum(self.confidences)
        
        if total_confidence == 0:
            return self._apply_moving_average()
        
        weighted_probs = np.zeros(len(PIZZA_CLASSES))
        for i, prob in enumerate(self.probabilities):
            weight = self.confidences[i] / total_confidence
            weighted_probs += weight * prob
        
        # Finde Klasse mit höchster gewichteter Wahrscheinlichkeit
        max_class = np.argmax(weighted_probs)
        max_prob = weighted_probs[max_class]
        
        return max_class, max_prob, weighted_probs
    
    def reset(self):
        self.predictions.clear()
        self.confidences.clear()
        self.probabilities.clear()


def generate_noisy_predictions(num_frames=100, true_class=0, noise_level=0.3, change_points=None):
    """
    Generiert eine Sequenz von simulierten Vorhersagen mit Rauschen.
    
    Args:
        num_frames: Anzahl der zu erzeugenden Frames
        true_class: Die tatsächliche Klasse
        noise_level: Stärke des Rauschens (0-1)
        change_points: Liste mit Framepositionen, an denen sich die wahre Klasse ändert
                      Format: [(frame_idx, new_class), ...]
    
    Returns:
        Tuple aus (class_indices, confidences, probabilities)
    """
    class_indices = []
    confidences = []
    probabilities = []
    
    current_class = true_class
    
    for i in range(num_frames):
        # Prüfe, ob die wahre Klasse sich ändert
        if change_points:
            for frame_idx, new_class in change_points:
                if i == frame_idx:
                    current_class = new_class
        
        # Generiere Rauschen
        probs = np.random.random(len(PIZZA_CLASSES)) * noise_level
        
        # Setze wahre Klasse als Basis
        probs[current_class] = 1.0 - noise_level + np.random.random() * noise_level * 0.5
        
        # Normalisiere zu Summe 1
        probs = probs / probs.sum()
        
        # Argmax für die vorhergesagte Klasse
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        class_indices.append(pred_class)
        confidences.append(confidence)
        probabilities.append(probs)
    
    return class_indices, confidences, probabilities


def test_smoothing_strategies(predictions, confidences, probabilities, true_classes, strategy_names=None):
    """
    Testet verschiedene Smoothing-Strategien auf einer Sequenz von Vorhersagen.
    
    Args:
        predictions: Liste von vorhergesagten Klassenindizes
        confidences: Liste von Konfidenzwerten
        probabilities: Liste von Wahrscheinlichkeits-Arrays
        true_classes: Liste oder Array der wahren Klassen pro Frame
        strategy_names: Liste der zu testenden Strategien
    
    Returns:
        Dictionary mit Ergebnissen für jede Strategie
    """
    if strategy_names is None:
        strategy_names = [s.name for s in SmoothingStrategy]
    
    strategies = {
        'MAJORITY_VOTE': SmoothingStrategy.MAJORITY_VOTE,
        'MOVING_AVERAGE': SmoothingStrategy.MOVING_AVERAGE,
        'EXPONENTIAL_MA': SmoothingStrategy.EXPONENTIAL_MA,
        'CONFIDENCE_WEIGHTED': SmoothingStrategy.CONFIDENCE_WEIGHTED
    }
    
    results = {}
    
    # Für jede Strategie
    for name in strategy_names:
        strategy = strategies[name]
        smoother = TemporalSmoother(window_size=5, strategy=strategy)
        
        smoothed_predictions = []
        smoothed_confidences = []
        
        # Simuliere Echtzeit-Verarbeitung
        for i in range(len(predictions)):
            smoother.add_prediction(predictions[i], confidences[i], probabilities[i])
            smooth_class, smooth_conf, _ = smoother.get_smoothed_prediction()
            
            smoothed_predictions.append(smooth_class)
            smoothed_confidences.append(smooth_conf)
        
        # Berechne Genauigkeit
        accuracy_raw = np.mean(np.array(predictions) == np.array(true_classes))
        accuracy_smoothed = np.mean(np.array(smoothed_predictions) == np.array(true_classes))
        
        # Speichere Ergebnisse
        results[name] = {
            'smoothed_predictions': smoothed_predictions,
            'smoothed_confidences': smoothed_confidences,
            'accuracy_raw': accuracy_raw,
            'accuracy_smoothed': accuracy_smoothed,
            'improvement': accuracy_smoothed - accuracy_raw
        }
    
    return results


def plot_results(predictions, confidences, true_classes, results, output_dir):
    """
    Visualisiert die Ergebnisse der verschiedenen Smoothing-Strategien.
    
    Args:
        predictions: Liste der Rohvorhersagen
        confidences: Liste der Rohkonfidenzwerte
        true_classes: Liste der wahren Klassen
        results: Dictionary mit Ergebnissen von test_smoothing_strategies
        output_dir: Verzeichnis für die Ausgabedateien
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Vergleich der Strategien
    plt.figure(figsize=(15, 10))
    
    frames = range(len(predictions))
    
    # Rohvorhersagen
    plt.subplot(len(results) + 1, 1, 1)
    plt.plot(frames, true_classes, 'g-', label='Wahre Klasse')
    plt.plot(frames, predictions, 'r.', label='Rohvorhersage')
    plt.title(f'Rohvorhersagen (Genauigkeit: {np.mean(np.array(predictions) == np.array(true_classes)):.2%})')
    plt.ylabel('Klasse')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Geglättete Vorhersagen für jede Strategie
    for i, (name, result) in enumerate(results.items(), 2):
        plt.subplot(len(results) + 1, 1, i)
        plt.plot(frames, true_classes, 'g-', label='Wahre Klasse')
        plt.plot(frames, result['smoothed_predictions'], 'b.', label='Geglättete Vorhersage')
        plt.title(f'{name} (Genauigkeit: {result["accuracy_smoothed"]:.2%}, Verbesserung: {result["improvement"]:.2%})')
        plt.ylabel('Klasse')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_comparison.png")
    plt.close()
    
    # 2. Stabilitätsvergleich
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(frames, confidences, 'r-', label='Roh-Konfidenz')
    for name, result in results.items():
        if name == 'MAJORITY_VOTE':  # Zeige nur eine Strategie zur Übersichtlichkeit
            plt.plot(frames, result['smoothed_confidences'], 'b-', label=f'{name} Konfidenz')
    plt.title('Stabilität der Konfidenzwerte')
    plt.ylabel('Konfidenz')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    
    # Berechne "Zittern" (Änderungsrate) in den Vorhersagen
    raw_changes = np.sum(np.diff(predictions) != 0)
    
    # Zeige Balkendiagramm der Änderungsraten
    strategy_names = []
    change_counts = []
    
    strategy_names.append("Rohvorhersage")
    change_counts.append(raw_changes)
    
    for name, result in results.items():
        smooth_changes = np.sum(np.diff(result['smoothed_predictions']) != 0)
        strategy_names.append(name)
        change_counts.append(smooth_changes)
    
    plt.bar(strategy_names, change_counts)
    plt.title('Anzahl der Klassenwechsel (niedrigere Werte = stabilere Vorhersage)')
    plt.ylabel('Anzahl der Änderungen')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "stability_comparison.png")
    plt.close()
    
    # 3. Zusammenfassung
    plt.figure(figsize=(12, 6))
    
    # Genauigkeitsvergleich
    accuracies = [np.mean(np.array(predictions) == np.array(true_classes))]
    names = ['Rohvorhersage']
    
    for name, result in results.items():
        accuracies.append(result['accuracy_smoothed'])
        names.append(name)
    
    plt.subplot(1, 2, 1)
    plt.bar(names, accuracies, color='skyblue')
    plt.title('Genauigkeit der Vorhersagen')
    plt.ylabel('Genauigkeit')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Verbesserungen (nur für geglättete Strategien)
    smoothed_names = names[1:]  # Ohne "Rohvorhersage"
    improvements = [result['improvement'] * 100 for result in results.values()]  # In Prozent
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(smoothed_names, improvements, color='lightgreen')
    plt.title('Verbesserung durch Temporal Smoothing')
    plt.ylabel('Verbesserung in Prozentpunkten')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Beschrifte Balken
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}pp', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_summary.png")
    plt.close()
    
    # 4. Erzeuge einen Bericht
    with open(output_dir / "temporal_smoothing_report.txt", "w") as f:
        f.write("=== Temporal Smoothing Ergebnisbericht ===\n\n")
        
        f.write("Testkonfiguration:\n")
        f.write(f"- Anzahl der Frames: {len(predictions)}\n")
        f.write(f"- Smoothing-Fenstergröße: 5\n\n")
        
        f.write("Rohvorhersagen:\n")
        f.write(f"- Genauigkeit: {np.mean(np.array(predictions) == np.array(true_classes)):.2%}\n")
        f.write(f"- Anzahl der Klassenwechsel: {raw_changes}\n\n")
        
        f.write("Ergebnisse nach Strategie:\n")
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"- Genauigkeit: {result['accuracy_smoothed']:.2%}\n")
            f.write(f"- Verbesserung: {result['improvement']:.2%}\n")
            f.write(f"- Anzahl der Klassenwechsel: {np.sum(np.diff(result['smoothed_predictions']) != 0)}\n")
        
        f.write("\n\nEmpfehlung:\n")
        best_strategy = max(results.items(), key=lambda x: x[1]['accuracy_smoothed'])
        f.write(f"Die beste Strategie für diesen Datensatz ist {best_strategy[0]} ")
        f.write(f"mit einer Genauigkeit von {best_strategy[1]['accuracy_smoothed']:.2%} ")
        f.write(f"(Verbesserung um {best_strategy[1]['improvement']:.2%}).\n")
        
        most_stable = min(results.items(), key=lambda x: np.sum(np.diff(x[1]['smoothed_predictions']) != 0))
        f.write(f"Die stabilste Strategie ist {most_stable[0]} ")
        f.write(f"mit nur {np.sum(np.diff(most_stable[1]['smoothed_predictions']) != 0)} Klassenwechseln.\n")
        
        f.write("\nHinweis: Die optimale Strategie kann je nach Anwendungsfall variieren. ")
        f.write("Wenn Stabilität wichtiger ist als Genauigkeit, könnte eine andere Strategie bevorzugt werden.")


def simulate_scenario(scenario_name, output_dir, window_size=5):
    """
    Simuliert verschiedene Szenarien für den Temporal-Smoothing-Test.
    
    Args:
        scenario_name: Name des zu simulierenden Szenarios
        output_dir: Ausgabeverzeichnis
        window_size: Fenstergröße für das Smoothing
    """
    output_dir = Path(output_dir) / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if scenario_name == "flackernde_erkennung":
        # Szenario: Klasse flackert zwischen zwei Werten
        true_classes = np.zeros(100, dtype=int)
        true_classes[20:80] = 1  # Wahrer Wechsel zu Klasse 1
        
        # Erzeugt Vorhersagen mit hohem Rauschen
        predictions, confidences, probabilities = generate_noisy_predictions(
            num_frames=100, 
            true_class=0, 
            noise_level=0.7,
            change_points=[(20, 1), (80, 0)]
        )
        
    elif scenario_name == "kurze_unterbrechung":
        # Szenario: Kurze Unterbrechung/Flackern in stabiler Erkennung
        true_classes = np.ones(100, dtype=int)  # Alles Klasse 1
        
        # Erzeugt Vorhersagen mit mittlerem Rauschen und kurzen falschen Erkennungen
        predictions, confidences, probabilities = generate_noisy_predictions(
            num_frames=100, 
            true_class=1, 
            noise_level=0.4
        )
        # Füge gezielt falsche Erkennungen ein
        for i in range(30, 35):
            probabilities[i][1] = 0.2
            probabilities[i][0] = 0.8
            predictions[i] = 0
            confidences[i] = 0.8
            
        for i in range(60, 63):
            probabilities[i][1] = 0.3
            probabilities[i][2] = 0.7
            predictions[i] = 2
            confidences[i] = 0.7
        
    elif scenario_name == "mehrere_klassen":
        # Szenario: Rotierende Erkennung mit mehreren Klassen
        true_classes = np.zeros(150, dtype=int)
        true_classes[30:60] = 1
        true_classes[60:90] = 2
        true_classes[90:120] = 3
        
        # Erzeugt Vorhersagen mit mehreren Klassenwechseln
        predictions, confidences, probabilities = generate_noisy_predictions(
            num_frames=150, 
            true_class=0, 
            noise_level=0.5,
            change_points=[(30, 1), (60, 2), (90, 3), (120, 0)]
        )
    
    elif scenario_name == "niedrige_konfidenz":
        # Szenario: Erkennung mit niedriger Konfidenz
        true_classes = np.ones(100, dtype=int)
        
        # Erzeugt Vorhersagen mit sehr niedrigen Konfidenzwerten
        predictions, confidences, probabilities = generate_noisy_predictions(
            num_frames=100, 
            true_class=1, 
            noise_level=0.8
        )
        # Reduziere alle Konfidenzwerte weiter
        for i in range(len(confidences)):
            confidences[i] *= 0.6
            probs = probabilities[i] * 0.6
            probabilities[i] = probs / probs.sum()
    
    else:
        # Standard-Szenario mit moderatem Rauschen
        true_classes = np.zeros(100, dtype=int)
        true_classes[40:70] = 1  # Wahrer Wechsel zu Klasse 1
        
        predictions, confidences, probabilities = generate_noisy_predictions(
            num_frames=100, 
            true_class=0, 
            noise_level=0.4,
            change_points=[(40, 1), (70, 0)]
        )
    
    # Teste verschiedene Strategien
    results = test_smoothing_strategies(
        predictions, 
        confidences, 
        probabilities, 
        true_classes
    )
    
    # Visualisiere Ergebnisse
    plot_results(predictions, confidences, true_classes, results, output_dir)
    
    print(f"Ergebnisse für Szenario '{scenario_name}' gespeichert in {output_dir}")


def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Test der Temporal-Smoothing-Strategien')
    parser.add_argument('--output-dir', default='output/temporal_smoothing_test', 
                        help='Ausgabeverzeichnis für Testergebnisse')
    parser.add_argument('--scenario', default='all',
                        choices=['all', 'flackernde_erkennung', 'kurze_unterbrechung', 
                                 'mehrere_klassen', 'niedrige_konfidenz', 'standard'],
                        help='Zu simulierendes Szenario')
    args = parser.parse_args()
    
    if args.scenario == 'all':
        scenarios = ['flackernde_erkennung', 'kurze_unterbrechung', 
                     'mehrere_klassen', 'niedrige_konfidenz', 'standard']
        for scenario in scenarios:
            simulate_scenario(scenario, args.output_dir)
    else:
        simulate_scenario(args.scenario, args.output_dir)


if __name__ == "__main__":
    main()