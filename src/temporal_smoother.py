"""
Temporal Smoothing für Mehrbild-Entscheidung in der Pizza-Erkennung.

Dieses Modul implementiert verschiedene Strategien für die Kombination mehrerer
aufeinanderfolgender Inferenz-Ergebnisse, um die Genauigkeit und Robustheit
der Pizza-Erkennung zu erhöhen.
"""

import numpy as np
import torch
from collections import deque
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union


class SmoothingStrategy(Enum):
    """Verfügbare Strategien für Temporal Smoothing."""
    MAJORITY_VOTE = "majority_vote"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_MOVING_AVERAGE = "exponential_moving_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class TemporalSmoother:
    """
    Implementiert verschiedene Methoden für Temporal Smoothing.
    
    Diese Klasse kann mehrere aufeinanderfolgende Inferenz-Ergebnisse kombinieren,
    um stabilere und verlässlichere Vorhersagen zu erhalten.
    """
    
    def __init__(
        self, 
        window_size: int = 5, 
        strategy: SmoothingStrategy = SmoothingStrategy.MAJORITY_VOTE,
        decay_factor: float = 0.7  # nur für EMA verwendet
    ):
        """
        Initialisiert den TemporalSmoother.
        
        Args:
            window_size: Anzahl der zu berücksichtigenden Frames
            strategy: Zu verwendende Smoothing-Strategie
            decay_factor: Abklingfaktor für Exponential Moving Average (0-1)
        """
        self.window_size = window_size
        self.strategy = strategy
        self.decay_factor = max(0.0, min(1.0, decay_factor))  # Begrenze auf 0-1
        
        # Speicher für bisherige Vorhersagen und Konfidenzwerte
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
    
    def add_result(self, class_index: int, confidence: float, class_probs: Dict[str, float] = None):
        """
        Fügt ein neues Inferenz-Ergebnis zur Historie hinzu.
        
        Args:
            class_index: Index der vorhergesagten Klasse
            confidence: Konfidenz der Vorhersage (0-1)
            class_probs: Dictionary mit Klassenwahrscheinlichkeiten
        """
        self.predictions.append(class_index)
        self.confidences.append(confidence)
        
        if class_probs:
            self.probabilities.append(class_probs)
    
    def get_smoothed_prediction(self) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """
        Berechnet die geglättete Vorhersage basierend auf der gewählten Strategie.
        
        Returns:
            Tuple aus (Klassen-Index, Konfidenz, Klassenwahrscheinlichkeiten)
            Falls keine Daten vorliegen, wird (0, 0.0, None) zurückgegeben
        """
        if not self.predictions:
            return 0, 0.0, None
        
        if self.strategy == SmoothingStrategy.MAJORITY_VOTE:
            return self._apply_majority_vote()
        elif self.strategy == SmoothingStrategy.MOVING_AVERAGE:
            return self._apply_moving_average()
        elif self.strategy == SmoothingStrategy.EXPONENTIAL_MOVING_AVERAGE:
            return self._apply_exponential_moving_average()
        elif self.strategy == SmoothingStrategy.CONFIDENCE_WEIGHTED:
            return self._apply_confidence_weighted()
        else:
            # Fallback: Verwende neueste Vorhersage
            idx = len(self.predictions) - 1
            return (
                self.predictions[idx], 
                self.confidences[idx],
                self.probabilities[idx] if self.probabilities else None
            )
    
    def _apply_majority_vote(self) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """Mehrheitsentscheidung über die letzten N Vorhersagen."""
        # Zähle Häufigkeiten
        from collections import Counter
        counter = Counter(self.predictions)
        
        # Finde häufigste Klasse
        most_common_class, count = counter.most_common(1)[0]
        
        # Berechne durchschnittliche Konfidenz für diese Klasse
        confidence_sum = 0.0
        confidence_count = 0
        
        for i, pred in enumerate(self.predictions):
            if (pred == most_common_class):
                confidence_sum += self.confidences[i]
                confidence_count += 1
        
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        # Falls Wahrscheinlichkeiten vorhanden, berechne den Durchschnitt für jede Klasse
        avg_probs = None
        if self.probabilities:
            # Extrahiere alle Klassen
            all_classes = set()
            for probs in self.probabilities:
                all_classes.update(probs.keys())
            
            # Initialisiere Dictionary für Durchschnittswerte
            avg_probs = {cls: 0.0 for cls in all_classes}
            
            # Summiere Wahrscheinlichkeiten
            for probs in self.probabilities:
                for cls in all_classes:
                    avg_probs[cls] += probs.get(cls, 0.0)
            
            # Teile durch Anzahl der Vorhersagen
            for cls in avg_probs:
                avg_probs[cls] /= len(self.probabilities)
        
        return most_common_class, avg_confidence, avg_probs
    
    def _apply_moving_average(self) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """Anwendung eines gleitenden Mittelwerts auf die Klassenwahrscheinlichkeiten."""
        if not self.probabilities:
            # Fallback zu Majority Vote, wenn keine Wahrscheinlichkeiten verfügbar
            return self._apply_majority_vote()
        
        # Extrahiere alle Klassenbezeichner
        all_classes = set()
        for probs in self.probabilities:
            all_classes.update(probs.keys())
        
        # Initialisiere Dictionary für Durchschnittswerte
        avg_probs = {cls: 0.0 for cls in all_classes}
        
        # Berechne Durchschnitt der Wahrscheinlichkeiten
        for probs in self.probabilities:
            for cls in all_classes:
                avg_probs[cls] += probs.get(cls, 0.0)
        
        for cls in avg_probs:
            avg_probs[cls] /= len(self.probabilities)
        
        # Finde Klasse mit höchster durchschnittlicher Wahrscheinlichkeit
        best_class = max(avg_probs, key=avg_probs.get)
        best_class_idx = list(all_classes).index(best_class)
        best_confidence = avg_probs[best_class]
        
        return best_class_idx, best_confidence, avg_probs
    
    def _apply_exponential_moving_average(self) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """Anwendung eines exponentiell gewichteten gleitenden Mittelwerts."""
        if not self.probabilities:
            # Fallback zu Majority Vote, wenn keine Wahrscheinlichkeiten verfügbar
            return self._apply_majority_vote()
        
        # Extrahiere alle Klassenbezeichner
        all_classes = set()
        for probs in self.probabilities:
            all_classes.update(probs.keys())
            
        # Initialisiere Dictionary für Durchschnittswerte
        ema_probs = {cls: 0.0 for cls in all_classes}
        
        # Gewichte berechnen
        weights = np.array([self.decay_factor ** i for i in range(len(self.probabilities) - 1, -1, -1)])
        weights = weights / weights.sum()  # Normalisieren
        
        # Berechne gewichteten Durchschnitt der Wahrscheinlichkeiten
        for i, probs in enumerate(self.probabilities):
            for cls in all_classes:
                ema_probs[cls] += weights[i] * probs.get(cls, 0.0)
        
        # Finde Klasse mit höchster gewichteter Wahrscheinlichkeit
        best_class = max(ema_probs, key=ema_probs.get)
        best_class_idx = list(all_classes).index(best_class)
        best_confidence = ema_probs[best_class]
        
        return best_class_idx, best_confidence, ema_probs
    
    def _apply_confidence_weighted(self) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """Gewichtete Entscheidung basierend auf Konfidenzwerten."""
        if not self.probabilities:
            # Fallback zu Majority Vote, wenn keine Wahrscheinlichkeiten verfügbar
            return self._apply_majority_vote()
        
        # Extrahiere alle Klassenbezeichner
        all_classes = set()
        for probs in self.probabilities:
            all_classes.update(probs.keys())
        
        # Initialisiere Dictionary für gewichtete Durchschnittswerte
        weighted_probs = {cls: 0.0 for cls in all_classes}
        total_confidence = sum(self.confidences)
        
        if total_confidence == 0:
            # Wenn alle Konfidenzwerte 0 sind, verwende einfachen Durchschnitt
            return self._apply_moving_average()
        
        # Berechne konfidenzgewichteten Durchschnitt der Wahrscheinlichkeiten
        for i, probs in enumerate(self.probabilities):
            weight = self.confidences[i] / total_confidence
            for cls in all_classes:
                weighted_probs[cls] += weight * probs.get(cls, 0.0)
        
        # Finde Klasse mit höchster gewichteter Wahrscheinlichkeit
        best_class = max(weighted_probs, key=weighted_probs.get)
        best_class_idx = list(all_classes).index(best_class)
        best_confidence = weighted_probs[best_class]
        
        return best_class_idx, best_confidence, weighted_probs
    
    def reset(self):
        """Setzt alle internen Zustände zurück."""
        self.predictions.clear()
        self.confidences.clear()
        self.probabilities.clear()


# Hilfsklasse für zeitliche Glättung bei kontinuierlicher Inferenz
class PizzaTemporalPredictor:
    """
    Konsolidierende Klasse für die zeitliche Glättung von Pizza-Erkennungen.
    
    Diese Klasse ist für die direkte Verwendung in Anwendungen gedacht
    und bietet eine einfache API für kontinuierliche Inferenz mit 
    integrierter zeitlicher Glättung.
    """
    
    def __init__(
        self, 
        model,
        class_names: List[str],
        window_size: int = 5,
        strategy: Union[str, SmoothingStrategy] = "majority_vote"
    ):
        """
        Initialisiert den temporalen Prädiktor.
        
        Args:
            model: Das zu verwendende PyTorch- oder TFLite-Modell
            class_names: Liste der Klassennamen
            window_size: Anzahl der zu speichernden Frames
            strategy: Zu verwendende Glättungsstrategie
        """
        self.model = model
        self.class_names = class_names
        
        # Konvertiere String-Strategie in Enum
        if isinstance(strategy, str):
            strategy = SmoothingStrategy(strategy)
        
        self.smoother = TemporalSmoother(
            window_size=window_size,
            strategy=strategy
        )
    
    def predict(self, input_tensor, device=None):
        """
        Führt eine Vorhersage durch und wendet zeitliche Glättung an.
        
        Args:
            input_tensor: Eingabetensor für das Modell
            device: Optional, das zu verwendende PyTorch-Gerät
            
        Returns:
            Ein Dictionary mit 'class' (String), 'confidence' (Float) und
            'probabilities' (Dict[str, float])
        """
        # Inferenz auf aktuellem Frame durchführen
        with torch.no_grad():
            if device:
                input_tensor = input_tensor.to(device)
                self.model = self.model.to(device)
            
            # Führe Inferenz durch
            outputs = self.model(input_tensor.unsqueeze(0))
            
            # Berechne Wahrscheinlichkeiten
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Bestimme die vorhergesagte Klasse
            predicted = torch.argmax(outputs, dim=1).item()
            confidence = probs[predicted].item()
            
            # Erstelle Wahrscheinlichkeits-Dict
            class_probs = {self.class_names[i]: prob.item() for i, prob in enumerate(probs)}
        
        # Füge Ergebnis zum Smoother hinzu
        self.smoother.add_result(predicted, confidence, class_probs)
        
        # Hole geglättetes Ergebnis
        smoothed_idx, smoothed_conf, smoothed_probs = self.smoother.get_smoothed_prediction()
        
        # Konvertiere Index zu Klassenname
        if 0 <= smoothed_idx < len(self.class_names):
            smoothed_class = self.class_names[smoothed_idx]
        else:
            smoothed_class = "unknown"
        
        return {
            'class': smoothed_class,
            'confidence': smoothed_conf,
            'probabilities': smoothed_probs,
            'raw_class': self.class_names[predicted],
            'raw_confidence': confidence,
            'raw_probabilities': class_probs,
            'is_smoothed': len(self.smoother.predictions) > 1
        }


# Funktionen für C/C++-Implementierung
def generate_c_implementation(window_size=5, strategy="majority_vote"):
    """
    Generiert C/C++-Code für die Temporal-Smoothing-Implementierung.
    
    Args:
        window_size: Größe des Glättungsfensters
        strategy: Zu verwendende Glättungsstrategie 
        
    Returns:
        String mit C/C++-Code
    """
    c_code = """
/**
 * Temporal Smoothing für Pizza-Erkennungsmodell
 * Implementiert {strategy} über {window_size} Frames
 */
 
#include <stdint.h>
#include <stdbool.h>

#define TS_WINDOW_SIZE {window_size}
#define TS_NUM_CLASSES {num_classes}

// Ringpuffer für letzte Vorhersagen
static int prediction_history[TS_WINDOW_SIZE]; 
static float confidence_history[TS_WINDOW_SIZE];
static float probability_history[TS_WINDOW_SIZE][TS_NUM_CLASSES];
static int history_count = 0;
static int history_index = 0;

/**
 * Initialisiert den Temporal-Smoothing-Puffer
 */
void ts_init(void) {{
    history_count = 0;
    history_index = 0;
}}

/**
 * Fügt eine neue Vorhersage zum Temporal-Smoothing-Puffer hinzu
 * 
 * @param predicted_class Der Index der vorhergesagten Klasse
 * @param confidence Die Konfidenz der Vorhersage (0.0-1.0)
 * @param probabilities Array mit Wahrscheinlichkeiten für alle Klassen
 */
void ts_add_prediction(int predicted_class, float confidence, const float probabilities[TS_NUM_CLASSES]) {{
    // Speichere Vorhersage im Ringpuffer
    prediction_history[history_index] = predicted_class;
    confidence_history[history_index] = confidence;
    
    // Speichere alle Klassenwahrscheinlichkeiten
    for (int i = 0; i < TS_NUM_CLASSES; i++) {{
        probability_history[history_index][i] = probabilities[i];
    }}
    
    // Aktualisiere Index und Zähler
    history_index = (history_index + 1) % TS_WINDOW_SIZE;
    if (history_count < TS_WINDOW_SIZE) {{
        history_count++;
    }}
}}

/**
 * Berechnet die geglättete Vorhersage basierend auf den letzten N Vorhersagen
 * 
 * @param smoothed_probabilities Array für die geglätteten Wahrscheinlichkeiten (kann NULL sein)
 * @return Der Index der geglätteten Klasse
 */
int ts_get_smoothed_prediction(float smoothed_probabilities[TS_NUM_CLASSES]) {{
    // Bei nur einer Vorhersage, keine Glättung anwenden
    if (history_count <= 1) {{
        if (smoothed_probabilities != NULL) {{
            for (int i = 0; i < TS_NUM_CLASSES; i++) {{
                int idx = (history_index - 1 + TS_WINDOW_SIZE) % TS_WINDOW_SIZE;
                smoothed_probabilities[i] = probability_history[idx][i];
            }}
        }}
        return prediction_history[(history_index - 1 + TS_WINDOW_SIZE) % TS_WINDOW_SIZE];
    }}
    
""".format(
        strategy=strategy,
        window_size=window_size,
        num_classes=6  # Anzahl der Klassen aus MODEL_CLASS_NAMES
    )
    
    # Strategie-spezifische Implementierung
    if strategy == "majority_vote":
        c_code += """
    // Implementierung der Mehrheitsentscheidung
    int counts[TS_NUM_CLASSES] = {0};
    int max_count = 0;
    int max_class = 0;
    
    // Zähle Häufigkeiten
    for (int i = 0; i < history_count; i++) {
        int pred = prediction_history[i];
        counts[pred]++;
        
        if (counts[pred] > max_count) {
            max_count = counts[pred];
            max_class = pred;
        }
    }
    
    // Berechne durchschnittliche Konfidenz für die Mehrheitsklasse
    float total_confidence = 0.0f;
    int conf_count = 0;
    
    for (int i = 0; i < history_count; i++) {
        if (prediction_history[i] == max_class) {
            total_confidence += confidence_history[i];
            conf_count++;
        }
    }
    
    // Berechne geglättete Wahrscheinlichkeiten (Durchschnitt)
    if (smoothed_probabilities != NULL) {
        // Initialisiere mit 0
        for (int i = 0; i < TS_NUM_CLASSES; i++) {
            smoothed_probabilities[i] = 0.0f;
        }
        
        // Summiere Wahrscheinlichkeiten
        for (int i = 0; i < history_count; i++) {
            for (int j = 0; j < TS_NUM_CLASSES; j++) {
                smoothed_probabilities[j] += probability_history[i][j];
            }
        }
        
        // Teile durch Anzahl der Vorhersagen
        for (int i = 0; i < TS_NUM_CLASSES; i++) {
            smoothed_probabilities[i] /= history_count;
        }
    }
    
    return max_class;
"""
    elif strategy == "moving_average":
        c_code += """
    // Implementierung des gleitenden Mittelwerts
    float avg_probabilities[TS_NUM_CLASSES] = {0};
    
    // Berechne durchschnittliche Wahrscheinlichkeiten
    for (int i = 0; i < TS_NUM_CLASSES; i++) {
        for (int j = 0; j < history_count; j++) {
            avg_probabilities[i] += probability_history[j][i];
        }
        avg_probabilities[i] /= history_count;
    }
    
    // Finde die Klasse mit der höchsten durchschnittlichen Wahrscheinlichkeit
    int max_class = 0;
    float max_prob = avg_probabilities[0];
    
    for (int i = 1; i < TS_NUM_CLASSES; i++) {
        if (avg_probabilities[i] > max_prob) {
            max_prob = avg_probabilities[i];
            max_class = i;
        }
    }
    
    // Kopiere die geglätteten Wahrscheinlichkeiten
    if (smoothed_probabilities != NULL) {
        for (int i = 0; i < TS_NUM_CLASSES; i++) {
            smoothed_probabilities[i] = avg_probabilities[i];
        }
    }
    
    return max_class;
"""
    
    # Abschluss der Funktion
    c_code += """
}

/**
 * Gibt die Konfidenz der geglätteten Vorhersage zurück
 * 
 * @return Die Konfidenz der geglätteten Vorhersage (0.0-1.0)
 */
float ts_get_smoothed_confidence(void) {
    float smoothed_probabilities[TS_NUM_CLASSES];
    int smoothed_class = ts_get_smoothed_prediction(smoothed_probabilities);
    return smoothed_probabilities[smoothed_class];
}

/**
 * Setzt den Temporal-Smoothing-Puffer zurück
 */
void ts_reset(void) {
    history_count = 0;
    history_index = 0;
}
"""
    
    return c_code