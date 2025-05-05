#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit-Tests für das Pizza-Erkennungsmodell

Diese Tests überprüfen die Genauigkeit und Stabilität des Modells
auf verschiedenen Testbildern und unter verschiedenen Bedingungen.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import warnings
import json
import logging
from typing import Dict, List, Any

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importiere notwendige Module
from src.pizza_detector import (
    RP2040Config, load_model, preprocess_image, get_prediction
)
from src.constants import (
    CLASS_NAMES, PROJECT_ROOT, MODELS_DIR, INPUT_SIZE
)
from scripts.automated_test_suite import (
    setup_test_environment, generate_test_images, evaluate_test_images
)

# Unterdrücke Warnungen für sauberere Testausgaben
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Konstanten
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "pizza_model_int8.pth")
MINIMAL_TEST_IMAGES = 5  # Minimale Anzahl von Testbildern pro Klasse
ACCURACY_THRESHOLD = 0.7  # Minimale Genauigkeit, um Tests zu bestehen

# Fixture für die Testumgebung
@pytest.fixture(scope="module")
def test_environment():
    """Bereitet die Testumgebung mit Testbildern vor"""
    
    class Args:
        data_dir = TEST_DATA_DIR
        model_path = DEFAULT_MODEL_PATH
        output_dir = os.path.join(PROJECT_ROOT, "output", "test_results")
        generate_images = True
        num_test_images = MINIMAL_TEST_IMAGES
        detailed_report = False
    
    args = Args()
    
    # Richte Testumgebung ein
    test_dir, output_dir, model_path = setup_test_environment(args)
    
    # Prüfe, ob Testbilder vorhanden sind, sonst erstelle sie
    has_images = False
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, "normal", class_name)
        if os.path.exists(class_dir) and len([f for f in os.listdir(class_dir) 
                                           if f.endswith(('.jpg', '.png', '.jpeg'))]) >= MINIMAL_TEST_IMAGES:
            has_images = True
        else:
            has_images = False
            break
    
    if not has_images:
        generate_test_images(test_dir, MINIMAL_TEST_IMAGES)
    
    return test_dir, output_dir, model_path

# Fixture für das Modell
@pytest.fixture(scope="module")
def model():
    """Lädt das Modell für Tests"""
    config = RP2040Config()
    return load_model(DEFAULT_MODEL_PATH, config, quantized=DEFAULT_MODEL_PATH.endswith('int8.pth'))

# Tests für die Pizza-Erkennung
class TestPizzaDetection:
    
    def test_model_loading(self, model):
        """Testet, ob das Modell korrekt geladen werden kann"""
        assert model is not None, "Modell konnte nicht geladen werden"
        assert isinstance(model, torch.nn.Module), "Geladenes Objekt ist kein PyTorch-Modell"
    
    def test_model_inference(self, model, test_environment):
        """Testet, ob Inferenz mit dem Modell funktioniert"""
        test_dir, _, _ = test_environment
        
        # Finde ein zufälliges Testbild für die Inferenz
        test_image_path = None
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(test_dir, "normal", class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
                if image_files:
                    test_image_path = os.path.join(class_dir, image_files[0])
                    break
        
        assert test_image_path is not None, "Kein Testbild gefunden"
        
        # Führe Inferenz durch
        img_tensor = preprocess_image(test_image_path, INPUT_SIZE)
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Prüfe Ausgabeform
        assert outputs is not None, "Modell lieferte keine Ausgabe"
        assert outputs.shape[1] == len(CLASS_NAMES), f"Ausgabeform falsch. Erwartet: {len(CLASS_NAMES)}, Bekommen: {outputs.shape[1]}"
    
    def test_model_predictions(self, model, test_environment):
        """Testet, ob Vorhersagen plausibel sind (Summe der Softmax-Werte ≈ 1)"""
        test_dir, _, _ = test_environment
        
        # Sammle ein paar Testbilder
        test_images = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(test_dir, "normal", class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
                if image_files:
                    test_images.append(os.path.join(class_dir, image_files[0]))
                    if len(test_images) >= 3:  # 3 Bilder sind genug für diesen Test
                        break
        
        assert len(test_images) > 0, "Keine Testbilder gefunden"
        
        for img_path in test_images:
            # Führe Vorhersage durch
            prediction, probs = get_prediction(model, img_path, INPUT_SIZE, CLASS_NAMES)
            
            # Prüfe Wahrscheinlichkeiten
            assert isinstance(prediction, str), "Vorhersage sollte ein String sein"
            assert prediction in CLASS_NAMES, f"Vorhersage '{prediction}' ist keine gültige Klasse"
            assert abs(sum(probs.values()) - 1.0) < 1e-5, f"Summe der Wahrscheinlichkeiten ist nicht 1.0: {sum(probs.values())}"
            assert all(0 <= p <= 1 for p in probs.values()), "Wahrscheinlichkeiten außerhalb [0,1]"
    
    def test_model_accuracy(self, test_environment):
        """Testet, ob das Modell eine Mindestgenauigkeit erreicht"""
        test_dir, output_dir, model_path = test_environment
        
        # Diese Funktion ist im Hauptskript definiert und führt eine vollständige Evaluierung durch
        results = evaluate_test_images(test_dir, model_path, output_dir)
        
        # Prüfe Gesamtgenauigkeit
        assert results["overall"]["accuracy"] >= ACCURACY_THRESHOLD, \
            f"Modellgenauigkeit ({results['overall']['accuracy']*100:.1f}%) unter Mindestschwelle ({ACCURACY_THRESHOLD*100:.1f}%)"
        
        # Prüfe auch F1-Score
        assert results["overall"]["f1_score"] >= ACCURACY_THRESHOLD - 0.1, \
            f"F1-Score ({results['overall']['f1_score']:.3f}) zu niedrig"
    
    def test_model_stability_across_conditions(self, test_environment):
        """Testet, ob das Modell unter verschiedenen Lichtverhältnissen stabil ist"""
        test_dir, output_dir, model_path = test_environment
        
        # Lade Ergebnisse der letzten Evaluierung
        result_files = [f for f in os.listdir(output_dir) if f.startswith("test_results_") and f.endswith(".json")]
        if not result_files:
            # Falls keine Ergebnisse vorhanden, führe Evaluierung durch
            results = evaluate_test_images(test_dir, model_path, output_dir)
        else:
            # Lade neuestes Ergebnis
            latest_result = sorted(result_files)[-1]
            with open(os.path.join(output_dir, latest_result), 'r') as f:
                results = json.load(f)
        
        # Prüfe Stabilität über verschiedene Bedingungen
        conditions = [c for c in results.keys() if c != "overall"]
        if not conditions:
            pytest.skip("Keine Bedingungsergebnisse gefunden")
        
        # Die Genauigkeit sollte in mindestens einer Bedingung über dem Schwellenwert liegen
        condition_accuracies = [results[c]["accuracy"] for c in conditions]
        assert max(condition_accuracies) >= ACCURACY_THRESHOLD, \
            f"Modell erreicht in keiner Bedingung die Mindestgenauigkeit. Beste: {max(condition_accuracies)*100:.1f}%"
        
        # Die Standardabweichung der Genauigkeiten sollte nicht zu hoch sein
        accuracy_std = np.std(condition_accuracies)
        assert accuracy_std <= 0.25, \
            f"Modellgenauigkeit variiert zu stark zwischen Bedingungen (Std: {accuracy_std:.3f})"
    
    def test_classes_balanced_accuracy(self, test_environment):
        """Testet, ob das Modell für alle Klassen ähnliche Genauigkeiten erreicht"""
        test_dir, output_dir, model_path = test_environment
        
        # Lade Ergebnisse der letzten Evaluierung
        result_files = [f for f in os.listdir(output_dir) if f.startswith("test_results_") and f.endswith(".json")]
        if not result_files:
            # Falls keine Ergebnisse vorhanden, führe Evaluierung durch
            results = evaluate_test_images(test_dir, model_path, output_dir)
        else:
            # Lade neuestes Ergebnis
            latest_result = sorted(result_files)[-1]
            with open(os.path.join(output_dir, latest_result), 'r') as f:
                results = json.load(f)
        
        # Prüfe Klassenbalance
        class_accuracies = []
        for class_name in CLASS_NAMES:
            if (results["overall"]["per_class"][class_name]["total_images"] > 0):
                class_acc = (results["overall"]["per_class"][class_name]["correctly_classified"] / 
                            results["overall"]["per_class"][class_name]["total_images"])
                class_accuracies.append(class_acc)
        
        if not class_accuracies:
            pytest.skip("Keine Klassengenauigkeiten verfügbar")
        
        # Die schlechteste Klasse sollte nicht zu schlecht sein
        assert min(class_accuracies) >= ACCURACY_THRESHOLD - 0.15, \
            f"Schlechteste Klassengenauigkeit zu niedrig: {min(class_accuracies)*100:.1f}%"
        
        # Der Unterschied zwischen bester und schlechtester Klasse sollte nicht zu groß sein
        accuracy_range = max(class_accuracies) - min(class_accuracies)
        assert accuracy_range <= 0.3, \
            f"Zu große Diskrepanz zwischen bester und schlechtester Klasse: {accuracy_range*100:.1f}%"


# Erlaube Ausführung der Tests direkt über pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])