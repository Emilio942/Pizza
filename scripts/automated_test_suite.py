#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatisierte Test-Suite für das Pizza-Erkennungssystem

Dieses Skript führt Tests mit simulierten und echten Pizza-Bildern durch,
um die Genauigkeit des Klassifikationsmodells zu prüfen und Regressionstests 
zu ermöglichen.

Es generiert einen detaillierten Bericht über die Genauigkeit und 
Testabdeckung für verschiedene Klassen und Bedingungen.

Verwendung:
    python scripts/automated_test_suite.py [--data-dir DIR] [--model-path FILE] 
                                          [--output-dir DIR] [--generate-images]
                                          [--num-test-images N] [--detailed-report]

Optionen:
    --data-dir: Verzeichnis mit Testdaten (Standard: data/test)
    --model-path: Pfad zum trainierten Modell (Standard: models/pizza_model_int8.pth)
    --output-dir: Verzeichnis für Testergebnisse (Standard: output/test_results)
    --generate-images: Testbilder automatisch generieren, wenn sie nicht existieren
    --num-test-images: Anzahl der zu generierenden Testbilder pro Klasse
    --detailed-report: Generiert einen ausführlichen HTML-Report mit Beispielbildern
"""

import os
import sys
import json
import argparse
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import cv2

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importiere Module aus dem Pizza-Projekt
from src.pizza_detector import (
    RP2040Config, create_optimized_dataloaders, load_model, evaluate_model,
    PizzaDatasetAnalysis
)
from src.utils.types import InferenceResult, ModelMetrics # MODIFIED: Corrected import path
from src.metrics import calculate_metrics, visualize_confusion_matrix
from src.constants import (
    CLASS_NAMES, CLASS_COLORS, PROJECT_ROOT, MODELS_DIR, OUTPUT_DIR,
    IMAGE_EXTENSIONS, IMAGE_MEAN, IMAGE_STD, INPUT_SIZE
)
from src.augmentation import (
    add_noise, adjust_brightness, adjust_contrast, rotate_image,
    apply_blur, apply_shadow, apply_highlight
)

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automated_test_suite.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automated_test_suite")

# Stellen Sie sicher, dass Warnungen als Fehler behandelt werden können
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Konstanten
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")
TEST_RESULTS_DIR = os.path.join(OUTPUT_DIR, "test_results")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "pizza_model_int8.pth")
NUM_TEST_IMAGES_PER_CLASS = 20
LIGHTING_CONDITIONS = [
    "normal",           # Normale Beleuchtung
    "dark",             # Dunkle Bedingungen
    "bright",           # Überbelichtet
    "uneven",           # Ungleichmäßige Beleuchtung
    "low_contrast"      # Geringer Kontrast
]


def setup_test_environment(args):
    """
    Richtet die Testumgebung ein, erstellt Verzeichnisse und prüft Voraussetzungen
    
    Args:
        args: Kommandozeilenargumente
    
    Returns:
        Tuple: (test_dir, output_dir, model_path)
    """
    # Prüfe und erstelle Testdatenverzeichnis
    test_dir = args.data_dir
    if not os.path.exists(test_dir):
        logger.info(f"Erstelle Testdatenverzeichnis: {test_dir}")
        os.makedirs(test_dir, exist_ok=True)
    
    # Prüfe und erstelle Unterordner für Lichtverhältnisse
    for condition in LIGHTING_CONDITIONS:
        condition_dir = os.path.join(test_dir, condition)
        if not os.path.exists(condition_dir):
            os.makedirs(condition_dir, exist_ok=True)
        
        # Klassenverzeichnisse
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(condition_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)
    
    # Prüfe und erstelle Ausgabeverzeichnis
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        logger.info(f"Erstelle Ausgabeverzeichnis: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Prüfe Modellpfad
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"Modell nicht gefunden: {model_path}")
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
    
    return test_dir, output_dir, model_path


def generate_test_images(test_dir: str, num_images_per_class: int = 20):
    """
    Generiert simulierte Testbilder für verschiedene Lichtverhältnisse
    
    Args:
        test_dir: Verzeichnis für die Testdaten
        num_images_per_class: Anzahl der Bilder pro Klasse und Lichtverhältnis
    """
    logger.info("Generiere simulierte Testbilder für verschiedene Lichtverhältnisse...")
    
    # Prüfe, ob Quelldaten verfügbar sind
    augmented_dir = os.path.join(PROJECT_ROOT, "data", "augmented")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    
    source_dir = augmented_dir if os.path.exists(augmented_dir) else raw_dir
    if not os.path.exists(source_dir):
        logger.error(f"Keine Quelldaten gefunden in {augmented_dir} oder {raw_dir}")
        raise FileNotFoundError(f"Keine Quelldaten gefunden")
    
    logger.info(f"Verwende Quelldaten aus: {source_dir}")
    
    # Sammle vorhandene Bilder nach Klasse
    source_images = {}
    
    # Wenn Quellordner nach Klassen strukturiert ist
    if os.path.isdir(os.path.join(source_dir, CLASS_NAMES[0])):
        for class_name in CLASS_NAMES:
            source_images[class_name] = []
            class_dir = os.path.join(source_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        source_images[class_name].append(os.path.join(class_dir, filename))
    # Sonst versuche, Bilder anhand von Namenskonventionen zuzuordnen
    else:
        for class_name in CLASS_NAMES:
            source_images[class_name] = []
            for filename in os.listdir(source_dir):
                if class_name in filename.lower() and any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    source_images[class_name].append(os.path.join(source_dir, filename))
    
    # Prüfe, ob Bilder für alle Klassen gefunden wurden
    missing_classes = [c for c in CLASS_NAMES if not source_images.get(c)]
    if missing_classes:
        logger.warning(f"Keine Bilder gefunden für Klassen: {', '.join(missing_classes)}")
        logger.warning("Generiere synthetische Bilder für fehlende Klassen")
        
        # Erstelle einfache synthetische Bilder für fehlende Klassen
        for class_name in missing_classes:
            source_images[class_name] = []
            synth_dir = os.path.join(PROJECT_ROOT, "data", "synthetic", class_name)
            os.makedirs(synth_dir, exist_ok=True)
            
            # Generiere einfache Bilder mit unterschiedlichen Formen/Farben je nach Klasse
            for i in range(max(10, num_images_per_class // 2)):
                img = Image.new('RGB', (64, 64), (240, 240, 240))
                draw = ImageDraw.Draw(img)
                
                if class_name == "basic":
                    # Einfacher Kreis für basic
                    draw.ellipse([(10, 10), (54, 54)], fill=(200, 180, 150))
                elif class_name == "burnt":
                    # Dunkler Kreis für burnt
                    draw.ellipse([(10, 10), (54, 54)], fill=(100, 60, 30))
                elif class_name == "perfect":
                    # Goldener Kreis für perfect
                    draw.ellipse([(10, 10), (54, 54)], fill=(220, 180, 120))
                    # Topping-Punkte
                    for _ in range(random.randint(5, 15)):
                        x = random.randint(15, 50)
                        y = random.randint(15, 50)
                        r = random.randint(1, 3)
                        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(200, 0, 0))
                else:
                    # Generisches Bild für andere Klassen
                    draw.rectangle([(10, 10), (54, 54)], fill=(180, 160, 140))
                
                save_path = os.path.join(synth_dir, f"synthetic_{class_name}_{i:03d}.png")
                img.save(save_path)
                source_images[class_name].append(save_path)
    
    # Generiere Testbilder für jede Klasse und jedes Lichtverhältnis
    for condition in tqdm(LIGHTING_CONDITIONS, desc="Verarbeite Lichtverhältnisse"):
        for class_name in tqdm(CLASS_NAMES, desc=f"Generiere {condition}-Bilder", leave=False):
            # Zielverzeichnis
            target_dir = os.path.join(test_dir, condition, class_name)
            
            # Prüfe, ob bereits genug Bilder vorhanden sind
            existing_images = [f for f in os.listdir(target_dir) 
                              if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
            
            if len(existing_images) >= num_images_per_class:
                logger.info(f"Bereits {len(existing_images)} Bilder für {class_name} unter {condition}-Bedingungen vorhanden")
                continue
            
            # Generiere oder kopiere Bilder
            num_to_generate = num_images_per_class - len(existing_images)
            logger.info(f"Generiere {num_to_generate} neue Bilder für {class_name} unter {condition}-Bedingungen")
            
            # Wähle zufällige Quellbilder (mit Zurücklegen, wenn nötig)
            source_choices = random.choices(source_images[class_name], k=num_to_generate)
            
            for i, src_path in enumerate(source_choices):
                try:
                    # Lade Quellbild
                    img = Image.open(src_path).convert('RGB')
                    
                    # Wende Transformationen je nach Bedingung an
                    if condition == "dark":
                        factor = random.uniform(0.3, 0.6)
                        img = adjust_brightness(img, factor)
                    elif condition == "bright":
                        factor = random.uniform(1.4, 1.8)
                        img = adjust_brightness(img, factor)
                    elif condition == "uneven":
                        img = apply_shadow(img, intensity=random.uniform(0.4, 0.7))
                    elif condition == "low_contrast":
                        factor = random.uniform(0.4, 0.7)
                        img = adjust_contrast(img, factor)
                    
                    # Zusätzliches zufälliges Rauschen, leichte Rotation
                    if random.random() < 0.7:
                        img = add_noise(img, intensity=random.uniform(0.01, 0.05))
                    
                    if random.random() < 0.5:
                        angle = random.uniform(-10, 10)
                        img = rotate_image(img, angle)
                    
                    # Speichere als Testbild
                    filename = f"test_{class_name}_{condition}_{len(existing_images) + i:03d}.png"
                    save_path = os.path.join(target_dir, filename)
                    img.save(save_path)
                    
                except Exception as e:
                    logger.error(f"Fehler beim Generieren von Testbild aus {src_path}: {e}")
    
    # Zähle die generierten Bilder
    total_images = 0
    for condition in LIGHTING_CONDITIONS:
        for class_name in CLASS_NAMES:
            target_dir = os.path.join(test_dir, condition, class_name)
            num_images = len([f for f in os.listdir(target_dir) 
                             if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)])
            total_images += num_images
    
    logger.info(f"Test-Suite enthält nun insgesamt {total_images} Testbilder in {len(LIGHTING_CONDITIONS)} Lichtverhältnissen")


def evaluate_test_images(test_dir: str, model_path: str, output_dir: str):
    """
    Evaluiert das Modell auf den Testbildern und erfasst die Ergebnisse
    
    Args:
        test_dir: Verzeichnis mit den Testbildern
        model_path: Pfad zum trainierten Modell
        output_dir: Verzeichnis für die Ausgabe der Ergebnisse
    
    Returns:
        Dict: Ergebnisse nach Lichtverhältnissen und Klassen
    """
    logger.info(f"Evaluiere Modell {model_path} auf Testbildern...")
    
    # Lade Konfiguration und Modell
    config = RP2040Config()
    model = load_model(model_path, config, quantized=model_path.endswith('int8.pth'))
    model.eval()
    
    # Dictionary für die Ergebnisse
    results = {
        "overall": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": None,
            "total_images": 0,
            "correctly_classified": 0,
            "per_class": {}
        }
    }
    
    # Initialisiere Ergebnisstruktur für jede Bedingung und Klasse
    for condition in LIGHTING_CONDITIONS:
        results[condition] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": None,
            "total_images": 0,
            "correctly_classified": 0,
            "per_class": {}
        }
        
        for class_name in CLASS_NAMES:
            results[condition]["per_class"][class_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "total_images": 0,
                "correctly_classified": 0,
                "example_correct": None,
                "example_incorrect": None
            }
    
    # Initialisiere overall per_class
    for class_name in CLASS_NAMES:
        results["overall"]["per_class"][class_name] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_images": 0,
            "correctly_classified": 0
        }
    
    # Vorverarbeitungsfunktion für Bilder
    def preprocess_image(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize(INPUT_SIZE)
        img_array = np.array(img) / 255.0
        # Normalisieren
        img_array = (img_array - np.array(IMAGE_MEAN)) / np.array(IMAGE_STD)
        # Zu Tensor konvertieren
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
        return img_tensor
    
    # Funktion für Bildklassifikation
    def classify_image(img_tensor):
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1).item()
            confidence = probs[0, predicted].item()
            return predicted, confidence, probs.squeeze().tolist()
    
    # Evaluiere Testbilder pro Lichtverhältnis und Klasse
    all_true_labels = []
    all_pred_labels = []
    
    for condition in tqdm(LIGHTING_CONDITIONS, desc="Evaluiere Lichtverhältnisse"):
        condition_true_labels = []
        condition_pred_labels = []
        
        for class_idx, class_name in enumerate(tqdm(CLASS_NAMES, desc=f"Evaluiere {condition}", leave=False)):
            class_dir = os.path.join(test_dir, condition, class_name)
            test_images = [f for f in os.listdir(class_dir) 
                          if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
            
            # Falls keine Bilder vorhanden, überspringe
            if not test_images:
                logger.warning(f"Keine Testbilder für {class_name} unter {condition}-Bedingungen")
                continue
            
            true_class_idx = CLASS_NAMES.index(class_name)
            results[condition]["per_class"][class_name]["total_images"] = len(test_images)
            results["overall"]["per_class"][class_name]["total_images"] += len(test_images)
            results[condition]["total_images"] += len(test_images)
            results["overall"]["total_images"] += len(test_images)
            
            # Speichere für jede Klasse ein korrektes und ein inkorrektes Beispiel
            correct_examples = []
            incorrect_examples = []
            
            for img_file in test_images:
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # Vorverarbeitung und Klassifikation
                    img_tensor = preprocess_image(img_path)
                    predicted_class_idx, confidence, probabilities = classify_image(img_tensor)
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    
                    # Sammle Labels für Confusion Matrix
                    condition_true_labels.append(true_class_idx)
                    condition_pred_labels.append(predicted_class_idx)
                    all_true_labels.append(true_class_idx)
                    all_pred_labels.append(predicted_class_idx)
                    
                    # Prüfe, ob korrekt klassifiziert
                    is_correct = (predicted_class_idx == true_class_idx)
                    
                    # Aktualisiere Zähler
                    if is_correct:
                        results[condition]["per_class"][class_name]["correctly_classified"] += 1
                        results["overall"]["per_class"][class_name]["correctly_classified"] += 1
                        results[condition]["correctly_classified"] += 1
                        results["overall"]["correctly_classified"] += 1
                        correct_examples.append((img_path, confidence))
                    else:
                        incorrect_examples.append((img_path, predicted_class, confidence))
                
                except Exception as e:
                    logger.error(f"Fehler bei der Klassifikation von {img_path}: {e}")
            
            # Speichere Beispielbilder
            if correct_examples:
                # Wähle Beispiel mit höchster Konfidenz
                correct_examples.sort(key=lambda x: x[1], reverse=True)
                results[condition]["per_class"][class_name]["example_correct"] = correct_examples[0][0]
            
            if incorrect_examples:
                # Wähle Beispiel mit höchster Konfidenz
                incorrect_examples.sort(key=lambda x: x[2], reverse=True)
                results[condition]["per_class"][class_name]["example_incorrect"] = (
                    incorrect_examples[0][0], incorrect_examples[0][1]
                )
        
        # Berechne Metriken für die Bedingung
        if condition_true_labels:
            condition_metrics = calculate_metrics(
                condition_true_labels, condition_pred_labels, CLASS_NAMES
            )
            
            results[condition]["accuracy"] = condition_metrics.accuracy
            results[condition]["precision"] = condition_metrics.precision
            results[condition]["recall"] = condition_metrics.recall
            results[condition]["f1_score"] = condition_metrics.f1_score
            results[condition]["confusion_matrix"] = condition_metrics.confusion_matrix.tolist()
            
            # Speichere klassenspezifische Metriken
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                condition_true_labels, condition_pred_labels, average=None, labels=range(len(CLASS_NAMES))
            )
            
            for i, class_name in enumerate(CLASS_NAMES):
                if results[condition]["per_class"][class_name]["total_images"] > 0:
                    results[condition]["per_class"][class_name]["precision"] = float(precision_per_class[i])
                    results[condition]["per_class"][class_name]["recall"] = float(recall_per_class[i])
                    results[condition]["per_class"][class_name]["f1_score"] = float(f1_per_class[i])
    
    # Berechne Gesamtmetriken
    if all_true_labels:
        overall_metrics = calculate_metrics(all_true_labels, all_pred_labels, CLASS_NAMES)
        
        results["overall"]["accuracy"] = overall_metrics.accuracy
        results["overall"]["precision"] = overall_metrics.precision
        results["overall"]["recall"] = overall_metrics.recall
        results["overall"]["f1_score"] = overall_metrics.f1_score
        results["overall"]["confusion_matrix"] = overall_metrics.confusion_matrix.tolist()
        
        # Speichere klassenspezifische Metriken
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average=None, labels=range(len(CLASS_NAMES))
        )
        
        for i, class_name in enumerate(CLASS_NAMES):
            if results["overall"]["per_class"][class_name]["total_images"] > 0:
                results["overall"]["per_class"][class_name]["precision"] = float(precision_per_class[i])
                results["overall"]["per_class"][class_name]["recall"] = float(recall_per_class[i])
                results["overall"]["per_class"][class_name]["f1_score"] = float(f1_per_class[i])
    
    # Speichere die Ergebnisse als JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluierung abgeschlossen. Ergebnisse gespeichert in {results_file}")
    
    return results


def generate_html_report(results, test_dir, output_dir, model_path):
    """
    Generiert einen detaillierten HTML-Bericht mit Visualisierungen
    
    Args:
        results: Evaluierungsergebnisse
        test_dir: Verzeichnis mit den Testbildern
        output_dir: Verzeichnis für die Ausgabe
        model_path: Pfad zum verwendeten Modell
    
    Returns:
        str: Pfad zum generierten Bericht
    """
    logger.info("Generiere detaillierten HTML-Bericht...")
    
    # Erstelle Verzeichnis für Bilder im Bericht
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    images_dir = os.path.join(report_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Generiere Confusion Matrix-Visualisierungen
    cm_overall_path = os.path.join(images_dir, "confusion_matrix_overall.png")
    if results["overall"]["confusion_matrix"]:
        cm_array = np.array(results["overall"]["confusion_matrix"])
        plt.figure(figsize=(10, 8))
        visualize_confusion_matrix(cm_array, CLASS_NAMES)
        plt.tight_layout()
        plt.savefig(cm_overall_path)
        plt.close()
    
    cm_per_condition = {}
    for condition in LIGHTING_CONDITIONS:
        if results[condition]["confusion_matrix"]:
            cm_path = os.path.join(images_dir, f"confusion_matrix_{condition}.png")
            cm_array = np.array(results[condition]["confusion_matrix"])
            plt.figure(figsize=(10, 8))
            visualize_confusion_matrix(cm_array, CLASS_NAMES)
            plt.title(f"Confusion Matrix - {condition.capitalize()}")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            cm_per_condition[condition] = f"images/confusion_matrix_{condition}.png"
    
    # Balkendiagramm für Genauigkeit nach Bedingung
    accuracy_path = os.path.join(images_dir, "accuracy_by_condition.png")
    conditions = LIGHTING_CONDITIONS
    accuracies = [results[c]["accuracy"] * 100 for c in conditions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, accuracies, color=[plt.cm.viridis(i/len(conditions)) for i in range(len(conditions))])
    
    # Beschriftungen hinzufügen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title("Genauigkeit nach Lichtverhältnis")
    plt.xlabel("Lichtverhältnis")
    plt.ylabel("Genauigkeit (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_path)
    plt.close()
    
    # F1-Score nach Klasse und Bedingung (Heatmap)
    f1_heatmap_path = os.path.join(images_dir, "f1_score_heatmap.png")
    
    f1_data = []
    for class_name in CLASS_NAMES:
        class_f1 = []
        for condition in conditions:
            if results[condition]["per_class"][class_name]["total_images"] > 0:
                class_f1.append(results[condition]["per_class"][class_name]["f1_score"])
            else:
                class_f1.append(0)  # Keine Daten
        f1_data.append(class_f1)
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(f1_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='F1-Score')
    
    # Achsenbeschriftungen
    plt.xticks(range(len(conditions)), [c.capitalize() for c in conditions], rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    
    plt.title("F1-Score nach Klasse und Lichtverhältnis")
    plt.tight_layout()
    plt.savefig(f1_heatmap_path)
    plt.close()
    
    # Kopiere alle Beispielbilder in das Berichtsverzeichnis und erstelle Verweise
    examples = {}
    for condition in LIGHTING_CONDITIONS:
        examples[condition] = {}
        for class_name in CLASS_NAMES:
            examples[condition][class_name] = {"correct": None, "incorrect": None}
            
            # Korrektes Beispiel
            correct_example = results[condition]["per_class"][class_name].get("example_correct")
            if correct_example:
                filename = f"example_{condition}_{class_name}_correct.png"
                dest_path = os.path.join(images_dir, filename)
                shutil.copy(correct_example, dest_path)
                examples[condition][class_name]["correct"] = f"images/{filename}"
            
            # Inkorrektes Beispiel
            incorrect_example = results[condition]["per_class"][class_name].get("example_incorrect")
            if incorrect_example:
                img_path, pred_class = incorrect_example
                filename = f"example_{condition}_{class_name}_incorrect_{pred_class}.png"
                dest_path = os.path.join(images_dir, filename)
                shutil.copy(img_path, dest_path)
                examples[condition][class_name]["incorrect"] = {
                    "path": f"images/{filename}",
                    "predicted_as": pred_class
                }
    
    # Jetzt generiere den HTML-Bericht
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pizza-Erkennungssystem: Automatisierte Test-Suite</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #3498db;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        .metrics-table th {{
            background-color: #f2f2f2;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metrics-table tr:hover {{
            background-color: #f1f1f1;
        }}
        .good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .bad {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .examples {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .example-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            width: 220px;
        }}
        .example-card img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
        }}
        .example-card .caption {{
            padding: 10px;
            background-color: #f8f9fa;
        }}
        .example-card.correct {{
            border-color: #27ae60;
        }}
        .example-card.incorrect {{
            border-color: #e74c3c;
        }}
        .tabs {{
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            background-color: #f1f1f1;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }}
        .tab.active {{
            background-color: #3498db;
            color: white;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-top: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 0.8em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pizza-Erkennungssystem: Automatisierte Test-Suite</h1>
            <p>Generiert am {timestamp}</p>
        </div>
        
        <div class="summary">
            <h2>Zusammenfassung</h2>
            <p><strong>Modell:</strong> {os.path.basename(model_path)}</p>
            <p><strong>Gesamtgenauigkeit:</strong> <span class="{
                'good' if results['overall']['accuracy'] >= 0.8 else 
                'medium' if results['overall']['accuracy'] >= 0.6 else 
                'bad'
            }">{results['overall']['accuracy']*100:.1f}%</span></p>
            <p><strong>F1-Score:</strong> {results['overall']['f1_score']:.3f}</p>
            <p><strong>Anzahl der Testbilder:</strong> {results['overall']['total_images']}</p>
            <p><strong>Korrekt klassifiziert:</strong> {results['overall']['correctly_classified']} ({results['overall']['correctly_classified']/results['overall']['total_images']*100:.1f}%)</p>
        </div>
        
        <div class="visualization">
            <h3>Gesamtgenauigkeit nach Lichtverhältnis</h3>
            <img src="images/accuracy_by_condition.png" alt="Accuracy by Condition" style="max-width:100%;">
        </div>
        
        <div class="visualization">
            <h3>F1-Score nach Klasse und Lichtverhältnis</h3>
            <img src="images/f1_score_heatmap.png" alt="F1-Score Heatmap" style="max-width:100%;">
        </div>
        
        <div class="visualization">
            <h3>Confusion Matrix (Gesamtdatensatz)</h3>
            <img src="images/confusion_matrix_overall.png" alt="Confusion Matrix" style="max-width:100%;">
        </div>
        
        <h2>Detaillierte Ergebnisse</h2>
        
        <div class="tabs">
            <div class="tab active" onclick="openTab(event, 'tab-overall')">Gesamtergebnis</div>
            
            {' '.join([
                f'<div class="tab" onclick="openTab(event, \'tab-{condition}\')">{condition.capitalize()}</div>'
                for condition in LIGHTING_CONDITIONS
            ])}
            
        </div>
        
        <div id="tab-overall" class="tab-content active">
            <h3>Gesamtergebnisse</h3>
            
            <table class="metrics-table">
                <tr>
                    <th>Klasse</th>
                    <th>Präzision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Bilder</th>
                    <th>Korrekt</th>
                    <th>Genauigkeit</th>
                </tr>
                
                {
                ''.join([
                f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{results['overall']['per_class'][class_name]['precision']:.3f}</td>
                    <td>{results['overall']['per_class'][class_name]['recall']:.3f}</td>
                    <td>{results['overall']['per_class'][class_name]['f1_score']:.3f}</td>
                    <td>{results['overall']['per_class'][class_name]['total_images']}</td>
                    <td>{results['overall']['per_class'][class_name]['correctly_classified']}</td>
                    <td class="{
                        'good' if results['overall']['per_class'][class_name]['correctly_classified'] / 
                                 results['overall']['per_class'][class_name]['total_images'] >= 0.8 else 
                        'medium' if results['overall']['per_class'][class_name]['correctly_classified'] / 
                                   results['overall']['per_class'][class_name]['total_images'] >= 0.6 else 
                        'bad'
                    }">{
                        results['overall']['per_class'][class_name]['correctly_classified'] / 
                        results['overall']['per_class'][class_name]['total_images'] * 100 if 
                        results['overall']['per_class'][class_name]['total_images'] > 0 else 0
                    :.1f}%</td>
                </tr>
                """
                for class_name in CLASS_NAMES if results['overall']['per_class'][class_name]['total_images'] > 0
                ])
                }
            </table>
        </div>
        
        {
        ''.join([
        f"""
        <div id="tab-{condition}" class="tab-content">
            <h3>Ergebnisse für {condition.capitalize()}</h3>
            
            <div class="summary">
                <p><strong>Genauigkeit:</strong> <span class="{
                    'good' if results[condition]['accuracy'] >= 0.8 else 
                    'medium' if results[condition]['accuracy'] >= 0.6 else 
                    'bad'
                }">{results[condition]['accuracy']*100:.1f}%</span></p>
                <p><strong>F1-Score:</strong> {results[condition]['f1_score']:.3f}</p>
                <p><strong>Testbilder:</strong> {results[condition]['total_images']}</p>
            </div>
            
            <div class="visualization">
                <h4>Confusion Matrix</h4>
                <img src="{cm_per_condition.get(condition, '')}" alt="Confusion Matrix {condition}" style="max-width:100%;">
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Klasse</th>
                    <th>Präzision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Bilder</th>
                    <th>Korrekt</th>
                    <th>Genauigkeit</th>
                </tr>
                
                {
                ''.join([
                f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{results[condition]['per_class'][class_name]['precision']:.3f}</td>
                    <td>{results[condition]['per_class'][class_name]['recall']:.3f}</td>
                    <td>{results[condition]['per_class'][class_name]['f1_score']:.3f}</td>
                    <td>{results[condition]['per_class'][class_name]['total_images']}</td>
                    <td>{results[condition]['per_class'][class_name]['correctly_classified']}</td>
                    <td class="{
                        'good' if results[condition]['per_class'][class_name]['correctly_classified'] / 
                                 results[condition]['per_class'][class_name]['total_images'] >= 0.8 else 
                        'medium' if results[condition]['per_class'][class_name]['correctly_classified'] / 
                                   results[condition]['per_class'][class_name]['total_images'] >= 0.6 else 
                        'bad'
                    }">{
                        results[condition]['per_class'][class_name]['correctly_classified'] / 
                        results[condition]['per_class'][class_name]['total_images'] * 100 if 
                        results[condition]['per_class'][class_name]['total_images'] > 0 else 0
                    :.1f}%</td>
                </tr>
                """
                for class_name in CLASS_NAMES if results[condition]['per_class'][class_name]['total_images'] > 0
                ])
                }
            </table>
            
            <h4>Beispielbilder</h4>
            
            {
            ''.join([
            f"""
            <h5>{class_name}</h5>
            <div class="examples">
                {
                f'''
                <div class="example-card correct">
                    <img src="{examples[condition][class_name]['correct']}" alt="Correct {class_name}">
                    <div class="caption">
                        <strong>Korrekt klassifiziert</strong><br>
                        Klasse: {class_name}
                    </div>
                </div>
                ''' if examples[condition][class_name]['correct'] else ''
                }
                
                {
                f'''
                <div class="example-card incorrect">
                    <img src="{examples[condition][class_name]['incorrect']['path']}" alt="Incorrect {class_name}">
                    <div class="caption">
                        <strong>Falsch klassifiziert</strong><br>
                        Tatsächlich: {class_name}<br>
                        Erkannt als: {examples[condition][class_name]['incorrect']['predicted_as']}
                    </div>
                </div>
                ''' if examples[condition][class_name]['incorrect'] else ''
                }
            </div>
            """
            for class_name in CLASS_NAMES if results[condition]['per_class'][class_name]['total_images'] > 0
            ])
            }
        </div>
        """
        for condition in LIGHTING_CONDITIONS
        ])
        }
        
        <div class="footer">
            <p>Automatisierte Test-Suite für das Pizza-Erkennungssystem</p>
            <p>Generiert am {timestamp}</p>
        </div>
    </div>
    
    <script>
        function openTab(evt, tabName) {{
            var i, tabContent, tabLinks;
            
            // Verstecke alle Tab-Inhalte
            tabContent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabContent.length; i++) {{
                tabContent[i].className = tabContent[i].className.replace(" active", "");
            }}
            
            // Entferne die aktive Klasse von allen Tabs
            tabLinks = document.getElementsByClassName("tab");
            for (i = 0; i < tabLinks.length; i++) {{
                tabLinks[i].className = tabLinks[i].className.replace(" active", "");
            }}
            
            // Aktiviere den aktuellen Tab und Inhalt
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
        }}
    </script>
</body>
</html>
"""
    
    # Speichere HTML-Bericht
    html_path = os.path.join(report_dir, "test_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML-Bericht erstellt: {html_path}")
    return html_path


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Automatisierte Test-Suite für das Pizza-Erkennungssystem')
    
    parser.add_argument('--data-dir', default=TEST_DATA_DIR, help='Verzeichnis mit Testdaten')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Pfad zum trainierten Modell')
    parser.add_argument('--output-dir', default=TEST_RESULTS_DIR, help='Verzeichnis für Testergebnisse')
    parser.add_argument('--generate-images', action='store_true', help='Testbilder automatisch generieren')
    parser.add_argument('--num-test-images', type=int, default=NUM_TEST_IMAGES_PER_CLASS, 
                        help='Anzahl der zu generierenden Testbilder pro Klasse')
    parser.add_argument('--detailed-report', action='store_true', help='Detaillierten HTML-Bericht erstellen')
    
    args = parser.parse_args()
    
    try:
        # Richte Testumgebung ein
        test_dir, output_dir, model_path = setup_test_environment(args)
        
        # Generiere Testbilder, falls gewünscht oder keine vorhanden
        if args.generate_images:
            generate_test_images(test_dir, args.num_test_images)
        else:
            # Prüfe, ob Testbilder vorhanden sind
            has_images = False
            for condition in LIGHTING_CONDITIONS:
                for class_name in CLASS_NAMES:
                    class_dir = os.path.join(test_dir, condition, class_name)
                    if os.path.exists(class_dir) and os.listdir(class_dir):
                        has_images = True
                        break
                if has_images:
                    break
            
            if not has_images:
                logger.warning("Keine Testbilder gefunden. Generiere automatisch...")
                generate_test_images(test_dir, args.num_test_images)
        
        # Evaluiere die Testbilder
        results = evaluate_test_images(test_dir, model_path, output_dir)
        
        # Erstelle Bericht
        if args.detailed_report:
            html_path = generate_html_report(results, test_dir, output_dir, model_path)
            logger.info(f"Öffne den HTML-Bericht, um die detaillierten Ergebnisse zu sehen: {html_path}")
        
        # Ausgabe der Gesamtergebnisse
        print("\n" + "="*80)
        print("ERGEBNISSE DER AUTOMATISIERTEN TEST-SUITE")
        print("="*80)
        print(f"Gesamtgenauigkeit: {results['overall']['accuracy']*100:.2f}%")
        print(f"F1-Score: {results['overall']['f1_score']:.4f}")
        print(f"Testbilder: {results['overall']['total_images']}")
        print(f"Korrekt klassifiziert: {results['overall']['correctly_classified']} ({results['overall']['correctly_classified']/results['overall']['total_images']*100:.2f}%)")
        
        print("\nGenauigkeit nach Lichtverhältnis:")
        for condition in LIGHTING_CONDITIONS:
            print(f"  {condition.capitalize()}: {results[condition]['accuracy']*100:.2f}% ({results[condition]['total_images']} Bilder)")
        
        print("\nProblematische Klassen:")
        problem_classes = []
        for class_name in CLASS_NAMES:
            if results['overall']['per_class'][class_name]['total_images'] > 0:
                accuracy = results['overall']['per_class'][class_name]['correctly_classified'] / results['overall']['per_class'][class_name]['total_images']
                if accuracy < 0.7:  # Weniger als 70% Genauigkeit
                    problem_classes.append((class_name, accuracy))
        
        if problem_classes:
            for class_name, accuracy in sorted(problem_classes, key=lambda x: x[1]):
                print(f"  {class_name}: {accuracy*100:.2f}%")
        else:
            print("  Keine Klassen unter 70% Genauigkeit")
        
        print("\nErgebnisse gespeichert in:")
        print(f"  {os.path.join(output_dir)}")
        
        if args.detailed_report:
            print(f"\nDetaillierter Bericht:\n  {html_path}")
        
        print("="*80)
        
        # Erfolgsbeurteilung
        if results['overall']['accuracy'] >= 0.8:
            logger.info("Test-Suite BESTANDEN: Genauigkeit ≥ 80%")
            return 0
        elif results['overall']['accuracy'] >= 0.7:
            logger.warning("Test-Suite TEILWEISE BESTANDEN: Genauigkeit ≥ 70%, aber < 80%")
            return 0
        else:
            logger.error("Test-Suite NICHT BESTANDEN: Genauigkeit < 70%")
            return 1
    
    except Exception as e:
        logger.error(f"Fehler in der automatisierten Test-Suite: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())