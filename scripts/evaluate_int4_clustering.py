#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INT4 Quantisierung für geclusterte Modelle
-------------------------------------------
Dieses Skript quantisiert ein geclustertes Modell auf INT4-Genauigkeit und
evaluiert die Auswirkungen auf Genauigkeit und Speicherbedarf. Der Fokus liegt
auf der Untersuchung, wie Gewichts-Clustering die Effizienz der INT4-Quantisierung
verbessert.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Modulpfad für Import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)

# Eigene Module importieren
from scripts.model_optimization.int4_quantization import quantize_model_to_int4
from scripts.model_optimization.weight_pruning import evaluate_model, create_comparison_report
from src.pizza_detector import (
    MicroPizzaNet, create_optimized_dataloaders, MemoryEstimator
)
from config.config import DefaultConfig as Config

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("int4_quantization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_int4_with_clustering(clustered_model_path, output_dir, device='cpu'):
    """
    Evaluiert die Auswirkungen von Gewichts-Clustering auf die INT4-Quantisierung.
    
    Args:
        clustered_model_path: Pfad zum geclusterten Modell
        output_dir: Ausgabeverzeichnis
        device: Gerät für die Berechnung
    """
    # Konfiguration laden
    config = Config()
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    # Stelle sicher, dass cuda verfügbar ist, wenn angegeben
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA nicht verfügbar, wechsle zu CPU")
        device = 'cpu'
    
    logger.info(f"Verwende Gerät: {device}")
    
    # Lade Daten für Evaluation
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    
    # Anzahl der Klassen
    num_classes = len(class_names)
    logger.info(f"Anzahl Klassen: {num_classes}")
    
    # Lade geclustertes Modell
    logger.info(f"Lade geclustertes Modell von: {clustered_model_path}")
    model = MicroPizzaNet(num_classes=num_classes)
    model.load_state_dict(torch.load(clustered_model_path, map_location=device))
    model = model.to(device)
    
    # Konfigurationsobjekt mit Device aktualisieren
    config.DEVICE = device
    
    # Evaluiere geclustertes Modell
    logger.info("Evaluiere geclustertes Modell vor INT4-Quantisierung")
    clustered_model_stats = evaluate_model(model, val_loader, class_names, device)
    
    # Führe INT4-Quantisierung durch
    logger.info("Starte INT4-Quantisierung des geclusterten Modells")
    int4_output_dir = os.path.join(output_dir, "int4_quantized")
    int4_results = quantize_model_to_int4(
        model=model,
        val_loader=val_loader, 
        config=config,
        class_names=class_names,
        output_dir=int4_output_dir
    )
    
    # Erstelle Vergleichsbericht
    logger.info("Erstelle Vergleichsbericht")
    comparison_data = {
        "clustered_model": {
            "accuracy": clustered_model_stats["accuracy"],
            "model_size_kb": clustered_model_stats.get("model_size_kb", 0),
            "memory_usage_kb": clustered_model_stats.get("memory_usage_kb", 0),
            "avg_inference_time_ms": clustered_model_stats.get("avg_inference_time_ms", 0)
        },
        "int4_model": {
            "accuracy": int4_results["int4_model"]["accuracy"],
            "model_size_kb": int4_results["int4_model"]["memory_kb"],
            "memory_reduction": int4_results["memory_reduction"],
            "avg_inference_time_ms": int4_results["int4_model"]["inference_time_ms"]
        }
    }
    
    # Speichere Vergleichsbericht
    report_path = os.path.join(output_dir, "clustering_int4_comparison.json")
    with open(report_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Erstelle Visualisierung
    create_int4_clustering_visualization(comparison_data, output_dir)
    
    logger.info(f"Evaluierung abgeschlossen. Ergebnisse gespeichert unter: {report_path}")
    return comparison_data

def create_int4_clustering_visualization(comparison_data, output_dir):
    """
    Erstellt eine Visualisierung des Vergleichs zwischen dem geclusterten und
    dem INT4-quantisierten Modell.
    
    Args:
        comparison_data: Vergleichsdaten
        output_dir: Ausgabeverzeichnis
    """
    plt.figure(figsize=(12, 8))
    
    # Genauigkeitsvergleich
    plt.subplot(2, 2, 1)
    accuracies = [
        comparison_data["clustered_model"]["accuracy"],
        comparison_data["int4_model"]["accuracy"]
    ]
    plt.bar(["Geclustertes Modell", "INT4-Modell"], accuracies, color=['blue', 'orange'])
    plt.ylabel("Genauigkeit (%)")
    plt.title("Genauigkeitsvergleich")
    
    # Modellgrößenvergleich
    plt.subplot(2, 2, 2)
    sizes = [
        comparison_data["clustered_model"]["model_size_kb"],
        comparison_data["int4_model"]["model_size_kb"]
    ]
    plt.bar(["Geclustertes Modell", "INT4-Modell"], sizes, color=['blue', 'orange'])
    plt.ylabel("Modellgröße (KB)")
    plt.title("Speicherverbrauch")
    
    # Inferenzzeit
    plt.subplot(2, 2, 3)
    inf_times = [
        comparison_data["clustered_model"]["avg_inference_time_ms"],
        comparison_data["int4_model"]["avg_inference_time_ms"]
    ]
    plt.bar(["Geclustertes Modell", "INT4-Modell"], inf_times, color=['blue', 'orange'])
    plt.ylabel("Inferenzzeit (ms)")
    plt.title("Durchschnittliche Inferenzzeit")
    
    # Speicherreduktion
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f"Speicherreduktion durch INT4:\n{comparison_data['int4_model']['memory_reduction']:.2%}",
             horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "int4_clustering_comparison.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='INT4-Quantisierung für geclusterte Modelle')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Pfad zum geclusterten Modell')
    parser.add_argument('--output_dir', type=str, default='output/int4_evaluation', 
                        help='Ausgabeverzeichnis')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Gerät für die Berechnung (cuda oder cpu)')
    
    args = parser.parse_args()
    
    # Führe Evaluierung durch
    evaluate_int4_with_clustering(
        clustered_model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    logger.info("INT4-Quantisierungsevaluierung abgeschlossen!")
    
if __name__ == "__main__":
    main()
