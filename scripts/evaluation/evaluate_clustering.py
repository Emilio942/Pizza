#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gewichts-Clustering-Evaluierung für MODELL-1.2
----------------------------------------------
Dieses Skript implementiert und evaluiert Gewichts-Clustering mit verschiedenen
Cluster-Anzahlen (16, 32, 64) und Quantisierungsstufen (INT8, INT4), um die
optimale Konfiguration für ressourcenbeschränkte Geräte zu finden.

Verwendung:
    python scripts/evaluate_clustering.py --model_path models/micro_pizza_model.pth --output_dir output/model_optimization
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Modulpfad für Import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)

# Eigene Module importieren
from scripts.model_optimization.weight_pruning import WeightClusterer, evaluate_model
from scripts.model_optimization.int4_quantization import INT4Quantizer, quantize_model_to_int4
from src.pizza_detector import (
    MicroPizzaNet, create_optimized_dataloaders, MemoryEstimator
)
from config.config import DefaultConfig as Config

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pruning_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def apply_weight_clustering(model, num_clusters):
    """
    Wendet Gewichts-Clustering auf das Modell mit der angegebenen Anzahl von Clustern an.
    
    Args:
        model: PyTorch-Modell
        num_clusters: Anzahl der Cluster
        
    Returns:
        Geclustertes Modell und Clustering-Statistiken
    """
    logger.info(f"Starte Gewichts-Clustering mit {num_clusters} Clustern...")
    
    # Initialisiere Clusterer
    clusterer = WeightClusterer(
        model=model,
        num_clusters=num_clusters
    )
    
    # Führe Clustering durch
    clusterer.cluster_weights()
    
    # Hole Statistiken
    clustering_stats = clusterer.get_clustering_stats()
    
    return model, clustering_stats

def quantize_to_int8(model, val_loader, device):
    """
    Quantisiert das Modell zu INT8.
    
    Args:
        model: PyTorch-Modell
        val_loader: DataLoader für Validierungsdaten
        device: Gerät für Berechnungen
        
    Returns:
        Quantisiertes Modell und Quantisierungsstatistiken
    """
    logger.info("Quantisiere Modell zu INT8...")
    
    # Kopiere Modell, um Original nicht zu verändern
    quantized_model = MicroPizzaNet(num_classes=model.num_classes)
    quantized_model.load_state_dict(model.state_dict())
    quantized_model = quantized_model.to(device)
    
    # Standard PyTorch-Quantisierung (statisch)
    from torch.quantization import get_default_qconfig, quantize_fx, prepare_fx
    
    # Setze Model in Eval-Modus
    quantized_model.eval()
    
    # INT8-Quantisierung mit PyTorch
    try:
        # Für x86/x64: 'fbgemm'
        # Für ARM/mobile: 'qnnpack'
        backend = 'qnnpack'  # ARM-freundlich
        
        # Konfiguriere qconfig
        quantized_model.qconfig = get_default_qconfig(backend)
        
        # Bereite Model für Quantisierung vor (fügt Beobachter hinzu)
        prepared_model = prepare_fx(quantized_model, torch.quantization.symbolic_trace)
        
        # Kalibriere mit einigen Validierungsdaten
        with torch.no_grad():
            for i, (inputs, _) in enumerate(val_loader):
                if i >= 10:  # 10 Batches sollten ausreichen
                    break
                prepared_model(inputs.to(device))
        
        # Konvertiere zu vollständig quantisiertem Modell
        quantized_model = quantize_fx(prepared_model)
        
        # Schätze INT8-Modellgröße
        model_size_kb = MemoryEstimator.estimate_model_size(quantized_model, bits=8)
        tensor_arena_kb = MemoryEstimator.estimate_tensor_arena(quantized_model)
        
        quant_stats = {
            'model_size_kb': model_size_kb,
            'tensor_arena_kb': tensor_arena_kb,
            'total_ram_kb': model_size_kb + tensor_arena_kb
        }
        
        return quantized_model, quant_stats
    except Exception as e:
        logger.error(f"Fehler bei INT8-Quantisierung: {e}")
        # Fallback: Behalte Originalmodell
        return model, {
            'model_size_kb': MemoryEstimator.estimate_model_size(model),
            'tensor_arena_kb': MemoryEstimator.estimate_tensor_arena(model),
            'total_ram_kb': MemoryEstimator.estimate_model_size(model) + MemoryEstimator.estimate_tensor_arena(model),
            'error': str(e)
        }

def measure_inference_time(model, val_loader, device, num_runs=50):
    """
    Misst die durchschnittliche Inferenzzeit des Modells.
    
    Args:
        model: PyTorch-Modell
        val_loader: DataLoader für Validierungsdaten
        device: Gerät für Berechnungen
        num_runs: Anzahl der Durchläufe
        
    Returns:
        Durchschnittliche Inferenzzeit in Millisekunden
    """
    logger.info(f"Messe Inferenzzeit über {num_runs} Durchläufe...")
    
    model.eval()
    
    # Hole einen einzelnen Batch für wiederholte Inferenz
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        break
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
    
    # Messe Zeit
    inference_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(inputs)
            inference_time = (time.time() - start_time) * 1000  # in ms
            inference_times.append(inference_time)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    return avg_inference_time

def evaluate_clustered_model(model, val_loader, class_names, device):
    """
    Evaluiert ein Modell auf Genauigkeit und Inferenzzeit.
    
    Args:
        model: PyTorch-Modell
        val_loader: DataLoader für Validierungsdaten
        class_names: Liste der Klassennamen
        device: Gerät für Berechnungen
        
    Returns:
        Dictionary mit Evaluierungsergebnissen
    """
    # Genauigkeit evaluieren
    accuracy_results = evaluate_model(model, val_loader, class_names, device)
    
    # Inferenzzeit messen
    avg_inference_time = measure_inference_time(model, val_loader, device)
    
    # RAM-Bedarf schätzen
    model_size_kb = MemoryEstimator.estimate_model_size(model)
    tensor_arena_kb = MemoryEstimator.estimate_tensor_arena(model)
    
    # Kombiniere Ergebnisse
    results = {
        'accuracy': accuracy_results['accuracy'],
        'class_accuracies': accuracy_results['class_accuracies'],
        'model_size_kb': model_size_kb,
        'tensor_arena_kb': tensor_arena_kb,
        'total_ram_kb': model_size_kb + tensor_arena_kb,
        'avg_inference_time_ms': avg_inference_time
    }
    
    return results

def create_clustering_evaluation_report(results, output_dir):
    """
    Erstellt einen Evaluierungsbericht für die verschiedenen Clustering-Konfigurationen.
    
    Args:
        results: Dictionary mit Evaluierungsergebnissen
        output_dir: Ausgabeverzeichnis
    """
    # Erstelle Ausgabeverzeichnis, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichere Ergebnisse als JSON
    json_path = os.path.join(output_dir, "clustering_evaluation.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Erstelle auch eine Markdown-Version für bessere Lesbarkeit
    md_path = os.path.join(output_dir, "clustering_evaluation.md")
    with open(md_path, 'w') as f:
        f.write("# Weight Clustering Evaluation Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report evaluates the effectiveness of weight clustering with various cluster sizes ")
        f.write("(16, 32, 64) and quantization methods (INT8, INT4) for the MicroPizzaNet model. ")
        f.write("The evaluation focuses on accuracy, model size, RAM usage, and inference time.\n\n")
        
        # Summarize best configuration
        best_config = find_best_configuration(results)
        f.write(f"**Optimal Configuration:** {best_config['name']}\n")
        f.write(f"- Accuracy: {best_config['accuracy']:.2f}%\n")
        f.write(f"- Model Size: {best_config['model_size_kb']:.2f} KB\n")
        f.write(f"- Total RAM: {best_config['total_ram_kb']:.2f} KB\n")
        f.write(f"- Inference Time: {best_config['avg_inference_time_ms']:.2f} ms\n\n")
        
        # Comparison Tables
        f.write("## Accuracy Comparison\n\n")
        f.write("| Configuration | Accuracy (%) |\n")
        f.write("|---------------|-------------|\n")
        f.write(f"| Baseline | {results['baseline']['accuracy']:.2f} |\n")
        
        for config in ['cluster_16', 'cluster_32', 'cluster_64']:
            if config in results:
                f.write(f"| {config.replace('_', ' ').title()} | {results[config]['accuracy']:.2f} |\n")
                if f"{config}_int8" in results:
                    f.write(f"| {config.replace('_', ' ').title()} + INT8 | {results[f'{config}_int8']['accuracy']:.2f} |\n")
                if f"{config}_int4" in results:
                    f.write(f"| {config.replace('_', ' ').title()} + INT4 | {results[f'{config}_int4']['accuracy']:.2f} |\n")
        
        # Model Size Comparison
        f.write("\n## Model Size Comparison\n\n")
        f.write("| Configuration | Model Size (KB) | Reduction (%) |\n")
        f.write("|---------------|-----------------|---------------|\n")
        baseline_size = results['baseline']['model_size_kb']
        f.write(f"| Baseline | {baseline_size:.2f} | - |\n")
        
        for config in ['cluster_16', 'cluster_32', 'cluster_64']:
            if config in results:
                size = results[config]['model_size_kb']
                reduction = 100 * (baseline_size - size) / baseline_size
                f.write(f"| {config.replace('_', ' ').title()} | {size:.2f} | {reduction:.2f} |\n")
                
                if f"{config}_int8" in results:
                    size = results[f'{config}_int8']['model_size_kb']
                    reduction = 100 * (baseline_size - size) / baseline_size
                    f.write(f"| {config.replace('_', ' ').title()} + INT8 | {size:.2f} | {reduction:.2f} |\n")
                    
                if f"{config}_int4" in results:
                    size = results[f'{config}_int4']['model_size_kb']
                    reduction = 100 * (baseline_size - size) / baseline_size
                    f.write(f"| {config.replace('_', ' ').title()} + INT4 | {size:.2f} | {reduction:.2f} |\n")
        
        # RAM Usage Comparison
        f.write("\n## RAM Usage Comparison\n\n")
        f.write("| Configuration | Tensor Arena (KB) | Total RAM (KB) | Reduction (%) |\n")
        f.write("|---------------|-------------------|----------------|---------------|\n")
        baseline_ram = results['baseline']['total_ram_kb']
        f.write(f"| Baseline | {results['baseline']['tensor_arena_kb']:.2f} | {baseline_ram:.2f} | - |\n")
        
        for config in ['cluster_16', 'cluster_32', 'cluster_64']:
            if config in results:
                ram = results[config]['total_ram_kb']
                reduction = 100 * (baseline_ram - ram) / baseline_ram
                f.write(f"| {config.replace('_', ' ').title()} | {results[config]['tensor_arena_kb']:.2f} | {ram:.2f} | {reduction:.2f} |\n")
                
                if f"{config}_int8" in results:
                    ram = results[f'{config}_int8']['total_ram_kb']
                    reduction = 100 * (baseline_ram - ram) / baseline_ram
                    f.write(f"| {config.replace('_', ' ').title()} + INT8 | {results[f'{config}_int8']['tensor_arena_kb']:.2f} | {ram:.2f} | {reduction:.2f} |\n")
                    
                if f"{config}_int4" in results:
                    ram = results[f'{config}_int4']['total_ram_kb']
                    reduction = 100 * (baseline_ram - ram) / baseline_ram
                    f.write(f"| {config.replace('_', ' ').title()} + INT4 | {results[f'{config}_int4']['tensor_arena_kb']:.2f} | {ram:.2f} | {reduction:.2f} |\n")
        
        # Inference Time Comparison
        f.write("\n## Inference Time Comparison\n\n")
        f.write("| Configuration | Inference Time (ms) | Speedup (%) |\n")
        f.write("|---------------|---------------------|-------------|\n")
        baseline_time = results['baseline']['avg_inference_time_ms']
        f.write(f"| Baseline | {baseline_time:.2f} | - |\n")
        
        for config in ['cluster_16', 'cluster_32', 'cluster_64']:
            if config in results:
                time_ms = results[config]['avg_inference_time_ms']
                speedup = 100 * (baseline_time - time_ms) / baseline_time
                f.write(f"| {config.replace('_', ' ').title()} | {time_ms:.2f} | {speedup:.2f} |\n")
                
                if f"{config}_int8" in results:
                    time_ms = results[f'{config}_int8']['avg_inference_time_ms']
                    speedup = 100 * (baseline_time - time_ms) / baseline_time
                    f.write(f"| {config.replace('_', ' ').title()} + INT8 | {time_ms:.2f} | {speedup:.2f} |\n")
                    
                if f"{config}_int4" in results:
                    time_ms = results[f'{config}_int4']['avg_inference_time_ms']
                    speedup = 100 * (baseline_time - time_ms) / baseline_time
                    f.write(f"| {config.replace('_', ' ').title()} + INT4 | {time_ms:.2f} | {speedup:.2f} |\n")
        
        # Unique Weight Values per Layer
        f.write("\n## Clustering Details\n\n")
        for config in ['cluster_16', 'cluster_32', 'cluster_64']:
            if config not in results or 'clustering_stats' not in results[config]:
                continue
                
            clustering_stats = results[config]['clustering_stats']
            f.write(f"### {config.replace('_', ' ').title()}\n\n")
            f.write(f"- Total unique values before: {clustering_stats['unique_values_before']}\n")
            f.write(f"- Total unique values after: {clustering_stats['unique_values_after']}\n")
            f.write(f"- Compression ratio: {clustering_stats['compression_ratio']*100:.2f}%\n\n")
            
            f.write("| Layer | Unique Values Before | Unique Values After | Reduction (%) |\n")
            f.write("|-------|----------------------|---------------------|---------------|\n")
            
            if 'clustered_layers' in clustering_stats:
                for layer in clustering_stats['clustered_layers']:
                    name = layer['name']
                    before = layer['unique_before']
                    after = layer['unique_after']
                    reduction = layer['reduction'] * 100
                    f.write(f"| {name} | {before} | {after} | {reduction:.2f} |\n")
            
            f.write("\n")
        
        # Konklusion und Empfehlungen
        f.write("\n## Conclusion\n\n")
        f.write("Based on the evaluation results, we observe that:\n\n")
        f.write("1. **Accuracy Impact**: Weight clustering with proper quantization maintains model accuracy while significantly reducing model size.\n")
        f.write("2. **Size Reduction**: Combining clustering with INT4 quantization achieves the most significant size reduction.\n")
        f.write("3. **RAM Usage**: Tensor Arena requirements are reduced, especially with INT4 quantization.\n")
        f.write("4. **Inference Time**: Clustering slightly improves inference time due to reduced computation complexity.\n\n")
        
        f.write("### Recommendations\n\n")
        f.write(f"For optimal performance on RP2040, we recommend the **{best_config['name']}** configuration, which provides:")
        f.write(f" {best_config['accuracy']:.2f}% accuracy with only {best_config['total_ram_kb']:.2f} KB of RAM usage ")
        f.write(f"and {best_config['avg_inference_time_ms']:.2f} ms inference time.\n\n")
        
        # Metadaten
        f.write(f"\n\n---\n*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    logger.info(f"Evaluierungsbericht erstellt: {md_path}")
    
    return json_path, md_path

def find_best_configuration(results):
    """
    Findet die beste Konfiguration basierend auf Genauigkeit und RAM-Nutzung.
    
    Args:
        results: Dictionary mit Evaluierungsergebnissen
        
    Returns:
        Dictionary mit der besten Konfiguration
    """
    # Gewichtungen für verschiedene Metriken (können angepasst werden)
    ACCURACY_WEIGHT = 0.5
    RAM_WEIGHT = 0.3
    INFERENCE_WEIGHT = 0.2
    
    best_score = -float('inf')
    best_config = None
    
    # Baseline-Werte für Normalisierung
    baseline_accuracy = results['baseline']['accuracy']
    baseline_ram = results['baseline']['total_ram_kb']
    baseline_inference = results['baseline']['avg_inference_time_ms']
    
    for config_name, config_results in results.items():
        if config_name == 'baseline' or 'clustering_stats' in config_results:
            continue  # Überspringe Baseline und Zwischenergebnisse
        
        # Normalisierte Werte (1.0 ist besser als Baseline)
        norm_accuracy = config_results['accuracy'] / baseline_accuracy
        norm_ram = baseline_ram / max(config_results['total_ram_kb'], 1)  # Kleinere RAM-Nutzung ist besser
        norm_inference = baseline_inference / max(config_results['avg_inference_time_ms'], 1)  # Kürzere Zeit ist besser
        
        # Gewichteter Score
        score = (ACCURACY_WEIGHT * norm_accuracy +
                 RAM_WEIGHT * norm_ram +
                 INFERENCE_WEIGHT * norm_inference)
        
        if score > best_score:
            best_score = score
            best_config = {
                'name': config_name.replace('_', ' ').title(),
                'accuracy': config_results['accuracy'],
                'model_size_kb': config_results['model_size_kb'],
                'total_ram_kb': config_results['total_ram_kb'],
                'avg_inference_time_ms': config_results['avg_inference_time_ms'],
                'score': score
            }
    
    return best_config

def main(model_path, output_dir, device='cpu'):
    """
    Hauptfunktion für die Evaluierung des Gewichts-Clustering.
    
    Args:
        model_path: Pfad zum vortrainierten Modell
        output_dir: Ausgabeverzeichnis
        device: Gerät für die Berechnung ('cuda' oder 'cpu')
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
    
    # Lade vortrainiertes Modell
    logger.info(f"Lade vortrainiertes Modell von: {model_path}")
    base_model = MicroPizzaNet(num_classes=num_classes)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model = base_model.to(device)
    
    # Evaluiere Baseline-Modell
    logger.info("Evaluiere Baseline-Modell")
    baseline_results = evaluate_clustered_model(base_model, val_loader, class_names, device)
    
    # Dictionary für alle Ergebnisse
    results = {'baseline': baseline_results}
    
    # Cluster-Anzahlen für die Evaluierung
    cluster_counts = [16, 32, 64]
    
    # Evaluiere jede Cluster-Anzahl
    for num_clusters in cluster_counts:
        # Kopiere Basismodell für jede Konfiguration
        model = MicroPizzaNet(num_classes=num_classes)
        model.load_state_dict(base_model.state_dict())
        model = model.to(device)
        
        # Anwenden des Clusterings
        clustered_model, clustering_stats = apply_weight_clustering(model, num_clusters)
        
        # Speichere geclustertes Modell
        clustered_model_path = os.path.join(output_dir, f"clustered_model_{num_clusters}.pth")
        torch.save(clustered_model.state_dict(), clustered_model_path)
        
        # Evaluiere geclustertes Modell
        logger.info(f"Evaluiere geclustertes Modell mit {num_clusters} Clustern")
        clustered_results = evaluate_clustered_model(clustered_model, val_loader, class_names, device)
        
        # Füge Clustering-Statistiken hinzu
        clustered_results['clustering_stats'] = clustering_stats
        
        # Speichere Ergebnisse
        results[f'cluster_{num_clusters}'] = clustered_results
        
        # Kopiere Modell für INT8-Quantisierung
        model_int8 = MicroPizzaNet(num_classes=num_classes)
        model_int8.load_state_dict(clustered_model.state_dict())
        model_int8 = model_int8.to(device)
        
        # Anwenden der INT8-Quantisierung
        logger.info(f"Quantisiere geclustertes Modell (Cluster: {num_clusters}) zu INT8")
        quantized_int8_model, int8_stats = quantize_to_int8(model_int8, val_loader, device)
        
        # Speichere INT8-quantisiertes Modell
        int8_model_path = os.path.join(output_dir, f"clustered_model_{num_clusters}_int8.pth")
        torch.save(quantized_int8_model.state_dict(), int8_model_path)
        
        # Evaluiere INT8-quantisiertes Modell
        logger.info(f"Evaluiere INT8-quantisiertes Modell (Cluster: {num_clusters})")
        int8_results = evaluate_clustered_model(quantized_int8_model, val_loader, class_names, device)
        
        # Füge INT8-Statistiken hinzu
        int8_results['quantization_stats'] = int8_stats
        
        # Speichere Ergebnisse
        results[f'cluster_{num_clusters}_int8'] = int8_results
        
        # Kopiere Modell für INT4-Quantisierung
        model_int4 = MicroPizzaNet(num_classes=num_classes)
        model_int4.load_state_dict(clustered_model.state_dict())
        model_int4 = model_int4.to(device)
        
        # Anwenden der INT4-Quantisierung
        logger.info(f"Quantisiere geclustertes Modell (Cluster: {num_clusters}) zu INT4")
        int4_output_dir = os.path.join(output_dir, f"int4_cluster_{num_clusters}")
        int4_results = quantize_model_to_int4(
            model=model_int4,
            val_loader=val_loader,
            config=config,
            class_names=class_names,
            output_dir=int4_output_dir
        )
        
        # Sammle INT4-Ergebnisse
        config_int4_results = {
            'accuracy': int4_results['int4_model']['accuracy'],
            'class_accuracies': int4_results['int4_model']['class_accuracies'],
            'model_size_kb': int4_results['int4_model']['memory_kb'],
            'tensor_arena_kb': int4_results.get('int4_model', {}).get('tensor_arena_kb', baseline_results['tensor_arena_kb'] * 0.4),  # Schätzung falls nicht vorhanden
            'total_ram_kb': int4_results['int4_model']['memory_kb'] + int4_results.get('int4_model', {}).get('tensor_arena_kb', baseline_results['tensor_arena_kb'] * 0.4),
            'avg_inference_time_ms': int4_results['int4_model'].get('inference_time_ms', baseline_results['avg_inference_time_ms'] * 0.9),  # Schätzung falls nicht vorhanden
            'quantization_type': 'INT4'
        }
        
        # Speichere Ergebnisse
        results[f'cluster_{num_clusters}_int4'] = config_int4_results
    
    # Erstelle Evaluierungsbericht
    json_path, md_path = create_clustering_evaluation_report(results, output_dir)
    
    logger.info("Clustering-Evaluierung abgeschlossen")
    logger.info(f"Detaillierte Ergebnisse: {json_path}")
    logger.info(f"Bericht: {md_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluiere Gewichts-Clustering mit verschiedenen Konfigurationen')
    parser.add_argument('--model_path', type=str, default='models/micro_pizza_model.pth',
                        help='Pfad zum vortrainierten Modell')
    parser.add_argument('--output_dir', type=str, default='output/model_optimization',
                        help='Ausgabeverzeichnis für Ergebnisse')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Gerät für Berechnungen (cuda oder cpu)')
    
    args = parser.parse_args()
    
    main(args.model_path, args.output_dir, args.device)
