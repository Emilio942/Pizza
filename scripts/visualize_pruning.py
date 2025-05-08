#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pizza Model Visualization: Pruning & Clustering
-----------------------------------------------
Dieses Skript erstellt visualisierungen für die Auswirkungen von
Pruning und Clustering auf die Modellgewichte des Pizza-Erkennungsmodells.

Verwendung:
    python visualize_pruning.py --model_path models/micro_pizza_model.pth --pruned_model_path models/pruned_model/pruned_pizza_model.pth

Die Visualisierung zeigt:
1. Gewichtsverteilung vor und nach Pruning/Clustering
2. Heatmaps der Gewichtsmatrizen
3. Auswirkungen auf Parameterreduktion und Modellgröße
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from pathlib import Path

# Modulpfad für Import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)

from src.pizza_detector import MicroPizzaNet

def plot_weight_distribution(original_weights, pruned_weights, layer_name, output_dir):
    """Erstellt Histogramm-Vergleich der Gewichtsverteilungen"""
    plt.figure(figsize=(12, 6))
    
    # Original Gewichtsverteilung
    plt.subplot(1, 2, 1)
    plt.hist(original_weights.flatten(), bins=50, alpha=0.7)
    plt.title(f"Original Gewichte: {layer_name}")
    plt.xlabel("Gewichtswert")
    plt.ylabel("Häufigkeit")
    
    # Pruned/Clustered Gewichtsverteilung
    plt.subplot(1, 2, 2)
    plt.hist(pruned_weights.flatten(), bins=50, alpha=0.7)
    plt.title(f"Optimierte Gewichte: {layer_name}")
    plt.xlabel("Gewichtswert")
    plt.ylabel("Häufigkeit")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"distribution_{layer_name.replace('.', '_')}.png"))
    plt.close()

def plot_weight_heatmap(original_weights, pruned_weights, layer_name, output_dir):
    """Erstellt Heatmap der Gewichtsmatrizen"""
    # Für convolutional Layer nehmen wir den ersten Filter für die Visualisierung
    if len(original_weights.shape) == 4:
        original_to_plot = original_weights[0, 0]
        pruned_to_plot = pruned_weights[0, 0]
    elif len(original_weights.shape) == 2:
        # Für vollvernetzte Layer nehmen wir eine Teilmenge der Gewichte
        max_vis_size = 20  # Maximale Visualisierungsgröße
        original_to_plot = original_weights[:min(max_vis_size, original_weights.shape[0]), 
                                           :min(max_vis_size, original_weights.shape[1])]
        pruned_to_plot = pruned_weights[:min(max_vis_size, pruned_weights.shape[0]), 
                                       :min(max_vis_size, pruned_weights.shape[1])]
    else:
        # Für andere Layer-Typen überspringen
        return
    
    plt.figure(figsize=(14, 6))
    
    # Original Gewichts-Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(original_to_plot.cpu().numpy(), cmap="viridis")
    plt.title(f"Original Gewichte: {layer_name}")
    
    # Pruned/Clustered Gewichts-Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(pruned_to_plot.cpu().numpy(), cmap="viridis")
    plt.title(f"Optimierte Gewichte: {layer_name}")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"heatmap_{layer_name.replace('.', '_')}.png"))
    plt.close()

def plot_unique_values(original_model, pruned_model, output_dir):
    """Visualisiert die Reduktion von eindeutigen Gewichtswerten (Clustering-Effekt)"""
    layers = []
    original_unique = []
    pruned_unique = []
    
    # Sammle die Anzahl eindeutiger Werte pro Layer
    for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), pruned_model.named_parameters()):
        if 'weight' in name1 and param1.dim() > 1:
            layers.append(name1.split('.')[-2])  # Layer-Namen ohne "weight"
            original_unique.append(len(torch.unique(param1)))
            pruned_unique.append(len(torch.unique(param2)))
    
    # Erstelle Balkendiagramm
    plt.figure(figsize=(12, 6))
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, original_unique, width, label='Original')
    plt.bar(x + width/2, pruned_unique, width, label='Optimiert')
    
    plt.ylabel('Anzahl eindeutiger Werte')
    plt.title('Reduktion eindeutiger Gewichtswerte durch Clustering')
    plt.xticks(x, layers, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "unique_values_reduction.png"))
    plt.close()

def plot_sparsity(original_model, pruned_model, output_dir):
    """Visualisiert die Modell-Sparsity (Anteil der Nullen) pro Layer"""
    layers = []
    original_sparsity = []
    pruned_sparsity = []
    
    # Berechne Sparsity pro Layer
    for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), pruned_model.named_parameters()):
        if 'weight' in name1 and param1.dim() > 1:
            layers.append(name1.split('.')[-2])  # Layer-Namen ohne "weight"
            
            # Sparsity = Anteil der Nullen
            original_zeros = torch.sum(param1 == 0).item()
            original_total = param1.numel()
            original_sparsity.append(100 * original_zeros / original_total)
            
            pruned_zeros = torch.sum(param2 == 0).item()
            pruned_total = param2.numel()
            pruned_sparsity.append(100 * pruned_zeros / pruned_total)
    
    # Erstelle Balkendiagramm
    plt.figure(figsize=(12, 6))
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, original_sparsity, width, label='Original')
    plt.bar(x + width/2, pruned_sparsity, width, label='Optimiert')
    
    plt.ylabel('Sparsity (% Nullen)')
    plt.title('Erhöhung der Sparsity durch Pruning')
    plt.xticks(x, layers, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sparsity_increase.png"))
    plt.close()

def create_model_summary(original_model, pruned_model, output_dir):
    """Erstellt eine Zusammenfassung der Modellparameter und Reduktion"""
    # Zähle Parameter und berechne Speichernutzung
    orig_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    
    # Zähle Nicht-Null-Parameter im optimierten Modell
    pruned_nonzero = sum((p != 0).sum().item() for p in pruned_model.parameters() if p.requires_grad)
    
    # Berechne eindeutige Werte
    orig_unique = sum(len(torch.unique(p)) for p in original_model.parameters() if p.requires_grad and p.dim() > 1)
    pruned_unique = sum(len(torch.unique(p)) for p in pruned_model.parameters() if p.requires_grad and p.dim() > 1)
    
    # Gewichtsreduktion durch Pruning
    param_reduction = 100 * (1 - pruned_nonzero / orig_params)
    
    # Speicherreduktion durch Clustering
    storage_reduction = 100 * (1 - pruned_unique / orig_unique)
    
    # Erstelle Zusammenfassungs-Diagramm
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parameter-Reduktion
    labels = ['Original', 'Optimiert\n(Nicht-Null)']
    values = [orig_params, pruned_nonzero]
    ax[0].bar(labels, values)
    ax[0].set_title(f'Parameterreduktion: {param_reduction:.1f}%')
    ax[0].set_ylabel('Anzahl Parameter')
    
    for i, v in enumerate(values):
        ax[0].text(i, v + 0.1, str(v), ha='center')
    
    # Eindeutige Werte Reduktion
    labels = ['Original', 'Optimiert']
    values = [orig_unique, pruned_unique]
    ax[1].bar(labels, values)
    ax[1].set_title(f'Speicherreduktion durch Clustering: {storage_reduction:.1f}%')
    ax[1].set_ylabel('Anzahl eindeutiger Werte')
    
    for i, v in enumerate(values):
        ax[1].text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "model_summary.png"))
    plt.close()
    
    # Erstelle zusätzlich eine Textdatei mit den Details
    with open(os.path.join(output_dir, "model_comparison.txt"), 'w') as f:
        f.write("Modellvergleich: Original vs. Optimiert\n")
        f.write("==========================================\n\n")
        f.write(f"Original Modell Parameter: {orig_params}\n")
        f.write(f"Optimiertes Modell Parameter: {pruned_params}\n")
        f.write(f"Nicht-Null Parameter nach Pruning: {pruned_nonzero}\n")
        f.write(f"Parameterreduktion durch Pruning: {param_reduction:.2f}%\n\n")
        f.write(f"Original eindeutige Gewichtswerte: {orig_unique}\n")
        f.write(f"Optimierte eindeutige Gewichtswerte: {pruned_unique}\n")
        f.write(f"Speicherreduktion durch Clustering: {storage_reduction:.2f}%\n")

def visualize_models(original_model_path, pruned_model_path, output_dir='visualizations/pruning'):
    """Hauptfunktion zur Visualisierung der Modelloptimierung"""
    # Lade die Modelle
    device = torch.device('cpu')  # Für Visualisierung verwenden wir CPU
    
    # Anzahl der Klassen bestimmen
    temp_model = torch.load(original_model_path, map_location=device)
    if isinstance(temp_model, dict):  # Wenn es ein state_dict ist
        # Versuche die Anzahl der Klassen aus dem letzten Layer zu bestimmen
        for k, v in temp_model.items():
            if 'classifier' in k and 'weight' in k and len(v.shape) == 2:
                num_classes = v.shape[0]
                break
    else:
        num_classes = 6  # Standardwert, falls nicht bestimmbar
    
    # Modelle laden
    original_model = MicroPizzaNet(num_classes=num_classes)
    pruned_model = MicroPizzaNet(num_classes=num_classes)
    
    # Lade state_dict
    original_state = torch.load(original_model_path, map_location=device)
    pruned_state = torch.load(pruned_model_path, map_location=device)
    
    # Prüfe, ob es sich um vollständige Modelle oder state_dicts handelt
    if isinstance(original_state, dict) and 'state_dict' in original_state:
        original_model.load_state_dict(original_state['state_dict'])
    else:
        original_model.load_state_dict(original_state)
        
    if isinstance(pruned_state, dict) and 'state_dict' in pruned_state:
        pruned_model.load_state_dict(pruned_state['state_dict'])
    else:
        pruned_model.load_state_dict(pruned_state)
    
    # Erstelle Ausgabeverzeichnis
    os.makedirs(output_dir, exist_ok=True)
    
    # Erstelle Visualisierungen für jede Gewichtsschicht
    for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), pruned_model.named_parameters()):
        if 'weight' in name1 and param1.dim() > 1:  # Nur Gewichtsmatrizen/Tensoren
            print(f"Visualisiere Layer: {name1}")
            
            # Gewichtsverteilung
            plot_weight_distribution(param1.data, param2.data, name1, output_dir)
            
            # Gewichts-Heatmap
            plot_weight_heatmap(param1.data, param2.data, name1, output_dir)
    
    # Weitere Modellvergleichs-Visualisierungen
    plot_unique_values(original_model, pruned_model, output_dir)
    plot_sparsity(original_model, pruned_model, output_dir)
    create_model_summary(original_model, pruned_model, output_dir)
    
    print(f"Visualisierungen erstellt im Verzeichnis: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualisierung von Pruning und Clustering Effekten')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Pfad zum Original-Modell')
    parser.add_argument('--pruned_model_path', type=str, required=True, 
                        help='Pfad zum optimierten Modell')
    parser.add_argument('--output_dir', type=str, default='visualizations/pruning', 
                        help='Ausgabeverzeichnis für Visualisierungen')
    
    args = parser.parse_args()
    
    visualize_models(args.model_path, args.pruned_model_path, args.output_dir)

if __name__ == "__main__":
    main()