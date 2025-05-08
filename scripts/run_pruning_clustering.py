#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pizza Model Optimization: Pruning & Clustering
----------------------------------------------
Dieses Skript führt eine komplette Optimierung des Pizza-Erkennungsmodells durch,
indem es Gewichts-Pruning und Clustering kombiniert, um die Modellgröße zu reduzieren
und die Inferenzgeschwindigkeit zu verbessern, während die Genauigkeit erhalten bleibt.

Verwendung:
    python run_pruning_clustering.py --model_path models/micro_pizza_model.pth --output_dir models/pruned

Die Optimierung umfasst:
1. Unstrukturiertes Pruning: Entfernt unwichtige Gewichte basierend auf ihrer Magnitude
2. Strukturelles Pruning: Entfernt ganze Kanäle/Filter basierend auf ihrer Wichtigkeit
3. Gewichts-Clustering: Fasst ähnliche Gewichtswerte zusammen
4. Fine-Tuning: Trainiert das optimierte Modell nach, um die Genauigkeit wiederherzustellen
"""

import os
import sys
import logging
import argparse
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Modulpfad für Import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)

# Eigenes Modul importieren
from scripts.model_optimization.weight_pruning import (
    PruningManager, WeightClusterer, evaluate_model, 
    fine_tune_pruned_model, create_comparison_report
)
from src.pizza_detector import (
    MicroPizzaNet, create_optimized_dataloaders, MemoryEstimator, 
    train_microcontroller_model, EarlyStopping, export_to_microcontroller
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

def optimize_model(model_path, output_dir, prune_ratio=0.3, structured_ratio=0.2, 
                   num_clusters=32, fine_tune_epochs=10, device='cuda', 
                   export_microcontroller=True):
    """
    Führt die vollständige Optimierungspipeline für das Modell durch.
    
    Args:
        model_path: Pfad zum vortrainierten Modell
        output_dir: Ausgabeverzeichnis
        prune_ratio: Verhältnis für unstrukturiertes Pruning (0.0-1.0)
        structured_ratio: Verhältnis für strukturelles Pruning (0.0-1.0)
        num_clusters: Anzahl der Gewichts-Cluster
        fine_tune_epochs: Anzahl der Epochen für das Fine-Tuning
        device: Gerät für die Berechnung ('cuda' oder 'cpu')
        export_microcontroller: Ob das Modell für RP2040 exportiert werden soll
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
    
    # Lade Daten für Evaluation und Fine-Tuning
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    
    # Anzahl der Klassen
    num_classes = len(class_names)
    logger.info(f"Anzahl Klassen: {num_classes}")
    
    # Lade vortrainiertes Modell
    logger.info(f"Lade vortrainiertes Modell von: {model_path}")
    model = MicroPizzaNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Evaluiere Basis-Modell
    logger.info("Evaluiere Basis-Modell vor Optimierung")
    base_model_stats = evaluate_model(model, val_loader, class_names, device)
    base_model_stats['class_names'] = class_names
    
    # Trainierte Modellkopie für später speichern
    base_model = MicroPizzaNet(num_classes=num_classes)
    base_model.load_state_dict(model.state_dict())
    
    # 1. Unstructured Pruning (magnitude-based)
    logger.info(f"Starte Pruning mit Prune-Ratio={prune_ratio}, Structured-Ratio={structured_ratio}")
    pruning_manager = PruningManager(
        model=model,
        prune_ratio=prune_ratio,
        structured_ratio=structured_ratio
    )
    
    pruning_manager.unstructured_pruning()
    
    # 2. Structured Pruning (channel/filter)
    pruning_manager.structured_pruning()
    
    # Aktualisiere Statistiken
    pruning_manager.recompute_statistics()
    pruning_stats = pruning_manager.get_pruning_stats()
    
    # 3. Fine-Tuning mit reduzierten Lernraten
    logger.info(f"Starte Fine-Tuning für {fine_tune_epochs} Epochen")
    config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('__')}
    config_dict['DEVICE'] = device
    config_dict['LEARNING_RATE'] = config.LEARNING_RATE * 0.1
    config_dict['EPOCHS'] = fine_tune_epochs
    config_dict['EARLY_STOPPING_PATIENCE'] = 5
    
    history, model = fine_tune_pruned_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=types.SimpleNamespace(**config_dict),
        class_names=class_names,
        num_epochs=fine_tune_epochs
    )
    
    # Stelle sicher, dass Nullen nach dem Training noch Null sind
    pruning_manager.apply_masks()
    
    # 4. Gewichts-Clustering
    logger.info(f"Starte Weight-Clustering mit {num_clusters} Clustern")
    clusterer = WeightClusterer(
        model=model,
        num_clusters=num_clusters
    )
    
    clusterer.cluster_weights()
    clustering_stats = clusterer.get_clustering_stats()
    
    # Speichere optimiertes Modell
    pruned_model_path = os.path.join(output_dir, "pruned_pizza_model.pth")
    torch.save(model.state_dict(), pruned_model_path)
    logger.info(f"Optimiertes Modell gespeichert unter: {pruned_model_path}")
    
    # Evaluiere optimiertes Modell
    logger.info("Evaluiere optimiertes Modell")
    pruned_model_stats = evaluate_model(model, val_loader, class_names, device)
    pruned_model_stats['class_names'] = class_names
    
    # Erstelle Vergleichsbericht
    create_comparison_report(
        base_model_stats=base_model_stats,
        pruned_model_stats=pruned_model_stats,
        pruning_stats=pruning_stats,
        clustering_stats=clustering_stats,
        output_dir=output_dir
    )
    
    # Optional: Für Mikrocontroller exportieren
    if export_microcontroller:
        logger.info("Exportiere optimiertes Modell für RP2040-Mikrocontroller")
        
        # Konfigurationsobjekt für den Export erweitern
        config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('__')}
        
        export_info = export_to_microcontroller(
            model=model,
            config=types.SimpleNamespace(**config_dict),
            class_names=class_names,
            preprocess_params=preprocessing_params,
            output_dir=os.path.join(output_dir, "rp2040_export")
        )
        
        logger.info(f"Modell exportiert: {export_info['export_dir']}")
        logger.info(f"Modellgröße: {export_info['model_size_kb']:.2f} KB")
        
    return {
        'base_model_stats': base_model_stats,
        'pruned_model_stats': pruned_model_stats,
        'pruning_stats': pruning_stats,
        'clustering_stats': clustering_stats,
        'model_path': pruned_model_path
    }

def main():
    parser = argparse.ArgumentParser(description='Pizza Model Optimierung mit Pruning und Clustering')
    parser.add_argument('--model_path', type=str, default='models/micro_pizza_model.pth', 
                       help='Pfad zum vortrainierten Modell')
    parser.add_argument('--output_dir', type=str, default='models/pruned_model', 
                       help='Ausgabeverzeichnis')
    parser.add_argument('--prune_ratio', type=float, default=0.3, 
                       help='Verhältnis für unstrukturiertes Pruning (0.0-1.0)')
    parser.add_argument('--structured_ratio', type=float, default=0.2, 
                       help='Verhältnis für strukturelles Pruning (0.0-1.0)')
    parser.add_argument('--num_clusters', type=int, default=32, 
                       help='Anzahl der Gewichts-Cluster')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, 
                       help='Anzahl der Fine-Tuning-Epochen')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Gerät für die Berechnung (cuda oder cpu)')
    parser.add_argument('--no_export', action='store_true', 
                       help='Nicht für Mikrocontroller exportieren')
    
    args = parser.parse_args()
    
    # Führe Optimierung durch
    optimize_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        prune_ratio=args.prune_ratio,
        structured_ratio=args.structured_ratio,
        num_clusters=args.num_clusters,
        fine_tune_epochs=args.fine_tune_epochs,
        device=args.device,
        export_microcontroller=not args.no_export
    )
    
    logger.info("Optimierung abgeschlossen!")
    
if __name__ == "__main__":
    main()