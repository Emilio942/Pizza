#!/usr/bin/env python3
"""
Architektur-Hyperparameter-Suche für das Pizza-Erkennungssystem

Dieses Skript führt eine systematische Suche über verschiedene Modellarchitekturen durch,
um die optimale Balance zwischen Genauigkeit, Modellgröße und Inferenzgeschwindigkeit
auf dem RP2040-Mikrocontroller zu finden.

Verwendung:
    python hyperparameter_search.py [--data-dir DIR] [--output-dir DIR] [--epochs N]

Optionen:
    --data-dir: Verzeichnis mit dem Bilddatensatz (Standard: data/augmented)
    --output-dir: Verzeichnis für die Ergebnisse (Standard: output/hyperparameter_search)
    --epochs: Anzahl der Trainingsepochen pro Modell (Standard: 25)
    --batch-size: Batch-Größe für das Training (Standard: 32)
    --search-size: Anzahl der zu testenden Konfigurationen (small/medium/full, Standard: medium)
    --early-stopping: Frühes Beenden bei Stagnation (Standard: True)
    --visualize: Visualisierungen erstellen (Standard: True)
"""

import os
import sys
import argparse
import time
import json
import csv
import itertools
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Importiere Module aus dem Pizza-Projekt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pizza_detector import RP2040Config, PizzaDatasetAnalysis, create_optimized_dataloaders, MemoryEstimator
from src.metrics import ModelMetrics

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModularMicroPizzaNet(nn.Module):
    """
    Modulares CNN für die Hyperparameter-Suche mit flexiblen Architekturparametern
    """
    def __init__(self, 
                 num_classes=6, 
                 input_channels=3,
                 img_size=48,
                 depth_multiplier=1.0,     # Skalierungsfaktor für Kanalbreiten
                 num_blocks=2,             # Anzahl der Faltungsblöcke
                 use_separable=True,       # Verwende separable Faltungen
                 use_gpool=True,           # Verwende Global Pooling
                 dropout_rate=0.2,         # Dropout-Rate
                 initial_channels=8,       # Anzahl der Kanäle im ersten Block
                 use_batchnorm=True,       # Verwende BatchNorm
                 kernel_size=3,            # Kerngröße der Faltung
                 ):
        super(ModularMicroPizzaNet, self).__init__()
        
        self.hyperparams = {
            'num_classes': num_classes,
            'input_channels': input_channels,
            'img_size': img_size,
            'depth_multiplier': depth_multiplier,
            'num_blocks': num_blocks,
            'use_separable': use_separable,
            'use_gpool': use_gpool,
            'dropout_rate': dropout_rate,
            'initial_channels': initial_channels,
            'use_batchnorm': use_batchnorm,
            'kernel_size': kernel_size,
        }
        
        # Skaliere die Kanalanzahl mit dem Depth-Multiplier
        initial_ch = int(initial_channels * depth_multiplier)
        
        # Erstelle Sequenz von Blöcken
        self.blocks = nn.ModuleList()
        
        # Erster Block ist immer ein Standard-Faltungsblock (kein separable conv)
        in_ch = input_channels
        out_ch = initial_ch
        
        # Erstelle erste Konvolution (immer Standard conv)
        block = []
        block.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                               stride=2, padding=kernel_size//2, bias=not use_batchnorm))
        if use_batchnorm:
            block.append(nn.BatchNorm2d(out_ch))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.blocks.append(nn.Sequential(*block))
        
        # Aktuelle Feature-Map-Größe berechnen
        fm_size = img_size // 4  # Nach ersten conv und pool halbiert sich die Größe zweimal
        
        # Folgenden Blöcke erstellen
        in_ch = out_ch
        for i in range(1, num_blocks):
            # Verdopple Kanäle in jedem Block (typisches CNN-Pattern)
            out_ch = in_ch * 2
            
            block = []
            
            # Bei Bedarf separable convolution verwenden
            if use_separable:
                # Depthwise Faltung
                block.append(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                      stride=1, padding=kernel_size//2, groups=in_ch, 
                                      bias=not use_batchnorm))
                if use_batchnorm:
                    block.append(nn.BatchNorm2d(in_ch))
                block.append(nn.ReLU(inplace=True))
                
                # Pointwise Faltung (1x1)
                block.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not use_batchnorm))
            else:
                # Standard-Faltung
                block.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                                      stride=1, padding=kernel_size//2, bias=not use_batchnorm))
            
            if use_batchnorm:
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            self.blocks.append(nn.Sequential(*block))
            
            # Feature-Map-Größe aktualisieren
            fm_size = fm_size // 2
            in_ch = out_ch
        
        # Global Pooling oder Flatten
        if use_gpool:
            self.pool = nn.AdaptiveAvgPool2d(1)
            pool_output_size = out_ch
        else:
            self.pool = nn.Flatten()
            pool_output_size = out_ch * fm_size * fm_size
        
        # Classifier
        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(dropout_rate))
        
        # Immer direkt vom Pool zur Ausgabeschicht (keine hidden layers im Classifier)
        classifier.append(nn.Linear(pool_output_size, num_classes))
        
        self.classifier = nn.Sequential(*classifier)
        
        # Initialisierung der Gewichte
        self._initialize_weights()
    
    def forward(self, x):
        # Durch alle Blöcke laufen
        for block in self.blocks:
            x = block(x)
        
        # Pooling
        x = self.pool(x)
        
        # Bei Verwendung von Global Pooling muss noch geflattened werden
        if isinstance(self.pool, nn.AdaptiveAvgPool2d):
            x = x.view(x.size(0), -1)
        
        # Klassifikation
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """Verbesserte Gewichtsinitialisierung"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EarlyStopping:
    """Early-Stopping-Implementierung mit Validation Loss Plateau-Erkennung"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.val_loss_history = []
    
    def __call__(self, val_loss, model):
        # Speichere alle Validierungsverluste
        self.val_loss_history.append(val_loss)
        
        score = -val_loss
        
        if self.best_score is None:
            # Erster Aufruf
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            # Verschlechterung oder Stagnation
            self.counter += 1
            
            # Prüfe auf Plateau oder Divergenz
            if len(self.val_loss_history) >= 5:
                # Berechne gleitenden Durchschnitt der letzten 3 Verluste
                recent_avg = sum(self.val_loss_history[-3:]) / 3
                # Wenn der Verlust konstant oder steigend ist
                if all(l >= self.val_loss_history[-4] for l in self.val_loss_history[-3:]):
                    self.counter = max(self.counter, self.patience - 2)  # Beschleunige Abbruch
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Verbesserung
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def restore_weights(self, model):
        """Stellt die besten Gewichte wieder her"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            return True
        return False


def train_model(model, train_loader, val_loader, config, class_names, 
                epochs=25, early_stopping=True, model_name="micro_pizza_model"):
    """Trainiert ein Modell und gibt Metriken zurück"""
    device = config.DEVICE
    model = model.to(device)
    
    # Verlustfunktion
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=5) if early_stopping else None
    
    # Trainingsstatistiken
    stats = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # --- Trainingsphase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Statistiken
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * train_correct / train_total
            })
        
        # Durchschnitt berechnen
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        
        # --- Validierungsphase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistiken
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Durchschnitt berechnen
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        
        # Statistiken aktualisieren
        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)
        stats['train_acc'].append(train_acc)
        stats['val_acc'].append(val_acc)
        
        # Beste Validierungsgenauigkeit speichern
        if val_acc > stats['best_val_acc']:
            stats['best_val_acc'] = val_acc
            stats['best_epoch'] = epoch
        
        # Ausgabe
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early Stopping prüfen
        if early_stopper:
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                logger.info(f"Early Stopping in Epoche {epoch+1}")
                break
    
    # Trainingszeit
    stats['training_time'] = time.time() - start_time
    
    # Beste Gewichte wiederherstellen, falls Early Stopping aktiv
    if early_stopper and early_stopper.restore_weights(model):
        logger.info("Beste Modellgewichte wiederhergestellt")
    
    return stats, model


def evaluate_model(model, val_loader, config, class_names):
    """Evaluiert ein Modell und gibt Metriken zurück"""
    device = config.DEVICE
    model.eval()
    
    # Speicher für Vorhersagen und Labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Sammeln
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metriken berechnen
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Konfusionsmatrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Ergebnis
    return {
        'accuracy': accuracy * 100,  # In Prozent
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def measure_inference_time(model, input_size, config, quantized=False, num_runs=50):
    """Misst die Inferenzzeit auf CPU (als Proxy für RP2040)"""
    device = torch.device('cpu')  # Verwende immer CPU für Timing
    model = model.to(device)
    model.eval()
    
    # Erstelle Dummy-Input
    dummy_input = torch.randn(1, *input_size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Messe Zeit
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    end_time = time.time()
    
    # Durchschnittliche Inferenzzeit pro Bild
    avg_time = (end_time - start_time) / num_runs
    
    # Geschätztes RP2040-Timing (CPU ist ~10x schneller als RP2040)
    rp2040_factor = 10  # Grobe Schätzung
    est_rp2040_time = avg_time * rp2040_factor
    
    return {
        'avg_inference_time_s': avg_time,
        'estimated_rp2040_time_s': est_rp2040_time,
        'fps_on_rp2040': 1.0 / est_rp2040_time if est_rp2040_time > 0 else 0,
        'num_runs': num_runs
    }


def generate_model_configs(args):
    """Generiert Modellkonfigurationen für die Grid-Suche basierend auf Suchgröße"""
    # Parameter-Grid je nach gewählter Suchgröße
    if args.search_size == 'small':
        # Kleine Suche (schnell)
        depth_multipliers = [0.75, 1.0, 1.25]
        num_blocks = [2, 3]
        initial_channels = [8, 16]
        use_separable = [True]
        use_gpool = [True]
        
    elif args.search_size == 'medium':
        # Mittlere Suche (Standard)
        depth_multipliers = [0.5, 0.75, 1.0, 1.25]
        num_blocks = [2, 3, 4]
        initial_channels = [8, 16]
        use_separable = [True, False]
        use_gpool = [True]
        
    else:  # 'full'
        # Große Suche (umfassend)
        depth_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        num_blocks = [2, 3, 4, 5]
        initial_channels = [8, 16, 24]
        use_separable = [True, False]
        use_gpool = [True, False]
    
    # Gemeinsame Parameter für alle Suchgrößen
    img_sizes = [48]  # RP2040 ist auf 48x48 limitiert
    dropout_rates = [0.2]
    use_batchnorm = [True]
    kernel_sizes = [3]
    
    # Konfigurationen generieren
    configs = []
    
    # Beschränke Anzahl der Kombinationen für effizientere Suche
    max_configs = 50 if args.search_size == 'full' else 30 if args.search_size == 'medium' else 12
    
    # Zufällige Auswahl, wenn es zu viele Kombinationen gibt
    all_combinations = list(itertools.product(
        depth_multipliers, num_blocks, initial_channels, 
        use_separable, use_gpool, img_sizes, dropout_rates,
        use_batchnorm, kernel_sizes
    ))
    
    if len(all_combinations) > max_configs:
        # Zufällige Auswahl ohne Zurücklegen
        selected_indices = np.random.choice(len(all_combinations), max_configs, replace=False)
        selected_combinations = [all_combinations[i] for i in selected_indices]
    else:
        selected_combinations = all_combinations
    
    # Configs erstellen
    for params in selected_combinations:
        depth_mult, n_blocks, init_ch, sep, gpool, img_size, dropout, batchnorm, k_size = params
        
        config = {
            'depth_multiplier': depth_mult,
            'num_blocks': n_blocks,
            'initial_channels': init_ch,
            'use_separable': sep,
            'use_gpool': gpool,
            'img_size': img_size,
            'dropout_rate': dropout,
            'use_batchnorm': batchnorm,
            'kernel_size': k_size
        }
        configs.append(config)
    
    logger.info(f"Generiert: {len(configs)} Modellkonfigurationen für die Suche")
    return configs


def create_model_name(model_config):
    """Erstellt einen beschreibenden Namen für ein Modell basierend auf seinen Konfigurationsparametern"""
    depth = model_config['depth_multiplier']
    blocks = model_config['num_blocks']
    init_ch = model_config['initial_channels']
    sep = 'sep' if model_config['use_separable'] else 'std'
    pool = 'gpool' if model_config['use_gpool'] else 'flatten'
    
    return f"micro_d{depth}_b{blocks}_ch{init_ch}_{sep}_{pool}"


def run_hyperparameter_search(args):
    """Führt die Hyperparameter-Suche aus"""
    logger.info("Starte Hyperparameter-Suche für Pizza Erkennung")
    
    # Erstelle die Ausgabeverzeichnisse
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Basiskonfiguration
    config = RP2040Config(data_dir=args.data_dir)
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    
    # Datensatz vorbereiten
    logger.info(f"Lade Datensatz aus {args.data_dir}")
    analyzer = PizzaDatasetAnalysis(config.DATA_DIR)
    preprocessing_params = analyzer.analyze(sample_size=50)
    
    # DataLoader erstellen
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(
        config, preprocessing_params
    )
    
    # Modellkonfigurationen generieren
    model_configs = generate_model_configs(args)
    
    # CSV für Ergebnisse vorbereiten
    csv_path = os.path.join(args.output_dir, 'hyperparameter_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'model_name', 'accuracy', 'f1_score', 'params_count', 'model_size_kb', 
            'int8_size_kb', 'inference_time_ms', 'estimated_rp2040_time_ms', 
            'ram_usage_kb', 'depth_multiplier', 'num_blocks', 'initial_channels', 
            'use_separable', 'use_gpool', 'training_time_s', 'best_epoch'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Ergebnis-Array für einfacheren DataFrame-Export
    results = []
    
    # Trainiere und evaluiere jedes Modell
    for i, model_config in enumerate(model_configs):
        model_name = create_model_name(model_config)
        logger.info(f"\n{'='*50}\nTrainiere Modell {i+1}/{len(model_configs)}: {model_name}\n{'='*50}")
        
        # Modell erstellen
        model = ModularMicroPizzaNet(
            num_classes=len(class_names),
            input_channels=3,
            **model_config
        )
        
        # Parameter und Speichernutzung analysieren
        params_count = model.count_parameters()
        memory_analysis = MemoryEstimator.check_memory_requirements(
            model, 
            (3, model_config['img_size'], model_config['img_size']), 
            config
        )
        
        logger.info(f"Modell hat {params_count:,} Parameter")
        logger.info(f"Geschätzte Modellgröße: {memory_analysis['model_size_float32_kb']:.2f} KB (float32), "
                   f"{memory_analysis['model_size_int8_kb']:.2f} KB (int8)")
        
        # Training
        training_stats, trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            config, 
            class_names,
            epochs=args.epochs,
            early_stopping=args.early_stopping,
            model_name=model_name
        )
        
        # Evaluierung
        eval_metrics = evaluate_model(trained_model, val_loader, config, class_names)
        
        # Inferenzzeit messen
        timing = measure_inference_time(
            trained_model, 
            (3, model_config['img_size'], model_config['img_size']),
            config
        )
        
        # Ergebnisse speichern
        result = {
            'model_name': model_name,
            'accuracy': eval_metrics['accuracy'],
            'f1_score': eval_metrics['f1'],
            'params_count': params_count,
            'model_size_kb': memory_analysis['model_size_float32_kb'],
            'int8_size_kb': memory_analysis['model_size_int8_kb'],
            'inference_time_ms': timing['avg_inference_time_s'] * 1000,
            'estimated_rp2040_time_ms': timing['estimated_rp2040_time_s'] * 1000,
            'ram_usage_kb': memory_analysis['total_runtime_memory_kb'],
            'depth_multiplier': model_config['depth_multiplier'],
            'num_blocks': model_config['num_blocks'],
            'initial_channels': model_config['initial_channels'],
            'use_separable': model_config['use_separable'],
            'use_gpool': model_config['use_gpool'],
            'training_time_s': training_stats['training_time'],
            'best_epoch': training_stats['best_epoch']
        }
        
        # In CSV schreiben
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
        
        # Zum Array hinzufügen
        results.append(result)
        
        # Modell speichern
        if args.save_models:
            model_path = os.path.join(args.output_dir, 'models', f"{model_name}.pth")
            torch.save(trained_model.state_dict(), model_path)
            logger.info(f"Modell gespeichert unter: {model_path}")
        
        # Visualisierungen für dieses Modell erstellen
        if args.visualize:
            # Trainingshistorie
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(training_stats['train_acc'], label='Training')
            plt.plot(training_stats['val_acc'], label='Validierung')
            plt.title(f'Modellgenauigkeit: {model_name}')
            plt.xlabel('Epoche')
            plt.ylabel('Genauigkeit (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(training_stats['train_loss'], label='Training')
            plt.plot(training_stats['val_loss'], label='Validierung')
            plt.title(f'Modellverlust: {model_name}')
            plt.xlabel('Epoche')
            plt.ylabel('Verlust')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name}_training.png"))
            plt.close()
            
            # Konfusionsmatrix
            cm = np.array(eval_metrics['confusion_matrix'])
            if cm.shape[0] > 1:  # Nur anzeigen, wenn mehr als eine Klasse
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Konfusionsmatrix: {model_name}')
                plt.ylabel('Tatsächlich')
                plt.xlabel('Vorhergesagt')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name}_confusion.png"))
                plt.close()
    
    # Zusammenfassungsdatei erstellen
    df = pd.DataFrame(results)
    
    # Nach Genauigkeit sortieren
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    # Ergebnisse als Excel speichern
    excel_path = os.path.join(args.output_dir, 'hyperparameter_results.xlsx')
    df_sorted.to_excel(excel_path, index=False)
    
    # Pareto-Optimale Modelle identifizieren (Accuracy vs. Model Size)
    pareto_optimal = []
    
    for _, row1 in df.iterrows():
        is_pareto = True
        for _, row2 in df.iterrows():
            if (row2['accuracy'] > row1['accuracy'] and 
                row2['int8_size_kb'] <= row1['int8_size_kb']):
                # Eine andere Konfiguration ist besser in Accuracy und gleich/besser in Size
                is_pareto = False
                break
        if is_pareto:
            pareto_optimal.append(row1['model_name'])
    
    # Visualisierungen der Gesamtergebnisse
    if args.visualize and df.shape[0] > 1:
        # Accuracy vs. Model Size (mit Pareto-Front)
        plt.figure(figsize=(10, 6))
        
        # Alle Punkte plotten
        plt.scatter(df['int8_size_kb'], df['accuracy'], alpha=0.7, s=50)
        
        # Pareto-Optimale Punkte hervorheben
        df_pareto = df[df['model_name'].isin(pareto_optimal)]
        plt.scatter(df_pareto['int8_size_kb'], df_pareto['accuracy'], 
                   color='red', s=100, label='Pareto-Optimale Modelle')
        
        # Labels für pareto-optimale Modelle
        for _, row in df_pareto.iterrows():
            plt.annotate(row['model_name'], 
                        (row['int8_size_kb'], row['accuracy']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        # Pareto-Front verbinden
        df_pareto_sorted = df_pareto.sort_values('int8_size_kb')
        plt.plot(df_pareto_sorted['int8_size_kb'], df_pareto_sorted['accuracy'], 
                'r--', alpha=0.5)
        
        plt.title('Genauigkeit vs. Modellgröße')
        plt.xlabel('Modellgröße (KB, Int8-Quantisiert)')
        plt.ylabel('Validierungs-Genauigkeit (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', 'accuracy_vs_size.png'))
        plt.close()
        
        # Accuracy vs. Inference Time
        plt.figure(figsize=(10, 6))
        plt.scatter(df['estimated_rp2040_time_ms'], df['accuracy'], alpha=0.7, s=50)
        plt.title('Genauigkeit vs. Inferenzzeit')
        plt.xlabel('Geschätzte Inferenzzeit auf RP2040 (ms)')
        plt.ylabel('Validierungs-Genauigkeit (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', 'accuracy_vs_time.png'))
        plt.close()
        
        # Architektur-Heatmap
        if len(df['depth_multiplier'].unique()) > 1 and len(df['num_blocks'].unique()) > 1:
            # Erstelle Pivot-Tabelle für Heatmap
            pivot = df.pivot_table(
                index='depth_multiplier', 
                columns='num_blocks',
                values='accuracy',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Genauigkeit nach Depth-Multiplier und Blockanzahl')
            plt.xlabel('Anzahl der Blöcke')
            plt.ylabel('Depth-Multiplier')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'plots', 'architecture_heatmap.png'))
            plt.close()
    
    # Zusammenfassung ausgeben
    logger.info("\n" + "="*50)
    logger.info("ZUSAMMENFASSUNG DER HYPERPARAMETER-SUCHE")
    logger.info("="*50)
    logger.info(f"Analysierte Modelle: {len(model_configs)}")
    logger.info(f"Pareto-optimale Modelle: {len(pareto_optimal)}")
    
    # Top-5 Modelle nach Genauigkeit
    logger.info("\nTop 5 Modelle nach Genauigkeit:")
    top5_acc = df_sorted.head(5)
    for i, (_, row) in enumerate(top5_acc.iterrows()):
        logger.info(f"{i+1}. {row['model_name']}: "
                   f"Acc={row['accuracy']:.2f}%, "
                   f"Size={row['int8_size_kb']:.1f}KB, "
                   f"Time={row['estimated_rp2040_time_ms']:.1f}ms")
    
    # Top-5 Modelle nach Speichereffizienz (Accuracy/KB)
    df_sorted['efficiency'] = df_sorted['accuracy'] / df_sorted['int8_size_kb']
    top5_eff = df_sorted.sort_values('efficiency', ascending=False).head(5)
    logger.info("\nTop 5 Modelle nach Speichereffizienz (Accuracy/KB):")
    for i, (_, row) in enumerate(top5_eff.iterrows()):
        logger.info(f"{i+1}. {row['model_name']}: "
                   f"Eff={row['efficiency']:.2f}%/KB, "
                   f"Acc={row['accuracy']:.2f}%, "
                   f"Size={row['int8_size_kb']:.1f}KB")
    
    logger.info("\nErgebnisse wurden gespeichert in:")
    logger.info(f"- CSV: {csv_path}")
    logger.info(f"- Excel: {excel_path}")
    if args.visualize:
        logger.info(f"- Plots: {os.path.join(args.output_dir, 'plots')}")
    if args.save_models:
        logger.info(f"- Modelle: {os.path.join(args.output_dir, 'models')}")
    
    return {
        'csv_path': csv_path,
        'excel_path': excel_path,
        'results_df': df_sorted,
        'pareto_optimal': pareto_optimal
    }


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Hyperparameter-Suche für Pizza-Erkennungsmodell')
    parser.add_argument('--data-dir', default='data/augmented', help='Datensatzverzeichnis')
    parser.add_argument('--output-dir', default='output/hyperparameter_search', help='Ausgabeverzeichnis')
    parser.add_argument('--epochs', type=int, default=25, help='Anzahl der Trainingsepochen pro Modell')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch-Größe für das Training')
    parser.add_argument('--search-size', default='medium', choices=['small', 'medium', 'full'], 
                        help='Größe der Hyperparameter-Suche')
    parser.add_argument('--early-stopping', action='store_true', help='Early Stopping verwenden')
    parser.add_argument('--save-models', action='store_true', help='Trainierte Modelle speichern')
    parser.add_argument('--visualize', action='store_true', help='Visualisierungen erstellen')
    
    args = parser.parse_args()
    
    # Defaults für bool-Flags, wenn nicht explizit gesetzt
    if not any([arg.startswith('--early-stopping') for arg in sys.argv]):
        args.early_stopping = True
    if not any([arg.startswith('--visualize') for arg in sys.argv]):
        args.visualize = True
    
    # Starte Suche
    run_hyperparameter_search(args)


if __name__ == "__main__":
    main()