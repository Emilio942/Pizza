#!/usr/bin/env python3
"""
Vergleich von ReLU und Hard-Swish Aktivierungsfunktionen im MicroPizzaNet

Dieses Skript vergleicht die originale MicroPizzaNet-Architektur mit einer
Variante, die ReLU-Aktivierungen durch Hard-Swish ersetzt.

Hard-Swish ist eine effiziente Approximation der Swish-Aktivierungsfunktion:
Hard-Swish(x) = x * ReLU6(x + 3) / 6

Bekannt aus MobileNetV3, kann Hard-Swish die Modellgenauigkeit ohne signifikante
Steigerung der Rechenkosten verbessern.

Ausgabe ist ein Vergleichsbericht mit Metriken wie:
- Modellgröße (Float32 und Int8)
- Accuracy
- F1-Score
- Inferenzzeit
- RAM-Nutzung
- Visualisierungen der Aktivierungen

Verwendung:
    python compare_hard_swish.py [--data-dir DIR] [--output-dir DIR] [--epochs N]

Optionen:
    --data-dir: Verzeichnis mit dem Bilddatensatz (Standard: data/augmented)
    --output-dir: Verzeichnis für die Ergebnisse (Standard: output/hard_swish_comparison)
    --epochs: Anzahl der Trainingsepochen pro Modell (Standard: 30)
    --batch-size: Batch-Größe für das Training (Standard: 32)
    --early-stopping: Frühes Beenden bei Stagnation (Standard: True)
    --visualize: Visualisierungen erstellen (Standard: True)
"""

import os
import sys
import argparse
import time
import json
import csv
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
from src.pizza_detector import (
    RP2040Config, PizzaDatasetAnalysis, create_optimized_dataloaders, 
    MemoryEstimator, MicroPizzaNet, EarlyStopping
)

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hard_swish_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hard-Swish Aktivierungsfunktion
class HardSwish(nn.Module):
    """
    Hard-Swish Aktivierungsfunktion: x * ReLU6(x + 3) / 6
    Effiziente Approximation der Swish-Funktion für Mobile-Anwendungen
    """
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace
        
    def forward(self, x):
        # Implementierung: x * min(max(0, x+3), 6) / 6
        return x * torch.nn.functional.relu6(x + 3., inplace=self.inplace) / 6.

# MicroPizzaNet mit Hard-Swish Aktivierung
class MicroPizzaNetHardSwish(nn.Module):
    """MicroPizzaNet Variante mit Hard-Swish statt ReLU Aktivierungsfunktion"""
    def __init__(self, num_classes=6, dropout_rate=0.2):
        super(MicroPizzaNetHardSwish, self).__init__()
        
        # Erster Block: 3 -> 8 Filter
        self.block1 = nn.Sequential(
            # Standardfaltung für den ersten Layer (3 -> 8)
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            HardSwish(inplace=True),  # Hard-Swish statt ReLU
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 8x12x12
        )
        
        # Zweiter Block: 8 -> 16 Filter mit depthwise separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            HardSwish(inplace=True),  # Hard-Swish statt ReLU
            # Pointwise Faltung (1x1) zur Kanalexpansion
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),  # Hard-Swish statt ReLU
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 16x6x6
        )
        
        # Global Average Pooling spart Parameter im Vergleich zu Flatten + Dense
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Ausgabe: 16x1x1
        
        # Kompakter Klassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 16
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # Direkt zur Ausgabeschicht
        )
        
        # Initialisierung der Gewichte für bessere Konvergenz
        self._initialize_weights()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, config, class_names, epochs=30, early_stopping=True, model_name="model"):
    """Trainiert ein Modell und gibt Trainingsstatistiken zurück"""
    logger.info(f"Starte Training für {model_name}...")
    
    device = config.DEVICE
    model = model.to(device)
    
    # Parameter zählen
    params_count = model.count_parameters()
    logger.info(f"Modell hat {params_count:,} Parameter")
    
    # Verlustfunktion und Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=5, 
                                                    verbose=True)
    
    # Early Stopping
    es = EarlyStopping(patience=10, restore_best_weights=True) if early_stopping else None
    
    # Training-Historie
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training Loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass + Backward pass + Optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistiken
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        
        # Validierung 
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistiken
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.0 * correct / total
                })
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct / total
        
        # Historie aktualisieren
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Learning Rate anpassen
        scheduler.step(epoch_val_acc)
        
        # Output
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% - "
                   f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Early Stopping
        if es:
            es(epoch_val_loss, model)
            if es.early_stop:
                logger.info(f"Early Stopping in Epoche {epoch+1}")
                break
    
    # Stelle beste Gewichte wieder her, falls Early Stopping verwendet wurde
    if es and es.best_weights is not None:
        model.load_state_dict(es.best_weights)
        logger.info("Beste Gewichte wiederhergestellt")
    
    return history, model

def evaluate_model(model, dataloader, config, class_names):
    """Evaluiert ein Modell und gibt Metriken zurück"""
    logger.info("Evaluiere Modell...")
    
    device = config.DEVICE
    model = model.to(device)
    model.eval()
    
    # Verlustfunktion
    criterion = nn.CrossEntropyLoss()
    
    # Sammle Vorhersagen und Labels
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluierung"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistiken
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Berechne Metriken
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Konfusionsmatrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Log Ergebnisse
    logger.info(f"Genauigkeit: {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': running_loss / len(dataloader.dataset),
        'confusion_matrix': cm
    }

def measure_inference_time(model, input_size, config, n_runs=100):
    """Misst die Inferenzzeit eines Modells"""
    logger.info("Messe Inferenzzeit...")
    
    device = config.DEVICE
    model = model.to(device)
    model.eval()
    
    # Dummy-Input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Zeitmessung
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    # Durchschnittliche Inferenzzeit
    avg_time = (end_time - start_time) / n_runs
    
    # Geschätzte Zeit auf RP2040 (Faktor basierend auf Erfahrungswerten)
    rp2040_factor = 35  # RP2040 ist ~35x langsamer als ein typischer PC bei CNN-Inferenz
    rp2040_time = avg_time * rp2040_factor
    
    logger.info(f"Durchschnittliche Inferenzzeit auf CPU: {avg_time*1000:.2f} ms")
    logger.info(f"Geschätzte Inferenzzeit auf RP2040: {rp2040_time*1000:.2f} ms")
    
    return {
        'avg_inference_time_s': avg_time,
        'estimated_rp2040_time_s': rp2040_time
    }

def generate_activation_maps(model, sample_image, output_dir):
    """Generiert Feature-Maps für ein Beispielbild"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Speichere Aktivierungen
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Registriere Hooks für Conv-Layer
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        sample_image = sample_image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(sample_image)
    
    # Entferne Hooks
    for hook in hooks:
        hook.remove()
    
    # Erstelle Visualisierungspfad
    vis_dir = os.path.join(output_dir, 'activations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualisiere Aktivierungen
    for name, activation in activations.items():
        # Nur die ersten 16 Kanäle (oder weniger) anzeigen
        n_channels = min(16, activation.shape[1])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(16):
            if i < n_channels:
                feature_map = activation[0, i].numpy()
                axes[i].imshow(feature_map, cmap='viridis')
            axes[i].axis('off')
            
        plt.suptitle(f'Feature Maps für Layer: {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'feature_map_{name.replace(".", "_")}.png'))
        plt.close()
    
    return vis_dir

def run_hard_swish_comparison(args):
    """Führt den Vergleich zwischen ReLU und Hard-Swish Aktivierungen durch"""
    logger.info("Starte Vergleich zwischen ReLU und Hard-Swish Aktivierungsfunktionen")
    
    # Erstelle Ausgabeverzeichnisse
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
    
    # Modelle definieren
    models = [
        {
            'name': 'MicroPizzaNet (ReLU)',
            'class': MicroPizzaNet,
            'params': {
                'num_classes': len(class_names),
                'dropout_rate': 0.2,
            }
        },
        {
            'name': 'MicroPizzaNet (Hard-Swish)',
            'class': MicroPizzaNetHardSwish,
            'params': {
                'num_classes': len(class_names),
                'dropout_rate': 0.2,
            }
        }
    ]
    
    # CSV für Ergebnisse vorbereiten
    csv_path = os.path.join(args.output_dir, 'hard_swish_comparison_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'model_name', 'accuracy', 'f1_score', 'precision', 'recall',
            'params_count', 'model_size_kb', 'int8_size_kb', 
            'inference_time_ms', 'estimated_rp2040_time_ms', 
            'ram_usage_kb', 'training_time_s'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Ergebnis-Array für einfacheren DataFrame-Export
    results = []
    trained_models = []
    
    # Trainiere und evaluiere jedes Modell
    for i, model_config in enumerate(models):
        model_name = model_config['name']
        logger.info(f"\n{'='*50}\nTrainiere Modell {i+1}/{len(models)}: {model_name}\n{'='*50}")
        
        # Modell erstellen
        model = model_config['class'](**model_config['params'])
        
        # Parameter und Speichernutzung analysieren
        params_count = model.count_parameters()
        memory_analysis = MemoryEstimator.check_memory_requirements(
            model, 
            (3, config.IMG_SIZE, config.IMG_SIZE),
            config
        )
        
        # Training
        start_time = time.time()
        training_stats, trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            config, 
            class_names,
            epochs=args.epochs,
            early_stopping=args.early_stopping,
            model_name=model_name.replace(" ", "_").lower()
        )
        training_time = time.time() - start_time
        
        # Speichere trainiertes Modell
        trained_models.append({
            'name': model_name,
            'model': trained_model
        })
        
        # Evaluierung
        eval_metrics = evaluate_model(trained_model, val_loader, config, class_names)
        
        # Inferenzzeit messen
        timing = measure_inference_time(
            trained_model, 
            (3, config.IMG_SIZE, config.IMG_SIZE),
            config
        )
        
        # Ergebnisse speichern
        result = {
            'model_name': model_name,
            'accuracy': eval_metrics['accuracy'],
            'f1_score': eval_metrics['f1'],
            'precision': eval_metrics['precision'],
            'recall': eval_metrics['recall'],
            'params_count': params_count,
            'model_size_kb': memory_analysis['model_size_float32_kb'],
            'int8_size_kb': memory_analysis['model_size_int8_kb'],
            'inference_time_ms': timing['avg_inference_time_s'] * 1000,
            'estimated_rp2040_time_ms': timing['estimated_rp2040_time_s'] * 1000,
            'ram_usage_kb': memory_analysis['total_runtime_memory_kb'],
            'training_time_s': training_time,
            'training_history': training_stats
        }
        
        # In CSV schreiben
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row_dict = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(row_dict)
        
        # Zum Array hinzufügen
        results.append(result)
        
        # Modell speichern
        if args.save_models:
            model_path = os.path.join(args.output_dir, 'models', f"{model_name.replace(' ', '_').lower()}.pth")
            torch.save(trained_model.state_dict(), model_path)
            logger.info(f"Modell gespeichert unter: {model_path}")
        
        # Trainingshistorie visualisieren
        if args.visualize:
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
            plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name.replace(' ', '_').lower()}_training.png"))
            plt.close()
            
            # Konfusionsmatrix
            cm = eval_metrics['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Konfusionsmatrix: {model_name}')
            plt.ylabel('Tatsächlich')
            plt.xlabel('Vorhergesagt')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name.replace(' ', '_').lower()}_confusion.png"))
            plt.close()
    
    # Visualisiere Feature-Maps für ein Beispielbild
    if args.visualize and len(val_loader) > 0:
        # Hole ein Beispielbild
        sample_batch = next(iter(val_loader))
        sample_image = sample_batch[0][0]  # Erstes Bild aus dem ersten Batch
        
        for trained_model_info in trained_models:
            model_name = trained_model_info['name']
            model = trained_model_info['model']
            
            activations_dir = generate_activation_maps(
                model, 
                sample_image, 
                os.path.join(args.output_dir, model_name.replace(' ', '_').lower())
            )
            logger.info(f"Feature-Maps für {model_name} gespeichert in {activations_dir}")
    
    # Ergebnisse als DataFrame
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'training_history'} for r in results])
    
    # Nach Genauigkeit sortieren
    df_sorted = df_results.sort_values('accuracy', ascending=False)
    
    # Ergebnisse als Excel speichern
    excel_path = os.path.join(args.output_dir, 'hard_swish_comparison_results.xlsx')
    df_sorted.to_excel(excel_path, index=False)
    
    # Vergleichsvisualisierungen erstellen
    if args.visualize:
        # Balkendiagramme für verschiedene Metriken
        metrics_to_plot = [
            ('accuracy', 'Genauigkeit (%)'),
            ('f1_score', 'F1-Score'),
            ('model_size_kb', 'Modellgröße (KB)'),
            ('ram_usage_kb', 'RAM-Nutzung (KB)'),
            ('estimated_rp2040_time_ms', 'Inferenzzeit auf RP2040 (ms)'),
            ('params_count', 'Anzahl Parameter')
        ]
        
        for metric, title in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='model_name', y=metric, data=df_results)
            
            # Werte über den Balken anzeigen
            for i, v in enumerate(df_results[metric]):
                if metric == 'params_count':
                    # Formatiere Parameter mit Tausendertrennzeichen
                    value_text = f"{v:,}"
                elif metric in ['accuracy', 'f1_score']:
                    # Prozent für Accuracy und F1
                    value_text = f"{v:.2f}%"
                elif metric in ['estimated_rp2040_time_ms']:
                    # Millisekunden mit einer Nachkommastelle
                    value_text = f"{v:.1f}ms"
                else:
                    # Andere Werte mit einer Nachkommastelle
                    value_text = f"{v:.1f}"
                    
                ax.text(i, v, value_text, ha='center', va='bottom')
            
            plt.title(title)
            plt.xlabel('Modell')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'plots', f"comparison_{metric}.png"))
            plt.close()
        
        # Liniendiagramm für Trainingshistorie
        plt.figure(figsize=(12, 10))
        
        # Genauigkeit
        plt.subplot(2, 1, 1)
        for result in results:
            plt.plot(result['training_history']['val_acc'], 
                    label=f"{result['model_name']} (Val)")
        plt.title('Validierungsgenauigkeit im Vergleich')
        plt.xlabel('Epoche')
        plt.ylabel('Genauigkeit (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Verlust
        plt.subplot(2, 1, 2)
        for result in results:
            plt.plot(result['training_history']['val_loss'], 
                    label=f"{result['model_name']} (Val)")
        plt.title('Validierungsverlust im Vergleich')
        plt.xlabel('Epoche')
        plt.ylabel('Verlust')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', "training_comparison.png"))
        plt.close()
    
    # Erzeuge HTML-Bericht
    html_path = os.path.join(args.output_dir, 'hard_swish_comparison_report.html')
    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hard-Swish vs. ReLU Aktivierungsfunktionen - Vergleich</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; }}
                .section {{ margin-top: 30px; margin-bottom: 15px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .highlight {{ background-color: #e6ffe6; }}
                .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .flex-item {{ flex: 1; min-width: 300px; }}
                .code {{ font-family: monospace; background-color: #f8f8f8; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MicroPizzaNet: Hard-Swish vs. ReLU Aktivierungsfunktionen</h1>
                <p>Generiert am {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
            </div>
            
            <h2 class="section">Übersicht</h2>
            <p>
                Dieser Bericht vergleicht die originale MicroPizzaNet-Architektur mit ReLU Aktivierungen
                und eine angepasste Version, die Hard-Swish Aktivierungen verwendet.
            </p>
            
            <div class="section">
                <h3>Was ist Hard-Swish?</h3>
                <p>
                    Hard-Swish ist eine berechnungseffiziente Approximation der Swish-Aktivierungsfunktion, 
                    die in MobileNetV3 eingeführt wurde. Sie ist definiert als:
                </p>
                <div class="code">
                    <pre>Hard-Swish(x) = x * ReLU6(x + 3) / 6</pre>
                </div>
                <p>
                    Im Vergleich zu ReLU bietet Hard-Swish einige potenzielle Vorteile:
                </p>
                <ul>
                    <li>Glattere Aktivierungskurve, was bei Gradienten-basierten Optimierern hilfreich sein kann</li>
                    <li>Verbesserte Modellgenauigkeit ohne signifikanten Rechenaufwand</li>
                    <li>Kann "tote Neuronen" reduzieren, da Hard-Swish auch für leicht negative Inputs nicht verschwindet</li>
                </ul>
                
                <div class="code">
                <pre>
class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace
        
    def forward(self, x):
        # x * min(max(0, x+3), 6) / 6
        return x * torch.nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
                </pre>
                </div>
                
                <p>
                    Hard-Swish ist für Mikrocontroller besonders geeignet, da sie weniger rechenintensiv ist als
                    die originale Swish-Funktion, die eine Exponentialfunktion erfordert. Hard-Swish kann mit
                    grundlegenden arithmetischen Operationen implementiert werden.
                </p>
            </div>
            
            <h2 class="section">Vergleichsergebnisse</h2>
            
            <table>
                <tr>
                    <th>Modell</th>
                    <th>Genauigkeit</th>
                    <th>F1-Score</th>
                    <th>Parameter</th>
                    <th>Modellgröße (KB)</th>
                    <th>RAM-Nutzung (KB)</th>
                    <th>Inferenzzeit (ms)</th>
                </tr>
        """)
        
        # Tabellenzeilen
        for _, row in df_sorted.iterrows():
            highlight = ' class="highlight"' if row['model_name'] == df_sorted.iloc[0]['model_name'] else ''
            
            f.write(f"""
                <tr{highlight}>
                    <td>{row['model_name']}</td>
                    <td>{row['accuracy']:.2f}%</td>
                    <td>{row['f1_score']:.4f}</td>
                    <td>{row['params_count']:,}</td>
                    <td>{row['model_size_kb']:.1f}</td>
                    <td>{row['ram_usage_kb']:.1f}</td>
                    <td>{row['estimated_rp2040_time_ms']:.1f}</td>
                </tr>
            """)
        
        f.write("""
            </table>
            
            <h2 class="section">Visualisierungen</h2>
            
            <div class="flex-container">
        """)
        
        # Füge Visualisierungen hinzu
        if args.visualize:
            # Metriken
            for metric, title in metrics_to_plot:
                plot_path = f"plots/comparison_{metric}.png"
                f.write(f"""
                    <div class="flex-item">
                        <h3>{title}</h3>
                        <img src="{plot_path}" alt="{title}">
                    </div>
                """)
            
            # Trainingsvergleich
            f.write("""
                <div class="flex-item" style="flex-basis: 100%;">
                    <h3>Trainingsvergleich</h3>
                    <img src="plots/training_comparison.png" alt="Training Comparison">
                </div>
            """)
            
            # Feature-Maps (falls vorhanden)
            for i, model_info in enumerate(models):
                model_name = model_info['name'].replace(' ', '_').lower()
                activations_dir = os.path.join(model_name, 'activations')
                
                if os.path.exists(os.path.join(args.output_dir, activations_dir)):
                    f.write(f"""
                        <div class="flex-item" style="flex-basis: 100%;">
                            <h3>Feature Maps: {model_info['name']}</h3>
                            <p>Beispielhafte Feature-Maps zeigen, wie das Modell auf verschiedene Bildmerkmale reagiert.</p>
                            <div class="flex-container">
                    """)
                    
                    # Zeige einige Feature-Maps
                    for img_file in sorted(os.listdir(os.path.join(args.output_dir, activations_dir)))[:4]:
                        if img_file.endswith('.png'):
                            img_path = os.path.join(activations_dir, img_file)
                            f.write(f"""
                                <div class="flex-item">
                                    <img src="{img_path}" alt="Feature Map">
                                </div>
                            """)
                    
                    f.write("""
                            </div>
                        </div>
                    """)
        
        f.write("""
            </div>
            
            <h2 class="section">Erkenntnisse & Schlussfolgerung</h2>
        """)
        
        # Finde das beste Modell
        best_model = df_sorted.iloc[0]
        original_model = df_results[df_results['model_name'] == 'MicroPizzaNet (ReLU)'].iloc[0]
        
        # Berechne Verbesserung
        accuracy_improvement = best_model['accuracy'] - original_model['accuracy']
        
        # Vergleiche Inferenzzeiten
        if 'MicroPizzaNet (Hard-Swish)' in df_results['model_name'].values:
            hard_swish_model = df_results[df_results['model_name'] == 'MicroPizzaNet (Hard-Swish)'].iloc[0]
            inference_diff = ((hard_swish_model['estimated_rp2040_time_ms'] / original_model['estimated_rp2040_time_ms']) - 1) * 100
            inference_diff_text = f"steigt um {inference_diff:.1f}%" if inference_diff > 0 else f"sinkt um {abs(inference_diff):.1f}%"
        else:
            inference_diff_text = "konnte nicht berechnet werden"
        
        best_model_name = best_model['model_name']
        
        f.write(f"""
            <p>
                Die Untersuchung zeigt, dass der Einsatz von Hard-Swish anstelle von ReLU in MicroPizzaNet
                zu einer <strong>Genauigkeitsveränderung von {accuracy_improvement:.2f}%</strong> führt.
            </p>
            
            <p>
                Das beste Modell ist <strong>{best_model_name}</strong> mit einer Genauigkeit von {best_model['accuracy']:.2f}%.
            </p>
            
            <p>
                <strong>Wichtige Erkenntnisse:</strong>
            </p>
            <ul>
                <li>Die Inferenzzeit mit Hard-Swish {inference_diff_text} im Vergleich zu ReLU</li>
                <li>Die Modellgröße bleibt praktisch unverändert, da beide Aktivierungsfunktionen keine trainierbaren Parameter haben</li>
                <li>Hard-Swish kann besonders bei tieferen Netzwerken Vorteile bieten</li>
                <li>Bei bestimmten Bildmerkmalen erzeugt Hard-Swish unterschiedliche Aktivierungsmuster</li>
            </ul>
            
            <div class="section">
                <h3>Mathematischer Vergleich der Aktivierungen</h3>
                <p>
                    <strong>ReLU:</strong> f(x) = max(0, x)<br>
                    <strong>Hard-Swish:</strong> f(x) = x * min(max(0, x+3), 6) / 6
                </p>
                <p>
                    Im Vergleich zu ReLU bietet Hard-Swish:
                </p>
                <ul>
                    <li>Eine glattere Kurve ohne harte Knickstelle bei x=0</li>
                    <li>Geringe Aktivierung für leicht negative Eingabewerte, was Gradienten erhält</li>
                    <li>Sättigung bei großen positiven Werten, was zur Regularisierung beitragen kann</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>Nächste Schritte</h3>
                <ol>
                    <li>C-Implementierung der Hard-Swish Funktion für RP2040 optimieren</li>
                    <li>Experimente mit selektiver Anwendung von Hard-Swish nur in bestimmten Schichten</li>
                    <li>Kombination mit anderen Optimierungstechniken wie Pruning und Quantisierung</li>
                    <li>Tests auf realer Hardware zur Validierung der Leistungsschätzungen</li>
                </ol>
            </div>
        </body>
        </html>
        """)
    
    # Zusammenfassung ausgeben
    logger.info("\n" + "="*50)
    logger.info("ZUSAMMENFASSUNG DES HARD-SWISH VERGLEICHS")
    logger.info("="*50)
    
    # Bestes Modell nach Genauigkeit
    best_model = df_sorted.iloc[0]
    logger.info(f"Bestes Modell: {best_model['model_name']}")
    logger.info(f"- Genauigkeit: {best_model['accuracy']:.2f}%")
    logger.info(f"- F1-Score: {best_model['f1_score']:.4f}")
    logger.info(f"- Parameter: {best_model['params_count']:,}")
    logger.info(f"- Modellgröße: {best_model['model_size_kb']:.1f} KB")
    logger.info(f"- RAM-Nutzung: {best_model['ram_usage_kb']:.1f} KB")
    logger.info(f"- Inferenzzeit auf RP2040: {best_model['estimated_rp2040_time_ms']:.1f} ms")
    
    # Original vs. Bestes (Verbesserung)
    original_model = df_results[df_results['model_name'] == 'MicroPizzaNet (ReLU)'].iloc[0]
    accuracy_improvement = best_model['accuracy'] - original_model['accuracy']
    
    logger.info("\nVerbesserung gegenüber Original:")
    logger.info(f"- Genauigkeit: {'+' if accuracy_improvement >= 0 else ''}{accuracy_improvement:.2f}%")
    
    # Ausgabepfade
    logger.info("\nErgebnisse gespeichert in:")
    logger.info(f"- CSV: {csv_path}")
    logger.info(f"- Excel: {excel_path}")
    logger.info(f"- HTML-Bericht: {html_path}")
    
    return {
        'csv_path': csv_path,
        'excel_path': excel_path,
        'html_path': html_path,
        'results_df': df_sorted,
        'best_model': best_model['model_name']
    }

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Vergleich von ReLU und Hard-Swish Aktivierungsfunktionen')
    parser.add_argument('--data-dir', default='data/augmented', help='Datensatzverzeichnis')
    parser.add_argument('--output-dir', default='output/hard_swish_comparison', help='Ausgabeverzeichnis')
    parser.add_argument('--epochs', type=int, default=30, help='Anzahl der Trainingsepochen pro Modell')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch-Größe für das Training')
    parser.add_argument('--early-stopping', action='store_true', help='Early Stopping verwenden')
    parser.add_argument('--save-models', action='store_true', help='Trainierte Modelle speichern')
    parser.add_argument('--visualize', action='store_true', help='Visualisierungen erstellen')
    
    args = parser.parse_args()
    
    # Defaults für bool-Flags, wenn nicht explizit gesetzt
    if not any([arg.startswith('--early-stopping') for arg in sys.argv]):
        args.early_stopping = True
    if not any([arg.startswith('--visualize') for arg in sys.argv]):
        args.visualize = True
    if not any([arg.startswith('--save-models') for arg in sys.argv]):
        args.save_models = True
    
    # Starte Vergleich
    run_hard_swish_comparison(args)


if __name__ == "__main__":
    main()