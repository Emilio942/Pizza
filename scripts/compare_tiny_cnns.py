#!/usr/bin/env python3
"""
Vergleich alternativer Tiny-CNNs für das Pizza-Erkennungssystem

Dieses Skript vergleicht verschiedene leichtgewichtige CNN-Architekturen
für die Verwendung auf dem RP2040-Mikrocontroller, darunter:
1. MicroPizzaNet (die aktuelle Baseline)
2. MCUNet (eine für Mikrocontroller optimierte Architektur)
3. MobilePizzaNet (eine reduzierte MobileNetV2-Variante)

Ausgabe ist ein Vergleichsbericht mit Metriken wie:
- Modellgröße (Float32 und Int8)
- Accuracy
- F1-Score
- Inferenzzeit
- RAM-Nutzung

Verwendung:
    python compare_tiny_cnns.py [--data-dir DIR] [--output-dir DIR] [--epochs N]

Optionen:
    --data-dir: Verzeichnis mit dem Bilddatensatz (Standard: data/augmented)
    --output-dir: Verzeichnis für die Ergebnisse (Standard: output/tiny_cnn_comparison)
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
from src.pizza_detector import RP2040Config, PizzaDatasetAnalysis, create_optimized_dataloaders, MemoryEstimator
from src.metrics import ModelMetrics
from scripts.hyperparameter_search import ModularMicroPizzaNet, train_model, evaluate_model, measure_inference_time, EarlyStopping

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tiny_cnn_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCUNet(nn.Module):
    """
    MCUNet-Implementierung für RP2040 - eine für Mikrocontroller optimierte Architektur
    Basierend auf dem Paper "MCUNet: Tiny Deep Learning on IoT Devices"
    Angepasst an die Ressourcenbeschränkungen des RP2040
    """
    def __init__(self, num_classes=6, input_channels=3, img_size=48, width_mult=0.5):
        super(MCUNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        self.width_mult = width_mult
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # Pointwise
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        
        # Berechne tatsächliche Kanalbreiten basierend auf width_mult
        def make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Stelle sicher, dass die Reduktion nicht mehr als 10% beträgt
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = make_divisible(32 * width_mult)
        
        # Architektur definieren - erheblich reduziert für RP2040
        layers = []
        
        # Ersten Layer - Standard Conv
        layers.append(conv_bn(input_channels, input_channel, 2))
        
        # Depthwise Separable Convs
        # Reduziert auf nur 3 Blöcke, um in RAM zu passen
        block_config = [
            # t, c, n, s - expansion, channels, num_blocks, stride
            [1, make_divisible(16 * width_mult), 1, 1],
            [6, make_divisible(24 * width_mult), 2, 2],
            [6, make_divisible(32 * width_mult), 2, 2],
        ]
        
        # Feature Extractor bauen
        for t, c, n, s in block_config:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(conv_dw(input_channel, output_channel, stride))
                input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        
        # Global Pooling und Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, num_classes),
        )
        
        # Gewichte initialisieren
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MobilePizzaNet(nn.Module):
    """
    Stark reduzierte MobileNetV2-Variante für den Einsatz auf dem RP2040
    Angepasst mit invertierten Residual-Blöcken und Bottleneck-Struktur
    """
    def __init__(self, num_classes=6, input_channels=3, img_size=48, width_mult=0.25):
        super(MobilePizzaNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        self.width_mult = width_mult
        
        # Block für invertierte Residual-Struktur
        class InvertedResidual(nn.Module):
            def __init__(self, inp, oup, stride, expand_ratio):
                super(InvertedResidual, self).__init__()
                self.stride = stride
                assert stride in [1, 2]

                hidden_dim = int(inp * expand_ratio)
                self.use_res_connect = self.stride == 1 and inp == oup

                layers = []
                # Expansion
                if expand_ratio != 1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True)
                    ))
                
                # Depthwise
                layers.append(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                ))
                
                # Projection
                layers.append(nn.Sequential(
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup)
                ))
                
                self.conv = nn.Sequential(*layers)

            def forward(self, x):
                if self.use_res_connect:
                    return x + self.conv(x)
                else:
                    return self.conv(x)
        
        # Berechne tatsächliche Kanalbreiten basierend auf width_mult
        def make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = make_divisible(32 * width_mult)
        
        # Ersten Layer - Standard Conv
        self.first_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Stark reduzierte Architektur für RP2040
        # t, c, n, s - expansion ratio, channel, num blocks, stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # Kein Expand im ersten Block
            [6, 24, 2, 2],  # Halbe Auflösung
            [6, 32, 2, 2],  # 1/4 Auflösung
        ]
        
        # Baue die Blöcke
        features = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Global Pooling und Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, num_classes),
        )
        
        # Gewichte initialisieren
        self._initialize_weights()
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_model_comparison(args):
    """Führt den Vergleich der verschiedenen CNN-Architekturen durch"""
    logger.info("Starte Vergleich alternativer Tiny-CNNs für Pizza-Erkennung")
    
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
    
    # Definiere die zu vergleichenden Modelle
    model_configs = [
        {
            'name': 'MicroPizzaNet (Baseline)',
            'class': ModularMicroPizzaNet,
            'params': {
                'num_classes': len(class_names),
                'input_channels': 3,
                'img_size': 48,
                'depth_multiplier': 1.0,
                'num_blocks': 3,
                'initial_channels': 8,
                'use_separable': True,
                'use_gpool': True
            }
        },
        {
            'name': 'MCUNet',
            'class': MCUNet,
            'params': {
                'num_classes': len(class_names),
                'input_channels': 3,
                'img_size': 48,
                'width_mult': 0.5  # 50% der Original-Kanalbreite
            }
        },
        {
            'name': 'MobilePizzaNet',
            'class': MobilePizzaNet,
            'params': {
                'num_classes': len(class_names),
                'input_channels': 3,
                'img_size': 48,
                'width_mult': 0.25  # 25% der Original-Kanalbreite
            }
        }
    ]
    
    # CSV für Ergebnisse vorbereiten
    csv_path = os.path.join(args.output_dir, 'model_comparison_results.csv')
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
    
    # Trainiere und evaluiere jedes Modell
    for i, model_config in enumerate(model_configs):
        model_name = model_config['name']
        logger.info(f"\n{'='*50}\nTrainiere Modell {i+1}/{len(model_configs)}: {model_name}\n{'='*50}")
        
        # Modell erstellen
        model = model_config['class'](**model_config['params'])
        
        # Parameter und Speichernutzung analysieren
        params_count = model.count_parameters()
        memory_analysis = MemoryEstimator.check_memory_requirements(
            model, 
            (3, 48, 48),  # Feste Bildgröße für RP2040
            config
        )
        
        logger.info(f"Modell hat {params_count:,} Parameter")
        logger.info(f"Geschätzte Modellgröße: {memory_analysis['model_size_float32_kb']:.2f} KB (float32), "
                   f"{memory_analysis['model_size_int8_kb']:.2f} KB (int8)")
        
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
        
        # Evaluierung
        eval_metrics = evaluate_model(trained_model, val_loader, config, class_names)
        
        # Inferenzzeit messen
        timing = measure_inference_time(
            trained_model, 
            (3, 48, 48),
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
            'training_time_s': training_time
        }
        
        # In CSV schreiben
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
        
        # Zum Array hinzufügen
        results.append(result)
        
        # Modell speichern
        if args.save_models:
            model_path = os.path.join(args.output_dir, 'models', f"{model_name.replace(' ', '_').lower()}.pth")
            torch.save(trained_model.state_dict(), model_path)
            logger.info(f"Modell gespeichert unter: {model_path}")
        
        # Konfusionsmatrix-Visualisierung
        if args.visualize:
            cm = np.array(eval_metrics['confusion_matrix'])
            if cm.shape[0] > 1:  # Nur anzeigen, wenn mehr als eine Klasse
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Konfusionsmatrix: {model_name}')
                plt.ylabel('Tatsächlich')
                plt.xlabel('Vorhergesagt')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name.replace(' ', '_').lower()}_confusion.png"))
                plt.close()
                
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
            plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name.replace(' ', '_').lower()}_training.png"))
            plt.close()
    
    # Vergleichstabelle als DataFrame
    df_results = pd.DataFrame(results)
    
    # Nach Genauigkeit sortieren
    df_sorted = df_results.sort_values('accuracy', ascending=False)
    
    # Ergebnisse als Excel speichern
    excel_path = os.path.join(args.output_dir, 'model_comparison_results.xlsx')
    df_sorted.to_excel(excel_path, index=False)
    
    # Vergleichsvisualisierungen erstellen
    if args.visualize:
        # Bar-Charts für Vergleich
        metrics_to_plot = [
            ('accuracy', 'Genauigkeit (%)'),
            ('int8_size_kb', 'Modellgröße (KB, Int8)'),
            ('estimated_rp2040_time_ms', 'Inferenzzeit auf RP2040 (ms)'),
            ('ram_usage_kb', 'RAM-Nutzung (KB)')
        ]
        
        for metric, title in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            # Sortiere nach aktuellem Metrik für bessere Darstellung
            temp_df = df_results.sort_values(metric)
            if metric == 'accuracy':  # Für Accuracy absteigend sortieren
                temp_df = temp_df.sort_values(metric, ascending=False)
                
            sns.barplot(x='model_name', y=metric, data=temp_df)
            plt.title(title)
            plt.xlabel('Modell')
            plt.ylabel(title)
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'plots', f"comparison_{metric}.png"))
            plt.close()
        
        # Radar-Chart für Gesamtvergleich
        categories = ['Genauigkeit', 'Kompaktheit', 'Geschwindigkeit', 'RAM-Effizienz']
        
        # Für Radar-Chart normalisieren (0-1)
        radar_data = {}
        
        for _, row in df_results.iterrows():
            model_name = row['model_name']
            
            # Normalisierte Werte berechnen (höher ist besser)
            accuracy_norm = row['accuracy'] / 100.0  # Accuracy ist bereits in Prozent
            
            # Für die restlichen Metriken: niedrigere Werte sind besser, daher 1 - norm_value
            size_max = df_results['int8_size_kb'].max()
            size_norm = 1 - (row['int8_size_kb'] / size_max if size_max > 0 else 0)
            
            time_max = df_results['estimated_rp2040_time_ms'].max()
            time_norm = 1 - (row['estimated_rp2040_time_ms'] / time_max if time_max > 0 else 0)
            
            ram_max = df_results['ram_usage_kb'].max()
            ram_norm = 1 - (row['ram_usage_kb'] / ram_max if ram_max > 0 else 0)
            
            radar_data[model_name] = [accuracy_norm, size_norm, time_norm, ram_norm]
        
        # Radar-Chart plotten
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Winkel für jede Kategorie
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Schließe den Plot
        
        # Plot für jedes Modell
        for model_name, values in radar_data.items():
            values += values[:1]  # Schließe die Werte
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Beschriftungen
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Modellvergleich: Radar-Chart")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', "model_comparison_radar.png"))
        plt.close()
            
    # Detaillierte Vergleichstabelle generieren
    table_data = []
    for _, row in df_sorted.iterrows():
        table_data.append({
            'Modell': row['model_name'],
            'Genauigkeit': f"{row['accuracy']:.2f}%",
            'F1-Score': f"{row['f1_score']:.3f}",
            'Parameter': f"{row['params_count']:,}",
            'Größe (Int8)': f"{row['int8_size_kb']:.1f} KB",
            'Inferenzzeit': f"{row['estimated_rp2040_time_ms']:.1f} ms",
            'FPS auf RP2040': f"{1000/row['estimated_rp2040_time_ms']:.1f}",
            'RAM-Nutzung': f"{row['ram_usage_kb']:.1f} KB"
        })
    
    # Erzeuge einen HTML-Vergleichsbericht
    html_path = os.path.join(args.output_dir, 'model_comparison_report.html')
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tiny CNN Modellvergleich - Pizza Erkennung</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .header { background-color: #4CAF50; color: white; padding: 10px; }
                .section { margin-top: 20px; margin-bottom: 10px; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                .highlight { background-color: #e6ffe6; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RP2040 Pizza-Erkennungssystem: Modellvergleich</h1>
                <p>Generiert am """ + datetime.now().strftime("%d.%m.%Y %H:%M") + """</p>
            </div>
            
            <h2 class="section">Modellvergleich</h2>
            <p>Dieser Bericht vergleicht verschiedene Tiny-CNN-Architekturen für das Pizza-Erkennungssystem.</p>
            
            <h3>Vergleichstabelle</h3>
            <table>
                <tr>
                    <th>Modell</th>
                    <th>Genauigkeit</th>
                    <th>F1-Score</th>
                    <th>Parameter</th>
                    <th>Größe (Int8)</th>
                    <th>Inferenzzeit</th>
                    <th>FPS auf RP2040</th>
                    <th>RAM-Nutzung</th>
                </tr>
        """)
        
        # Zeilen der Tabelle
        for i, row in enumerate(table_data):
            highlight = ' class="highlight"' if i == 0 else ''
            f.write(f"""
                <tr{highlight}>
                    <td>{row['Modell']}</td>
                    <td>{row['Genauigkeit']}</td>
                    <td>{row['F1-Score']}</td>
                    <td>{row['Parameter']}</td>
                    <td>{row['Größe (Int8)']}</td>
                    <td>{row['Inferenzzeit']}</td>
                    <td>{row['FPS auf RP2040']}</td>
                    <td>{row['RAM-Nutzung']}</td>
                </tr>
            """)
        
        f.write("""
            </table>
            
            <h3 class="section">Visualisierungen</h3>
        """)
        
        # Füge Visualisierungen hinzu
        if args.visualize:
            for metric, title in metrics_to_plot:
                plot_path = f"plots/comparison_{metric}.png"
                f.write(f"""
                    <div>
                        <h4>{title}</h4>
                        <img src="{plot_path}" alt="{title}">
                    </div>
                """)
            
            # Radar-Chart
            f.write("""
                <div>
                    <h4>Gesamtvergleich (Radar-Chart)</h4>
                    <img src="plots/model_comparison_radar.png" alt="Model Comparison Radar Chart">
                </div>
            """)
        
        # Schlussbemerkungen
        f.write("""
            <h3 class="section">Zusammenfassung</h3>
            <p>Alle drei Modelle wurden mit den gleichen Daten trainiert und auf dem gleichen Validierungssatz getestet.</p>
            <p>Die Inferenzzeiten sind Schätzungen, basierend auf CPU-Messungen und einem Faktor für die RP2040-Performance.</p>
            <p>Die RAM-Nutzung umfasst Speicher für Aktivierungen, Gewichte und Zwischenergebnisse während der Inferenz.</p>
            
            <div class="section">
                <h3>Nächste Schritte</h3>
                <ul>
                    <li>Implementierung des besten Modells auf dem RP2040</li>
                    <li>Weitere Optimierung durch Pruning oder Quantisierung</li>
                    <li>Verfeinerung des Trainings mit Fokus auf die problematischen Klassen</li>
                </ul>
            </div>
        </body>
        </html>
        """)
    
    # Zusammenfassung ausgeben
    logger.info("\n" + "="*50)
    logger.info("ZUSAMMENFASSUNG DES MODELLVERGLEICHS")
    logger.info("="*50)
    logger.info(f"Verglichene Modelle: {len(model_configs)}")
    
    # Bestes Modell nach Genauigkeit
    best_model = df_sorted.iloc[0]
    logger.info(f"\nBestes Modell nach Genauigkeit: {best_model['model_name']}")
    logger.info(f"- Genauigkeit: {best_model['accuracy']:.2f}%")
    logger.info(f"- F1-Score: {best_model['f1_score']:.3f}")
    logger.info(f"- Modellgröße (Int8): {best_model['int8_size_kb']:.1f} KB")
    logger.info(f"- Inferenzzeit auf RP2040: {best_model['estimated_rp2040_time_ms']:.1f} ms")
    logger.info(f"- RAM-Nutzung: {best_model['ram_usage_kb']:.1f} KB")
    
    # Kompaktestes Modell
    smallest_model = df_results.sort_values('int8_size_kb').iloc[0]
    logger.info(f"\nKompaktestes Modell: {smallest_model['model_name']}")
    logger.info(f"- Größe (Int8): {smallest_model['int8_size_kb']:.1f} KB")
    logger.info(f"- Genauigkeit: {smallest_model['accuracy']:.2f}%")
    
    # Schnellstes Modell
    fastest_model = df_results.sort_values('estimated_rp2040_time_ms').iloc[0]
    logger.info(f"\nSchnellstes Modell: {fastest_model['model_name']}")
    logger.info(f"- Inferenzzeit: {fastest_model['estimated_rp2040_time_ms']:.1f} ms")
    logger.info(f"- Genauigkeit: {fastest_model['accuracy']:.2f}%")
    
    # Ausgabepfade
    logger.info("\nErgebnisse wurden gespeichert in:")
    logger.info(f"- CSV: {csv_path}")
    logger.info(f"- Excel: {excel_path}")
    logger.info(f"- HTML-Bericht: {html_path}")
    if args.visualize:
        logger.info(f"- Plots: {os.path.join(args.output_dir, 'plots')}")
    if args.save_models:
        logger.info(f"- Modelle: {os.path.join(args.output_dir, 'models')}")
    
    return {
        'csv_path': csv_path,
        'excel_path': excel_path,
        'html_path': html_path,
        'results_df': df_sorted,
        'best_model': best_model['model_name']
    }


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Vergleich alternativer Tiny-CNNs für Pizza-Erkennung')
    parser.add_argument('--data-dir', default='data/augmented', help='Datensatzverzeichnis')
    parser.add_argument('--output-dir', default='output/tiny_cnn_comparison', help='Ausgabeverzeichnis')
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
    run_model_comparison(args)


if __name__ == "__main__":
    main()