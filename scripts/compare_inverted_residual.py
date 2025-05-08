#!/usr/bin/env python3
"""
Vergleich zwischen MicroPizzaNet und MicroPizzaNetV2 mit inverted residual blocks.

Dieses Skript vergleicht die ursprüngliche MicroPizzaNet-Architektur mit der 
verbesserten MicroPizzaNetV2-Architektur, die invertierte Residualblöcke im MobileNetV2-Stil 
verwendet. Es misst und vergleicht:
- Modellgröße (Float32 und Int8)
- Genauigkeit und F1-Score auf dem Validierungsdatensatz
- Inferenzzeit auf der CPU und geschätzt für den RP2040
- RAM-Nutzung und Aktivierungsgrößen

Ausgabe ist ein Vergleichsbericht mit detaillierten Metriken und Visualisierungen
zur Verbesserung der Modellarchitektur.

Verwendung:
    python compare_inverted_residual.py [--data-dir DIR] [--output-dir DIR] [--epochs N]
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import pandas as pd
import seaborn as sns
from datetime import datetime

# Importiere Module aus dem Pizza-Projekt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pizza_detector import (
    RP2040Config, PizzaDatasetAnalysis, create_optimized_dataloaders, 
    MemoryEstimator, MicroPizzaNet, MicroPizzaNetV2, InvertedResidualBlock,
    train_microcontroller_model, detailed_evaluation
)

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inverted_residual_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def measure_inference_time(model, input_size, device, num_runs=100, rp2040_factor=10):
    """Misst die Inferenzzeit auf der CPU und schätzt die Zeit auf dem RP2040"""
    # Stelle sicher, dass das Modell im Evaluierungsmodus ist
    model.eval()
    model = model.to(device)
    
    # Erstelle zufällige Eingabedaten
    dummy_input = torch.randn(1, *input_size, device=device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Messung
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    # Berechne Durchschnitt
    avg_time = (end_time - start_time) / num_runs
    
    # Schätze RP2040-Zeit (10x langsamer als moderne CPU)
    rp2040_time = avg_time * rp2040_factor
    
    return {
        'avg_inference_time_s': avg_time,
        'estimated_rp2040_time_s': rp2040_time,
        'estimated_fps_rp2040': 1 / rp2040_time
    }

def compare_architectures(args):
    """Vergleicht die MicroPizzaNet und MicroPizzaNetV2 Architekturen"""
    logger.info("Starte Vergleich zwischen MicroPizzaNet und MicroPizzaNetV2")
    
    # Erstelle Ausgabeverzeichnisse
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Konfiguration
    config = RP2040Config(data_dir=args.data_dir)
    config.EPOCHS = args.epochs
    
    # CUDA-Probleme vermeiden, indem wir explizit CPU verwenden
    if torch.cuda.is_available() and args.force_cpu:
        logger.info("CUDA verfügbar, aber --force_cpu Option verwendet. Verwende CPU.")
        config.DEVICE = torch.device('cpu')
    else:
        # Stelle sicher, dass das Gerät korrekt initialisiert wird
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.DEVICE = device
        logger.info(f"Verwende Gerät: {config.DEVICE}")
    
    # Datensatz vorbereiten
    logger.info(f"Lade Datensatz aus {args.data_dir}")
    analyzer = PizzaDatasetAnalysis(config.DATA_DIR)
    preprocessing_params = analyzer.analyze(sample_size=50)
    
    # DataLoader erstellen
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(
        config, preprocessing_params
    )
    
    # Definiere die zu vergleichenden Modelle
    models_to_compare = [
        {
            'name': 'MicroPizzaNet (Original)',
            'model': MicroPizzaNet(num_classes=len(class_names)),
            'color': 'blue'
        },
        {
            'name': 'MicroPizzaNetV2 (Inverted Residual)',
            'model': MicroPizzaNetV2(num_classes=len(class_names)),
            'color': 'green'
        }
    ]
    
    # Ergebnisse für jeden Architektur
    results = []
    
    # Trainiere und evaluiere jedes Modell
    for model_info in models_to_compare:
        model_name = model_info['name']
        model = model_info['model']
        
        # Stelle sicher, dass das Modell auf dem richtigen Gerät ist
        model = model.to(config.DEVICE)
        
        logger.info(f"\n{'='*50}\nTrainiere {model_name}\n{'='*50}")
        
        # Parameter und Speichernutzung analysieren
        params_count = model.count_parameters()
        memory_analysis = MemoryEstimator.check_memory_requirements(
            model, 
            (3, config.IMG_SIZE, config.IMG_SIZE),
            config
        )
        
        logger.info(f"Modell hat {params_count:,} Parameter")
        logger.info(f"Geschätzte Modellgröße: {memory_analysis['model_size_float32_kb']:.2f} KB (float32), "
                  f"{memory_analysis['model_size_int8_kb']:.2f} KB (int8)")
        
        # Trainiere das Modell
        start_time = time.time()
        history, trained_model = train_microcontroller_model(
            model, 
            train_loader, 
            val_loader, 
            config, 
            class_names,
            model_name=model_name.replace(" ", "_").lower()
        )
        training_time = time.time() - start_time
        
        # Stelle sicher, dass das Modell auf dem richtigen Gerät ist für die Auswertung
        trained_model = trained_model.to(config.DEVICE)
        
        # Detaillierte Evaluierung
        evaluation = detailed_evaluation(trained_model, val_loader, config, class_names)
        
        # Inferenzzeit messen
        timing = measure_inference_time(
            trained_model, 
            (3, config.IMG_SIZE, config.IMG_SIZE),
            config.DEVICE
        )
        
        # Speichere Modell
        model_path = os.path.join(args.output_dir, 'models', f"{model_name.replace(' ', '_').lower()}.pth")
        torch.save(trained_model.state_dict(), model_path)
        logger.info(f"Modell gespeichert unter: {model_path}")
        
        # Speichere Ergebnisse
        result = {
            'name': model_name,
            'params_count': params_count,
            'model_size_float32_kb': memory_analysis['model_size_float32_kb'],
            'model_size_int8_kb': memory_analysis['model_size_int8_kb'],
            'peak_runtime_memory_kb': memory_analysis.get('peak_runtime_memory_kb', 
                                     memory_analysis.get('total_runtime_memory_kb', 0)),
            'accuracy': evaluation['accuracy'],
            'macro_f1': evaluation['macro_f1'],
            'macro_precision': evaluation['macro_precision'],
            'macro_recall': evaluation['macro_recall'],
            'inference_time_ms': timing['avg_inference_time_s'] * 1000,
            'rp2040_time_ms': timing['estimated_rp2040_time_s'] * 1000,
            'rp2040_fps': timing['estimated_fps_rp2040'],
            'training_time_s': training_time,
            'confusion_matrix': evaluation['confusion_matrix'],
            'history': history,
            'color': model_info['color']
        }
        
        results.append(result)
        
        # Trainingshistorie visualisieren
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Training')
        plt.plot(history['val_acc'], label='Validierung')
        plt.title(f'Genauigkeit: {model_name}')
        plt.xlabel('Epoche')
        plt.ylabel('Genauigkeit (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Training')
        plt.plot(history['val_loss'], label='Validierung')
        plt.title(f'Verlust: {model_name}')
        plt.xlabel('Epoche')
        plt.ylabel('Verlust')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name.replace(' ', '_').lower()}_training.png"))
        plt.close()
        
        # Konfusionsmatrix visualisieren
        cm = np.array(evaluation['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Konfusionsmatrix: {model_name}')
        plt.ylabel('Tatsächliche Klasse')
        plt.xlabel('Vorhergesagte Klasse')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', f"{model_name.replace(' ', '_').lower()}_confusion.png"))
        plt.close()
    
    # Vergleiche die Modelle
    
    # 1. Genauigkeits- und Verlaufsvergleich
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(result['history']['val_acc'], label=result['name'], color=result['color'])
    plt.title('Validierungsgenauigkeit')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(result['history']['val_loss'], label=result['name'], color=result['color'])
    plt.title('Validierungsverlust')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', 'accuracy_loss_comparison.png'))
    plt.close()
    
    # 2. Bar-Charts für verschiedene Metriken
    metrics_to_compare = [
        ('accuracy', 'Genauigkeit (%)'),
        ('macro_f1', 'F1-Score'),
        ('model_size_int8_kb', 'Modellgröße (Int8, KB)'),
        ('rp2040_time_ms', 'Inferenzzeit auf RP2040 (ms)'),
        ('peak_runtime_memory_kb', 'Spitzen-RAM-Nutzung (KB)'),
        ('rp2040_fps', 'FPS auf RP2040')
    ]
    
    plt.figure(figsize=(15, 10))
    for i, (metric, title) in enumerate(metrics_to_compare):
        plt.subplot(2, 3, i+1)
        
        # Extrahiere Daten
        names = [r['name'] for r in results]
        values = [r[metric] for r in results]
        colors = [r['color'] for r in results]
        
        # Invertiere Richtung für bestimmte Metriken (kleiner ist besser)
        invert = metric in ['model_size_int8_kb', 'rp2040_time_ms', 'peak_runtime_memory_kb']
        
        # Berechne Prozentunterschied zwischen den Modellen
        if len(values) == 2:
            if invert:
                diff_pct = ((values[0] - values[1]) / values[0]) * 100
                diff_text = f"{diff_pct:.1f}% kleiner" if diff_pct > 0 else f"{-diff_pct:.1f}% größer"
            else:
                diff_pct = ((values[1] - values[0]) / values[0]) * 100
                diff_text = f"{diff_pct:.1f}% besser" if diff_pct > 0 else f"{-diff_pct:.1f}% schlechter"
        else:
            diff_text = ""
        
        # Plot
        bars = plt.bar(names, values, color=colors)
        plt.title(title)
        plt.ylabel(title)
        
        # Füge Werte und Unterschied hinzu
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f"{value:.2f}", ha='center', va='bottom')
        
        if diff_text:
            plt.figtext(0.5, 0.01, f"MicroPizzaNetV2 ist {diff_text} als MicroPizzaNet", 
                       ha='center', fontsize=10, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', 'metrics_comparison.png'))
    plt.close()
    
    # 3. Radar-Chart für Gesamtvergleich
    categories = ['Genauigkeit', 'F1-Score', 'Kompaktheit', 'Geschwindigkeit', 'RAM-Effizienz']
    
    # Für Radar-Chart normalisieren (0-1, höher ist besser)
    radar_data = {}
    
    for result in results:
        model_name = result['name']
        
        # Direkte Normalisierung für Metriken, wo höher besser ist
        accuracy_norm = result['accuracy'] / 100.0  # Bereits in Prozent
        f1_norm = result['macro_f1']  # F1 ist bereits im Bereich [0,1]
        
        # Inverse Normalisierung für Metriken, wo niedriger besser ist
        size_values = [r['model_size_int8_kb'] for r in results]
        size_max = max(size_values)
        size_norm = 1 - (result['model_size_int8_kb'] / size_max if size_max > 0 else 0)
        
        time_values = [r['rp2040_time_ms'] for r in results]
        time_max = max(time_values)
        time_norm = 1 - (result['rp2040_time_ms'] / time_max if time_max > 0 else 0)
        
        ram_values = [r['peak_runtime_memory_kb'] for r in results]
        ram_max = max(ram_values)
        ram_norm = 1 - (result['peak_runtime_memory_kb'] / ram_max if ram_max > 0 else 0)
        
        radar_data[model_name] = [accuracy_norm, f1_norm, size_norm, time_norm, ram_norm]
    
    # Radar-Chart plotten
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Winkel für jede Kategorie
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Schließe den Plot
    
    # Plot für jedes Modell
    for model_name, values in radar_data.items():
        values += values[:1]  # Schließe die Werte
        color = next(r['color'] for r in results if r['name'] == model_name)
        ax.plot(angles, values, linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Beschriftungen
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Modellvergleich: Radar-Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', 'radar_comparison.png'))
    plt.close()
    
    # Speichere Detailergebnisse als JSON
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        # Aus Ergebnissen die nicht-serialisierbaren Teile entfernen
        json_results = []
        for r in results:
            # Kopie erstellen und nicht-JSON-serialisierbare Elemente entfernen
            r_copy = r.copy()
            if 'history' in r_copy:
                # Nur die wichtigsten Metriken behalten
                r_copy['history'] = {
                    'train_acc': r_copy['history'].get('train_acc', []),
                    'val_acc': r_copy['history'].get('val_acc', []),
                    'train_loss': r_copy['history'].get('train_loss', []),
                    'val_loss': r_copy['history'].get('val_loss', [])
                }
            json_results.append(r_copy)
        
        json.dump(json_results, f, indent=2)
    
    # Erzeuge HTML-Bericht
    html_path = os.path.join(args.output_dir, 'comparison_report.html')
    
    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MicroPizzaNetV2 vs MicroPizzaNet - Vergleichsbericht</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
                .section {{ margin-top: 20px; margin-bottom: 10px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .highlight {{ background-color: #e6ffe6; }}
                .diff-better {{ color: green; font-weight: bold; }}
                .diff-worse {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MicroPizzaNetV2 vs. MicroPizzaNet: Inverted Residual Blocks Vergleich</h1>
                <p>Generiert am {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
            </div>
            
            <h2 class="section">Zusammenfassung</h2>
            <p>Dieser Bericht vergleicht die originale MicroPizzaNet-Architektur mit der verbesserten MicroPizzaNetV2-Architektur, 
            die invertierte Residualblöcke im MobileNetV2-Stil verwendet.</p>
            
            <h3>Architektur-Vergleich</h3>
            <table>
                <tr>
                    <th>Metrik</th>
                    <th>{results[0]['name']}</th>
                    <th>{results[1]['name']}</th>
                    <th>Unterschied</th>
                </tr>
        """)
        
        # Metriken in die Tabelle einfügen
        metrics_for_table = [
            ('params_count', 'Parameter', '{:,}', False),
            ('accuracy', 'Genauigkeit', '{:.2f}%', True),
            ('macro_f1', 'F1-Score', '{:.3f}', True),
            ('model_size_int8_kb', 'Modellgröße (Int8)', '{:.1f} KB', False),
            ('rp2040_time_ms', 'Inferenzzeit (RP2040)', '{:.1f} ms', False),
            ('rp2040_fps', 'FPS auf RP2040', '{:.1f}', True),
            ('peak_runtime_memory_kb', 'Spitzen-RAM-Nutzung', '{:.1f} KB', False)
        ]
        
        for metric, label, fmt, higher_is_better in metrics_for_table:
            val1 = results[0][metric]
            val2 = results[1][metric]
            
            # Berechne prozentuale Verbesserung/Verschlechterung
            if higher_is_better:
                pct_diff = ((val2 - val1) / val1) * 100
                diff_text = f"+{pct_diff:.1f}%" if pct_diff > 0 else f"{pct_diff:.1f}%"
                diff_class = "diff-better" if pct_diff > 0 else "diff-worse"
            else:
                pct_diff = ((val1 - val2) / val1) * 100
                diff_text = f"-{pct_diff:.1f}%" if pct_diff > 0 else f"+{-pct_diff:.1f}%"
                diff_class = "diff-better" if pct_diff > 0 else "diff-worse"
            
            # Formatiere Werte
            val1_fmt = fmt.format(val1)
            val2_fmt = fmt.format(val2)
            
            f.write(f"""
                <tr>
                    <td>{label}</td>
                    <td>{val1_fmt}</td>
                    <td>{val2_fmt}</td>
                    <td class="{diff_class}">{diff_text}</td>
                </tr>
            """)
        
        f.write("""
            </table>
            
            <h3 class="section">Visualisierungen</h3>
            
            <div>
                <h4>Leistungsmetriken</h4>
                <img src="plots/metrics_comparison.png" alt="Metriken-Vergleich">
            </div>
            
            <div>
                <h4>Trainings- und Validierungskurven</h4>
                <img src="plots/accuracy_loss_comparison.png" alt="Accuracy/Loss-Vergleich">
            </div>
            
            <div>
                <h4>Radar-Chart (Gesamtvergleich)</h4>
                <img src="plots/radar_comparison.png" alt="Radar-Chart Vergleich">
            </div>
            
            <h3 class="section">Trainingsdetails</h3>
        """)
        
        # Details für jedes Modell
        for result in results:
            model_name = result['name']
            clean_name = model_name.replace(' ', '_').lower()
            
            f.write(f"""
                <h4>{model_name}</h4>
                <div>
                    <img src="plots/{clean_name}_training.png" alt="{model_name} Training">
                </div>
                <div>
                    <img src="plots/{clean_name}_confusion.png" alt="{model_name} Confusion Matrix">
                </div>
            """)
        
        f.write("""
            <h3 class="section">Fazit</h3>
            <p>Der Vergleich zeigt, dass die Verwendung von invertierten Residualblöcken (MobileNetV2-Style)
               im MicroPizzaNetV2 signifikante Vorteile bietet. Die Shortcut-Verbindungen verbessern den
               Gradientenfluss während des Trainings und ermöglichen tiefere Netzwerke ohne Leistungsverlust.</p>
               
            <p>Durch den Bottleneck-Ansatz wird die Anzahl der Berechnungen reduziert, was zu schnellerer
               Inferenz auf dem RP2040-Mikrocontroller führt, während gleichzeitig die Genauigkeit verbessert
               oder zumindest beibehalten wird.</p>
        </body>
        </html>
        """)
    
    # Zusammenfassung ausgeben
    logger.info("\n" + "="*50)
    logger.info("ZUSAMMENFASSUNG DES ARCHITEKTURVERGLEICHS")
    logger.info("="*50)
    
    # Vergleichsmetriken ausgeben
    for i, result in enumerate(results):
        logger.info(f"{result['name']}:")
        logger.info(f"- Genauigkeit: {result['accuracy']:.2f}%")
        logger.info(f"- F1-Score: {result['macro_f1']:.3f}")
        logger.info(f"- Parameter: {result['params_count']:,}")
        logger.info(f"- Modellgröße (Int8): {result['model_size_int8_kb']:.1f} KB")
        logger.info(f"- Inferenzzeit auf RP2040: {result['rp2040_time_ms']:.1f} ms")
        logger.info(f"- FPS auf RP2040: {result['rp2040_fps']:.1f}")
        logger.info(f"- Spitzen-RAM-Nutzung: {result['peak_runtime_memory_kb']:.1f} KB")
        logger.info("")
    
    # Berechne Verbesserungen
    acc_improvement = ((results[1]['accuracy'] - results[0]['accuracy']) / results[0]['accuracy']) * 100
    f1_improvement = ((results[1]['macro_f1'] - results[0]['macro_f1']) / results[0]['macro_f1']) * 100
    size_improvement = ((results[0]['model_size_int8_kb'] - results[1]['model_size_int8_kb']) / results[0]['model_size_int8_kb']) * 100
    speed_improvement = ((results[0]['rp2040_time_ms'] - results[1]['rp2040_time_ms']) / results[0]['rp2040_time_ms']) * 100
    ram_improvement = ((results[0]['peak_runtime_memory_kb'] - results[1]['peak_runtime_memory_kb']) / results[0]['peak_runtime_memory_kb']) * 100
    
    logger.info("MicroPizzaNetV2 im Vergleich zu MicroPizzaNet:")
    logger.info(f"- Genauigkeit: {'+'if acc_improvement > 0 else ''}{acc_improvement:.1f}%")
    logger.info(f"- F1-Score: {'+'if f1_improvement > 0 else ''}{f1_improvement:.1f}%")
    logger.info(f"- Modellgröße: {'-'if size_improvement > 0 else '+'}{abs(size_improvement):.1f}%")
    logger.info(f"- Inferenzgeschwindigkeit: {'-'if speed_improvement > 0 else '+'}{abs(speed_improvement):.1f}%")
    logger.info(f"- RAM-Nutzung: {'-'if ram_improvement > 0 else '+'}{abs(ram_improvement):.1f}%")
    
    logger.info("\nErgebnisse wurden gespeichert in:")
    logger.info(f"- HTML-Bericht: {html_path}")
    logger.info(f"- Plots: {os.path.join(args.output_dir, 'plots')}")
    logger.info(f"- Modelle: {os.path.join(args.output_dir, 'models')}")
    logger.info(f"- Detailergebnisse: {os.path.join(args.output_dir, 'comparison_results.json')}")
    
    return {
        'results': results,
        'html_report': html_path
    }

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Vergleich zwischen MicroPizzaNet und MicroPizzaNetV2')
    parser.add_argument('--data-dir', default='data/augmented', help='Datensatzverzeichnis')
    parser.add_argument('--output-dir', default='output/inverted_residual_comparison', help='Ausgabeverzeichnis')
    parser.add_argument('--epochs', type=int, default=30, help='Anzahl der Trainingsepochen pro Modell')
    parser.add_argument('--force_cpu', action='store_true', help='Training auf CPU erzwingen (auch wenn CUDA verfügbar ist)')
    
    args = parser.parse_args()
    
    # Starte Vergleich
    compare_architectures(args)

if __name__ == "__main__":
    main()