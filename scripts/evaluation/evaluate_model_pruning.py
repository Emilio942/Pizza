#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strukturbasiertes Pruning-Evaluierungsskript

Dieses Skript führt strukturbasiertes Pruning mit verschiedenen Sparsity-Raten durch 
und evaluiert die resultierenden Modelle auf Genauigkeit, Modellgröße, RAM-Bedarf 
und Inferenzzeit.

Verwendung:
    python evaluate_model_pruning.py [--sparsities 0.1 0.2 0.3] [--fine-tune] [--output-dir OUTPUT_DIR]

Optionen:
    --sparsities: Liste von Sparsity-Raten (z.B. 0.1 0.2 0.3)
    --fine-tune: Modelle nach dem Pruning feintunen
    --output-dir: Ausgabeverzeichnis für Berichte
"""

import os
import sys
import argparse
import json
import time
import logging
import subprocess
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importiere benötigte Module aus dem Projekt
from scripts.pruning_tool import parse_arguments, get_model, create_dataloaders
from scripts.pruning_tool import get_filter_importance, create_pruned_model, quantize_model, save_pruned_model

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'pruning_evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pruning_evaluation')

def run_pruning(sparsity, fine_tune=False, fine_tune_epochs=5, model_path=None, output_dir="models_optimized"):
    """
    Führt strukturbasiertes Pruning für eine bestimmte Sparsity-Rate durch
    
    Args:
        sparsity: Anteil der zu entfernenden Filter (0.0-1.0)
        fine_tune: Ob das Modell nach dem Pruning feingetuned werden soll
        fine_tune_epochs: Anzahl der Epochen für Finetuning
        model_path: Pfad zum vortrainierten Modell
        output_dir: Ausgabeverzeichnis für geprunte Modelle
        
    Returns:
        Pfad zum geprunten Modell
    """
    logger.info(f"Starte Pruning mit Sparsity {sparsity:.2f}")
    
    # Setze Argumente für pruning_tool.py
    sys.argv = [
        "pruning_tool.py",
        f"--sparsity={sparsity}",
        f"--output_dir={output_dir}"
    ]
    
    if model_path:
        sys.argv.append(f"--model_path={model_path}")
        
    if fine_tune:
        sys.argv.append("--fine_tune")
        sys.argv.append(f"--fine_tune_epochs={fine_tune_epochs}")
    
    # Parse Argumente und erstelle Konfiguration
    args = parse_arguments()
    
    # Lade oder trainiere das Modell
    model = get_model()
    logger.info(f"Modell geladen. Parameter: {model.count_parameters():,}")
    
    # Erstelle DataLoader für Training/Validierung
    train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)
    
    # Berechne Filter-Wichtigkeit
    importance_dict = get_filter_importance(model)
    
    # Erstelle gepruntes Modell
    pruned_model = create_pruned_model(model, importance_dict, sparsity)
    logger.info(f"Gepruntes Modell erstellt. Parameter: {pruned_model.count_parameters():,}")
    
    # Optional: Feintuning
    if fine_tune:
        logger.info(f"Starte Feintuning für {fine_tune_epochs} Epochen")
        # Setup für Feintuning
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.0005)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pruned_model.to(device)
        
        # Feintuning-Schleife
        for epoch in range(fine_tune_epochs):
            pruned_model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            logger.info(f'Feintuning Epoche {epoch+1}/{fine_tune_epochs}, Verlust: {epoch_loss:.4f}')
            
            # Evaluiere nach jeder Epoche
            pruned_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = pruned_model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            logger.info(f'Validierungsgenauigkeit: {accuracy:.2f}%')
    
    # Quantisiere das Modell
    quantized_model = quantize_model(pruned_model, train_loader)
    
    # Speichere Modell und Bericht
    model_type = "pruned_finetuned" if fine_tune else "pruned"
    quantized = True
    sparsity_str = int(sparsity * 100)
    model_name = f"micropizzanetv2_{model_type}_s{sparsity_str}"
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pth")
    
    # Speichere das Modell
    torch.save(quantized_model.state_dict(), model_path)
    logger.info(f"Quantisiertes, gepruntes Modell gespeichert: {model_path}")
    
    return model_path

def measure_tensor_arena(model_path):
    """
    Misst den Tensor-Arena-Speicherbedarf für ein Modell
    
    Args:
        model_path: Pfad zum Modell
        
    Returns:
        dict: Dictionary mit Messergebnissen
    """
    logger.info(f"Messe Tensor-Arena-Speicherbedarf für {model_path}")
    
    try:
        # Führe measure_tensor_arena.py aus
        cmd = [sys.executable, os.path.join(project_root, "scripts/measure_tensor_arena.py"), 
               "--model", model_path, "--verbose"]
        
        # Führe den Befehl aus und erfasse die Ausgabe
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Fehler bei der Messung des Tensor-Arena-Speicherbedarfs: {result.stderr}")
            return None
        
        # Parse die Ausgabe, um die Ergebnisse zu extrahieren
        # Suche nach den JSON-Daten in der Ausgabe
        output_lines = result.stdout.split('\n')
        for i, line in enumerate(output_lines):
            if "Tensor-Arena-Größe:" in line:
                # Extrahiere die Zahl aus dieser Zeile
                arena_size_kb = float(line.split(':')[1].split()[0])
                break
        else:
            # Wenn wir keine Zeile mit der Tensor-Arena-Größe finden, versuchen wir,
            # den Wert aus der Standardausgabe zu extrahieren
            arena_size_kb = None
            
        # Versuche, die Ergebnisdatei zu lesen
        try:
            report_path = os.path.join(project_root, "output/ram_analysis/tensor_arena_report.json")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    tensor_report = json.load(f)
                    arena_size_kb = tensor_report.get('tensor_arena_size_bytes', 0) / 1024
            else:
                logger.warning(f"Tensor-Arena-Bericht nicht gefunden: {report_path}")
        except Exception as e:
            logger.error(f"Fehler beim Lesen des Tensor-Arena-Berichts: {e}")
            
        return {
            'tensor_arena_kb': arena_size_kb
        }
        
    except Exception as e:
        logger.error(f"Fehler bei der Messung des Tensor-Arena-Speicherbedarfs: {e}")
        return None

def evaluate_accuracy(model_path):
    """
    Evaluiert die Genauigkeit eines Modells mit dem Testdatensatz
    
    Args:
        model_path: Pfad zum Modell
        
    Returns:
        dict: Dictionary mit Evaluierungsergebnissen
    """
    logger.info(f"Evaluiere Genauigkeit für {model_path}")
    
    try:
        # Führe run_pizza_tests.py aus
        cmd = [sys.executable, os.path.join(project_root, "scripts/run_pizza_tests.py"), 
               "--model", model_path, "--quiet"]
        
        # Führe den Befehl aus und erfasse die Ausgabe
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Fehler bei der Genauigkeitsevaluierung: {result.stderr}")
            return None
        
        # Versuche, die Genauigkeit aus der Ausgabe zu extrahieren
        accuracy = None
        for line in result.stdout.split('\n'):
            if "Accuracy:" in line:
                accuracy = float(line.split(':')[1].strip().rstrip('%'))
                break
                
        # Wenn wir die Genauigkeit nicht aus der Ausgabe extrahieren konnten,
        # versuchen wir, sie aus der Ergebnisdatei zu lesen
        if accuracy is None:
            try:
                # Suche nach der neuesten Evaluierungsdatei
                eval_dir = os.path.join(project_root, "output/test_results")
                if os.path.exists(eval_dir):
                    eval_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) 
                                 if f.endswith('.json') and 'evaluation' in f]
                    if eval_files:
                        # Sortiere nach Änderungsdatum (neueste zuerst)
                        eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        with open(eval_files[0], 'r') as f:
                            eval_data = json.load(f)
                            accuracy = eval_data.get('accuracy', None) * 100  # Als Prozent
            except Exception as e:
                logger.error(f"Fehler beim Lesen der Evaluierungsdatei: {e}")
        
        return {
            'accuracy': accuracy
        }
        
    except Exception as e:
        logger.error(f"Fehler bei der Genauigkeitsevaluierung: {e}")
        return None

def measure_inference_time(model_path):
    """
    Misst die Inferenzzeit eines Modells
    
    Args:
        model_path: Pfad zum Modell
        
    Returns:
        dict: Dictionary mit Messergebnissen
    """
    logger.info(f"Messe Inferenzzeit für {model_path}")
    
    try:
        # Erstelle und führe ein einfaches Skript zur Zeitmessung aus
        # Wir messen die Zeit für 100 Inferenzen und berechnen den Durchschnitt
        
        # Lade das Modell
        model = torch.load(model_path) if model_path.endswith('.pth') else torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Erstelle Dummy-Input
        dummy_input = torch.randn(1, 3, 48, 48)  # Annahme: 48x48 Bilder, RGB
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Messe die Zeit
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        end_time = time.time()
        
        avg_inference_time_ms = (end_time - start_time) * 1000 / num_runs
        
        return {
            'inference_time_ms': avg_inference_time_ms
        }
        
    except Exception as e:
        logger.error(f"Fehler bei der Inferenzzeitmessung: {e}")
        return None

def create_evaluation_report(evaluation_results, output_dir="output/model_optimization"):
    """
    Erstellt einen Bericht über die Evaluierungsergebnisse
    
    Args:
        evaluation_results: Dictionary mit Evaluierungsergebnissen
        output_dir: Ausgabeverzeichnis für den Bericht
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "pruning_evaluation.json")
    
    # Speichere den Bericht als JSON
    with open(report_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Evaluierungsbericht gespeichert: {report_path}")
    
    # Erstelle Visualisierungen
    create_visualization(evaluation_results, output_dir)
    
    return report_path

def create_visualization(evaluation_results, output_dir):
    """
    Erstellt Visualisierungen der Evaluierungsergebnisse
    
    Args:
        evaluation_results: Dictionary mit Evaluierungsergebnissen
        output_dir: Ausgabeverzeichnis für die Visualisierungen
    """
    # Extrahiere die Daten
    models = list(evaluation_results.keys())
    sparsities = [evaluation_results[model].get('sparsity', 0.0) for model in models]
    accuracies = [evaluation_results[model].get('accuracy', 0.0) for model in models]
    model_sizes = [evaluation_results[model].get('model_size_kb', 0.0) for model in models]
    tensor_arenas = [evaluation_results[model].get('tensor_arena_kb', 0.0) for model in models]
    inference_times = [evaluation_results[model].get('inference_time_ms', 0.0) for model in models]
    
    # Sortiere die Daten nach Sparsity
    sorted_indices = sorted(range(len(sparsities)), key=lambda i: sparsities[i])
    sparsities = [sparsities[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    model_sizes = [model_sizes[i] for i in sorted_indices]
    tensor_arenas = [tensor_arenas[i] for i in sorted_indices]
    inference_times = [inference_times[i] for i in sorted_indices]
    
    # Berechne die x-Achsenbeschriftungen (Sparsity als Prozent)
    x_labels = [f"{int(s * 100)}%" for s in sparsities]
    
    # Erstelle eine Abbildung mit 2x2 Subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Genauigkeit vs. Sparsity
    axs[0, 0].plot(x_labels, accuracies, 'o-', color='blue')
    axs[0, 0].set_title('Genauigkeit vs. Sparsity')
    axs[0, 0].set_xlabel('Sparsity')
    axs[0, 0].set_ylabel('Genauigkeit (%)')
    axs[0, 0].grid(True)
    
    # Subplot 2: Modellgröße vs. Sparsity
    axs[0, 1].plot(x_labels, model_sizes, 'o-', color='green')
    axs[0, 1].set_title('Modellgröße vs. Sparsity')
    axs[0, 1].set_xlabel('Sparsity')
    axs[0, 1].set_ylabel('Modellgröße (KB)')
    axs[0, 1].grid(True)
    
    # Subplot 3: Tensor-Arena-Größe vs. Sparsity
    axs[1, 0].plot(x_labels, tensor_arenas, 'o-', color='red')
    axs[1, 0].set_title('Tensor-Arena-Größe vs. Sparsity')
    axs[1, 0].set_xlabel('Sparsity')
    axs[1, 0].set_ylabel('Tensor-Arena-Größe (KB)')
    axs[1, 0].grid(True)
    
    # Subplot 4: Inferenzzeit vs. Sparsity
    axs[1, 1].plot(x_labels, inference_times, 'o-', color='purple')
    axs[1, 1].set_title('Inferenzzeit vs. Sparsity')
    axs[1, 1].set_xlabel('Sparsity')
    axs[1, 1].set_ylabel('Inferenzzeit (ms)')
    axs[1, 1].grid(True)
    
    # Layout anpassen
    plt.tight_layout()
    
    # Speichere die Abbildung
    plot_path = os.path.join(output_dir, "pruning_evaluation_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Visualisierung gespeichert: {plot_path}")
    
    # Schließe die Abbildung
    plt.close(fig)

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Strukturbasiertes Pruning-Evaluierungsskript")
    parser.add_argument("--sparsities", type=float, nargs='+', default=[0.1, 0.2, 0.3],
                        help="Liste von Sparsity-Raten (z.B. 0.1 0.2 0.3)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Modelle nach dem Pruning feintunen")
    parser.add_argument("--output-dir", type=str, default="output/model_optimization",
                        help="Ausgabeverzeichnis für Berichte")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Pfad zum Basismodell")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Auch das Basismodell evaluieren (Sparsity 0.0)")
    
    args = parser.parse_args()
    
    # Füge das Basismodell hinzu, wenn gewünscht
    sparsities = args.sparsities
    if args.include_baseline:
        sparsities = [0.0] + sparsities
    
    logger.info(f"Starte Evaluierung mit Sparsity-Raten: {sparsities}")
    logger.info(f"Feintuning: {args.fine_tune}")
    
    # Erstelle das Ausgabeverzeichnis
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Führe Pruning und Evaluierung für jede Sparsity-Rate durch
    evaluation_results = {}
    
    for sparsity in sparsities:
        if sparsity == 0.0:
            # Verwende das Basismodell
            model_path = args.model_path
            if model_path is None:
                # Wenn kein Basismodell angegeben wurde, trainiere eines
                logger.info("Kein Basismodell angegeben. Verwende ein neu trainiertes Modell.")
                model = get_model()
                model_path = os.path.join("models", "micropizzanetv2_base.pth")
                torch.save(model.state_dict(), model_path)
            model_name = os.path.basename(model_path)
        else:
            # Führe Pruning durch
            model_path = run_pruning(
                sparsity=sparsity, 
                fine_tune=args.fine_tune,
                model_path=args.model_path,
                output_dir="models_optimized"
            )
            model_name = os.path.basename(model_path)
        
        # Messe Modellgröße
        model_size_kb = os.path.getsize(model_path) / 1024
        
        # Messe Tensor-Arena-Speicherbedarf
        tensor_arena_result = measure_tensor_arena(model_path)
        tensor_arena_kb = tensor_arena_result.get('tensor_arena_kb') if tensor_arena_result else None
        
        # Evaluiere Genauigkeit
        accuracy_result = evaluate_accuracy(model_path)
        accuracy = accuracy_result.get('accuracy') if accuracy_result else None
        
        # Messe Inferenzzeit
        inference_time_result = measure_inference_time(model_path)
        inference_time_ms = inference_time_result.get('inference_time_ms') if inference_time_result else None
        
        # Speichere die Ergebnisse
        evaluation_results[model_name] = {
            'sparsity': sparsity,
            'fine_tuned': args.fine_tune,
            'model_path': model_path,
            'model_size_kb': model_size_kb,
            'tensor_arena_kb': tensor_arena_kb,
            'accuracy': accuracy,
            'inference_time_ms': inference_time_ms
        }
        
        logger.info(f"Evaluierung für Sparsity {sparsity:.2f} abgeschlossen:")
        logger.info(f"  Modellgröße: {model_size_kb:.2f} KB")
        logger.info(f"  Tensor-Arena: {tensor_arena_kb:.2f} KB" if tensor_arena_kb else "  Tensor-Arena: N/A")
        logger.info(f"  Genauigkeit: {accuracy:.2f}%" if accuracy else "  Genauigkeit: N/A")
        logger.info(f"  Inferenzzeit: {inference_time_ms:.2f} ms" if inference_time_ms else "  Inferenzzeit: N/A")
    
    # Erstelle einen Bericht
    report_path = create_evaluation_report(evaluation_results, args.output_dir)
    
    logger.info(f"Evaluierung abgeschlossen. Bericht: {report_path}")

if __name__ == "__main__":
    main()
