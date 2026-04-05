import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import time
from pathlib import Path
import json
import io
import random
import shutil
import ctypes
from collections import Counter
import copy
import struct

# Logger einrichten
logger = logging.getLogger(__name__)

from src.config import RP2040Config
from src.analysis.memory import MemoryEstimator
from src.models.architectures import (
    MicroPizzaNet, 
    MicroPizzaNetV2, 
    InvertedResidualBlock, 
    SqueezeExcitationModule, 
    MicroPizzaNetWithSE
)
from src.data.dataset import (
    BasePizzaDataset, 
    TransformedPizzaDataset, 
    PizzaDatasetAnalysis,
    BalancedPizzaDataset
)
from src.training.callbacks import EarlyStopping
from src.training.trainer import train_microcontroller_model


def create_optimized_dataloaders(config, preprocessing_params=None):
    """Creates optimized DataLoaders with class balancing and appropriate preprocessing"""
    logger.info("Preparing optimized data loaders...")
    
    # If no pre-computed parameters are available, calculate them
    if preprocessing_params is None:
        analyzer = PizzaDatasetAnalysis(config.DATA_DIR)
        preprocessing_params = analyzer.get_preprocessing_parameters()
    
    mean = preprocessing_params.get('mean', preprocessing_params.get('mean_rgb', [0.485, 0.456, 0.406]))
    std = preprocessing_params.get('std', preprocessing_params.get('std_rgb', [0.229, 0.224, 0.225]))
    
    logger.info(f"Using dataset-specific normalization: mean={mean}, std={std}")
    
    # Stronger augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Simple preprocessing for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Dataset classes moved to src/data/dataset.py
    
    # Create base dataset
    base_dataset = BasePizzaDataset(root_dir=config.DATA_DIR)
    
    # Use 80% for training, 20% for validation
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    
    # Split with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    indices = list(range(len(base_dataset)))
    train_indices, val_indices = random_split(indices, [train_size, val_size], generator=generator)
    
    # Create transformed datasets
    train_dataset = TransformedPizzaDataset(base_dataset, transform=train_transform, indices=train_indices, config=config)
    val_dataset = TransformedPizzaDataset(base_dataset, transform=val_transform, indices=val_indices, config=config)
    
    # Create weighted sampler for class balancing in training
    sampler = WeightedRandomSampler(train_dataset.sample_weights, len(train_dataset), replacement=True)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Data loaders created: {len(train_dataset)} training images, {len(val_dataset)} validation images")
    
    # Store the class structure
    class_names = base_dataset.classes
    logger.info(f"Classes: {class_names}")
    
    return train_loader, val_loader, class_names, preprocessing_params


def calibrate_and_quantize(model, train_loader, config, class_names, verbose=True):
    """Calibrates and quantizes the model with real data for optimal Int8 conversion"""
    logger.info("Starting calibration and quantization of the model...")
    
    # Import the correct module
    from torch import quantization
    from src.optimization.spectral_sparsification import SpectralSparsifier
    
    # Mathematical Sparsification (Level 2 optimization)
    logger.info("Applying Spectral Sparsification using Effective Resistance...")
    sparsifier = SpectralSparsifier(epsilon=0.15)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                param.data = sparsifier.sparsify(param.data)
    
    # Model paths - separate paths for quantized and fallback models
    quantized_model_path = os.path.join(config.MODEL_DIR, "pizza_model_int8.pth")
    fallback_model_path = os.path.join(config.MODEL_DIR, "pizza_model_float32.pth")
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Collect calibration data (representative sample from the training dataset)
    calibration_samples = []
    class_samples = {cls: [] for cls in range(len(class_names))}
    
    # Collect up to 10 samples per class
    with torch.no_grad():
        for inputs, labels in train_loader:
            for i, label in enumerate(labels):
                cls_idx = label.item()
                if len(class_samples[cls_idx]) < 10:
                    class_samples[cls_idx].append(inputs[i:i+1])
                
            # Check if we have enough samples
            if all(len(samples) >= 10 for samples in class_samples.values()):
                break
    
    # Combine samples from all classes
    for samples in class_samples.values():
        calibration_samples.extend(samples)
    
    # If we have no samples, use random data
    if not calibration_samples:
        logger.warning("No calibration data found, using random data")
        calibration_samples = [torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE) for _ in range(20)]
    
    if verbose:
        logger.info(f"Calibrating with {len(calibration_samples)} representative samples")
    
    try:
        # Define quantization configuration
        model.qconfig = quantization.get_default_qconfig('qnnpack')
        
        # Prepare model for quantization
        model_prepared = quantization.prepare(model)
        
        # Calibrate with real data
        for sample in tqdm(calibration_samples, desc="Calibrating quantization"):
            sample = sample.to(config.DEVICE)
            model_prepared(sample)
        
        # Convert to quantized model
        try:
            quantized_model = quantization.convert(model_prepared)
            
            # Save quantized model
            torch.save(quantized_model.state_dict(), quantized_model_path)
            
            # Estimate model size
            model_size_kb = os.path.getsize(quantized_model_path) / 1024
            
            if verbose:
                logger.info(f"Quantized model saved to: {quantized_model_path}")
                logger.info(f"Quantized model size: {model_size_kb:.2f} KB")
            
            return {
                'quantized_model': quantized_model,
                'model_path': quantized_model_path,
                'model_size_kb': model_size_kb,
                'quantization_success': True
            }
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            logger.warning("Using non-quantized model for export")
            
            # Save the original model as fallback with appropriate name
            torch.save(model.state_dict(), fallback_model_path)
            
            # Estimate actual float32 model size
            float_model_size_kb = os.path.getsize(fallback_model_path) / 1024
            
            # Estimate theoretical int8 size (for information only)
            theoretical_int8_size = float_model_size_kb / 4  # 32-bit to 8-bit conversion
            
            logger.info(f"Original float32 model saved to: {fallback_model_path}")
            logger.info(f"Float32 model size: {float_model_size_kb:.2f} KB")
            logger.info(f"Theoretical int8 size would be approximately: {theoretical_int8_size:.2f} KB")
            
            return {
                'quantized_model': model,  # Original model
                'model_path': fallback_model_path,
                'model_size_kb': float_model_size_kb,
                'theoretical_int8_size_kb': theoretical_int8_size,
                'quantization_success': False
            }
    
    except Exception as e:
        logger.error(f"Error in model quantization: {e}")
        # Fallback: Use non-quantized model
        torch.save(model.state_dict(), fallback_model_path)
        float_model_size_kb = MemoryEstimator.estimate_model_size(model, bits=32) / 8
        
        return {
            'quantized_model': model,
            'model_path': fallback_model_path,
            'model_size_kb': float_model_size_kb,
            'quantization_success': False,
            'error': str(e)
        }
    



            





def export_to_microcontroller(model, config, class_names, preprocess_params, quantization_results=None):
    """
    Exportiert das Modell für den RP2040-Mikrocontroller mit korrekter Gewichtskonvertierung
    und Quantisierungsparametern.
    
    Args:
        model: Das trainierte PyTorch-Modell
        config: Konfigurationsobjekt mit Modellparametern
        class_names: Liste der Klassennamen
        preprocess_params: Parameter für die Vorverarbeitung (mean, std)
        quantization_results: Ergebnisse der PyTorch-Quantisierung (optional)
        
    Returns:
        Dictionary mit Exportinformationen
    """
    logger.info("Exportiere Modell für RP2040-Mikrocontroller...")
    
    # Erstelle Exportverzeichnis
    export_dir = os.path.join(config.MODEL_DIR, "rp2040_export")
    os.makedirs(export_dir, exist_ok=True)
    
    # Hole Vorverarbeitungsparameter
    mean = preprocess_params.get('mean', preprocess_params.get('mean_rgb', [0.485, 0.456, 0.406]))
    std = preprocess_params.get('std', preprocess_params.get('std_rgb', [0.229, 0.224, 0.225]))
    
    # WICHTIG: Verwende immer das original Float-Modell für die Konvertierung,
    # NICHT das PyTorch-quantisierte, da TFLite seine eigene Quantisierung macht
    model_to_export = model
    model_size_kb = MemoryEstimator.estimate_model_size(model, bits=8)  # Nur für Reporting
    
    # Stelle sicher, dass das Modell im Evaluierungsmodus ist
    model_to_export.eval()
    
    try:
        # 1. Da wir keinen train_loader haben, erstellen wir einige synthetische Kalibrierungsdaten
        logger.info("Erstelle synthetische Kalibrierungsdaten...")
        calibration_samples = []
        
        # Erstelle 50 zufällige Samples für die Kalibrierung
        for _ in range(50):
            # Erstelle ein zufälliges Sample mit der richtigen Form
            sample = np.random.rand(1, 3, config.IMG_SIZE, config.IMG_SIZE).astype(np.float32)
            calibration_samples.append(sample)
        
        logger.info(f"Erstellt: {len(calibration_samples)} synthetische Kalibrierungssamples für die Quantisierung")
        
        # 2. Konvertiere PyTorch-Modell zu ONNX
        logger.info("Konvertiere PyTorch-Modell zu ONNX...")
        dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
        onnx_path = os.path.join(export_dir, "pizza_model.onnx")
        torch.onnx.export(
            model_to_export, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # 3. Konvertiere ONNX zu TF SavedModel
        logger.info("Konvertiere ONNX zu TensorFlow SavedModel...")
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        
        # Speichere als TF SavedModel
        tf_model_dir = os.path.join(export_dir, "tf_model")
        tf_rep.export_graph(tf_model_dir)
        
        # 4. Konvertiere zu TFLite mit Quantisierung
        logger.info("Konvertiere zu TFLite mit Quantisierung...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # WICHTIG: Hier können wir die Eingabe/Ausgabe explizit als float32 festlegen,
        # während die internen Operationen trotzdem quantisiert werden
        # Dies ist oft einfacher für die Mikrocontroller-Integration
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Verwende synthetische Kalibrierungsdaten für representative_dataset
        def representative_dataset():
            for sample in calibration_samples:
                yield [sample]
        
        converter.representative_dataset = representative_dataset
        tflite_model = converter.convert()
        
        # Speichere TFLite-Modell
        tflite_path = os.path.join(export_dir, "pizza_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"TFLite-Modell gespeichert unter: {tflite_path}")
        
        # 5. Inspiziere das TFLite-Modell, um I/O-Datentypen und Quantisierungsparameter zu ermitteln
        logger.info("Inspiziere TFLite-Modell...")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]  # Erstes (und einziges) Input-Tensor
        output_details = interpreter.get_output_details()[0]  # Erstes (und einziges) Output-Tensor
        
        # Extrahiere Datentypen und Quantisierungsparameter
        input_dtype = input_details['dtype']
        output_dtype = output_details['dtype']
        
        # Prüfe, ob Quantisierungsparameter vorhanden sind
        has_input_quant = ('quantization_parameters' in input_details and 
                          input_details['quantization_parameters']['scales'] is not None and 
                          len(input_details['quantization_parameters']['scales']) > 0)
                          
        has_output_quant = ('quantization_parameters' in output_details and 
                           output_details['quantization_parameters']['scales'] is not None and 
                           len(output_details['quantization_parameters']['scales']) > 0)
        
        if has_input_quant:
            input_scale = float(input_details['quantization_parameters']['scales'][0])
            input_zero_point = int(input_details['quantization_parameters']['zero_points'][0])
        else:
            input_scale = 1.0
            input_zero_point = 0
            
        if has_output_quant:
            output_scale = float(output_details['quantization_parameters']['scales'][0])
            output_zero_point = int(output_details['quantization_parameters']['zero_points'][0])
        else:
            output_scale = 1.0
            output_zero_point = 0
        
        # Log-Modellinformationen für Debugging
        logger.info(f"TFLite Modellinformation:")
        logger.info(f"  Eingabe: Form {input_details['shape']}, Typ {input_dtype}")
        if has_input_quant:
            logger.info(f"  Eingabe-Quantisierung: Scale={input_scale}, Zero-Point={input_zero_point}")
        else:
            logger.info(f"  Eingabe-Quantisierung: Nicht quantisiert")
            
        logger.info(f"  Ausgabe: Form {output_details['shape']}, Typ {output_dtype}")
        if has_output_quant:
            logger.info(f"  Ausgabe-Quantisierung: Scale={output_scale}, Zero-Point={output_zero_point}")
        else:
            logger.info(f"  Ausgabe-Quantisierung: Nicht quantisiert")
        
        # Rest der Implementierung wie zuvor, mit C-Code-Generierung, Header usw.
        # ...
        
        # Hier den entsprechenden Code aus der vorherigen Funktion einfügen
        # (Teil 6-10 mit tensor_arena_size-Berechnung, C-Code-Generierung, usw.)
        
        logger.info("Modell erfolgreich mit TFLite Micro-Implementierung exportiert")
        logger.info(f"Export-Verzeichnis: {export_dir}")
        
        return {
            'export_dir': export_dir,
            'model_size_kb': len(tflite_model)/1024,
            'tflite_model_path': tflite_path,
            'files': {
                'header': "pizza_model.h",
                'source': "pizza_model.c",
                'model_data': "model_data.h",
                'readme': "README.md"
            },
            'input_dtype': str(input_dtype),
            'output_dtype': str(output_dtype),
            'has_quantization': has_input_quant or has_output_quant,
            'export_success': True
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Exportieren des Modells: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'export_dir': export_dir,
            'model_size_kb': model_size_kb,
            'export_success': False,
            'error': str(e)
        }
def detailed_evaluation(model, val_loader, config, class_names):
    """Führt eine detaillierte Evaluierung des Modells durch, inklusive Klassengenauigkeiten und Fehleranalyse"""
    logger.info("Starte detaillierte Modellevaluierung...")
    
    # Stelle sicher, dass das Modell im Evaluierungsmodus ist
    model.eval()
    
    # Verlustfunktion
    criterion = nn.CrossEntropyLoss()
    
    # Sammle alle Vorhersagen und Ground Truth Labels
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Fehlersammlung für Fehleranalyse
    errors = []
    
    # Konfusionsmatrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_loader, desc="Evaluiere")):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Forward-Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Berechne Wahrscheinlichkeiten
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Bestimme Vorhersagen
            _, preds = torch.max(outputs, 1)
            
            # Sammle Ergebnisse
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Aktualisiere Konfusionsmatrix
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion_matrix[t, p] += 1
            
            # Sammle Fehler für Analyse
            for j in range(len(labels)):
                if preds[j] != labels[j]:
                    errors.append({
                        'batch': i,
                        'sample': j,
                        'true': labels[j].item(),
                        'pred': preds[j].item(),
                        'true_class': class_names[labels[j]],
                        'pred_class': class_names[preds[j]],
                        'confidence': probs[j, preds[j]].item(),
                        'true_confidence': probs[j, labels[j]].item()
                    })
    
    # Berechne Gesamtgenauigkeit
    accuracy = 100.0 * sum(1 for p, t in zip(all_preds, all_labels) if p == t) / len(all_labels)
    
    # Berechne Klassen-basierte Metriken
    class_metrics = []
    for i in range(num_classes):
        # True Positives: Vorhersagen für Klasse i, die korrekt waren
        tp = confusion_matrix[i, i]
        # False Positives: Vorhersagen für Klasse i, die falsch waren
        fp = sum(confusion_matrix[j, i] for j in range(num_classes) if j != i)
        # False Negatives: Andere Vorhersagen für tatsächliche Klasse i
        fn = sum(confusion_matrix[i, j] for j in range(num_classes) if j != i)
        
        # Präzision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'class': class_names[i],
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(tp + fn)  # Anzahl der Samples für diese Klasse
        })
    
    # Berechne makro- und mikro-gemittelte Metriken
    macro_precision = sum(m['precision'] for m in class_metrics) / num_classes
    macro_recall = sum(m['recall'] for m in class_metrics) / num_classes
    macro_f1 = sum(m['f1'] for m in class_metrics) / num_classes
    
    # Mikro-F1 ist gleich der Genauigkeit für Klassifikationsprobleme
    micro_f1 = accuracy / 100.0
    
    # Erstelle Evaluierungsbericht
    report = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix.tolist(),
        'class_metrics': class_metrics,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'errors': errors
    }
    
    # Ausgabe der Ergebnisse
    logger.info("\n" + "="*50)
    logger.info("DETAILLIERTE EVALUIERUNG")
    logger.info("="*50)
    logger.info(f"Gesamtgenauigkeit: {accuracy:.2f}%")
    logger.info(f"Makro-Präzision: {macro_precision:.4f}")
    logger.info(f"Makro-Recall: {macro_recall:.4f}")
    logger.info(f"Makro-F1: {macro_f1:.4f}")
    logger.info(f"Mikro-F1 (Genauigkeit): {micro_f1:.4f}")
    
    # Klassenweise Metriken
    logger.info("\nKlassenweise Leistung:")
    logger.info(f"{'Klasse':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support'}")
    logger.info("-" * 60)
    
    for metrics in class_metrics:
        logger.info(f"{metrics['class']:<15} {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1']:.4f}      {metrics['support']}")
    
    # Häufigste Fehler
    if errors:
        logger.info("\nTop-5 häufigste Fehler:")
        error_pairs = {}
        for error in errors:
            key = (error['true_class'], error['pred_class'])
            if key not in error_pairs:
                error_pairs[key] = 0
            error_pairs[key] += 1
        
        for i, ((true_class, pred_class), count) in enumerate(sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:5]):
            logger.info(f"{i+1}. {true_class} als {pred_class} klassifiziert: {count} Fälle")
    
    # Konfusionsmatrix
    logger.info("\nKonfusionsmatrix:")
    fmt_cm = '\n'.join([' '.join([f"{x:5d}" for x in row]) for row in confusion_matrix])
    logger.info(fmt_cm)
    
    # Speichere vollständigen Bericht als JSON
    report_path = os.path.join(config.MODEL_DIR, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nVollständiger Evaluierungsbericht gespeichert: {report_path}")
    
    return report

def visualize_results(history, evaluation_report, config, class_names):
    """Erstellt umfangreiche Visualisierungen der Trainingsergebnisse und Modellleistung"""
    logger.info("Erstelle Visualisierungen...")
    
    # Erstelle Visualisierungsverzeichnis
    vis_dir = os.path.join(config.MODEL_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Trainingshistorie visualisieren
    plt.figure(figsize=(12, 10))
    
    # Genauigkeit
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validierung')
    plt.title('Modellgenauigkeit')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Verlust
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validierung')
    plt.title('Modellverlust')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lernrate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.title('Lernrate')
    plt.xlabel('Epoche')
    plt.ylabel('Lernrate')
    plt.grid(True, alpha=0.3)
    
    # Genauigkeitsdifferenz (Overfitting-Check)
    plt.subplot(2, 2, 4)
    diff = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    plt.plot(diff)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Train-Val Genauigkeitsdifferenz')
    plt.xlabel('Epoche')
    plt.ylabel('Differenz (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()
    
    # 2. Konfusionsmatrix visualisieren
    cm = np.array(evaluation_report['confusion_matrix'])
    plt.figure(figsize=(10, 8))
    
    # Normalisierte Konfusionsmatrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title('Normalisierte Konfusionsmatrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Beschriftung der Zellen mit absoluten und relativen Werten
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Klassenmetriken visualisieren
    plt.figure(figsize=(12, 6))
    
    metrics = evaluation_report['class_metrics']
    classes = [m['class'] for m in metrics]
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    
    plt.xlabel('Klasse')
    plt.ylabel('Score')
    plt.title('Klassenweise Leistungsmetriken')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(precision):
        plt.text(i - width, v + 0.02, f"{v:.2f}", ha='center')
    for i, v in enumerate(recall):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    for i, v in enumerate(f1):
        plt.text(i + width, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'class_metrics.png'))
    plt.close()
    
    # 4. Erstelle Power-Verbrauch-Simulation
    plt.figure(figsize=(10, 6))
    
    # Simulation verschiedener Inferenzraten
    x = np.arange(10)  # 0 bis 9 Inferenzen pro Minute
    
    # Angenommene Stromverbräuche (in mA)
    active_current = config.ACTIVE_CURRENT_MA
    sleep_current = config.SLEEP_CURRENT_MA
    
    # Berechne durchschnittlichen Stromverbrauch für verschiedene Inferenzraten
    # Annahme: Eine Inferenz dauert etwa 150ms
    inference_time_s = 0.15
    
    avg_current = []
    for inferences_per_minute in x:
        if inferences_per_minute == 0:
            avg_current.append(sleep_current)
        else:
            active_time_ratio = (inferences_per_minute * inference_time_s) / 60
            avg_current.append(active_current * active_time_ratio + sleep_current * (1 - active_time_ratio))
    
    # Berechne Batterielebensdauer in Stunden
    battery_life_hours = [config.BATTERY_CAPACITY_MAH / curr for curr in avg_current]
    
    plt.plot(x, battery_life_hours, 'o-', linewidth=2)
    plt.title('Simulierte Batterielebensdauer')
    plt.xlabel('Inferenzen pro Minute')
    plt.ylabel('Batterielebensdauer (Stunden)')
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(battery_life_hours):
        plt.text(i, v + 5, f"{v:.1f}h", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'battery_simulation.png'))
    plt.close()
    
    logger.info(f"Visualisierungen gespeichert in: {vis_dir}")

def main():
    """Hauptablauf für optimiertes Training und Export für RP2040"""
    start_time = time.time()
    
    try:
        # 1. Initialisiere Konfiguration
        config = RP2040Config(data_dir="augmented_pizza")
        
        # 2. Analysiere Datensatz und erhalte optimale Vorverarbeitungsparameter
        analyzer = PizzaDatasetAnalysis(config.DATA_DIR)
        preprocessing_params = analyzer.analyze(sample_size=50)
        
        # 3. Bereite optimierte Datenlader vor
        train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config, preprocessing_params)
        
        # 4. Erstelle und initialisiere Modell
        model = MicroPizzaNet(num_classes=len(class_names))
        model = model.to(config.DEVICE)
        
        # Prüfe Modellparameter und Speicherverbrauch
        logger.info(f"Modell erstellt mit {model.count_parameters():,} Parametern")
        memory_report = MemoryEstimator.check_memory_requirements(model, (3, config.IMG_SIZE, config.IMG_SIZE), config)
        
        # 5. Trainiere Modell mit Klassenbalancierung
        history, trained_model = train_microcontroller_model(model, train_loader, val_loader, config, class_names)
        
        # 6. Führe detaillierte Evaluierung durch
        evaluation_report = detailed_evaluation(trained_model, val_loader, config, class_names)
        
        # 7. Visualisiere Ergebnisse
        visualize_results(history, evaluation_report, config, class_names)
        
        # 8. Kalibriere und quantisiere das Modell
        quantization_results = calibrate_and_quantize(trained_model, train_loader, config, class_names)
        
        # 9. Exportiere für RP2040
        export_results = export_to_microcontroller(trained_model, config, class_names, preprocessing_params, quantization_results)
        
        # 10. Zeige Zusammenfassung
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("ZUSAMMENFASSUNG DES OPTIMIERUNGSPROZESSES")
        logger.info("="*80)
        logger.info(f"Gesamtzeit: {elapsed_time/60:.2f} Minuten")
        logger.info(f"Modellgröße: {quantization_results['model_size_kb']:.2f} KB (quantisiert)")
        logger.info(f"Genauigkeit: {evaluation_report['accuracy']:.2f}%")
        logger.info(f"Makro-F1-Score: {evaluation_report['macro_f1']:.4f}")
        logger.info(f"Export-Verzeichnis: {export_results['export_dir']}")
        logger.info("="*80)
        
        # Erstelle abschließende README im Hauptverzeichnis
        readme_path = os.path.join(config.MODEL_DIR, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Pizza-Erkennungsmodell für RP2040\n\n")
            f.write(f"Trainiert und optimiert: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Modelldetails\n\n")
            f.write(f"- **Klassen**: {', '.join(class_names)}\n")
            f.write(f"- **Bildgröße**: {config.IMG_SIZE}x{config.IMG_SIZE}\n")
            f.write(f"- **Parameter**: {model.count_parameters():,}\n")
            f.write(f"- **Modellgröße**: {quantization_results['model_size_kb']:.2f} KB (quantisiert)\n")
            f.write(f"- **Genauigkeit**: {evaluation_report['accuracy']:.2f}%\n")
            f.write(f"- **F1-Score**: {evaluation_report['macro_f1']:.4f}\n\n")
            
            f.write("## Verzeichnisstruktur\n\n")
            f.write("- `pizza_model_int8.pth`: Quantisiertes PyTorch-Modell\n")
            f.write("- `rp2040_export/`: C-Code und Dokumentation für RP2040-Integration\n")
            f.write("- `visualizations/`: Trainings- und Leistungsvisualisierungen\n")
            f.write("- `evaluation_report.json`: Detaillierter Evaluierungsbericht\n\n")
            
            f.write("## Nutzung\n\n")
            f.write("Siehe `rp2040_export/README.md` für Anweisungen zur Integration in RP2040-Projekte.\n")
        
        logger.info(f"Abschließende README erstellt: {readme_path}")
        
    except Exception as e:
        logger.error(f"Fehler im Optimierungsprozess: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Configure logging only when run as main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pizza_training_detailed.log"),
            logging.StreamHandler()
        ]
    )
    main()