#!/usr/bin/env python3
"""
MicroPizzaNet mit Early Exit Branch nach Block 3
Implementiert dynamische Inferenz für RP2040 zur Stromeinsparung

Diese Implementierung fügt einen Early-Exit-Branch nach dem zweiten Block
der MicroPizzaNet-Architektur hinzu, der bei hoher Konfidenzschwelle 
eine frühzeitige Klassifizierung ermöglicht, um Rechenzeit zu sparen.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Füge Projektwurzel zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    MicroPizzaNet, RP2040Config, MemoryEstimator,
    create_optimized_dataloaders, PizzaDatasetAnalysis
)

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("early_exit_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MicroPizzaNetWithEarlyExit(nn.Module):
    """
    Erweitertes MicroPizzaNet mit Early-Exit-Funktionalität nach Block 2.
    Kann bei hoher Konfidenzschwelle frühzeitig Vorhersagen treffen, um Rechenzeit einzusparen.
    """
    def __init__(self, num_classes=6, dropout_rate=0.2, confidence_threshold=0.8):
        super(MicroPizzaNetWithEarlyExit, self).__init__()
        
        # Speichern des Konfidenz-Schwellenwerts für Early Exit
        self.confidence_threshold = confidence_threshold
        
        # Block 1: Standard-Faltung und Pooling (3 -> 8 Kanäle)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 8x12x12
        )
        
        # Block 2: Depthwise Separable Faltung (8 -> 16 Kanäle)
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1) zur Kanalexpansion
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 16x6x6
        )
        
        # Early Exit Classifier nach Block 2
        self.early_exit_pooling = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.early_exit_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
        
        # Block 3: Zweite Depthwise Separable Faltung (16 -> 32 Kanäle)
        self.block3 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1) zur Kanalexpansion
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 32x3x3
        )
        
        # Haupt-Klassifikator (nach Block 3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        # Gewichtsinitialisierung für bessere Konvergenz
        self._initialize_weights()
    
    def forward(self, x, use_early_exit=True):
        """
        Forward-Pass mit optionalem Early Exit
        
        Args:
            x: Eingabe-Tensor
            use_early_exit: Ob Early Exit aktiviert werden soll (kann für Evaluation deaktiviert werden)
            
        Returns:
            tuple: (outputs, early_exit_used)
                - outputs: Logits der Vorhersage
                - early_exit_used: Boolean, ob Early Exit verwendet wurde
        """
        # Erste Feature-Extraktion
        x = self.block1(x)
        x = self.block2(x)
        
        # Early Exit nach Block 2, wenn aktiviert
        early_exit_used = False
        
        if use_early_exit:
            # Early Exit probieren
            early_features = self.early_exit_pooling(x)
            early_exit_output = self.early_exit_classifier(early_features)
            
            # Konfidenzen für Early Exit berechnen
            early_probs = F.softmax(early_exit_output, dim=1)
            
            # Maximale Konfidenz ermitteln
            max_confidence, _ = torch.max(early_probs, dim=1)
            
            # Wenn im Inference-Modus (keine Gradienten) und Konfidenz hoch genug
            if not self.training and torch.all(max_confidence >= self.confidence_threshold):
                early_exit_used = True
                return early_exit_output, early_exit_used
        
        # Wenn Early Exit nicht verwendet wird, normal weiter
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x, early_exit_used
    
    def _initialize_weights(self):
        """Optimierte Gewichtsinitialisierung für bessere Konvergenz"""
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
        """Zählt die trainierbaren Parameter des Modells"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_early_exit_model(model, train_loader, val_loader, config, class_names, 
                           epochs=50, early_stopping_patience=10, 
                           lambda_ee=0.3, model_name="micropizzanet_early_exit"):
    """
    Trainiert das MicroPizzaNet-Modell mit Early-Exit-Branch
    
    Args:
        model: Das zu trainierende Modell
        train_loader: DataLoader für Trainingsdaten
        val_loader: DataLoader für Validierungsdaten
        config: Konfigurationsobjekt
        class_names: Liste der Klassennamen
        epochs: Anzahl der Trainingsiterationen
        early_stopping_patience: Nach wievielen Epochen ohne Verbesserung abgebrochen wird
        lambda_ee: Gewichtung des Early-Exit-Verlusts (0-1)
        model_name: Name für das gespeicherte Modell
        
    Returns:
        tuple: (history, model)
    """
    logger.info(f"Starte Training von {model_name} mit Early Exit (λ={lambda_ee})...")
    
    # Modellpfad festlegen
    model_dir = os.path.join(project_root, "models_optimized")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    # Gerät festlegen
    device = config.DEVICE
    model = model.to(device)
    
    # Verlustfunktion
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer mit Gewichtsverfall
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    # Learning Rate Scheduler für bessere Konvergenz
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )
    
    # Early Stopping initialisieren
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    
    # Training History
    history = {
        'train_loss': [],
        'train_loss_main': [],
        'train_loss_early': [],
        'val_loss': [],
        'train_acc': [],
        'train_acc_early': [],
        'val_acc': [],
        'val_acc_early': [],
        'early_exit_rate': [],
        'lr': []
    }
    
    # Training Loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_loss_main = 0.0
        running_loss_early = 0.0
        correct_main = 0
        correct_early = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Gradienten zurücksetzen
            optimizer.zero_grad()
            
            # Forward-Pass (deaktiviere Early Exit für Training, wir wollen beide Ausgänge trainieren)
            main_output, _ = model(inputs, use_early_exit=False)
            
            # Early-Exit-Ausgabe separat berechnen (für Training)
            early_features = model.early_exit_pooling(model.block2(model.block1(inputs)))
            early_output = model.early_exit_classifier(early_features)
            
            # Verluste berechnen
            loss_main = criterion(main_output, labels)
            loss_early = criterion(early_output, labels)
            
            # Kombinierter Verlust mit Gewichtung
            loss = (1 - lambda_ee) * loss_main + lambda_ee * loss_early
            
            # Backward-Pass und Optimierung
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Statistiken aktualisieren
            running_loss += loss.item() * inputs.size(0)
            running_loss_main += loss_main.item() * inputs.size(0)
            running_loss_early += loss_early.item() * inputs.size(0)
            
            _, preds_main = torch.max(main_output, 1)
            _, preds_early = torch.max(early_output, 1)
            
            total += labels.size(0)
            correct_main += (preds_main == labels).sum().item()
            correct_early += (preds_early == labels).sum().item()
            
            # Progressbar aktualisieren
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc_main': 100 * correct_main / total,
                'acc_early': 100 * correct_early / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Trainingsmetriken berechnen
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_loss_main = running_loss_main / len(train_loader.dataset)
        epoch_train_loss_early = running_loss_early / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct_main / total
        epoch_train_acc_early = 100.0 * correct_early / total
        
        # Validierung Phase
        model.eval()
        running_val_loss = 0.0
        correct_main = 0
        correct_early = 0
        early_exit_used_count = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward-Pass mit aktiviertem Early Exit
                outputs, early_exit_used = model(inputs, use_early_exit=True)
                
                # Wenn Early Exit verwendet wurde, ist outputs bereits die Early-Exit-Ausgabe
                if early_exit_used:
                    early_exit_used_count += inputs.size(0)
                    # Verlust der Early-Exit-Ausgabe
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                else:
                    # Wir müssen den Early-Exit-Teil manuell berechnen
                    early_features = model.early_exit_pooling(model.block2(model.block1(inputs)))
                    early_output = model.early_exit_classifier(early_features)
                    
                    # Hauptverlust
                    loss_main = criterion(outputs, labels)
                    loss_early = criterion(early_output, labels)
                    loss = (1 - lambda_ee) * loss_main + lambda_ee * loss_early
                    
                    # Vorhersagen
                    _, preds_main = torch.max(outputs, 1)
                    _, preds_early = torch.max(early_output, 1)
                    
                    # Für die Statistik
                    preds = preds_main
                    correct_early += (preds_early == labels).sum().item()
                
                # Statistiken aktualisieren
                running_val_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct_main += (preds == labels).sum().item()
                
                # Progressbar aktualisieren
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * correct_main / total,
                    'early_exit_rate': 100 * early_exit_used_count / total
                })
        
        # Validierungsmetriken berechnen
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct_main / total
        epoch_val_acc_early = 100.0 * correct_early / total
        early_exit_rate = 100.0 * early_exit_used_count / total
        
        # History aktualisieren
        history['train_loss'].append(epoch_train_loss)
        history['train_loss_main'].append(epoch_train_loss_main)
        history['train_loss_early'].append(epoch_train_loss_early)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['train_acc_early'].append(epoch_train_acc_early)
        history['val_acc'].append(epoch_val_acc)
        history['val_acc_early'].append(epoch_val_acc_early)
        history['early_exit_rate'].append(early_exit_rate)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Ausgabe
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f} (Main: {epoch_train_loss_main:.4f}, Early: {epoch_train_loss_early:.4f})")
        logger.info(f"  Train Acc: {epoch_train_acc:.2f}% (Early Exit: {epoch_train_acc_early:.2f}%)")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        logger.info(f"  Early Exit Rate: {early_exit_rate:.2f}%")
        
        # Early Stopping und Modell speichern
        if epoch_val_loss < best_val_loss:
            logger.info(f"  Validierungsverlust verbessert: {best_val_loss:.4f} -> {epoch_val_loss:.4f}")
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Checkpoint speichern
            torch.save(model.state_dict(), model_path)
            logger.info(f"  Modell gespeichert: {model_path}")
        else:
            patience_counter += 1
            logger.info(f"  Keine Verbesserung. Early Stopping Counter: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early Stopping in Epoche {epoch+1}. Beste Epoche war {best_epoch+1}.")
                break
    
    # Beste Gewichte wiederherstellen
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info(f"Beste Modellgewichte aus Epoche {best_epoch+1} wiederhergestellt.")
    
    # Modell speichern (nochmal sicherstellen)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Modell gespeichert: {model_path}")
    
    return history, model


def evaluate_early_exit_model(model, val_loader, config, class_names, 
                             confidence_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):
    """
    Evaluiert das Early-Exit-Modell mit verschiedenen Konfidenz-Schwellenwerten
    und analysiert die Effizienzgewinne.
    
    Args:
        model: Das trainierte Modell
        val_loader: DataLoader für Validierungsdaten
        config: Konfigurationsobjekt
        class_names: Liste der Klassennamen
        confidence_thresholds: Liste der zu testenden Konfidenz-Schwellenwerte
        
    Returns:
        dict: Evaluierungsergebnisse
    """
    logger.info("Evaluiere Early-Exit-Modell mit verschiedenen Konfidenz-Schwellenwerten...")
    
    # Gerät
    device = config.DEVICE
    model = model.to(device)
    model.eval()
    
    # Ergebnisse für jeden Schwellenwert
    results = {}
    
    # Genauigkeit ohne Early Exit als Baseline messen
    baseline_correct = 0
    baseline_early_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Baseline (ohne Early Exit)"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward-Pass ohne Early Exit
            outputs, _ = model(inputs, use_early_exit=False)
            
            # Early-Exit-Ausgabe separat berechnen
            early_features = model.early_exit_pooling(model.block2(model.block1(inputs)))
            early_output = model.early_exit_classifier(early_features)
            
            # Vorhersagen
            _, preds_main = torch.max(outputs, 1)
            _, preds_early = torch.max(early_output, 1)
            
            # Korrekte Zählen
            total += labels.size(0)
            baseline_correct += (preds_main == labels).sum().item()
            baseline_early_correct += (preds_early == labels).sum().item()
    
    baseline_accuracy = 100.0 * baseline_correct / total
    baseline_early_accuracy = 100.0 * baseline_early_correct / total
    
    logger.info(f"Baseline Accuracy (ohne Early Exit): {baseline_accuracy:.2f}%")
    logger.info(f"Early Exit Branch Accuracy: {baseline_early_accuracy:.2f}%")
    
    # Evaluierung für jeden Schwellenwert
    for threshold in confidence_thresholds:
        # Setze den Konfidenz-Schwellenwert
        model.confidence_threshold = threshold
        
        correct = 0
        early_exit_used_count = 0
        early_exit_correct = 0
        early_exit_incorrect = 0
        total = 0
        
        main_block_time = 0
        early_exit_time = 0
        
        # Für Konfusionsmatrix
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Evaluiere (threshold={threshold:.2f})"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                batch_size = inputs.size(0)
                total += batch_size
                
                # Zeitmessung für Block 1 und 2 (immer ausgeführt)
                start_time = time.time()
                
                # Block 1 und 2 ausführen
                x = model.block1(inputs)
                x = model.block2(x)
                
                # Early-Exit-Berechnung
                early_features = model.early_exit_pooling(x)
                early_output = model.early_exit_classifier(early_features)
                early_probs = F.softmax(early_output, dim=1)
                
                # Messe Zeit für Early Exit
                early_exit_time += time.time() - start_time
                
                # Prüfe Konfidenz für Early Exit
                max_confidence, early_preds = torch.max(early_probs, dim=1)
                
                # Bereite Speicher für die Vorhersagen vor
                final_preds = torch.zeros_like(early_preds)
                
                # Für Samples, die den Early Exit nehmen
                early_exit_mask = max_confidence >= threshold
                early_exit_count = early_exit_mask.sum().item()
                early_exit_used_count += early_exit_count
                
                # Vorhersagen für Early Exit
                final_preds[early_exit_mask] = early_preds[early_exit_mask]
                
                # Für Samples, die den vollen Durchlauf benötigen
                full_inference_mask = ~early_exit_mask
                full_inference_count = full_inference_mask.sum().item()
                
                # Wenn es Samples gibt, die den vollen Durchlauf benötigen
                if full_inference_count > 0:
                    # Zeitmessung für Block 3 und Klassifikator
                    start_time = time.time()
                    
                    # Nur für die notwendigen Samples Block 3 ausführen
                    x_continue = x[full_inference_mask]
                    x_continue = model.block3(x_continue)
                    x_continue = model.global_pool(x_continue)
                    main_output_continue = model.classifier(x_continue)
                    
                    # Messe Zeit für Hauptblock
                    main_block_time += time.time() - start_time
                    
                    # Vorhersagen für den vollen Durchlauf
                    _, main_preds_continue = torch.max(main_output_continue, dim=1)
                    
                    # Ordne den entsprechenden Indizes im Gesamtbatch zu
                    final_preds[full_inference_mask] = main_preds_continue
                
                # Korrekte Vorhersagen zählen
                correct += (final_preds == labels).sum().item()
                
                # Early-Exit-Statistiken
                if early_exit_count > 0:
                    early_correct = (early_preds[early_exit_mask] == labels[early_exit_mask]).sum().item()
                    early_exit_correct += early_correct
                    early_exit_incorrect += early_exit_count - early_correct
                
                # Für Konfusionsmatrix
                all_preds.extend(final_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Berechne Konfusionsmatrix
        confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
        for t, p in zip(all_labels, all_preds):
            confusion_matrix[t, p] += 1
        
        # Berechne Metriken
        accuracy = 100.0 * correct / total
        early_exit_rate = 100.0 * early_exit_used_count / total
        early_exit_accuracy = 100.0 * early_exit_correct / early_exit_used_count if early_exit_used_count > 0 else 0
        
        # Berechne normierte Konfusionsmatrix
        confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Berechnete Zeitersparnis durch Early Exit
        # Annahme: Block 3 und Klassifikator machen etwa 50% der Inferenzzeit aus
        time_saved_percent = early_exit_rate / 100 * 50  # ~ % der Zeit gespart
        
        # Speichere Ergebnisse
        results[threshold] = {
            'accuracy': accuracy,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_diff': accuracy - baseline_accuracy,
            'early_exit_rate': early_exit_rate,
            'early_exit_accuracy': early_exit_accuracy,
            'early_exit_correct': early_exit_correct,
            'early_exit_incorrect': early_exit_incorrect,
            'time_saved_percent': time_saved_percent,
            'confusion_matrix': confusion_matrix.tolist(),
            'confusion_matrix_norm': confusion_matrix_norm.tolist()
        }
        
        logger.info(f"Schwellenwert {threshold:.2f}:")
        logger.info(f"  Accuracy: {accuracy:.2f}% (vs. Baseline: {baseline_accuracy:.2f}%, Diff: {accuracy - baseline_accuracy:.2f}%)")
        logger.info(f"  Early Exit Rate: {early_exit_rate:.2f}%")
        logger.info(f"  Early Exit Accuracy: {early_exit_accuracy:.2f}%")
        logger.info(f"  Geschätzte Zeitersparnis: ~{time_saved_percent:.1f}%")
    
    return {
        'thresholds': confidence_thresholds,
        'results': results,
        'baseline_accuracy': baseline_accuracy,
        'baseline_early_accuracy': baseline_early_accuracy
    }


def visualize_early_exit_results(history, evaluation_results, config, output_dir=None):
    """
    Visualisiert die Trainingsergebnisse und Effizienzgewinne des Early-Exit-Modells
    
    Args:
        history: Trainingshistorie
        evaluation_results: Evaluierungsergebnisse
        config: Konfigurationsobjekt
        output_dir: Ausgabeverzeichnis für die Visualisierungen
        
    Returns:
        None
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, "models_optimized", "visualizations", "early_exit")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Trainingshistorie visualisieren
    plt.figure(figsize=(15, 10))
    
    # Genauigkeiten
    plt.subplot(2, 3, 1)
    plt.plot(history['train_acc'], label='Training (Haupt)')
    plt.plot(history['train_acc_early'], label='Training (Early)')
    plt.plot(history['val_acc'], label='Validierung (Haupt)')
    plt.title('Modellgenauigkeit während Training')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Verluste
    plt.subplot(2, 3, 2)
    plt.plot(history['train_loss'], label='Gesamt')
    plt.plot(history['train_loss_main'], label='Haupt')
    plt.plot(history['train_loss_early'], label='Early')
    plt.plot(history['val_loss'], label='Validierung')
    plt.title('Verlustfunktionen während Training')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Early-Exit-Rate
    plt.subplot(2, 3, 3)
    plt.plot(history['early_exit_rate'])
    plt.title('Early-Exit-Rate während Validierung')
    plt.xlabel('Epoche')
    plt.ylabel('Early-Exit-Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # Threshold vs. Accuracy Tradeoff
    plt.subplot(2, 3, 4)
    thresholds = evaluation_results['thresholds']
    accuracies = [evaluation_results['results'][t]['accuracy'] for t in thresholds]
    exit_rates = [evaluation_results['results'][t]['early_exit_rate'] for t in thresholds]
    
    plt.plot(thresholds, accuracies, 'o-', label='Genauigkeit')
    plt.axhline(y=evaluation_results['baseline_accuracy'], color='r', linestyle='--', label='Baseline')
    plt.title('Genauigkeit vs. Konfidenz-Schwellenwert')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Threshold vs. Early Exit Rate
    plt.subplot(2, 3, 5)
    plt.plot(thresholds, exit_rates, 'o-')
    plt.title('Early-Exit-Rate vs. Konfidenz-Schwellenwert')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Early-Exit-Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # Time Savings vs. Accuracy Loss
    plt.subplot(2, 3, 6)
    time_saved = [evaluation_results['results'][t]['time_saved_percent'] for t in thresholds]
    acc_diff = [evaluation_results['results'][t]['accuracy_diff'] for t in thresholds]
    
    plt.scatter(time_saved, acc_diff, c=thresholds, cmap='viridis')
    plt.colorbar(label='Schwellenwert')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Zeitersparnis vs. Genauigkeitsverlust')
    plt.xlabel('Geschätzte Zeitersparnis (%)')
    plt.ylabel('Genauigkeitsdifferenz zur Baseline (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'early_exit_evaluation.png'), dpi=300)
    plt.close()
    
    # 2. Detaillierte Konfidenz-Analysen
    plt.figure(figsize=(12, 8))
    
    best_threshold_idx = np.argmax([evaluation_results['results'][t]['accuracy'] for t in thresholds])
    best_threshold = thresholds[best_threshold_idx]
    
    # Frühere vs. spätere Genauigkeit
    early_acc = [evaluation_results['results'][t]['early_exit_accuracy'] for t in thresholds]
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, early_acc, 'o-', label='Early Exit')
    plt.axhline(y=evaluation_results['baseline_accuracy'], color='r', linestyle='--', label='Baseline')
    plt.title('Early-Exit vs. Baseline Genauigkeit')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Korrekte vs. inkorrekte Early Exits
    correct = [evaluation_results['results'][t]['early_exit_correct'] for t in thresholds]
    incorrect = [evaluation_results['results'][t]['early_exit_incorrect'] for t in thresholds]
    
    plt.subplot(2, 2, 2)
    plt.bar(thresholds, correct, label='Korrekt', alpha=0.7)
    plt.bar(thresholds, incorrect, bottom=correct, label='Inkorrekt', alpha=0.7)
    plt.title('Korrekte vs. Inkorrekte Early Exits')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Anzahl der Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Effizienz-Frontend Visualisierung
    plt.subplot(2, 2, 3)
    
    # Stromeinsparung basierend auf Early-Exit-Rate simulieren
    # Annahme: Jedes 1% Early Exit spart 0.5% Strom (vereinfachte Annahme)
    power_saved = [rate * 0.5 for rate in exit_rates]
    
    # Batterielebensdauer mit CR123A (1500mAh)
    # Annahme: Baseline Batterielebensdauer von 9.1 Tagen (vom Projektstatus)
    baseline_battery_days = 9.1
    battery_life = [baseline_battery_days * (1 + ps/100) for ps in power_saved]
    
    plt.plot(thresholds, battery_life, 'o-')
    plt.axhline(y=baseline_battery_days, color='r', linestyle='--', label='Baseline')
    plt.title('Simulierte Batterielebensdauer')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Lebensdauer (Tage)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Effizienz-Genauigkeits-Tradeoff
    plt.subplot(2, 2, 4)
    plt.plot(exit_rates, accuracies, 'o-')
    
    # Idealer Bereich hervorheben
    best_idx = np.argmax(accuracies)
    plt.scatter([exit_rates[best_idx]], [accuracies[best_idx]], c='r', s=100, label=f'Bester Wert (t={thresholds[best_idx]:.2f})')
    
    plt.title('Genauigkeits-Effizienz Tradeoff')
    plt.xlabel('Early-Exit-Rate (%)')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'early_exit_efficiency.png'), dpi=300)
    plt.close()
    
    # 3. Erstelle eine zusammenfassende HTML-Datei
    html_path = os.path.join(output_dir, 'early_exit_report.html')
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Early Exit Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
                .section {{ margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #f9f9f9; }}
                .conclusion {{ margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-left: 5px solid #4CAF50; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Early Exit Evaluation Report</h1>
                <p>MicroPizzaNet with Dynamic Inference</p>
            </div>
            
            <div class="section">
                <h2>Überblick</h2>
                <p>Dieses Dokument analysiert die Leistung des MicroPizzaNet-Modells mit Early-Exit-Funktionalität. 
                   Early Exit ermöglicht dynamische Inferenz, bei der einfachere Beispiele früher klassifiziert werden können,
                   was Rechenleistung und Energie spart.</p>
            </div>
            
            <div class="section">
                <h2>Training und Validierung</h2>
                <img src="early_exit_evaluation.png" alt="Early Exit Evaluation">
                
                <h3>Trainingsergebnisse:</h3>
                <ul>
                    <li>Baseline Accuracy: {:.2f}%</li>
                    <li>Early Exit Branch Accuracy: {:.2f}%</li>
                    <li>Trainingszeit: {} Epochen</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Effizienzanalyse</h2>
                <img src="early_exit_efficiency.png" alt="Early Exit Efficiency">
                
                <h3>Effizienzgewinne:</h3>
                <table>
                    <tr>
                        <th>Schwellenwert</th>
                        <th>Accuracy</th>
                        <th>Diff zur Baseline</th>
                        <th>Early Exit Rate</th>
                        <th>Geschätzte Zeitersparnis</th>
                        <th>Geschätzte zusätzliche Batterielebensdauer</th>
                    </tr>
        """.format(
            evaluation_results['baseline_accuracy'],
            evaluation_results['baseline_early_accuracy'],
            len(history['train_loss'])
        ))
        
        # Tabelle mit Ergebnissen für jeden Schwellenwert
        for threshold in thresholds:
            result = evaluation_results['results'][threshold]
            exit_rate = result['early_exit_rate']
            time_saved = result['time_saved_percent']
            
            # Simulierte Batterielebensdauer
            baseline_battery_days = 9.1
            battery_life_gain = baseline_battery_days * (time_saved / 100)
            
            # Highlight den besten Schwellenwert (höchste Genauigkeit)
            row_class = " class='highlight'" if abs(result['accuracy'] - max([evaluation_results['results'][t]['accuracy'] for t in thresholds])) < 0.01 else ""
            
            f.write(f"""
                <tr{row_class}>
                    <td>{threshold:.2f}</td>
                    <td>{result['accuracy']:.2f}%</td>
                    <td>{result['accuracy_diff']:.2f}%</td>
                    <td>{exit_rate:.2f}%</td>
                    <td>{time_saved:.1f}%</td>
                    <td>+{battery_life_gain:.1f} Tage</td>
                </tr>
            """)
        
        # Beste Schwellenwerte identifizieren
        best_accuracy_idx = np.argmax([evaluation_results['results'][t]['accuracy'] for t in thresholds])
        best_accuracy_threshold = thresholds[best_accuracy_idx]
        
        # Bester Effizienz-Genauigkeits-Kompromiss
        # Maximiere: accuracy - baseline_accuracy + time_saved_percent/5
        compromise_score = [evaluation_results['results'][t]['accuracy_diff'] + evaluation_results['results'][t]['time_saved_percent']/5 for t in thresholds]
        best_compromise_idx = np.argmax(compromise_score)
        best_compromise_threshold = thresholds[best_compromise_idx]
        
        f.write("""
                </table>
            </div>
            
            <div class="conclusion">
                <h2>Schlussfolgerungen und Empfehlungen</h2>
                
                <h3>Optimale Konfiguration:</h3>
                <ul>
                    <li><strong>Höchste Genauigkeit:</strong> Schwellenwert = {:.2f} (Genauigkeit: {:.2f}%, Zeitersparnis: {:.1f}%)</li>
                    <li><strong>Bester Kompromiss:</strong> Schwellenwert = {:.2f} (Genauigkeit: {:.2f}%, Zeitersparnis: {:.1f}%)</li>
                </ul>
                
                <h3>Vorteile des Early Exits:</h3>
                <ul>
                    <li>Kann die Batterielebensdauer um bis zu {:.1f} Tage verlängern (+{:.1f}%)</li>
                    <li>Reduziert die Inferenzzeit um bis zu {:.1f}%</li>
                    <li>Verbessert die Reaktionszeit des Systems</li>
                </ul>
                
                <h3>Empfohlene Konfiguration für Deployment:</h3>
                <p>Für den RP2040-Mikrocontroller empfehlen wir einen Konfidenz-Schwellenwert von <strong>{:.2f}</strong>,
                   der eine gute Balance zwischen Genauigkeit ({:.2f}%) und Energieeffizienz (Zeitersparnis: {:.1f}%) bietet.</p>
            </div>
            
            <div class="section">
                <h2>Implementierungsdetails</h2>
                <p>Das Modell verwendet einen Early-Exit-Branch nach Block 2, der bei ausreichender Konfidenz die Berechnung 
                   von Block 3 und dem Hauptklassifikator überspringt.</p>
                
                <h3>Modellarchitektur:</h3>
                <ul>
                    <li>Block 1: Standard Convolution (3 → 8 Kanäle)</li>
                    <li>Block 2: Depthwise Separable Convolution (8 → 16 Kanäle)</li>
                    <li>Early Exit Branch: Global Average Pooling + Classifier (16 → {} Klassen)</li>
                    <li>Block 3: Depthwise Separable Convolution (16 → 32 Kanäle)</li>
                    <li>Hauptklassifikator: Global Average Pooling + Classifier (32 → {} Klassen)</li>
                </ul>
            </div>
        </body>
        </html>
        """.format(
            best_accuracy_threshold,
            evaluation_results['results'][best_accuracy_threshold]['accuracy'],
            evaluation_results['results'][best_accuracy_threshold]['time_saved_percent'],
            
            best_compromise_threshold,
            evaluation_results['results'][best_compromise_threshold]['accuracy'],
            evaluation_results['results'][best_compromise_threshold]['time_saved_percent'],
            
            # Max additional battery life
            max([baseline_battery_days * (evaluation_results['results'][t]['time_saved_percent'] / 100) for t in thresholds]),
            max([evaluation_results['results'][t]['time_saved_percent'] * baseline_battery_days / 9.1 for t in thresholds]),
            
            # Max time saved
            max([evaluation_results['results'][t]['time_saved_percent'] for t in thresholds]),
            
            # Recommended threshold (use the compromise)
            best_compromise_threshold,
            evaluation_results['results'][best_compromise_threshold]['accuracy'],
            evaluation_results['results'][best_compromise_threshold]['time_saved_percent'],
            
            # Class count (for architecture description)
            len(evaluation_results['results'][thresholds[0]]['confusion_matrix']),
            len(evaluation_results['results'][thresholds[0]]['confusion_matrix'])
        ))
    
    logger.info(f"Evaluierungsbericht erstellt: {html_path}")
    return html_path


def main(args):
    """Hauptfunktion zum Trainieren und Evaluieren des Early-Exit-Modells"""
    # Konfiguration initialisieren
    config = RP2040Config(data_dir=args.data_dir)
    
    # Datensatz analysieren
    logger.info("Analysiere Datensatz...")
    analyzer = PizzaDatasetAnalysis(config.DATA_DIR)
    preprocessing_params = analyzer.analyze(sample_size=50)
    
    # DataLoader erstellen
    logger.info("Erstelle DataLoader...")
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(
        config, preprocessing_params
    )
    
    if args.mode == 'train' or args.mode == 'both':
        # Modell erstellen
        logger.info("Erstelle Early-Exit-Modell...")
        model = MicroPizzaNetWithEarlyExit(
            num_classes=len(class_names),
            dropout_rate=0.2,
            confidence_threshold=args.confidence_threshold
        )
        
        # Parameter und Speicherverbrauch prüfen
        logger.info(f"Modell erstellt mit {model.count_parameters():,} Parametern")
        memory_report = MemoryEstimator.check_memory_requirements(
            model, (3, config.IMG_SIZE, config.IMG_SIZE), config
        )
        
        # Modell trainieren
        history, trained_model = train_early_exit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            class_names=class_names,
            epochs=args.epochs,
            early_stopping_patience=args.patience,
            lambda_ee=args.lambda_ee,
            model_name=args.model_name
        )
    
    if args.mode == 'evaluate' or args.mode == 'both':
        # Wenn nur Evaluierung, lade vortrainiertes Modell
        if args.mode == 'evaluate':
            logger.info(f"Lade vortrainiertes Modell: {args.model_path}")
            model = MicroPizzaNetWithEarlyExit(
                num_classes=len(class_names),
                confidence_threshold=args.confidence_threshold
            )
            model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
            trained_model = model
        
        # Evaluiere das Modell mit verschiedenen Schwellenwerten
        evaluation_results = evaluate_early_exit_model(
            model=trained_model,
            val_loader=val_loader,
            config=config,
            class_names=class_names,
            confidence_thresholds=args.thresholds
        )
        
        # Visualisiere die Ergebnisse
        if args.mode == 'both':
            html_path = visualize_early_exit_results(
                history=history,
                evaluation_results=evaluation_results,
                config=config,
                output_dir=args.output_dir
            )
        else:
            # Erstelle eine leere History für Visualisierung
            dummy_history = {
                'train_acc': [], 'train_acc_early': [], 'val_acc': [],
                'train_loss': [], 'train_loss_main': [], 'train_loss_early': [], 'val_loss': [],
                'early_exit_rate': [], 'lr': []
            }
            html_path = visualize_early_exit_results(
                history=dummy_history,
                evaluation_results=evaluation_results,
                config=config,
                output_dir=args.output_dir
            )
        
        logger.info(f"Evaluierung und Visualisierung abgeschlossen. Bericht: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroPizzaNet mit Early Exit trainieren und evaluieren")
    
    parser.add_argument("--mode", type=str, choices=['train', 'evaluate', 'both'], default='both',
                        help="Betriebsmodus: train, evaluate oder both")
    parser.add_argument("--data-dir", type=str, default="data/augmented",
                        help="Verzeichnis mit dem Datensatz")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Anzahl der Trainingsiterationen")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early Stopping Geduld")
    parser.add_argument("--lambda-ee", type=float, default=0.3,
                        help="Gewichtung des Early-Exit-Verlusts (0-1)")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                        help="Konfidenz-Schwellenwert für Early Exit")
    parser.add_argument("--thresholds", type=float, nargs='+', 
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                        help="Liste der zu evaluierenden Konfidenz-Schwellenwerte")
    parser.add_argument("--model-name", type=str, default="micropizzanet_early_exit",
                        help="Name für das gespeicherte Modell")
    parser.add_argument("--model-path", type=str, 
                        default="models_optimized/micropizzanet_early_exit.pth",
                        help="Pfad zum vortrainierten Modell (für Evaluierungsmodus)")
    parser.add_argument("--output-dir", type=str, 
                        default=None,
                        help="Ausgabeverzeichnis für Visualisierungen")
    
    args = parser.parse_args()
    main(args)