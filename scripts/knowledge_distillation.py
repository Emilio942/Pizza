#!/usr/bin/env python3
"""
Knowledge Distillation für Pizza-Erkennungsmodell.

Dieses Skript implementiert Knowledge Distillation, bei der ein kleineres 'Student'-Modell
von einem größeren, vortrainierten 'Teacher'-Modell lernt. Das Student-Modell lernt sowohl
von den harten Labels als auch von den "soften" Wahrscheinlichkeitsverteilungen des
Teacher-Modells, was zu einem kompakteren, aber effektiveren Modell führen kann.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm

# Projekt-Root zum Pythonpfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    MicroPizzaNet, MicroPizzaNetV2, RP2040Config, 
    create_optimized_dataloaders, MemoryEstimator
)

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_distillation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """
    Implementiert die Knowledge Distillation Loss-Funktion nach Hinton et al. (2015).
    Kombiniert einen harten Verlust (Cross-Entropy mit wahren Labels) mit einem weichen
    Verlust (Kullback-Leibler-Divergenz zwischen den Outputs von Teacher und Student).
    """
    
    def __init__(self, alpha=0.5, temperature=4.0):
        """
        Initialisiert die Distillation Loss-Funktion.
        
        Args:
            alpha: Gewichtung zwischen hartem Verlust (1-alpha) und weichem Verlust (alpha)
            temperature: Temperatur für Softmax, höhere Werte erzeugen weichere Verteilungen
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Berechnet den kombinierten Distillation Loss.
        
        Args:
            student_outputs: Logits vom Student-Modell
            teacher_outputs: Logits vom Teacher-Modell
            targets: Wahre Klassen-Labels
        
        Returns:
            Kombinierter Distillation Loss
        """
        # Harter Verlust: Cross-Entropy mit wahren Labels
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Weicher Verlust: KL-Divergenz zwischen Teacher und Student
        # Skaliere die Logits beider Modelle mit der Temperatur
        soft_student = self.log_softmax(student_outputs / self.temperature)
        soft_teacher = self.softmax(teacher_outputs / self.temperature)
        
        soft_loss = self.kl_loss(soft_student, soft_teacher)
        
        # Kombinierter Verlust (skaliere soft_loss mit dem Quadrat der Temperatur
        # gemäß dem Paper von Hinton)
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss * (self.temperature ** 2)
        
        return loss, hard_loss, soft_loss

def load_teacher_model(model_path, num_classes, device):
    """
    Lädt ein vortrainiertes Teacher-Modell.
    
    Args:
        model_path: Pfad zum gespeicherten Modell
        num_classes: Anzahl der Klassen
        device: Gerät für das Training ('cuda' oder 'cpu')
    
    Returns:
        Das geladene Modell
    """
    logger.info(f"Lade Teacher-Modell von {model_path}")
    
    # Versuche zuerst, ein MicroPizzaNetV2 (komplexeres Modell) zu laden
    try:
        model = MicroPizzaNetV2(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Teacher-Modell als MicroPizzaNetV2 geladen")
    except Exception as e:
        logger.info(f"Konnte nicht als MicroPizzaNetV2 laden: {e}")
        logger.info("Versuche als MicroPizzaNet zu laden")
        
        # Versuche als Fallback, ein MicroPizzaNet (einfacheres Modell) zu laden
        model = MicroPizzaNet(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Teacher-Modell als MicroPizzaNet geladen")
    
    # Fixiere die Weights des Teacher-Modells
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    model.eval()  # Teacher immer im Evaluierungsmodus
    
    return model

def create_student_model(model_type, num_classes, device):
    """
    Erstellt ein Student-Modell des angegebenen Typs.
    
    Args:
        model_type: Typ des Student-Modells ('micro' oder 'custom')
        num_classes: Anzahl der Klassen
        device: Gerät für das Training ('cuda' oder 'cpu')
    
    Returns:
        Das erstellte Student-Modell
    """
    if model_type.lower() == 'micro':
        # Standard MicroPizzaNet (kleiner)
        model = MicroPizzaNet(num_classes=num_classes)
        logger.info("Student-Modell: Standard MicroPizzaNet")
    else:
        # Ein noch kleineres, angepasstes Modell
        model = CustomMicroPizzaNet(num_classes=num_classes)
        logger.info("Student-Modell: CustomMicroPizzaNet (extra klein)")
    
    model = model.to(device)
    return model

class CustomMicroPizzaNet(nn.Module):
    """
    Eine noch kompaktere Version des MicroPizzaNet mit weniger Parametern.
    Optimiert für maximale Effizienz auf dem RP2040.
    """
    def __init__(self, num_classes=6, dropout_rate=0.1):
        super(CustomMicroPizzaNet, self).__init__()
        
        # Erster Block: 3 -> 6 Filter (reduziert von 8)
        self.block1 = nn.Sequential(
            # Standardfaltung für den ersten Layer (3 -> 6)
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 6x12x12
        )
        
        # Zweiter Block: 6 -> 12 Filter mit depthwise separable Faltung (reduziert von 16)
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, groups=6, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1) zur Kanalexpansion
            nn.Conv2d(6, 12, kernel_size=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 12x6x6
        )
        
        # Global Average Pooling spart Parameter im Vergleich zu Flatten + Dense
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Ausgabe: 12x1x1
        
        # Kompakter Klassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 12
            nn.Dropout(dropout_rate),
            nn.Linear(12, num_classes)  # Direkt zur Ausgabeschicht
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

def train_with_distillation(student_model, teacher_model, train_loader, val_loader, 
                           num_epochs=50, alpha=0.5, temperature=4.0, 
                           learning_rate=0.001, device='cuda'):
    """
    Trainiert das Student-Modell mit Knowledge Distillation.
    
    Args:
        student_model: Das zu trainierende Student-Modell
        teacher_model: Das vortrainierte Teacher-Modell
        train_loader: DataLoader für Trainingsdaten
        val_loader: DataLoader für Validierungsdaten
        num_epochs: Anzahl der Trainingsepochen
        alpha: Gewichtung zwischen hartem Verlust (1-alpha) und weichem Verlust (alpha)
        temperature: Temperatur für Softmax
        learning_rate: Lernrate für den Optimierer
        device: Gerät für das Training ('cuda' oder 'cpu')
    
    Returns:
        Das trainierte Student-Modell und ein Dictionary mit der Trainingshistorie
    """
    logger.info(f"Starte Training mit Knowledge Distillation (alpha={alpha}, temp={temperature})")
    
    # Verlustfunktion und Optimizer
    distillation_criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Trainingshistorie
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'hard_loss': [],
        'soft_loss': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    # Beste Modellgewichte speichern
    best_model_weights = None
    best_val_acc = 0.0
    
    # Early Stopping Zähler
    patience = 10
    early_stop_counter = 0
    
    # Training Loop
    for epoch in range(num_epochs):
        # Training
        student_model.train()
        teacher_model.eval()  # Lehrer immer im Evaluierungsmodus
        
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward-Pass durch Teacher-Modell (ohne Gradient-Berechnung)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            # Forward-Pass durch Student-Modell
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            
            # Berechne Distillation Loss
            loss, hard_loss, soft_loss = distillation_criterion(
                student_outputs, teacher_outputs, targets
            )
            
            # Backward und optimize
            loss.backward()
            optimizer.step()
            
            # Statistiken
            running_loss += loss.item() * inputs.size(0)
            running_hard_loss += hard_loss.item() * inputs.size(0)
            running_soft_loss += soft_loss.item() * inputs.size(0)
            
            _, predicted = torch.max(student_outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update der Progressbar
            train_bar.set_postfix({
                'loss': loss.item(),
                'hard_loss': hard_loss.item(),
                'soft_loss': soft_loss.item(),
                'acc': 100.0 * correct / total
            })
        
        # Berechne Trainingsmetriken
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_hard_loss = running_hard_loss / len(train_loader.dataset)
        epoch_train_soft_loss = running_soft_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        
        # Speichere Metriken in der Historie
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['hard_loss'].append(epoch_train_hard_loss)
        history['soft_loss'].append(epoch_train_soft_loss)
        
        # Validierung
        student_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward-Pass durch beide Modelle
                teacher_outputs = teacher_model(inputs)
                student_outputs = student_model(inputs)
                
                # Berechne Distillation Loss für Tracking
                loss, _, _ = distillation_criterion(
                    student_outputs, teacher_outputs, targets
                )
                
                # Statistiken
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(student_outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update der Progressbar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.0 * correct / total
                })
        
        # Berechne Validierungsmetriken
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct / total
        
        # Speichere Metriken in der Historie
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Learning Rate anpassen
        scheduler.step(epoch_val_loss)
        
        # Ausgabe des Fortschritts
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% - "
              f"Hard Loss: {epoch_train_hard_loss:.4f}, Soft Loss: {epoch_train_soft_loss:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Speichere bestes Modell
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = student_model.state_dict().copy()
            history['best_epoch'] = epoch
            history['best_val_acc'] = best_val_acc
            early_stop_counter = 0
            logger.info(f"Neues bestes Modell in Epoch {epoch+1} mit Validation Accuracy: {best_val_acc:.2f}%")
        else:
            early_stop_counter += 1
        
        # Early Stopping
        if early_stop_counter >= patience:
            logger.info(f"Early Stopping in Epoch {epoch+1}")
            break
    
    # Lade die besten Gewichte
    if best_model_weights is not None:
        student_model.load_state_dict(best_model_weights)
        logger.info(f"Beste Modellgewichte aus Epoch {history['best_epoch']+1} geladen")
    
    return student_model, history

def evaluate_model(model, data_loader, device, class_names):
    """
    Evaluiert das Modell auf dem angegebenen DataLoader.
    
    Args:
        model: Das zu evaluierende Modell
        data_loader: DataLoader mit den Testdaten
        device: Gerät für die Evaluation
        class_names: Liste der Klassennamen
    
    Returns:
        Ein Dictionary mit Evaluationsmetriken
    """
    logger.info("Evaluiere Modell...")
    
    model.eval()
    
    # Sammle Vorhersagen und Ground-Truth-Labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluiere"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward-Pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Sammle Ergebnisse
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Berechne Genauigkeit
    accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Berechne Konfusionsmatrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        confusion_matrix[t, p] += 1
    
    # Berechne klassenweise Genauigkeiten
    class_accuracies = []
    for i in range(num_classes):
        class_acc = 100.0 * confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0.0
        class_accuracies.append(class_acc)
        logger.info(f"Genauigkeit für Klasse '{class_names[i]}': {class_acc:.2f}%")
    
    logger.info(f"Gesamtgenauigkeit: {accuracy:.2f}%")
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': confusion_matrix.tolist()
    }

def plot_training_history(history, output_path):
    """
    Visualisiert die Trainingshistorie.
    
    Args:
        history: Dictionary mit Trainingsmetriken
        output_path: Pfad zum Speichern der Visualisierung
    """
    plt.figure(figsize=(15, 10))
    
    # Genauigkeit
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], 'b-', label='Training')
    plt.plot(history['val_acc'], 'r-', label='Validierung')
    plt.axvline(x=history['best_epoch'], color='g', linestyle='--', 
                label=f'Bestes Modell (Epoch {history["best_epoch"]+1})')
    plt.title('Modellgenauigkeit')
    plt.ylabel('Genauigkeit (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Verlust
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], 'b-', label='Training')
    plt.plot(history['val_loss'], 'r-', label='Validierung')
    plt.axvline(x=history['best_epoch'], color='g', linestyle='--')
    plt.title('Gesamtverlust')
    plt.ylabel('Verlust')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Hard vs. Soft Loss
    plt.subplot(2, 2, 3)
    plt.plot(history['hard_loss'], 'b-', label='Hard Loss (CE)')
    plt.plot(history['soft_loss'], 'r-', label='Soft Loss (KL)')
    plt.axvline(x=history['best_epoch'], color='g', linestyle='--')
    plt.title('Verlustkomponenten')
    plt.ylabel('Verlust')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Zusammenfassung
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.05, 0.95, 'Knowledge Distillation Zusammenfassung',
             fontsize=14, fontweight='bold')
    plt.text(0.05, 0.85, f'Beste Validation Accuracy: {history["best_val_acc"]:.2f}%')
    plt.text(0.05, 0.80, f'Beste Epoch: {history["best_epoch"]+1}')
    plt.text(0.05, 0.70, f'Training Accuracy: {history["train_acc"][-1]:.2f}%')
    plt.text(0.05, 0.65, f'Validation Accuracy: {history["val_acc"][-1]:.2f}%')
    plt.text(0.05, 0.55, f'Final Hard Loss: {history["hard_loss"][-1]:.4f}')
    plt.text(0.05, 0.50, f'Final Soft Loss: {history["soft_loss"][-1]:.4f}')
    
    plt.tight_layout()
    
    # Speichere die Visualisierung
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Trainingshistorie gespeichert unter {output_path}")

def compare_models(student_model, teacher_model, val_loader, device, output_dir):
    """
    Vergleicht die Performance und Größe von Student- und Teacher-Modell.
    
    Args:
        student_model: Das trainierte Student-Modell
        teacher_model: Das Teacher-Modell
        val_loader: DataLoader mit Validierungsdaten
        device: Gerät für die Evaluation
        output_dir: Ausgabeverzeichnis für den Vergleichsbericht
    """
    logger.info("Vergleiche Student- und Teacher-Modell...")
    
    # Modellgrößen (Parameter und Speicher)
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    
    student_size_kb = os.path.getsize(os.path.join(output_dir, "student_model.pth")) / 1024
    teacher_size_kb = os.path.getsize(os.path.join(output_dir, "teacher_model.pth")) / 1024
    
    # Parameterreduktion in Prozent
    if teacher_params > 0:
        param_reduction = (1 - student_params / teacher_params) * 100
    else:
        param_reduction = 0
        logger.warning("Teacher-Modell hat 0 Parameter, Reduktionsberechnung nicht möglich")
    
    if teacher_size_kb > 0:
        size_reduction = (1 - student_size_kb / teacher_size_kb) * 100
    else:
        size_reduction = 0
        logger.warning("Teacher-Modell hat 0 KB Größe, Größenreduktionsberechnung nicht möglich")
    
    # Genauigkeitsvergleich
    student_model.eval()
    teacher_model.eval()
    
    student_correct = 0
    teacher_correct = 0
    total = 0
    
    # Inferenzzeiten
    student_times = []
    teacher_times = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Vergleiche"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Teacher-Inferenz mit Zeitmessung
            start_time = time.time()
            teacher_outputs = teacher_model(inputs)
            teacher_times.append(time.time() - start_time)
            
            # Student-Inferenz mit Zeitmessung
            start_time = time.time()
            student_outputs = student_model(inputs)
            student_times.append(time.time() - start_time)
            
            # Bestimme Vorhersagen
            _, teacher_preds = torch.max(teacher_outputs, 1)
            _, student_preds = torch.max(student_outputs, 1)
            
            # Zähle korrekte Vorhersagen
            teacher_correct += (teacher_preds == labels).sum().item()
            student_correct += (student_preds == labels).sum().item()
            total += labels.size(0)
    
    # Berechne Genauigkeiten
    teacher_accuracy = 100.0 * teacher_correct / total if total > 0 else 0
    student_accuracy = 100.0 * student_correct / total if total > 0 else 0
    
    # Berechne durchschnittliche Inferenzzeiten
    avg_teacher_time = np.mean(teacher_times) * 1000 if teacher_times else 0  # in ms
    avg_student_time = np.mean(student_times) * 1000 if student_times else 0  # in ms
    
    # Vermeiden der Division durch Null
    speedup = avg_teacher_time / avg_student_time if avg_student_time > 0 else 0
    
    # Erstelle Vergleichsbericht
    comparison = {
        'teacher_model': {
            'parameters': teacher_params,
            'size_kb': teacher_size_kb,
            'accuracy': teacher_accuracy,
            'inference_time_ms': avg_teacher_time
        },
        'student_model': {
            'parameters': student_params,
            'size_kb': student_size_kb,
            'accuracy': student_accuracy,
            'inference_time_ms': avg_student_time
        },
        'comparison': {
            'parameter_reduction_percent': param_reduction,
            'size_reduction_percent': size_reduction,
            'accuracy_change': student_accuracy - teacher_accuracy,
            'speedup_factor': speedup
        }
    }
    
    # Ausgabe des Vergleichs
    logger.info("\n" + "="*60)
    logger.info("MODELLVERGLEICH: TEACHER VS. STUDENT")
    logger.info("="*60)
    logger.info(f"TEACHER Modell:")
    logger.info(f"  - Parameter: {teacher_params:,}")
    logger.info(f"  - Modellgröße: {teacher_size_kb:.2f} KB")
    logger.info(f"  - Genauigkeit: {teacher_accuracy:.2f}%")
    logger.info(f"  - Inferenzzeit: {avg_teacher_time:.2f} ms")
    
    logger.info(f"\nSTUDENT Modell:")
    logger.info(f"  - Parameter: {student_params:,}")
    logger.info(f"  - Modellgröße: {student_size_kb:.2f} KB")
    logger.info(f"  - Genauigkeit: {student_accuracy:.2f}%")
    logger.info(f"  - Inferenzzeit: {avg_student_time:.2f} ms")
    
    logger.info(f"\nVERGLEICH:")
    logger.info(f"  - Parameterreduktion: {param_reduction:.2f}%")
    logger.info(f"  - Größenreduktion: {size_reduction:.2f}%")
    logger.info(f"  - Genauigkeitsänderung: {student_accuracy - teacher_accuracy:.2f}%")
    logger.info(f"  - Beschleunigungsfaktor: {speedup:.2f}x")
    logger.info("="*60)
    
    # Speichere Vergleichsbericht als JSON
    report_path = os.path.join(output_dir, "model_comparison.json")
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Vergleichsbericht gespeichert unter {report_path}")
    
    # Erstelle Visualisierung des Vergleichs
    plot_comparison(comparison, os.path.join(output_dir, "model_comparison.png"))
    
    return comparison

def plot_comparison(comparison, output_path):
    """
    Erstellt eine Visualisierung des Modellvergleichs.
    
    Args:
        comparison: Dictionary mit Vergleichsdaten
        output_path: Pfad zum Speichern der Visualisierung
    """
    plt.figure(figsize=(12, 8))
    
    # Barishe Darstellung von Größe, Parametern, Genauigkeit und Inferenzzeit
    metrics = ["Modellgröße (KB)", "Parameter (K)", "Genauigkeit (%)", "Inferenzzeit (ms)"]
    
    # Normalisiere Parameterzahlen auf Tausend
    teacher_values = [
        comparison['teacher_model']['size_kb'],
        comparison['teacher_model']['parameters'] / 1000,
        comparison['teacher_model']['accuracy'],
        comparison['teacher_model']['inference_time_ms']
    ]
    
    student_values = [
        comparison['student_model']['size_kb'],
        comparison['student_model']['parameters'] / 1000,
        comparison['student_model']['accuracy'],
        comparison['student_model']['inference_time_ms']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, teacher_values, width, label='Teacher', color='#3498db')
    plt.bar(x + width/2, student_values, width, label='Student', color='#2ecc71')
    
    plt.xlabel('Metrik')
    plt.ylabel('Wert')
    plt.title('Vergleich: Teacher vs. Student Modell')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Füge Werte über den Balken hinzu
    for i, v in enumerate(teacher_values):
        plt.text(i - width/2, v + (max(teacher_values[i], student_values[i]) * 0.02), 
                f"{v:.1f}", ha='center')
    
    for i, v in enumerate(student_values):
        plt.text(i + width/2, v + (max(teacher_values[i], student_values[i]) * 0.02), 
                f"{v:.1f}", ha='center')
    
    # Füge Zusammenfassung hinzu
    plt.figtext(0.5, 0.01, 
               f"Parameterreduktion: {comparison['comparison']['parameter_reduction_percent']:.1f}% | "
               f"Größenreduktion: {comparison['comparison']['size_reduction_percent']:.1f}% | "
               f"Genauigkeitsänderung: {comparison['comparison']['accuracy_change']:.2f}% | "
               f"Beschleunigung: {comparison['comparison']['speedup_factor']:.2f}x",
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Vergleichsvisualisierung gespeichert unter {output_path}")

def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Knowledge Distillation für Pizza-Erkennungsmodell')
    parser.add_argument('--teacher', default='models_optimized/micropizzanetv2_(inverted_residual).pth', 
                        help='Pfad zum Teacher-Modell')
    parser.add_argument('--data', default='data/augmented', help='Verzeichnis mit Trainingsdaten')
    parser.add_argument('--epochs', type=int, default=50, help='Anzahl der Trainingsepochen')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch-Größe')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Gewichtung zwischen hartem und weichem Verlust')
    parser.add_argument('--temperature', type=float, default=4.0, 
                        help='Temperatur für die Softmax-Funktion')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Lernrate')
    parser.add_argument('--output-dir', default='output/knowledge_distillation', 
                        help='Ausgabeverzeichnis für Modell und Visualisierungen')
    parser.add_argument('--student-type', default='micro', choices=['micro', 'custom'], 
                        help='Typ des Student-Modells')
    parser.add_argument('--no-cuda', action='store_true', help='Deaktiviert CUDA-Beschleunigung')
    args = parser.parse_args()
    
    # Ausgabeverzeichnis erstellen
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gerät für Training festlegen
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    logger.info(f"Verwende Gerät: {device}")
    
    # Konfiguration für DataLoader
    config = RP2040Config(data_dir=args.data)
    config.BATCH_SIZE = args.batch_size
    
    # Datenlader erstellen
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    logger.info(f"Klassenamen: {class_names}")
    
    # Teacher-Modell laden
    teacher_model = load_teacher_model(args.teacher, len(class_names), device)
    
    # Speichere Teacher-Modell für den Vergleich
    torch.save(teacher_model.state_dict(), output_dir / "teacher_model.pth")
    
    # Student-Modell erstellen
    student_model = create_student_model(args.student_type, len(class_names), device)
    
    # Vergleiche Modellgrößen
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Teacher-Modell: {teacher_params:,} Parameter")
    logger.info(f"Student-Modell: {student_params:,} Parameter")
    
    # Vermeide Division durch Null
    if teacher_params > 0:
        logger.info(f"Parameterreduktion: {(1 - student_params / teacher_params) * 100:.2f}%")
    else:
        logger.warning("Teacher-Modell hat 0 Parameter, Reduktionsberechnung nicht möglich")
    
    # Evaluiere Teacher-Modell
    teacher_eval = evaluate_model(teacher_model, val_loader, device, class_names)
    logger.info(f"Teacher-Modell Genauigkeit: {teacher_eval['accuracy']:.2f}%")
    
    # Trainiere Student-Modell mit Knowledge Distillation
    trained_student, history = train_with_distillation(
        student_model, 
        teacher_model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs, 
        alpha=args.alpha, 
        temperature=args.temperature, 
        learning_rate=args.learning_rate, 
        device=device
    )
    
    # Speichere Student-Modell
    student_model_path = output_dir / "student_model.pth"
    torch.save(trained_student.state_dict(), student_model_path)
    logger.info(f"Student-Modell gespeichert unter {student_model_path}")
    
    # Erstelle Visualisierungen
    plot_training_history(history, output_dir / "training_history.png")
    
    # Evaluiere Student-Modell
    student_eval = evaluate_model(trained_student, val_loader, device, class_names)
    logger.info(f"Student-Modell Genauigkeit: {student_eval['accuracy']:.2f}%")
    
    # Speichere Evaluationsbericht
    eval_report = {
        'teacher': teacher_eval,
        'student': student_eval,
        'history': history,
        'config': {
            'alpha': args.alpha,
            'temperature': args.temperature,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'student_type': args.student_type
        }
    }
    
    with open(output_dir / "evaluation_report.json", 'w') as f:
        json.dump(eval_report, f, indent=2)
    
    # Vergleiche Modelle
    compare_models(trained_student, teacher_model, val_loader, device, output_dir)
    
    logger.info("\n" + "="*50)
    logger.info("KNOWLEDGE DISTILLATION ABGESCHLOSSEN")
    logger.info("="*50)
    logger.info(f"Teacher-Genauigkeit: {teacher_eval['accuracy']:.2f}%")
    logger.info(f"Student-Genauigkeit: {student_eval['accuracy']:.2f}%")
    logger.info(f"Genauigkeitsänderung: {student_eval['accuracy'] - teacher_eval['accuracy']:.2f}%")
    
    # Vermeide Division durch Null
    if teacher_params > 0:
        logger.info(f"Parameterreduktion: {(1 - student_params / teacher_params) * 100:.2f}%")
    else:
        logger.warning("Teacher-Modell hat 0 Parameter, Reduktionsberechnung nicht möglich")
    
    logger.info(f"Modell und Berichte gespeichert unter: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())