#!/usr/bin/env python3
"""
Trainingsscript für das Pizza-Erkennungsmodell.
Trainiert ein MicroPizzaNet-Modell mit den verfügbaren Daten.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
from torchvision import transforms
import glob
import logging

# Projekt-Root zum Pythonpfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import MicroPizzaNet, RP2040Config, PizzaDatasetAnalysis, MemoryEstimator
from src.constants import INPUT_SIZE, IMAGE_MEAN, IMAGE_STD

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pizza_training_new.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Die 6 Klassen aus dem vorherigen Modell
MODEL_CLASS_NAMES = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']

class PizzaDataset(Dataset):
    """Dataset für Pizza-Bilder mit Unterstützung für verschiedene Klassen."""
    
    def __init__(self, data_dir, transform=None, class_names=MODEL_CLASS_NAMES):
        """
        Initialisiert das Dataset.
        
        Args:
            data_dir: Verzeichnis mit den Bilddaten
            transform: Transformationen für die Bilder
            class_names: Liste mit den Klassennamen
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        
        # Bilder und Labels sammeln
        self.samples = self._collect_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Keine Bilder im Verzeichnis {data_dir} gefunden!")
        
        # Klassenverteilung berechnen
        self._compute_class_weights()
        
    def _collect_samples(self):
        """Sammelt alle Bilder und ordnet sie den entsprechenden Klassen zu."""
        samples = []
        
        # Methode 1: Bilder in Klassenverzeichnissen (z.B. data/augmented/basic/*.jpg)
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists() and class_dir.is_dir():
                for img_path in class_dir.glob("*.jpg"):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.jpeg"):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.png"):
                    samples.append((str(img_path), self.class_to_idx[class_name]))
        
        # Methode 2: Bilder mit Klassennamen im Dateinamen (z.B. pizza_X_basic_Y_Z.jpg)
        for img_path in self.data_dir.glob("**/*.jpg"):
            img_name = img_path.name.lower()
            for class_name in self.class_names:
                if f"_{class_name}_" in img_name:
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                    break
        
        for img_path in self.data_dir.glob("**/*.jpeg"):
            img_name = img_path.name.lower()
            for class_name in self.class_names:
                if f"_{class_name}_" in img_name:
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                    break
                
        for img_path in self.data_dir.glob("**/*.png"):
            img_name = img_path.name.lower()
            for class_name in self.class_names:
                if f"_{class_name}_" in img_name:
                    samples.append((str(img_path), self.class_to_idx[class_name]))
                    break
        
        return samples
    
    def _compute_class_weights(self):
        """Berechnet Gewichtungen für Klassen-Balancierung."""
        # Zähle Bilder pro Klasse
        class_counts = {}
        for _, label in self.samples:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Für Logging: Klassenverteilung
        logger.info("Klassenverteilung im Dataset:")
        for cls_idx, count in class_counts.items():
            if cls_idx < len(self.class_names):  # Sicherheitscheck
                logger.info(f"  {self.class_names[cls_idx]}: {count} Bilder")
        
        # Gesamtzahl und Anzahl Klassen
        num_samples = len(self.samples)
        num_classes = len(self.class_names)
        
        # Berechne Gewichte: (N/K) / n_c wo N=Gesamtanzahl, K=Anzahl Klassen, n_c=Anzahl in Klasse c
        self.class_weights = {}
        for c in range(num_classes):
            count = class_counts.get(c, 0)
            # Wenn keine Samples in dieser Klasse, setze ein hohes Gewicht
            if count == 0:
                self.class_weights[c] = 1.0
            else:
                self.class_weights[c] = num_samples / (num_classes * count)
        
        # Gewichte für jedes Sample
        self.sample_weights = [self.class_weights[label] for _, label in self.samples]
        
        # Logging der Gewichte
        logger.info("Klassengewichte für balanciertes Training:")
        for cls_idx, weight in self.class_weights.items():
            if cls_idx < len(self.class_names):  # Sicherheitscheck
                logger.info(f"  {self.class_names[cls_idx]}: {weight:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Lade und transformiere das Bild
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
        except Exception as e:
            logger.warning(f"Fehler beim Laden von {img_path}: {e}")
            # Fallback: Erstelle ein schwarzes Bild
            if self.transform:
                # Größe bestimmen
                h, w = INPUT_SIZE
                # Schwarzes Bild als Tensor
                return torch.zeros(3, h, w), label
            else:
                # Schwarzes PIL-Bild
                return Image.new('RGB', INPUT_SIZE, (0, 0, 0)), label

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """
    Trainiert das Modell.
    
    Args:
        model: Das zu trainierende Modell
        train_loader: DataLoader für Trainingsdaten
        val_loader: DataLoader für Validierungsdaten
        num_epochs: Anzahl der Trainingsepochen
        learning_rate: Lernrate
        device: Gerät für das Training ('cuda' oder 'cpu')
    
    Returns:
        Das trainierte Modell und ein Dictionary mit der Trainingshistorie
    """
    # Verlustfunktion und Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward und optimize
            loss.backward()
            optimizer.step()
            
            # Statistiken
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Berechne Trainingsmetriken
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        
        # Validierung
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistiken
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Berechne Validierungsmetriken
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct / total
        
        # Learning Rate anpassen
        scheduler.step(epoch_val_loss)
        
        # Speichere Metriken in der Historie
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        # Ausgabe des Fortschritts
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Speichere bestes Modell
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = model.state_dict().copy()
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
        model.load_state_dict(best_model_weights)
        logger.info(f"Beste Modellgewichte aus Epoch {history['best_epoch']+1} geladen")
    
    return model, history

def plot_training_history(history, output_path=None):
    """
    Visualisiert die Trainingshistorie.
    
    Args:
        history: Dictionary mit Trainingsmetriken
        output_path: Pfad zum Speichern der Visualisierung
    """
    plt.figure(figsize=(12, 8))
    
    # Genauigkeit
    plt.subplot(2, 1, 1)
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
    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], 'b-', label='Training')
    plt.plot(history['val_loss'], 'r-', label='Validierung')
    plt.axvline(x=history['best_epoch'], color='g', linestyle='--')
    plt.title('Modellverlust')
    plt.ylabel('Verlust')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Trainingshistorie gespeichert unter {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Training des Pizza-Erkennungsmodells')
    parser.add_argument('--data', default='data/augmented', help='Verzeichnis mit Trainingsdaten')
    parser.add_argument('--epochs', type=int, default=50, help='Anzahl der Trainingsepochen')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch-Größe')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Lernrate')
    parser.add_argument('--output-dir', default='output/models', help='Ausgabeverzeichnis für Modell und Visualisierungen')
    parser.add_argument('--img-size', type=int, default=48, help='Bildgröße für das Training')
    parser.add_argument('--no-cuda', action='store_true', help='Deaktiviert CUDA-Beschleunigung')
    args = parser.parse_args()
    
    # Ausgabeverzeichnis erstellen
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gerät für Training festlegen
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    logger.info(f"Verwende Gerät: {device}")
    
    # Bildgröße
    img_size = (args.img_size, args.img_size)
    
    # Datensatz analysieren für Normalisierungsparameter
    analyzer = PizzaDatasetAnalysis(args.data)
    try:
        params = analyzer.analyze(sample_size=100)
        preprocess_params = analyzer.get_preprocessing_parameters()
        mean = preprocess_params.get('mean', [0.485, 0.456, 0.406])
        std = preprocess_params.get('std', [0.229, 0.224, 0.225])
        logger.info(f"Normalisierungsparameter aus Datensatz: mean={mean}, std={std}")
    except Exception as e:
        logger.warning(f"Konnte Datensatz nicht analysieren: {e}")
        logger.warning("Verwende Standard-Normalisierungsparameter")
        mean = [0.485, 0.456, 0.406]  # ImageNet Mittelwerte
        std = [0.229, 0.224, 0.225]   # ImageNet Standardabweichungen
    
    # Transformationen für Training
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Transformationen für Validierung
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Datensatz laden
    dataset = PizzaDataset(args.data, transform=None, class_names=MODEL_CLASS_NAMES)
    
    # Split in Training und Validierung (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = random_split(
        list(range(len(dataset))), [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Datensatz-Klassen mit entsprechenden Transformationen
    class TransformedSubset(Dataset):
        def __init__(self, dataset, indices, transform=None):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            
            # Falls bereits ein Tensor, konvertiere zurück zu PIL
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
    
    # Erstelle Teilmengen mit entsprechenden Transformationen
    train_dataset = TransformedSubset(dataset, train_indices, train_transform)
    val_dataset = TransformedSubset(dataset, val_indices, val_transform)
    
    # Gewichteter Sampler für Klassenbalancierung
    weights = [dataset.sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Datensatz geladen: {train_size} Trainingsbilder, {val_size} Validierungsbilder")
    
    # Modell erstellen
    num_classes = len(MODEL_CLASS_NAMES)
    logger.info(f"Erstelle MicroPizzaNet für {num_classes} Klassen: {MODEL_CLASS_NAMES}")
    
    model = MicroPizzaNet(num_classes=num_classes)
    model.to(device)
    
    # Modell trainieren
    logger.info("Starte Training...")
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs, 
        learning_rate=args.learning_rate, 
        device=device
    )
    
    # Modell speichern
    model_path = output_dir / "pizza_model_new.pth"
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Modell gespeichert unter {model_path}")
    
    # Visualisierungen erstellen
    plt.switch_backend('agg')  # Nicht-interaktives Backend für Server
    plot_training_history(history, output_dir / "training_history.png")
    
    # Abschließende Zusammenfassung
    logger.info("\n" + "="*50)
    logger.info("TRAINING ABGESCHLOSSEN")
    logger.info("="*50)
    logger.info(f"Beste Validation Accuracy: {history['best_val_acc']:.2f}% in Epoch {history['best_epoch']+1}")
    logger.info(f"Modell gespeichert unter: {model_path}")
    logger.info(f"Trainingsvisualisierung: {output_dir}/training_history.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
