import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter

from src.analysis.memory import MemoryEstimator
from src.training.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

def train_microcontroller_model(model, train_loader, val_loader, config, class_names, model_name="micro_pizza_model"):
    """Optimiertes Training mit LR-Scheduling und Loss-Gewichtung für unbalancierte Klassen"""
    logger.info(f"Starte optimiertes Training für Mikrocontroller-Modell...")
    
    # Modellpfad festlegen
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    
    # Parameter- und Speicherschätzungen
    params_count = model.count_parameters()
    memory_report = MemoryEstimator.check_memory_requirements(model, (3, config.IMG_SIZE, config.IMG_SIZE), config)
    
    # Gewichteter Verlust für Klassenbalancierung
    class_counts = Counter()
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1
    
    # Berechne Gewichte invers proportional zur Klassenhäufigkeit
    num_samples = sum(class_counts.values())
    num_classes = len(class_names)  # Verwende class_names für die Gesamtzahl der Klassen
    class_weights = []
    
    # Gewichte für alle Klassen berechnen, auch für die ohne Samples
    for i in range(num_classes):
        if i in class_counts and class_counts[i] > 0:
            class_weights.append(num_samples / (num_classes * class_counts[i]))
        else:
            # Standard-Gewicht für Klassen ohne Samples
            class_weights.append(1.0)
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(config.DEVICE)
    
    logger.info(f"Klassengewichte für Loss-Funktion: {[round(w, 2) for w in class_weights]}")
    
    # Gewichtete Verlustfunktion
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer mit Gewichtsverfall für bessere Generalisierung
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    # OneCycle Learning Rate Scheduler für effizienteres Training
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
        pct_start=0.3,  # 30% der Zeit aufwärmen
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training Tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(config.EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress Bar für Training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        # Batches durchlaufen
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Gradienten zurücksetzen
            optimizer.zero_grad()
            
            # Forward-Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward-Pass und Optimierung
            loss.backward()
            
            # Gradient Clipping gegen explodierende Gradienten
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Statistiken sammeln
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update der Progressbar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Durchschnittliche Trainingsmetriken berechnen
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Klassenweise Genauigkeiten
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                
                # Forward-Pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistiken sammeln
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Klassenweise Genauigkeiten
                correct_mask = (predicted == labels)
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += correct_mask[i].item()
                    class_total[label] += 1
                
                # Update der Progressbar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.0 * correct / total
                })
        
        # Durchschnittliche Validierungsmetriken berechnen
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Ausgabe der Ergebnisse
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Ausgabe der klassenweisen Genauigkeiten
        logger.info("Klassenweise Genauigkeiten:")
        for i in range(len(class_names)):
            if class_total[i] > 0:
                accuracy = 100.0 * class_correct[i] / class_total[i]
                logger.info(f"  {class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # Early Stopping überprüfen
        early_stopping(epoch_val_loss, model)
        
        # Checkpoint speichern (alle 5 Epochen und bei Verbesserung)
        if (epoch + 1) % 5 == 0 or epoch_val_acc > max(history['val_acc'][:-1] + [0]):
            checkpoint_path = os.path.join(config.MODEL_DIR, f"{model_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint gespeichert: {checkpoint_path}")
        
        if early_stopping.early_stop:
            logger.info(f"Early Stopping in Epoche {epoch+1}")
            break
    
    # Trainingszeit
    training_time = time.time() - start_time
    logger.info(f"Training abgeschlossen in {training_time:.2f} Sekunden")
    
    # Stelle beste Gewichte wieder her
    if early_stopping.restore_weights(model):
        logger.info("Beste Modellgewichte wiederhergestellt")
    
    # Speichere finales Modell
    torch.save(model.state_dict(), model_path)
    logger.info(f"Modell gespeichert als: {model_path}")
    
    # Return Training History
    history['training_time'] = training_time
    
    return history, model
