#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight Pruning & Clustering für MicroPizzaNet
---------------------------------------------
Implementiert strukturelle und unstrukturelle Pruning-Techniken sowie
Weight-Clustering zur Optimierung des Modells für RP2040.

Die Strategie umfasst:
1. Unstructured Pruning: Entfernt individuelle unwichtige Gewichte (kleiner Magnitude)
2. Structured Pruning: Entfernt ganze Kanäle/Filter basierend auf ihrer Wichtigkeit
3. Weight Clustering: Gruppiert ähnliche Gewichte, um die Parameter-Redundanz zu reduzieren

Kombiniert führen diese Techniken zu kleineren Modellen mit schnellerer Inferenz
bei minimalem Genauigkeitsverlust.
"""

import os
import sys
import time
import json
import types
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Eigenes Modul importieren
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pizza_detector import (
    MicroPizzaNet, create_optimized_dataloaders, MemoryEstimator, 
    train_microcontroller_model, EarlyStopping, export_to_microcontroller
)
from config.config import DefaultConfig as Config

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pruning_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PruningManager:
    """
    Verwaltet das Pruning eines neuronalen Netzwerks durch verschiedene Techniken.
    Unterstützt sowohl strukturelles als auch unstrukturelles Pruning.
    """
    
    def __init__(self, model, prune_ratio=0.3, structured_ratio=0.2, exclude_bn=True):
        """
        Initialisiert den PruningManager.
        
        Args:
            model: Das zu optimierende PyTorch-Modell
            prune_ratio: Anteil der zu entfernenden Gewichte (0.0-1.0)
            structured_ratio: Anteil der strukturellen Pruning (0.0-1.0)
            exclude_bn: BatchNorm-Layer vom Pruning ausschließen (empfohlen)
        """
        self.model = model
        self.prune_ratio = prune_ratio
        self.structured_ratio = structured_ratio
        self.exclude_bn = exclude_bn
        self.original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        self.masks = {}
        self.channel_masks = {}
        
        # Statistik für die Dokumentation
        self.pruning_stats = {
            'total_params_before': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'pruned_params': 0,
            'total_params_after': 0,
            'pruned_percent': 0.0,
            'pruned_layers': [],
            'structured_pruning': {
                'removed_channels': {},
                'total_channels_removed': 0
            }
        }
    
    def _find_pruneable_layers(self):
        """Identifiziert Layer, die für Pruning in Frage kommen."""
        pruneable_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Separable conv (depthwise) hat spezielle Anforderungen
                is_depthwise = module.groups > 1 and module.groups == module.in_channels
                
                # Pointwise convs und normale convs können stark geprunt werden
                if not is_depthwise:
                    pruneable_layers.append((name, module, 'weight'))
                else:
                    # Depthwise convs brauchen vorsichtigeres Pruning
                    pruneable_layers.append((name, module, 'weight', 'depthwise'))
            
            elif isinstance(module, nn.Linear):
                pruneable_layers.append((name, module, 'weight'))
                
            elif isinstance(module, nn.BatchNorm2d) and not self.exclude_bn:
                pruneable_layers.append((name, module, 'weight'))
        
        return pruneable_layers
    
    def unstructured_pruning(self):
        """
        Führt unstrukturiertes Pruning durch - entfernt Gewichte mit niedriger Magnitude.
        """
        logger.info(f"Starte unstrukturiertes Pruning mit Ratio {self.prune_ratio}...")
        
        # Identifiziere zu prunende Layer
        pruneable_layers = self._find_pruneable_layers()
        
        if not pruneable_layers:
            logger.warning("Keine pruning-fähigen Layer gefunden!")
            return
            
        # Für jede pruneable Schicht eine Maske erstellen
        for layer_info in pruneable_layers:
            name = layer_info[0]
            module = layer_info[1]
            param_name = layer_info[2]
            is_depthwise = len(layer_info) > 3 and layer_info[3] == 'depthwise'
            
            # Parameter holen
            parameter = getattr(module, param_name)
            
            # Weniger aggressive Pruning für depthwise convs
            local_prune_ratio = self.prune_ratio * 0.5 if is_depthwise else self.prune_ratio
            
            # Magnitude-basiertes Pruning - berechne Schwellenwert 
            tensor = parameter.data.cpu().abs().numpy()
            threshold = np.percentile(tensor, local_prune_ratio * 100)
            
            # Erstelle Binärmaske (1 für behalten, 0 für prunen)
            mask = torch.ones_like(parameter.data)
            mask[parameter.data.abs() < threshold] = 0
            
            # Speichere Maske und wende sie an
            self.masks[name + '.' + param_name] = mask
            parameter.data.mul_(mask)  # Pruning anwenden
            
            # Statistiken aktualisieren
            pruned = torch.sum(mask == 0).item()
            total = mask.numel()
            self.pruning_stats['pruned_params'] += pruned
            
            logger.info(f"  Layer {name}: {pruned}/{total} Parameter entfernt ({pruned/total:.2%})")
            
            self.pruning_stats['pruned_layers'].append({
                'name': name,
                'param_name': param_name,
                'total_params': total,
                'pruned_params': pruned,
                'pruned_percent': pruned/total * 100
            })
        
        # Aktualisiere Statistiken
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.pruning_stats['total_params_after'] = total_params
        self.pruning_stats['pruned_percent'] = self.pruning_stats['pruned_params'] / self.pruning_stats['total_params_before'] * 100
        
        logger.info(f"Unstrukturiertes Pruning abgeschlossen. Entfernt: {self.pruning_stats['pruned_params']} Parameter ({self.pruning_stats['pruned_percent']:.2f}%)")
    
    def structured_pruning(self):
        """
        Führt strukturelles Pruning durch - entfernt ganze Kanäle basierend auf L1-Norm.
        Dies führt zu einer effizienteren Matrix-Multiplikation während der Inferenz.
        """
        logger.info(f"Starte strukturelles Pruning mit Ratio {self.structured_ratio}...")
        
        # Wir speichern die zu entfernenden Kanäle pro Layer
        channels_to_prune = {}
        channels_to_prune_indices = {}
        
        # 1. Identifiziere Convolution-Layer für strukturelles Pruning
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Für nicht-depthwise Convs können wir Output-Kanäle prunen
                if module.groups == 1:
                    weight = module.weight.data.clone()
                    
                    # L1-Norm der Filter (entlang der Input, H, W Dimensionen)
                    importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
                    
                    # Anzahl der zu prunenden Kanäle berechnen (mindestens 1 Kanal behalten)
                    num_channels = importance.size(0)
                    num_to_prune = int(num_channels * self.structured_ratio)
                    num_to_prune = min(num_to_prune, num_channels - 1)  # Mindestens einen Kanal behalten
                    
                    if num_to_prune > 0:
                        # Sortiere Kanäle nach Wichtigkeit (aufsteigend)
                        _, indices = torch.sort(importance)
                        to_prune_indices = indices[:num_to_prune].tolist()
                        
                        channels_to_prune[name] = num_to_prune
                        channels_to_prune_indices[name] = to_prune_indices
                        
                        logger.info(f"  Layer {name}: {num_to_prune}/{num_channels} Kanäle für Pruning ausgewählt")
                        
                        self.pruning_stats['structured_pruning']['removed_channels'][name] = num_to_prune
                        self.pruning_stats['structured_pruning']['total_channels_removed'] += num_to_prune
        
        # 2. Erstelle Layer-Mapping - wir müssen wissen, welcher Conv den Input eines anderen bereitstellt
        # Dies ist wichtig, weil Pruning eines Output-Channels in Layer1 bedeutet, dass der entsprechende
        # Input-Channel in Layer2 auch entfernt werden muss
        layer_dependency = self._find_layer_dependencies()
        
        # 3. Führe das tatsächliche Pruning durch
        if channels_to_prune:
            self._apply_structured_pruning(channels_to_prune_indices, layer_dependency)
        else:
            logger.info("Keine geeigneten Layer für strukturelles Pruning gefunden.")
    
    def _find_layer_dependencies(self):
        """
        Erstellt eine Abhängigkeitsmap zwischen Layern.
        Da wir keinen direkte Strukturinformation haben, müssen wir eine Heuristik verwenden:
        Wir gehen davon aus, dass Layer mit passenden Input/Output-Dimensionen verbunden sind.
        """
        dependencies = {}
        
        # Sammle alle Convolutional-Layer
        conv_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers[name] = {
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'groups': module.groups,
                    'module': module
                }
        
        # Für jeden Conv-Layer, finde mögliche Verbindungen
        for target_name, target_info in conv_layers.items():
            dependencies[target_name] = []
            
            # Skip für depthwise convs da diese spezielle Behandlung brauchen
            if target_info['groups'] > 1 and target_info['groups'] == target_info['in_channels']:
                continue
                
            target_in_channels = target_info['in_channels']
            
            for source_name, source_info in conv_layers.items():
                if source_name != target_name:
                    source_out_channels = source_info['out_channels']
                    
                    # Wenn Output-Channels von source = Input-Channels von target, haben wir
                    # wahrscheinlich eine Verbindung (mit Vorsicht zu genießen)
                    if source_out_channels == target_in_channels:
                        dependencies[target_name].append(source_name)
        
        return dependencies
    
    def _apply_structured_pruning(self, channels_to_prune_indices, layer_dependency):
        """
        Wendet strukturelles Pruning auf das Modell an.
        
        Args:
            channels_to_prune_indices: Dict mit Layer-Namen und zu prunenden Kanal-Indizes
            layer_dependency: Dict mit Layer-Namen und deren abhängigen Layern
        """
        # 1. Erstelle für jedes Conv-Layer eine Maske zum Prunen der Output-Kanäle
        for layer_name, indices in channels_to_prune_indices.items():
            module = dict(self.model.named_modules())[layer_name]
            
            # Erstelle Maske für den Weight-Tensor des Layers (zuerst alle behalten)
            mask = torch.ones_like(module.weight.data)
            
            # Setze die zu prunenden Output-Kanäle auf 0
            mask[indices, :, :, :] = 0
            
            # Speichere und wende die Maske an
            mask_name = layer_name + '.weight'
            self.masks[mask_name] = mask
            module.weight.data.mul_(mask)
            
            # Wenn der Layer ein Bias hat, entferne auch die entsprechenden Bias-Werte
            if module.bias is not None:
                bias_mask = torch.ones_like(module.bias.data)
                bias_mask[indices] = 0
                self.masks[layer_name + '.bias'] = bias_mask
                module.bias.data.mul_(bias_mask)
            
            # Speichere auch die Channel-Maske für spätere Referenz
            self.channel_masks[layer_name] = {
                'pruned_channels': indices,
                'kept_channels': [i for i in range(module.out_channels) if i not in indices]
            }
        
        # 2. Passe entsprechende Input-Kanäle in abhängigen Layern an
        for layer_name, dependent_layers in layer_dependency.items():
            if layer_name in channels_to_prune_indices:
                pruned_indices = channels_to_prune_indices[layer_name]
                
                # Für jeden abhängigen Layer
                for dep_layer_name in dependent_layers:
                    try:
                        dep_module = dict(self.model.named_modules())[dep_layer_name]
                        
                        # Prüfe, ob es sich um eine Standard-Convolution handelt (keine Depthwise oder Grouped)
                        if dep_module.groups == 1:
                            # Erstelle Maske für den Weight-Tensor des abhängigen Layers
                            dep_mask = torch.ones_like(dep_module.weight.data)
                            
                            # Stelle sicher, dass die zu prunenden Indizes innerhalb des gültigen Bereichs liegen
                            valid_pruned_indices = [idx for idx in pruned_indices if idx < dep_mask.shape[1]]
                            
                            if len(valid_pruned_indices) > 0:
                                # Setze die Input-Kanäle entsprechend der geprunten Output-Kanäle auf 0
                                dep_mask[:, valid_pruned_indices, :, :] = 0
                                
                                # Speichere und wende die Maske an
                                mask_name = dep_layer_name + '.weight'
                                if mask_name in self.masks:
                                    # Wenn bereits eine Maske existiert, kombinieren
                                    self.masks[mask_name] = self.masks[mask_name] * dep_mask
                                else:
                                    self.masks[mask_name] = dep_mask
                                
                                dep_module.weight.data.mul_(dep_mask)
                            else:
                                logger.info(f"  Keine gültigen Indizes für Layer {dep_layer_name} gefunden, überspringe")
                    except (KeyError, IndexError) as e:
                        logger.warning(f"  Konnte abhängigen Layer {dep_layer_name} nicht anpassen: {e}")
                        continue
        
        logger.info(f"Strukturelles Pruning abgeschlossen. {self.pruning_stats['structured_pruning']['total_channels_removed']} Kanäle entfernt.")
    
    def recompute_statistics(self):
        """Aktualisiert die Statistiken nach dem Pruning."""
        # Zähle verbliebene Parameter
        non_zero_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                non_zero_params += torch.sum(param != 0).item()
        
        # Aktualisiere Statistiken
        self.pruning_stats['total_params_after'] = non_zero_params
        self.pruning_stats['pruned_params'] = self.pruning_stats['total_params_before'] - non_zero_params
        self.pruning_stats['pruned_percent'] = self.pruning_stats['pruned_params'] / self.pruning_stats['total_params_before'] * 100
        
        logger.info(f"Finale Pruning-Statistik: {self.pruning_stats['pruned_params']} Parameter entfernt ({self.pruning_stats['pruned_percent']:.2f}%)")
        logger.info(f"Modellgröße reduziert von {self.pruning_stats['total_params_before']} auf {non_zero_params} Parameter")
    
    def apply_masks(self):
        """Wendet die gespeicherten Masken auf das Modell an (z.B. nach Finetuning)."""
        for name, mask in self.masks.items():
            # Trenne Layer-Name und Parameter-Name
            parts = name.split('.')
            if len(parts) < 2:
                continue
                
            param_name = parts[-1]
            layer_name = '.'.join(parts[:-1])
            
            # Finde das entsprechende Modul
            try:
                module = dict(self.model.named_modules())[layer_name]
                parameter = getattr(module, param_name)
                parameter.data.mul_(mask)
            except (KeyError, AttributeError) as e:
                logger.warning(f"Konnte Maske nicht anwenden für {name}: {e}")
    
    def get_pruning_stats(self):
        """Gibt die Pruning-Statistiken zurück."""
        return self.pruning_stats


class WeightClusterer:
    """
    Implementiert Gewichts-Clustering für Modellkompression.
    Dies reduziert die Anzahl einzigartiger Gewichte durch Zusammenfassen
    ähnlicher Werte, was die Modellgröße reduziert und Inference beschleunigt.
    """
    
    def __init__(self, model, num_clusters=32, exclude_bn=True):
        """
        Initialisiert den WeightClusterer.
        
        Args:
            model: Das zu optimierende PyTorch-Modell
            num_clusters: Anzahl der Gewichts-Cluster (typisch: 16, 32, 64, 128)
            exclude_bn: BatchNorm-Layer ausschließen
        """
        self.model = model
        self.num_clusters = num_clusters
        self.exclude_bn = exclude_bn
        self.clustering_stats = {
            'total_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'clustered_layers': [],
            'unique_values_before': 0,
            'unique_values_after': 0,
            'compression_ratio': 0.0
        }
        self.centroid_map = {}  # Speichert die Clusterzentren pro Layer
    
    def _find_clusterable_layers(self):
        """Identifiziert Layer, die für Clustering geeignet sind."""
        clusterable_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                clusterable_layers.append((name, module, 'weight'))
            elif isinstance(module, nn.BatchNorm2d) and not self.exclude_bn:
                clusterable_layers.append((name, module, 'weight'))
        
        return clusterable_layers
    
    def cluster_weights(self, min_elements=64):
        """
        Clustert Gewichte des Modells mit k-Means-Algorithmus.
        
        Args:
            min_elements: Minimale Anzahl von Elementen für das Clustering
        """
        logger.info(f"Starte Gewichts-Clustering mit {self.num_clusters} Clustern...")
        
        # Zähle insgesamt eindeutige Werte vor dem Clustering
        total_unique_before = 0
        
        # Identifiziere zu clusternde Layer
        clusterable_layers = self._find_clusterable_layers()
        
        if not clusterable_layers:
            logger.warning("Keine clusterfähigen Layer gefunden!")
            return
            
        # Für jeden clusterfähigen Layer
        for layer_info in clusterable_layers:
            name = layer_info[0]
            module = layer_info[1]
            param_name = layer_info[2]
            
            # Parameter holen
            parameter = getattr(module, param_name)
            tensor = parameter.data
            
            # Clustere nur, wenn genügend Elemente vorhanden sind
            if tensor.numel() < min_elements:
                logger.info(f"  Überspringe Layer {name}: Zu wenige Elemente ({tensor.numel()} < {min_elements})")
                continue
                
            # Zähle eindeutige Werte vor dem Clustering
            unique_before = len(torch.unique(tensor))
            total_unique_before += unique_before
            
            # Entscheide über Cluster-Anzahl basierend auf Tensorgrößen
            # Für kleinere Tensoren verwenden wir weniger Cluster
            adaptive_clusters = min(self.num_clusters, max(8, tensor.numel() // 32))
            
            # Gewichte für k-Means-Clustering vorbereiten
            weights_flat = tensor.view(-1).cpu().numpy()
            
            # Führe k-Means-Clustering durch
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=adaptive_clusters, random_state=0).fit(weights_flat.reshape(-1, 1))
            
            # Zentroiden der Cluster
            centroids = kmeans.cluster_centers_.flatten()
            
            # Labels für jedes Gewicht
            labels = kmeans.labels_
            
            # Erstelle geclusterten Tensor
            clustered_weights = torch.from_numpy(centroids[labels]).view(tensor.shape).to(tensor.device)
            
            # Wende geclusterte Gewichte auf das Modell an
            parameter.data = clustered_weights
            
            # Speichere Clusterzentren für spätere Analyse
            self.centroid_map[name + '.' + param_name] = centroids
            
            # Zähle eindeutige Werte nach dem Clustering
            unique_after = len(torch.unique(clustered_weights))
            
            # Aktualisiere Statistiken
            self.clustering_stats['clustered_layers'].append({
                'name': name,
                'param_name': param_name,
                'tensor_size': tensor.numel(),
                'unique_before': unique_before,
                'unique_after': unique_after,
                'centroids': centroids.tolist(),
                'reduction': 1.0 - (unique_after / unique_before)
            })
            
            logger.info(f"  Layer {name}: Eindeutige Werte reduziert von {unique_before} auf {unique_after} ({1.0 - (unique_after / unique_before):.2%} Reduktion)")
        
        # Aktualisiere Gesamt-Statistiken
        total_unique_after = sum(layer['unique_after'] for layer in self.clustering_stats['clustered_layers'])
        self.clustering_stats['unique_values_before'] = total_unique_before
        self.clustering_stats['unique_values_after'] = total_unique_after
        
        if total_unique_before > 0:
            self.clustering_stats['compression_ratio'] = 1.0 - (total_unique_after / total_unique_before)
            
        logger.info(f"Clustering abgeschlossen. Eindeutige Werte reduziert von {total_unique_before} auf {total_unique_after} ({self.clustering_stats['compression_ratio']:.2%} Reduktion)")
    
    def get_clustering_stats(self):
        """Gibt die Clustering-Statistiken zurück."""
        return self.clustering_stats


def fine_tune_pruned_model(model, train_loader, val_loader, config, class_names, num_epochs=10):
    """Feintuned das geprunte Modell, um die Genauigkeit wiederherzustellen."""
    logger.info(f"Starte Feintuning des geprunten Modells für {num_epochs} Epochen...")

    # Erstelle eine Kopie der Konfiguration, um sie anzupassen
    config_copy = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if not k.startswith('__')})
    config_copy.EPOCHS = num_epochs
    config_copy.LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.001) * 0.1 # Reduzierte Lernrate für Fine-Tuning
    config_copy.EARLY_STOPPING_PATIENCE = getattr(config, 'EARLY_STOPPING_PATIENCE', 10) // 2 # Kürzere Geduld
    # Stelle sicher, dass DEVICE korrekt an train_microcontroller_model übergeben wird
    if not hasattr(config_copy, 'DEVICE'):
        config_copy.DEVICE = getattr(config, 'DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')


    # Rufe die Trainingsfunktion aus pizza_detector.py auf
    return train_microcontroller_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_copy,
        class_names=class_names,
        model_name="pruned_pizza_model"
    )


def evaluate_model(model, val_loader, class_names, device):
    """
    Evaluiert das Modell auf dem Validierungsdatensatz.
    
    Args:
        model: Das zu evaluierende Modell
        val_loader: Validierungsdatenlader
        class_names: Namen der Klassen
        device: Gerät für die Berechnung
        
    Returns:
        Dictionary mit Evaluierungsergebnissen
    """
    model.eval()
    correct = 0
    total = 0
    
    # Klassenweise Genauigkeit
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    # Inferenzzeit messen
    inference_times = []
    
    # Konfusionsmatrix
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluiere Modell"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Inferenzzeit messen
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Inferenzergebnisse auswerten
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Klassenweise Genauigkeit
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
            
            # Konfusionsmatrix aktualisieren
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[t, p] += 1
    
    # Berechne Gesamtgenauigkeit
    accuracy = 100.0 * correct / total
    
    # Berechne klassenweise Genauigkeit
    class_accuracies = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_accuracies.append(100.0 * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)
    
    # Berechne durchschnittliche Inferenzzeit
    avg_inference_time = 1000 * np.mean(inference_times)  # ms
    
    logger.info(f"Modell-Evaluation abgeschlossen:")
    logger.info(f"  Genauigkeit: {accuracy:.2f}%")
    logger.info(f"  Durchschnittliche Inferenzzeit: {avg_inference_time:.2f} ms")
    
    for i in range(len(class_names)):
        if class_total[i] > 0:
            logger.info(f"  Klasse {class_names[i]}: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Ergebnisse zurückgeben
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': confusion_matrix.tolist(),
        'avg_inference_time_ms': avg_inference_time
    }


def create_comparison_report(
    base_model_stats, 
    pruned_model_stats, 
    pruning_stats, 
    clustering_stats, 
    output_dir
):
    """
    Erstellt einen Vergleichsbericht für die Modelloptimierung.
    
    Args:
        base_model_stats: Statistiken des Basismodells
        pruned_model_stats: Statistiken des optimierten Modells
        pruning_stats: Statistiken des Pruning-Prozesses
        clustering_stats: Statistiken des Clustering-Prozesses
        output_dir: Verzeichnis für die Ausgabe
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichere Rohstatistiken als JSON
    with open(os.path.join(output_dir, 'optimization_stats.json'), 'w') as f:
        json.dump({
            'base_model': base_model_stats,
            'optimized_model': pruned_model_stats,
            'pruning': pruning_stats,
            'clustering': clustering_stats
        }, f, indent=2)
    
    # Berechne Verbesserungen
    accuracy_change = pruned_model_stats['accuracy'] - base_model_stats['accuracy']
    inference_speedup = base_model_stats['avg_inference_time_ms'] / max(0.001, pruned_model_stats['avg_inference_time_ms'])
    
    # Erstelle HTML-Bericht
    with open(os.path.join(output_dir, 'optimization_report.html'), 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MicroPizzaNet Optimierungsbericht</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .highlight {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .comparison {{
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
        }}
        .comparison > div {{
            width: 48%;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .warning {{
            color: orange;
        }}
        .chart {{
            margin: 30px 0;
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MicroPizzaNet Optimierungsbericht</h1>
        
        <div class="highlight">
            <h2>Zusammenfassung der Optimierung</h2>
            <p>
                Das Modell wurde durch Gewichts-Pruning und Clustering optimiert, um die Modellgröße zu reduzieren 
                und die Inferenzzeit zu verbessern, während die Genauigkeit weitgehend erhalten blieb.
            </p>
            <table>
                <tr>
                    <th>Metrik</th>
                    <th>Basis-Modell</th>
                    <th>Optimiertes Modell</th>
                    <th>Veränderung</th>
                </tr>
                <tr>
                    <td>Genauigkeit</td>
                    <td>{base_model_stats['accuracy']:.2f}%</td>
                    <td>{pruned_model_stats['accuracy']:.2f}%</td>
                    <td class="{('positive' if accuracy_change >= 0 else 'negative')}">
                        {('+' if accuracy_change >= 0 else '')}{accuracy_change:.2f}%
                    </td>
                </tr>
                <tr>
                    <td>Parameter (gesamt)</td>
                    <td>{pruning_stats['total_params_before']}</td>
                    <td>{pruning_stats['total_params_after']}</td>
                    <td class="positive">
                        -{pruning_stats['pruned_percent']:.2f}%
                    </td>
                </tr>
                <tr>
                    <td>Inferenzzeit</td>
                    <td>{base_model_stats['avg_inference_time_ms']:.2f} ms</td>
                    <td>{pruned_model_stats['avg_inference_time_ms']:.2f} ms</td>
                    <td class="positive">
                        {inference_speedup:.2f}x schneller
                    </td>
                </tr>
                <tr>
                    <td>Eindeutige Gewichtswerte</td>
                    <td>{clustering_stats['unique_values_before']}</td>
                    <td>{clustering_stats['unique_values_after']}</td>
                    <td class="positive">
                        -{clustering_stats['compression_ratio']*100:.2f}%
                    </td>
                </tr>
            </table>
        </div>
        
        <h2>Details zum Pruning-Prozess</h2>
        <p>
            Es wurden insgesamt <strong>{pruning_stats['pruned_params']}</strong> Parameter entfernt, 
            was <strong>{pruning_stats['pruned_percent']:.2f}%</strong> des Gesamtmodells entspricht.
        </p>
        
        <h3>Unstrukturiertes Pruning</h3>
        <p>
            Beim unstrukturierten Pruning wurden individuelle Gewichte mit niedriger Magnitude entfernt.
            Die folgende Tabelle zeigt die Layer mit dem höchsten Pruning-Anteil:
        </p>
        <table>
            <tr>
                <th>Layer</th>
                <th>Entfernte Parameter</th>
                <th>Gesamtparameter</th>
                <th>Anteil</th>
            </tr>
""")
        
        # Zeige Top-5 Layer mit höchstem Pruning-Anteil
        sorted_layers = sorted(pruning_stats['pruned_layers'], 
                              key=lambda x: x['pruned_percent'], reverse=True)
        for layer in sorted_layers[:5]:
            f.write(f"""
            <tr>
                <td>{layer['name']}</td>
                <td>{layer['pruned_params']}</td>
                <td>{layer['total_params']}</td>
                <td>{layer['pruned_percent']:.2f}%</td>
            </tr>""")
            
        f.write(f"""
        </table>
        
        <h3>Strukturelles Pruning</h3>
        <p>
            Beim strukturellen Pruning wurden ganze Kanäle/Filter basierend auf ihrer Wichtigkeit entfernt.
        </p>
        <table>
            <tr>
                <th>Layer</th>
                <th>Entfernte Kanäle</th>
            </tr>
""")
        
        # Zeige Layer mit strukturellem Pruning
        for layer_name, num_channels in pruning_stats['structured_pruning']['removed_channels'].items():
            f.write(f"""
            <tr>
                <td>{layer_name}</td>
                <td>{num_channels}</td>
            </tr>""")
            
        f.write(f"""
        </table>
        
        <h2>Details zum Clustering-Prozess</h2>
        <p>
            Durch Clustering ähnlicher Gewichte wurde die Anzahl eindeutiger Werte von 
            <strong>{clustering_stats['unique_values_before']}</strong> auf 
            <strong>{clustering_stats['unique_values_after']}</strong> reduziert.
            Dies entspricht einer Kompressionsrate von <strong>{clustering_stats['compression_ratio']*100:.2f}%</strong>.
        </p>
        
        <h3>Layer mit höchster Clustering-Effizienz</h3>
        <table>
            <tr>
                <th>Layer</th>
                <th>Eindeutige Werte vorher</th>
                <th>Eindeutige Werte nachher</th>
                <th>Reduktion</th>
            </tr>
""")
        
        # Zeige Top-5 Layer mit höchster Clustering-Effizienz
        sorted_clustering = sorted(clustering_stats['clustered_layers'], 
                                  key=lambda x: x['reduction'], reverse=True)
        for layer in sorted_clustering[:5]:
            f.write(f"""
            <tr>
                <td>{layer['name']}</td>
                <td>{layer['unique_before']}</td>
                <td>{layer['unique_after']}</td>
                <td>{layer['reduction']*100:.2f}%</td>
            </tr>""")
            
        f.write(f"""
        </table>
        
        <h2>Klassenweise Genauigkeit</h2>
        <p>
            Die folgende Tabelle zeigt die Genauigkeit pro Klasse vor und nach der Optimierung:
        </p>
        <table>
            <tr>
                <th>Klasse</th>
                <th>Basis-Modell</th>
                <th>Optimiertes Modell</th>
                <th>Veränderung</th>
            </tr>
""")
        
        # Zeige klassenweise Genauigkeit
        for i, class_name in enumerate(pruned_model_stats.get('class_names', [f"Klasse {i}" for i in range(len(base_model_stats['class_accuracies']))])):
            base_acc = base_model_stats['class_accuracies'][i]
            pruned_acc = pruned_model_stats['class_accuracies'][i]
            diff = pruned_acc - base_acc
            
            f.write(f"""
            <tr>
                <td>{class_name}</td>
                <td>{base_acc:.2f}%</td>
                <td>{pruned_acc:.2f}%</td>
                <td class="{('positive' if diff >= 0 else 'negative')}">
                    {('+' if diff >= 0 else '')}{diff:.2f}%
                </td>
            </tr>""")
            
        f.write(f"""
        </table>
        
        <h2>Schlussfolgerungen</h2>
        <div class="highlight">
            <p>
                Die Kombination aus unstrukturiertem Pruning, strukturellem Pruning und Weight-Clustering 
                hat die Modellgröße deutlich reduziert und die Inferenzzeit verbessert, 
                während die Genauigkeit {'verbessert wurde' if accuracy_change > 0 else 'weitgehend erhalten blieb'}.
            </p>
            <p>
                <strong>Optimale Parameter für RP2040-Deployment:</strong>
            </p>
            <ul>
                <li>Unstrukturiertes Pruning: {pruning_stats.get('prune_ratio', 0.3) * 100:.0f}%</li>
                <li>Strukturelles Pruning: {pruning_stats.get('structured_ratio', 0.2) * 100:.0f}%</li>
                <li>Gewichts-Clustering: {clustering_stats.get('num_clusters', 32)} Cluster</li>
            </ul>
            <p>
                Diese Optimierungen ermöglichen eine effizientere Ausführung auf dem RP2040-Mikrocontroller
                durch reduzierte Speicheranforderungen und schnellere Inferenz.
            </p>
        </div>
    </div>
</body>
</html>""")
    
    logger.info(f"Vergleichsbericht erstellt unter: {os.path.join(output_dir, 'optimization_report.html')}")
    
    # Erstelle Visualisierungen
    try:
        plt.figure(figsize=(10, 6))
        
        # Balkendiagramm für Parameter-Reduktion
        plt.subplot(1, 2, 1)
        plt.bar(['Original', 'Optimiert'], 
                [pruning_stats['total_params_before'], pruning_stats['total_params_after']])
        plt.title('Parameter-Reduktion')
        plt.ylabel('Anzahl Parameter')
        
        # Balkendiagramm für Inferenzzeit
        plt.subplot(1, 2, 2)
        plt.bar(['Original', 'Optimiert'], 
                [base_model_stats['avg_inference_time_ms'], pruned_model_stats['avg_inference_time_ms']])
        plt.title('Inferenzzeit-Verbesserung')
        plt.ylabel('Zeit (ms)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_comparison.png'))
        logger.info(f"Visualisierung erstellt unter: {os.path.join(output_dir, 'optimization_comparison.png')}")
    except Exception as e:
        logger.warning(f"Konnte Visualisierung nicht erstellen: {e}")


def main():
    parser = argparse.ArgumentParser(description='Führt Pruning und Weight-Clustering für MicroPizzaNet durch')
    parser.add_argument('--model_path', type=str, help='Pfad zum trainierten Modell')
    parser.add_argument('--output_dir', type=str, default='output/pruned_model', help='Ausgabeverzeichnis')
    parser.add_argument('--prune_ratio', type=float, default=0.3, help='Pruning-Verhältnis (0.0-1.0)')
    parser.add_argument('--structured_ratio', type=float, default=0.2, help='Strukturelles Pruning-Verhältnis (0.0-1.0)')
    parser.add_argument('--num_clusters', type=int, default=32, help='Anzahl der Gewichts-Cluster')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Anzahl der Fine-Tuning-Epochen')
    parser.add_argument('--skip_fine_tuning', action='store_true', help='Fine-Tuning überspringen')
    parser.add_argument('--export_microcontroller', action='store_true', help='Für Mikrocontroller exportieren')
    parser.add_argument('--device', type=str, default='cuda', help='Gerät für die Berechnung (cuda oder cpu)')
    parser.add_argument('--num_classes', type=int, default=6, help='Anzahl der Klassen im Modell')
    
    args = parser.parse_args()
    
    # Verzeichnisstruktur
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Lade Konfiguration
    config = Config()
    
    # Anzahl der Klassen
    num_classes = args.num_classes
    
    # Stelle sicher, dass cuda verfügbar ist, wenn in der Konfiguration spezifiziert
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA nicht verfügbar, wechsle zu CPU")
        device = 'cpu'
    
    logger.info(f"Verwende Gerät: {device}")
    logger.info(f"Anzahl Klassen: {num_classes}")
    
    # Lade oder trainiere Modell
    if args.model_path:
        logger.info(f"Lade vortrainiertes Modell von: {args.model_path}")
        model = MicroPizzaNet(num_classes=num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        logger.info("Trainiere neues Modell da kein Modellpfad angegeben wurde")
        model = MicroPizzaNet(num_classes=num_classes)
        
        # Lade Daten
        train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
        
        # Konfigurationsobjekt für das Training erweitern
        config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('__')}
        config_dict['DEVICE'] = device
        
        # Trainiere Modell
        history, model = train_microcontroller_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=types.SimpleNamespace(**config_dict),
            class_names=class_names,
            model_name="base_pizza_model"
        )
    
    # Lade Daten für Evaluation und Fine-Tuning
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    
    # Setze Modell auf das konfigurierte Gerät
    model = model.to(device)
    
    # Evaluiere Basis-Modell
    logger.info("Evaluiere Basis-Modell vor Optimierung")
    base_model_stats = evaluate_model(model, val_loader, class_names, device)
    base_model_stats['class_names'] = class_names
    
    # Trainierte Modellkopie für später speichern
    base_model = MicroPizzaNet(num_classes=num_classes)
    base_model.load_state_dict(model.state_dict())
    
    # Pruning
    logger.info(f"Starte Pruning mit Prune-Ratio={args.prune_ratio}, Structured-Ratio={args.structured_ratio}")
    pruning_manager = PruningManager(
        model=model,
        prune_ratio=args.prune_ratio,
        structured_ratio=args.structured_ratio
    )
    
    # Unstructured Pruning (magnitude-based)
    pruning_manager.unstructured_pruning()
    
    # Structured Pruning (channel/filter)
    pruning_manager.structured_pruning()
    
    # Aktualisiere Statistiken
    pruning_manager.recompute_statistics()
    pruning_stats = pruning_manager.get_pruning_stats()
    
    # Führe Fine-Tuning durch, wenn nicht übersprungen
    if not args.skip_fine_tuning:
        # Fine-Tuning mit reduzierten Lernraten
        # Konfigurationsobjekt für das Training erweitern
        config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('__')}
        config_dict['DEVICE'] = device
        config_dict['LEARNING_RATE'] = config.LEARNING_RATE * 0.1
        config_dict['EPOCHS'] = args.fine_tune_epochs
        config_dict['EARLY_STOPPING_PATIENCE'] = 5
        
        history, model = train_microcontroller_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=types.SimpleNamespace(**config_dict),
            class_names=class_names,
            model_name="pruned_pizza_model"
        )
        
        # Stelle sicher, dass Nullen nach dem Training noch Null sind
        pruning_manager.apply_masks()
    
    # Gewichts-Clustering
    logger.info(f"Starte Weight-Clustering mit {args.num_clusters} Clustern")
    clusterer = WeightClusterer(
        model=model,
        num_clusters=args.num_clusters
    )
    
    clusterer.cluster_weights()
    clustering_stats = clusterer.get_clustering_stats()
    
    # Speichere optimiertes Modell
    pruned_model_path = os.path.join(output_dir, "pruned_clustered_model.pth")
    torch.save(model.state_dict(), pruned_model_path)
    logger.info(f"Optimiertes Modell gespeichert unter: {pruned_model_path}")
    
    # Evaluiere optimiertes Modell
    logger.info("Evaluiere optimiertes Modell")
    pruned_model_stats = evaluate_model(model, val_loader, class_names, device)
    pruned_model_stats['class_names'] = class_names
    
    # Erstelle Vergleichsbericht
    create_comparison_report(
        base_model_stats=base_model_stats,
        pruned_model_stats=pruned_model_stats,
        pruning_stats=pruning_stats,
        clustering_stats=clustering_stats,
        output_dir=output_dir
    )
    
    # Optional: Für Mikrocontroller exportieren
    if args.export_microcontroller:
        logger.info("Exportiere optimiertes Modell für RP2040-Mikrocontroller")
        
        # Konfigurationsobjekt für den Export erweitern
        config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('__')}
        
        export_info = export_to_microcontroller(
            model=model,
            config=types.SimpleNamespace(**config_dict),
            class_names=class_names,
            preprocess_params=preprocessing_params
        )
        
        logger.info(f"Modell exportiert: {export_info['export_dir']}")
        logger.info(f"Modellgröße: {export_info['model_size_kb']:.2f} KB")


if __name__ == "__main__":
    main()