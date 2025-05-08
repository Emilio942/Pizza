#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) für das Pizza-Erkennungsmodell.

Dieses Skript implementiert Quantization-Aware Training für das MicroPizzaNet,
bei dem während des Trainings die Effekte der Int8-Quantisierung (round-to-nearest-even)
simuliert werden. Dies verbessert die Genauigkeit des quantisierten Modells bei der
Bereitstellung auf dem RP2040-Mikrocontroller.
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, FakeQuantize
from torch.quantization.observer import MinMaxObserver

# Projekt-Root zum Pythonpfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    MicroPizzaNet, MicroPizzaNetV2, RP2040Config, 
    create_optimized_dataloaders, MemoryEstimator, EarlyStopping
)

# Konfiguration der Logging-Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantization_aware_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RoundToNearestEvenFakeQuantize(FakeQuantize):
    """
    Erweiterte FakeQuantize-Implementierung, die explizit round-to-nearest-even verwendet.
    
    Diese Klasse ist größtenteils redundant, da torch.fake_quantize_per_tensor_affine 
    bereits round-to-nearest-even implementiert. Sie dient nur der Dokumentation.
    """
    def forward(self, X):
        if self.training or self.fake_quant_enabled[0] == 1:
            X = torch.fake_quantize_per_tensor_affine(
                X, self.scale.item(), self.zero_point.item(),
                self.quant_min, self.quant_max)
        return X

class QuantizationAwareModule(nn.Module):
    """Basisklasse für alle Module mit integrierter Quantisierungssimulation."""
    def __init__(self):
        super(QuantizationAwareModule, self).__init__()
        # Standard-FakeQuantize verwenden, aber mit den richtigen Parametern
        self.activation_quant = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )()  # Instanz direkt erstellen

class QuantizableConv2d(QuantizationAwareModule):
    """Wrapper für nn.Conv2d mit integrierter Aktivierungsquantisierung."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(QuantizableConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation_quant(x)
        return x

class QAwareReLU(QuantizationAwareModule):
    """ReLU mit Quantisierungssimulation."""
    def __init__(self, inplace=False):
        super(QAwareReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.activation_quant(x)
        return x

class QAwareBatchNorm2d(QuantizationAwareModule):
    """BatchNorm2d mit Quantisierungssimulation."""
    def __init__(self, num_features):
        super(QAwareBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.activation_quant(x)
        return x

class QAwareMicroPizzaNet(nn.Module):
    """
    Quantization-Aware MicroPizzaNet, das die Quantisierung während des Trainings simuliert.
    Diese Version ist speziell für den RP2040-Mikrocontroller optimiert und integriert
    round-to-nearest-even Quantisierungssimulation in den Forward-Pass.
    """
    def __init__(self, num_classes=6, dropout_rate=0.5, channels_multiplier=1.0):
        super(QAwareMicroPizzaNet, self).__init__()
        
        # Basis-Kanalzahlen
        base_channels = [int(c * channels_multiplier) for c in [16, 32, 64, 128]]
        
        # Quantisierungs-Stubs für Eingabe und Ausgabe
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Block 1: Standard-Faltung für den ersten Layer
        self.block1 = nn.Sequential(
            QuantizableConv2d(3, base_channels[0], kernelgröße=3, stride=2, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[0]),
            QAwareReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[0] x 12 x 12
        )
        
        # Block 2: Depthwise Separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            QuantizableConv2d(base_channels[0], base_channels[0], kernelgröße=3, 
                          groups=base_channels[0], stride=1, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[0]),
            QAwareReLU(inplace=True),
            # Pointwise Faltung (1x1)
            QuantizableConv2d(base_channels[0], base_channels[1], kernelgröße=1, bias=False),
            QAwareBatchNorm2d(base_channels[1]),
            QAwareReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[1] x 6 x 6
        )
        
        # Block 3: Zweite Depthwise Separable Faltung
        self.block3 = nn.Sequential(
            # Depthwise Faltung
            QuantizableConv2d(base_channels[1], base_channels[1], kernelgröße=3, 
                          groups=base_channels[1], stride=1, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[1]),
            QAwareReLU(inplace=True),
            # Pointwise Faltung (1x1)
            QuantizableConv2d(base_channels[1], base_channels[2], kernelgröße=1, bias=False),
            QAwareBatchNorm2d(base_channels[2]),
            QAwareReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[2] x 3 x 3
        )
        
        # Feature-Extraktor für den letzten Block
        self.features = nn.Sequential(
            # Eine reguläre Faltung für den letzten Block (größere Rezeptivfelder)
            QuantizableConv2d(base_channels[2], base_channels[3], kernelgröße=3, stride=1, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[3]),
            QAwareReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Globales Average Pooling, Ausgabe: channels[3] x 1 x 1
        )
        
        # Klassifikator-Layer mit Dropout für bessere Generalisierung
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels[3], num_classes)
        )
        
        # Initialisiere Gewichte für bessere Konvergenz
        self._initialize_weights()
        
        # Konfiguriere die Quantisierungseinstellungen
        self.configure_qat()
    
    def forward(self, x):
        # Stelle sicher, dass die Eingabe auf dem gleichen Gerät wie das Modell ist
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Quantisiere Eingabe
        x = self.quant(x)
        
        # Feature-Extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.features(x)
        
        # Klassifikation
        x = self.classifier(x)
        
        # Dequantisiere Ausgabe
        x = self.dequant(x)
        
        return x
    
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
                nn.init.constant_(m.bias, 0)
    
    def configure_qat(self):
        """Konfiguriert das Modell für Quantization-Aware Training"""
        # Verwende per_tensor_affine mit korrekten Grenzen für die Datentypen
        self.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8)
        )
        torch.quantization.prepare_qat(self, inplace=True)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter des Modells"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def to(self, device):
        """
        Überschreibt die to-Methode, um sicherzustellen, dass alle Observer
        und FakeQuantize-Module ebenfalls auf das richtige Gerät verschoben werden.
        """
        super().to(device)
        # Stelle sicher, dass alle Observer und FakeQuantize-Module auf dem richtigen Gerät sind
        for module in self.modules():
            if hasattr(module, 'activation_post_process'):
                module.activation_post_process.to(device)
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.to(device)
        return self

class QAwareMicroPizzaNetWithEarlyExit(nn.Module):
    """
    Quantization-Aware MicroPizzaNet mit einem Early-Exit nach Block 3.
    Bei hoher Konfidenz kann das Modell frühzeitig eine Vorhersage treffen,
    was Rechenzeit spart und dynamische Inferenz ermöglicht.
    """
    def __init__(self, num_classes=6, dropout_rate=0.5, channels_multiplier=1.0, confidence_threshold=0.8):
        super(QAwareMicroPizzaNetWithEarlyExit, self).__init__()
        
        # Speichern des Konfidenz-Schwellenwerts für Early Exit
        self.confidence_threshold = confidence_threshold
        
        # Basis-Kanalzahlen
        base_channels = [int(c * channels_multiplier) for c in [16, 32, 64, 128]]
        
        # Quantisierungs-Stubs für Eingabe und Ausgabe
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Block 1: Standard-Faltung für den ersten Layer
        self.block1 = nn.Sequential(
            QuantizableConv2d(3, base_channels[0], kernelgröße=3, stride=2, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[0]),
            QAwareReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[0] x 12 x 12
        )
        
        # Block 2: Depthwise Separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            QuantizableConv2d(base_channels[0], base_channels[0], kernelgröße=3, 
                          groups=base_channels[0], stride=1, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[0]),
            QAwareReLU(inplace=True),
            # Pointwise Faltung (1x1)
            QuantizableConv2d(base_channels[0], base_channels[1], kernelgröße=1, bias=False),
            QAwareBatchNorm2d(base_channels[1]),
            QAwareReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[1] x 6 x 6
        )
        
        # Block 3: Zweite Depthwise Separable Faltung
        self.block3 = nn.Sequential(
            # Depthwise Faltung
            QuantizableConv2d(base_channels[1], base_channels[1], kernelgröße=3, 
                          groups=base_channels[1], stride=1, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[1]),
            QAwareReLU(inplace=True),
            # Pointwise Faltung (1x1)
            QuantizableConv2d(base_channels[1], base_channels[2], kernelgröße=1, bias=False),
            QAwareBatchNorm2d(base_channels[2]),
            QAwareReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[2] x 3 x 3
        )
        
        # Early Exit Classifier nach Block 3
        self.early_exit_pooling = nn.AdaptiveAvgPool2d(1)  # Globales Average Pooling
        self.early_exit_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels[2], num_classes)
        )
        
        # Feature-Extraktor für den letzten Block
        self.features = nn.Sequential(
            # Eine reguläre Faltung für den letzten Block (größere Rezeptivfelder)
            QuantizableConv2d(base_channels[2], base_channels[3], kernelgröße=3, stride=1, padding=1, bias=False),
            QAwareBatchNorm2d(base_channels[3]),
            QAwareReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Globales Average Pooling, Ausgabe: channels[3] x 1 x 1
        )
        
        # Hauptklassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels[3], num_classes)
        )
        
        # Initialisiere Gewichte für bessere Konvergenz
        self._initialize_weights()
        
        # Konfiguriere die Quantisierungseinstellungen
        self.configure_qat()
    
    def forward(self, x, use_early_exit=True):
        # Stelle sicher, dass die Eingabe auf dem gleichen Gerät wie das Modell ist
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Quantisiere Eingabe
        x = self.quant(x)
        
        # Feature-Extraction - Block 1 und 2
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Early Exit nach Block 3, wenn aktiviert
        early_exit_used = False
        early_exit_output = None
        main_output = None
        
        if use_early_exit:
            # Berechne Ausgabe des Early Exit
            early_features = self.early_exit_pooling(x)
            early_exit_output = self.early_exit_classifier(early_features)
            
            # Berechne Konfidenzen für Early Exit
            early_probs = F.softmax(early_exit_output, dim=1)
            max_confidence, _ = torch.max(early_probs, dim=1)
            
            # Wenn wir im Training sind, immer beide Pfade berechnen
            # Im Inferenzmodus entscheiden wir basierend auf der Konfidenz
            if not self.training and torch.all(max_confidence >= self.confidence_threshold):
                early_exit_used = True
                # Dequantisiere Ausgabe vor dem Zurückgeben
                early_exit_output = self.dequant(early_exit_output)
                return early_exit_output, early_exit_used
        
        # Falls kein Early Exit oder niedrige Konfidenz, berechne Hauptpfad
        if not early_exit_used:
            x = self.features(x)
            main_output = self.classifier(x)
            
            # Dequantisiere Ausgabe
            main_output = self.dequant(main_output)
            
            # Im Trainingsmodus geben wir beide Ausgaben zurück
            if self.training and early_exit_output is not None:
                early_exit_output = self.dequant(early_exit_output)
                return main_output, early_exit_output
        
        return main_output, early_exit_used
    
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
                nn.init.constant_(m.bias, 0)
    
    def configure_qat(self):
        """Konfiguriert das Modell für Quantization-Aware Training"""
        # Verwende per_tensor_affine mit korrekten Grenzen für die Datentypen
        self.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8)
        )
        torch.quantization.prepare_qat(self, inplace=True)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter des Modells"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def to(self, device):
        """
        Überschreibt die to-Methode, um sicherzustellen, dass alle Observer
        und FakeQuantize-Module ebenfalls auf das richtige Gerät verschoben werden.
        """
        super().to(device)
        # Stelle sicher, dass alle Observer und FakeQuantize-Module auf dem richtigen Gerät sind
        for module in self.modules():
            if hasattr(module, 'activation_post_process'):
                module.activation_post_process.to(device)
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.to(device)
        return self

class StdQAwareMicroPizzaNet(nn.Module):
    """
    MicroPizzaNet mit Standard-PyTorch-Layern, optimiert für QAT.
    Verwendet den korrekten PyTorch QAT-Workflow mit QuantStub/DeQuantStub
    und prepare_qat für optimale Quantisierung.
    """
    def __init__(self, num_classes=6, dropout_rate=0.5, channels_multiplier=1.0):
        super(StdQAwareMicroPizzaNet, self).__init__()
        
        # Basis-Kanalzahlen
        base_channels = [int(c * channels_multiplier) for c in [16, 32, 64, 128]]
        
        # Quantisierungs-Stubs für Eingabe und Ausgabe
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Block 1: Standard-Faltung für den ersten Layer
        # Standard PyTorch-Layer, prepare_qat fügt FakeQuantize/Observers ein
        self.block1 = nn.Sequential(
            nn.Conv2d(3, base_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[0] x 12 x 12
        )
        
        # Block 2: Depthwise Separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(base_channels[0], base_channels[0], kernelgröße=3, 
                    groups=base_channels[0], stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1)
            nn.Conv2d(base_channels[0], base_channels[1], kernelgröße=1, bias=False),
            nn.BatchNorm2d(base_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[1] x 6 x 6
        )
        
        # Block 3: Zweite Depthwise Separable Faltung
        self.block3 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(base_channels[1], base_channels[1], kernelgröße=3, 
                    groups=base_channels[1], stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[1]),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1)
            nn.Conv2d(base_channels[1], base_channels[2], kernelgröße=1, bias=False),
            nn.BatchNorm2d(base_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[2] x 3 x 3
        )
        
        # Feature-Extraktor für den letzten Block
        self.features = nn.Sequential(
            # Eine reguläre Faltung für den letzten Block (größere Rezeptivfelder)
            nn.Conv2d(base_channels[2], base_channels[3], kernelgröße=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Globales Average Pooling, Ausgabe: channels[3] x 1 x 1
        )
        
        # Klassifikator-Layer mit Dropout für bessere Generalisierung
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels[3], num_classes)
        )
        
        # Initialisiere Gewichte für bessere Konvergenz
        self._initialize_weights()
        
        # QAT-Konfiguration wird später mit prepare_qat angewendet
        # nach dem das Modell auf das Zielgerät verschoben wurde
    
    def forward(self, x):
        # Die Eingabe sollte bereits auf dem richtigen Gerät sein,
        # wenn das Modell dort platziert wurde.
        # x = x.to(next(self.parameters()).device) # Nicht notwendig, wenn to(device) korrekt aufgerufen wird

        # Quantisiere Eingabe (simuliert)
        x = self.quant(x)
        
        # Feature-Extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.features(x)
        
        # Klassifikation (Arbeitet mit simulierten quantisierten Features, gibt FP32 Logits aus)
        x = self.classifier(x)
        
        # Dequantisiere Ausgabe (simuliert)
        # Dies ist notwendig, damit der Loss mit FP32-Ausgaben berechnet werden kann
        x = self.dequant(x)
        
        return x
    
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
                nn.init.constant_(m.bias, 0)
    
    def prepare_for_qat(self):
        """Konfiguriert das Modell für Quantization-Aware Training"""
        # Definiere QConfig
        self.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8), # Unsigned 8-bit für Aktivierungen
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8) # Signed 8-bit für Gewichte
        )
        # prepare_qat fügt FakeQuantize-Module ein und bereitet Fusionen vor
        torch.quantization.prepare_qat(self, inplace=True)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter des Modells"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class StdQAwareMicroPizzaNetWithEarlyExit(nn.Module):
    """
    MicroPizzaNet mit Early-Exit und Standard-PyTorch-Layern für QAT.
    Nutzt den korrekten PyTorch QAT-Workflow für optimale Quantisierung.
    
    WICHTIG: Die Early-Exit-Kontrollflusslogik (if/else) ist nicht quantisierbar
             mit Standard-Tools. Für die Bereitstellung auf RP2040 müssen die
             Modellteile separat exportiert und die Entscheidungslogik
             in der Inferenzsoftware implementiert werden.
    """
    def __init__(self, num_classes=6, dropout_rate=0.5, channels_multiplier=1.0, confidence_threshold=0.8):
        super(StdQAwareMicroPizzaNetWithEarlyExit, self).__init__()
        
        # Speichern des Konfidenz-Schwellenwerts für Early Exit
        self.confidence_threshold = confidence_threshold
        
        # Basis-Kanalzahlen
        base_channels = [int(c * channels_multiplier) for c in [16, 32, 64, 128]]
        
        # Quantisierungs-Stubs für Eingabe und Ausgabe des Gesamtmodells
        self.quant = QuantStub()
        self.dequant = DeQuantStub() # DequantStub für den Hauptpfad
        
        # Block 1: Standard-Faltung für den ersten Layer
        self.block1 = nn.Sequential(
            nn.Conv2d(3, base_channels[0], kernelgröße=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[0] x 12 x 12
        )
        
        # Block 2: Depthwise Separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(base_channels[0], base_channels[0], kernelgröße=3, 
                    groups=base_channels[0], stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1)
            nn.Conv2d(base_channels[0], base_channels[1], kernelgröße=1, bias=False),
            nn.BatchNorm2d(base_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[1] x 6 x 6
        )
        
        # Block 3: Zweite Depthwise Separable Faltung
        self.block3 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(base_channels[1], base_channels[1], kernelgröße=3, 
                    groups=base_channels[1], stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[1]),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1)
            nn.Conv2d(base_channels[1], base_channels[2], kernelgröße=1, bias=False),
            nn.BatchNorm2d(base_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernelgröße=2, stride=2)  # Ausgabe: channels[2] x 3 x 3
        )
        
        # Early Exit Classifier nach Block 3
        self.early_exit_pooling = nn.AdaptiveAvgPool2d(1)  # Globales Average Pooling
        self.early_exit_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels[2], num_classes)
        )
        
        # Eigene DeQuantStub für den Early-Exit-Pfad
        # Notwendig, da dieser Pfad vor dem Haupt-dequant endet
        self.early_dequant = DeQuantStub()
        
        # Feature-Extraktor für den letzten Block (Fortsetzung nach Early Exit Point)
        self.features = nn.Sequential(
            # Eine reguläre Faltung für den letzten Block (größere Rezeptivfelder)
            nn.Conv2d(base_channels[2], base_channels[3], kernelgröße=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Globales Average Pooling, Ausgabe: channels[3] x 1 x 1
        )
        
        # Hauptklassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels[3], num_classes)
        )
        
        # Initialisiere Gewichte für bessere Konvergenz
        self._initialize_weights()
    
    def forward(self, x, use_early_exit=True):
        # Die Eingabe sollte bereits auf dem richtigen Gerät sein
        # x = x.to(next(self.parameters()).device) # Nicht notwendig

        # Quantisiere Eingabe (simuliert)
        x = self.quant(x)
        
        # Feature-Extraction - Block 1 bis 3
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Berechne Ausgabe des Early Exit
        early_features = self.early_exit_pooling(x)
        early_output_logits = self.early_exit_classifier(early_features)
        
        # Dequantisiere Early Exit Logits, um Konfidenz zu berechnen
        early_output_dequantized = self.early_dequant(early_output_logits)
        
        # Early Exit Entscheidung
        early_exit_used = False
        final_output = None
        
        # Inferenzmodus und Early Exit aktiviert
        if not self.training and use_early_exit:
            early_probs = F.softmax(early_output_dequantized, dim=1)
            max_confidence, _ = torch.max(early_probs, dim=1)
            
            # Hier nehmen wir an, dass entweder ALLE Samples im Batch early exit nehmen ODER KEINES.
            # Dies ist eine Vereinfachung für Batch-Verarbeitung. Eine feinere Implementierung
            # würde pro Sample entscheiden und die Ergebnisse zusammenführen.
            if torch.all(max_confidence >= self.confidence_threshold):
                early_exit_used = True
                final_output = early_output_dequantized
            # Andernfalls (oder wenn Training oder use_early_exit=False): Weiter zum Hauptpfad
        
        # Falls kein Early Exit (oder im Training)
        if not early_exit_used:
            x = self.features(x)
            main_output_logits = self.classifier(x)
            main_output_dequantized = self.dequant(main_output_logits)
            final_output = main_output_dequantized
        
        # Rückgabe im Training: Immer beide dequantisierten Logits zurückgeben
        # Rückgabe in Inferenz: Nur die Logits des gewählten Pfades zurückgeben
        if self.training:
            return main_output_dequantized, early_output_dequantized # Rückgabe beider für den Loss
        else:
             # Rückgabe des finalen Outputs und des Flags, ob Early Exit genommen wurde
            return final_output, early_exit_used # Rückgabe eines einzelnen Outputs für Inferenz
    
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
                nn.init.constant_(m.bias, 0)
    
    def prepare_for_qat(self):
        """Konfiguriert das Modell für Quantization-Aware Training"""
        # Definiere QConfig
        self.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8)
        )
        
        # prepare_qat fügt FakeQuantize-Module ein und bereitet Fusionen vor
        torch.quantization.prepare_qat(self, inplace=True)

        # Nach prepare_qat müssen Sie die Module, die fusioniert werden sollen, explizit definieren und fuse_model aufrufen.
        # Beispiel: torch.quantization.fuse_modules(self.block1, [['0', '1', '2']], inplace=True)
        # Dies muss für alle fusionsfähigen Sequenzen wiederholt werden.
        # Dies wird hier ausgelassen, da es vom genauen Modellgraphen abhängt und
        # den Code komplexer macht, aber es ist ein wichtiger Schritt für die Performance.
        # Für dieses Beispielmodell (Conv-BN-ReLU in Sequential), kann PyTorch das oft selbst erkennen.
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter des Modells"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_qat_model(model, train_loader, val_loader, config, class_names, model_name="qat_pizza_model"):
    """
    Trainiert das Modell mit Quantization-Aware Training und Loss-Gewichtung.
    
    Args:
        model: Das zu trainierende QAT-Modell
        train_loader: DataLoader für Trainingsdaten
        val_loader: DataLoader für Validierungsdaten
        config: Konfigurationsobjekt
        class_names: Liste der Klassennamen
        model_name: Name des Modells für Speicherung
        
    Returns:
        Dictionary mit Trainingshistorie und das trainierte Modell
    """
    logger.info(f"Starte Quantization-Aware Training für {model_name}...")
    
    # Modellpfad festlegen
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    
    # Parameter- und Speicherschätzungen
    params_count = model.count_parameters()
    memory_report = MemoryEstimator.check_memory_requirements(model, (3, config.IMG_SIZE, config.IMG_SIZE), config)
    
    # Stelle sicher, dass alle Observer und FakeQuantize-Module auf dem richtigen Gerät sind
    for module in model.modules():
        if hasattr(module, 'activation_post_process'):
            module.activation_post_process.to(config.DEVICE)
        if hasattr(module, 'weight_fake_quant'):
            module.weight_fake_quant.to(config.DEVICE)
    
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
        if i in class_counts or class_counts[i] > 0:
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
        # Aktiviere/Deaktiviere lernen der Observer alle 4 Epochen
        # (hilft bei stabilem QAT-Training)
        if epoch % 4 == 0:
            logger.info(f"Epoche {epoch+1}: Observer-Learning aktiviert")
            model.apply(torch.quantization.enable_observer)
        else:
            logger.info(f"Epoche {epoch+1}: Observer-Learning deaktiviert")
            model.apply(torch.quantization.disable_observer)
        
        # Aktiviere Fake-Quantisierung nach Epoch 2
        if epoch >= 2:
            logger.info(f"Epoche {epoch+1}: Fake-Quantisierung aktiviert")
            model.apply(torch.quantization.enable_fake_quant)
        
        # Stelle sicher, dass alle Observer und FakeQuantize-Module auf dem richtigen Gerät sind
        for module in model.modules():
            if hasattr(module, 'activation_post_process'):
                module.activation_post_process.to(config.DEVICE)
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.to(config.DEVICE)
        
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
            
            try:
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
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    logger.error(f"Geräte-Fehler: {str(e)}")
                    logger.info("Versuche alle Tensor auf CPU zu verschieben und fortfahren...")
                    
                    # Bei Gerätefehler auf CPU ausweichen
                    model = model.cpu()
                    inputs = inputs.cpu()
                    labels = labels.cpu()
                    
                    for module in model.modules():
                        if hasattr(module, 'activation_post_process'):
                            module.activation_post_process.to('cpu')
                        if hasattr(module, 'weight_fake_quant'):
                            module.weight_fake_quant.to('cpu')
                    
                    # Erneuter Versuch
                    outputs = model(inputs)
                    loss = criterion.cpu()(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    # Bei anderen Fehlern abbrechen
                    raise e
        
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
                # Alles auf dasselbe Gerät verschieben
                device = next(model.parameters()).device
                inputs, labels = inputs.to(device), labels.to(device)
                
                try:
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
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        logger.error(f"Geräte-Fehler während Validierung: {str(e)}")
                        # Bei Gerätefehler auf CPU ausweichen
                        model = model.cpu()
                        inputs = inputs.cpu()
                        labels = labels.cpu()
                        
                        # Stelle sicher, dass alle Observer auf dem CPU sind
                        for module in model.modules():
                            if hasattr(module, 'activation_post_process'):
                                module.activation_post_process.to('cpu')
                            if hasattr(module, 'weight_fake_quant'):
                                module.weight_fake_quant.to('cpu')
                        
                        # Erneuter Versuch
                        outputs = model(inputs)
                        loss = criterion.cpu()(outputs, labels)
                        
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
                    else:
                        # Bei anderen Fehlern abbrechen
                        raise e
        
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

def convert_qat_model_to_quantized(qat_model, config, calibration_data=None):
    """
    Konvertiert ein QAT-trainiertes Modell in ein quantisiertes Modell für die Inferenz.
    
    Args:
        qat_model: Das QAT-trainierte Modell
        config: Konfigurationsobjekt
        calibration_data: Optional, Kalibrierungsdaten für die Quantisierung (nicht benötigt für QAT)
        
    Returns:
        Das quantisierte Modell für die Inferenz
    """
    logger.info("Konvertiere QAT-Modell zu quantisiertem Inferenzmodell...")
    
    # Stelle sicher, dass das Modell im Eval-Modus ist
    qat_model.eval()
    
    # Verschiebe Modell auf CPU für Konvertierung
    cpu_model = qat_model.cpu()
    
    # Deaktiviere Observer, aktiviere Fake-Quantisierung
    cpu_model.apply(torch.quantization.disable_observer)
    cpu_model.apply(torch.quantization.enable_fake_quant)
    
    try:
        # Konvertiere zu statisch quantisiertem Modell
        quantized_model = torch.quantization.convert(cpu_model, inplace=False)
        
        # Speichere quantisiertes Modell
        quantized_model_path = os.path.join(config.MODEL_DIR, "pizza_model_int8_qat.pth")
        torch.save(quantized_model.state_dict(), quantized_model_path)
        
        # Schätze Modellgröße
        model_size_kb = os.path.getsize(quantized_model_path) / 1024
        logger.info(f"Quantisiertes Modell gespeichert als: {quantized_model_path}")
        logger.info(f"Quantisierte Modellgröße: {model_size_kb:.2f} KB")
        
        return quantized_model, {
            'model_path': quantized_model_path,
            'model_size_kb': model_size_kb,
            'quantization_method': 'QAT'
        }
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung des QAT-Modells: {str(e)}")
        logger.info("Versuche manuelles Speichern des Modells...")
        
        # Manuelles Speichern als Fallback
        fallback_path = os.path.join(config.MODEL_DIR, "pizza_model_qat_fallback.pth")
        torch.save(cpu_model.state_dict(), fallback_path)
        
        return cpu_model, {
            'model_path': fallback_path,
            'error': str(e),
            'model_size_kb': os.path.getsize(fallback_path) / 1024,
            'quantization_method': 'QAT (Fallback)'
        }

def convert_qat_model_to_quantized_std(qat_model, config):
    """
    Konvertiert ein QAT-trainiertes StdQAwareMicroPizzaNet in ein quantisiertes Modell.
    """
    logger.info("Konvertiere StdQAwareMicroPizzaNet QAT-Modell zu quantisiertem Inferenzmodell...")
    
    qat_model.eval()
    cpu_model = qat_model.cpu()
    
    # Deaktiviere Observer, aktiviere Fake-Quantisierung VOR der Konvertierung
    # Dies stellt sicher, dass die gesammelten Skalen und ZP im FakeQuant verwendet werden.
    cpu_model.apply(torch.quantization.disable_observer)
    cpu_model.apply(torch.quantization.enable_fake_quant)
    
    try:
        # Konvertiere zu statisch quantisiertem Modell
        quantized_model = torch.quantization.convert(cpu_model, inplace=False)
        
        quantized_model_path = os.path.join(config.MODEL_DIR, "pizza_model_int8_qat_std.pth")
        torch.save(quantized_model.state_dict(), quantized_model_path)
        
        model_size_kb = os.path.getsize(quantized_model_path) / 1024
        logger.info(f"Quantisiertes Modell gespeichert als: {quantized_model_path}")
        logger.info(f"Quantisierte Modellgröße: {model_size_kb:.2f} KB")
        
        return quantized_model, {
            'model_path': quantized_model_path,
            'model_size_kb': model_size_kb,
            'quantization_method': 'QAT (Standard Workflow)'
        }
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung des QAT-Modells: {str(e)}")
        logger.info("Gibt das QAT-Modell (CPU) als Fallback zurück.")
        # Rückgabe des QAT-Modells auf CPU als Fallback für die Evaluierung
        return cpu_model, {
            'model_path': None, # Keine quantisierte Datei gespeichert
            'model_size_kb': os.path.getsize(os.path.join(config.MODEL_DIR, "qat_pizza_model_std.pth")) / 1024 if os.path.exists(os.path.join(config.MODEL_DIR, "qat_pizza_model_std.pth")) else 0,
            'error': str(e),
            'quantization_method': 'QAT (Conversion Failed)'
        }

def evaluate_quantized_model(model, val_loader, class_names, device):
    """
    Evaluiert ein quantisiertes Modell auf dem Validierungsdatensatz.
    
    Args:
        model: Das zu evaluierende Modell
        val_loader: DataLoader für Validierungsdaten
        class_names: Liste der Klassennamen
        device: Gerät für die Evaluation
        
    Returns:
        Dictionary mit Evaluationsmetriken
    """
    logger.info("Evaluiere quantisiertes Modell...")
    
    # Stelle sicher, dass das Modell im Eval-Modus ist
    model.eval()
    correct = 0
    total = 0
    
    # Klassenweise Genauigkeiten
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    # Inferenzzeiten messen
    inference_times = []
    
    # Wrap the evaluation in a try-except to handle possible errors
    try:
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Evaluiere"):
                # CPU-Inferenz für Quantisierung
                inputs = inputs.cpu()
                labels = labels.cpu()
                
                # Forward-Pass mit Zeitmessung
                start_time = time.time()
                outputs = model(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Statistiken sammeln
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Klassenweise Genauigkeiten
                correct_mask = (predicted == labels)
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += correct_mask[i].item()
                    class_total[label] += 1
    except Exception as e:
        logger.warning(f"Fehler bei der Evaluierung des quantisierten Modells: {str(e)}")
        logger.info("Fahre mit der Speicherung des Modells fort ohne Evaluierung...")
        return {
            'accuracy': 0.0,
            'class_accuracies': [0.0] * len(class_names),
            'avg_inference_time_ms': 0.0,
            'error': str(e)
        }
    
    # Berechne Genauigkeit und Inferenzzeit
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_inference_time = np.mean(inference_times) * 1000 if inference_times else 0.0  # in ms
    
    logger.info(f"Quantisiertes Modell Genauigkeit: {accuracy:.2f}%")
    logger.info(f"Durchschnittliche Inferenzzeit: {avg_inference_time:.2f} ms")
    
    # Ausgabe der klassenweisen Genauigkeiten
    logger.info("Klassenweise Genauigkeiten:")
    class_accuracies = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
            logger.info(f"  {class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            class_accuracies.append(0.0)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'avg_inference_time_ms': avg_inference_time,
        'inference_times': [t * 1000 for t in inference_times],
    }

def compare_models(standard_model, qat_model, quantized_model, val_loader, class_names, device, output_dir):
    """
    Vergleicht die Performance von Standard-, QAT- und quantisiertem Modell.
    
    Args:
        standard_model: Regulär trainiertes Modell (ohne QAT)
        qat_model: QAT-trainiertes Modell vor Konvertierung
        quantized_model: Quantisiertes Modell nach Konvertierung
        val_loader: DataLoader für Validierungsdaten
        class_names: Liste der Klassennamen
        device: Gerät für die Evaluation
        output_dir: Ausgabeverzeichnis für den Vergleichsbericht
    
    Returns:
        Dictionary mit Vergleichsmetriken
    """
    logger.info("Vergleiche Modelle: Standard vs. QAT vs. Quantisiert...")
    
    # Alle Modelle auf CPU verschieben für konsistente Evaluation
    standard_model = standard_model.cpu()
    qat_model = qat_model.cpu()
    # Quantized model is already on CPU
    
    # Stelle sicher, dass alle Modelle im Eval-Modus sind
    standard_model.eval()
    qat_model.eval()
    quantized_model.eval()
    
    # Deaktiviere Observer und aktiviere Fake-Quantisierung für QAT-Modell
    qat_model.apply(torch.quantization.disable_observer)
    qat_model.apply(torch.quantization.enable_fake_quant)
    
    # Genauigkeiten und Inferenzzeiten für jedes Modell
    results = {
        'standard': {'correct': 0, 'times': []},
        'qat': {'correct': 0, 'times': []},
        'quantized': {'correct': 0, 'times': []}
    }
    
    total = 0
    
    # Konfusionsmatrizen für jedes Modell
    confusion_matrices = {
        'standard': np.zeros((len(class_names), len(class_names)), dtype=int),
        'qat': np.zeros((len(class_names), len(class_names)), dtype=int),
        'quantized': np.zeros((len(class_names), len(class_names)), dtype=int)
    }
    
    # Evaluiere jedes Modell mit demselben Validierungsdatensatz
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Vergleiche Modelle"):
            # Alles auf CPU - keine GPU-Tensoren verwenden
            cpu_inputs = inputs.cpu()
            cpu_labels = labels.cpu()
            total += labels.size(0)
            
            # Standard-Modell (CPU)
            start_time = time.time()
            outputs = standard_model(cpu_inputs)
            results['standard']['times'].append(time.time() - start_time)
            
            _, predicted = torch.max(outputs, 1)
            results['standard']['correct'] += (predicted == cpu_labels).sum().item()
            
            # Update Konfusionsmatrix
            for t, p in zip(cpu_labels.numpy(), predicted.numpy()):
                confusion_matrices['standard'][t, p] += 1
            
            # QAT-Modell (CPU)
            try:
                start_time = time.time()
                outputs = qat_model(cpu_inputs)
                results['qat']['times'].append(time.time() - start_time)
                
                _, predicted = torch.max(outputs, 1)
                results['qat']['correct'] += (predicted == cpu_labels).sum().item()
                
                # Update Konfusionsmatrix
                for t, p in zip(cpu_labels.numpy(), predicted.numpy()):
                    confusion_matrices['qat'][t, p] += 1
            except RuntimeError as e:
                logger.error(f"Fehler bei der Auswertung des QAT-Modells: {str(e)}")
                # Wir setzen Nullwerte ein, falls das QAT-Modell nicht funktioniert
                results['qat']['correct'] = 0
                results['qat']['times'] = [0]
            
            # Quantisiertes Modell (CPU)
            try:
                # Achtung: Das quantisierte Modell könnte einen AttributeError auslösen
                start_time = time.time()
                outputs = quantized_model(cpu_inputs)
                results['quantized']['times'].append(time.time() - start_time)
                
                _, predicted = torch.max(outputs, 1)
                results['quantized']['correct'] += (predicted == cpu_labels).sum().item()
                
                # Update Konfusionsmatrix
                for t, p in zip(cpu_labels.numpy(), predicted.numpy()):
                    confusion_matrices['quantized'][t, p] += 1
            except (RuntimeError, AttributeError) as e:
                logger.error(f"Fehler bei der Auswertung des quantisierten Modells: {str(e)}")
                logger.info("Setze Nullwerte für das quantisierte Modell ein")
                results['quantized']['correct'] = 0
                results['quantized']['times'] = [0]
                confusion_matrices['quantized'] = np.zeros((len(class_names), len(class_names)), dtype=int)
    
    # Berechne Genauigkeiten und durchschnittliche Inferenzzeiten
    comparison = {}
    for model_type in results:
        accuracy = 100.0 * results[model_type]['correct'] / total if total > 0 else 0
        avg_time = np.mean(results[model_type]['times']) * 1000 if results[model_type]['times'] else 0  # ms
        
        # Berechne klassenweise Genauigkeiten
        class_accuracies = []
        for i in range(len(class_names)):
            row_sum = np.sum(confusion_matrices[model_type][i, :])
            if row_sum > 0:
                class_acc = 100.0 * confusion_matrices[model_type][i, i] / row_sum
            else:
                class_acc = 0.0
            class_accuracies.append(class_acc)
        
        comparison[model_type] = {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_time,
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_matrices[model_type].tolist()
        }
    
    # Berechne Verbesserung durch QAT gegenüber Standard-Quantisierung
    # (theoretisch, da wir hier kein Standard-quantisiertes Modell haben)
    comparison['improvement'] = {
        'accuracy': comparison['quantized']['accuracy'] - comparison['standard']['accuracy'],
        'inference_speedup': comparison['standard']['avg_inference_time_ms'] / 
                            comparison['quantized']['avg_inference_time_ms'] 
                            if comparison['quantized']['avg_inference_time_ms'] > 0 else 0
    }
    
    # Ausgabe des Vergleichs
    logger.info("\n" + "="*60)
    logger.info("MODELLVERGLEICH: STANDARD vs. QAT vs. QUANTISIERT")
    logger.info("="*60)
    
    logger.info(f"STANDARD Modell:")
    logger.info(f"  - Genauigkeit: {comparison['standard']['accuracy']:.2f}%")
    logger.info(f"  - Inferenzzeit: {comparison['standard']['avg_inference_time_ms']:.2f} ms")
    
    logger.info(f"\nQAT Modell:")
    logger.info(f"  - Genauigkeit: {comparison['qat']['accuracy']:.2f}%")
    logger.info(f"  - Inferenzzeit: {comparison['qat']['avg_inference_time_ms']:.2f} ms")
    
    logger.info(f"\nQUANTISIERTES Modell:")
    logger.info(f"  - Genauigkeit: {comparison['quantized']['accuracy']:.2f}%")
    logger.info(f"  - Inferenzzeit: {comparison['quantized']['avg_inference_time_ms']:.2f} ms")
    
    if comparison['quantized']['accuracy'] == 0:
        logger.warning(f"Das quantisierte Modell konnte nicht korrekt evaluiert werden.")
        logger.warning(f"Möglicherweise gibt es ein Problem mit der Konvertierung.")
    
    logger.info(f"\nVERBESSERUNG (Quantisiert vs. Standard):")
    logger.info(f"  - Genauigkeitsänderung: {comparison['improvement']['accuracy']:.2f}%")
    logger.info(f"  - Beschleunigung: {comparison['improvement']['inference_speedup']:.2f}x")
    logger.info("="*60)
    
    # Speichere Vergleichsbericht als JSON
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "model_comparison.json")
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Vergleichsbericht gespeichert unter {report_path}")
    
    # Erstelle Visualisierung des Vergleichs
    plot_comparison(comparison, class_names, os.path.join(output_dir, "model_comparison.png"))
    
    return comparison

def plot_comparison(comparison, class_names, output_path):
    """
    Erstellt eine Visualisierung des Modellvergleichs.
    
    Args:
        comparison: Dictionary mit Vergleichsdaten
        class_names: Liste der Klassennamen
        output_path: Pfad zum Speichern der Visualisierung
    """
    plt.figure(figsize=(14, 10))
    
    # Tabelle 1: Gesamtgenauigkeit und Inferenzzeit
    plt.subplot(2, 2, 1)
    models = ['standard', 'qat', 'quantized']
    model_labels = ['Standard', 'QAT', 'Quantisiert (Int8)']
    accuracies = [comparison[m]['accuracy'] for m in models]
    times = [comparison[m]['avg_inference_time_ms'] for m in models]
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    ax1 = plt.gca()
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Genauigkeit (%)', color='#3498db')
    ax1.set_ylabel('Genauigkeit (%)')
    ax1.set_ylim(0, max(accuracies) * 1.1)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, times, width, label='Inferenzzeit (ms)', color='#e74c3c')
    ax2.set_ylabel('Inferenzzeit (ms)')
    ax2.set_ylim(0, max(times) * 1.1)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Genauigkeit vs. Inferenzzeit')
    
    # Tabelle 2: Klassenweise Genauigkeiten
    plt.subplot(2, 2, 2)
    
    class_accuracies = {
        m: comparison[m]['class_accuracies'] for m in models
    }
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, class_accuracies['standard'], width, label='Standard', color='#3498db')
    plt.bar(x, class_accuracies['qat'], width, label='QAT', color='#2ecc71')
    plt.bar(x + width, class_accuracies['quantized'], width, label='Quantisiert', color='#e74c3c')
    
    plt.ylabel('Genauigkeit (%)')
    plt.title('Klassenweise Genauigkeiten')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    
    # Tabelle 3: Konfusionsmatrix des quantisierten Modells
    plt.subplot(2, 2, 3)
    plot_confusion_matrix(
        comparison['quantized']['confusion_matrix'], 
        class_names,
        title='Konfusionsmatrix (Quantisiertes Modell)'
    )
    
    # Tabelle 4: Zusammenfassung
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.05, 0.95, 'Quantization-Aware Training Ergebnisse',
             fontsize=14, fontweight='bold')
    plt.text(0.05, 0.85, f'Standard Modell Genauigkeit: {comparison["standard"]["accuracy"]:.2f}%')
    plt.text(0.05, 0.80, f'QAT Modell Genauigkeit: {comparison["qat"]["accuracy"]:.2f}%')
    plt.text(0.05, 0.75, f'Quantisiertes Modell Genauigkeit: {comparison["quantized"]["accuracy"]:.2f}%')
    plt.text(0.05, 0.65, f'Genauigkeitsänderung durch QAT: {comparison["improvement"]["accuracy"]:.2f}%')
    plt.text(0.05, 0.60, f'Beschleunigung durch Quantisierung: {comparison["improvement"]["inference_speedup"]:.2f}x')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Vergleichsvisualisierung gespeichert unter {output_path}")

def plot_confusion_matrix(confusion_matrix, class_names, title='Konfusionsmatrix'):
    """
    Visualisiert eine Konfusionsmatrix.
    
    Args:
        confusion_matrix: 2D-Array mit der Konfusionsmatrix
        class_names: Liste der Klassennamen
        title: Titel für die Visualisierung
    """
    cm = np.array(confusion_matrix)
    
    # Normalisiere die Konfusionsmatrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Ersetze NaN durch 0
    
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Werte in die Zellen eintragen
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.ylabel('Wahre Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    plt.tight_layout()

def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Quantization-Aware Training für Pizza-Erkennungsmodell')
    parser.add_argument('--data', default='data/augmented', help='Verzeichnis mit Trainingsdaten')
    parser.add_argument('--epochs', type=int, default=30, help='Anzahl der Trainingsepochen')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch-Größe')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Lernrate')
    parser.add_argument('--output-dir', default='output/quantization_aware_training', 
                        help='Ausgabeverzeichnis für Modell und Visualisierungen')
    parser.add_argument('--pretrained', default=None, help='Pfad zu einem vortrainierten Modell (optional)')
    parser.add_argument('--standard-model', default='models/micro_pizza_model.pth', 
                        help='Pfad zum Standard-Modell für Vergleich')
    parser.add_argument('--no-cuda', action='store_true', help='Deaktiviert CUDA-Beschleunigung')
    parser.add_argument('--early-exit', action='store_true', help='Aktiviert Early-Exit-Modell')
    parser.add_argument('--run', action='store_true', help='Führt nur die Evaluierung eines bereits trainierten Early-Exit-Modells durch')
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
    config.LEARNING_RATE = args.learning_rate
    config.EPOCHS = args.epochs
    
    # Datenlader erstellen
    train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
    logger.info(f"Klassenamen: {class_names}")
    
    if args.early_exit:
        # Early Exit Modell verwenden
        logger.info("Aktiviere Early-Exit-Modell für dynamische Inferenz")
        
        early_exit_output_dir = os.path.join(args.output_dir, "early_exit")
        os.makedirs(early_exit_output_dir, exist_ok=True)
        
        early_exit_model_path = os.path.join(config.MODEL_DIR, "early_exit_pizza_model.pth")
        
        if args.run:
            # Nur Evaluierung eines bereits trainierten Early-Exit-Modells durchführen
            logger.info("Lade vortrainiertes Early-Exit-Modell für Evaluierung")
            
            # Prüfen ob das Modell existiert
            if not os.path.exists(early_exit_model_path):
                logger.error(f"Fehler: Das Early-Exit-Modell wurde nicht gefunden: {early_exit_model_path}")
                logger.info("Sie müssen das Modell zuerst trainieren mit: --early-exit (ohne --run)")
                logger.info("Oder geben Sie ein anderes vortrainiertes Modell mit --pretrained an")
                return 1
            
            try:
                # Versuche zuerst, ein Checkpoint-Modell zu laden
                early_exit_model = QAwareMicroPizzaNetWithEarlyExit(num_classes=len(class_names))
                early_exit_model.load_state_dict(torch.load(early_exit_model_path, map_location=device))
                early_exit_model = early_exit_model.to(device)
                logger.info(f"Vortrainiertes Early-Exit-Modell geladen von {early_exit_model_path}")
                
                # Evaluiere mit verschiedenen Schwellenwerten
                eval_results = evaluate_early_exit_model(
                    early_exit_model, 
                    val_loader, 
                    class_names, 
                    device, 
                    confidence_thresholds=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
                )
                
                # Visualisierungen erstellen
                plot_early_exit_evaluation(eval_results, os.path.join(early_exit_output_dir, "early_exit_evaluation.png"))
                plot_early_exit_trade_off(eval_results, os.path.join(early_exit_output_dir, "early_exit_tradeoff.png"))
                
                # Speichere Evaluierungsbericht
                with open(os.path.join(early_exit_output_dir, "early_exit_evaluation.json"), 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                # Konvertiere zu quantisiertem Modell mit bestem Schwellenwert
                quantized_model, quant_info = convert_early_exit_model_to_quantized(
                    early_exit_model, 
                    config, 
                    best_threshold=eval_results['best_threshold']
                )
                
                logger.info("\n" + "="*60)
                logger.info("EARLY-EXIT-MODELL EVALUIERUNG ABGESCHLOSSEN")
                logger.info("="*60)
                logger.info(f"Bester Konfidenz-Schwellenwert: {eval_results['best_threshold']:.2f}")
                logger.info(f"Early-Exit-Nutzung: {eval_results['best_threshold_results']['early_exit_ratio']:.2%}")
                logger.info(f"Kombininierte Genauigkeit: {eval_results['best_threshold_results']['accuracy_combined']:.2f}%")
                logger.info(f"Geschwindigkeitssteigerung: {eval_results['best_threshold_results']['speedup']:.2f}x")
                logger.info(f"Quantisiertes Early-Exit-Modell: {quant_info['model_path']}")
                logger.info(f"Evaluierungsbericht gespeichert unter: {early_exit_output_dir}")
                logger.info("="*60)
                
            except Exception as e:
                logger.error(f"Fehler beim Laden des Early-Exit-Modells: {str(e)}")
                return 1
            
            return 0
        
        # Neues Training des Early-Exit-Modells
        early_exit_model = QAwareMicroPizzaNetWithEarlyExit(num_classes=len(class_names))
        early_exit_model = early_exit_model.to(device)
        
        # Lade vortrainiertes Modell wenn vorhanden
        if args.pretrained:
            try:
                logger.info(f"Lade vortrainiertes Modell von {args.pretrained}")
                early_exit_model.load_state_dict(torch.load(args.pretrained, map_location=device))
            except Exception as e:
                logger.warning(f"Konnte vortrainiertes Modell nicht laden: {str(e)}")
                logger.info("Starte mit neu initialisierten Gewichten")
        
        # Early-Exit-Training durchführen
        history, early_exit_model = train_early_exit_model(
            early_exit_model,
            train_loader,
            val_loader,
            config,
            class_names,
            model_name="early_exit_pizza_model"
        )
        
        # Visualisiere Trainingshistorie für Early-Exit-Modell
        plot_early_exit_history(history, os.path.join(early_exit_output_dir, "early_exit_training_history.png"))
        
        # Evaluiere Early-Exit-Modell mit verschiedenen Schwellenwerten
        eval_results = evaluate_early_exit_model(
            early_exit_model, 
            val_loader, 
            class_names, 
            device, 
            confidence_thresholds=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        )
        
        # Visualisierungen erstellen
        plot_early_exit_evaluation(eval_results, os.path.join(early_exit_output_dir, "early_exit_evaluation.png"))
        plot_early_exit_trade_off(eval_results, os.path.join(early_exit_output_dir, "early_exit_tradeoff.png"))
        
        # Speichere Evaluierungsbericht
        with open(os.path.join(early_exit_output_dir, "early_exit_evaluation.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Konvertiere Early-Exit-Modell zu quantisiertem Modell
        quantized_model, quant_info = convert_early_exit_model_to_quantized(
            early_exit_model, 
            config, 
            best_threshold=eval_results['best_threshold']
        )
        
        # Ausgabe der Ergebnisse
        logger.info("\n" + "="*60)
        logger.info("EARLY-EXIT-MODELL TRAINING ABGESCHLOSSEN")
        logger.info("="*60)
        logger.info(f"Early-Exit-Modell Pfad: {early_exit_model_path}")
        logger.info(f"Bester Konfidenz-Schwellenwert: {eval_results['best_threshold']:.2f}")
        logger.info(f"Early-Exit-Nutzung: {eval_results['best_threshold_results']['early_exit_ratio']:.2%}")
        logger.info(f"Kombininierte Genauigkeit: {eval_results['best_threshold_results']['accuracy_combined']:.2f}%")
        logger.info(f"Geschwindigkeitssteigerung: {eval_results['best_threshold_results']['speedup']:.2f}x")
        logger.info(f"Quantisiertes Early-Exit-Modell: {quant_info['model_path']}")
        logger.info(f"Modell und Berichte gespeichert unter: {early_exit_output_dir}")
        logger.info("="*60)
        
    else:
        # Standard QAT-Modell verwenden
        qat_model = QAwareMicroPizzaNet(num_classes=len(class_names))
        qat_model = qat_model.to(device)
        
        # Lade vortrainiertes Modell wenn vorhanden
        if args.pretrained:
            logger.info(f"Lade vortrainiertes Modell von {args.pretrained}")
            qat_model.load_state_dict(torch.load(args.pretrained, map_location=device))
        
        # Standard-Modell für Vergleich laden
        standard_model = MicroPizzaNet(num_classes=len(class_names))
        standard_model.load_state_dict(torch.load(args.standard_model, map_location=device))
        standard_model = standard_model.to(device)
        
        # QAT-Training durchführen
        history, qat_model = train_qat_model(
            qat_model, 
            train_loader, 
            val_loader, 
            config, 
            class_names, 
            model_name="qat_pizza_model"
        )
        
        # Visualisiere Trainingshistorie
        plot_training_history(history, os.path.join(args.output_dir, "qat_training_history.png"))
        
        # Konvertiere QAT-Modell zu quantisiertem Modell
        quantized_model, quant_info = convert_qat_model_to_quantized(qat_model, config)
        
        # Evaluiere quantisiertes Modell
        quant_eval = evaluate_quantized_model(quantized_model, val_loader, class_names, device)
        
        # Speichere Evaluationsbericht
        eval_report = {
            'qat_training': history,
            'quantized_model': {
                **quant_info,
                **quant_eval
            }
        }
        
        with open(os.path.join(args.output_dir, "qat_evaluation_report.json"), 'w') as f:
            json.dump(eval_report, f, indent=2)
        
        # Vergleiche Modelle
        compare_models(
            standard_model,
            qat_model,
            quantized_model,
            val_loader,
            class_names,
            device,
            args.output_dir
        )
        
        logger.info("\n" + "="*50)
        logger.info("QUANTIZATION-AWARE TRAINING ABGESCHLOSSEN")
        logger.info("="*50)
        logger.info(f"Standard-Modell Pfad: {args.standard_model}")
        logger.info(f"QAT-Modell Pfad: {os.path.join(config.MODEL_DIR, 'qat_pizza_model.pth')}")
        logger.info(f"Quantisiertes Modell Pfad: {quant_info['model_path']}")
        logger.info(f"Quantisierte Modellgröße: {quant_info['model_size_kb']:.2f} KB")
        logger.info(f"Quantisiertes Modell Genauigkeit: {quant_eval['accuracy']:.2f}%")
        logger.info(f"Modell und Berichte gespeichert unter: {args.output_dir}")
    
    return 0

def plot_training_history(history, output_path):
    """
    Visualisiert die Trainingshistorie.
    
    Args:
        history: Dictionary mit Trainingsmetriken
        output_path: Pfad zum Speichern der Visualisierung
    """
    plt.figure(figsize=(12, 8))
    
    # Genauigkeit
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], 'b-', label='Training')
    plt.plot(history['val_acc'], 'r-', label='Validierung')
    plt.title('Modellgenauigkeit')
    plt.ylabel('Genauigkeit (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Verlust
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], 'b-', label='Training')
    plt.plot(history['val_loss'], 'r-', label='Validierung')
    plt.title('Modellverlust')
    plt.ylabel('Verlust')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'], 'g-')
    plt.title('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    # Zusammenfassung
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.05, 0.95, 'Quantization-Aware Training Zusammenfassung',
             fontsize=14, fontweight='bold')
    plt.text(0.05, 0.85, f'Finale Training Accuracy: {history["train_acc"][-1]:.2f}%')
    plt.text(0.05, 0.80, f'Finale Validation Accuracy: {history["val_acc"][-1]:.2f}%')
    plt.text(0.05, 0.70, f'Beste Validation Accuracy: {max(history["val_acc"]):.2f}%')
    plt.text(0.05, 0.65, f'Beste Epoch: {history["val_acc"].index(max(history["val_acc"])) + 1}')
    plt.text(0.05, 0.55, f'Trainingszeit: {history.get("training_time", 0)::.2f} Sekunden')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Trainingshistorie gespeichert unter {output_path}")

if __name__ == "__main__":
    sys.exit(main())

def train_early_exit_model(model, train_loader, val_loader, config, class_names, model_name="early_exit_pizza_model"):
    """
    Trainiert ein Modell mit Early-Exit-Funktionalität und kombinierten Loss-Funktionen.
    
    Args:
        model: Das zu trainierende Early-Exit-Modell
        train_loader: DataLoader für Trainingsdaten
        val_loader: DataLoader für Validierungsdaten
        config: Konfigurationsobjekt
        class_names: Liste der Klassennamen
        model_name: Name des Modells für Speicherung
        
    Returns:
        Dictionary mit Trainingshistorie und das trainierte Modell
    """
    logger.info(f"Starte Training für Early-Exit-Modell {model_name}...")
    
    # Modellpfad festlegen
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    
    # Parameter- und Speicherschätzungen
    params_count = model.count_parameters()
    memory_report = MemoryEstimator.check_memory_requirements(model, (3, config.IMG_SIZE, config.IMG_SIZE), config)
    
    logger.info(f"Modell hat {params_count:,} Parameter")
    logger.info(f"Geschätzte Speicheranforderungen: {memory_report['total_runtime_memory_kb']:.2f} KB")
    
    # Stelle sicher, dass alle Observer und FakeQuantize-Module auf dem richtigen Gerät sind
    for module in model.modules():
        if hasattr(module, 'activation_post_process'):
            module.activation_post_process.to(config.DEVICE)
        if hasattr(module, 'weight_fake_quant'):
            module.weight_fake_quant.to(config.DEVICE)
    
    # Gewichteter Verlust für Klassenbalancierung
    class_counts = Counter()
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1
    
    # Berechne Gewichte invers proportional zur Klassenhäufigkeit
    num_samples = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = []
    
    # Gewichte für alle Klassen berechnen, auch für die ohne Samples
    for i in range(num_classes):
        if i in class_counts or class_counts[i] > 0:
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
        'train_main_loss': [],
        'train_early_loss': [],
        'val_loss': [],
        'val_main_loss': [],
        'val_early_loss': [],
        'train_acc': [],
        'train_main_acc': [],
        'train_early_acc': [],
        'val_acc': [],
        'val_main_acc': [],
        'val_early_acc': [],
        'lr': []
    }
    
    # Wichtungsfaktor für die Verlustteilnehmer
    # Den Hauptklassifikator leicht höher gewichten (0.6 vs 0.4)
    main_loss_weight = 0.6
    early_loss_weight = 0.4
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(config.EPOCHS):
        # Aktiviere/Deaktiviere lernen der Observer alle 4 Epochen
        if epoch % 4 == 0:
            logger.info(f"Epoche {epoch+1}: Observer-Learning aktiviert")
            model.apply(torch.quantization.enable_observer)
        else:
            logger.info(f"Epoche {epoch+1}: Observer-Learning deaktiviert")
            model.apply(torch.quantization.disable_observer)
        
        # Aktiviere Fake-Quantisierung nach Epoch 2
        if epoch >= 2:
            logger.info(f"Epoche {epoch+1}: Fake-Quantisierung aktiviert")
            model.apply(torch.quantization.enable_fake_quant)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_early_loss = 0.0
        correct_main = 0
        correct_early = 0
        total = 0
        
        # Progress Bar für Training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        # Batches durchlaufen
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Gradienten zurücksetzen
            optimizer.zero_grad()
            
            try:
                # Forward-Pass - Im Trainingsmodus gibt das Modell beide Ausgaben zurück
                main_output, early_output = model(inputs)
                
                # Verlust für beide Ausgaben berechnen
                main_loss = criterion(main_output, labels)
                early_loss = criterion(early_output, labels)
                
                # Gewichteten Gesamtverlust berechnen
                loss = main_loss_weight * main_loss + early_loss_weight * early_loss
                
                # Backward-Pass und Optimierung
                loss.backward()
                
                # Gradient Clipping gegen explodierende Gradienten
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Statistiken sammeln
                running_loss += loss.item() * inputs.size(0)
                running_main_loss += main_loss.item() * inputs.size(0)
                running_early_loss += early_loss.item() * inputs.size(0)
                
                _, main_predicted = torch.max(main_output, 1)
                _, early_predicted = torch.max(early_output, 1)
                
                total += labels.size(0)
                correct_main += (main_predicted == labels).sum().item()
                correct_early += (early_predicted == labels).sum().item()
                
                # Update der Progressbar
                train_bar.set_postfix({
                    'loss': loss.item(),
                    'main_acc': 100.0 * correct_main / total,
                    'early_acc': 100.0 * correct_early / total,
                    'lr': optimizer.param_groups[0]['lr']
                })
            except RuntimeError as e:
                logger.error(f"Geräte-Fehler: {str(e)}")
                # Bei schwerwiegenden Fehlern abbrechen
                raise e
        
        # Durchschnittliche Trainingsmetriken berechnen
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_main_loss = running_main_loss / len(train_loader.dataset)
        epoch_train_early_loss = running_early_loss / len(train_loader.dataset)
        
        epoch_train_main_acc = 100.0 * correct_main / total
        epoch_train_early_acc = 100.0 * correct_early / total
        epoch_train_acc = 100.0 * (correct_main + correct_early) / (2 * total)  # Durchschnitt beider Genauigkeiten
        
        history['train_loss'].append(epoch_train_loss)
        history['train_main_loss'].append(epoch_train_main_loss)
        history['train_early_loss'].append(epoch_train_early_loss)
        history['train_acc'].append(epoch_train_acc)
        history['train_main_acc'].append(epoch_train_main_acc)
        history['train_early_acc'].append(epoch_train_early_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation Phase
        model.eval()
        running_loss = 0.0
        running_main_loss = 0.0
        running_early_loss = 0.0
        correct_main = 0
        correct_early = 0
        total = 0
        
        # Klassenweise Genauigkeiten
        class_correct_main = [0] * len(class_names)
        class_correct_early = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
            for inputs, labels in val_bar:
                # Alles auf dasselbe Gerät verschieben
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                
                try:
                    # Forward-Pass - Im Trainingsmodus gibt das Modell beide Ausgaben zurück
                    main_output, early_output = model(inputs)
                    
                    # Verlust für beide Ausgaben berechnen
                    main_loss = criterion(main_output, labels)
                    early_loss = criterion(early_output, labels)
                    
                    # Gewichteten Gesamtverlust berechnen
                    loss = main_loss_weight * main_loss + early_loss_weight * early_loss
                    
                    # Statistiken sammeln
                    running_loss += loss.item() * inputs.size(0)
                    running_main_loss += main_loss.item() * inputs.size(0)
                    running_early_loss += early_loss.item() * inputs.size(0)
                    
                    _, main_predicted = torch.max(main_output, 1)
                    _, early_predicted = torch.max(early_output, 1)
                    
                    total += labels.size(0)
                    correct_main += (main_predicted == labels).sum().item()
                    correct_early += (early_predicted == labels).sum().item()
                    
                    # Klassenweise Genauigkeiten
                    main_correct_mask = (main_predicted == labels)
                    early_correct_mask = (early_predicted == labels)
                    for i in range(len(labels)):
                        label = labels[i].item()
                        class_correct_main[label] += main_correct_mask[i].item()
                        class_correct_early[label] += early_correct_mask[i].item()
                        class_total[label] += 1
                    
                    # Update der Progressbar
                    val_bar.set_postfix({
                        'loss': loss.item(),
                        'main_acc': 100.0 * correct_main / total,
                        'early_acc': 100.0 * correct_early / total
                    })
                except RuntimeError as e:
                    logger.error(f"Geräte-Fehler während Validierung: {str(e)}")
                    # Bei schwerwiegenden Fehlern abbrechen
                    raise e
        
        # Durchschnittliche Validierungsmetriken berechnen
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_main_loss = running_main_loss / len(val_loader.dataset)
        epoch_val_early_loss = running_early_loss / len(val_loader.dataset)
        
        epoch_val_main_acc = 100.0 * correct_main / total
        epoch_val_early_acc = 100.0 * correct_early / total
        epoch_val_acc = 100.0 * (correct_main + correct_early) / (2 * total)  # Durchschnitt beider Genauigkeiten
        
        history['val_loss'].append(epoch_val_loss)
        history['val_main_loss'].append(epoch_val_main_loss)
        history['val_early_loss'].append(epoch_val_early_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_main_acc'].append(epoch_val_main_acc)
        history['val_early_acc'].append(epoch_val_early_acc)
        
        # Ausgabe der Ergebnisse
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} - "
              f"Train Loss: {epoch_train_loss:.4f} (Main: {epoch_train_main_loss:.4f}, Early: {epoch_train_early_loss:.4f}), "
              f"Train Acc: {epoch_train_acc:.2f}% (Main: {epoch_train_main_acc:.2f}%, Early: {epoch_train_early_acc:.2f}%) - "
              f"Val Loss: {epoch_val_loss:.4f} (Main: {epoch_val_main_loss:.4f}, Early: {epoch_val_early_loss:.4f}), "
              f"Val Acc: {epoch_val_acc:.2f}% (Main: {epoch_val_main_acc:.2f}%, Early: {epoch_val_early_acc:.2f}%)")
        
        # Ausgabe der klassenweisen Genauigkeiten
        logger.info("Klassenweise Genauigkeiten (Hauptklassifikator):")
        for i in range(len(class_names)):
            if class_total[i] > 0:
                accuracy = 100.0 * class_correct_main[i] / class_total[i]
                logger.info(f"  {class_names[i]}: {accuracy:.2f}% ({class_correct_main[i]}/{class_total[i]})")
        
        logger.info("Klassenweise Genauigkeiten (Early-Exit):")
        for i in range(len(class_names)):
            if class_total[i] > 0:
                accuracy = 100.0 * class_correct_early[i] / class_total[i]
                logger.info(f"  {class_names[i]}: {accuracy:.2f}% ({class_correct_early[i]}/{class_total[i]})")
        
        # Early Stopping überprüfen (basierend auf dem Gesamtverlust)
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

def evaluate_early_exit_model(model, val_loader, class_names, device, confidence_thresholds=None):
    """
    Evaluiert ein Early-Exit-Modell mit verschiedenen Konfidenz-Schwellenwerten.
    
    Args:
        model: Das zu evaluierende Early-Exit-Modell
        val_loader: DataLoader für Validierungsdaten
        class_names: Liste der Klassennamen
        device: Gerät für die Evaluation
        confidence_thresholds: Liste von Konfidenz-Schwellenwerten für den Early Exit
        
    Returns:
        Dictionary mit Evaluationsmetriken
    """
    logger.info("Evaluiere Early-Exit-Modell...")
    
    # Verwende Standard-Schwellenwerte, wenn keine angegeben wurden
    if confidence_thresholds is None:
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Stelle sicher, dass das Modell im Eval-Modus ist
    model.eval()
    
    results = {}
    
    for threshold in confidence_thresholds:
        logger.info(f"Evaluiere mit Konfidenz-Schwellenwert: {threshold:.2f}")
        
        # Setze Schwellenwert für das Modell
        model.confidence_threshold = threshold
        
        # Metriken für diesen Schwellenwert
        correct_main = 0
        correct_early = 0
        correct_total = 0
        total = 0
        early_exit_count = 0
        
        # Inferenzzeiten messen
        times_main_path = []
        times_early_path = []
        
        # Klassenweise Genauigkeiten
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Threshold={threshold:.2f}"):
                # Alles auf dem richtigen Gerät
                inputs, labels = inputs.to(device), labels.to(device)
                
                batch_size = labels.size(0)
                total += batch_size
                
                # Führe Early-Exit-Inferenz durch und messe Zeit
                start_time = time.time()
                outputs, early_exit_used = model(inputs, use_early_exit=True)
                inferenz_time = time.time() - start_time
                
                # Vorhersagen sammeln
                _, predicted = torch.max(outputs, 1)
                
                # Führe auch den kompletten Pfad aus, um die Genauigkeit beider Zweige zu messen
                # aber berücksichtige bei der Zeitmessung nur den tatsächlich benutzten Pfad
                with torch.no_grad():
                    # Hier deaktivieren wir den Early Exit, um den vollen Pfad zu erzwingen
                    full_output, _ = model(inputs, use_early_exit=False)
                    # Early Exit-Ausgabe separat berechnen
                    early_features = model.early_exit_pooling(model.block3(model.block2(model.block1(model.quant(inputs)))))
                    early_output = model.early_exit_classifier(early_features)
                
                _, main_predicted = torch.max(full_output, 1)
                _, early_predicted = torch.max(early_output, 1)
                
                # Zähle korrekte Vorhersagen für jeden Pfad
                correct_main += (main_predicted == labels).sum().item()
                correct_early += (early_predicted == labels).sum().item()
                
                # Zähle korrekte Vorhersagen für den tatsächlich benutzten Pfad
                correct_total += (predicted == labels).sum().item()
                
                # Zähle Early Exit-Nutzung
                if isinstance(early_exit_used, bool):
                    # Einzelnes Boolean für den ganzen Batch
                    early_exit_batch_count = batch_size if early_exit_used else 0
                else:
                    # Pro-Sample Booleans
                    early_exit_batch_count = early_exit_used.sum().item()
                
                early_exit_count += early_exit_batch_count
                
                # Sammle Inferenzzeiten basierend auf dem benutzten Pfad
                if early_exit_batch_count > 0:
                    times_early_path.append(inferenz_time)
                else:
                    times_main_path.append(inferenz_time)
                
                # Klassenweise Genauigkeiten des kombinierten Modells
                correct_mask = (predicted == labels)
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += correct_mask[i].item()
                    class_total[label] += 1
        
        # Berechne Metriken für diesen Schwellenwert
        early_exit_ratio = early_exit_count / total
        accuracy_main = 100.0 * correct_main / total
        accuracy_early = 100.0 * correct_early / total
        accuracy_combined = 100.0 * correct_total / total
        
        # Berechne durchschnittliche Inferenzzeiten
        avg_main_time = np.mean(times_main_path) * 1000 if times_main_path else 0  # in ms
        avg_early_time = np.mean(times_early_path) * 1000 if times_early_path else 0  # in ms
        
        # Berechne gewichtete durchschnittliche Inferenzzeit basierend auf der Nutzung
        if early_exit_ratio == 0:
            avg_time = avg_main_time
        elif early_exit_ratio == 1:
            avg_time = avg_early_time
        else:
            avg_time = (early_exit_ratio * avg_early_time + 
                       (1 - early_exit_ratio) * avg_main_time)
        
        # Berechne theoretische Geschwindigkeitssteigerung gegenüber Standardmodell
        if avg_main_time > 0:
            speedup = avg_main_time / avg_time if avg_time > 0 else 1.0
        else:
            speedup = 1.0
            
        # Berechne klassenweise Genauigkeiten
        class_accuracies = []
        for i in range(len(class_names)):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        # Sammle die Ergebnisse für diesen Schwellenwert
        threshold_results = {
            'threshold': threshold,
            'early_exit_ratio': early_exit_ratio,
            'accuracy_combined': accuracy_combined,
            'accuracy_main': accuracy_main,
            'accuracy_early': accuracy_early,
            'avg_inference_time_ms': avg_time,
            'avg_main_time_ms': avg_main_time,
            'avg_early_time_ms': avg_early_time,
            'speedup': speedup,
            'class_accuracies': class_accuracies
        }
        
        results[str(threshold)] = threshold_results
        
        # Ausgabe der Ergebnisse für diesen Schwellenwert
        logger.info(f"Schwellenwert {threshold:.2f}:")
        logger.info(f"  - Early Exit-Nutzung: {early_exit_ratio:.2%}")
        logger.info(f"  - Genauigkeit (kombiniert): {accuracy_combined:.2f}%")
        logger.info(f"  - Genauigkeit (Hauptpfad): {accuracy_main:.2f}%")
        logger.info(f"  - Genauigkeit (Early Exit): {accuracy_early:.2f}%")
        logger.info(f"  - Durchschnittliche Inferenzzeit: {avg_time:.2f} ms")
        logger.info(f"  - Geschwindigkeitssteigerung: {speedup:.2f}x")
    
    # Finde den besten Schwellenwert basierend auf einem Kompromiss zwischen 
    # Genauigkeit und Geschwindigkeit (höhere Werte sind besser)
    best_threshold = None
    best_score = -1
    
    for threshold, res in results.items():
        # Gewichte Genauigkeit höher als Geschwindigkeit (70:30)
        norm_accuracy = res['accuracy_combined'] / 100.0  # Normalisiere auf [0,1]
        combined_score = 0.7 * norm_accuracy + 0.3 * (res['speedup'] / max(r['speedup'] for r in results.values()))
        
        if combined_score > best_score:
            best_score = combined_score
            best_threshold = float(threshold)
    
    # Sammle zusammenfassende Statistiken
    summary = {
        'best_threshold': best_threshold,
        'best_threshold_results': results[str(best_threshold)],
        'thresholds_evaluated': list(map(float, results.keys())),
        'all_results': results
    }
    
    logger.info(f"\nBester Konfidenz-Schwellenwert: {best_threshold:.2f}")
    logger.info(f"  - Early Exit-Nutzung: {results[str(best_threshold)]['early_exit_ratio']:.2%}")
    logger.info(f"  - Genauigkeit: {results[str(best_threshold)]['accuracy_combined']:.2f}%")
    logger.info(f"  - Geschwindigkeitssteigerung: {results[str(best_threshold)]['speedup']:.2f}x")
    
    return summary

def plot_early_exit_evaluation(evaluation_results, output_path):
    """
    Erstellt eine Visualisierung der Early-Exit-Evaluationsergebnisse.
    
    Args:
        evaluation_results: Dictionary mit Evaluationsergebnissen
        output_path: Pfad zum Speichern der Visualisierung
    """
    thresholds = evaluation_results['thresholds_evaluated']
    
    # Extrahiere Metriken für jeden Schwellenwert
    early_exit_ratios = [evaluation_results['all_results'][str(t)]['early_exit_ratio'] for t in thresholds]
    accuracies = [evaluation_results['all_results'][str(t)]['accuracy_combined'] for t in thresholds]
    speedups = [evaluation_results['all_results'][str(t)]['speedup'] for t in thresholds]
    
    plt.figure(figsize=(14, 10))
    
    # Diagramm 1: Early Exit-Nutzung vs. Schwellenwert
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, [ratio * 100 for ratio in early_exit_ratios], 'bo-')
    plt.title('Early Exit-Nutzung')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Early Exit Nutzung (%)')
    plt.grid(True, alpha=0.3)
    
    # Diagramm 2: Genauigkeit vs. Schwellenwert
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, accuracies, 'go-')
    plt.title('Modellgenauigkeit')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Genauigkeit (%)')
    plt.grid(True, alpha=0.3)
    
    # Diagramm 3: Geschwindigkeitssteigerung vs. Schwellenwert
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, speedups, 'ro-')
    plt.title('Geschwindigkeitssteigerung')
    plt.xlabel('Konfidenz-Schwellenwert')
    plt.ylabel('Speedup (x-fach)')
    plt.grid(True, alpha=0.3)
    
    # Diagramm 4: Zusammenfassung
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    best_threshold = evaluation_results['best_threshold']
    best_results = evaluation_results['best_threshold_results']
    
    plt.text(0.05, 0.95, 'Early Exit-Evaluierung Zusammenfassung', fontsize=14, fontweight='bold')
    plt.text(0.05, 0.85, f'Bester Konfidenz-Schwellenwert: {best_threshold:.2f}')
    plt.text(0.05, 0.80, f'Early Exit-Nutzung: {best_results["early_exit_ratio"]:.2%}')
    plt.text(0.05, 0.75, f'Genauigkeit (kombiniert): {best_results["accuracy_combined"]:.2f}%')
    plt.text(0.05, 0.70, f'Genauigkeit (Hauptpfad): {best_results["accuracy_main"]:.2f}%')
    plt.text(0.05, 0.65, f'Genauigkeit (Early Exit): {best_results["accuracy_early"]:.2f}%')
    plt.text(0.05, 0.60, f'Durchschnittliche Inferenzzeit: {best_results["avg_inference_time_ms"]:.2f} ms')
    plt.text(0.05, 0.55, f'Geschwindigkeitssteigerung: {best_results["speedup"]:.2f}x')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Early Exit-Evaluierung visualisiert und gespeichert unter {output_path}")

def plot_early_exit_trade_off(evaluation_results, output_path):
    """
    Erstellt ein Scatter-Plot, das den Trade-off zwischen Genauigkeit und 
    Geschwindigkeit für verschiedene Konfidenz-Schwellenwerte zeigt.
    
    Args:
        evaluation_results: Dictionary mit Evaluationsergebnissen
        output_path: Pfad zum Speichern der Visualisierung
    """
    thresholds = evaluation_results['thresholds_evaluated']
    
    # Extrahiere Metriken für jeden Schwellenwert
    accuracies = [evaluation_results['all_results'][str(t)]['accuracy_combined'] for t in thresholds]
    speedups = [evaluation_results['all_results'][str(t)]['speedup'] for t in thresholds]
    
    plt.figure(figsize=(10, 8))
    
    # Erstelle Scatter-Plot mit Genauigkeit vs. Geschwindigkeit
    scatter = plt.scatter(accuracies, speedups, c=thresholds, cmap='viridis', s=100, alpha=0.8)
    
    # Füge Farbbalken hinzu
    cbar = plt.colorbar(scatter)
    cbar.set_label('Konfidenz-Schwellenwert')
    
    # Beschrifte den besten Schwellenwert
    best_threshold = evaluation_results['best_threshold']
    best_accuracy = evaluation_results['best_threshold_results']['accuracy_combined']
    best_speedup = evaluation_results['best_threshold_results']['speedup']
    
    plt.annotate(f'Bester Schwellenwert: {best_threshold:.2f}',
                xy=(best_accuracy, best_speedup), xytext=(best_accuracy-5, best_speedup+0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Beschriftungen und Titel
    plt.title('Trade-off zwischen Genauigkeit und Geschwindigkeit')
    plt.xlabel('Genauigkeit (%)')
    plt.ylabel('Geschwindigkeitssteigerung (x-fach)')
    plt.grid(True, alpha=0.3)
    
    # Speichern der Visualisierung
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Trade-off-Analyse visualisiert und gespeichert unter {output_path}")

def convert_early_exit_model_to_quantized(qat_model, config, best_threshold=None):
    """
    Konvertiert ein Early-Exit-QAT-Modell in ein quantisiertes Modell für die Inferenz.
    
    Args:
        qat_model: Das QAT-trainierte Early-Exit-Modell
        config: Konfigurationsobjekt
        best_threshold: Optionaler bester Konfidenz-Schwellenwert für Early Exit
        
    Returns:
        Das quantisierte Modell für die Inferenz und Metadaten
    """
    logger.info("Konvertiere Early-Exit-QAT-Modell zu quantisiertem Inferenzmodell...")
    
    # Stelle sicher, dass das Modell im Eval-Modus ist
    qat_model.eval()
    
    # Setze besten Schwellenwert, falls angegeben
    if best_threshold is not None:
        qat_model.confidence_threshold = best_threshold
        logger.info(f"Konfidenz-Schwellenwert auf {best_threshold:.2f} gesetzt")
    
    # Verschiebe Modell auf CPU für Konvertierung
    cpu_model = qat_model.cpu()
    
    # Deaktiviere Observer, aktiviere Fake-Quantisierung
    cpu_model.apply(torch.quantization.disable_observer)
    cpu_model.apply(torch.quantization.enable_fake_quant)
    
    try:
        # In Early-Exit-Modellen ist die Konvertierung komplexer, da wir
        # Pfade mit dynamischer Logik haben. Für den RP2040 müssen wir die 
        # Kernkomponenten separat quantisieren
        
        # 1. Feature-Extraktor bis Block 3 (vor Early Exit)
        feature_extractor = nn.Sequential(
            cpu_model.quant,   # Eingabe-Quantisierungsstub
            cpu_model.block1,
            cpu_model.block2,
            cpu_model.block3
        )
        
        # 2. Early-Exit-Pfad
        early_exit_path = nn.Sequential(
            cpu_model.early_exit_pooling,
            cpu_model.early_exit_classifier,
            cpu_model.early_dequant  # Ausgabe-Dequantisierungsstub für Early Exit
        )
        
        # 3. Hauptpfad
        main_path = nn.Sequential(
            cpu_model.features,
            cpu_model.classifier,
            cpu_model.dequant  # Ausgabe-Dequantisierungsstub für Hauptpfad
        )
        
        # Konvertiere jeden Teilpfad separat
        feature_extractor_quantized = torch.quantization.convert(feature_extractor, inplace=False)
        early_exit_path_quantized = torch.quantization.convert(early_exit_path, inplace=False)
        main_path_quantized = torch.quantization.convert(main_path, inplace=False)
        
        # Speichere die quantisierten Teilkomponenten
        quantized_model_path = os.path.join(config.MODEL_DIR, "pizza_model_int8_early_exit.pth")
        torch.save({
            'feature_extractor': feature_extractor_quantized.state_dict(),
            'early_exit_path': early_exit_path_quantized.state_dict(),
            'main_path': main_path_quantized.state_dict(),
            'confidence_threshold': cpu_model.confidence_threshold
        }, quantized_model_path)
        
        # Schätze Modellgröße
        model_size_kb = os.path.getsize(quantized_model_path) / 1024
        
        # Wir behalten das original cpu_model als Rückgabewert bei,
        # da wir die Teilkomponenten nicht wieder zu einem funktionierenden 
        # PyTorch-Modell mit dynamischer Logik zusammensetzen können.
        # Bei der Bereitstellung auf dem RP2040 wird die Entscheidungslogik
        # separat implementiert werden müssen.
        
        logger.info(f"Quantisierte Early-Exit-Modellkomponenten gespeichert als: {quantized_model_path}")
        logger.info(f"Quantisierte Modellgröße: {model_size_kb:.2f} KB")
        logger.info(f"Hinweis: Für die RP2040-Bereitstellung müssen die Modellkomponenten separat geladen")
        logger.info(f"und die Early-Exit-Logik in der Inferenzsoftware implementiert werden.")
        
        return cpu_model, {
            'model_path': quantized_model_path,
            'model_size_kb': model_size_kb,
            'confidence_threshold': cpu_model.confidence_threshold,
            'quantization_method': 'Component-wise QAT with Early Exit',
            'components': ['feature_extractor', 'early_exit_path', 'main_path']
        }
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung des Early-Exit-Modells: {str(e)}")
        logger.info("Versuche manuelles Speichern des Modells...")
        
        # Manuelles Speichern als Fallback
        fallback_path = os.path.join(config.MODEL_DIR, "pizza_model_early_exit_fallback.pth")
        torch.save({
            'model_state_dict': cpu_model.state_dict(),
            'confidence_threshold': cpu_model.confidence_threshold
        }, fallback_path)
        
        return cpu_model, {
            'model_path': fallback_path,
            'confidence_threshold': cpu_model.confidence_threshold,
            'error': str(e),
            'quantization_method': 'QAT with Early Exit (Fallback)'
        }

def plot_early_exit_history(history, output_path):
    """
    Visualisiert die Trainingshistorie eines Early-Exit-Modells mit Metriken
    für beide Ausgabepfade.
    
    Args:
        history: Dictionary mit Trainingsmetriken
        output_path: Pfad zum Speichern der Visualisierung
    """
    plt.figure(figsize=(15, 10))
    
    # Genauigkeit - Hauptpfad vs. Early Exit
    plt.subplot(2, 3, 1)
    plt.plot(history['train_main_acc'], 'b-', label='Hauptpfad (Train)')
    plt.plot(history['val_main_acc'], 'b--', label='Hauptpfad (Val)')
    plt.plot(history['train_early_acc'], 'g-', label='Early Exit (Train)')
    plt.plot(history['val_early_acc'], 'g--', label='Early Exit (Val)')
    plt.title('Genauigkeit: Hauptpfad vs. Early Exit')
    plt.ylabel('Genauigkeit (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Genauigkeit - Kombiniert
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], 'b-', label='Training')
    plt.plot(history['val_acc'], 'r-', label='Validierung')
    plt.title('Kombinierte Modellgenauigkeit')
    plt.ylabel('Genauigkeit (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Verlust - Hauptpfad vs. Early Exit
    plt.subplot(2, 3, 3)
    plt.plot(history['train_main_loss'], 'b-', label='Hauptpfad (Train)')
    plt.plot(history['val_main_loss'], 'b--', label='Hauptpfad (Val)')
    plt.plot(history['train_early_loss'], 'g-', label='Early Exit (Train)')
    plt.plot(history['val_early_loss'], 'g--', label='Early Exit (Val)')
    plt.title('Verlust: Hauptpfad vs. Early Exit')
    plt.ylabel('Verlust')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Verlust - Kombiniert
    plt.subplot(2, 3, 4)
    plt.plot(history['train_loss'], 'b-', label='Training')
    plt.plot(history['val_loss'], 'r-', label='Validierung')
    plt.title('Kombinierter Modellverlust')
    plt.ylabel('Verlust')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(2, 3, 5)
    plt.plot(history['lr'], 'g-')
    plt.title('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    # Zusammenfassung
    plt.subplot(2, 3, 6)
    plt.axis('off')
    plt.text(0.05, 0.95, 'Early-Exit-Modell Training Zusammenfassung',
             fontsize=12, fontweight='bold')
    plt.text(0.05, 0.85, f'Finale Genauigkeit (kombiniert): {history["val_acc"][-1]:.2f}%')
    plt.text(0.05, 0.80, f'Finale Hauptpfad-Genauigkeit: {history["val_main_acc"][-1]:.2f}%')
    plt.text(0.05, 0.75, f'Finale Early-Exit-Genauigkeit: {history["val_early_acc"][-1]:.2f}%')
    plt.text(0.05, 0.65, f'Beste kombinierte Genauigkeit: {max(history["val_acc"]):.2f}%')
    plt.text(0.05, 0.60, f'Beste Hauptpfad-Genauigkeit: {max(history["val_main_acc"]):.2f}%')
    plt.text(0.05, 0.55, f'Beste Early-Exit-Genauigkeit: {max(history["val_early_acc"]):.2f}%')
    plt.text(0.05, 0.45, f'Trainingszeit: {history.get("training_time", 0):.2f} Sekunden')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Early-Exit-Trainingshistorie gespeichert unter {output_path}")

def convert_early_exit_model_to_quantized_std(qat_model, config, best_threshold=None):
    """
    Konvertiert die Komponenten eines Early-Exit-QAT-Modells für die Bereitstellung.
    Speichert quantisierte Komponenten und den Schwellenwert.
    """
    logger.info("Konvertiere Early-Exit-QAT-Modellkomponenten zu quantisierten Modulen...")
    
    qat_model.eval()
    cpu_model = qat_model.cpu()
    
    # Setze besten Schwellenwert im Modell
    if best_threshold is not None:
        cpu_model.confidence_threshold = best_threshold
        logger.info(f"Konfidenz-Schwellenwert auf {best_threshold:.2f} gesetzt für Metadaten.")
    
    # Deaktiviere Observer, aktiviere Fake-Quantisierung auf dem gesamten Modell VOR dem Extrahieren der Subgraphen
    # Dies stellt sicher, dass die Skalen und ZP für alle Teile korrekt sind
    cpu_model.apply(torch.quantization.disable_observer)
    cpu_model.apply(torch.quantization.enable_fake_quant)
    
    try:
        # Extrahieren und Konvertieren der Teilpfade
        # Wichtig: Die Stubs (quant, dequant, early_dequant) müssen am Anfang/Ende der Subgraphen platziert werden.
        
        # 1. Feature-Extraktor bis Block 3 (gemeinsamer Teil)
        # Endet mit FakeQuantize Modulen nach Block 3, die durch prepare_qat eingefügt wurden.
        # Diese Ausgabe wird die Eingabe für die nachfolgenden Pooling/Classifier Layer sein.
        # Wir brauchen hier keinen DeQuantStub, da die Ausgabe im FP32-Format (simuliert quantisiert) benötigt wird.
        feature_extractor = nn.Sequential(
             cpu_model.quant, # Eingabe-Quantisierungsstub des Gesamtmodells
             cpu_model.block1,
             cpu_model.block2,
             cpu_model.block3
             # prepare_qat hat hier FakeQuants eingefügt
        )
        
        # 2. Early-Exit-Pfad ab dem Pooling
        # Beginnt mit der Ausgabe von Block 3 (simuliert quantisiert), braucht also keinen QuantStub
        # Endet mit Early Classifier und early_dequant
        early_exit_path = nn.Sequential(
            cpu_model.early_exit_pooling,
            cpu_model.early_exit_classifier,
            cpu_model.early_dequant # Ausgabe-Dequantisierungsstub
        )
        
        # 3. Hauptpfad ab Features
        # Beginnt mit der Ausgabe von Block 3 (simuliert quantisiert), braucht also keinen QuantStub
        # Endet mit Haupt Classifier und dequant
        main_path = nn.Sequential(
            cpu_model.features,
            cpu_model.classifier,
            cpu_model.dequant # Ausgabe-Dequantisierungsstub
        )
        
        # Konvertiere jeden Teilpfad separat
        # Wichtig: calibrate=True ist hier unnötig, da es QAT-Modelle sind.
        # Wir wollen convert die Skalen/ZP aus den FakeQuants übernehmen lassen.
        feature_extractor_quantized = torch.quantization.convert(feature_extractor, inplace=False)
        early_exit_path_quantized = torch.quantization.convert(early_exit_path, inplace=False)
        main_path_quantized = torch.quantization.convert(main_path, inplace=False)
        
        # Speichere die quantisierten Teilkomponenten
        quantized_model_path = os.path.join(config.MODEL_DIR, "pizza_model_int8_early_exit_components.pth")
        torch.save({
            'feature_extractor_state_dict': feature_extractor_quantized.state_dict(),
            'early_exit_path_state_dict': early_exit_path_quantized.state_dict(),
            'main_path_state_dict': main_path_quantized.state_dict(),
            'confidence_threshold': cpu_model.confidence_threshold # Speichere den Schwellenwert als Metadaten
        }, quantized_model_path)
        
        # Schätze Modellgröße (Summe der Komponenten-State_dicts? Oder einfach die Gesamtdatei?)
        # Die Größe der Datei ist wahrscheinlich aussagekräftiger für den Speicherbedarf.
        model_size_kb = os.path.getsize(quantized_model_path) / 1024
        
        logger.info(f"Quantisierte Early-Exit-Modellkomponenten gespeichert als: {quantized_model_path}")
        logger.info(f"Quantisierte Modellgröße (Gesamtdatei): {model_size_kb:.2f} KB")
        logger.info(f"Hinweis: Für die RP2040-Bereitstellung müssen die Modellkomponenten separat geladen")
        logger.info(f"und die Early-Exit-Logik in der Inferenzsoftware implementiert werden.")
        
        # Rückgabe des ursprünglichen QAT-Modells (CPU) zur Simulation der quantisierten
        # Leistung auf dem Host in compare_models
        return cpu_model, {
            'model_path': quantized_model_path,
            'model_size_kb': model_size_kb,
            'confidence_threshold': cpu_model.confidence_threshold,
            'quantization_method': 'Component-wise QAT with Early Exit',
            'components_saved': True
        }
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung der Early-Exit-Modellkomponenten: {str(e)}")
        logger.info("Gibt das QAT-Modell (CPU) als Fallback zurück.")
        # Rückgabe des QAT-Modells auf CPU als Fallback für die Evaluierung
        return cpu_model, {
            'model_path': None, # Keine quantisierte Datei gespeichert
            'model_size_kb': os.path.getsize(os.path.join(config.MODEL_DIR, "early_exit_pizza_model_std.pth")) / 1024 if os.path.exists(os.path.join(config.MODEL_DIR, "early_exit_pizza_model_std.pth")) else 0,
            'confidence_threshold': qat_model.confidence_threshold,
            'error': str(e),
            'quantization_method': 'QAT with Early Exit (Conversion Failed)'
        }

def train_qat_model_std(model, train_loader, val_loader, config, class_names, model_name="qat_pizza_model_std"):
    """
    Trainiert das Standard-QAT-Modell (StdQAwareMicroPizzaNet).
    Verwendet den korrekten PyTorch QAT-Workflow.
    """
    logger.info(f"Starte Quantization-Aware Training für {model_name}...")
    
    # Modellpfad festlegen
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    
    # Parameter- und Speicherschätzungen (Optional, zur Info)
    try:
        params_count = model.count_parameters()
        memory_report = MemoryEstimator.check_memory_requirements(model, (3, config.IMG_SIZE, config.IMG_SIZE), config)
        logger.info(f"Modell hat {params_count:,} Parameter")
        logger.info(f"Geschätzte Speicheranforderungen: {memory_report['total_runtime_memory_kb']:.2f} KB")
    except Exception as e:
         logger.warning(f"Fehler bei Parameter-/Speicherschätzung: {e}")

    # Gewichteter Verlust für Klassenbalancierung
    class_counts = Counter()
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1
    
    num_samples = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = []
    for i in range(num_classes):
        class_weights.append(num_samples / (num_classes * max(class_counts[i], 1))) # avoid division by zero
    class_weights_tensor = torch.FloatTensor(class_weights).to(config.DEVICE)
    
    logger.info(f"Klassengewichte für Loss-Funktion: {[round(w, 2) for w in class_weights]}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
        pct_start=0.3,
    )
    
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # QAT Observer/FakeQuant Toggling
        if epoch >= 2: # Start Fake Quantization after a few epochs
             model.apply(torch.quantization.enable_fake_quant)
             logger.debug(f"Epoche {epoch+1}: Fake-Quantisierung aktiviert")
        
        if epoch % 4 == 0: # Enable Observer learning periodically
             model.apply(torch.quantization.enable_observer)
             logger.debug(f"Epoche {epoch+1}: Observer-Learning aktiviert")
        else:
             model.apply(torch.quantization.disable_observer)
             logger.debug(f"Epoche {epoch+1}: Observer-Learning deaktiviert")
             
        # Make sure FakeQuant is always enabled after epoch 2, even if observers are disabled
        if epoch >= 2:
             model.apply(torch.quantization.enable_fake_quant)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation Phase
        model.eval()
        # Ensure fake quant is enabled during validation for QAT evaluation
        model.apply(torch.quantization.enable_fake_quant)
        model.apply(torch.quantization.disable_observer) # Observers should not learn during eval

        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                correct_mask = (predicted == labels)
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += correct_mask[i].item()
                    class_total[label] += 1
                
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.0 * correct / total
                })
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        logger.info("Klassenweise Genauigkeiten:")
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100.0 * class_correct[i] / class_total[i]
                logger.info(f"  {class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                 logger.info(f"  {class_names[i]}: N/A (0/{class_total[i]})")

        early_stopping(epoch_val_loss, model)
        
        if early_stopping.early_stop:
            logger.info(f"Early Stopping in Epoche {epoch+1}")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training abgeschlossen in {training_time:.2f} Sekunden")
    
    if early_stopping.restore_weights(model):
        logger.info("Beste Modellgewichte wiederhergestellt")
    
    torch.save(model.state_dict(), model_path)
    logger.info(f"Modell gespeichert als: {model_path}")
    
    history['training_time'] = training_time
    return history, model