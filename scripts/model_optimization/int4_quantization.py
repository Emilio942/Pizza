#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INT4 Quantisierung für MicroPizzaNet
------------------------------------
Erweitert die vorhandene Quantisierung auf INT4 und fügt spezielle
Optimierungen für Modelle mit reduzierter Weight-Entropy durch Clustering.

Diese Implementierung bietet:
1. Post-Training INT4 Quantisierung
2. Optimierte Speicherabschätzung für INT4-Modelle
3. Exportfunktionen für RP2040 mit INT4-Unterstützung
4. Evaluierungsfunktionen für INT4-quantisierte Modelle
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import tensorflow as tf

# Projektpfad hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pizza_detector import MemoryEstimator, export_to_microcontroller
from scripts.quantization_aware_training import evaluate_quantized_model

logger = logging.getLogger(__name__)

class INT4Quantizer:
    """
    Implementiert die Quantisierung eines PyTorch-Modells auf INT4-Genauigkeit.
    Optimiert für Modelle, die bereits Gewichts-Clustering durchlaufen haben.
    """
    
    def __init__(self, model, calibration_loader=None):
        """
        Initialisiert den INT4-Quantisierer
        
        Args:
            model: PyTorch-Modell, das quantisiert werden soll
            calibration_loader: DataLoader für Kalibrierungsdaten (optional)
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Statistiken für Auswertung
        self.quantization_stats = {
            'original_model_size': MemoryEstimator.estimate_model_size(model),
            'quantized_model_size': 0,
            'compression_ratio': 0.0,
            'unique_values': {},
            'int4_layers': []
        }
        
    def _find_quantizable_layers(self):
        """Identifiziert Layer, die für INT4-Quantisierung in Frage kommen."""
        quantizable_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                quantizable_layers.append((name, module, 'weight'))
                
        return quantizable_layers
    
    def _pack_int4_weights(self, tensor):
        """
        Packt INT4-Gewichte (4-Bit) in INT8-Tensor (8-Bit)
        Zwei 4-Bit-Werte werden in einem 8-Bit-Wert gespeichert.
        
        Args:
            tensor: Original Float32-Tensor
            
        Returns:
            Tuple aus gepacktem INT8-Tensor, Skalierungsfaktor und Nullpunkt
        """
        # Bereite Tensor für Quantisierung vor
        tensor_np = tensor.detach().cpu().numpy()
        min_val = np.min(tensor_np)
        max_val = np.max(tensor_np)
        
        # Berechne Quantisierungsparameter für INT4 (-8 bis 7 für signed INT4)
        q_min, q_max = -8, 7
        scale = (max_val - min_val) / (q_max - q_min)
        zero_point = q_min - int(round(min_val / scale))
        
        # Quantisiere auf INT4-Bereich
        q_tensor = np.clip(np.round(tensor_np / scale) + zero_point, q_min, q_max).astype(np.int8)
        
        # Forme Tensor um, damit wir ihn packen können
        original_shape = q_tensor.shape
        flat_tensor = q_tensor.flatten()
        
        # Packe jeweils 2 INT4-Werte in einen INT8-Wert
        # Erster Wert in hohe 4 Bits, zweiter Wert in niedrige 4 Bits
        if len(flat_tensor) % 2 == 1:
            # Füge Padding hinzu, wenn ungerade Anzahl an Elementen
            flat_tensor = np.append(flat_tensor, q_min)
            
        packed_shape = (len(flat_tensor) // 2,)
        packed_tensor = np.zeros(packed_shape, dtype=np.int8)
        
        for i in range(0, len(flat_tensor), 2):
            high_bits = (flat_tensor[i] & 0xF) << 4
            low_bits = flat_tensor[i+1] & 0xF
            packed_tensor[i//2] = high_bits | low_bits
            
        return packed_tensor, original_shape, scale, zero_point
    
    def _unpack_int4_weights(self, packed_tensor, original_shape, scale, zero_point):
        """
        Entpackt INT4-Gewichte aus einem INT8-Tensor zurück in die Originalform.
        
        Args:
            packed_tensor: Gepackter INT8-Tensor
            original_shape: Originalform des Tensors
            scale: Skalierungsfaktor
            zero_point: Nullpunkt
            
        Returns:
            Entpackter Float32-Tensor in Originalform
        """
        # Entpacke INT4-Werte
        flat_size = np.prod(original_shape)
        unpacked = np.zeros(flat_size, dtype=np.int8)
        
        for i in range(len(packed_tensor)):
            # Extrahiere hohe und niedrige 4 Bits
            high_val = (packed_tensor[i] >> 4) & 0xF
            low_val = packed_tensor[i] & 0xF
            
            # Behandle negative Werte korrekt (signed INT4)
            if high_val > 7:
                high_val -= 16
            if low_val > 7:
                low_val -= 16
                
            idx = i * 2
            if idx < flat_size:
                unpacked[idx] = high_val
            if idx + 1 < flat_size:
                unpacked[idx + 1] = low_val
                
        # Dequantisierung zu Float
        unpacked_float = (unpacked.astype(np.float32) - zero_point) * scale
        return torch.from_numpy(unpacked_float.reshape(original_shape))
    
    def quantize(self):
        """
        Quantisiert das Modell auf INT4-Genauigkeit.
        
        Returns:
            Quantisiertes Modell und Quantisierungsstatistiken
        """
        logger.info("Starte INT4-Quantisierung des Modells...")
        
        # Finde quantisierbare Layer
        quantizable_layers = self._find_quantizable_layers()
        
        if not quantizable_layers:
            logger.warning("Keine quantisierbaren Layer gefunden!")
            return self.model, self.quantization_stats
            
        # INT4 Quantisierungsinformationen für spätere Wiederherstellung
        self.int4_info = {}
        
        # Für jeden quantisierbaren Layer
        for name, module, param_name in quantizable_layers:
            # Hole Parameter
            param = getattr(module, param_name)
            tensor = param.data
            
            # Zähle eindeutige Werte vor der Quantisierung
            unique_before = len(torch.unique(tensor))
            
            # Packe in INT4-Format (simuliert)
            packed_tensor, original_shape, scale, zero_point = self._pack_int4_weights(tensor)
            
            # Speichere Quantisierungsinformationen
            self.int4_info[name + '.' + param_name] = {
                'packed_tensor': packed_tensor,
                'original_shape': original_shape,
                'scale': scale,
                'zero_point': zero_point
            }
            
            # Entpacke für die weitere Verwendung (da PyTorch INT4 nicht nativ unterstützt)
            # Dies simuliert nur das Verhalten, spart aber nicht wirklich Speicher während der Laufzeit
            unpacked_tensor = self._unpack_int4_weights(
                packed_tensor, original_shape, scale, zero_point
            )
            
            # Setze quantisierten Tensor zurück ins Modell
            param.data = unpacked_tensor
            
            # Zähle eindeutige Werte nach der Quantisierung
            unique_after = len(torch.unique(unpacked_tensor))
            
            # Dokumentiere Layer-Statistiken
            memory_saving = tensor.numel() * 4 - (len(packed_tensor) * 8) / 8  # in Bytes
            self.quantization_stats['int4_layers'].append({
                'name': name,
                'param_name': param_name,
                'tensor_size': tensor.numel(),
                'unique_before': unique_before,
                'unique_after': unique_after,
                'memory_saving_bytes': memory_saving,
                'scale': float(scale),
                'zero_point': int(zero_point)
            })
            
            # Log-Information
            logger.info(f"  Layer {name}: Quantisiert auf INT4. Eindeutige Werte: {unique_before} -> {unique_after}")
            logger.info(f"    Speichereinsparung: {memory_saving/1024:.2f} KB")
            
        # Berechne Modellgröße mit INT4-Gewichten
        self.quantization_stats['quantized_model_size'] = MemoryEstimator.estimate_model_size(
            self.model, custom_bits={'int4_layers': 4}
        )
        
        self.quantization_stats['compression_ratio'] = 1.0 - (
            self.quantization_stats['quantized_model_size'] / 
            self.quantization_stats['original_model_size']
        )
        
        logger.info(f"INT4-Quantisierung abgeschlossen.")
        logger.info(f"  Originalgröße: {self.quantization_stats['original_model_size']:.2f} KB")
        logger.info(f"  Quantisierte Größe: {self.quantization_stats['quantized_model_size']:.2f} KB")
        logger.info(f"  Kompressionsrate: {self.quantization_stats['compression_ratio']:.2%}")
        
        return self.model, self.quantization_stats
    
    def export_model(self, output_dir, config, class_names, preprocess_params):
        """
        Exportiert das INT4-quantisierte Modell für den RP2040-Mikrocontroller.
        
        Args:
            output_dir: Ausgabeverzeichnis
            config: Konfigurationsobjekt
            class_names: Klassennamen
            preprocess_params: Vorverarbeitungsparameter
            
        Returns:
            Exportinformationen
        """
        logger.info(f"Exportiere INT4-quantisiertes Modell nach {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Speichere INT4-Quantisierungsinformationen
        int4_info_path = os.path.join(output_dir, "int4_quantization_info.json")
        with open(int4_info_path, 'w') as f:
            # Konvertiere NumPy-Arrays zu Listen für JSON-Serialisierung
            int4_info_json = {}
            for key, value in self.int4_info.items():
                int4_info_json[key] = {
                    'packed_tensor': value['packed_tensor'].tolist(),
                    'original_shape': value['original_shape'],
                    'scale': float(value['scale']),
                    'zero_point': int(value['zero_point'])
                }
            json.dump(int4_info_json, f, indent=2)
            
        # Speichere Quantisierungsstatistiken
        stats_path = os.path.join(output_dir, "int4_quantization_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.quantization_stats, f, indent=2)
            
        # Verwende die normale Exportfunktion mit zusätzlichen Quantisierungsinformationen
        export_info = export_to_microcontroller(
            model=self.model,
            config=config,
            class_names=class_names,
            preprocess_params=preprocess_params,
            quantization_results={
                'type': 'int4',
                'int4_info_path': int4_info_path,
                'stats_path': stats_path
            }
        )
        
        return export_info


def quantize_model_to_int4(model, val_loader, config, class_names, output_dir):
    """
    Quantisiert ein Modell auf INT4 und evaluiert die Ergebnisse.
    
    Args:
        model: PyTorch-Modell für die Quantisierung
        val_loader: Validierungsdaten für die Evaluierung
        config: Konfigurationsobjekt
        class_names: Klassennamen
        output_dir: Ausgabeverzeichnis
        
    Returns:
        Dictionary mit Evaluierungsergebnissen
    """
    logger.info(f"Starte INT4-Quantisierung und Evaluierung...")
    
    # Backup des ursprünglichen Modells
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Evaluiere Basismodell
    logger.info("Evaluiere Original-Modell vor INT4-Quantisierung...")
    base_eval = evaluate_quantized_model(model, val_loader, config.DEVICE, class_names)
    
    # INT4 Quantisierung durchführen
    quantizer = INT4Quantizer(model, val_loader)
    quantized_model, quant_stats = quantizer.quantize()
    
    # Evaluiere quantisiertes Modell
    logger.info("Evaluiere INT4-quantisiertes Modell...")
    int4_eval = evaluate_quantized_model(quantized_model, val_loader, config.DEVICE, class_names)
    
    # Erstelle Ausgabeverzeichnis
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichere quantisiertes Modell
    int4_model_path = os.path.join(output_dir, "int4_model.pth")
    torch.save(quantized_model.state_dict(), int4_model_path)
    
    # Speichere Evaluierungsergebnisse
    results = {
        'original_model': {
            'accuracy': base_eval['accuracy'],
            'memory_kb': base_eval.get('memory_kb', 0),
            'inference_time_ms': base_eval.get('avg_inference_time_ms', 0)
        },
        'int4_model': {
            'accuracy': int4_eval['accuracy'],
            'memory_kb': int4_eval.get('memory_kb', 0),
            'inference_time_ms': int4_eval.get('avg_inference_time_ms', 0),
            'quantization_stats': quant_stats
        },
        'accuracy_diff': int4_eval['accuracy'] - base_eval['accuracy'],
        'memory_reduction': 1.0 - (int4_eval.get('memory_kb', 0) / max(base_eval.get('memory_kb', 1), 1)),
        'int4_model_path': int4_model_path
    }
    
    # Speichere Ergebnisse als JSON
    results_path = os.path.join(output_dir, "int4_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"INT4-Quantisierung abgeschlossen und gespeichert unter: {int4_model_path}")
    logger.info(f"Genauigkeit: Original {base_eval['accuracy']:.2f}% -> INT4 {int4_eval['accuracy']:.2f}%")
    logger.info(f"Speicherreduktion: {results['memory_reduction']:.2%}")
    
    return results
