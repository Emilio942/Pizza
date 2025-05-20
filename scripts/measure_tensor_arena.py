#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misst die tatsächliche Tensor-Arena-Größe für TFLite-Modelle

Dieses Skript führt eine detaillierte Analyse eines TFLite-Modells durch,
um den tatsächlichen Speicherbedarf während der Inferenz zu bestimmen.
Im Gegensatz zur einfachen prozentualen Schätzung (EMU-02) berechnet es
den exakten Speicherbedarf basierend auf der Modellarchitektur.

Verwendung:
    python scripts/measure_tensor_arena.py [--model MODEL_PATH] [--verbose]

Optionen:
    --model: Pfad zum Modell (Standard: models/pizza_model_int8.pth)
    --verbose: Gibt detaillierte Informationen aus
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
import logging
import json
import subprocess
import tempfile

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.pizza_detector import load_model, RP2040Config, export_to_microcontroller
    from src.emulation.emulator import RP2040Emulator
    from src.emulation.emulator import CameraEmulator
except ImportError:
    print("Fehler beim Importieren der Modellmodule. Stellen Sie sicher, dass Sie im richtigen Verzeichnis sind.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_tensor_memory_usage(model_path, verbose=False, input_size=None):
    """
    Berechnet den tatsächlichen Tensor-Arena-Speicherbedarf eines TFLite-Modells.
    
    Args:
        model_path (str): Pfad zum PyTorch-Modell
        verbose (bool): Ob detaillierte Informationen ausgegeben werden sollen
        input_size (int, optional): Bildgröße für Eingabe. Falls None, wird der Wert aus config verwendet.
        
    Returns:
        dict: Enthält tatsächlichen RAM-Bedarf und Berechnungsdetails
    """
    # 1. Lade das PyTorch-Modell
    logger.info(f"Lade Modell aus {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        return None
    
    # 2. Exportiere als TFLite (falls noch nicht geschehen)
    tflite_path = os.path.join(tempfile.gettempdir(), "temp_model.tflite")
    logger.info(f"Exportiere Modell nach TFLite: {tflite_path}")
    config = RP2040Config()
    
    img_size = input_size if input_size is not None else config.IMG_SIZE
    logger.info(f"Verwende Eingabebildgröße: {img_size}x{img_size}")
    
    try:
        # Exportiere das Modell nach TFLite
        result = export_to_microcontroller(model, config, input_shape=(3, img_size, img_size), 
                                          quantize=True, output_path=tflite_path)
        if not os.path.exists(tflite_path):
            logger.error(f"TFLite-Modell wurde nicht erstellt: {tflite_path}")
            return None
    except Exception as e:
        logger.error(f"Fehler beim Exportieren nach TFLite: {e}")
        return None
    
    # 3. Analysiere das TFLite-Modell für tatsächlichen Speicherbedarf
    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow nicht installiert. Installieren mit 'pip install tensorflow'")
        # Versuche, es mit der TFLite Interpreter-API direkt zu machen
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            logger.error("Weder TensorFlow noch tflite_runtime verfügbar")
            return None
    
    try:
        # Versuche zuerst mit TensorFlow, falls verfügbar
        if 'tf' in locals():
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
        else:
            # Alternativ mit tflite_runtime
            interpreter = Interpreter(model_path=tflite_path)
        
        # Allocate tensors - dies zeigt den tatsächlichen Speicherbedarf
        interpreter.allocate_tensors()
        
        # Hole Speichernutzungsinformationen
        memory_info = interpreter._get_tensor_details()
        
        # Berechne Speicherbedarf für jeden Tensor
        tensor_sizes = {}
        activation_memory = 0
        weights_memory = 0
        
        for tensor in memory_info:
            tensor_name = tensor['name']
            shape = tensor['shape']
            dtype = tensor['dtype']
            
            # Berechne Byte-Größe basierend auf Datentyp
            if dtype == np.float32:
                bytes_per_element = 4
            elif dtype in [np.int8, np.uint8]:
                bytes_per_element = 1
            elif dtype in [np.int16, np.uint16]:
                bytes_per_element = 2
            elif dtype in [np.int32, np.uint32]:
                bytes_per_element = 4
            else:
                bytes_per_element = 8  # Fallback
            
            # Berechne Gesamtgröße des Tensors
            size_bytes = np.prod(shape) * bytes_per_element
            
            # Kategorisiere in Gewichte oder Aktivierungen
            if 'const' in tensor_name.lower() or 'weight' in tensor_name.lower() or 'bias' in tensor_name.lower():
                weights_memory += size_bytes
            else:
                activation_memory += size_bytes
            
            tensor_sizes[tensor_name] = {
                'shape': shape.tolist() if hasattr(shape, 'tolist') else shape,
                'dtype': str(dtype),
                'size_bytes': int(size_bytes)
            }
        
        # Summiere die Tensoren für die Tensor-Arena-Größe
        tensor_arena_size = activation_memory
        
        # In der Praxis fügt der TFLite Interpreter einen Overhead hinzu
        # und benötigt etwas mehr Speicher für Verwaltungsstrukturen
        overhead_factor = 1.1  # 10% Overhead
        total_tensor_arena_size = int(tensor_arena_size * overhead_factor)
        
        # Erstelle detaillierten Bericht
        report = {
            'model_file_size_bytes': os.path.getsize(tflite_path),
            'weights_memory_bytes': int(weights_memory),
            'activation_memory_bytes': int(activation_memory),
            'tensor_arena_size_bytes': int(tensor_arena_size),
            'estimated_total_arena_size_bytes': total_tensor_arena_size,
            'tensor_details': tensor_sizes if verbose else None
        }
        
        # Gib Speicherplatz frei
        del interpreter
        
        # Vergleiche mit der EMU-02 Schätzung
        model_size_kb = os.path.getsize(tflite_path) / 1024
        emu02_estimated_ram = int(model_size_kb * 0.2 * 1024)  # 20% der Modellgröße für INT8
        
        report['emu02_estimated_ram_bytes'] = emu02_estimated_ram
        report['difference_bytes'] = total_tensor_arena_size - emu02_estimated_ram
        report['difference_percentage'] = ((total_tensor_arena_size - emu02_estimated_ram) / emu02_estimated_ram) * 100
        
        return report
    
    except Exception as e:
        logger.error(f"Fehler bei der Analyse des TFLite-Modells: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Lösche temporäre Datei
        if os.path.exists(tflite_path):
            os.remove(tflite_path)

def generate_report(report, output_dir='output/ram_analysis'):
    """Erstellt einen strukturierten Bericht über die Tensor-Arena-Größe"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, 'tensor_arena_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Bericht gespeichert unter: {report_path}")
    
    # Gib eine Zusammenfassung aus
    print("\n" + "=" * 80)
    print("TENSOR-ARENA-ANALYSE ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"Modellgröße: {report['model_file_size_bytes']/1024:.2f} KB")
    print(f"Gewichte-Speicher: {report['weights_memory_bytes']/1024:.2f} KB")
    print(f"Aktivierungs-Speicher: {report['activation_memory_bytes']/1024:.2f} KB")
    print(f"Tensor-Arena-Größe: {report['tensor_arena_size_bytes']/1024:.2f} KB")
    print(f"Geschätzte Gesamtgröße mit Overhead: {report['estimated_total_arena_size_bytes']/1024:.2f} KB")
    print(f"EMU-02 Schätzung (20% der Modellgröße): {report['emu02_estimated_ram_bytes']/1024:.2f} KB")
    print(f"Unterschied: {report['difference_bytes']/1024:.2f} KB ({report['difference_percentage']:.2f}%)")
    
    if abs(report['difference_percentage']) > 5:
        print("\nANALYSE: Die EMU-02 Schätzung weicht um mehr als 5% vom tatsächlichen Wert ab.")
        print("EMPFEHLUNG: Die Berechnungsmethode in der Simulation sollte korrigiert werden.")
    else:
        print("\nANALYSE: Die EMU-02 Schätzung ist innerhalb der akzeptablen Toleranz von 5%.")
        print("EMPFEHLUNG: Keine Änderung an der Berechnungsmethode erforderlich.")
    
    return report_path

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Misst die tatsächliche Tensor-Arena-Größe für TFLite-Modelle")
    parser.add_argument("--model", type=str, default="models/pizza_model_int8.pth",
                        help="Pfad zum Modell (Standard: models/pizza_model_int8.pth)")
    parser.add_argument("--verbose", action="store_true",
                        help="Gibt detaillierte Informationen aus")
    parser.add_argument("--input_size", type=int, default=None,
                        help="Eingabebildgröße (Standard: Wert aus constants.py)")
    
    args = parser.parse_args()
    
    logger.info(f"Starte Tensor-Arena-Messung für Modell: {args.model}")
    
    report = calculate_tensor_memory_usage(args.model, args.verbose, args.input_size)
    
    if report:
        generate_report(report)
    else:
        logger.error("Konnte keinen Bericht generieren. Siehe obige Fehler.")
        sys.exit(1)

if __name__ == "__main__":
    main()
