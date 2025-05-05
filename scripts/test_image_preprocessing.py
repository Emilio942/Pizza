#!/usr/bin/env python3
"""
Test-Skript für die CLAHE-Bildvorverarbeitung.
Demonstriert die Verbesserung der Bildqualität unter schwierigen Beleuchtungsbedingungen.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import argparse
import random
import time

# Füge das Projekt-Root zum Pythonpfad hinzu, um Imports zu ermöglichen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import INPUT_SIZE

def apply_clahe_opencv(image, clip_limit=4.0, grid_size=8):
    """
    Wendet CLAHE (Contrast Limited Adaptive Histogram Equalization) auf ein Bild an.
    Dies simuliert die on-device C-Implementierung mit OpenCV.
    """
    # Konvertiere zu BGR wenn es ein RGB-Bild ist
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Wende CLAHE auf jeden Kanal an
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # Wende CLAHE nur auf den Y-Kanal (Helligkeit) an
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        
        # Zurück zu RGB
        processed = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        # Für Graustufenbilder
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        processed = clahe.apply(image)
    
    return processed

def simulate_poor_lighting(image, mode='dark', intensity=0.5):
    """
    Simuliert schlechte Beleuchtungsbedingungen.
    
    Mode kann sein:
    - 'dark': Reduziert die Helligkeit
    - 'overexposed': Erhöht die Beleuchtung zu stark
    - 'uneven': Ungleichmäßige Beleuchtung
    - 'lowcontrast': Reduziert den Kontrast
    """
    image_float = image.astype(np.float32) / 255.0
    
    if mode == 'dark':
        # Verdunkle das Bild
        darkened = image_float * (1.0 - intensity)
        return (darkened * 255).astype(np.uint8)
    
    elif mode == 'overexposed':
        # Überbelichte das Bild
        overexposed = image_float * (1.0 + intensity)
        return np.clip(overexposed * 255, 0, 255).astype(np.uint8)
    
    elif mode == 'uneven':
        # Erzeuge ungleichmäßige Beleuchtung mit einem Gradienten
        height, width = image.shape[:2]
        gradient = np.zeros((height, width), dtype=np.float32)
        center_x, center_y = width // 2, height // 2
        
        for y in range(height):
            for x in range(width):
                # Radialer Gradient vom Zentrum
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                gradient[y, x] = 1.0 - (dist / max_dist) * intensity
        
        # Wende den Gradienten auf jeden Kanal an
        if len(image.shape) == 3:
            gradient = np.stack([gradient] * 3, axis=2)
        
        uneven = image_float * gradient
        return (uneven * 255).astype(np.uint8)
    
    elif mode == 'lowcontrast':
        # Reduziere den Kontrast
        mean = np.mean(image_float)
        lowcontrast = (image_float - mean) * (1.0 - intensity) + mean
        return (lowcontrast * 255).astype(np.uint8)
    
    else:
        return image

def find_test_images(data_dir, num_images=5):
    """Findet einige zufällige Testbilder aus dem Datensatz."""
    # Suche nach allen Bildern im Datenverzeichnis
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_paths.extend(list(Path(data_dir).glob(f'**/*{ext}')))
    
    if not image_paths:
        print(f"Keine Bilder im Verzeichnis {data_dir} gefunden!")
        sys.exit(1)
    
    # Wähle zufällige Bilder aus
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))
    print(f"{len(selected_images)} Testbilder ausgewählt.")
    
    return selected_images

def test_clahe_preprocessing(image_paths, output_dir, modes=None):
    """Testet die CLAHE-Vorverarbeitung mit verschiedenen simulierten Beleuchtungsproblemen."""
    if modes is None:
        modes = ['dark', 'overexposed', 'uneven', 'lowcontrast']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Für jedes Testbild
    for i, image_path in enumerate(image_paths):
        # Lade das Originalbild
        try:
            img = np.array(Image.open(image_path).convert('RGB'))
        except Exception as e:
            print(f"Fehler beim Laden von {image_path}: {e}")
            continue
        
        # Erstelle eine Unterverzeichnis für jedes Bild
        img_dir = output_dir / f"img_{i+1}"
        img_dir.mkdir(exist_ok=True)
        
        # Speichere das Original
        Image.fromarray(img).save(img_dir / "original.jpg")
        
        # Erstelle eine Collage für jedes Beleuchtungsproblem
        for mode in modes:
            plt.figure(figsize=(15, 5))
            
            # Original
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Original")
            plt.axis('off')
            
            # Simuliertes Beleuchtungsproblem
            degraded = simulate_poor_lighting(img, mode=mode)
            plt.subplot(1, 3, 2)
            plt.imshow(degraded)
            plt.title(f"Simuliertes Problem: {mode}")
            plt.axis('off')
            
            # Korrigiert mit CLAHE
            corrected = apply_clahe_opencv(degraded)
            plt.subplot(1, 3, 3)
            plt.imshow(corrected)
            plt.title("Mit CLAHE korrigiert")
            plt.axis('off')
            
            # Speichere die Collage
            plt.tight_layout()
            plt.savefig(img_dir / f"{mode}.jpg")
            plt.close()
            
            # Speichere auch die einzelnen Bilder für weitere Analysen
            Image.fromarray(degraded).save(img_dir / f"{mode}_degraded.jpg")
            Image.fromarray(corrected).save(img_dir / f"{mode}_corrected.jpg")
            
        print(f"Verarbeitung für {image_path.name} abgeschlossen.")
    
    print(f"Alle Tests abgeschlossen. Ergebnisse gespeichert in {output_dir}")

def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Test der on-device CLAHE-Bildvorverarbeitung')
    parser.add_argument('--data', default='data/classified', help='Verzeichnis mit Testbildern')
    parser.add_argument('--num-images', type=int, default=5, help='Anzahl der zu testenden Bilder')
    parser.add_argument('--output-dir', default='output/clahe_test', help='Ausgabeverzeichnis für Testergebnisse')
    args = parser.parse_args()
    
    # Finde einige Testbilder
    image_paths = find_test_images(args.data, args.num_images)
    
    # Führe den Test durch
    test_clahe_preprocessing(image_paths, args.output_dir)

if __name__ == "__main__":
    main()