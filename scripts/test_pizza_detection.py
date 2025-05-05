#!/usr/bin/env python3
"""
Test-Skript für das Pizza-Erkennungsmodell.
Lädt das vortrainierte Modell und führt Inferenzen mit einigen Testbildern durch.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
import time
import argparse
from torchvision import transforms
import glob
import cv2

# Füge das Projekt-Root zum Pythonpfad hinzu, um Imports zu ermöglichen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import MicroPizzaNet
from src.constants import INPUT_SIZE, IMAGE_MEAN, IMAGE_STD
from src.types import InferenceResult

# Die Klassen aus der Modell-README, nicht aus constants.py
MODEL_CLASS_NAMES = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']

# Definiere Farben für alle Klassen
MODEL_CLASS_COLORS = {
    'basic': (0, 255, 0),       # Grün
    'burnt': (255, 0, 0),       # Rot
    'combined': (255, 165, 0),  # Orange
    'mixed': (128, 0, 128),     # Lila
    'progression': (0, 0, 255), # Blau
    'segment': (255, 255, 0)    # Gelb
}

def load_model(model_path, num_classes=len(MODEL_CLASS_NAMES), device=None):
    """Lädt das vortrainierte Modell."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Verwende Gerät: {device}")
    
    # Modell erstellen
    model = MicroPizzaNet(num_classes=num_classes)
    
    # Lade die Gewichte
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modell erfolgreich geladen: {model_path}")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        sys.exit(1)
    
    # Setze das Modell in den Evaluationsmodus
    model.eval()
    model.to(device)
    
    return model, device

def preprocess_image(image_path, img_size=INPUT_SIZE):
    """Verarbeitet ein Bild für die Inferenz vor."""
    # Lade das Bild
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Fehler beim Laden des Bildes {image_path}: {e}")
        return None, None
    
    # Definiere die Transformation
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    # Transformiere das Bild
    img_tensor = transform(img)
    
    return img_tensor, img

def find_test_images(data_dir, num_images=5):
    """Findet einige zufällige Testbilder aus dem Datensatz."""
    # Suche nach allen Bildern im Datenverzeichnis
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_paths.extend(glob.glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True))
    
    if not image_paths:
        print(f"Keine Bilder im Verzeichnis {data_dir} gefunden!")
        sys.exit(1)
    
    # Wähle zufällige Bilder aus
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))
    print(f"{len(selected_images)} Testbilder ausgewählt.")
    
    return selected_images

def perform_inference(model, img_tensor, device):
    """Führt die Inferenz mit dem Modell durch."""
    # Bereite das Bild für die Inferenz vor
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    # Deaktiviere den Gradienten für die Inferenz
    with torch.no_grad():
        start_time = time.time()
        
        # Führe die Vorhersage durch
        outputs = model(input_batch)
        
        # Berechne die Wahrscheinlichkeiten mit Softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Bestimme die vorhergesagte Klasse
        _, predicted = torch.max(outputs, 1)
        
        inference_time = (time.time() - start_time) * 1000  # ms
    
    # Konvertiere zu Python-Typen
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item()
    
    # Erstelle ein Dict für alle Klassenwahrscheinlichkeiten
    class_probs = {MODEL_CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    # Erstelle ein InferenceResult-Objekt
    result = InferenceResult(
        class_name=MODEL_CLASS_NAMES[predicted_class],
        confidence=confidence,
        probabilities=class_probs,
        prediction=predicted_class
    )
    
    return result, inference_time

def annotate_image(image, result, draw_confidence=True):
    """Fügt Erkennungsergebnisse zu einem Bild hinzu."""
    # Konvertiere PIL-Bild zu NumPy-Array (falls nötig)
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    annotated = image.copy()
    
    # Rahmenfarbe basierend auf Klasse
    color = MODEL_CLASS_COLORS[result.class_name]
    
    # Zeichne Rahmen
    height, width = image.shape[:2]
    thickness = max(2, int(min(height, width) / 200))
    cv2.rectangle(
        annotated,
        (0, 0),
        (width-1, height-1),
        color,
        thickness
    )
    
    # Textgröße basierend auf Bildgröße
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(font_scale * 2))
    
    # Klassenname
    text = result.class_name
    if draw_confidence:
        text += f' ({result.confidence:.1%})'
    
    # Texthintergrund
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (text_width + 10, text_height + 10),
        color,
        -1  # Gefüllt
    )
    
    # Text
    cv2.putText(
        annotated,
        text,
        (5, text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # Weiß
        font_thickness
    )
    
    return annotated

def plot_inference_result(image, result, output_path=None):
    """Visualisiert ein Inferenzergebnis mit Wahrscheinlichkeiten."""
    plt.figure(figsize=(12, 6))
    
    # Bild anzeigen
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Erkannt: {result.class_name}\nKonfidenz: {result.confidence:.2%}')
    plt.axis('off')
    
    # Balkendiagramm der Wahrscheinlichkeiten
    plt.subplot(1, 2, 2)
    classes = list(result.probabilities.keys())
    probs = list(result.probabilities.values())
    
    # Farben für Balkendiagramm
    colors = [MODEL_CLASS_COLORS[c] for c in classes]
    
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, probs, color=[np.array(c)/255 for c in colors])
    plt.yticks(y_pos, classes)
    plt.xlabel('Wahrscheinlichkeit')
    plt.title('Klassenwahrscheinlichkeiten')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisierung gespeichert unter {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(description='Test des Pizza-Erkennungsmodells')
    parser.add_argument('--model', default='models/micro_pizza_model.pth', help='Pfad zum vortrainierten Modell')
    parser.add_argument('--data', default='data/augmented', help='Verzeichnis mit Testbildern')
    parser.add_argument('--num-images', type=int, default=5, help='Anzahl der zu testenden Bilder')
    parser.add_argument('--output-dir', default='output/test_results', help='Ausgabeverzeichnis für Ergebnisse')
    args = parser.parse_args()
    
    # Erstelle Ausgabeverzeichnis falls nötig
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lade das Modell
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        sys.exit(1)
    
    model, device = load_model(model_path)
    
    # Finde Testbilder
    test_images = find_test_images(args.data, num_images=args.num_images)
    
    # Führe Inferenz für jedes Bild durch
    results = []
    for i, image_path in enumerate(test_images):
        print(f"\nBild {i+1}/{len(test_images)}: {image_path}")
        
        # Verarbeite das Bild vor
        img_tensor, original_img = preprocess_image(image_path)
        if img_tensor is None:
            continue
        
        # Führe die Inferenz durch
        result, inference_time = perform_inference(model, img_tensor, device)
        
        # Zeige Ergebnis
        print(f"Vorhersage: {result.class_name} mit {result.confidence*100:.2f}% Konfidenz")
        print(f"Inferenzzeit: {inference_time:.2f} ms")
        
        # Visualisiere und speichere das Ergebnis
        output_path = output_dir / f"result_{i+1}.png"
        
        # Annotiere das Bild
        annotated_img = annotate_image(original_img, result, draw_confidence=True)
        
        # Visualisiere das Ergebnis
        plot_inference_result(annotated_img, result, output_path)
        
        # Speichere Ergebnis für die Zusammenfassung
        results.append({
            'image_path': image_path,
            'predicted_class': result.class_name,
            'confidence': result.confidence,
            'inference_time': inference_time
        })
    
    # Zeige Zusammenfassung
    print("\n" + "="*50)
    print("ZUSAMMENFASSUNG DER ERGEBNISSE")
    print("="*50)
    print(f"Getestete Bilder: {len(results)}")
    
    # Berechne durchschnittliche Inferenzzeit
    avg_inference_time = sum(r['inference_time'] for r in results) / len(results)
    print(f"Durchschnittliche Inferenzzeit: {avg_inference_time:.2f} ms")
    
    # Zähle Vorhersagen pro Klasse
    class_counts = {}
    for r in results:
        if r['predicted_class'] not in class_counts:
            class_counts[r['predicted_class']] = 0
        class_counts[r['predicted_class']] += 1
    
    print("\nVorhersagen pro Klasse:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\nTest abgeschlossen. Alle Ergebnisse wurden gespeichert in:", output_dir)

if __name__ == "__main__":
    main()
