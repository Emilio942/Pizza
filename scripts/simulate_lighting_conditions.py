"""
Simulation der Pizza-Erkennung unter verschiedenen Lichtverhältnissen.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Füge das Stammverzeichnis zum Python-Pfad hinzu, damit die src-Module gefunden werden
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.emulator import RP2040Emulator
from src.pizza_detector import preprocess_image
from src.constants import CLASS_NAMES

# Simulationsparameter
LIGHT_CONDITIONS = [
    {"name": "Sehr dunkel", "mean": 50, "std": 20},
    {"name": "Innenbeleuchtung", "mean": 120, "std": 30},
    {"name": "Tageslicht", "mean": 180, "std": 40},
    {"name": "Überbelichtet", "mean": 220, "std": 25}
]

NUM_ITERATIONS = 50
OUTPUT_DIR = Path("/home/emilio/Documents/ai/pizza/output/simulations")

def simulate_image_with_lighting(light_condition, size=(48, 48)):
    """Generiert ein simuliertes Bild mit den gegebenen Lichtverhältnissen."""
    # Erstelle Basisbild
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * light_condition["mean"]
    
    # Füge Rauschen hinzu
    noise = np.random.normal(0, light_condition["std"], image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Simuliere einfache Pizza-Formen
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2 - 5
    
    # Kreisform für Pizza
    for i in range(size[0]):
        for j in range(size[1]):
            dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if dist < radius:
                # Pizza-Inneres (dunkler/röter)
                image[i, j, 0] = min(255, image[i, j, 0] * 1.2)  # Mehr Rot
                image[i, j, 1] = max(0, image[i, j, 1] * 0.8)    # Weniger Grün
                image[i, j, 2] = max(0, image[i, j, 2] * 0.7)    # Weniger Blau
    
    return image

def run_simulation():
    """Führt die Simulation für verschiedene Lichtverhältnisse durch."""
    emulator = RP2040Emulator()
    emulator.load_firmware({
        'path': 'test.bin',
        'total_size_bytes': 100 * 1024,
        'model_size_bytes': 50 * 1024,
        'ram_usage_bytes': 40 * 1024,
        'model_input_size': (48, 48)
    })
    
    # Ergebnisstruktur
    results = {
        "light_conditions": [],
        "accuracies": [],
        "confidence_scores": [],
        "inference_times": []
    }
    
    # Für jede Lichtbedingung
    for light in LIGHT_CONDITIONS:
        print(f"Simuliere {light['name']} Bedingungen...")
        
        correct_predictions = 0
        confidences = []
        times = []
        
        # Mehrere Iterationen
        for i in range(NUM_ITERATIONS):
            # Simuliere ein Bild
            image = simulate_image_with_lighting(light)
            
            # Simuliere Erkennung
            start_time = time.time()
            result = emulator.simulate_inference(image)
            times.append(time.time() - start_time)
            
            # Auswerten des Ergebnisses (simuliert)
            # Für Simulationszwecke: "Pizza" hat höhere Chance bei mittleren Lichtverh.
            light_quality = 1.0 - abs(light["mean"] - 150) / 150  # 1.0 bei perfektem Licht (150)
            correct_class_prob = 0.5 + (light_quality * 0.4)  # 50%-90% je nach Lichtqualität
            
            # Simuliertes Ergebnis basierend auf Lichtqualität
            predicted_class = 0 if np.random.random() < correct_class_prob else np.random.randint(1, len(CLASS_NAMES))
            confidence = 0.5 + (light_quality * 0.4) + np.random.normal(0, 0.05)
            confidence = max(0.1, min(0.99, confidence))
            
            result["class_id"] = predicted_class
            result["confidence"] = confidence
            
            # Pizza (Klasse 0) ist die "richtige" Klasse für unsere Simulation
            if predicted_class == 0:
                correct_predictions += 1
            
            confidences.append(confidence)
        
        # Speichere Ergebnisse
        accuracy = correct_predictions / NUM_ITERATIONS
        avg_confidence = sum(confidences) / len(confidences)
        avg_time = sum(times) / len(times)
        
        results["light_conditions"].append(light["name"])
        results["accuracies"].append(accuracy)
        results["confidence_scores"].append(avg_confidence)
        results["inference_times"].append(avg_time)
        
        print(f"  Genauigkeit: {accuracy:.2f}, Konfidenz: {avg_confidence:.2f}, Zeit: {avg_time*1000:.1f}ms")
    
    return results

def plot_results(results):
    """Erstellt Diagramme aus den Simulationsergebnissen."""
    plt.figure(figsize=(14, 8))
    
    # Plot für Genauigkeit
    plt.subplot(1, 3, 1)
    plt.bar(results["light_conditions"], results["accuracies"], color='blue')
    plt.title('Erkennungsgenauigkeit bei\nverschiedenen Lichtverhältnissen')
    plt.ylabel('Genauigkeit')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    # Plot für Konfidenz
    plt.subplot(1, 3, 2)
    plt.bar(results["light_conditions"], results["confidence_scores"], color='green')
    plt.title('Durchschnittliche Konfidenz bei\nverschiedenen Lichtverhältnissen')
    plt.ylabel('Konfidenz')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    # Plot für Inferenzzeit
    plt.subplot(1, 3, 3)
    plt.bar(results["light_conditions"], [t * 1000 for t in results["inference_times"]], color='red')
    plt.title('Inferenzzeit bei\nverschiedenen Lichtverhältnissen')
    plt.ylabel('Zeit (ms)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Speichere die Ergebnisse
    output_path = OUTPUT_DIR / "light_conditions_simulation.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    # Stelle sicher, dass der Ausgabeordner existiert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starte Simulation für Pizza-Erkennung unter verschiedenen Lichtverhältnissen...")
    results = run_simulation()
    
    output_path = plot_results(results)
    print(f"Simulationsergebnisse gespeichert unter: {output_path}")