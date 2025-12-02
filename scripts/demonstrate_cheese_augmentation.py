# scripts/demonstrate_cheese_augmentation.py

import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import importlib.util

# Füge das Hauptverzeichnis zum Python-Pfad hinzu, um die src-Module zu finden
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dynamisches Laden des Moduls pizza_baking_detection_final
# Dies umgeht Probleme mit dem Python-Pfad und der Modulauflösung
module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'pizza_baking_detection_final.py')
spec = importlib.util.spec_from_file_location("pizza_baking_detection_final", module_path)
pizza_baking_detection_final = importlib.util.module_from_spec(spec)
sys.modules["pizza_baking_detection_final"] = pizza_baking_detection_final
spec.loader.exec_module(pizza_baking_detection_final)

# Importiere die benötigten Klassen aus dem dynamisch geladenen Modul
RP2040Config = pizza_baking_detection_final.RP2040Config
AddCheese = pizza_baking_detection_final.AddCheese # AddCheese ist jetzt in temp_pizza_baking_detection_final.py importiert

def demonstrate_augmentation(sample_image_path: str, cheese_render_path: str, output_dir: str = "temp_demonstrations"):
    """
    Demonstriert die Käse-Augmentierung visuell.

    Args:
        sample_image_path (str): Pfad zu einem Beispiel-Pizza-Bild.
        cheese_render_path (str): Pfad zu einem gerenderten Käsebild.
        output_dir (str): Verzeichnis, in dem die Demonstrationsbilder gespeichert werden.
    """
    print(f"Starte Demonstration der Käse-Augmentierung...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Konfiguration initialisieren
    config = RP2040Config()
    
    # Lade das Beispiel-Pizza-Bild
    original_pizza_img = Image.open(sample_image_path).convert("RGB")
    print(f"Original-Pizza-Bild geladen: {sample_image_path}")

    # Lade ein Beispiel-Käse-Render
    cheese_render_img = Image.open(cheese_render_path).convert("RGBA")
    print(f"Käse-Render geladen: {cheese_render_path}")
    cheese_render_img.save(os.path.join(output_dir, "01_cheese_render.png"))
    print(f"Gespeichert: {os.path.join(output_dir, '01_cheese_render.png')}")

    # Definiere Normalisierungsparameter (könnten auch aus PizzaDatasetAnalysis kommen)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transform-Pipeline ohne Käse (für das "Vorher"-Bild)
    base_transform_pipeline = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Transform-Pipeline mit Käse (für das "Nachher"-Bild)
    # Hier verwenden wir eine AddCheese-Instanz, die nur das spezifische Käsebild verwendet
    # und es immer anwendet (p=1.0)
    add_cheese_specific = AddCheese(cheese_renders_path=os.path.dirname(cheese_render_path), p=1.0)
    # Um sicherzustellen, dass immer das gleiche Käsebild verwendet wird, müsste AddCheese angepasst werden
    # Für diese Demo nehmen wir an, dass es ein zufälliges aus dem Pfad wählt, was für die Demonstration ok ist.
    # Wenn wir das exakte Käsebild wollen, müssten wir AddCheese eine Option geben, ein spezifisches Bild zu laden.
    # Für jetzt lassen wir es zufällig, aber wir zeigen das geladene Käsebild separat.

    augmented_transform_pipeline = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        add_cheese_specific, # Fügt den Käse hinzu
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Wende die Basis-Transformation an (Pizza ohne Käse, aber skaliert und normalisiert)
    pizza_without_cheese_tensor = base_transform_pipeline(original_pizza_img)
    pizza_without_cheese_img = transforms.ToPILImage()(
        torch.clamp(pizza_without_cheese_tensor * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1), 0, 1)
    )
    pizza_without_cheese_img.save(os.path.join(output_dir, "02_pizza_without_cheese.png"))
    print(f"Gespeichert: {os.path.join(output_dir, '02_pizza_without_cheese.png')}")

    # Wende die augmentierte Transformation an (Pizza mit Käse)
    pizza_with_cheese_tensor = augmented_transform_pipeline(original_pizza_img)
    pizza_with_cheese_img = transforms.ToPILImage()(
        torch.clamp(pizza_with_cheese_tensor * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1), 0, 1)
    )
    pizza_with_cheese_img.save(os.path.join(output_dir, "03_pizza_with_cheese.png"))
    print(f"Gespeichert: {os.path.join(output_dir, '03_pizza_with_cheese.png')}")

    print(f"\nDemonstration abgeschlossen. Bilder im Verzeichnis '{output_dir}' gespeichert.")
    print("Bitte überprüfen Sie die Bilder visuell.")

if __name__ == "__main__":
    # Beispielpfade (müssen existieren)
    sample_pizza_path = "augmented_pizza/lighting_perspective_test/augmentation_preview.png"
    sample_cheese_render_path = "data/generated_cheese_renders/cheese_00000.png"
    
    # Erstelle das temporäre Verzeichnis, falls es nicht existiert
    if not os.path.exists("temp_demonstrations"):
        os.makedirs("temp_demonstrations")

    demonstrate_augmentation(sample_pizza_path, sample_cheese_render_path)
