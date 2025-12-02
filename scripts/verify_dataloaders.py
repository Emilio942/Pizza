# scripts/verify_dataloaders.py

import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
from torchvision import transforms # Hinzugefügt: Fehlenden Import hinzufügen

# Dynamisches Laden des Moduls pizza_baking_detection_final
# Dies umgeht Probleme mit dem Python-Pfad und der Modulauflösung
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_pizza_baking_detection_final.py')

print(f"Attempting to load module from (temp): {module_path}")

if not os.path.exists(module_path):
    print(f"Error: Temporary file not found at the specified path: {module_path}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("pizza_baking_detection_final", module_path)
pizza_baking_detection_final = importlib.util.module_from_spec(spec)
sys.modules["pizza_baking_detection_final"] = pizza_baking_detection_final
spec.loader.exec_module(pizza_baking_detection_final)

# Importiere die benötigten Klassen aus dem dynamisch geladenen Modul
RP2040Config = pizza_baking_detection_final.RP2040Config
create_optimized_dataloaders = pizza_baking_detection_final.create_optimized_dataloaders


def verify_dataloaders():
    """
    Verifiziert die Funktionalität der Datenlader und der AddCheese-Transformation.
    """
    print("Starte Verifizierung der Datenlader...")

    # Konfiguration initialisieren
    config = RP2040Config(data_dir='augmented_pizza') # Verwende das Top-Level-Verzeichnis
    
    # Datenlader erstellen
    try:
        train_loader, val_loader, class_names, preprocessing_params = create_optimized_dataloaders(config)
        print(f"Datenlader erfolgreich erstellt. Klassen: {class_names}")
        print(f"Trainingsbilder: {len(train_loader.dataset)}, Validierungsbilder: {len(val_loader.dataset)}")

        # Überprüfe einige Batches des Trainingsladers
        print("\nIteriere durch Trainingsbatches zur Überprüfung der Augmentierung...")
        num_batches_to_check = 2
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i+1}: Inputs Shape={inputs.shape}, Labels Shape={labels.shape}")
            
            # Speichere ein paar Bilder zur visuellen Inspektion
            if i < 2: # Speichere Bilder aus den ersten 2 Batches
                for j in range(min(4, inputs.shape[0])): # Speichere bis zu 4 Bilder pro Batch
                    img_tensor = inputs[j].cpu()
                    
                    # Denormalisiere das Bild, um es korrekt anzuzeigen
                    mean = torch.tensor(preprocessing_params['mean']).view(3, 1, 1)
                    std = torch.tensor(preprocessing_params['std']).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1) # Stelle sicher, dass Werte im Bereich [0,1] liegen
                    
                    img_pil = transforms.ToPILImage()(img_tensor)
                    output_path = f"temp_augmented_image_batch{i+1}_sample{j+1}.png"
                    img_pil.save(output_path)
                    print(f"Gespeichertes augmentiertes Bild: {output_path}")

            if i >= num_batches_to_check - 1:
                break
        
        print("\nVerifizierung abgeschlossen. Überprüfen Sie die 'temp_augmented_image_*.png' Dateien.")

    except Exception as e:
        print(f"\nFehler bei der Verifizierung der Datenlader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataloaders()