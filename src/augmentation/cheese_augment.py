# src/augmentation/cheese_augment.py

import os
import random
from PIL import Image
import torch
from torchvision import transforms

class AddCheese(torch.nn.Module):
    """
    Eine Transformation, die prozedural generierte Käsebilder auf Pizza-Bilder legt.
    """
    def __init__(self, cheese_renders_path: str, p: float = 0.5):
        """
        Initialisiert die Transformation.

        Args:
            cheese_renders_path (str): Pfad zum Verzeichnis mit den vorab gerenderten Käsebildern.
            p (float): Wahrscheinlichkeit, mit der die Transformation angewendet wird.
        """
        super().__init__()
        self.cheese_renders_path = cheese_renders_path
        self.p = p
        self.cheese_images = self._load_cheese_images()
        if not self.cheese_images:
            raise ValueError(f"Keine Käsebilder im Verzeichnis gefunden: {cheese_renders_path}")

    def _load_cheese_images(self):
        """
        Lädt die Pfade aller Käsebilder im angegebenen Verzeichnis.
        """
        image_files = [os.path.join(self.cheese_renders_path, f) 
                       for f in os.listdir(self.cheese_renders_path) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
        return image_files

    def forward(self, img: Image.Image) -> Image.Image:
        """
        Wendet die Transformation auf das Bild an.

        Args:
            img (PIL.Image.Image): Das Eingabebild (Pizza).

        Returns:
            PIL.Image.Image: Das augmentierte Bild.
        """
        if random.random() < self.p:
            if not self.cheese_images:
                return img # Keine Käsebilder zum Hinzufügen

            # Wähle ein zufälliges Käsebild
            cheese_img_path = random.choice(self.cheese_images)
            cheese_img = Image.open(cheese_img_path).convert("RGBA")

            # Skaliere das Käsebild zufällig
            img_w, img_h = img.size
            cheese_w, cheese_h = cheese_img.size

            scale_factor = random.uniform(0.3, 0.8) # Käse soll einen Teil der Pizza bedecken
            new_cheese_w = int(img_w * scale_factor)
            new_cheese_h = int(img_h * scale_factor)
            
            # Behalte das Seitenverhältnis bei
            if cheese_w > cheese_h:
                new_cheese_h = int(new_cheese_w * (cheese_h / cheese_w))
            else:
                new_cheese_w = int(new_cheese_h * (cheese_w / cheese_h))

            cheese_img = cheese_img.resize((new_cheese_w, new_cheese_h), Image.LANCZOS)

            # Zufällige Position auf dem Pizza-Bild
            # Stelle sicher, dass der Käse nicht über den Rand hinausgeht
            max_x = img_w - new_cheese_w
            max_y = img_h - new_cheese_h
            
            if max_x < 0 or max_y < 0: # Käse ist größer als das Bild, zentrieren
                paste_x = (img_w - new_cheese_w) // 2
                paste_y = (img_h - new_cheese_h) // 2
            else:
                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)

            # Komponiere das Käsebild auf das Pizza-Bild
            # Das Pizza-Bild muss in RGBA konvertiert werden, um den Alpha-Kanal zu nutzen
            img = img.convert("RGBA")
            img.alpha_composite(cheese_img, (paste_x, paste_y))
            
            # Konvertiere zurück zum Originalmodus, falls es nicht RGBA war
            # (z.B. RGB, wenn das Modell RGB erwartet)
            return img.convert("RGB") if img.mode == "RGBA" else img
        else:
            return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cheese_renders_path='{self.cheese_renders_path}', p={self.p})"

if __name__ == '__main__':
    # Beispiel für die Verwendung
    # Erstelle ein Dummy-Pizza-Bild
    dummy_pizza = Image.new('RGB', (224, 224), color = 'brown')
    
    # Pfad zu den generierten Käsebildern
    cheese_path = '../../data/generated_cheese_renders/' # Angepasster Pfad für den Testlauf

    try:
        add_cheese_transform = AddCheese(cheese_renders_path=cheese_path, p=1.0)
        augmented_pizza = add_cheese_transform(dummy_pizza)
        augmented_pizza.save('temp_augmented_pizza.png')
        print("Augmentiertes Pizza-Bild wurde als 'temp_augmented_pizza.png' gespeichert.")
    except ValueError as e:
        print(f"Fehler: {e}")
        print("Stellen Sie sicher, dass Sie zuerst 'scripts/generate_cheese_renders.py' ausgeführt haben.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
