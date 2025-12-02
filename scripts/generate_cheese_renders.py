# scripts/generate_cheese_renders.py

import os
import sys
from tqdm import tqdm
import numpy as np

# Füge das Hauptverzeichnis zum Python-Pfad hinzu, um die src-Module zu finden
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentation.cheese_generator_3d import CheeseGenerator3D
from src.augmentation.renderer_2d import Renderer2D, get_random_camera_pose

def generate_renders(num_images: int, output_dir: str, image_size: tuple = (224, 224)):
    """
    Generiert und speichert eine bestimmte Anzahl von 2D-Käse-Renderings.

    Args:
        num_images (int): Die Anzahl der zu generierenden Bilder.
        output_dir (str): Das Verzeichnis, in dem die Bilder gespeichert werden sollen.
        image_size (tuple): Die Größe der Ausgabebilder.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Verzeichnis erstellt: {output_dir}")

    generator = CheeseGenerator3D()
    renderer = Renderer2D(image_size=image_size)

    print(f"Beginne mit der Generierung von {num_images} Käse-Renderings...")

    try:
        for i in tqdm(range(num_images), desc="Generiere Käse-Renderings"):
            # 1. Erzeuge ein zufälliges 3D-Käse-Mesh
            blobiness = np.random.uniform(0.4, 0.8)
            complexity = np.random.randint(4, 7)
            cheese_mesh = generator.create_melted_cheese_blob(
                blobiness=blobiness,
                complexity=complexity
            )

            # 2. Wähle eine zufällige Kameraposition und Lichtintensität
            camera_pose = get_random_camera_pose()
            light_intensity = np.random.uniform(5.0, 8.0)

            # 3. Rendere das Mesh
            rendered_image = renderer.render_mesh(
                mesh=cheese_mesh,
                camera_pose=camera_pose,
                light_intensity=light_intensity
            )

            # 4. Speichere das Bild
            output_path = os.path.join(output_dir, f"cheese_{i:05d}.png")
            rendered_image.save(output_path)

    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")
        print("Dies kann passieren, wenn die grafische Umgebung für das Offscreen-Rendering (EGL) nicht korrekt eingerichtet ist.")
        print("Stellen Sie sicher, dass alle Abhängigkeiten von 'pyrender' und 'trimesh' korrekt via 'pip install -r requirements.txt' installiert wurden.")
    
    finally:
        # Gib die Renderer-Ressourcen frei
        del renderer
        print(f"\nGenerierung abgeschlossen. {len(os.listdir(output_dir))} Bilder wurden in {output_dir} gespeichert.")


if __name__ == "__main__":
    NUM_IMAGES_TO_GENERATE = 100  # Klein halten für einen schnellen Test
    OUTPUT_DIRECTORY = "data/generated_cheese_renders/"
    
    generate_renders(NUM_IMAGES_TO_GENERATE, OUTPUT_DIRECTORY)
