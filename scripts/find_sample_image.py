# scripts/find_sample_image.py

import os
import sys

def find_first_image(directory):
    """
    Durchsucht ein Verzeichnis rekursiv nach der ersten Bilddatei.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return os.path.join(root, file)
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        search_dir = "augmented_pizza" # Standardverzeichnis

    image_path = find_first_image(search_dir)
    if image_path:
        print(f"Gefundenes Bild: {image_path}")
    else:
        print(f"Kein Bild im Verzeichnis {search_dir} gefunden.")
