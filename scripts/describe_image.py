# scripts/describe_image.py

from PIL import Image
import sys
import os

def describe_image(image_path):
    """
    Lädt ein Bild und gibt seine Eigenschaften aus.
    """
    try:
        with Image.open(image_path) as img:
            print(f"Bildpfad: {image_path}")
            print(f"Format: {img.format}")
            print(f"Modus: {img.mode}")
            print(f"Größe: {img.size} (Breite, Höhe)")
            if 'A' in img.mode:
                print("Enthält einen Alpha-Kanal (Transparenz).")
            else:
                print("Enthält keinen Alpha-Kanal (keine Transparenz).")
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {image_path}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        describe_image(image_file)
    else:
        print("Verwendung: python describe_image.py <bildpfad>")
