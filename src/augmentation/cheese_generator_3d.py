# src/augmentation/cheese_generator_3d.py

import trimesh
import numpy as np

class CheeseGenerator3D:
    """
    Generiert prozedurale 3D-Modelle von Käsestrukturen.
    """
    def __init__(self, num_vertices=1000, seed=None):
        """
        Initialisiert den Generator.

        Args:
            num_vertices (int): Die ungefähre Anzahl der Vertices für das Mesh.
            seed (int, optional): Seed für die Zufallszahlengenerierung.
        """
        self.num_vertices = num_vertices
        self.rng = np.random.default_rng(seed)

    def create_melted_cheese_blob(self, blobiness=0.5, complexity=5):
        """
        Erzeugt ein einzelnes, geschmolzenes Käse-Blob-Mesh.

        Args:
            blobiness (float): Wie unregelmäßig und "blob-artig" das Mesh ist. 
                               Werte zwischen 0.2 (glatter) and 1.0 (sehr unregelmäßig).
            complexity (int): Anzahl der Frequenzen für das prozedurale Rauschen.

        Returns:
            trimesh.Trimesh: Ein 3D-Mesh des Käse-Blobs.
        """
        # Erzeuge eine Kugel als Basisgeometrie
        mesh = trimesh.creation.icosphere(subdivisions=3)

        # Wende prozedurales Rauschen an, um die Oberfläche zu verformen
        vertices = mesh.vertices
        
        # Skaliere die Vertices basierend auf einem Perlin-ähnlichen Rauschen
        noise = np.zeros(len(vertices))
        frequency = 1.0
        amplitude = blobiness
        
        for i in range(complexity):
            noise += (self.rng.random(len(vertices)) * 2 - 1) * amplitude
            frequency *= 2
            amplitude /= 2

        # Normalisiere das Rauschen und wende es auf den Radius an
        scale = 1.0 + noise
        mesh.vertices *= scale[:, np.newaxis]

        # Füge eine zufällige Skalierung hinzu, um die Form zu variieren
        scale_factors = self.rng.uniform(0.8, 1.5, 3)
        scale_factors[2] *= 0.3 # Mache es flacher
        mesh.apply_scale(scale_factors)

        # Stelle sicher, dass das Mesh "wasserdicht" ist
        mesh.fill_holes()
        
        # Glätte das Mesh leicht, um scharfe Kanten zu entfernen
        trimesh.smoothing.filter_humphrey(mesh, iterations=5)

        # Setze die Farbe auf ein käsiges Gelb
        cheese_color = [255, 215, 0, 255]  # RGBA
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=cheese_color)

        return mesh

if __name__ == '__main__':
    # Beispiel für die Verwendung
    generator = CheeseGenerator3D()
    cheese_mesh = generator.create_melted_cheese_blob()

    # Zeige das Mesh an (erfordert eine Desktop-Umgebung)
    # cheese_mesh.show()

    # Speichere das Mesh als STL-Datei zur Überprüfung
    cheese_mesh.export('temp_cheese_blob.stl')
    print("Test-Käse-Mesh wurde als 'temp_cheese_blob.stl' gespeichert.")
