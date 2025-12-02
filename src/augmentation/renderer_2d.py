# src/augmentation/renderer_2d.py

import os
# Setzt die Umgebungsvariable für Offscreen-Rendering, bevor pyrender importiert wird
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import trimesh
import numpy as np
from PIL import Image

class Renderer2D:
    """
    Rendert 3D-Trimesh-Objekte in 2D-Bilder mit Transparenz.
    """
    def __init__(self, image_size=(256, 256)):
        """
        Initialisiert den Renderer.

        Args:
            image_size (tuple): Die Größe des gerenderten Bildes (Breite, Höhe).
        """
        self.image_width, self.image_height = image_size
        self.renderer = pyrender.OffscreenRenderer(self.image_width, self.image_height)
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])

    def render_mesh(self, mesh: trimesh.Trimesh, camera_pose: np.ndarray, light_intensity=6.0):
        """
        Rendert ein einzelnes Mesh.

        Args:
            mesh (trimesh.Trimesh): Das zu rendernde 3D-Mesh.
            camera_pose (np.ndarray): 4x4-Transformationsmatrix für die Kameraposition.
            light_intensity (float): Die Intensität des gerichteten Lichts.

        Returns:
            PIL.Image.Image: Ein RGBA-Bild des gerenderten Objekts.
        """
        # Leere die Szene, um alte Objekte zu entfernen
        self.scene.clear()

        # Füge das Mesh zur Szene hinzu
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        self.scene.add(render_mesh)

        # Füge eine Kamera hinzu
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        self.scene.add(camera, pose=camera_pose)

        # Füge eine Lichtquelle hinzu
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
        self.scene.add(light, pose=camera_pose)

        # Rendere die Szene
        # Das Flag 'RGBA' ist entscheidend für den Alpha-Kanal
        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        
        return Image.fromarray(color, 'RGBA')

    def __del__(self):
        """
        Stellt sicher, dass der Renderer ordnungsgemäß freigegeben wird.
        """
        self.renderer.delete()

def get_random_camera_pose():
    """
    Erzeugt eine zufällige Kameraposition, die von oben auf das Objekt blickt.
    """
    # Zufälliger Winkel von oben
    angle = np.random.uniform(0, 2 * np.pi)
    # Leichte Neigung
    tilt = np.random.uniform(-np.pi / 8, np.pi / 8)
    
    eye = np.array([0, 0, 2.5])  # Kameraposition (über dem Objekt)
    target = np.array([0, 0, 0]) # Ziel (Objektzentrum)
    up = np.array([0, 1, 0])     # Oben-Vektor

    # Manuelle Implementierung von look_at
    f = (target - eye)
    f = f / np.linalg.norm(f)
    
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    M = np.eye(4)
    M[0:3, 0] = s
    M[0:3, 1] = u
    M[0:3, 2] = -f
    M[0:3, 3] = eye

    # Wende zufällige Rotationen an
    rotation_z = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    rotation_x = trimesh.transformations.rotation_matrix(tilt, [1, 0, 0])
    
    return rotation_z @ rotation_x @ M


if __name__ == '__main__':
    # Beispiel für die Verwendung
    from cheese_generator_3d import CheeseGenerator3D

    # 1. Erzeuge ein 3D-Mesh
    generator = CheeseGenerator3D()
    cheese_mesh = generator.create_melted_cheese_blob()

    # 2. Initialisiere den Renderer
    renderer = Renderer2D(image_size=(512, 512))

    # 3. Erzeuge eine zufällige Kameraposition
    camera_pose = get_random_camera_pose()

    # 4. Rendere das Mesh
    try:
        rendered_image = renderer.render_mesh(cheese_mesh, camera_pose)
        
        # 5. Speichere das Bild zur Überprüfung
        rendered_image.save('temp_cheese_render.png')
        print("Test-Rendering wurde als 'temp_cheese_render.png' gespeichert.")

    except Exception as e:
        print(f"Fehler beim Rendern. Dies kann passieren, wenn keine geeignete grafische Umgebung (wie EGL oder Xvfb) gefunden wird.")
        print(f"Stellen Sie sicher, dass die Abhängigkeiten von pyrender korrekt installiert sind.")
        print(f"Fehlermeldung: {e}")

    finally:
        # Renderer-Ressourcen freigeben
        del renderer
