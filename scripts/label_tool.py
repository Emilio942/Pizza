#!/usr/bin/env python3
"""
Pizza Labeling Tool - Erweiterte Version

Ein fortschrittliches Tool zum schnellen und konsistenten Labeln von Pizza-Bildern.
Bietet eine verbesserte Benutzeroberfläche mit Zoom, Histogramm, Batch-Processing,
KI-Vorschlägen und automatischem Git-Tracking.

Verwendung:
    python label_tool.py [--source-dir DIR] [--output-dir DIR] [--class-file FILE]

Optionen:
    --source-dir: Verzeichnis mit unklassifizierten Bildern (Standard: data/raw)
    --output-dir: Verzeichnis für sortierte Bilder (Standard: data/classified)
    --class-file: JSON-Datei mit Klassendefinitionen (Standard: data/class_definitions.json)
    --stats-file: JSON-Datei für Label-Statistiken (Standard: data/classified/classification_stats.json)
    --git-track: Versioniere Änderungen mit Git (Standard: True)
    --batch-size: Anzahl der gleichzeitig anzuzeigenden Bilder für Batch-Labeling (Standard: 4)
    --review-mode: Starte im Review-Modus für bereits klassifizierte Bilder (Standard: False)
    --model-predict: Verwende vortrainiertes Modell für Label-Vorschläge (Standard: True)
"""

import os
import json
import argparse
import shutil
import datetime
import glob
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, List, Tuple, Any, Optional, Union
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, StringVar, IntVar, BooleanVar
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
import threading
import random
import sys
from pathlib import Path

# Füge Projekt-Root zum Python-Pfad hinzu für Importe
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Versuche, das Pizza-Modell zu importieren (für KI-Vorschläge)
try:
    from src.pizza_detector import PizzaDetector
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Hinweis: Pizza-Modell nicht verfügbar. KI-Vorschläge deaktiviert.")

# Standard Pizza-Klassen (können durch eine JSON-Datei überschrieben werden)
DEFAULT_CLASSES = {
    "basic": {"description": "Roher Teig/Pizza-Grundzustand", "shortcut": "b", "color": "#5b9bd5"},
    "burnt": {"description": "Pizza ist verbrannt oder stark angebrannt", "shortcut": "v", "color": "#ed7d31"},
    "combined": {"description": "Kombinierte Pizza mit mehreren Zuständen", "shortcut": "c", "color": "#a5a5a5"},
    "mixed": {"description": "Mischzustand zwischen roh und fertig", "shortcut": "m", "color": "#ffc000"},
    "progression": {"description": "Prozessphase mit erkennbarem Übergang", "shortcut": "p", "color": "#70ad47"},
    "segment": {"description": "Einzelnes Pizzasegment oder Stück", "shortcut": "s", "color": "#4472c4"}
}

class PizzaLabeler:
    """Hauptklasse für das erweiterte Pizza Labeling Tool"""
    
    def __init__(self, 
                 source_dir: str = "data/raw", 
                 output_dir: str = "data/classified",
                 class_file: str = None,
                 stats_file: str = "data/classified/classification_stats.json",
                 git_track: bool = True,
                 batch_size: int = 4,
                 review_mode: bool = False,
                 model_predict: bool = True):
        """
        Initialisiert das Labeling Tool.
        
        Args:
            source_dir: Verzeichnis mit unklassifizierten Bildern
            output_dir: Verzeichnis für klassifizierte Bilder
            class_file: JSON-Datei mit Klassendefinitionen (optional)
            stats_file: JSON-Datei für Label-Statistiken
            git_track: Ob Änderungen mit Git verfolgt werden sollen
            batch_size: Anzahl der gleichzeitig anzuzeigenden Bilder für Batch-Labeling
            review_mode: Ob im Review-Modus gestartet werden soll
            model_predict: Ob das Modell für Vorhersagen verwendet werden soll
        """
        self.source_dir = os.path.abspath(source_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.stats_file = os.path.abspath(stats_file)
        self.git_track = git_track
        self.batch_size = max(1, min(batch_size, 6))  # Zwischen 1 und 6 begrenzen
        self.review_mode = review_mode
        self.model_predict = model_predict and MODEL_AVAILABLE
        
        # Lade Klassen aus JSON oder verwende Standard
        self.classes = self._load_classes(class_file)
        
        # Erstelle Verzeichnisse, falls nicht vorhanden
        self._create_directories()
        
        # Lade bisherige Statistiken oder initialisiere neue
        self.stats = self._load_stats()
        
        # Git Repository vorbereiten, falls aktiviert
        if self.git_track:
            self._setup_git_repository()
        
        # Suche nach Bildern je nach Modus
        if self.review_mode:
            self.image_files = self._get_classified_image_files()
            self.working_dir = self.output_dir
        else:
            self.image_files = self._get_image_files()
            self.working_dir = self.source_dir
            
        if not self.image_files:
            print(f"Keine Bilder gefunden.")
            
        # Initialisiere Pizza-Detektor, falls verfügbar
        self.detector = None
        if self.model_predict and MODEL_AVAILABLE:
            try:
                self.detector = PizzaDetector()
                print("Pizza-Detektor initialisiert für KI-Vorschläge.")
            except Exception as e:
                print(f"Fehler beim Initialisieren des Pizza-Detektors: {e}")
                self.model_predict = False
        
        # Initialisiere UI-Komponenten
        self.root = None
        self.current_image_label = None
        self.info_label = None
        self.buttons = []
        self.current_index = 0
        self.current_image_path = None
        self.current_image_obj = None
        self.is_fullscreen = False
        self.batch_mode_active = False
        self.batch_images = []
        self.batch_selections = []
        
        # Für die Bildvorschau
        self.preview_size = (600, 600)
        self.zoom_factor = 1.0
        self.contrast_factor = 1.0
        self.brightness_factor = 1.0
        
        # Für Bildannotationen und Metadaten
        self.annotations = {}
        self.metadata = {}
        self.predictions = {}
        
        # Für Histogramm und Analysen
        self.histogram_visible = False
        self.histogram_canvas = None
        
        # Für Keyboard-Shortcuts und Modi
        self.last_label = None  # Speichert das zuletzt verwendete Label
        self.auto_advance = True  # Automatisch zum nächsten Bild
    
    # ... bestehende Methoden beibehalten ...
    
    def _load_classes(self, class_file: str) -> Dict[str, Dict[str, str]]:
        """Lädt Klassendefinitionen aus einer JSON-Datei oder verwendet Standard"""
        if class_file and os.path.exists(class_file):
            try:
                with open(class_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Fehler beim Laden der Klassendefinitionen: {e}")
                print("Verwende Standard-Klassendefinitionen.")
        
        return DEFAULT_CLASSES
    
    def _create_directories(self):
        """Erstellt die notwendigen Verzeichnisse"""
        # Erstelle Hauptverzeichnisse
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Erstelle ein Verzeichnis für jede Klasse
        for class_name in self.classes.keys():
            class_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    def _load_stats(self) -> Dict[str, Any]:
        """Lädt bisherige Klassifizierungsstatistiken oder initialisiert neue"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Fehler beim Laden der Statistik: {e}")
                print("Initialisiere neue Statistik.")
        
        # Initialisiere neue Statistik
        stats = {
            "processed": 0,
            "classifications": {class_name: 0 for class_name in self.classes.keys()},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return stats
    
    def _get_image_files(self) -> List[str]:
        """Sucht nach Bildern im Quellverzeichnis"""
        extensions = ['jpg', 'jpeg', 'png']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.source_dir, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(self.source_dir, f"*.{ext.upper()}")))
        
        return sorted(image_files)
    
    def _get_classified_image_files(self) -> List[str]:
        """Sucht nach bereits klassifizierten Bildern (für Review-Modus)"""
        extensions = ['jpg', 'jpeg', 'png']
        image_files = []
        
        # Durchsuche jedes Klassenverzeichnis
        for class_name in self.classes.keys():
            class_dir = os.path.join(self.output_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(class_dir, f"*.{ext}")))
                image_files.extend(glob.glob(os.path.join(class_dir, f"*.{ext.upper()}")))
        
        return sorted(image_files)
    
    def _save_stats(self):
        """Speichert aktuelle Klassifizierungsstatistiken"""
        # Aktualisiere Zeitstempel
        self.stats["timestamp"] = datetime.datetime.now().isoformat()
        
        # Speichere in JSON-Datei
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _setup_git_repository(self):
        """Bereitet Git-Repository für Versionierung vor"""
        if not self.git_track:
            return
            
        try:
            # Prüfe, ob wir in einem Git-Repository sind
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Nicht in einem Git-Repository, initialisiere eines
                print("Kein Git-Repository gefunden. Initialisiere neues Repository...")
                project_dir = os.path.dirname(os.path.dirname(self.output_dir))
                subprocess.run(["git", "init", project_dir])
                
                # Erstelle .gitignore für temporäre Dateien
                gitignore_path = os.path.join(project_dir, ".gitignore")
                if not os.path.exists(gitignore_path):
                    with open(gitignore_path, "w") as f:
                        f.write("__pycache__/\n*.py[cod]\n*.so\n.DS_Store\n")
            
            # Erstelle README für Datensatz, falls nicht vorhanden
            readme_path = os.path.join(self.output_dir, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write("# Pizza-Datensatz\n\n")
                    f.write("Dieser Datensatz enthält klassifizierte Pizza-Bilder für das RP2040-Projekt.\n\n")
                    f.write("## Klassen\n\n")
                    for class_name, info in self.classes.items():
                        f.write(f"- **{class_name}**: {info['description']}\n")
                
                # Commit README
                subprocess.run(["git", "add", readme_path])
                subprocess.run(["git", "commit", "-m", "Initialisiere Datensatz-README"])
                
            print("Git-Repository bereit für Versionierung.")
        except Exception as e:
            print(f"Git: Fehler bei der Repository-Initialisierung - {e}")
            self.git_track = False
    
    def _git_commit_changes(self, image_path: str, label: str):
        """Führt Git-Commits für Änderungen durch"""
        if not self.git_track:
            return
        
        try:
            # Prüfe, ob wir in einem Git-Repository sind
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Nicht in einem Git-Repository, keine Versionierung möglich
                return
            
            # Erstelle sinnvolle Commit-Nachricht
            image_name = os.path.basename(image_path)
            commit_message = f"Label: {image_name} as '{label}'"
            
            # Füge Änderungen hinzu und committe
            subprocess.run(["git", "add", self.stats_file])
            subprocess.run(["git", "add", os.path.join(self.output_dir, label, os.path.basename(image_path))])
            subprocess.run(["git", "commit", "-m", commit_message])
            
            print(f"Git: Änderungen committed.")
        except Exception as e:
            print(f"Git: Fehler bei der Versionierung - {e}")
    
    def _git_pull_push(self):
        """Führt Git Pull und Push durch, um Änderungen zu synchronisieren"""
        if not self.git_track:
            return
            
        try:
            # Prüfe, ob Remote-Repository existiert
            result = subprocess.run(
                ["git", "remote"],
                capture_output=True,
                text=True
            )
            
            if "origin" in result.stdout:
                # Pull aktuelle Änderungen
                pull_result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    capture_output=True,
                    text=True
                )
                if pull_result.returncode == 0:
                    print("Git: Änderungen erfolgreich gepullt.")
                
                # Push unsere Änderungen
                push_result = subprocess.run(
                    ["git", "push", "origin", "main"],
                    capture_output=True,
                    text=True
                )
                if push_result.returncode == 0:
                    print("Git: Änderungen erfolgreich gepusht.")
                else:
                    print(f"Git: Push fehlgeschlagen - {push_result.stderr}")
            else:
                print("Git: Kein Remote-Repository konfiguriert.")
        except Exception as e:
            print(f"Git: Fehler bei Pull/Push - {e}")
    
    def _classify_image(self, image_path: str, label: str):
        """Klassifiziert ein Bild und aktualisiert die Statistiken"""
        if not os.path.exists(image_path):
            print(f"Fehler: Bild {image_path} existiert nicht mehr.")
            return False
        
        # Zielverzeichnis
        target_dir = os.path.join(self.output_dir, label)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        # Zieldatei
        file_name = os.path.basename(image_path)
        target_path = os.path.join(target_dir, file_name)
        
        # Kopiere oder verschiebe das Bild
        try:
            shutil.copy2(image_path, target_path)  # Kopieren mit Metadaten
            print(f"Bild {file_name} als '{label}' klassifiziert.")
            
            # Aktualisiere Statistiken
            self.stats["processed"] += 1
            self.stats["classifications"][label] += 1
            self._save_stats()
            
            # Versioniere die Änderungen mit Git
            self._git_commit_changes(image_path, label)
            
            return True
        except Exception as e:
            print(f"Fehler beim Klassifizieren von {file_name}: {e}")
            return False
    
    def _predict_label(self, image_path: str) -> Tuple[str, float]:
        """Verwendet das Modell, um ein Label vorherzusagen"""
        if not self.model_predict or not self.detector:
            return None, 0.0
        
        try:
            # Lade Bild
            img = Image.open(image_path)
            
            # Führe Vorhersage durch
            class_name, confidence = self.detector.predict_image(img)
            
            return class_name, confidence
        except Exception as e:
            print(f"Fehler bei der Vorhersage: {e}")
            return None, 0.0
    
    def _load_current_image(self):
        """Lädt das aktuelle Bild für die Anzeige"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return None, None
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Öffne das Bild
            img = Image.open(image_path)
            
            # Berechne Skalierung, um in Vorschau zu passen
            img.thumbnail(self.preview_size, Image.LANCZOS)
            
            return image_path, img
        except Exception as e:
            print(f"Fehler beim Laden des Bildes {image_path}: {e}")
            return image_path, None
    
    def _process_image(self, img: Image.Image) -> Image.Image:
        """Verarbeitet das Bild für die Anzeige (Kontrast, Helligkeit, etc.)"""
        if not img:
            return None
            
        try:
            # Anwenden von Bildverbesserungen
            if self.contrast_factor != 1.0:
                img = ImageEnhance.Contrast(img).enhance(self.contrast_factor)
                
            if self.brightness_factor != 1.0:
                img = ImageEnhance.Brightness(img).enhance(self.brightness_factor)
            
            # Zoom anwenden
            if self.zoom_factor != 1.0:
                width, height = img.size
                new_width = int(width * self.zoom_factor)
                new_height = int(height * self.zoom_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Zentrieren
                if new_width > self.preview_size[0] or new_height > self.preview_size[1]:
                    left = (new_width - self.preview_size[0]) // 2 if new_width > self.preview_size[0] else 0
                    top = (new_height - self.preview_size[1]) // 2 if new_height > self.preview_size[1] else 0
                    right = left + self.preview_size[0]
                    bottom = top + self.preview_size[1]
                    img = img.crop((left, top, right, bottom))
            
            return img
        except Exception as e:
            print(f"Fehler bei der Bildverarbeitung: {e}")
            return img
    
    def _create_histogram(self, img: Image.Image):
        """Erstellt ein Histogramm des Bildes"""
        if not img or not self.histogram_canvas:
            return
            
        try:
            # Konvertiere zu NumPy Array
            img_array = np.array(img.convert('L'))  # In Graustufen umwandeln
            
            # Lösche vorherige Histogramm-Figur
            self.histogram_canvas.figure.clear()
            ax = self.histogram_canvas.figure.add_subplot(111)
            
            # Erstelle Histogramm
            ax.hist(img_array.flatten(), bins=256, range=(0, 255), color='gray', alpha=0.7)
            ax.set_title("Histogramm")
            ax.set_xlabel("Pixel-Intensität")
            ax.set_ylabel("Häufigkeit")
            ax.grid(alpha=0.3)
            
            # Zeichne vertikale Linien für wichtige Werte
            ax.axvline(np.mean(img_array), color='blue', linestyle='dashed', linewidth=1)
            ax.axvline(np.median(img_array), color='red', linestyle='dashed', linewidth=1)
            
            # Aktualisiere Canvas
            self.histogram_canvas.draw()
        except Exception as e:
            print(f"Fehler beim Erstellen des Histogramms: {e}")
    
    def _update_display(self):
        """Aktualisiert die Bildanzeige und Informationsleiste"""
        self.current_image_path, img_obj = self._load_current_image()
        
        if img_obj:
            # Verarbeite das Bild
            processed_img = self._process_image(img_obj)
            self.current_image_obj = processed_img
            
            # Konvertiere für Tkinter
            photo = ImageTk.PhotoImage(processed_img)
            
            # Aktualisiere Bildanzeige
            self.current_image_label.config(image=photo)
            self.current_image_label.image = photo  # Halte Referenz
            
            # Aktualisiere Histogramm, falls sichtbar
            if self.histogram_visible:
                self._create_histogram(img_obj)
            
            # Mache KI-Vorhersage, falls aktiviert
            prediction = None
            confidence = 0.0
            if self.model_predict and self.detector:
                prediction, confidence = self._predict_label(self.current_image_path)
                self.predictions[self.current_image_path] = (prediction, confidence)
                
            # Aktualisiere Infotext
            file_name = os.path.basename(self.current_image_path)
            remaining = len(self.image_files) - self.current_index
            info_text = f"Bild: {file_name} ({self.current_index + 1}/{len(self.image_files)}, {remaining} verbleibend)"
            
            if prediction:
                info_text += f" | KI-Vorschlag: {prediction} ({confidence:.1%})"
                
            self.info_label.config(text=info_text)
            
            # Hebe vorgeschlagenen Button hervor, falls Vorhersage verfügbar
            if prediction:
                for btn, class_name in zip(self.buttons, self.classes.keys()):
                    if class_name == prediction:
                        btn.config(style="Accent.TButton")
                    else:
                        btn.config(style="TButton")
        else:
            # Keine Bilder mehr oder Fehler
            self.current_image_label.config(image='')
            self.info_label.config(text="Keine weiteren Bilder verfügbar.")
    
    def _update_batch_display(self):
        """Aktualisiert die Anzeige im Batch-Modus"""
        self.batch_images = []
        self.batch_selections = [None] * self.batch_size
        
        # Leere alle Batch-Frames
        for frame in self.batch_frames:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # Lade Batch-Bilder
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx >= len(self.image_files):
                break
                
            image_path = self.image_files[idx]
            try:
                img = Image.open(image_path)
                img.thumbnail((200, 200), Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                
                # Erstelle Label für Bild
                img_label = ttk.Label(self.batch_frames[i])
                img_label.image = photo
                img_label.config(image=photo)
                img_label.pack(pady=5)
                
                # Erstelle Label für Dateiname
                name_label = ttk.Label(self.batch_frames[i], text=os.path.basename(image_path))
                name_label.pack(pady=2)
                
                # Erstelle Dropdown für Klassen
                var = StringVar(self.root)
                
                # Mache KI-Vorhersage, falls aktiviert
                if self.model_predict and self.detector:
                    prediction, confidence = self._predict_label(image_path)
                    if prediction and confidence > 0.5:
                        var.set(prediction)
                    else:
                        var.set("Auswählen...")
                else:
                    var.set("Auswählen...")
                
                dropdown = ttk.Combobox(self.batch_frames[i], 
                                        textvariable=var,
                                        values=list(self.classes.keys()))
                dropdown.pack(pady=5, fill=tk.X)
                
                # Speichere Informationen
                self.batch_images.append((image_path, img, var))
            except Exception as e:
                print(f"Fehler beim Laden des Batch-Bildes {image_path}: {e}")
        
        # Aktualisiere Infotext
        remaining = len(self.image_files) - self.current_index
        info_text = f"Batch: {self.current_index + 1}-{min(self.current_index + self.batch_size, len(self.image_files))}/{len(self.image_files)} ({remaining} verbleibend)"
        self.info_label.config(text=info_text)
    
    def _next_image(self):
        """Geht zum nächsten Bild"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._update_display()
        else:
            messagebox.showinfo("Ende erreicht", "Keine weiteren Bilder verfügbar.")
    
    def _next_batch(self):
        """Geht zum nächsten Batch von Bildern"""
        if self.current_index + self.batch_size < len(self.image_files):
            self.current_index += self.batch_size
            self._update_batch_display()
        else:
            messagebox.showinfo("Ende erreicht", "Keine weiteren Bilder verfügbar.")
    
    def _prev_image(self):
        """Geht zum vorherigen Bild"""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()
    
    def _prev_batch(self):
        """Geht zum vorherigen Batch von Bildern"""
        if self.current_index >= self.batch_size:
            self.current_index -= self.batch_size
            self._update_batch_display()
    
    def _handle_label_button(self, label: str):
        """Verarbeitet einen Klick auf einen Label-Button"""
        if not self.current_image_path:
            return
        
        # Speichere Klassifizierung
        success = self._classify_image(self.current_image_path, label)
        
        if success:
            # Merke das letzte Label
            self.last_label = label
            
            # Gehe zum nächsten Bild, falls aktiviert
            if self.auto_advance:
                self._next_image()
    
    def _process_batch(self):
        """Verarbeitet alle ausgewählten Labels im Batch-Modus"""
        processed_count = 0
        
        for i, (image_path, _, var) in enumerate(self.batch_images):
            label = var.get()
            if label and label != "Auswählen...":
                success = self._classify_image(image_path, label)
                if success:
                    processed_count += 1
        
        # Aktualisiere nach Verarbeitung
        if processed_count > 0:
            messagebox.showinfo("Batch verarbeitet", f"{processed_count} Bilder erfolgreich klassifiziert.")
            
            # Gehe zum nächsten Batch
            if self.current_index + self.batch_size < len(self.image_files):
                self.current_index += self.batch_size
                self._update_batch_display()
            else:
                # Aktualisiere die Bilderliste, da wir am Ende sind
                if self.review_mode:
                    self.image_files = self._get_classified_image_files()
                else:
                    self.image_files = self._get_image_files()
                
                self.current_index = 0
                if self.image_files:
                    self._update_batch_display()
                else:
                    messagebox.showinfo("Fertig", "Alle Bilder wurden klassifiziert!")
                    self.batch_mode_active = False
                    self._toggle_batch_mode()
    
    def _toggle_histogram(self):
        """Schaltet die Histogramm-Anzeige ein oder aus"""
        self.histogram_visible = not self.histogram_visible
        
        if self.histogram_visible:
            # Erstelle Histogramm-Frame
            if self.histogram_canvas is None:
                fig = plt.Figure(figsize=(4, 3), dpi=100)
                self.histogram_canvas = FigureCanvasTkAgg(fig, self.histogram_frame)
                self.histogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.histogram_frame.pack(side=tk.RIGHT, fill=tk.Y)
            if self.current_image_obj:
                self._create_histogram(self.current_image_obj)
        else:
            # Verstecke Histogramm-Frame
            self.histogram_frame.pack_forget()
    
    def _toggle_batch_mode(self):
        """Wechselt zwischen Einzel- und Batch-Modus"""
        self.batch_mode_active = not self.batch_mode_active
        
        if self.batch_mode_active:
            # Verstecke Einzelbild-Anzeige
            self.img_frame.pack_forget()
            
            # Zeige Batch-Frame
            self.batch_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Aktualisiere Batch-Anzeige
            self._update_batch_display()
        else:
            # Verstecke Batch-Frame
            self.batch_container.pack_forget()
            
            # Zeige Einzelbild-Anzeige
            self.img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Aktualisiere Einzelbild-Anzeige
            self._update_display()
    
    def _toggle_auto_advance(self):
        """Schaltet den automatischen Bildwechsel ein oder aus"""
        self.auto_advance = not self.auto_advance
        text = "Auto-Vorschub: " + ("AN" if self.auto_advance else "AUS")
        self.auto_advance_btn.config(text=text)
    
    def _apply_last_label(self):
        """Wendet das zuletzt verwendete Label auf das aktuelle Bild an"""
        if self.last_label and self.current_image_path:
            self._handle_label_button(self.last_label)
    
    def _handle_zoom(self, factor):
        """Ändert den Zoom-Faktor"""
        self.zoom_factor *= factor
        self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))  # Begrenze zwischen 0.5x und 5x
        self._update_display()
    
    def _handle_contrast(self, delta):
        """Ändert den Kontrast"""
        self.contrast_factor += delta
        self.contrast_factor = max(0.5, min(self.contrast_factor, 2.0))  # Begrenze zwischen 0.5 und 2.0
        self._update_display()
    
    def _handle_brightness(self, delta):
        """Ändert die Helligkeit"""
        self.brightness_factor += delta
        self.brightness_factor = max(0.5, min(self.brightness_factor, 2.0))  # Begrenze zwischen 0.5 und 2.0
        self._update_display()
    
    def _reset_image_adjustments(self):
        """Setzt Bildanpassungen zurück"""
        self.zoom_factor = 1.0
        self.contrast_factor = 1.0
        self.brightness_factor = 1.0
        self._update_display()
    
    def _handle_keypress(self, event):
        """Verarbeitet Tastaturereignisse"""
        # Bildnavigation
        if event.keysym == "Right" or event.keysym == "space":
            if self.batch_mode_active:
                self._next_batch()
            else:
                self._next_image()
        elif event.keysym == "Left":
            if self.batch_mode_active:
                self._prev_batch()
            else:
                self._prev_image()
        
        # App-Steuerung
        elif event.keysym == "Escape":
            if self.is_fullscreen:
                self._toggle_fullscreen()
            else:
                self.root.destroy()
        elif event.keysym == "F11":
            self._toggle_fullscreen()
        elif event.keysym == "F5":
            self._toggle_batch_mode()
        elif event.keysym == "h":
            self._toggle_histogram()
        elif event.keysym == "a":
            self._toggle_auto_advance()
        elif event.keysym == "r":
            self._reset_image_adjustments()
        
        # Bild-Bearbeitung
        elif event.keysym == "plus" or event.keysym == "equal":
            self._handle_zoom(1.2)  # Zoom in
        elif event.keysym == "minus":
            self._handle_zoom(0.8)  # Zoom out
        elif event.keysym == "bracketright":
            self._handle_contrast(0.1)  # Erhöhe Kontrast
        elif event.keysym == "bracketleft":
            self._handle_contrast(-0.1)  # Verringere Kontrast
        elif event.keysym == "braceright":
            self._handle_brightness(0.1)  # Erhöhe Helligkeit
        elif event.keysym == "braceleft":
            self._handle_brightness(-0.1)  # Verringere Helligkeit
        
        # Spezielle Funktionen
        elif event.keysym == "Return":
            if self.batch_mode_active:
                self._process_batch()
            elif self.last_label:
                self._apply_last_label()
        
        # Klassifizierung über Tastenkürzel
        for class_name, class_info in self.classes.items():
            if hasattr(event, 'char') and event.char == class_info["shortcut"]:
                if self.batch_mode_active:
                    # Bei Batch-Mode abrufen, welches Bild aktuell fokussiert ist
                    # Hier vereinfacht: wähle das erste Bild mit leerem Label
                    for i, (_, _, var) in enumerate(self.batch_images):
                        if var.get() == "Auswählen...":
                            var.set(class_name)
                            break
                else:
                    self._handle_label_button(class_name)
    
    def _create_button_frame(self, parent):
        """Erstellt den Rahmen mit Label-Buttons"""
        button_frame = ttk.Frame(parent, padding=10)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Erstelle ein Farb-Schema für die Buttons
        s = ttk.Style()
        s.configure("Accent.TButton", background="lightblue")
        
        # Erstelle Buttons für jede Klasse
        for class_name, class_info in self.classes.items():
            btn = ttk.Button(
                button_frame,
                text=f"{class_name} ({class_info['shortcut']}) - {class_info['description']}",
                command=lambda name=class_name: self._handle_label_button(name)
            )
            btn.pack(fill=tk.X, pady=2)
            self.buttons.append(btn)
        
        # Zusätzliche Steuerungselemente
        control_frame = ttk.Frame(button_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Zoom-Steuerung
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="-", width=3, 
                   command=lambda: self._handle_zoom(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", width=6, 
                   command=self._reset_image_adjustments).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", width=3, 
                   command=lambda: self._handle_zoom(1.2)).pack(side=tk.LEFT, padx=2)
        
        # Histogramm-Toggle
        hist_btn = ttk.Button(zoom_frame, text="Histogramm (h)", 
                             command=self._toggle_histogram)
        hist_btn.pack(side=tk.LEFT, padx=10)
        
        # Auto-Advance Toggle
        self.auto_advance_btn = ttk.Button(zoom_frame, 
                                          text="Auto-Vorschub: AN", 
                                          command=self._toggle_auto_advance)
        self.auto_advance_btn.pack(side=tk.LEFT, padx=10)
        
        # Batch-Modus Toggle
        batch_btn = ttk.Button(zoom_frame, text="Batch-Modus (F5)", 
                              command=self._toggle_batch_mode)
        batch_btn.pack(side=tk.LEFT, padx=10)
        
        # Navigation
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        prev_btn = ttk.Button(nav_frame, text="< Zurück", command=self._prev_image)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = ttk.Button(nav_frame, text="Weiter >", command=self._next_image)
        next_btn.pack(side=tk.RIGHT, padx=5)
        
        # Git-Synchronisierung
        if self.git_track:
            git_btn = ttk.Button(nav_frame, text="Git Sync", 
                                command=self._git_pull_push)
            git_btn.pack(side=tk.RIGHT, padx=15)
    
    def _create_batch_frame(self, parent):
        """Erstellt den Rahmen für Batch-Bearbeitung"""
        # Container für das Batch-Layout
        self.batch_container = ttk.Frame(parent)
        
        # Grid-Layout für Bilder
        batch_grid = ttk.Frame(self.batch_container)
        batch_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Berechne Zeilen und Spalten für das Layout
        cols = min(3, self.batch_size)
        rows = (self.batch_size + cols - 1) // cols  # Aufrunden
        
        # Erstelle Frames für jedes Bild
        self.batch_frames = []
        for i in range(self.batch_size):
            row = i // cols
            col = i % cols
            
            frame = ttk.Frame(batch_grid, borderwidth=2, relief="groove", padding=5)
            frame.grid(row=row, col=col, padx=10, pady=10, sticky="nsew")
            
            self.batch_frames.append(frame)
        
        # Gleichmäßige Verteilung im Grid
        for i in range(rows):
            batch_grid.rowconfigure(i, weight=1)
        for i in range(cols):
            batch_grid.columnconfigure(i, weight=1)
        
        # Steuerung für Batch-Modus
        batch_control = ttk.Frame(self.batch_container)
        batch_control.pack(fill=tk.X, padx=10, pady=5)
        
        # Batch-Navigation
        prev_batch_btn = ttk.Button(batch_control, text="< Vorheriger Batch", 
                                   command=self._prev_batch)
        prev_batch_btn.pack(side=tk.LEFT, padx=5)
        
        # Anwenden-Button
        apply_batch_btn = ttk.Button(batch_control, text="Batch verarbeiten (Enter)", 
                                    command=self._process_batch)
        apply_batch_btn.pack(side=tk.LEFT, padx=20)
        
        # Nächster Batch
        next_batch_btn = ttk.Button(batch_control, text="Nächster Batch >", 
                                   command=self._next_batch)
        next_batch_btn.pack(side=tk.RIGHT, padx=5)
        
        # Helper-Button für Batch-Modus
        help_btn = ttk.Button(batch_control, text="Hilfe", 
                             command=lambda: messagebox.showinfo(
                                 "Batch-Modus Hilfe",
                                 "Im Batch-Modus kannst du mehrere Bilder gleichzeitig klassifizieren.\n\n"
                                 "- Wähle für jedes Bild eine Klasse aus dem Dropdown\n"
                                 "- Verwende Tastenkürzel, um schnell auszuwählen\n"
                                 "- Drücke ENTER, um den Batch zu verarbeiten\n"
                                 "- Drücke F5, um zum Einzelbild-Modus zurückzukehren"
                             ))
        help_btn.pack(side=tk.RIGHT, padx=20)
    
    def start_ui(self):
        """Startet die grafische Benutzeroberfläche"""
        if not self.image_files:
            print("Keine Bilder zum Labeln gefunden.")
            return
        
        # Erstelle Hauptfenster
        self.root = tk.Tk()
        self.root.title("Pizza Labeling Tool (Erweitert)")
        self.root.geometry("1024x768")
        
        # Erstelle ein Farb-Theme für die UI
        style = ttk.Style()
        style.theme_use('clam')  # oder 'alt', 'default', 'classic'
        
        # Hauptframe
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Informationsleiste
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.info_label = ttk.Label(info_frame, text="", anchor=tk.W)
        self.info_label.pack(side=tk.LEFT, fill=tk.X, pady=5)
        
        # Statusleiste für Bildanpassungen
        self.status_label = ttk.Label(info_frame, text="", anchor=tk.E)
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Bildanzeigebereich mit Histogramm
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Histogramm-Frame (rechts, anfangs versteckt)
        self.histogram_frame = ttk.Frame(content_frame, width=300, borderwidth=1, relief="groove")
        
        # Bildanzeige
        self.img_frame = ttk.Frame(content_frame)
        self.img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.current_image_label = ttk.Label(self.img_frame)
        self.current_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Batch-Rahmen erstellen (anfangs versteckt)
        self._create_batch_frame(content_frame)
        
        # Button-Frame
        self._create_button_frame(main_frame)
        
        # Tastatursteuerung einrichten
        self.root.bind("<Key>", self._handle_keypress)
        self.root.bind("<F11>", self._toggle_fullscreen)
        
        # Lade erstes Bild
        self._update_display()
        
        # Zeige Hilfe beim ersten Start
        messagebox.showinfo(
            "Pizza Labeling Tool",
            "Tastenkürzel:\n"
            "- Pfeiltasten/Leertaste: Navigation\n"
            "- Buchstaben (b,v,c,m,p,s): Klassenzuweisung\n"
            "- +/-: Zoom ein/aus\n"
            "- [/]: Kontrast-Einstellung\n"
            "- {/}: Helligkeits-Einstellung\n"
            "- h: Histogramm ein/aus\n"
            "- a: Auto-Vorschub ein/aus\n"
            "- r: Bildeinstellungen zurücksetzen\n"
            "- F5: Batch-Modus ein/aus\n"
            "- F11: Vollbild\n"
            "- ESC: Beenden"
        )
        
        # Starte UI-Loop
        self.root.mainloop()

def main():
    """Hauptfunktion zum Starten des Label-Tools"""
    parser = argparse.ArgumentParser(description="Pizza Labeling Tool (Erweitert)")
    parser.add_argument("--source-dir", default="data/raw", help="Verzeichnis mit unklassifizierten Bildern")
    parser.add_argument("--output-dir", default="data/classified", help="Verzeichnis für klassifizierte Bilder")
    parser.add_argument("--class-file", default=None, help="JSON-Datei mit Klassendefinitionen")
    parser.add_argument("--stats-file", default="data/classified/classification_stats.json", help="JSON-Datei für Statistiken")
    parser.add_argument("--no-git", action="store_true", help="Keine Git-Versionierung verwenden")
    parser.add_argument("--batch-size", type=int, default=4, help="Anzahl der Bilder im Batch-Modus")
    parser.add_argument("--review-mode", action="store_true", help="Review-Modus für bereits klassifizierte Bilder")
    parser.add_argument("--no-model", action="store_true", help="Keine KI-Vorschläge verwenden")
    
    args = parser.parse_args()
    
    # Erstelle und starte Labeler
    labeler = PizzaLabeler(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        class_file=args.class_file,
        stats_file=args.stats_file,
        git_track=not args.no_git,
        batch_size=args.batch_size,
        review_mode=args.review_mode,
        model_predict=not args.no_model
    )
    
    labeler.start_ui()

if __name__ == "__main__":
    main()