#!/usr/bin/env python3
"""
Offline-Visualisierung mit Grad-CAM für das Pizza-Erkennungssystem

Dieses Skript generiert Grad-CAM-Heatmaps für falsch klassifizierte Bilder,
um die Entscheidungsgrundlage des Netzes besser zu verstehen.

Verwendung:
    python visualize_gradcam.py [--model-path PATH] [--data-dir DIR] [--output-dir DIR]
                                [--target-layer LAYER] [--num-samples N] [--all]

Optionen:
    --model-path: Pfad zum trainierten Modell (Standard: models/micro_pizza_model.pth)
    --data-dir: Verzeichnis mit dem Validierungsdatensatz (Standard: data/augmented)
    --output-dir: Verzeichnis für die Ausgabe (Standard: output/gradcam)
    --target-layer: Ziel-Layer für Grad-CAM (Standard: auto)
    --num-samples: Anzahl der zu visualisierenden Bilder (Standard: 20)
    --all: Visualisiere alle Bilder, nicht nur falsch klassifizierte
    --format: Ausgabeformat (html, pdf oder beide, Standard: html)
    --class-names: Pfad zur JSON-Datei mit Klassennamen (Standard: data/class_definitions.json)
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')  # Nicht-interaktives Backend
import logging
import cv2
from pathlib import Path
import shutil
import webbrowser
import traceback
from jinja2 import Template

# Importiere Module aus dem Pizza-Projekt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pizza_detector import RP2040Config, create_optimized_dataloaders
from scripts.compare_tiny_cnns import MCUNet, MobilePizzaNet

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gradcam_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Angepasste MicroPizzaNet-Klasse, die mit der Struktur des gespeicherten Modells kompatibel ist
class OriginalMicroPizzaNet(nn.Module):
    """
    Diese Klasse repliziert die ursprüngliche MicroPizzaNet-Architektur
    mit den gleichen Layer-Benennungen und Dimensionen wie im gespeicherten Modell
    """
    def __init__(self, num_classes=6, input_channels=3, img_size=48):
        super(OriginalMicroPizzaNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        
        # Feste Konfiguration für das gespeicherte Modell
        channels = 8  # 8 Kanäle im ersten Block, wie im gespeicherten Modell
        
        # Erster Konvolutionsblock
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Zweiter Block - Depthwise Separable Convolution
        self.block2 = nn.Sequential(
            # Depthwise convolution - 8 Kanäle, eine Gruppe pro Kanal (depthwise)
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            # Pointwise convolution - Erhöhe auf 16 Kanäle
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU()
        )
        
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Flatten und Classifier (16 -> 6 Klassen)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(channels * 2, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class GradCAM:
    """
    Grad-CAM-Implementierung zur Visualisierung von CNN-Entscheidungen
    """
    def __init__(self, model, target_layer_name='auto', device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Feature-Maps und Gradienten speichern
        self.gradients = None
        self.activations = None
        
        # Bestimme den zu visualisierenden Layer
        if target_layer_name == 'auto':
            # Wähle automatisch den letzten Convolutional Layer
            target_layer = self._find_last_conv_layer(model)
            if target_layer is None:
                raise ValueError("Konnte keinen Convolutional Layer im Modell finden!")
            logger.info(f"Automatisch ausgewählter Layer für Grad-CAM: {target_layer.__class__.__name__}")
        else:
            # Finde den angegebenen Layer
            target_layer = self._find_layer_by_name(model, target_layer_name)
            if target_layer is None:
                raise ValueError(f"Layer '{target_layer_name}' nicht im Modell gefunden!")
        
        # Registriere Hooks für Forward und Backward Pass
        self.target_layer = target_layer
        self.hook_handles = []
        
        # Forward-Hook registrieren
        handle = target_layer.register_forward_hook(self._save_activation)
        self.hook_handles.append(handle)
        
        # Backward-Hook registrieren
        handle = target_layer.register_full_backward_hook(self._save_gradient)
        self.hook_handles.append(handle)
    
    def _find_last_conv_layer(self, model):
        """Findet den letzten Convolutional Layer im Modell"""
        last_conv = None
        
        # Spezielle Behandlung für die verschiedenen Modelltypen
        if isinstance(model, OriginalMicroPizzaNet):
            # Bei OriginalMicroPizzaNet ist der letzte Conv Layer in block2
            for module in model.block2:
                if isinstance(module, nn.Conv2d):
                    last_conv = module
        
        elif isinstance(model, MCUNet):
            # Bei MCUNet ist der letzte Conv Layer in features
            for module in model.features:
                for layer in module:
                    if isinstance(layer, nn.Conv2d):
                        last_conv = layer
        
        elif isinstance(model, MobilePizzaNet):
            # Bei MobilePizzaNet ist der letzte Conv Layer in features
            if len(model.features) > 0:
                for block in model.features:
                    for layer in block.conv:
                        if isinstance(layer, nn.Conv2d):
                            last_conv = layer
        
        # Fallback: Durchsuche das gesamte Modell rekursiv
        if last_conv is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
        
        return last_conv
    
    def _find_layer_by_name(self, model, name):
        """Findet einen Layer anhand seines Namens"""
        for n, module in model.named_modules():
            if n == name:
                return module
        return None
    
    def _save_activation(self, module, input, output):
        """Speichert die Aktivierungen des Forward-Passes"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Speichert die Gradienten des Backward-Passes"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generiert eine Klassen-Aktivierungs-Map (CAM) für das Eingabebild
        
        Args:
            input_image (torch.Tensor): Eingabebild als Tensor (1, C, H, W)
            target_class (int, optional): Zielklasse für CAM. Wenn None, wird die
                                         vorhergesagte Klasse verwendet.
        
        Returns:
            Tuple: (cam, class_idx, output_prob)
                - cam: Normalisierte CAM als numpy array
                - class_idx: Index der verwendeten Klasse 
                - output_prob: Wahrscheinlichkeit der Vorhersage
        """
        # Sicherstellen, dass wir ein Batch mit einem Bild haben
        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)
        
        # Forward-Pass und Vorhersage
        input_image = input_image.to(self.device)
        
        # Zurücksetzen der gespeicherten Aktivierungen und Gradienten
        self.activations = None
        self.gradients = None
        
        # Forward-Pass mit Gradient-Tracking
        input_image.requires_grad = True
        output = self.model(input_image)
        
        # Bestimme Klasse (vorhergesagt oder vorgegeben)
        if target_class is None:
            class_idx = output.argmax().item()
        else:
            class_idx = target_class
        
        # Softmax für Wahrscheinlichkeit
        probs = F.softmax(output, dim=1)
        output_prob = probs[0, class_idx].item()
        
        # Backpropagation für die ausgewählte Klasse
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Überprüfen, ob wir Gradienten und Aktivierungen haben
        if self.gradients is None or self.activations is None:
            logger.error("Gradienten oder Aktivierungen konnten nicht erfasst werden!")
            return None, class_idx, output_prob
        
        # Gewichtete Summe der Gradient-Kanäle für CAM
        with torch.no_grad():
            # Global Average Pooling der Gradienten
            alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # Gewichtete Summe der Aktivierungen
            cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)
            
            # ReLU anwenden, um nur positive Einflüsse zu zeigen
            cam = F.relu(cam)
            
            # Größe an Eingabebild anpassen
            cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
            
            # Normalisieren für Visualisierung
            cam = cam - cam.min()
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max
            
            # Zu numpy konvertieren
            cam = cam.squeeze().cpu().numpy()
        
        return cam, class_idx, output_prob
    
    def overlay_cam_on_image(self, img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Überlagert ein Bild mit der CAM-Heatmap
        
        Args:
            img (numpy.ndarray): Originalbild als numpy-Array (H, W, C) im Format 0-255
            cam (numpy.ndarray): CAM-Heatmap als numpy-Array (H, W) im Format 0-1
            alpha (float): Transparenz der Heatmap (0-1)
            colormap: OpenCV-Colormap für die Heatmap
        
        Returns:
            numpy.ndarray: Bild mit überlagerter Heatmap im Format (H, W, C)
        """
        # Konvertiere CAM zu 8-bit und wende Colormap an
        cam_8bit = (cam * 255).astype(np.uint8)
        cam_color = cv2.applyColorMap(cam_8bit, colormap)
        
        # Sicherstellen, dass das Bild RGB ist
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # Mit Alpha-Kanal
            img = img[:, :, :3]
        
        # Sicherstellen, dass die Größen übereinstimmen
        if img.shape[:2] != cam_color.shape[:2]:
            cam_color = cv2.resize(cam_color, (img.shape[1], img.shape[0]))
        
        # Overlay erstellen
        overlay = cv2.addWeighted(img, 1 - alpha, cam_color, alpha, 0)
        
        return overlay
    
    def __del__(self):
        """Entfernt die registrierten Hooks beim Löschen der Instanz"""
        for handle in self.hook_handles:
            handle.remove()


def load_model(model_path, num_classes=6):
    """Lädt das Modell aus der angegebenen Datei"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
    
    # Bestimme Modelltyp anhand des Dateinamens
    filename = os.path.basename(model_path)
    
    if "micro_pizza" in filename.lower() or "pizza_model" in filename.lower():
        logger.info("Lade MicroPizzaNet Modell")
        model = OriginalMicroPizzaNet(
            num_classes=num_classes,
            input_channels=3,
            img_size=48
        )
    elif "mcunet" in filename.lower():
        logger.info("Lade MCUNet Modell")
        model = MCUNet(
            num_classes=num_classes,
            input_channels=3,
            img_size=48,
            width_mult=0.5
        )
    elif "mobile" in filename.lower():
        logger.info("Lade MobilePizzaNet Modell")
        model = MobilePizzaNet(
            num_classes=num_classes,
            input_channels=3,
            img_size=48,
            width_mult=0.25
        )
    else:
        # Standard: MicroPizzaNet
        logger.info("Modelltyp nicht erkannt, verwende MicroPizzaNet")
        model = OriginalMicroPizzaNet(
            num_classes=num_classes,
            input_channels=3,
            img_size=48
        )
    
    # Lade Gewichte
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        logger.info(f"Modell aus {model_path} geladen")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        raise
    
    return model


def get_misclassified_samples(model, data_loader, num_samples=20, all_samples=False, device='cpu'):
    """
    Identifiziert falsch klassifizierte Bilder aus dem Datensatz
    
    Args:
        model: Das zu evaluierende Modell
        data_loader: DataLoader mit Validierungsdaten
        num_samples: Maximale Anzahl zurückzugebender Bilder
        all_samples: Wenn True, werden auch korrekt klassifizierte Bilder zurückgegeben
        device: Gerät für die Berechnung ('cpu' oder 'cuda')
    
    Returns:
        Liste von Tupeln (image, true_label, pred_label, pred_prob, correct)
    """
    model.to(device)
    model.eval()
    
    misclassified = []
    correct_classified = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Suche nach falsch klassifizierten Bildern"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Hole Top-1 Prediction
            _, preds = torch.max(outputs, 1)
            
            # Für jedes Bild im Batch
            for i in range(images.size(0)):
                image = images[i]
                label = labels[i].item()
                pred = preds[i].item()
                prob = probs[i, pred].item()
                correct = (label == pred)
                
                # Speichere Bild mit Metadaten
                if not correct:
                    misclassified.append((image.cpu(), label, pred, prob, correct))
                elif all_samples and len(correct_classified) < num_samples:
                    correct_classified.append((image.cpu(), label, pred, prob, correct))
    
    # Shuffle und begrenze Anzahl
    random.shuffle(misclassified)
    misclassified = misclassified[:num_samples]
    
    if all_samples:
        # Kombiniere falsch und korrekt klassifizierte Bilder
        samples = misclassified + correct_classified
        random.shuffle(samples)
        return samples[:num_samples]
    else:
        return misclassified


def create_html_report(results, output_dir, class_names):
    """
    Erstellt einen HTML-Bericht mit GradCAM-Visualisierungen
    
    Args:
        results: Liste mit (image_path, original_image, gradcam_image, true_label, pred_label, 
                            pred_prob, is_correct, comment)
        output_dir: Ausgabeverzeichnis
        class_names: Liste der Klassennamen
    
    Returns:
        Pfad zur generierten HTML-Datei
    """
    # HTML-Template
    template_str = """<!DOCTYPE html>
    <html>
    <head>
        <title>Pizza-Erkennungssystem: GradCAM-Visualisierung</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; border-bottom: 1px solid #34495e; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .sample { 
                display: flex;
                border: 1px solid #ddd;
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .sample.correct { background-color: #e8f5e9; }
            .sample.incorrect { background-color: #ffebee; }
            .images { 
                display: flex;
                flex-direction: column;
                margin-right: 20px;
            }
            .image-container { 
                text-align: center;
                margin-bottom: 10px;
            }
            .info {
                flex: 1;
            }
            .prediction {
                font-weight: bold;
                margin-bottom: 15px;
            }
            .correct-pred { color: #4caf50; }
            .incorrect-pred { color: #f44336; }
            .confidence {
                display: inline-block;
                height: 20px;
                background: linear-gradient(to right, #4caf50, #8bc34a);
                border-radius: 3px;
                text-align: right;
                color: white;
                padding-right: 5px;
                box-sizing: border-box;
                margin-top: 5px;
            }
            .low-confidence {
                background: linear-gradient(to right, #ff9800, #ffeb3b);
            }
            .very-low-confidence {
                background: linear-gradient(to right, #f44336, #ff9800);
            }
            .comment {
                border-left: 3px solid #34495e;
                padding-left: 10px;
                margin-top: 15px;
                font-style: italic;
                color: #555;
            }
            .timestamp {
                text-align: right;
                color: #7f8c8d;
                font-size: 0.8em;
                margin-top: 50px;
            }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; text-align: center; }
            tr:hover { background-color: #f5f5f5; }
            .summary { margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>GradCAM-Visualisierung für Pizza-Erkennungssystem</h1>
        
        <div class="summary">
            <h2>Zusammenfassung</h2>
            <p>Modell: {{ model_name }}</p>
            <p>Analysierte Bilder: {{ total_samples }}</p>
            <p>Davon falsch klassifiziert: {{ misclassified_count }} ({{ misclassified_percent }}%)</p>
            <p>Durchschnittliche Konfidenz bei korrekten Vorhersagen: {{ avg_correct_confidence }}%</p>
            <p>Durchschnittliche Konfidenz bei falschen Vorhersagen: {{ avg_incorrect_confidence }}%</p>
        </div>

        <h2>Klassenverteilung</h2>
        <table>
            <tr>
                <th>Klasse</th>
                <th>Richtig erkannt</th>
                <th>Falsch erkannt</th>
                <th>Genauigkeit</th>
            </tr>
            {% for class_name, correct, total in class_stats %}
            <tr>
                <td>{{ class_name }}</td>
                <td style="text-align: center;">{{ correct }}</td>
                <td style="text-align: center;">{{ total - correct }}</td>
                <td style="text-align: center;">{{ "%.1f"|format(100 * correct / total if total > 0 else 0) }}%</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>GradCAM-Visualisierungen</h2>
        
        {% for sample in samples %}
        <div class="sample {% if sample.correct %}correct{% else %}incorrect{% endif %}">
            <div class="images">
                <div class="image-container">
                    <img src="{{ sample.original_image }}" alt="Original" width="224">
                    <div>Originalbild</div>
                </div>
                <div class="image-container">
                    <img src="{{ sample.gradcam_image }}" alt="GradCAM" width="224">
                    <div>GradCAM-Heatmap</div>
                </div>
            </div>
            <div class="info">
                <div class="prediction {% if sample.correct %}correct-pred{% else %}incorrect-pred{% endif %}">
                    Vorhersage: {{ sample.pred_label }} ({{ "%.1f"|format(sample.pred_prob * 100) }}%)
                    {% if not sample.correct %}
                    <br>Tatsächlich: {{ sample.true_label }}
                    {% endif %}
                </div>
                
                <div>Konfidenz:</div>
                <div class="confidence {% if sample.pred_prob < 0.7 %}low-confidence{% endif %} {% if sample.pred_prob < 0.5 %}very-low-confidence{% endif %}" 
                     style="width: {{ sample.pred_prob * 100 }}%;">
                    {{ "%.1f"|format(sample.pred_prob * 100) }}%
                </div>
                
                <div class="comment">{{ sample.comment }}</div>
            </div>
        </div>
        {% endfor %}
        
        <div class="timestamp">
            Erstellt am {{ timestamp }}
        </div>
    </body>
    </html>
    """
    
    template = Template(template_str)
    
    # Pfade für HTML und Bilder
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    html_path = os.path.join(output_dir, "gradcam_report.html")
    
    # Berechne Statistiken
    total_samples = len(results)
    misclassified_count = sum(1 for r in results if not r[6])  # is_correct ist False
    misclassified_percent = 100 * misclassified_count / total_samples if total_samples > 0 else 0
    
    correct_confidences = [r[5] for r in results if r[6]]  # confidence where is_correct is True
    incorrect_confidences = [r[5] for r in results if not r[6]]  # confidence where is_correct is False
    
    avg_correct_confidence = 100 * sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    avg_incorrect_confidence = 100 * sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
    
    # Klassenstatistiken
    class_counts = {name: {"correct": 0, "total": 0} for name in class_names}
    for _, _, _, true_label, pred_label, _, is_correct, _ in results:
        true_name = class_names[true_label]
        class_counts[true_name]["total"] += 1
        if is_correct:
            class_counts[true_name]["correct"] += 1
    
    class_stats = [(name, stats["correct"], stats["total"]) 
                  for name, stats in class_counts.items()]
    
    # Vorbereiten der Daten für das Template
    samples_for_template = []
    
    for i, (image_path, original, gradcam, true_label, pred_label, confidence, is_correct, comment) in enumerate(results):
        # Speichere Bilder
        orig_filename = f"images/original_{i}.png"
        gradcam_filename = f"images/gradcam_{i}.png"
        
        cv2.imwrite(os.path.join(output_dir, orig_filename), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, gradcam_filename), cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR))
        
        # Füge Sample zum Template hinzu
        samples_for_template.append({
            "original_image": orig_filename,
            "gradcam_image": gradcam_filename,
            "true_label": class_names[true_label],
            "pred_label": class_names[pred_label],
            "pred_prob": confidence,
            "correct": is_correct,
            "comment": comment
        })
    
    # Render Template
    html_content = template.render(
        model_name=os.path.basename(args.model_path),
        total_samples=total_samples,
        misclassified_count=misclassified_count,
        misclassified_percent=f"{misclassified_percent:.1f}",
        avg_correct_confidence=f"{avg_correct_confidence:.1f}",
        avg_incorrect_confidence=f"{avg_incorrect_confidence:.1f}",
        class_stats=class_stats,
        samples=samples_for_template,
        timestamp=datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    )
    
    # Schreibe HTML-Datei
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"HTML-Bericht erstellt: {html_path}")
    return html_path


def create_pdf_report(results, output_dir, class_names):
    """
    Erstellt einen PDF-Bericht mit GradCAM-Visualisierungen
    
    Args:
        results: Liste mit (image_path, original_image, gradcam_image, true_label, pred_label, 
                            pred_prob, class_names, comment)
        output_dir: Ausgabeverzeichnis
        class_names: Liste der Klassennamen
    
    Returns:
        Pfad zur generierten PDF-Datei
    """
    try:
        from fpdf import FPDF
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Pizza-Erkennungssystem: GradCAM-Visualisierung', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')
        
        # Erstelle PDF
        pdf = PDF()
        pdf.add_page()
        
        # Statistiken
        total_samples = len(results)
        misclassified_count = sum(1 for r in results if not r[4] == r[3])
        misclassified_percent = 100 * misclassified_count / total_samples if total_samples > 0 else 0
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Zusammenfassung', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 7, f'Modell: {os.path.basename(args.model_path)}', 0, 1)
        pdf.cell(0, 7, f'Analysierte Bilder: {total_samples}', 0, 1)
        pdf.cell(0, 7, f'Davon falsch klassifiziert: {misclassified_count} ({misclassified_percent:.1f}%)', 0, 1)
        pdf.ln(5)
        
        # Klassenstatistiken
        class_counts = {name: {"correct": 0, "total": 0} for name in class_names}
        for _, _, true_label, pred_label, _, _ in results:
            true_name = class_names[true_label]
            class_counts[true_name]["total"] += 1
            if true_label == pred_label:
                class_counts[true_name]["correct"] += 1
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Klassenverteilung', 0, 1)
        pdf.set_font('Arial', 'B', 10)
        
        # Tabellenkopf
        pdf.cell(50, 7, 'Klasse', 1, 0, 'C')
        pdf.cell(40, 7, 'Richtig erkannt', 1, 0, 'C')
        pdf.cell(40, 7, 'Falsch erkannt', 1, 0, 'C')
        pdf.cell(40, 7, 'Genauigkeit', 1, 1, 'C')
        
        # Tabellenzeilen
        pdf.set_font('Arial', '', 10)
        for class_name, stats in class_counts.items():
            correct = stats["correct"]
            total = stats["total"]
            accuracy = 100 * correct / total if total > 0 else 0
            
            pdf.cell(50, 7, class_name, 1, 0)
            pdf.cell(40, 7, str(correct), 1, 0, 'C')
            pdf.cell(40, 7, str(total - correct), 1, 0, 'C')
            pdf.cell(40, 7, f"{accuracy:.1f}%", 1, 1, 'C')
        
        pdf.ln(10)
        
        # GradCAM-Visualisierungen
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'GradCAM-Visualisierungen', 0, 1)
        
        # Bilder speichern für PDF
        image_paths = []
        
        for i, (_, original, gradcam, true_label, pred_label, confidence, is_correct, comment) in enumerate(results):
            # Erstelle kombiniertes Bild (Original + GradCAM)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(original)
            ax[0].set_title('Originalbild')
            ax[0].axis('off')
            
            ax[1].imshow(gradcam)
            ax[1].set_title('GradCAM-Heatmap')
            ax[1].axis('off')
            
            fig.tight_layout()
            
            # Speichere Bild
            img_path = os.path.join(output_dir, f"combined_{i}.png")
            fig.savefig(img_path, dpi=100)
            plt.close(fig)
            
            image_paths.append((img_path, true_label, pred_label, confidence, is_correct, comment))
        
        # Füge Bilder zum PDF hinzu
        for i, (img_path, true_label, pred_label, confidence, is_correct, comment) in enumerate(image_paths):
            if i > 0 and i % 2 == 0:
                pdf.add_page()
            
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            
            pdf.set_font('Arial', 'B', 10)
            status = "Korrekt" if is_correct else "Falsch"
            pdf.cell(0, 10, f"Bild {i+1}: {status} klassifiziert", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            if is_correct:
                pdf.cell(0, 7, f"Vorhersage: {pred_name} ({confidence*100:.1f}%)", 0, 1)
            else:
                pdf.cell(0, 7, f"Vorhersage: {pred_name} ({confidence*100:.1f}%), Tatsächlich: {true_name}", 0, 1)
            
            pdf.image(img_path, x=10, w=pdf.w - 20)
            
            pdf.set_font('Arial', 'I', 9)
            pdf.multi_cell(0, 5, f"Kommentar: {comment}")
            
            pdf.ln(10)
        
        # Speichere PDF
        pdf_path = os.path.join(output_dir, "gradcam_report.pdf")
        pdf.output(pdf_path)
        
        logger.info(f"PDF-Bericht erstellt: {pdf_path}")
        return pdf_path
        
    except ImportError:
        logger.warning("FPDF nicht installiert. PDF-Bericht konnte nicht erstellt werden.")
        logger.warning("Installiere FPDF mit: pip install fpdf")
        return None


def generate_comment(true_label, pred_label, confidence, class_names):
    """Generiert einen Kommentar basierend auf der Klassifikation"""
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    
    if true_label == pred_label:
        if confidence > 0.9:
            return f"Sehr sichere Klassifikation als '{pred_name}' mit hoher Konfidenz."
        elif confidence > 0.7:
            return f"Korrekte Klassifikation als '{pred_name}' mit guter Konfidenz."
        else:
            return f"Korrekte, aber unsichere Klassifikation als '{pred_name}'. Die Heatmap zeigt möglicherweise auf unspezifische Merkmale."
    else:
        if confidence > 0.9:
            return f"Falsche Klassifikation mit hoher Konfidenz. Das Modell verwechselt '{true_name}' mit '{pred_name}', konzentriert sich möglicherweise auf irreführende Merkmale."
        elif confidence > 0.7:
            return f"Falsche Klassifikation mit mittlerer Konfidenz. Die Textur/Farbe könnte zu Verwechslung zwischen '{true_name}' und '{pred_name}' führen."
        else:
            return f"Unsichere und falsche Klassifikation. Das Modell scheint zwischen '{true_name}' und '{pred_name}' zu zögern, mit Fokus auf unklaren Merkmalen."


def main(args):
    """Hauptfunktion"""
    # Ausgabeverzeichnis erstellen
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Klassenamen laden
    try:
        with open(args.class_names, 'r') as f:
            class_data = json.load(f)
            # Format der Klassennamen anpassen: Wir verwenden die Schlüssel des JSON-Objekts
            class_names = list(class_data.keys())
            logger.info(f"Klassen geladen: {class_names}")
    except Exception as e:
        logger.warning(f"Fehler beim Laden der Klassennamen: {e}")
        class_names = [f"Klasse_{i}" for i in range(6)]  # Fallback
    
    # Modell laden
    model = load_model(args.model_path, num_classes=len(class_names))
    
    # Datensatz direkt laden ohne die optimierte Funktion
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Versuche, den Datensatz zu laden
        dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
        logger.info(f"Datensatz mit {len(dataset)} Bildern geladen")
        
        # Aufteilen in Trainings- und Validierungsdaten (80/20)
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    except Exception as e:
        logger.error(f"Fehler beim Laden des Datensatzes: {e}")
        logger.error("Erstelle einen Dummy-Datensatz für den Test")
        
        # Erstelle einen Dummy-Validierungsloader mit zufälligen Daten
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100, num_classes=6):
                self.size = size
                self.num_classes = num_classes
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                img = torch.randn(3, 48, 48)  # Zufälliges Bild
                label = random.randint(0, self.num_classes - 1)  # Zufälliges Label
                return img, label
        
        val_dataset = DummyDataset(size=100, num_classes=len(class_names))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        logger.warning("Verwende Dummy-Datensatz! Die Ergebnisse sind nicht repräsentativ.")
    
    # GradCAM initialisieren
    gradcam = GradCAM(model, target_layer_name=args.target_layer)
    
    # Falsch klassifizierte Bilder finden
    samples = get_misclassified_samples(
        model, 
        val_loader, 
        num_samples=args.num_samples,
        all_samples=args.all
    )
    
    # Warnmeldung, wenn nicht genug falsch klassifizierte Bilder gefunden wurden
    if len(samples) < args.num_samples:
        logger.warning(f"Nur {len(samples)} falsch klassifizierte Bilder gefunden. "
                     f"Verwenden Sie --all, um auch korrekt klassifizierte Bilder anzuzeigen.")
    
    # GradCAM-Visualisierungen erstellen
    results = []
    
    for i, (image, true_label, pred_label, confidence, is_correct) in enumerate(tqdm(samples, desc="Erstelle GradCAM-Visualisierungen")):
        try:
            # GradCAM-Heatmap generieren
            cam, cam_class, cam_prob = gradcam.generate_cam(image, target_class=pred_label)
            
            if cam is None:
                logger.warning(f"Konnte keine CAM für Bild {i} generieren, überspringe...")
                continue
            
            # Bild zu numpy konvertieren und normalisieren
            img_np = image.permute(1, 2, 0).numpy()
            
            # Denormalisieren (falls das Bild bereits normalisiert wurde)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = ((img_np * std + mean) * 255).astype(np.uint8)
            
            # Heatmap überlagern
            overlay = gradcam.overlay_cam_on_image(img_np, cam)
            
            # Kommentar generieren
            comment = generate_comment(true_label, pred_label, confidence, class_names)
            
            # Ergebnis speichern
            image_path = f"image_{i}.png"
            results.append((image_path, img_np, overlay, true_label, pred_label, confidence, is_correct, comment))
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von Bild {i}: {e}")
            logger.error(traceback.format_exc())
    
    # Berichte erstellen
    if args.format in ['html', 'both']:
        html_path = create_html_report(results, args.output_dir, class_names)
        # HTML öffnen
        if html_path:
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
    
    if args.format in ['pdf', 'both']:
        pdf_path = create_pdf_report(results, args.output_dir, class_names)
    
    logger.info("GradCAM-Visualisierung abgeschlossen!")
    logger.info(f"Ergebnisse gespeichert in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GradCAM-Visualisierung für Pizza-Erkennungssystem')
    parser.add_argument('--model-path', default='models/micro_pizza_model.pth',
                        help='Pfad zum trainierten Modell')
    parser.add_argument('--data-dir', default='data/augmented',
                        help='Verzeichnis mit dem Validierungsdatensatz')
    parser.add_argument('--output-dir', default='output/gradcam',
                        help='Verzeichnis für die Ausgabe')
    parser.add_argument('--target-layer', default='auto',
                        help='Ziel-Layer für GradCAM')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Anzahl der zu visualisierenden Bilder')
    parser.add_argument('--all', action='store_true',
                        help='Visualisiere alle Bilder, nicht nur falsch klassifizierte')
    parser.add_argument('--format', choices=['html', 'pdf', 'both'], default='html',
                        help='Ausgabeformat (html, pdf oder beide)')
    parser.add_argument('--class-names', default='data/class_definitions.json',
                        help='Pfad zur JSON-Datei mit Klassennamen')
    
    args = parser.parse_args()
    main(args)