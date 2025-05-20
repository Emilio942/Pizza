#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal Verification Framework für Pizza-Erkennungssystem

Dieses Modul implementiert formale Verifikation für MicroPizzaNet-Modelle,
mit Fokus auf Robustheit gegen kleine Störungen und Invarianz gegenüber
bestimmten Transformationen.

Es verwendet ERAN/α,β-CROWN für Neural Network Verification.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
from enum import Enum

# Füge Projektverzeichnis zum Pfad hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE
from src.utils.validation import validate_model_config
from src.constants import DEFAULT_CLASSES as CLASS_NAMES, MODELS_DIR as MODEL_DIR, PROJECT_ROOT

# Versuche, die formale Verifikations-Abhängigkeiten zu importieren
# Diese müssen separat installiert werden (siehe README.md)
VERIFICATION_DEPENDENCIES_INSTALLED = False
try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    VERIFICATION_DEPENDENCIES_INSTALLED = True
except ImportError:
    logging.warning(
        "α,β-CROWN nicht gefunden. Formale Verifikation wird nicht verfügbar sein. "
        "Installiere mit: pip install auto_LiRPA"
    )

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("formal_verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("formal_verification")

# Make VERIFICATION_DEPENDENCIES_INSTALLED accessible
__all__ = ['ModelVerifier', 'VerificationResult', 'VerificationProperty', 'load_model_for_verification', 'CLASS_NAMES', 'VERIFICATION_DEPENDENCIES_INSTALLED']

class VerificationProperty(Enum):
    """Verifizierbare Eigenschaften für das Pizza-Erkennungsmodell."""
    ROBUSTNESS = "robustness"                    # Robustheit gegen kleine Störungen (L_p Norm)
    BRIGHTNESS_INVARIANCE = "brightness"         # Invarianz gegenüber Helligkeitsänderungen
    CLASS_SEPARATION = "class_separation"        # Vermeidung von Verwechslungen
    MONOTONICITY = "monotonicity"                # Monotonie bei bestimmten Transformationen

class VerificationResult:
    """Ergebnis einer Verifikation."""
    def __init__(
        self,
        verified: bool,
        property_type: VerificationProperty,
        time_seconds: float,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        counterexample: Optional[np.ndarray] = None,
        details: Dict[str, Any] = None
    ):
        self.verified = verified
        self.property_type = property_type
        self.time_seconds = time_seconds
        self.bounds = bounds
        self.counterexample = counterexample
        self.details = details or {}
        
    def __str__(self) -> str:
        status = "✓ VERIFIZIERT" if self.verified else "✗ NICHT VERIFIZIERT"
        result = f"Eigenschaft: {self.property_type.value} - Status: {status}\n"
        result += f"Verifikationszeit: {self.time_seconds:.2f} Sekunden\n"
        
        if not self.verified and self.counterexample is not None:
            result += "Gegenbeispiel gefunden\n"
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Ergebnis in ein Dictionary zur Speicherung."""
        result_dict = {
            "verified": self.verified,
            "property_type": self.property_type.value,
            "time_seconds": self.time_seconds,
            "details": self.details
        }
        
        # Nur relevante Teile des Gegenbeispiels speichern (kann sehr groß sein)
        if not self.verified and self.counterexample is not None:
            if isinstance(self.counterexample, np.ndarray):
                # Speichere komprimierte Statistiken anstatt des ganzen Arrays
                result_dict["counterexample_stats"] = {
                    "shape": self.counterexample.shape,
                    "min": float(np.min(self.counterexample)),
                    "max": float(np.max(self.counterexample)),
                    "mean": float(np.mean(self.counterexample)),
                    "std": float(np.std(self.counterexample))
                }
            else:
                result_dict["counterexample"] = str(self.counterexample)
                
        return result_dict

class ModelVerifier:
    """
    Framework zur formalen Verifikation von MicroPizzaNet-Modellen.
    
    Diese Klasse bietet Methoden zur Überprüfung von:
    - Robustheit gegen kleine Störungen (Adversarial Robustness)
    - Invarianz gegenüber bestimmten Transformationen (z.B. Helligkeit)
    - Separation von Klassen (keine Überschneidung kritischer Klassen)
    """
    
    def __init__(
        self, 
        model: nn.Module,
        input_size: Tuple[int, int] = (48, 48),
        device: str = 'cpu',
        epsilon: float = 0.01,
        norm_type: str = 'L_inf',
        verify_backend: str = 'crown'
    ):
        """
        Initialisiert den Model Verifier.
        
        Args:
            model: Das zu verifizierende Modell
            input_size: Eingabegröße des Bildes (H, W)
            device: Das zu verwendende Gerät ('cpu' oder 'cuda')
            epsilon: Störungsparameter für Robustheit (Default: 0.01)
            norm_type: Norm für Robustheitsanalyse ('L_inf', 'L_1', 'L_2')
            verify_backend: Verifikations-Backend ('crown' oder 'beta-crown')
        """
        if not VERIFICATION_DEPENDENCIES_INSTALLED:
            raise ImportError(
                "Formale Verifikation erfordert zusätzliche Abhängigkeiten. "
                "Bitte folge der Installationsanleitung in README.md."
            )
        
        self.model = model
        self.input_size = input_size
        self.device = torch.device(device)
        self.epsilon = epsilon
        
        # Konvertiere Norm-String in die entsprechende p-Norm
        self.norm_type = {
            'L_inf': np.inf,
            'L_1': 1,
            'L_2': 2
        }.get(norm_type, np.inf)
        
        self.verify_backend = verify_backend
        self.model.to(self.device)
        self.model.eval()
        
        # Für α,β-CROWN Verifikation
        self._bound_opts = {
            "optimize_bound_args": {
                "iteration": 100, 
                "lr_alpha": 0.1,
                "verbose": 0
            }
        }
        
        logger.info(f"ModelVerifier initialisiert für Modell: {type(model).__name__}")
    
    def verify_robustness(
        self, 
        input_image: np.ndarray, 
        true_class: int, 
        epsilon: Optional[float] = None
    ) -> VerificationResult:
        """
        Verifiziert die Robustheit des Modells gegen kleine Störungen.
        
        Überprüft, ob für alle Eingaben innerhalb der ε-Umgebung (bzgl. gewählter Norm)
        des Eingangsbildes die Vorhersage unverändert bleibt.
        
        Args:
            input_image: Das Eingabebild als numpy-Array mit Shape (H, W, C) oder (C, H, W)
            true_class: Die korrekte Klasse des Bildes
            epsilon: Der Störungsparameter (optional, falls überschrieben werden soll)
            
        Returns:
            Ein VerificationResult-Objekt mit dem Ergebnis der Verifikation
        """
        start_time = time.time()
        epsilon = epsilon or self.epsilon
        
        # Prüfe Bildformat und konvertiere nach Bedarf
        if input_image.shape[-1] == 3:  # Format: H,W,C
            input_image = np.transpose(input_image, (2, 0, 1))  # -> C,H,W
        
        # Normalisiere Bild sofern nötig (je nach Vorverarbeitung des Modells)
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
            
        # Konvertiere zu Torch Tensor
        x = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)  # hinzufügen der Batch-Dimension
        x = x.to(self.device)
        
        try:
            # Modellspezifisches Setup für α,β-CROWN
            bounded_model = BoundedModule(self.model, x)
            
            # Definiere Parameter für die Störung (Perturbation)
            ptb = PerturbationLpNorm(norm=self.norm_type, eps=epsilon)
            
            # Erstelle ein BoundedTensor für die Eingabe
            bounded_input = BoundedTensor(x, ptb)
            
            # Forward-Pass durch das Modell
            prediction = bounded_model(bounded_input)
            pred_label = prediction.argmax(dim=1).item()
            
            # Wenn die Vorhersage falsch ist, dann ist das Modell nicht robust
            if pred_label != true_class:
                end_time = time.time()
                return VerificationResult(
                    verified=False,
                    property_type=VerificationProperty.ROBUSTNESS,
                    time_seconds=end_time - start_time,
                    details={
                        "error": "Modell hat falsche Vorhersage für ungestörte Eingabe gemacht",
                        "predicted": int(pred_label),
                        "expected": true_class
                    }
                )
            
            # Berechne die Bounds für alle Klassen
            if self.verify_backend == 'beta-crown':
                bounds = bounded_model.compute_bounds(
                    x=(bounded_input,), method='crown-ibp', **self._bound_opts
                )
            else:
                bounds = bounded_model.compute_bounds(x=(bounded_input,), method='IBP+backward')
            
            # Extrahiere untere Schranken der Logits
            lower_bounds = bounds[0][0]
            
            # Minimum der Differenz zwischen dem Logit der wahren Klasse und allen anderen
            robust = True
            lb_diff_mins = []  # Differenz zwischen true_class und anderen Klassen
            
            for i in range(len(CLASS_NAMES)):
                if i != true_class:
                    # (true_class - i): Wenn positiv, wird true_class höher bewertet als i
                    lb_diff = lower_bounds[0, true_class] - lower_bounds[0, i]
                    lb_diff_mins.append(lb_diff.item())
                    if lb_diff <= 0:
                        robust = False
            
            end_time = time.time()
            
            if robust:
                return VerificationResult(
                    verified=True,
                    property_type=VerificationProperty.ROBUSTNESS,
                    time_seconds=end_time - start_time,
                    details={
                        "epsilon": epsilon, 
                        "norm": self.norm_type,
                        "min_logit_diff": min(lb_diff_mins) if lb_diff_mins else None
                    }
                )
            else:
                # In diesem Fall ist die Eigenschaft nicht verifiziert, aber wir haben
                # kein konkretes Gegenbeispiel (wir wissen nur, dass es existiert)
                return VerificationResult(
                    verified=False,
                    property_type=VerificationProperty.ROBUSTNESS,
                    time_seconds=end_time - start_time,
                    details={
                        "epsilon": epsilon, 
                        "norm": self.norm_type,
                        "min_logit_diff": min(lb_diff_mins) if lb_diff_mins else None
                    }
                )
                
        except Exception as e:
            end_time = time.time()
            logger.error(f"Fehler bei der Robustheitsverifikation: {str(e)}")
            return VerificationResult(
                verified=False,
                property_type=VerificationProperty.ROBUSTNESS,
                time_seconds=end_time - start_time,
                details={"error": str(e)}
            )
    
    def verify_brightness_invariance(
        self,
        input_image: np.ndarray,
        true_class: int,
        brightness_range: Tuple[float, float] = (0.8, 1.2)
    ) -> VerificationResult:
        """
        Verifiziert die Invarianz des Modells gegenüber Helligkeitsänderungen.
        
        Überprüft, ob das Modell für das Eingabebild bei verschiedenen Helligkeitsstufen
        innerhalb des angegebenen Bereichs konsistente Vorhersagen macht.
        
        Args:
            input_image: Das Eingabebild als numpy-Array
            true_class: Die korrekte Klasse des Bildes
            brightness_range: Der zu überprüfende Helligkeitsbereich als Faktor (min, max)
            
        Returns:
            Ein VerificationResult-Objekt mit dem Ergebnis der Verifikation
        """
        start_time = time.time()
        
        # Prüfe Bildformat und konvertiere nach Bedarf
        if input_image.shape[-1] == 3:  # Format: H,W,C
            input_image = np.transpose(input_image, (2, 0, 1))  # -> C,H,W
        
        # Normalisiere Bild sofern nötig
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
            
        # Konvertiere zu Torch Tensor
        x = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)
        
        try:
            # Erstelle das erweiterte Modell für die formale Verifikation
            # Wir modellieren Helligkeitsänderungen als lineare Transformation der Eingabe
            
            # Erstelle das bounded model mit der ursprünglichen Eingabe
            bounded_model = BoundedModule(self.model, x)
            
            # Definiere das minimale und maximale Bild
            brightness_min, brightness_max = brightness_range
            x_min = x * brightness_min
            x_max = x * brightness_max
            
            # Stelle sicher, dass die Werte im gültigen Bereich [0,1] liegen
            x_min = torch.clamp(x_min, 0.0, 1.0)
            x_max = torch.clamp(x_max, 0.0, 1.0)
            
            # Erstelle ein BoundedTensor mit den berechneten Grenzen
            bounded_input = BoundedTensor(x, None)
            bounded_input.ptb = ptb = PerturbationLpNorm(norm=np.inf, x_L=x_min, x_U=x_max)
            
            # Forward-Pass durch das Modell mit der ursprünglichen Eingabe
            with torch.no_grad():
                prediction = self.model(x)
            pred_label = prediction.argmax(dim=1).item()
            
            # Wenn die Vorhersage falsch ist, dann ist die Verifikation sinnlos
            if pred_label != true_class:
                end_time = time.time()
                return VerificationResult(
                    verified=False,
                    property_type=VerificationProperty.BRIGHTNESS_INVARIANCE,
                    time_seconds=end_time - start_time,
                    details={
                        "error": "Modell hat falsche Vorhersage für ungestörte Eingabe gemacht",
                        "predicted": int(pred_label),
                        "expected": true_class
                    }
                )
            
            # Berechne bounds für das erweiterte Modell
            if self.verify_backend == 'beta-crown':
                bounds = bounded_model.compute_bounds(
                    x=(bounded_input,), method='crown-ibp', **self._bound_opts
                )
            else:
                bounds = bounded_model.compute_bounds(
                    x=(bounded_input,), method='IBP+backward'
                )
            
            # Extrahiere untere Schranken der Logits
            lower_bounds = bounds[0][0]
            
            # Überprüfe, ob die wahre Klasse immer die höchste Wahrscheinlichkeit hat
            invariant = True
            lb_diff_mins = []
            
            for i in range(len(CLASS_NAMES)):
                if i != true_class:
                    lb_diff = lower_bounds[0, true_class] - lower_bounds[0, i]
                    lb_diff_mins.append(lb_diff.item())
                    if lb_diff <= 0:
                        invariant = False
            
            end_time = time.time()
            
            if invariant:
                return VerificationResult(
                    verified=True,
                    property_type=VerificationProperty.BRIGHTNESS_INVARIANCE,
                    time_seconds=end_time - start_time,
                    details={
                        "brightness_range": brightness_range,
                        "min_logit_diff": min(lb_diff_mins) if lb_diff_mins else None
                    }
                )
            else:
                return VerificationResult(
                    verified=False,
                    property_type=VerificationProperty.BRIGHTNESS_INVARIANCE,
                    time_seconds=end_time - start_time,
                    details={
                        "brightness_range": brightness_range,
                        "min_logit_diff": min(lb_diff_mins) if lb_diff_mins else None
                    }
                )
                
        except Exception as e:
            end_time = time.time()
            logger.error(f"Fehler bei der Helligkeitsinvarianz-Verifikation: {str(e)}")
            return VerificationResult(
                verified=False,
                property_type=VerificationProperty.BRIGHTNESS_INVARIANCE,
                time_seconds=end_time - start_time,
                details={"error": str(e)}
            )
            
    def verify_class_separation(
        self,
        class1: int,
        class2: int,
        examples: List[np.ndarray],
        robustness_eps: float = 0.03
    ) -> VerificationResult:
        """
        Verifiziert die strikte Trennung zwischen zwei kritischen Klassen.
        
        Diese Eigenschaft prüft, ob bestimmte Klassen niemals verwechselt werden,
        selbst bei kleinen Störungen. Besonders wichtig für Sicherheitseigenschaften
        wie "niemals eine rohe Pizza als durchgebacken klassifizieren".
        
        Args:
            class1: Index der ersten Klasse (z.B. "roh")
            class2: Index der zweiten Klasse (z.B. "durchgebacken")
            examples: Liste von Beispielbildern für die Verifikation
            robustness_eps: Störungsparameter für die Robustheitsprüfung
            
        Returns:
            Ein VerificationResult-Objekt mit dem Ergebnis der Verifikation
        """
        start_time = time.time()
        
        if not examples:
            end_time = time.time()
            return VerificationResult(
                verified=False,
                property_type=VerificationProperty.CLASS_SEPARATION,
                time_seconds=end_time - start_time,
                details={"error": "Keine Beispiele für die Verifikation angegeben"}
            )
        
        # Überprüfe, ob die beiden Klassen im gültigen Bereich liegen
        if not (0 <= class1 < len(CLASS_NAMES)) or not (0 <= class2 < len(CLASS_NAMES)):
            end_time = time.time()
            return VerificationResult(
                verified=False,
                property_type=VerificationProperty.CLASS_SEPARATION,
                time_seconds=end_time - start_time,
                details={
                    "error": f"Ungültige Klassenindizes: {class1}, {class2}. "
                            f"Gültiger Bereich: 0-{len(CLASS_NAMES)-1}"
                }
            )
            
        # Prüfe für jedes Beispielbild die Robustheit zwischen den beiden Klassen
        all_verified = True
        results = []
        
        for i, example in enumerate(examples):
            try:
                # Prüfe zuerst ohne Störung, ob das Bild korrekt klassifiziert wird
                # und ob die Klassenwahrscheinlichkeit deutlich über der anderen liegt
                
                # Prüfe Bildformat und konvertiere nach Bedarf
                img = example.copy()
                if img.shape[-1] == 3:  # Format: H,W,C
                    img = np.transpose(img, (2, 0, 1))  # -> C,H,W
                
                # Normalisiere Bild sofern nötig
                if img.max() > 1.0:
                    img = img / 255.0
                    
                # Konvertiere zu Torch Tensor
                x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                x = x.to(self.device)
                
                # Forward-Pass durch das Modell
                with torch.no_grad():
                    prediction = self.model(x)
                pred_label = prediction.argmax(dim=1).item()
                
                # Wenn das Bild falsch klassifiziert wird, überspringen
                if pred_label != class1 and pred_label != class2:
                    continue
                
                # Erstelle das bounded model mit der Eingabe
                bounded_model = BoundedModule(self.model, x)
                
                # Definiere Parameter für die Störung
                ptb = PerturbationLpNorm(norm=self.norm_type, eps=robustness_eps)
                
                # Erstelle ein BoundedTensor für die Eingabe
                bounded_input = BoundedTensor(x, ptb)
                
                # Berechne bounds
                if self.verify_backend == 'beta-crown':
                    bounds = bounded_model.compute_bounds(
                        x=(bounded_input,), method='crown-ibp', **self._bound_opts
                    )
                else:
                    bounds = bounded_model.compute_bounds(
                        x=(bounded_input,), method='IBP+backward'
                    )
                
                # Extrahiere untere Schranken der Logits
                lower_bounds = bounds[0][0]
                
                # Berechne die Differenz zwischen den beiden Klassen
                if pred_label == class1:
                    # class1 sollte größer sein als class2
                    lb_diff = lower_bounds[0, class1] - lower_bounds[0, class2]
                    if lb_diff <= 0:
                        all_verified = False
                else:
                    # class2 sollte größer sein als class1
                    lb_diff = lower_bounds[0, class2] - lower_bounds[0, class1]
                    if lb_diff <= 0:
                        all_verified = False
                
                results.append({
                    "example": i,
                    "predicted": pred_label,
                    "verified": lb_diff > 0,
                    "lb_diff": lb_diff.item()
                })
                
            except Exception as e:
                logger.error(f"Fehler bei der Klassentrennung-Verifikation für Beispiel {i}: {str(e)}")
                all_verified = False
                results.append({
                    "example": i,
                    "error": str(e)
                })
        
        end_time = time.time()
        
        return VerificationResult(
            verified=all_verified,
            property_type=VerificationProperty.CLASS_SEPARATION,
            time_seconds=end_time - start_time,
            details={
                "class1": class1,
                "class2": class2,
                "class1_name": CLASS_NAMES[class1],
                "class2_name": CLASS_NAMES[class2],
                "robustness_eps": robustness_eps,
                "results": results
            }
        )
        
    def verify_all_properties(
        self,
        input_images: List[np.ndarray],
        true_classes: List[int],
        critical_class_pairs: List[Tuple[int, int]] = None,
        robustness_eps: float = 0.01,
        brightness_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Dict[str, List[VerificationResult]]:
        """
        Führt eine umfassende Verifikation aller Eigenschaften durch.
        
        Args:
            input_images: Liste von Eingabebildern
            true_classes: Liste der korrekten Klassen für jedes Bild
            critical_class_pairs: Liste von Paaren kritischer Klassen (z.B. [(0, 2), (1, 3)])
            robustness_eps: Epsilon für Robustheitsprüfung
            brightness_range: Bereich für Helligkeitsprüfung
            
        Returns:
            Ein Dictionary mit Verifikationsergebnissen für jede Eigenschaft
        """
        if len(input_images) != len(true_classes):
            raise ValueError("Anzahl der Bilder und Klassen muss übereinstimmen")
            
        results = {
            VerificationProperty.ROBUSTNESS.value: [],
            VerificationProperty.BRIGHTNESS_INVARIANCE.value: [],
            VerificationProperty.CLASS_SEPARATION.value: []
        }
        
        # Robustheit und Helligkeitsinvarianz für jedes Bild prüfen
        for i, (img, cls) in enumerate(zip(input_images, true_classes)):
            logger.info(f"Prüfe Robustheit für Bild {i+1}/{len(input_images)}")
            
            robustness_result = self.verify_robustness(
                input_image=img,
                true_class=cls,
                epsilon=robustness_eps
            )
            results[VerificationProperty.ROBUSTNESS.value].append(robustness_result)
            
            logger.info(f"Prüfe Helligkeitsinvarianz für Bild {i+1}/{len(input_images)}")
            
            brightness_result = self.verify_brightness_invariance(
                input_image=img,
                true_class=cls,
                brightness_range=brightness_range
            )
            results[VerificationProperty.BRIGHTNESS_INVARIANCE.value].append(brightness_result)
        
        # Klassentrennung für kritische Klassenpaare prüfen
        if critical_class_pairs:
            for class1, class2 in critical_class_pairs:
                logger.info(f"Prüfe Klassentrennung für Klassen {CLASS_NAMES[class1]} und {CLASS_NAMES[class2]}")
                
                # Wähle Beispiele für diese Klassen aus
                examples_class1 = [img for img, cls in zip(input_images, true_classes) if cls == class1]
                examples_class2 = [img for img, cls in zip(input_images, true_classes) if cls == class2]
                
                # Prüfe Trennung in beide Richtungen
                if examples_class1:
                    separation_result1 = self.verify_class_separation(
                        class1=class1,
                        class2=class2,
                        examples=examples_class1,
                        robustness_eps=robustness_eps
                    )
                    results[VerificationProperty.CLASS_SEPARATION.value].append(separation_result1)
                
                if examples_class2:
                    separation_result2 = self.verify_class_separation(
                        class1=class2,
                        class2=class1,
                        examples=examples_class2,
                        robustness_eps=robustness_eps
                    )
                    results[VerificationProperty.CLASS_SEPARATION.value].append(separation_result2)
        
        return results
    
    def generate_verification_report(
        self,
        results: Dict[str, List[VerificationResult]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generiert einen strukturierten Verifikationsbericht.
        
        Args:
            results: Die Ergebnisse der Verifikation
            output_path: Pfad, unter dem der Bericht gespeichert werden soll (optional)
            
        Returns:
            Ein Dictionary mit dem vollständigen Bericht
        """
        import json
        from datetime import datetime
        
        # Erstelle Bericht-Struktur
        report = {
            "model_name": type(self.model).__name__,
            "verification_date": datetime.now().isoformat(),
            "properties": {},
            "summary": {}
        }
        
        # Verarbeite Ergebnisse pro Eigenschaft
        for prop_name, prop_results in results.items():
            prop_summary = {
                "total": len(prop_results),
                "verified": sum(1 for r in prop_results if r.verified),
                "failed": sum(1 for r in prop_results if not r.verified),
                "verification_rate": 0.0,
                "avg_time": 0.0,
                "details": []
            }
            
            if prop_results:
                prop_summary["verification_rate"] = prop_summary["verified"] / prop_summary["total"]
                prop_summary["avg_time"] = sum(r.time_seconds for r in prop_results) / len(prop_results)
            
            # Detaillierte Ergebnisse hinzufügen
            for i, result in enumerate(prop_results):
                prop_summary["details"].append(result.to_dict())
            
            report["properties"][prop_name] = prop_summary
        
        # Gesamtzusammenfassung
        all_results = [r for results_list in results.values() for r in results_list]
        report["summary"] = {
            "total_properties_checked": len(all_results),
            "total_verified": sum(1 for r in all_results if r.verified),
            "total_failed": sum(1 for r in all_results if not r.verified),
            "overall_verification_rate": (
                sum(1 for r in all_results if r.verified) / len(all_results)
                if all_results else 0.0
            ),
            "total_time_seconds": sum(r.time_seconds for r in all_results)
        }
        
        # Speichern, falls Pfad angegeben
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        
        return report


def load_model_for_verification(
    model_path: str,
    model_type: str = 'MicroPizzaNet',
    num_classes: int = 6,
    device: str = 'cpu'
) -> nn.Module:
    """
    Lädt ein vortrainiertes Modell für die formale Verifikation.
    
    Args:
        model_path: Pfad zur Modelldatei (.pth)
        model_type: Typ des Modells ('MicroPizzaNet', 'MicroPizzaNetV2', 'MicroPizzaNetWithSE')
        num_classes: Anzahl der Klassen
        device: Zielgerät ('cpu' oder 'cuda')
    
    Returns:
        Das geladene PyTorch-Modell
    """
    # Modellarchitektur auswählen
    if model_type == 'MicroPizzaNet':
        model = MicroPizzaNet(num_classes=num_classes)
    elif model_type == 'MicroPizzaNetV2':
        model = MicroPizzaNetV2(num_classes=num_classes)
    elif model_type == 'MicroPizzaNetWithSE':
        model = MicroPizzaNetWithSE(num_classes=num_classes)
    else:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")
    
    # Lade Gewichte
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    
    return model