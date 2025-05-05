#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo-Skript für die Statusanzeige des Pizza-Erkennungssystems

Dieses Skript demonstriert die Verwendung der Statusanzeige (RGB-LED oder OLED)
während eines simulierten Pizza-Erkennungsprozesses.

Verwendung:
    python scripts/demo_status_display.py [--display-type TYPE] [--hardware]

Optionen:
    --display-type: Typ der Anzeige ('rgb_led', 'oled', 'simulated', 'auto')
    --hardware: Bei Verwendung echter Hardware-Pins (sonst Simulation)
    --red-pin: GPIO-Pin für rote LED-Komponente (Standard: 16)
    --green-pin: GPIO-Pin für grüne LED-Komponente (Standard: 17)
    --blue-pin: GPIO-Pin für blaue LED-Komponente (Standard: 18)
    --common-anode: Verwende LED mit gemeinsamer Anode (Standard: True)
    --i2c-id: I2C-Bus-ID für OLED (Standard: 0)
    --sda-pin: SDA-Pin für OLED (Standard: 0)
    --scl-pin: SCL-Pin für OLED (Standard: 1)
"""

import os
import sys
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Füge das Projektverzeichnis zum Pythonpfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importiere Module aus dem Pizza-Projekt
from src.status_display import (
    StatusDisplay, SimulatedRGBLED, RP2040RGBLED, OLEDDisplay,
    create_status_display, LEDState
)
from src.types import InferenceResult
from src.constants import CLASS_NAMES

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("status_display_demo")

# Standard-Klassen für Pizza-Erkennung
PIZZA_CLASSES = [
    "basic",     # Roher Teig/Grundzustand
    "burnt",     # Verbrannt
    "combined",  # Kombinierte Zustände
    "mixed",     # Mischzustand
    "progression", # Prozessphase
    "segment"    # Einzelnes Segment
]

# Verwende die Klassen aus der Konstanten-Datei, wenn vorhanden
if CLASS_NAMES:
    PIZZA_CLASSES = CLASS_NAMES

def simulate_pizza_detection(display: StatusDisplay, num_iterations: int = 5,
                            delay: float = 2.0, with_errors: bool = True):
    """
    Simuliert einen Pizza-Erkennungsprozess mit Statusanzeige
    
    Args:
        display: Die zu verwendende Statusanzeige
        num_iterations: Anzahl der Erkennungsdurchläufe
        delay: Verzögerung zwischen den Statusänderungen (in Sekunden)
        with_errors: Ob gelegentlich Fehler simuliert werden sollen
    """
    try:
        # System-Start
        logger.info("System startet...")
        display.show_status("System startet...", (0, 0, 255), LEDState.PULSE)
        time.sleep(delay)
        
        # Initialisierung der Kamera
        logger.info("Kamera wird initialisiert...")
        display.show_status("Kamera initialisieren", (0, 255, 255), LEDState.BLINK_SLOW)
        time.sleep(delay)
        
        # Modell laden
        logger.info("Pizza-Erkennungsmodell wird geladen...")
        display.show_status("Modell laden", (255, 165, 0), LEDState.BLINK_SLOW)
        time.sleep(delay)
        
        # System bereit
        logger.info("System bereit für Pizza-Erkennung")
        display.show_status("System bereit", (0, 255, 0), LEDState.ON)
        time.sleep(delay)
        
        # Erkennungsschleife
        for i in range(num_iterations):
            # Simuliere gelegentlich einen Fehler
            if with_errors and random.random() < 0.2:
                error_types = [
                    "Kamera nicht gefunden",
                    "Bild zu dunkel",
                    "Keine Pizza erkannt",
                    "Inferenz-Timeout"
                ]
                error = random.choice(error_types)
                logger.error(f"Fehler: {error}")
                display.show_error(error)
                time.sleep(delay)
                continue
                
            # Simuliere eine Aufnahme
            logger.info(f"Bild {i+1} wird aufgenommen...")
            display.show_status("Bild aufnehmen", (0, 0, 255), LEDState.BLINK_FAST)
            time.sleep(delay / 2)
            
            # Simuliere die Vorverarbeitung
            logger.info("Bild wird vorverarbeitet...")
            display.show_status("Vorverarbeitung", (0, 255, 255), LEDState.ON)
            time.sleep(delay / 2)
            
            # Simuliere die Inferenz
            logger.info("Inferenz wird durchgeführt...")
            display.show_status("Inferenz läuft", (255, 165, 0), LEDState.ON)
            time.sleep(delay / 2)
            
            # Zufälliges Erkennungsergebnis generieren
            class_idx = random.randint(0, len(PIZZA_CLASSES)-1)
            class_name = PIZZA_CLASSES[class_idx]
            
            # Zufällige Konfidenz (gewichtet je nach Klasse)
            if class_name == "burnt":
                # "Burnt" wird mit höherer Konfidenz erkannt
                confidence = random.uniform(0.8, 0.98)
            elif class_name == "mixed":
                # "Mixed" wird mit niedrigerer Konfidenz erkannt
                confidence = random.uniform(0.55, 0.85)
            else:
                # Normale Konfidenzverteilung
                confidence = random.uniform(0.6, 0.95)
            
            # Erstelle simulierte Wahrscheinlichkeiten für alle Klassen
            probabilities = {}
            remaining_prob = 1.0 - confidence
            for idx, name in enumerate(PIZZA_CLASSES):
                if idx == class_idx:
                    probabilities[name] = confidence
                else:
                    # Verteile restliche Wahrscheinlichkeit zufällig
                    prob = remaining_prob * random.random()
                    remaining_prob -= prob
                    probabilities[name] = prob
            
            # Letzte Klasse bekommt den Rest (um Summe 1.0 zu garantieren)
            last_idx = len(PIZZA_CLASSES) - 1
            if last_idx != class_idx:
                last_class = PIZZA_CLASSES[last_idx]
                probabilities[last_class] += remaining_prob
            
            # Erstelle Inferenzergebnis
            result = InferenceResult(
                class_name=class_name,
                confidence=confidence,
                prediction=class_idx,
                probabilities=probabilities
            )
            
            # Zeige Ergebnis an
            logger.info(f"Pizza erkannt: {class_name} mit {confidence:.2f} Konfidenz")
            display.show_inference_result(result)
            
            # Warte vor nächster Iteration
            time.sleep(delay)
        
        # Ende der Simulation
        logger.info("Demo beendet.")
        display.show_status("Demo beendet", (0, 255, 0), LEDState.PULSE)
        time.sleep(delay)
        
    finally:
        # Aufräumen
        display.close()


def parse_args():
    """Parst die Kommandozeilenargumente"""
    parser = argparse.ArgumentParser(description="Demo für die Statusanzeige des Pizza-Erkennungssystems")
    
    parser.add_argument('--display-type', choices=['rgb_led', 'oled', 'simulated', 'auto'], 
                        default='simulated', help="Typ der Statusanzeige")
    parser.add_argument('--hardware', action='store_true', 
                        help="Verwende echte Hardware-Pins (sonst Simulation)")
    
    # RGB-LED-Parameter
    parser.add_argument('--red-pin', type=int, default=16, 
                        help="GPIO-Pin für rote LED-Komponente")
    parser.add_argument('--green-pin', type=int, default=17, 
                        help="GPIO-Pin für grüne LED-Komponente")
    parser.add_argument('--blue-pin', type=int, default=18, 
                        help="GPIO-Pin für blaue LED-Komponente")
    parser.add_argument('--common-anode', action='store_true', default=True,
                        help="Verwende LED mit gemeinsamer Anode")
    
    # OLED-Display-Parameter
    parser.add_argument('--i2c-id', type=int, default=0, 
                        help="I2C-Bus-ID für OLED-Display")
    parser.add_argument('--sda-pin', type=int, default=0, 
                        help="SDA-Pin für OLED-Display")
    parser.add_argument('--scl-pin', type=int, default=1, 
                        help="SCL-Pin für OLED-Display")
    
    # Demo-Parameter
    parser.add_argument('--iterations', type=int, default=10, 
                        help="Anzahl der Erkennungsdurchläufe")
    parser.add_argument('--delay', type=float, default=2.0, 
                        help="Verzögerung zwischen den Statusänderungen (in Sekunden)")
    parser.add_argument('--no-errors', action='store_true',
                        help="Keine Fehler simulieren")
    
    return parser.parse_args()


def main():
    """Hauptfunktion"""
    args = parse_args()
    
    # Force simulated mode unless --hardware is specified
    if not args.hardware and args.display_type != 'simulated':
        logger.warning("Ohne --hardware wird der simulierte Modus verwendet")
        display_type = 'simulated'
    else:
        display_type = args.display_type
    
    # Display-Optionen
    display_options = {}
    
    # RGB-LED-Optionen
    if display_type == 'rgb_led':
        display_options = {
            'red_pin': args.red_pin,
            'green_pin': args.green_pin,
            'blue_pin': args.blue_pin,
            'common_anode': args.common_anode
        }
    
    # OLED-Display-Optionen
    elif display_type == 'oled':
        display_options = {
            'i2c_id': args.i2c_id,
            'sda_pin': args.sda_pin,
            'scl_pin': args.scl_pin
        }
    
    # Statusanzeige erstellen
    try:
        display = create_status_display(display_type, **display_options)
        
        # Demo starten
        simulate_pizza_detection(
            display,
            num_iterations=args.iterations,
            delay=args.delay,
            with_errors=not args.no_errors
        )
        
    except KeyboardInterrupt:
        logger.info("Demo durch Benutzer abgebrochen.")
    except Exception as e:
        logger.error(f"Fehler: {e}")
        if display_type != 'simulated':
            logger.error("Möglicherweise ist die benötigte Hardware nicht angeschlossen.")
            logger.error("Verwenden Sie '--display-type simulated' für den Simulationsmodus.")


if __name__ == "__main__":
    main()