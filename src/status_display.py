#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status-Anzeige für das Pizza-Erkennungssystem

Dieses Modul ermöglicht die Ansteuerung einer RGB-LED oder eines OLED-Displays
zur Anzeige von Status-Informationen des Pizza-Erkennungssystems.

Es unterstützt sowohl Software-Simulation für Testzwecke als auch
die direkte Hardwareansteuerung auf dem RP2040.

Autor: Pizza Detection Team
Datum: 2025-05-05
"""

import time
import logging
from enum import Enum
from typing import Dict, Tuple, Optional, Union, Any

# Importiere Module aus dem Pizza-Projekt
from src.types import InferenceResult
from src.constants import CLASS_NAMES, CLASS_COLORS

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Typaliase
RGBColor = Tuple[int, int, int]  # (R, G, B) mit Werten von 0-255

# Standard-Farben für jede Pizza-Klasse
DEFAULT_CLASS_COLORS: Dict[str, RGBColor] = {
    "basic": (0, 255, 0),       # Grün: Roher Teig/Grundzustand
    "burnt": (255, 0, 0),       # Rot: Verbrannt
    "combined": (255, 165, 0),  # Orange: Kombinierte Zustände
    "mixed": (128, 0, 128),     # Lila: Mischzustand
    "progression": (0, 0, 255), # Blau: Prozessphase
    "segment": (255, 255, 0)    # Gelb: Einzelnes Segment
}

# Verwende die Farben aus constants.py, wenn vorhanden
if CLASS_COLORS:
    DEFAULT_CLASS_COLORS.update(CLASS_COLORS)

# Status-LED-Zustände
class LEDState(Enum):
    OFF = "off"
    ON = "on"
    BLINK_SLOW = "blink_slow"  # 1Hz
    BLINK_FAST = "blink_fast"  # 2Hz
    PULSE = "pulse"            # Sanftes Pulsieren


class StatusDisplay:
    """Basisklasse für Status-Anzeigen (LED oder Display)"""
    
    def __init__(self):
        """Initialisiert die Status-Anzeige"""
        self.class_colors = DEFAULT_CLASS_COLORS.copy()
        self._is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialisiert die Hardware. Muss von Unterklassen implementiert werden.
        
        Returns:
            bool: True bei erfolgreicher Initialisierung, sonst False
        """
        raise NotImplementedError("Basisklasse implementiert keine Hardware-Initialisierung")
    
    def set_class_colors(self, color_map: Dict[str, RGBColor]) -> None:
        """
        Setzt benutzerdefinierte Farben für Klassen
        
        Args:
            color_map: Dictionary mit Klassen und RGB-Farbwerten
        """
        self.class_colors.update(color_map)
    
    def show_status(self, status: str, color: RGBColor = (255, 255, 255), 
                    state: LEDState = LEDState.ON) -> None:
        """
        Zeigt einen Status mit angegebener Farbe und Zustand an
        
        Args:
            status: Statustext (für Displays) oder Beschreibung (für LEDs)
            color: RGB-Farbwert als Tuple (R, G, B)
            state: Zustand der Anzeige (an, aus, blinkend, etc.)
        """
        raise NotImplementedError("Basisklasse implementiert keine Status-Anzeige")
    
    def show_inference_result(self, result: InferenceResult) -> None:
        """
        Zeigt das Ergebnis einer Inferenz an
        
        Args:
            result: Inferenzergebnis mit Klassenname und Konfidenz
        """
        if result.class_name in self.class_colors:
            color = self.class_colors[result.class_name]
            # Wähle Zustand basierend auf Konfidenz
            if result.confidence > 0.9:
                state = LEDState.ON  # Hohe Konfidenz: Dauerhaft an
            elif result.confidence > 0.6:
                state = LEDState.PULSE  # Mittlere Konfidenz: Pulsieren
            else:
                state = LEDState.BLINK_SLOW  # Niedrige Konfidenz: Langsames Blinken
            
            self.show_status(f"Erkannt: {result.class_name} ({result.confidence:.0%})", 
                            color, state)
        else:
            # Fallback für unbekannte Klassen
            self.show_status(f"Unbekannt: {result.class_name}", (128, 128, 128), LEDState.BLINK_FAST)
    
    def show_error(self, error_message: str) -> None:
        """
        Zeigt einen Fehlerstatus an
        
        Args:
            error_message: Fehlermeldung
        """
        self.show_status(f"Fehler: {error_message}", (255, 0, 0), LEDState.BLINK_FAST)
    
    def clear(self) -> None:
        """Löscht die Anzeige oder schaltet die LED aus"""
        self.show_status("", (0, 0, 0), LEDState.OFF)
    
    def close(self) -> None:
        """Gibt Ressourcen frei"""
        self.clear()
        self._is_initialized = False


class SimulatedRGBLED(StatusDisplay):
    """
    Simulierte RGB-LED für Testzwecke.
    Gibt LED-Zustandsänderungen im Logger aus.
    """
    
    def __init__(self):
        super().__init__()
        self._current_color = (0, 0, 0)
        self._current_state = LEDState.OFF
        self._blinking_thread = None
    
    def initialize(self) -> bool:
        """
        Initialisiert die simulierte LED
        
        Returns:
            bool: Immer True, da keine tatsächliche Hardware
        """
        logger.info("Simulierte RGB-LED initialisiert")
        self._is_initialized = True
        return True
    
    def show_status(self, status: str, color: RGBColor = (255, 255, 255), 
                   state: LEDState = LEDState.ON) -> None:
        """
        Zeigt einen Status auf der simulierten LED an
        
        Args:
            status: Beschreibung des Status
            color: RGB-Farbwert
            state: Zustand der LED
        """
        if not self._is_initialized:
            logger.warning("LED wurde nicht initialisiert")
            return
        
        self._current_color = color
        self._current_state = state
        
        # Protokolliere die Zustandsänderung
        r, g, b = color
        state_str = state.value
        
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        logger.info(f"LED Status: {status} | Farbe: {hex_color} | Zustand: {state_str}")
        
        # In einer echten Anwendung würde hier ein Thread für das Blinken gestartet
        # Für die Simulation geben wir nur eine Nachricht aus
        if state == LEDState.BLINK_SLOW:
            logger.info("LED blinkt langsam (1Hz)")
        elif state == LEDState.BLINK_FAST:
            logger.info("LED blinkt schnell (2Hz)")
        elif state == LEDState.PULSE:
            logger.info("LED pulsiert")


class RP2040RGBLED(StatusDisplay):
    """
    Hardware-Implementation für die RGB-LED auf dem RP2040
    
    Verwendet die GPIO-Pins des RP2040 zur Ansteuerung einer RGB-LED.
    Unterstützt sowohl gemeinsame Anode als auch gemeinsame Kathode LEDs.
    """
    
    def __init__(self, red_pin: int, green_pin: int, blue_pin: int, 
                 common_anode: bool = True):
        """
        Initialisiert die RGB-LED
        
        Args:
            red_pin: GPIO-Pin für die rote Komponente
            green_pin: GPIO-Pin für die grüne Komponente
            blue_pin: GPIO-Pin für die blaue Komponente
            common_anode: True für LED mit gemeinsamer Anode, False für gemeinsame Kathode
        """
        super().__init__()
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self.common_anode = common_anode
        self._is_initialized = False
        self._current_color = (0, 0, 0)
        self._current_state = LEDState.OFF
        self._blinking_thread = None
        self._running = False
    
    def initialize(self) -> bool:
        """
        Initialisiert die GPIO-Pins für die RGB-LED
        
        Returns:
            bool: True bei erfolgreicher Initialisierung, sonst False
        """
        try:
            # Importiere MicroPython-Bibliotheken für RP2040
            # Dies wird nur auf der tatsächlichen Hardware funktionieren
            try:
                import machine
                import time
                self._machine = machine
                self._time = time
            except ImportError:
                logger.error("MicroPython-Bibliotheken nicht gefunden. Läuft auf RP2040?")
                return False
            
            # Konfiguriere GPIO-Pins als Ausgänge
            self._red = machine.PWM(machine.Pin(self.red_pin))
            self._green = machine.PWM(machine.Pin(self.green_pin))
            self._blue = machine.PWM(machine.Pin(self.blue_pin))
            
            # PWM-Frequenz einstellen (typischerweise 1000Hz)
            self._red.freq(1000)
            self._green.freq(1000)
            self._blue.freq(1000)
            
            # LED initial ausschalten
            self.clear()
            
            # Starte Hintergrund-Thread für Blink-/Pulseffekte
            self._running = True
            self._start_effect_thread()
            
            logger.info(f"RGB-LED initialisiert (Pins: R={self.red_pin}, G={self.green_pin}, B={self.blue_pin})")
            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der RGB-LED: {e}")
            return False
    
    def _set_color(self, color: RGBColor) -> None:
        """
        Setzt die RGB-LED auf die angegebene Farbe
        
        Args:
            color: RGB-Farbwert als Tuple (R, G, B)
        """
        if not self._is_initialized:
            return
        
        r, g, b = color
        
        # Skaliere von 0-255 auf 0-65535 (MicroPython PWM-Bereich)
        r_scaled = int(r * 65535 / 255)
        g_scaled = int(g * 65535 / 255)
        b_scaled = int(b * 65535 / 255)
        
        # Bei gemeinsamer Anode sind die Werte invertiert
        if self.common_anode:
            r_scaled = 65535 - r_scaled
            g_scaled = 65535 - g_scaled
            b_scaled = 65535 - b_scaled
        
        # PWM-Duty-Cycle setzen
        self._red.duty_u16(r_scaled)
        self._green.duty_u16(g_scaled)
        self._blue.duty_u16(b_scaled)
    
    def _start_effect_thread(self) -> None:
        """Startet einen Thread für LED-Effekte"""
        try:
            import _thread
            _thread.start_new_thread(self._effect_loop, ())
        except ImportError:
            logger.warning("Thread-Unterstützung nicht verfügbar, LED-Effekte deaktiviert")
    
    def _effect_loop(self) -> None:
        """Hauptschleife für LED-Effekte"""
        while self._running:
            if self._current_state == LEDState.BLINK_SLOW:
                self._set_color(self._current_color)
                self._time.sleep(0.5)
                self._set_color((0, 0, 0))
                self._time.sleep(0.5)
            
            elif self._current_state == LEDState.BLINK_FAST:
                self._set_color(self._current_color)
                self._time.sleep(0.25)
                self._set_color((0, 0, 0))
                self._time.sleep(0.25)
            
            elif self._current_state == LEDState.PULSE:
                # Sanftes Pulsieren mit Sinus-Kurve
                for i in range(20):
                    brightness = i / 20.0
                    r, g, b = self._current_color
                    scaled_color = (int(r * brightness), int(g * brightness), int(b * brightness))
                    self._set_color(scaled_color)
                    self._time.sleep(0.05)
                
                for i in range(20, 0, -1):
                    brightness = i / 20.0
                    r, g, b = self._current_color
                    scaled_color = (int(r * brightness), int(g * brightness), int(b * brightness))
                    self._set_color(scaled_color)
                    self._time.sleep(0.05)
            
            elif self._current_state == LEDState.ON:
                self._set_color(self._current_color)
                self._time.sleep(0.1)  # Kurze Pause zur CPU-Entlastung
            
            elif self._current_state == LEDState.OFF:
                self._set_color((0, 0, 0))
                self._time.sleep(0.1)  # Kurze Pause zur CPU-Entlastung
    
    def show_status(self, status: str, color: RGBColor = (255, 255, 255), 
                   state: LEDState = LEDState.ON) -> None:
        """
        Zeigt einen Status auf der RGB-LED an
        
        Args:
            status: Beschreibung des Status (wird nicht angezeigt, nur für Logging)
            color: RGB-Farbwert
            state: Zustand der LED
        """
        if not self._is_initialized:
            logger.warning("LED wurde nicht initialisiert")
            return
        
        # Speichere aktuellen Zustand für den Effekt-Thread
        self._current_color = color
        self._current_state = state
        
        # Direktes Setzen der Farbe für sofortige Änderung
        # Der Effekt-Thread übernimmt dann die Animation
        if state == LEDState.ON:
            self._set_color(color)
        elif state == LEDState.OFF:
            self._set_color((0, 0, 0))
    
    def close(self) -> None:
        """Gibt Ressourcen frei und stoppt den Effekt-Thread"""
        self._running = False
        self.clear()
        
        if hasattr(self, '_red'):
            self._red.deinit()
        if hasattr(self, '_green'):
            self._green.deinit()
        if hasattr(self, '_blue'):
            self._blue.deinit()
        
        self._is_initialized = False
        logger.info("RGB-LED-Ressourcen freigegeben")


class OLEDDisplay(StatusDisplay):
    """
    Implementation für ein OLED-Display (SSD1306 oder ähnlich) über I2C.
    
    Zeigt detailliertere Statusinformationen wie Klassennamen, 
    Konfidenzen und kleine Icons an.
    """
    
    def __init__(self, i2c_id: int = 0, sda_pin: int = 0, scl_pin: int = 1,
                 width: int = 128, height: int = 64, address: int = 0x3C):
        """
        Initialisiert das OLED-Display
        
        Args:
            i2c_id: ID der I2C-Schnittstelle (normalerweise 0 oder 1)
            sda_pin: GPIO-Pin für SDA (Serial Data Line)
            scl_pin: GPIO-Pin für SCL (Serial Clock Line)
            width: Breite des Displays in Pixeln
            height: Höhe des Displays in Pixeln
            address: I2C-Adresse des Displays (typischerweise 0x3C oder 0x3D)
        """
        super().__init__()
        self.i2c_id = i2c_id
        self.sda_pin = sda_pin
        self.scl_pin = scl_pin
        self.width = width
        self.height = height
        self.address = address
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """
        Initialisiert das OLED-Display über I2C
        
        Returns:
            bool: True bei erfolgreicher Initialisierung, sonst False
        """
        try:
            # Importiere MicroPython-Bibliotheken für RP2040
            try:
                import machine
                import ssd1306
                self._machine = machine
                self._ssd1306 = ssd1306
            except ImportError:
                logger.error("MicroPython-Bibliotheken nicht gefunden. Läuft auf RP2040?")
                return False
            
            # I2C-Schnittstelle initialisieren
            i2c = machine.I2C(self.i2c_id, 
                              sda=machine.Pin(self.sda_pin), 
                              scl=machine.Pin(self.scl_pin))
            
            # OLED-Display initialisieren
            self._display = ssd1306.SSD1306_I2C(self.width, self.height, i2c, addr=self.address)
            
            # Display löschen und Startnachricht anzeigen
            self._display.fill(0)
            self._display.text("Pizza Detection", 0, 0, 1)
            self._display.text("System Ready", 0, 20, 1)
            self._display.show()
            
            logger.info(f"OLED-Display initialisiert ({self.width}x{self.height})")
            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des OLED-Displays: {e}")
            return False
    
    def show_status(self, status: str, color: RGBColor = (255, 255, 255),
                   state: LEDState = LEDState.ON) -> None:
        """
        Zeigt einen Status auf dem OLED-Display an
        
        Args:
            status: Anzuzeigender Text
            color: Wird ignoriert (OLED ist monochrom)
            state: Wird für Blink-Effekte verwendet
        """
        if not self._is_initialized:
            logger.warning("Display wurde nicht initialisiert")
            return
        
        try:
            # Display löschen
            self._display.fill(0)
            
            # Status-Text in Zeilen aufteilen (max. 16 Zeichen pro Zeile bei 8x8 Font)
            lines = []
            words = status.split()
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) > 16 and current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Text anzeigen
            for i, line in enumerate(lines[:4]):  # Maximal 4 Zeilen
                self._display.text(line, 0, i * 10, 1)
            
            # Anzeigen
            self._display.show()
            
        except Exception as e:
            logger.error(f"Fehler bei der Anzeige auf dem OLED-Display: {e}")
    
    def show_inference_result(self, result: InferenceResult) -> None:
        """
        Zeigt das Ergebnis einer Inferenz auf dem Display an
        
        Args:
            result: Inferenzergebnis mit Klassenname und Konfidenz
        """
        if not self._is_initialized:
            return
        
        try:
            self._display.fill(0)
            
            # Titel
            self._display.text("Pizza Detected:", 0, 0, 1)
            
            # Klassenname
            if len(result.class_name) > 16:
                # Kürzen, wenn zu lang
                class_name = result.class_name[:14] + ".."
            else:
                class_name = result.class_name
            self._display.text(class_name, 0, 16, 1)
            
            # Konfidenz
            conf_text = f"Conf: {result.confidence*100:.1f}%"
            self._display.text(conf_text, 0, 32, 1)
            
            # Zeitstempel
            time_str = time.strftime("%H:%M:%S")
            self._display.text(time_str, 0, 48, 1)
            
            # Balken für Konfidenz
            bar_width = int(result.confidence * self.width)
            for i in range(bar_width):
                self._display.pixel(i, 42, 1)
            
            self._display.show()
            
        except Exception as e:
            logger.error(f"Fehler bei der Anzeige des Inferenzergebnisses: {e}")
    
    def clear(self) -> None:
        """Löscht das Display"""
        if self._is_initialized:
            try:
                self._display.fill(0)
                self._display.show()
            except Exception as e:
                logger.error(f"Fehler beim Löschen des Displays: {e}")
    
    def close(self) -> None:
        """Gibt Ressourcen frei"""
        self.clear()
        self._is_initialized = False
        logger.info("OLED-Display-Ressourcen freigegeben")


# Globale Helfer-Funktionen zum einfachen Erstellen von Status-Anzeigen

def create_status_display(display_type: str = "auto", **kwargs) -> StatusDisplay:
    """
    Erstellt eine Status-Anzeige passend zur Laufzeitumgebung
    
    Args:
        display_type: "rgb_led", "oled", oder "auto"
        **kwargs: Zusätzliche Parameter für den Constructor
    
    Returns:
        Eine initialisierte StatusDisplay-Instanz
    """
    if display_type == "auto":
        # Automatische Erkennung
        try:
            # Prüfe, ob wir auf einem MicroPython-System laufen
            import machine
            
            # Versuche LED zu erstellen
            display = RP2040RGBLED(
                red_pin=kwargs.get("red_pin", 16),  # Standardwerte anpassen
                green_pin=kwargs.get("green_pin", 17),
                blue_pin=kwargs.get("blue_pin", 18),
                common_anode=kwargs.get("common_anode", True)
            )
            
            if display.initialize():
                return display
            
            # Wenn LED fehlschlägt, versuche OLED
            display = OLEDDisplay(
                i2c_id=kwargs.get("i2c_id", 0),
                sda_pin=kwargs.get("sda_pin", 0),
                scl_pin=kwargs.get("scl_pin", 1),
                width=kwargs.get("width", 128),
                height=kwargs.get("height", 64),
                address=kwargs.get("address", 0x3C)
            )
            
            if display.initialize():
                return display
                
        except ImportError:
            # Keine Machine-Bibliothek, verwende Simulation
            logger.info("Keine Hardwareunterstützung gefunden, verwende Simulation")
        except Exception as e:
            logger.warning(f"Fehler bei automatischer Anzeigeerkennung: {e}")
        
        # Fallback auf Simulation
        display = SimulatedRGBLED()
        display.initialize()
        return display
    
    elif display_type == "rgb_led":
        display = RP2040RGBLED(
            red_pin=kwargs.get("red_pin", 16),
            green_pin=kwargs.get("green_pin", 17),
            blue_pin=kwargs.get("blue_pin", 18),
            common_anode=kwargs.get("common_anode", True)
        )
        
    elif display_type == "oled":
        display = OLEDDisplay(
            i2c_id=kwargs.get("i2c_id", 0),
            sda_pin=kwargs.get("sda_pin", 0),
            scl_pin=kwargs.get("scl_pin", 1),
            width=kwargs.get("width", 128),
            height=kwargs.get("height", 64),
            address=kwargs.get("address", 0x3C)
        )
    
    elif display_type == "simulated":
        display = SimulatedRGBLED()
    
    else:
        logger.warning(f"Unbekannter Display-Typ: {display_type}. Verwende Simulation.")
        display = SimulatedRGBLED()
    
    # Initialisiere die Anzeige
    display.initialize()
    return display


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Erstelle eine simulierte LED für Tests
    display = create_status_display("simulated")
    
    # Zeige verschiedene Status-Meldungen
    print("Demo der Status-Anzeige:")
    
    # System-Start
    print("\nSystem-Start...")
    display.show_status("System startet...", (0, 0, 255), LEDState.PULSE)
    time.sleep(2)
    
    # Initialisierung
    print("\nInitialisierung...")
    display.show_status("Initialisierung...", (0, 255, 255), LEDState.BLINK_SLOW)
    time.sleep(2)
    
    # Bereit
    print("\nSystem bereit...")
    display.show_status("System bereit", (0, 255, 0), LEDState.ON)
    time.sleep(2)
    
    # Inferenz-Ergebnisse
    print("\nSimuliere Inferenz-Ergebnisse:")
    
    # Roher Teig mit hoher Konfidenz
    result1 = InferenceResult(
        class_name="basic",
        confidence=0.95,
        prediction=0,
        probabilities={"basic": 0.95, "burnt": 0.01, "combined": 0.01, 
                      "mixed": 0.01, "progression": 0.01, "segment": 0.01}
    )
    print(f"\nInferenz: {result1.class_name} ({result1.confidence:.2f})")
    display.show_inference_result(result1)
    time.sleep(2)
    
    # Verbrannte Pizza mit mittlerer Konfidenz
    result2 = InferenceResult(
        class_name="burnt",
        confidence=0.75,
        prediction=1,
        probabilities={"basic": 0.05, "burnt": 0.75, "combined": 0.05, 
                      "mixed": 0.05, "progression": 0.05, "segment": 0.05}
    )
    print(f"\nInferenz: {result2.class_name} ({result2.confidence:.2f})")
    display.show_inference_result(result2)
    time.sleep(2)
    
    # Unsichere Erkennung
    result3 = InferenceResult(
        class_name="mixed",
        confidence=0.45,
        prediction=3,
        probabilities={"basic": 0.25, "burnt": 0.15, "combined": 0.10, 
                      "mixed": 0.45, "progression": 0.03, "segment": 0.02}
    )
    print(f"\nInferenz: {result3.class_name} ({result3.confidence:.2f})")
    display.show_inference_result(result3)
    time.sleep(2)
    
    # Fehlermeldung
    print("\nSimuliere Fehler...")
    display.show_error("Kamera nicht gefunden")
    time.sleep(2)
    
    # Aufräumen
    print("\nBeende Demonstration...")
    display.close()
    print("Demo beendet.")