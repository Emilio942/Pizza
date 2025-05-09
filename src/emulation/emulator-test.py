import os
import time
import numpy as np
from PIL import Image
import argparse

class ModelConverter:
    """
    Konvertiert ein PyTorch-Modell in ein Format für RP2040
    """
    def __init__(self):
        self.model_size_bytes = 0
        self.ram_usage_bytes = 0
        self.quantized = False
        self.model_input_size = (96, 96)  # Standardgröße, kann überschrieben werden
        
    def estimate_model_size(self, original_size_mb, quantized=True):
        """
        Schätzt die Größe des konvertierten Modells
        
        Args:
            original_size_mb (float): Größe des PyTorch-Modells in MB
            quantized (bool): Ob das Modell quantisiert werden soll
            
        Returns:
            dict: Informationen über das konvertierte Modell
        
        Raises:
            ValueError: Wenn die Eingabegröße negativ ist
        """
        # Validiere Eingabe
        if original_size_mb < 0:
            raise ValueError("Modellgröße kann nicht negativ sein")
            
        # Konvertierung von MB zu Bytes
        original_size_bytes = original_size_mb * 1024 * 1024
        
        # Bei Quantisierung (float32 zu int8) wird das Modell um ca. Faktor 4 kleiner
        if quantized:
            self.model_size_bytes = int(original_size_bytes / 4)
            self.quantized = True
        else:
            self.model_size_bytes = original_size_bytes
            self.quantized = False
            
        # Schätze RAM-Bedarf (für Aktivierungen während Inferenz)
        # Bei int8-Quantisierung ca. 20% der Modellgröße, bei float32 ca. 50%
        if quantized:
            self.ram_usage_bytes = int(self.model_size_bytes * 0.2)
        else:
            self.ram_usage_bytes = int(self.model_size_bytes * 0.5)
            
        return {
            "model_size_bytes": self.model_size_bytes,
            "ram_usage_bytes": self.ram_usage_bytes,
            "quantized": self.quantized,
            "model_input_size": self.model_input_size
        }
    
    def convert_pytorch_to_tflite(self, pth_path, quantize=True):
        """
        Simuliert die Konvertierung von PyTorch zu TFLite
        
        Args:
            pth_path (str): Pfad zur PyTorch-Modelldatei
            quantize (bool): Ob das Modell quantisiert werden soll
            
        Returns:
            dict: Informationen über das konvertierte Modell
            
        Raises:
            FileNotFoundError: Wenn die Modelldatei nicht gefunden wird
        """
        print(f"Simuliere Konvertierung von {pth_path} zu TFLite...")
        
        # Prüfe, ob die Datei existiert
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Modelldatei {pth_path} nicht gefunden")
        
        # Für die Simulation ermitteln wir die Dateigröße
        original_size_mb = os.path.getsize(pth_path) / (1024 * 1024)
        
        # Simuliere Verarbeitungszeit
        print("Konvertiere von PyTorch zu ONNX...")
        time.sleep(1)
        print("Konvertiere von ONNX zu TensorFlow...")
        time.sleep(1)
        print("Konvertiere" + (" und quantisiere" if quantize else "") + " zu TFLite...")
        time.sleep(1)
        
        # Modellgröße schätzen
        model_info = self.estimate_model_size(original_size_mb, quantized=quantize)
        
        print(f"Konvertierung abgeschlossen!")
        print(f"Ursprüngliche Modellgröße: {original_size_mb:.2f} MB")
        print(f"TFLite-Modell: {model_info['model_size_bytes']/1024/1024:.2f} MB " +
              f"({'Quantisiert (INT8)' if quantize else 'Nicht quantisiert (FLOAT32)'})")
        
        return model_info
    
    def convert_tflite_to_c_array(self, model_info):
        """
        Simuliert die Konvertierung von TFLite zu C-Array für Einbettung in Firmware
        
        Args:
            model_info (dict): Informationen über das konvertierte Modell
            
        Returns:
            str: Simulierter C-Code
        """
        print("Konvertiere TFLite-Modell zu C-Array für Einbettung in Firmware...")
        
        # Validiere model_info
        required_keys = ['model_size_bytes', 'quantized']
        for key in required_keys:
            if key not in model_info:
                raise KeyError(f"model_info fehlt der Schlüssel '{key}'")
        
        # Generiere simulierten C-Code
        c_code = f"""
/* Automatisch generiertes C-Array für ein TFLite-Modell */
#include <stdint.h>

const unsigned int model_tflite_len = {model_info['model_size_bytes']};
const unsigned char model_tflite_data[] = {{
  /* {model_info['model_size_bytes']} Bytes {'quantisiertes' if model_info['quantized'] else 'nicht-quantisiertes'} Modell */
  0x00, 0x00, 0x00, 0x00, /* ... usw. ... */
}};
        """
        
        print(f"C-Array generiert: {model_info['model_size_bytes']} Bytes " +
              f"({'Quantisiert' if model_info['quantized'] else 'Nicht quantisiert'})")
        return c_code


class FirmwareBuilder:
    """
    Simuliert den Build-Prozess der Firmware für RP2040
    """
    def __init__(self):
        self.code_size_bytes = 0
        self.total_firmware_size = 0
        
    def build_firmware(self, model_info, extra_code_kb=50):
        """
        Simuliert den Firmware-Build-Prozess
        
        Args:
            model_info (dict): Informationen über das konvertierte Modell
            extra_code_kb (float): Größe des zusätzlichen Codes in KB
            
        Returns:
            dict: Informationen über die erstellte Firmware
            
        Raises:
            ValueError: Wenn extra_code_kb negativ ist
            KeyError: Wenn model_info nicht alle erforderlichen Schlüssel enthält
        """
        print("\nStarte Firmware-Build-Prozess...")
        
        # Validiere Eingabe
        if extra_code_kb < 0:
            raise ValueError("Code-Größe kann nicht negativ sein")
            
        # Validiere model_info
        required_keys = ['model_size_bytes', 'ram_usage_bytes', 'quantized', 'model_input_size']
        for key in required_keys:
            if key not in model_info:
                raise KeyError(f"model_info fehlt der Schlüssel '{key}'")
        
        # Simuliere C/C++ Code Größe (RP2040 SDK, TFLM Runtime, eigener Code)
        self.code_size_bytes = int(extra_code_kb * 1024)
        
        # Gesamtgröße der Firmware
        self.total_firmware_size = self.code_size_bytes + model_info['model_size_bytes']
        
        print(f"SDK und eigener Code: {self.code_size_bytes/1024:.1f} KB")
        print(f"Neural Network Modell: {model_info['model_size_bytes']/1024:.1f} KB")
        print(f"Gesamtgröße der Firmware: {self.total_firmware_size/1024:.1f} KB")
        
        # Simuliere Kompilierung
        print("\nKompiliere C/C++ Code...")
        time.sleep(0.5)
        print("Linke Objekte...")
        time.sleep(0.5)
        print("Generiere Binärdatei...")
        time.sleep(0.5)
        
        # Erstelle ein simuliertes Firmware-Binary-Objekt
        firmware = {
            "total_size_bytes": self.total_firmware_size,
            "model_size_bytes": model_info['model_size_bytes'],
            "code_size_bytes": self.code_size_bytes,
            "ram_usage_bytes": model_info['ram_usage_bytes'],
            "quantized": model_info['quantized'],
            "model_input_size": model_info['model_input_size'],
            "entry_point": 0x10000000,  # RP2040 typischer Code-Startpunkt
            "timestamp": time.time()
        }
        
        print(f"Firmware erfolgreich gebaut: firmware.bin ({self.total_firmware_size/1024:.1f} KB)")
        return firmware


class RP2040Emulator:
    """
    Erweiterter Emulator für ein RP2040-basiertes Objekterkennungssystem
    mit Unterstützung für Firmware-Simulation
    """
    
    def __init__(self):
        # Hardware-Spezifikationen
        self.cpu_speed_mhz = 133
        self.cores = 2
        self.flash_size_bytes = 2 * 1024 * 1024  # 2MB Flash
        self.ram_size_bytes = 264 * 1024  # 264KB SRAM
        self.used_flash = 0
        self.used_ram = 0
        self.system_ram_overhead = 20 * 1024  # 20KB für System-Overhead
        
        # Firmware
        self.firmware = None
        self.firmware_loaded = False
        
        # Kamera-Einstellungen
        self.camera_resolution = (320, 240)  # QVGA
        self.max_fps = 10
        
        # Stromverbrauch
        self.active_current_ma = 180  # Durchschnitt
        self.standby_current_ma = 0.5
        self.battery_capacity_mah = 1500  # CR123A typische Kapazität
        self.battery_remaining_percent = 100
        
        # GPIO-Zustände
        self.status_led = False
        self.power_led = True
        self.buzzer_active = False
        
        # Bildverarbeitung
        self.current_frame = None
        self.processed_frame = None
        self.model_input_size = (96, 96)  # Standardwert
        
        # Inferenz-Metriken
        self.inference_time_ms = 0
        self.detection_score = 0
        self.detection_threshold = 0.7
        
        # System-Status
        self.system_active = True
        self.deep_sleep = False
        self.start_time = time.time()
        self.operation_time = 0
        
        print("RP2040 System-Emulator gestartet")
        print(f"Flash: {self.flash_size_bytes/1024/1024:.1f}MB, RAM: {self.ram_size_bytes/1024:.1f}KB")
        print(f"CPU: {self.cores} Cores @ {self.cpu_speed_mhz}MHz")
    
    def load_firmware(self, firmware):
        """
        Lädt eine simulierte Firmware in den Emulator
        
        Args:
            firmware (dict): Informationen über die zu ladende Firmware
            
        Returns:
            bool: True, wenn die Firmware erfolgreich geladen wurde, sonst False
            
        Raises:
            KeyError: Wenn firmware nicht alle erforderlichen Schlüssel enthält
        """
        print("\n--- Firmware wird geladen ---")
        
        # Validiere firmware-Objekt
        required_keys = ['total_size_bytes', 'ram_usage_bytes', 'model_input_size']
        for key in required_keys:
            if key not in firmware:
                raise KeyError(f"firmware fehlt der Schlüssel '{key}'")
        
        # Berechne den tatsächlichen RAM-Bedarf (Modell + System-Overhead)
        total_ram_needed = firmware['ram_usage_bytes'] + self.system_ram_overhead
        
        # Prüfe, ob die Firmware in den Flash passt
        if firmware['total_size_bytes'] > self.flash_size_bytes:
            print(f"FEHLER: Firmware zu groß ({firmware['total_size_bytes']/1024:.1f}KB) " +
                  f"für Flash ({self.flash_size_bytes/1024:.1f}KB)")
            return False
        
        # Prüfe, ob der RAM-Bedarf in den verfügbaren RAM passt
        if total_ram_needed > self.ram_size_bytes:
            print(f"FEHLER: Gesamt-RAM-Bedarf ({total_ram_needed/1024:.1f}KB) " +
                  f"überschreitet verfügbaren RAM ({self.ram_size_bytes/1024:.1f}KB)")
            print(f"  - Modell-RAM: {firmware['ram_usage_bytes']/1024:.1f}KB")
            print(f"  - System-Overhead: {self.system_ram_overhead/1024:.1f}KB")
            return False
        
        self.firmware = firmware
        self.firmware_loaded = True
        self.used_flash = firmware['total_size_bytes']
        self.used_ram = total_ram_needed
        
        # Übernehme die Modelleingabegröße aus der Firmware
        if 'model_input_size' in firmware:
            self.model_input_size = firmware['model_input_size']
        
        print(f"Firmware geladen: {firmware['total_size_bytes']/1024:.1f}KB Flash")
        print(f"RAM-Nutzung: {total_ram_needed/1024:.1f}KB " +
              f"({firmware['ram_usage_bytes']/1024:.1f}KB Modell + " +
              f"{self.system_ram_overhead/1024:.1f}KB System)")
        print(f"Verbleibender Flash: {(self.flash_size_bytes - self.used_flash)/1024:.1f}KB")
        print(f"Verbleibender RAM: {(self.ram_size_bytes - self.used_ram)/1024:.1f}KB")
        
        # Initialisiere simulierten RP2040
        print("Initialisiere RP2040...")
        time.sleep(0.5)
        print("Hardware-Peripherie wird konfiguriert...")
        time.sleep(0.5)
        print("Neural Network Runtime wird initialisiert...")
        time.sleep(0.5)
        print("System bereit!")
        
        return True
    
    def calculate_inference_time(self):
        """
        Berechnet die simulierte Inferenz-Zeit basierend auf Modellgröße und Quantisierung
        
        Returns:
            float: Geschätzte Inferenzzeit in Millisekunden
        """
        if not self.firmware_loaded:
            return 200  # Standardwert
        
        model_size_kb = self.firmware['model_size_bytes'] / 1024
        
        # Einfache Formel für die Simulation: 
        # Größeres Modell = längere Inferenzzeit
        # Quantisierung = schnellere Inferenzzeit
        if self.firmware['quantized']:
            # Quantisierte Modelle sind ~4x schneller
            base_time = model_size_kb * 0.05  # 0.05ms pro KB
        else:
            base_time = model_size_kb * 0.2   # 0.2ms pro KB
        
        # Füge etwas Varianz hinzu
        variance = np.random.normal(0, base_time * 0.1)
        
        # Mindestens 50ms, höchstens 500ms
        return min(500, max(50, base_time + variance))
    
    def capture_image(self, input_image_path=None):
        """
        Simuliert die Bildaufnahme vom OV2640 Sensor
        
        Args:
            input_image_path (str, optional): Pfad zu einem Testbild
            
        Returns:
            bool: True, wenn die Bildaufnahme erfolgreich war, sonst False
        """
        if not self.firmware_loaded:
            print("FEHLER: Keine Firmware geladen.")
            return False
            
        if input_image_path and os.path.exists(input_image_path):
            try:
                # Lade ein Testbild für die Simulation
                img = Image.open(input_image_path)
                # Skaliere auf Kameraauflösung
                img = img.resize(self.camera_resolution)
                # Konvertiere zu Numpy-Array
                self.current_frame = np.array(img)
                print(f"Bild erfasst: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
                return True
            except (IOError, PermissionError) as e:
                print(f"Fehler beim Laden des Bildes: {e}")
                return False
            except Exception as e:
                print(f"Unerwarteter Fehler beim Laden des Bildes: {e}")
                return False
        else:
            if input_image_path:
                print(f"WARNUNG: Bilddatei {input_image_path} nicht gefunden. Verwende simuliertes Bild.")
            # Erzeuge ein Zufallsbild wenn kein Eingabebild vorhanden
            self.current_frame = np.random.randint(0, 255, (*self.camera_resolution, 3), dtype=np.uint8)
            print(f"Simuliertes Zufallsbild erzeugt: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
            return True
    
    def preprocess_image(self):
        """
        Simuliert die Bildvorverarbeitung
        
        Returns:
            bool: True, wenn die Vorverarbeitung erfolgreich war, sonst False
        """
        if self.current_frame is None:
            print("FEHLER: Kein Bild zum Verarbeiten vorhanden.")
            return False
        
        # Simulierte Vorverarbeitung: Graustufenkonvertierung, Skalierung
        if len(self.current_frame.shape) == 3:
            gray = np.mean(self.current_frame, axis=2).astype(np.uint8)
        else:
            gray = self.current_frame
            
        # Verwende die Modelleingabegröße aus der Firmware oder Standardwert
        model_input_size = self.model_input_size
        
        # Tatsächliche Skalierung des Bildes auf die Modelleingabegröße
        try:
            # Konvertiere zu PIL für bessere Skalierung
            pil_image = Image.fromarray(gray)
            resized_image = pil_image.resize(model_input_size, Image.BICUBIC)
            # Zurück zu Numpy-Array
            self.processed_frame = np.array(resized_image)
            
            print(f"Bild vorverarbeitet: Graustufen, Skaliert auf {model_input_size[0]}x{model_input_size[1]}")
            return True
        except Exception as e:
            print(f"Fehler bei der Bildvorverarbeitung: {e}")
            return False
    
    def run_inference(self, simulated_score=None):
        """
        Simuliert die Inferenz des neuronalen Netzes
        
        Args:
            simulated_score (float, optional): Simulierte Detektionsbewertung
            
        Returns:
            bool: True, wenn die Inferenz erfolgreich war, sonst False
        """
        if self.processed_frame is None:
            print("FEHLER: Kein vorverarbeitetes Bild für Inferenz vorhanden.")
            return False
            
        if not self.firmware_loaded:
            print("FEHLER: Keine Firmware geladen.")
            return False
        
        # Simuliere die Inferenzzeit basierend auf Firmware-Eigenschaften
        self.inference_time_ms = self.calculate_inference_time()
        
        # Simuliere eine Detektionsbewertung
        if simulated_score is None:
            self.detection_score = np.random.random()  # Zufallswert zwischen 0 und 1
        else:
            self.detection_score = simulated_score
            
        # Simuliere die Ausführungszeit
        print(f"Führe Neural Network Inferenz aus...")
        time.sleep(self.inference_time_ms / 1000)
        
        # Aktualisiere den Batterieverbrauch
        self._update_battery_usage(self.inference_time_ms / 1000)
        
        print(f"Inferenz abgeschlossen in {self.inference_time_ms:.1f}ms")
        print(f"Detektionsbewertung: {self.detection_score:.4f} (Schwellwert: {self.detection_threshold:.2f})")
        
        if self.detection_score > self.detection_threshold:
            print("Objekt erkannt!")
            self.buzzer_double_beep()
            self.set_status_led(True)
        else:
            print("Kein Objekt erkannt.")
            self.set_status_led(False)
        
        return True
    
    def set_status_led(self, state):
        """
        Setzt den Status der Status-LED
        
        Args:
            state (bool): Neuer Status der LED
        """
        self.status_led = state
        print(f"Status-LED: {'AN' if state else 'AUS'}")
    
    def buzzer_double_beep(self):
        """
        Simuliert einen doppelten Piepton des Buzzers
        """
        self.buzzer_active = True
        print("PIEPTON! PIEPTON!")
        time.sleep(0.2)  # Simuliere Zeit für Pieptöne
        self.buzzer_active = False
    
    def enter_sleep_mode(self):
        """
        Versetzt das System in den Energiesparmodus
        
        Returns:
            bool: True, wenn der Modus erfolgreich gewechselt wurde
        """
        print("System wechselt in Deep-Sleep-Modus...")
        self.deep_sleep = True
        self.system_active = False
        # Schalte Kamera aus (simuliert)
        print("Kamera ausgeschaltet.")
        # Simuliere Stromverbrauchsänderung
        print(f"Stromverbrauch reduziert auf {self.standby_current_ma}mA")
        return True
    
    def wake_up(self):
        """
        Weckt das System aus dem Energiesparmodus auf
        
        Returns:
            bool: True, wenn das System erfolgreich aufgeweckt wurde
        """
        if not self.deep_sleep:
            print("System ist bereits aktiv.")
            return True
        
        print("System wird aufgeweckt...")
        self.deep_sleep = False
        self.system_active = True
        # Schalte Kamera ein (simuliert)
        print("Kamera eingeschaltet.")
        # Simuliere Stromverbrauchsänderung
        print(f"Stromverbrauch erhöht auf {self.active_current_ma}mA")
        return True
    
    def detect_object(self, input_image_path=None, simulated_score=None):
        """
        Führt den vollständigen Objekterkennungsprozess durch
        
        Args:
            input_image_path (str, optional): Pfad zu einem Testbild
            simulated_score (float, optional): Simulierte Detektionsbewertung
            
        Returns:
            bool: True, wenn die Objekterkennung erfolgreich war, sonst False
        """
        if not self.firmware_loaded:
            print("FEHLER: Keine Firmware geladen. Bitte zuerst Firmware laden.")
            return False
            
        if not self.system_active:
            print("System ist im Ruhemodus. Bitte erst aufwecken.")
            return False
        
        print("\n--- Starte Objekterkennung ---")
        result = self.capture_image(input_image_path)
        if not result:
            return False
            
        result = self.preprocess_image()
        if not result:
            return False
            
        result = self.run_inference(simulated_score)
        if not result:
            return False
            
        print("--- Objekterkennung abgeschlossen ---\n")
        return True
    
    def _update_battery_usage(self, duration_seconds):
        """
        Aktualisiert den Batteriestatus basierend auf der Betriebszeit
        
        Args:
            duration_seconds (float): Zeitdauer in Sekunden
        """
        if self.battery_capacity_mah <= 0:
            print("WARNUNG: Batterie hat ungültige Kapazität.")
            return
            
        if self.deep_sleep:
            current = self.standby_current_ma
        else:
            current = self.active_current_ma
            
        # Berechne den Batterieverbrauch in mAh
        consumed_mah = (current * duration_seconds) / 3600
        # Aktualisiere den verbleibenden Prozentsatz
        self.battery_remaining_percent -= (consumed_mah / self.battery_capacity_mah) * 100
        self.battery_remaining_percent = max(0, self.battery_remaining_percent)
        
        self.operation_time += duration_seconds
        
    def get_system_stats(self):
        """
        Gibt aktuelle Systemstatistiken zurück
        """
        if self.start_time is not None:
            self._update_battery_usage(time.time() - self.start_time)
            self.start_time = time.time()
        
        print("\n--- Systemstatus ---")
        print(f"Betriebszeit: {self.operation_time:.1f} Sekunden")
        print(f"Batteriestatus: {self.battery_remaining_percent:.1f}%")
        print(f"Flash verwendet: {self.used_flash/1024:.1f}KB von {self.flash_size_bytes/1024:.1f}KB")
        print(f"RAM verwendet: {self.used_ram/1024:.1f}KB von {self.ram_size_bytes/1024:.1f}KB")
        if self.firmware_loaded:
            print(f"Firmware-Größe: {self.firmware['total_size_bytes']/1024:.1f}KB")
            print(f"Modell-Größe: {self.firmware['model_size_bytes']/1024:.1f}KB")
            print(f"Modell-RAM-Bedarf: {self.firmware['ram_usage_bytes']/1024:.1f}KB")
            print(f"Modell quantisiert: {'Ja' if self.firmware['quantized'] else 'Nein'}")
            print(f"Modell-Eingabegröße: {self.firmware['model_input_size'][0]}x{self.firmware['model_input_size'][1]}")
        print(f"Status-LED: {'AN' if self.status_led else 'AUS'}")
        print(f"Power-LED: {'AN' if self.power_led else 'AUS'}")
        print(f"Systemmodus: {'Aktiv' if self.system_active else 'Deep-Sleep'}")
        print(f"Letzte Inferenzzeit: {self.inference_time_ms:.1f}ms")
        print(f"Letzte Detektionsbewertung: {self.detection_score:.4f}")
        print("-------------------\n")
        
        
class EmulationWorkflow:
    """
    Kompletter Workflow für die Konvertierung und Emulation
    """
    def __init__(self):
        self.converter = ModelConverter()
        self.builder = FirmwareBuilder()
        self.emulator = RP2040Emulator()
        self.model_info = None
        self.firmware = None
    
    def run_full_workflow(self, pth_path="model.pth", image_path=None, quantize=True):
        """
        Führt den gesamten Workflow aus: Konvertierung, Build, Emulation
        
        Args:
            pth_path (str): Pfad zur PyTorch-Modelldatei
            image_path (str, optional): Pfad zum Testbild
            quantize (bool): Ob das Modell quantisiert werden soll
            
        Returns:
            bool: True, wenn der Workflow erfolgreich war, sonst False
        """
        print("\n=== KOMPLETTER RP2040 EMULATIONS-WORKFLOW ===\n")
        
        try:
            # Schritt 1: Modellkonvertierung
            print("SCHRITT 1: MODELLKONVERTIERUNG")
            print("-------------------------------")
            self.model_info = self.converter.convert_pytorch_to_tflite(pth_path, quantize)
            self.converter.convert_tflite_to_c_array(self.model_info)
            
            # Schritt 2: Firmware-Aufbau
            print("\nSCHRITT 2: FIRMWARE-BUILD")
            print("------------------------")
            self.firmware = self.builder.build_firmware(self.model_info)
            
            # Schritt 3: Firmware in Emulator laden
            print("\nSCHRITT 3: FIRMWARE-EMULATION")
            print("----------------------------")
            if not self.emulator.load_firmware(self.firmware):
                print("FEHLER: Firmware konnte nicht geladen werden.")
                return False
            
            # Schritt 4: Objekterkennung testen
            print("\nSCHRITT 4: OBJEKTERKENNUNG TESTEN")
            print("--------------------------------")
            if not self.emulator.detect_object(image_path):
                print("FEHLER: Objekterkennung fehlgeschlagen.")
                return False
            
            # Schritt 5: Systemstatistiken
            print("\nSCHRITT 5: SYSTEMSTATISTIKEN")
            print("---------------------------")
            self.emulator.get_system_stats()
            
            # Schritt 6: Energieverbrauch testen
            print("\nSCHRITT 6: ENERGIEVERBRAUCH TESTEN")
            print("---------------------------------")
            print("Teste Deep-Sleep-Modus...")
            self.emulator.enter_sleep_mode()
            time.sleep(1)
            self.emulator.wake_up()
            
            # Weitere Tests
            self.emulator.detect_object(image_path, 0.8)
            self.emulator.get_system_stats()
            
            print("\n=== EMULATIONS-WORKFLOW ABGESCHLOSSEN ===\n")
            return True
            
        except Exception as e:
            print(f"\nFEHLER im Emulations-Workflow: {e}")
            return False
    
    def analyze_model_feasibility(self, pth_path="model.pth", quantize=True):
        """
        Analysiert, ob ein Modell auf dem RP2040 funktionieren würde
        
        Args:
            pth_path (str): Pfad zur PyTorch-Modelldatei
            quantize (bool): Ob das Modell quantisiert werden soll
            
        Returns:
            dict: Ergebnis der Machbarkeitsanalyse
        """
        print("\n=== MODELL-MACHBARKEITSANALYSE ===\n")
        
        try:
            # Konvertiere das Modell
            model_info = self.converter.convert_pytorch_to_tflite(pth_path, quantize)
            
            # Berechne den tatsächlichen RAM-Bedarf (Modell + System-Overhead)
            total_ram_needed = model_info['ram_usage_bytes'] + self.emulator.system_ram_overhead
            
            # Prüfe Speicherplatz
            flash_ratio = model_info['model_size_bytes'] / self.emulator.flash_size_bytes
            ram_ratio = total_ram_needed / self.emulator.ram_size_bytes
            
            # Erstelle eine temporäre Firmware für die Inferenzzeitberechnung
            temp_firmware = {
                'model_size_bytes': model_info['model_size_bytes'],
                'quantized': model_info['quantized'],
                'ram_usage_bytes': model_info['ram_usage_bytes'],
                'model_input_size': model_info['model_input_size']
            }
            
            # Lade die temporäre Firmware nur für die Berechnung, nicht für die tatsächliche Ausführung
            self.emulator.firmware = temp_firmware
            self.emulator.firmware_loaded = True
            
            # Berechne die Inferenzzeit
            inference_time = self.emulator.calculate_inference_time()
            
            # Zurücksetzen des Emulators
            self.emulator.firmware = None
            self.emulator.firmware_loaded = False
            
            print("\n--- Machbarkeitsanalyse ---")
            print(f"Modellgröße: {model_info['model_size_bytes']/1024:.1f}KB " + 
                  f"({flash_ratio*100:.1f}% des verfügbaren Flash)")
            
            print(f"Geschätzter RAM-Bedarf: {total_ram_needed/1024:.1f}KB " +
                  f"({ram_ratio*100:.1f}% des verfügbaren RAM)")
            print(f"  - Modell-RAM: {model_info['ram_usage_bytes']/1024:.1f}KB")
            print(f"  - System-Overhead: {self.emulator.system_ram_overhead/1024:.1f}KB")
            
            print(f"Geschätzte Inferenzzeit: {inference_time:.1f}ms")
            print(f"Modell-Eingabegröße: {model_info['model_input_size'][0]}x{model_info['model_input_size'][1]}")
            
            # Schlussfolgerung
            print("\nAnalyse-Ergebnis:")
            
            flash_status = "KRITISCH" if flash_ratio > 0.9 else "WARNUNG" if flash_ratio > 0.7 else "GUT"
            ram_status = "KRITISCH" if ram_ratio > 0.8 else "WARNUNG" if ram_ratio > 0.6 else "GUT"
            time_status = "WARNUNG" if inference_time > 300 else "GUT"
            
            if flash_ratio > 0.9:
                print("- KRITISCH: Das Modell ist zu groß für den Flash-Speicher des RP2040")
                print("  Empfehlung: Modell stärker quantisieren oder Modellgröße reduzieren")
            elif flash_ratio > 0.7:
                print("- WARNUNG: Das Modell nimmt einen großen Teil des Flash-Speichers ein")
                print("  Empfehlung: Modellgröße für mehr freien Speicher reduzieren")
            else:
                print("- GUT: Das Modell passt gut in den Flash-Speicher")
            
            if ram_ratio > 0.8:
                print("- KRITISCH: Der RAM-Bedarf ist zu hoch für den RP2040")
                print("  Empfehlung: Modell vereinfachen oder Aktivierungen reduzieren")
            elif ram_ratio > 0.6:
                print("- WARNUNG: Der RAM-Bedarf ist hoch und lässt wenig Platz für andere Funktionen")
                print("  Empfehlung: RAM-Optimierungen in Betracht ziehen")
            else:
                print("- GUT: Der RAM-Bedarf ist akzeptabel")
                
            if inference_time > 300:
                print("- WARNUNG: Die Inferenzzeit ist lang, was die Batterielebensdauer verkürzt")
                print("  Empfehlung: Modell vereinfachen oder aggressiver quantisieren")
            else:
                print("- GUT: Die geschätzte Inferenzzeit ist akzeptabel")
                
            # Gesamtbewertung
            feasible = flash_ratio < 0.9 and ram_ratio < 0.8 and inference_time < 500
            print(f"\nGesamtbewertung: {'MACHBAR' if feasible else 'NICHT MACHBAR'}")
            
            return {
                "model_info": model_info,
                "flash_usage_ratio": flash_ratio,
                "ram_usage_ratio": ram_ratio,
                "estimated_inference_time_ms": inference_time,
                "flash_status": flash_status,
                "ram_status": ram_status,
                "time_status": time_status,
                "feasible": feasible
            }
            
        except Exception as e:
            print(f"\nFEHLER bei der Machbarkeitsanalyse: {e}")
            return {"error": str(e), "feasible": False}


# Beispiel für die Verwendung
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RP2040 Neural Network Emulator')
    parser.add_argument('--model', type=str, default="model.pth", help='Pfad zur PyTorch-Modelldatei (.pth)')
    parser.add_argument('--image', type=str, help='Pfad zum Testbild (optional)')
    parser.add_argument('--analyze', action='store_true', help='Nur Machbarkeitsanalyse durchführen')
    parser.add_argument('--no-quantize', action='store_true', help='Modell nicht quantisieren (float32 statt int8)')
    args = parser.parse_args()
    
    workflow = EmulationWorkflow()
    
    if args.analyze:
        workflow.analyze_model_feasibility(args.model, not args.no_quantize)
    else:
        workflow.run_full_workflow(args.model, args.image, not args.no_quantize)