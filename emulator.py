import time
import numpy as np
from PIL import Image
import os
import sys

class RP2040Emulator:
    """
    Emulator für ein RP2040-basiertes Objekterkennungssystem
    """
    
    def __init__(self):
        # Hardware-Spezifikationen
        self.cpu_speed_mhz = 133
        self.cores = 2
        self.flash_size_bytes = 2 * 1024 * 1024  # 2MB Flash
        self.ram_size_bytes = 264 * 1024  # 264KB SRAM
        self.used_flash = 0
        self.used_ram = 0
        
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
        
    def load_model(self, model_size_bytes, ram_usage_bytes):
        """
        Lädt ein Modell und prüft, ob es in den Flash-Speicher passt
        """
        if model_size_bytes > self.flash_size_bytes:
            print(f"FEHLER: Modell ({model_size_bytes/1024:.1f}KB) passt nicht in Flash ({self.flash_size_bytes/1024:.1f}KB)")
            return False
        
        if ram_usage_bytes > self.ram_size_bytes:
            print(f"FEHLER: Modell RAM-Bedarf ({ram_usage_bytes/1024:.1f}KB) überschreitet verfügbaren RAM ({self.ram_size_bytes/1024:.1f}KB)")
            return False
        
        self.used_flash += model_size_bytes
        self.used_ram += ram_usage_bytes
        
        print(f"Modell geladen: {model_size_bytes/1024:.1f}KB Flash, {ram_usage_bytes/1024:.1f}KB RAM")
        print(f"Verbleibender Flash: {(self.flash_size_bytes - self.used_flash)/1024:.1f}KB")
        print(f"Verbleibender RAM: {(self.ram_size_bytes - self.used_ram)/1024:.1f}KB")
        return True
    
    def capture_image(self, input_image_path=None):
        """
        Simuliert die Bildaufnahme vom OV2640 Sensor
        """
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
            except Exception as e:
                print(f"Fehler beim Laden des Bildes: {e}")
                return False
        else:
            # Erzeuge ein Zufallsbild wenn kein Eingabebild vorhanden
            self.current_frame = np.random.randint(0, 255, (*self.camera_resolution, 3), dtype=np.uint8)
            print(f"Simuliertes Zufallsbild erzeugt: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
            return True
    
    def preprocess_image(self):
        """
        Simuliert die Bildvorverarbeitung
        """
        if self.current_frame is None:
            print("FEHLER: Kein Bild zum Verarbeiten vorhanden.")
            return False
        
        # Simulierte Vorverarbeitung: Graustufenkonvertierung, Skalierung
        if len(self.current_frame.shape) == 3:
            gray = np.mean(self.current_frame, axis=2).astype(np.uint8)
        else:
            gray = self.current_frame
            
        # Skaliere auf typische Netzwerkgröße
        model_input_size = (96, 96)  # Typische Größe für kleine Netze
        
        # Simuliere Skalierung (tatsächliche Implementierung würde PIL oder OpenCV nutzen)
        self.processed_frame = gray
        print(f"Bild vorverarbeitet: Graustufen, {model_input_size[0]}x{model_input_size[1]}")
        return True
    
    def run_inference(self, simulated_score=None, simulated_time_ms=None):
        """
        Simuliert die Inferenz des neuronalen Netzes
        """
        if self.processed_frame is None:
            print("FEHLER: Kein vorverarbeitetes Bild für Inferenz vorhanden.")
            return False
        
        # Simuliere die Inferenzzeit basierend auf Modellgröße und CPU-Geschwindigkeit
        if simulated_time_ms is None:
            # Typischer Bereich für kleine Modelle auf RP2040
            self.inference_time_ms = np.random.randint(100, 200)
        else:
            self.inference_time_ms = simulated_time_ms
        
        # Simuliere eine Detektionsbewertung
        if simulated_score is None:
            self.detection_score = np.random.random()  # Zufallswert zwischen 0 und 1
        else:
            self.detection_score = simulated_score
            
        # Simuliere die Ausführungszeit
        time.sleep(self.inference_time_ms / 1000)
        
        # Aktualisiere den Batterieverbrauch
        self._update_battery_usage(self.inference_time_ms / 1000)
        
        print(f"Inferenz abgeschlossen in {self.inference_time_ms}ms")
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
    
    def buzzer_alarm(self, duration_ms):
        """
        Simuliert einen Alarm-Ton für eine bestimmte Dauer
        """
        self.buzzer_active = True
        print(f"ALARM für {duration_ms}ms!")
        time.sleep(duration_ms / 1000)
        self.buzzer_active = False
    
    def enter_sleep_mode(self):
        """
        Versetzt das System in den Energiesparmodus
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
        """
        if not self.deep_sleep:
            return
        
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
        """
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
        """
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
        self._update_battery_usage(time.time() - self.start_time)
        self.start_time = time.time()
        
        print("\n--- Systemstatus ---")
        print(f"Betriebszeit: {self.operation_time:.1f} Sekunden")
        print(f"Batteriestatus: {self.battery_remaining_percent:.1f}%")
        print(f"Flash verwendet: {self.used_flash/1024:.1f}KB von {self.flash_size_bytes/1024:.1f}KB")
        print(f"RAM verwendet: {self.used_ram/1024:.1f}KB von {self.ram_size_bytes/1024:.1f}KB")
        print(f"Status-LED: {'AN' if self.status_led else 'AUS'}")
        print(f"Power-LED: {'AN' if self.power_led else 'AUS'}")
        print(f"Systemmodus: {'Aktiv' if self.system_active else 'Deep-Sleep'}")
        print(f"Letzte Inferenzzeit: {self.inference_time_ms}ms")
        print(f"Letzte Detektionsbewertung: {self.detection_score:.4f}")
        print("-------------------\n")


# Beispiel für die Verwendung des Emulators
if __name__ == "__main__":
    # Erstelle den Emulator
    emulator = RP2040Emulator()
    
    # Lade ein Beispiel-Modell (100KB Modellgröße, 50KB RAM-Nutzung)
    emulator.load_model(100 * 1024, 50 * 1024)
    
    # Führe Objekterkennung mit Testbild durch (Falls vorhanden, sonst Zufallsbild)
    test_image = "test_image.jpg" if len(sys.argv) > 1 else None
    emulator.detect_object(test_image)
    
    # Zeige Systemstatistiken an
    emulator.get_system_stats()
    
    # Energiesparmodus testen
    emulator.enter_sleep_mode()
    time.sleep(2)
    emulator.wake_up()
    
    # Nochmals Objekterkennung durchführen
    emulator.detect_object(test_image, 0.8)  # Simuliere hohen Erkennungswert
    
    # Abschließende Statistiken
    emulator.get_system_stats()