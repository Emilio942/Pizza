# Umfassende AI-zugängliche Dokumentation: Pizza-Erkennungssystem

## 1. Projektübersicht

Das Pizza-Erkennungssystem ist ein minimalistisches Bilderkennungssystem für den RP2040 Mikrocontroller, das zur Erkennung von Pizza-Zuständen entwickelt wurde. Es nutzt optimierte CNN-Modelle für ressourcenbeschränkte Hardware und implementiert verschiedene Techniken zur Effizienzsteigerung.

### Hauptfunktionen:
- Erkennung verschiedener Pizza-Zustände (roh, gebacken, verbrannt, etc.)
- Optimiert für ressourcenbeschränkte RP2040-Mikrocontroller
- Implementiert Energiesparmaßnahmen für Batteriebetrieb
- Umfassende Datenaugmentierung und Modelloptimierung

## 2. Projektstruktur und Organisation

```
/pizza/
├── config/           # Konfigurationsdateien
├── data/             # Datensätze (raw, processed, augmented, etc.)
├── docs/             # Dokumentation
├── hardware/         # Hardware-Dateien und Fertigungsunterlagen
├── models/           # Trainierte Modelle und Checkpoints
├── output/           # Ausgabeverzeichnisse für Logs und temporäre Dateien
├── scripts/          # Hilfsskripte für verschiedene Aufgaben
├── src/              # Quellcode
│   ├── augmentation/ # Code für Datenaugmentierung
│   ├── chatlist_ki/  # KI-Chat-Schnittstelle
│   ├── emulation/    # RP2040-Emulation
│   ├── integration/  # Integration mit anderen Systemen
│   └── utils/        # Hilfsfunktionen
└── tests/            # Testdateien
```

## 3. Systemarchitektur

Das System besteht aus mehreren Kernkomponenten:

1. **Datenvorverarbeitung und Augmentierung**: Erzeugt einen robusten Datensatz durch verschiedene Bildtransformationen.
2. **CNN-Modellarchitektur**: Speziell für Mikrocontroller optimierte Netzwerkarchitektur (MicroPizzaNet).
3. **Training und Quantisierung**: Trainingsroutinen mit Quantisierungsunterstützung für Int8-Inferenz.
4. **Emulation**: Simuliert die Ausführung auf RP2040-Hardware ohne physisches Gerät.
5. **Hardware-Abstraktion**: Verwaltet Kamera, Energiemanagement und Statusanzeige.
6. **CI/CD-Pipeline**: Automatisiert Training, Tests und Deployment.

## 4. Datenmodelle und Schemas

### 4.1 Klassifikationsmodell
Das Projekt verwendet ein CNN-Modell mit diesen spezifischen Charakteristiken:
- **Eingabegröße**: 48x48 Pixel RGB-Bilder
- **Ausgabe**: Klassenwahrscheinlichkeiten für verschiedene Pizza-Zustände
- **Klassen**: basic, burnt, combined, mixed, progression, segment

### 4.2 Konfigurationsparameter
Wichtige Systemparameter sind in der RP2040Config-Klasse definiert:
- RP2040_FLASH_SIZE_KB = 2048 (2MB Flash)
- RP2040_RAM_SIZE_KB = 264 (264KB RAM)
- RP2040_CLOCK_SPEED_MHZ = 133
- IMG_SIZE = 48
- QUANTIZATION_BITS = 8

## 5. API-Dokumentation

### 5.1 Kernmodule

#### pizza_detector.py
Hauptmodul zum Training und Exportieren von Modellen für die Pizza-Zustandserkennung.

```python
def train(data_dir="data/augmented", epochs=50, batch_size=16, learning_rate=0.002):
    """Trainiert das CNN-Modell mit den angegebenen Parametern."""
    
def export(model_path="models/pizza_model.pth", quantize=True, target="rp2040"):
    """Exportiert das trainierte Modell für RP2040-Deployment."""
```

#### emulator.py
Emuliert die Ausführung von CNN-Modellen auf RP2040-Hardware.

```python
class RP2040Emulator:
    def load_model(self, model_path):
        """Lädt ein quantisiertes Modell in den Emulator."""
        
    def run_inference(self, image):
        """Führt Inferenz auf einem Bild mit dem geladenen Modell durch."""
```

#### power_manager.py
Verwaltet Stromverbrauch und Batterielebensdauer.

```python
class PowerManager:
    def set_mode(self, mode):
        """Setzt den Energiemodus (ACTIVE, SLEEP, STANDBY)."""
        
    def estimate_battery_life(self):
        """Schätzt die verbleibende Batterielebensdauer basierend auf Nutzungsmuster."""
```

## 6. Konfiguration und Installationsanleitung

### 6.1 Voraussetzungen
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.5+
- Weitere Abhängigkeiten in requirements.txt

### 6.2 Installation

```bash
# Python-Umgebung einrichten
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 6.3 Hardware-Anforderungen
- RP2040 Mikrocontroller (z.B. Raspberry Pi Pico)
- OV2640 Kamerasensor
- CR123A Batterie mit LDO-Regler

## 7. Workflow und Prozesse

### 7.1 Datenerstellung und Augmentierung
1. Sammeln von Pizza-Bildern in verschiedenen Zuständen
2. Datenaugmentierung zur Erzeugung von Trainingsbeispielen
3. Bilder in Klassen einteilen und Datensatz aufbereiten

```bash
python src/augmentation/enhanced_pizza_augmentation.py --input-dir data/raw --output-dir data/augmented
```

### 7.2 Modelltraining
1. Datensatz in Trainings- und Validierungssplit aufteilen
2. Modell mit optimierten Hyperparametern trainieren
3. Quantisierungsbewusstes Training für Int8-Modelle

```bash
python src/pizza_detector.py train
```

### 7.3 Modell-Deployment
1. Modell für RP2040 exportieren
2. C-Code für Inferenz generieren
3. Firmware bauen und auf RP2040 flashen

```bash
python src/pizza_detector.py export
```

## 8. Fehlerbehandlung und Troubleshooting

### 8.1 Bekannte Probleme und Lösungen

| Problem | Ursache | Lösung |
|---------|---------|--------|
| Importfehler | Falsche Import-Pfade in Scripts | Import-Pfade korrigieren |
| Syntax-Fehler | Doppelte Schlüsselwortargumente | Redundanz entfernen |
| Fehlende Abhängigkeiten | Module nicht an erwarteter Stelle | Module anlegen oder Imports anpassen |

### 8.2 Ressourcenlimits
- Wenn das Modell zu groß für den RP2040-Flash ist, kann Pruning angewendet werden
- Bei RAM-Überschreitung können Aktivierungen neu berechnet statt gespeichert werden
- Batterielebensdauer kann durch Anpassung des Duty Cycles verbessert werden

## 9. Beispielcode und Nutzungsszenarien

### 9.1 Beispiel: Echtzeit-Inferenz

```python
from src.pizza_detector import load_model, preprocess_image, get_prediction
from src.utils.constants import CLASS_NAMES

# Modell laden
model = load_model("models/pizza_model_int8.pth")

# Bild vorverarbeiten
img = preprocess_image("data/test/pizza_sample.jpg")

# Inferenz durchführen
prediction, confidence = get_prediction(model, img)
print(f"Erkannter Pizza-Zustand: {CLASS_NAMES[prediction]} mit {confidence:.2f}% Sicherheit")
```

### 9.2 Beispiel: Batterieoptimierter Betrieb

```python
from src.utils.power_manager import PowerManager, AdaptiveMode

# Power Manager initialisieren
power_mgr = PowerManager()

# Adaptiven Modus setzen
power_mgr.set_adaptive_mode(AdaptiveMode.BATTERY_OPTIMIZED)

# Duty Cycle anpassen
power_mgr.set_duty_cycle(0.1)  # 10% aktive Zeit, 90% Schlafmodus

# Batteriestatus überprüfen
status = power_mgr.get_battery_status()
print(f"Verbleibende Batterielaufzeit: {status.remaining_hours:.1f} Stunden")
```

## 10. Validierungs- und Testergebnisse

Bei der umfassenden Validierung aller Skripte im Projekt wurden verschiedene Probleme identifiziert:

### 10.1 Syntaxfehler
- **scripts/generate_pizza_dataset.py**: Wiederholte Schlüsselwortargumente `image_size`
- **scripts/hyperparameter_search.py**: Ungültige Syntax
- **config/config.py**: Ungültige Syntax oder fehlendes Komma
- **src/augmentation/augmentation.py**: Parameter ohne Standardwert folgt Parameter mit Standardwert
- **src/augmentation/optimized-pizza-augmentation.py**: Parameter ohne Standardwert folgt Parameter mit Standardwert
- **hardware/manufacturing/pcb_export.py**: Ungültige Syntax

### 10.2 Import-/Abhängigkeitsprobleme
- **src/emulation/emulator.py**: Importiert fehlende Constants-Datei
- **scripts/demo_status_display.py**: Falscher Import-Pfad für status_display
- **scripts/demo_power_management.py**: Falscher Import-Pfad für RP2040Emulator
- **tests/conftest.py**: Falscher Import-Pfad für ModelConfig

### 10.3 Empfehlungen
1. Import-Pfade korrigieren
2. Fehlende Abhängigkeiten erstellen
3. Syntaxfehler beheben
4. Projekt als Paket installierbar machen

## 11. Internationale Standards und Best Practices

Das Pizza-Erkennungssystem folgt mehreren Best Practices für eingebettete ML-Systeme:

- **Quantisierung**: Standardtechnik zur Modellkomprimierung für ressourcenbeschränkte Geräte
- **Duty-Cycle-Management**: Energiesparstrategien für batteriebetriebene Geräte
- **CI/CD-Pipeline**: Automatisierte Tests und Deployment-Prozesse
- **Resource Budgeting**: Strikte Überwachung von Speicher- und Rechenressourcen

## 12. Metadaten und Versionierung

- **Projektname**: RP2040 Pizza-Erkennungssystem
- **Version**: 1.0.0
- **Zuletzt aktualisiert**: 2025-05-13
- **Erstellt von**: Pizza-Erkennungsteam
- **Lizenz**: Open Source (siehe LICENSE-Datei)
- **Sprache**: Deutsch
- **Technologien**: Python, PyTorch, RP2040, Computer Vision

## 13. Datenverarbeitungspipeline

### 13.1 Datenerfassung
- Kameradaten vom OV2640-Sensor (320x240 RGB)
- Bildvorverarbeitung (Skalierung auf 48x48, Normalisierung)

### 13.2 Inferenz-Pipeline
1. Bildaufnahme
2. Vorverarbeitung und Normalisierung
3. Modellvorhersage (quantisiertes Int8-Modell)
4. Temporal Smoothing für robuste Ergebnisse
5. Statusanzeige und Aktionen basierend auf Erkennung

### 13.3 Hardwarespezifische Optimierungen
- In-place Operationen zur RAM-Minimierung
- Wiederverwendung von Aktivierungstensoren
- Präzise Planung des Speicherlayouts für Cacheoptimierung

## 14. Validierung und Benchmarks

| Aspekt | Ergebnis | Einheit |
|--------|---------|---------|
| Modellgröße | 9.24 | KB |
| RAM-Verbrauch | 97 | KB |
| Inferenzzeit | 74 | ms |
| Klassifikationsgenauigkeit | 87.5 | % |
| Batterielebensdauer | 76 | Stunden |

## 15. Skript-Inventar

### 15.1 Core Scripts
- `src/pizza_detector.py`: Hauptmodul für Training und Export
- `src/pizza-baking-detection-final.py`: Finales Detektionsmodul
- `src/emulation/emulator.py`: Hardware-Emulator für RP2040

### 15.2 Utilities
- `src/utils/power_manager.py`: Energiemanagement
- `src/utils/memory.py`: Speicherverwaltung und -analyse
- `src/utils/constants.py`: Systemweite Konstanten
- `src/utils/types.py`: Typdefinitionen

### 15.3 Hilfsskripte
- `scripts/cleanup.py`: Projekt aufräumen
- `scripts/automated_test_suite.py`: Automatisierte Tests
- `scripts/knowledge_distillation.py`: Wissenstransfer für Modellkomprimierung
- `scripts/visualize_gradcam.py`: Visualisierung von Modellaktivierungen

## 16. Model-Architektur-Details

### 16.1 MicroPizzaNet (Original)
- 3x3 Konvolutionen mit reduzierten Filteranzahlen
- Depthwise Separable Convolutions für Effizienz
- Global Average Pooling statt Fully Connected Layers

### 16.2 MicroPizzaNetV2 (Inverted Residual)
- Inverted Residual Blocks für bessere Gradientenflüsse
- Skip-Connections für verbesserte Konvergenz
- Optimiert für Int8-Quantisierung

### 16.3 MicroPizzaNetWithSE (Squeeze-and-Excitation)
- Channel Attention Mechanism für verbesserte Genauigkeit
- Effizientes Parameter-Sharing
- Erhöhte Robustheit gegen Belichtungsvariationen

## 17. Knowledge Graph

```
Pizza-Erkennungssystem
├── Hardwarekomponenten
│   ├── RP2040
│   ├── OV2640
│   └── CR123A
├── Softwarekomponenten
│   ├── CNN-Modelle
│   │   ├── MicroPizzaNet
│   │   ├── MicroPizzaNetV2
│   │   └── MicroPizzaNetWithSE
│   ├── Datenverarbeitung
│   │   ├── Augmentierung
│   │   ├── Temporal Smoothing
│   │   └── CLAHE
│   └── System-Management
│       ├── Power Manager
│       ├── Memory Estimator
│       └── Status Display
└── Pizza-Zustände
    ├── Basic
    ├── Burnt
    ├── Combined
    ├── Mixed
    ├── Progression
    └── Segment
```

## 18. Fehlercodes und Diagnoseinformationen

| Fehlercode | Beschreibung | Empfohlene Maßnahme |
|------------|--------------|---------------------|
| E001 | Kamera nicht initialisiert | Hardware-Verbindung prüfen |
| E002 | Modell zu groß für Flash | Pruning anwenden |
| E003 | RAM-Überlauf | Inferenz-Batch-Größe reduzieren |
| E004 | Batterie schwach | Wechseln oder aufladen |
| E005 | Inferenzzeit überschritten | Modell weiter optimieren |

---

Dieses Dokument wurde erstellt, um eine umfassende, global zugängliche Dokumentation des Pizza-Erkennungssystems bereitzustellen. Es ist strukturiert, um von anderen KI-Systemen leicht verarbeitet zu werden, und folgt internationalen Standards für technische Dokumentation.
