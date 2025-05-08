# Python Code Dokumentation für das Pizza-Erkennungsprojekt

## 1. emulator-test.py

Dieses Skript simuliert die gesamte Umgebung eines RP2040-Mikrocontrollers und emuliert verschiedene Aspekte des Pizza-Erkennungssystems.

### Klassen und Funktionen:

#### 1.1 ModelConverter
```python
class ModelConverter:
    """Konvertiert ein PyTorch-Modell in ein Format für RP2040"""
```
- **Funktionalität**: Konvertiert PyTorch-Modelle in ein TFLite-Format und anschließend in ein C-Array für die Einbettung in Mikrocontroller-Firmware.
- **Wichtige Methoden**:
  - `estimate_model_size(original_size_mb, quantized=True)`: Schätzt die Größe des konvertierten Modells basierend auf der ursprünglichen Größe und Quantisierungsstatus.
  - `convert_pytorch_to_tflite(pth_path, quantize=True)`: Simuliert den mehrstufigen Konvertierungsprozess.
  - `convert_tflite_to_c_array(model_info)`: Generiert ein simuliertes C-Array für das konvertierte Modell.

#### 1.2 FirmwareBuilder
```python
class FirmwareBuilder:
    """Simuliert den Build-Prozess der Firmware für RP2040"""
```
- **Funktionalität**: Simuliert den Prozess der Firmware-Erstellung, die das Modell und den notwendigen Code für die Ausführung enthält.
- **Methoden**:
  - `build_firmware(model_info, extra_code_kb=50)`: Berechnet die Gesamtgröße der Firmware basierend auf Modellgröße und zusätzlichem Code.

#### 1.3 RP2040Emulator
```python
class RP2040Emulator:
    """Erweiterter Emulator für ein RP2040-basiertes Objekterkennungssystem mit Unterstützung für Firmware-Simulation"""
```
- **Funktionalität**: Emuliert die Hardware und den Betrieb eines RP2040-basierten Systems, inklusive Speicherverwaltung, Bildverarbeitung und Inferenz.
- **Wichtige Attribute**:
  - Hardware-Spezifikationen (CPU, Flash, RAM)
  - Kameraeinstellungen (Auflösung, FPS)
  - Stromverbrauchssimulation
  - GPIO-Zustände und Status-Indikatoren
- **Kernmethoden**:
  - `load_firmware(firmware)`: Lädt die simulierte Firmware und überprüft Ressourcenbeschränkungen.
  - `capture_image(input_image_path=None)`: Simuliert die Bildaufnahme von der OV2640 Kamera.
  - `preprocess_image()`: Simuliert die Bildvorverarbeitung (Graustufenkonvertierung, Skalierung).
  - `run_inference(simulated_score=None)`: Simuliert die Modellausführung und berechnet die Inferenzzeit.
  - `detect_object(input_image_path=None, simulated_score=None)`: Führt den vollständigen Objekterkennungsprozess durch.

#### 1.4 EmulationWorkflow
```python
class EmulationWorkflow:
    """Orchestriert den gesamten Simulationsprozess von der Modellkonvertierung bis zur Emulation"""
```
- **Funktionalität**: Verbindet die einzelnen Simulations-Komponenten zu einem vollständigen Workflow.
- **Methoden**:
  - `run_full_workflow(pth_path="model.pth", image_path=None, quantize=True)`: Führt alle Schritte der Simulation aus.

### Kritische Punkte:

- **EMU-01 (Kritisch)**: Fehlende Simulation des Kamera-Framebuffers im RAM. Dies führt zu einer deutlichen Unterschätzung des tatsächlichen RAM-Bedarfs.
- **EMU-02 (Hoch)**: Ungenaue Schätzung des Modell-RAM-Bedarfs (Tensor Arena). Die Berechnung basiert auf einem festen Prozentsatz der Modellgröße, während der tatsächliche Bedarf stark von der Modellarchitektur abhängt.

## 2. augmentation.py (enhanced_pizza_augmentation.py)

Dieses Skript implementiert eine fortschrittliche Augmentierungspipeline für Pizza-Bilder, speziell optimiert für die Erkennung verschiedener Verbrennungsgrade.

### Hauptkomponenten:

#### 2.1 Setup und Konfiguration
- `parse_arguments()`: Verarbeitet Kommandozeilenargumente für Eingabe-/Ausgabeverzeichnisse, Seed, Batch-Größe, etc.
- `setup_environment()`: Konfiguriert die Ausführungsumgebung (CPU/GPU), setzt Random Seeds und prüft verfügbare Bibliotheken.
- `validate_and_prepare_paths()`: Validiert Pfade und erstellt notwendige Verzeichnisse.
- `get_image_files()`: Lädt und validiert Bilddateien aus dem Eingabeverzeichnis.
- `get_optimal_batch_size()`: Ermittelt eine optimale Batch-Größe basierend auf verfügbarem GPU-Speicher.
- `AugmentationStats`: Eine Klasse zur Verfolgung und Speicherung von Augmentierungsstatistiken.

#### 2.2 Dataset und Utilities
- `PizzaAugmentationDataset`: Eine PyTorch Dataset-Klasse für effizientes Laden und Caching von Bildern.
- `open_image()`: Kontextmanager für sicheres Öffnen und Schließen von Bildern.
- `show_images()`: Hilfsfunktion zur Visualisierung von Bildern in einem Grid.
- `save_augmented_images()`: Speichert augmentierte Bilder batchweise mit einheitlicher Größe.

#### 2.3 Augmentierungsmodule
- `EnhancedPizzaBurningEffect`: Simuliert realistische Verbrennungseffekte mit verschiedenen Mustern (Rand, Flecken, Streifen).
- `EnhancedOvenEffect`: Generiert Ofeneffekte wie Dampf, Wärme, Schatten und ungleichmäßige Beleuchtung.
- `PizzaSegmentEffect`: Wendet Augmentierungen auf einzelne Pizza-Segmente an.

#### 2.4 Augmentierungspipeline
- `apply_basic_augmentation()`: Grundlegende Transformationen (Rotation, Zuschnitt, Spiegelung, Farbe).
- `apply_burning_augmentation()`: Anwendung von Verbrennungseffekten.
- `apply_mixed_augmentation()`: Kombiniert mehrere Bilder mit Techniken wie MixUp, CutMix und CopyPaste.
- `apply_progression_augmentation()`: Erzeugt Sequenzen mit zunehmenden Verbrennungsgraden.
- `apply_segment_augmentation()`: Erzeugt Variationen auf Pizza-Segment-Ebene.
- `apply_combination_augmentation()`: Kombiniert mehrere Augmentierungstechniken.

#### 2.5 Hauptprogramm
- `main()`: Orchestriert den gesamten Augmentierungsprozess mit einer Verteilungsstrategie für verschiedene Augmentierungstypen.

## 3. augmentation_optimized.py

Dieses Skript ist eine optimierte Variante der Augmentierungspipeline mit Schwerpunkt auf Speichereffizienz durch generatorbasierte Verarbeitung.

### Hauptunterschied zu augmentation.py:

#### 3.1 Einfachere Effektmodule
- `PizzaBurningEffect`: Vereinfachte Version des Verbrennungseffekts.
- `SimpleOvenEffect`: Vereinfachte Simulation von Ofeneffekten.

#### 3.2 Generator-basierte Funktionen
- `pizza_basic_augmentation()`: Generator für grundlegende Augmentierungen.
- `pizza_burning_augmentation()`: Generator für Verbrennungseffekte.
- `pizza_mixup()`, `pizza_cutmix()`: Implementations für die Bildkombination.
- `pizza_burning_progression()`: Erzeugt eine Sequenz mit zunehmendem Verbrennungsgrad.

#### 3.3 Optimierte Hauptpipeline
- `augment_pizza_dataset()`: Hauptfunktion, die den Augmentierungsprozess mit spezifischen Verteilungen koordiniert.

## 4. pizza_detector.py

Dieses Skript ist das Herzstück des Projekts und implementiert das Training, die Optimierung und den Export des Modells für die RP2040-Plattform.

### Hauptkomponenten:

#### 4.1 RP2040Config
```python
class RP2040Config:
    """Ausführliche Konfiguration für RP2040-basierte Bildklassifikation mit Speicher- und Leistungsanalyse"""
```
- **Funktionalität**: Enthält alle Konfigurationsparameter für das RP2040-System, das Training und die Deployment-Parameter.
- **Wichtige Attribute**:
  - Hardware-Spezifikationen (Flash, RAM, CPU-Geschwindigkeit)
  - Kamera-Parameter (Auflösung, FPS)
  - Batterieparameter
  - Modellparameter (Bildgröße, Batch-Größe, Lernrate)
  - Speicherbeschränkungen (Maximale Modellgröße, maximaler RAM-Verbrauch)

#### 4.2 PizzaDatasetAnalysis
```python
class PizzaDatasetAnalysis:
    """Analysiert den Datensatz für optimale Vorverarbeitung und Klassenbalancierung"""
```
- **Funktionalität**: Analysiert den Bilddatensatz, um optimale Parameter für die Vorverarbeitung zu bestimmen.
- **Methoden**:
  - `analyze(sample_size=None)`: Führt die Analyse durch und sammelt Statistiken über Bildgrößen, RGB-Verteilungen und Klassenverteilungen.
  - `get_preprocessing_parameters()`: Liefert die ermittelten Parameter für die Vorverarbeitung.

#### 4.3 MemoryEstimator
```python
class MemoryEstimator:
    """Schätzt Speicherverbrauch von Modellen und Operationen für RP2040"""
```
- **Funktionalität**: Bewertet den Speicherverbrauch von Modellen in Bezug auf die RP2040-Beschränkungen.
- **Methoden**:
  - `estimate_model_size(model, bits=32)`: Schätzt die Modellgröße in KB.
  - `estimate_activation_memory(model, input_size)`: Schätzt den Speicherverbrauch durch Aktivierungen während der Inferenz.
  - `check_memory_requirements(model, input_size, config)`: Überprüft, ob das Modell in die Speicherbeschränkungen passt.

#### 4.4 BalancedPizzaDataset
```python
class BalancedPizzaDataset(Dataset):
    """Erweiterter Dataset mit Augmentierung und Klassenbalancierung für Pizza-Erkennung"""
```
- **Funktionalität**: Implementiert einen balancierten Datensatz für das Training, der die Klassenverteilung berücksichtigt.
- **Methoden**:
  - `_collect_samples()`: Sammelt alle Bilder und ihre Klassen.
  - `_compute_class_weights()`: Berechnet Gewichte für die Klassenbalancierung.

#### 4.5 Modellarchitekturen
- `MicroPizzaNet`: Ein ultrakompaktes CNN für die Pizza-Erkennung auf RP2040.
- `MicroPizzaNetV2`: Verbesserte Version mit Inverted Residual Blocks.
- `InvertedResidualBlock`: MobileNetV2-Style Block für effiziente Faltungsoperationen.
- `SqueezeExcitationModule`: Implementiert den Squeeze-and-Excitation-Mechanismus für verbesserte Modellleistung.
- `MicroPizzaNetWithSE`: Integration von Squeeze-and-Excitation in das Basismodell.

#### 4.6 Training und Optimierung
- `create_optimized_dataloaders()`: Erstellt DataLoader mit Klassenbalancierung und spezifischer Vorverarbeitung.
- `EarlyStopping`: Implementiert Early Stopping mit Plateau-Erkennung.
- `train_microcontroller_model()`: Trainingsroutine mit LR-Scheduling und Gewichtung.
- `calibrate_and_quantize()`: Kalibriert und quantisiert das Modell für optimale Int8-Konvertierung.
- `export_to_microcontroller()`: Exportiert das Modell für den RP2040-Mikrocontroller.

#### 4.7 Evaluation und Visualisierung
- `detailed_evaluation()`: Führt eine detaillierte Evaluierung des Modells durch.
- `visualize_results()`: Erstellt Visualisierungen der Trainingsergebnisse und Modellleistung.

#### 4.8 Hauptprogramm
- `main()`: Orchestriert den gesamten Prozess von der Datenanalyse bis zum Export.

## 5. temporal_smoother.py

Dieses Modul implementiert verschiedene Strategien für Temporal Smoothing, um die Stabilität und Zuverlässigkeit von Bilderkennungen über mehrere Frames hinweg zu verbessern.

### Hauptkomponenten:

#### 5.1 SmoothingStrategy
```python
class SmoothingStrategy(Enum):
    """Verfügbare Strategien für Temporal Smoothing."""
```
- **Funktionalität**: Definiert verschiedene Strategien zur zeitlichen Glättung von Erkennungsergebnissen.
- **Verfügbare Strategien**:
  - `MAJORITY_VOTE`: Mehrheitsentscheidung über die letzten N Frames.
  - `MOVING_AVERAGE`: Gleitender Mittelwert über die Klassenwahrscheinlichkeiten.
  - `EXPONENTIAL_MOVING_AVERAGE`: Exponentiell gewichteter gleitender Mittelwert (neuere Frames haben höheres Gewicht).
  - `CONFIDENCE_WEIGHTED`: Gewichtung basierend auf den Konfidenzwerten der Vorhersagen.

#### 5.2 TemporalSmoother
```python
class TemporalSmoother:
    """Implementiert verschiedene Methoden für Temporal Smoothing."""
```
- **Funktionalität**: Hauptklasse, die mehrere aufeinanderfolgende Inferenz-Ergebnisse kombiniert, um stabilere Vorhersagen zu treffen.
- **Attribute**:
  - `window_size`: Anzahl der zu berücksichtigenden Frames (default: 5)
  - `strategy`: Zu verwendende Glättungsstrategie
  - `decay_factor`: Abklingfaktor für Exponential Moving Average (0-1)
  - `predictions`: Ringpuffer der letzten Klassenvorhersagen
  - `confidences`: Ringpuffer der letzten Konfidenzwerte
  - `probabilities`: Ringpuffer der letzten Klassenwahrscheinlichkeiten
- **Kernmethoden**:
  - `add_result(class_index, confidence, class_probs=None)`: Fügt ein neues Inferenz-Ergebnis zur Historie hinzu.
  - `get_smoothed_prediction()`: Berechnet die geglättete Vorhersage basierend auf der gewählten Strategie.
  - `_apply_majority_vote()`: Implementiert die Mehrheitsentscheidung über die letzten N Frames.
  - `_apply_moving_average()`: Wendet einen gleitenden Mittelwert auf die Klassenwahrscheinlichkeiten an.
  - `_apply_exponential_moving_average()`: Implementiert einen exponentiell gewichteten gleitenden Mittelwert.
  - `_apply_confidence_weighted()`: Implementiert eine konfidenzbasierte Gewichtungsstrategie.
  - `reset()`: Setzt alle internen Zustände zurück.

#### 5.3 PizzaTemporalPredictor
```python
class PizzaTemporalPredictor:
    """Konsolidierende Klasse für die zeitliche Glättung von Pizza-Erkennungen."""
```
- **Funktionalität**: High-Level-API für die kontinuierliche Inferenz mit integrierter zeitlicher Glättung.
- **Attribute**:
  - `model`: Das zu verwendende PyTorch- oder TFLite-Modell
  - `class_names`: Liste der Klassennamen
  - `smoother`: Instanz von TemporalSmoother
- **Methoden**:
  - `predict(input_tensor, device=None)`: Führt eine Vorhersage durch und wendet zeitliche Glättung an.
  
#### 5.4 C-Code-Generator für Mikrocontroller
```python
def generate_c_implementation(window_size=5, strategy="majority_vote"):
    """Generiert C/C++-Code für die Temporal-Smoothing-Implementierung."""
```
- **Funktionalität**: Erzeugt C-Code für die Implementierung auf ressourcenbeschränkten Mikrocontrollern (z.B. RP2040).
- **Parameter**:
  - `window_size`: Größe des Glättungsfensters
  - `strategy`: Zu verwendende Glättungsstrategie
- **Implementierte C-Funktionen**:
  - `ts_init()`: Initialisiert den Temporal-Smoothing-Puffer
  - `ts_add_prediction()`: Fügt eine neue Vorhersage zum Puffer hinzu
  - `ts_get_smoothed_prediction()`: Berechnet die geglättete Vorhersage
  - `ts_get_smoothed_confidence()`: Gibt die Konfidenz der geglätteten Vorhersage zurück
  - `ts_reset()`: Setzt den Puffer zurück

### Anwendungsfälle:

1. **Stabilisierung flackernder Erkennungen**:
   - Situationen mit wechselnden Lichtverhältnissen, Bewegung oder Bildstörungen können zu instabilen Frame-zu-Frame-Erkennungen führen.
   - Das Temporal Smoothing reduziert dieses "Flackern" und erzeugt eine stabilere Erkennung.

2. **Überbrückung kurzer Unterbrechungen**:
   - Falls ein einzelner Frame eine falsche Erkennung liefert (z.B. durch kurze Verdeckung oder Störung), kann das Temporal Smoothing die Erkennungsleistung aufrechterhalten.

3. **Erhöhte Konfidenz**:
   - Durch die Kombination mehrerer Frames kann die Gesamtkonfidenz der Erkennungen gesteigert werden.

4. **Reduzierung der Rechenleistung**:
   - In einigen Konfigurationen kann die Inferenz weniger häufig durchgeführt werden, während die Erkennungsqualität durch Temporal Smoothing aufrechterhalten wird.

### Integrierung mit dem Hauptsystem:

Der `PizzaTemporalPredictor` dient als Schnittstelle zwischen dem Inferenzmodell und dem Temporal Smoothing. Er nimmt einen Bildframe entgegen, führt die Modellvorhersage durch und wendet die zeitliche Glättung an. Das Ergebnis enthält sowohl die geglättete als auch die Original-Vorhersage, sodass der Anwender bei Bedarf beide verwenden kann.

Die C-Implementierung ermöglicht die Verwendung der Temporal-Smoothing-Algorithmen direkt auf dem RP2040-Mikrocontroller mit minimalen Ressourcenanforderungen.

## 6. test_temporal_smoothing.py

Dieses Skript dient zum Testen und Visualisieren der verschiedenen Temporal-Smoothing-Strategien unter verschiedenen simulierten Bedingungen.

### Hauptkomponenten:

#### 6.1 Simulationsfunktionen
- `generate_noisy_predictions()`: Generiert simulierte verrauschte Vorhersagen mit einstellbarem Rauschgrad.
- `test_smoothing_strategies()`: Testet verschiedene Glättungsstrategien mit den simulierten Daten.
- `plot_results()`: Erstellt visuelle Darstellungen der Ergebnisse verschiedener Strategien.
- `simulate_scenario()`: Simuliert verschiedene reale Szenarien (z.B. flackernde Erkennung, kurze Unterbrechungen).

#### 6.2 Simulierte Szenarien
- **Flackernde Erkennung**: Simuliert schnelle Wechsel zwischen Klassen durch Rauschen.
- **Kurze Unterbrechung**: Testet, wie gut Smoothing-Strategien mit kurzen Unterbrechungen umgehen können.
- **Mehrere Klassen**: Simuliert Übergänge zwischen mehr als zwei Klassen.
- **Niedrige Konfidenz**: Testet das Verhalten bei Vorhersagen mit niedriger Konfidenz.
- **Standard**: Ein einfacher Klassenwechsel mit moderatem Rauschen.

#### 6.3 Ausgabedaten
- Genauigkeitsstatistiken für jede Strategie.
- Visualisierung der Vorhersagen über die Zeit.
- Vergleich der Leistung verschiedener Strategien.
- Empfehlungen für die beste Strategie basierend auf den simulierten Daten.

### Anwendung:

Das Skript kann genutzt werden, um die optimale Smoothing-Strategie für spezifische Anwendungsfälle zu bestimmen, indem verschiedene Szenarien simuliert und die Ergebnisse analysiert werden. Dies hilft bei der Auswahl der besten Konfiguration für das reale Pizza-Erkennungssystem auf dem RP2040.

## Erweiterte Projektarchitektur mit Temporal Smoothing

Mit der Integration des Temporal-Smoothing-Moduls erweitert sich die Architektur des Pizza-Erkennungssystems:

1. **Bilderfassung**: Die Kamera (OV2640) nimmt kontinuierlich Bilder auf.
2. **Vorverarbeitung**: Jedes Bild wird vorverarbeitet (Größenanpassung, Normalisierung).
3. **Inferenz**: Das optimierte Modell (MicroPizzaNet) führt die Klassifikation durch.
4. **Temporal Smoothing**: 
   - Die letzten N Inferenzergebnisse werden im `TemporalSmoother` gespeichert.
   - Die gewählte Glättungsstrategie wird angewendet, um ein stabileres Ergebnis zu erhalten.
5. **Entscheidungsfindung**: Basierend auf dem geglätteten Ergebnis werden Aktionen ausgelöst (z.B. Status-LEDs, Benachrichtigungen).

Das Temporal Smoothing bildet somit eine kritische Komponente für die Robustheit und Zuverlässigkeit des Systems, insbesondere unter realen Bedingungen mit variablen Lichtverhältnissen, Bewegungen und anderen Störfaktoren.

## Architekturübersicht und Datenfluss

Das Pizza-Erkennungsprojekt besteht aus mehreren Schlüsselkomponenten, die zusammenarbeiten:

1. **Datenvorbereitung**:
   - `augmentation.py` und `augmentation_optimized.py` generieren einen großen Datensatz von augmentierten Pizza-Bildern mit verschiedenen Verbrennungsgraden.
   - Die Augmentierung ist entscheidend, um ein robustes Modell mit begrenzter Datenmenge zu trainieren.

2. **Modelltraining und Optimierung**:
   - `pizza_detector.py` führt die Datenanalyse, das Training, die Optimierung und den Export des Modells durch.
   - Verschiedene Modellarchitekturen (`MicroPizzaNet`, `MicroPizzaNetV2`, `MicroPizzaNetWithSE`) werden für die ressourcenbeschränkte RP2040-Plattform optimiert.
   - Speicheranalyse (`MemoryEstimator`) stellt sicher, dass das Modell auf dem Zielsystem ausgeführt werden kann.
   - Die Quantisierung reduziert die Modellgröße durch Konvertierung von Float32 zu Int8.

3. **Systemsimulation**:
   - `emulator-test.py` simuliert die Ausführung des Modells auf der RP2040-Hardware.
   - Der Workflow von der Bildaufnahme bis zur Inferenz wird emuliert, um die Machbarkeit zu bewerten.

4. **Export und Deployment**:
   - Das trainierte Modell wird für den RP2040-Mikrocontroller exportiert, entweder als quantisiertes TFLite-Modell oder als C-Code.
   - Begleitender C-Code für die Modellausführung wird generiert.

## Herausforderungen und Optimierungen

- **Speicherbeschränkungen**:
  - Der RP2040 hat nur 264KB RAM und 2MB Flash, was spezielle Architekturentscheidungen erfordert.
  - Depthwise Separable Convolutions und Inverted Residual Blocks reduzieren die Parameteranzahl.
  - Int8-Quantisierung reduziert die Modellgröße um etwa den Faktor 4.

- **Rechenleistung**:
  - Der Dual-Core Arm Cortex M0+ bei 133MHz bietet begrenzte Rechenleistung.
  - Die Modellarchitektur ist für die Inferenzgeschwindigkeit optimiert.

- **Batterielebensdauer**:
  - Die Simulation berücksichtigt den Stromverbrauch im aktiven und Standby-Zustand.
  - Das System wechselt zwischen diesen Modi, um die Batterielebensdauer zu maximieren.

## Zusammenfassung

Das Pizza-Erkennungsprojekt demonstriert fortschrittliche Techniken für die Entwicklung eines effizienten Bildklassifikationssystems auf einer ressourcenbeschränkten Hardware. Durch maßgeschneiderte Datenaugmentierung, speicheroptimierte Modellarchitekturen und umfassende Systemsimulation wird ein robustes System entwickelt, das auf dem RP2040-Mikrocontroller laufen kann.

## 7. constants.py

Dieses Modul definiert zentrale Konstanten und Konfigurationsparameter für das gesamte Pizza-Erkennungssystem.

### Hauptkomponenten:

#### 7.1 Hardware-Spezifikationen
- Definition von Hardwareparametern für den RP2040 Mikrocontroller
  - Speichergrenzen (Flash, RAM)
  - Taktfrequenz
  - Maximale Modellgröße und RAM-Bedarf

#### 7.2 Kamerakonfiguration
- Einstellungen für die OV2640-Kamera
  - Auflösung (160x120)
  - Bildrate (7 FPS)
  - Bildformat (RGB565)
  - Mindestintervall zwischen Aufnahmen

#### 7.3 Bildverarbeitung
- Normalisierungsparameter für Eingabebilder
  - Bildgröße (64x64)
  - RGB-Mittelwerte und Standardabweichungen
  - Unterstützte Bildformate

#### 7.4 Modellparameter
- Konfiguration des Klassifikationsmodells
  - Klassenanzahl und -namen
  - Farbzuordnung für verschiedene Pizzaklassen
  - Kanalgrößen für Faltungs- und vollständig verbundene Schichten
  - Dropout-Rate und Batch-Normalisierung

#### 7.5 Trainingskonfiguration
- Parameter für das Modelltraining
  - Batch-Größe, Lernrate, Epochenanzahl
  - Verhältnis Trainings-/Validierungsdaten
  - Zufallssamen für Reproduzierbarkeit
  - Augmentierungsfaktor

#### 7.6 Quantisierung
- Parameter für die Modellquantisierung
  - Quantisierungstyp und -modus
  - Anzahl der Kalibrierungsschritte

#### 7.7 Systemgrenzen
- Leistungsbezogene Grenzwerte
  - Maximale Inferenzzeit
  - Minimaler Konfidenzschwellwert
  - Maximaler Stromverbrauch

#### 7.8 Dateisystemstruktur
- Definition der Projektverzeichnisstruktur
  - Pfade für Daten, Modelle, Ausgaben und Logs
  - Unterverzeichnisse für verschiedene Datentypen

#### 7.9 Logging-Konfiguration
- Einstellungen für die Protokollierung
  - Logdateien und Rotationsgröße
  - Anzahl der Backup-Dateien

#### 7.10 Visualisierungsparameter
- Konfiguration für Datenvisualisierung
  - Auflösung und Größe für Plots
  - Farbpalette für verschiedene Visualisierungselemente

### Bedeutung für das Projekt:

Das `constants.py`-Modul dient als zentrale Konfigurationsdatei für das gesamte System und fördert eine konsistente Parametrisierung über alle Projektkomponenten hinweg. Durch die Zentralisierung der Konfiguration wird die Wartbarkeit verbessert und die Anpassung des Systems an verschiedene Anforderungen erleichtert.

## 8. devices.py

Dieses Modul implementiert die Hardware-Abstraktion und das Management für die physischen Komponenten des Pizza-Erkennungssystems auf dem RP2040 Mikrocontroller.

### Hauptkomponenten:

#### 8.1 PowerMode Enum
```python
class PowerMode(Enum):
    """Energiesparmodi des Systems."""
    ACTIVE = "active"
    SLEEP = "sleep"
    DEEP_SLEEP = "deep_sleep"
```
- **Funktionalität**: Definiert die verschiedenen Energiesparmodi des Systems für eine optimierte Batterielebensdauer.

#### 8.2 DeviceStatus Enum
```python
class DeviceStatus(Enum):
    """Gerätestatus."""
    OK = "ok"
    ERROR = "error"
    BUSY = "busy"
    LOW_BATTERY = "low_battery"
    CRITICAL_BATTERY = "critical_battery"
```
- **Funktionalität**: Definiert die möglichen Betriebszustände des Geräts für Statusberichte und Entscheidungsfindung.

#### 8.3 BatteryStatus
```python
class BatteryStatus:
    """Batteriestatusüberwachung."""
```
- **Funktionalität**: Überwacht und verwaltet den Ladezustand der Batterie.
- **Methoden**:
  - `update(voltage_mv)`: Aktualisiert den Batteriestatus basierend auf der gemessenen Spannung.
  - `is_low()`: Prüft, ob die Batterie einen niedrigen Ladezustand hat.
  - `is_critical()`: Prüft, ob die Batterie einen kritischen Ladezustand erreicht hat.

#### 8.4 Camera
```python
class Camera:
    """OV2640 Kamera-Management."""
```
- **Funktionalität**: Steuert die OV2640-Kamera über GPIO-Pins.
- **Methoden**:
  - `initialize()`: Initialisiert die Kamera.
  - `capture_frame()`: Nimmt ein Bild auf und gibt es als NumPy-Array zurück.
  - `set_power_mode(mode)`: Setzt die Kamera in einen bestimmten Energiesparmodus.

#### 8.5 SystemController
```python
class SystemController:
    """RP2040 System-Controller."""
```
- **Funktionalität**: Zentraler Controller, der alle Hardwarekomponenten koordiniert.
- **Attribute**:
  - `camera`: Instanz der Kameraklasse.
  - `battery`: Instanz der Batteriestatusklasse.
  - `status`: Aktueller Gerätestatus.
  - `power_mode`: Aktueller Energiesparmodus.
- **Methoden**:
  - `initialize()`: Initialisiert alle Systemkomponenten.
  - `get_system_stats()`: Liefert einen detaillierten Systembericht.
  - `update_power_state()`: Aktualisiert den Energiezustand basierend auf dem Batteriestatus.
  - `set_power_mode(mode)`: Setzt das System in einen bestimmten Energiesparmodus.

### Integration in das Gesamtsystem:

Das `devices.py`-Modul bildet die Brücke zwischen der Software-Logik und der physischen Hardware. Es abstrahiert die Hardwaredetails und bietet eine einheitliche Schnittstelle für den Zugriff auf Kamera, Energiemanagement und Systemstatus. Die Implementierung berücksichtigt die besonderen Anforderungen ressourcenbeschränkter Systeme und ermöglicht einen effizienten Betrieb auf dem RP2040 Mikrocontroller.

## 9. power_manager.py

Dieses Modul implementiert ein fortschrittliches Energiemanagement für das RP2040-basierte Pizza-Erkennungssystem, um die Batterielebensdauer zu maximieren und den Betrieb unter verschiedenen Bedingungen zu optimieren.

### Hauptkomponenten:

#### 9.1 PowerUsage
```python
@dataclass
class PowerUsage:
    """Energieverbrauchsdaten für verschiedene Systemzustände."""
```
- **Funktionalität**: Speichert und verwaltet Stromverbrauchsdaten für verschiedene Betriebszustände des Systems.
- **Attribute**:
  - Stromverbrauch in verschiedenen Modi (Sleep, Idle, Active, etc.)
- **Methoden**:
  - `get_total_active_current()`: Berechnet den Gesamtstromverbrauch im aktiven Zustand.
  - `scale_for_temperature(temperature_c)`: Passt den Stromverbrauch basierend auf der Temperatur an.

#### 9.2 AdaptiveMode Enum
```python
class AdaptiveMode(Enum):
    """Betriebsmodi für das adaptive Energiemanagement."""
```
- **Funktionalität**: Definiert verschiedene Betriebsmodi für das adaptive Energiemanagement, die unterschiedliche Kompromisse zwischen Leistung und Energieeffizienz bieten.
- **Modi**:
  - `PERFORMANCE`: Maximale Leistung, höchster Energieverbrauch
  - `BALANCED`: Ausgewogenes Verhältnis zwischen Leistung und Batterielebensdauer
  - `POWER_SAVE`: Maximale Batterielebensdauer, reduzierte Leistung
  - `ULTRA_LOW_POWER`: Extrem niedriger Energieverbrauch für kritische Batteriebedingungen
  - `ADAPTIVE`: Passt sich automatisch an Erkennungsmuster an
  - `CONTEXT_AWARE`: Passt sich basierend auf erkannten Pizzatypen an

#### 9.3 PowerManager
```python
class PowerManager:
    """Intelligentes Energiemanagement für das RP2040-System."""
```
- **Funktionalität**: Hauptklasse für das adaptive Energiemanagement, die verschiedene Strategien zur Optimierung der Batterielebensdauer implementiert.
- **Kernmethoden**:
  - `set_mode(mode)`: Ändert den Betriebsmodus des Energiemanagements.
  - `update_temperature(temperature_c)`: Aktualisiert die Temperaturmessung und passt Energieberechnungen an.
  - `update_detection_class(class_id)`: Aktualisiert den Erkennungskontext für kontextbasiertes Energiemanagement.
  - `get_next_sampling_interval()`: Berechnet das optimale Abtastintervall basierend auf dem aktuellen Modus.
  - `should_enter_sleep()` & `should_wake_up()`: Entscheiden über Schlafmodus-Übergänge.
  - `update_energy_consumption()`: Aktualisiert Energieverbrauchsstatistiken.
  - `recommend_power_mode()`: Empfiehlt einen optimalen Energiemodus basierend auf Systemdaten.

### Fortschrittliche Funktionen:

1. **Adaptive Abtastraten**:
   - Das System passt die Häufigkeit der Messungen dynamisch an, basierend auf:
     - Erkannter Aktivität (Änderungen in der Pizzaklassifikation)
     - Temperaturbedingungen
     - Batterieladezustand

2. **Kontextbasiertes Energiemanagement**:
   - Passt das Verhalten basierend auf erkannten Pizzatypen an:
     - Häufigere Überprüfungen bei kritischen Zuständen (Pizza kurz vor dem Verbrennen)
     - Seltenere Überprüfungen bei stabilen Zuständen

3. **Temperaturbasierte Anpassung**:
   - Berücksichtigt den erhöhten Stromverbrauch bei höheren Temperaturen
   - Passt Messintervalle und Schlafperioden entsprechend an

4. **Präzise Batterielebensdauerprognose**:
   - Berechnet die geschätzte Laufzeit basierend auf:
     - Tatsächlichem Duty-Cycle (Verhältnis aktive Zeit / Gesamtzeit)
     - Temperaturangepasstem Stromverbrauch
     - Historischen Verbrauchsmustern

### Integration in das Gesamtsystem:

Der PowerManager ist eine kritische Komponente für den Einsatz des Pizza-Erkennungssystems in batteriebetriebenen Anwendungen. Er ermöglicht einen effizienten Betrieb und maximiert die Nutzungsdauer zwischen den Ladevorgängen. Die adaptive Natur des Systems sorgt dafür, dass kritische Erkennungen (z.B. eine fast verbrannte Pizza) nicht verpasst werden, während gleichzeitig unnötige Aktivität vermieden wird.

## 10. status_display.py

Dieses Modul implementiert verschiedene Anzeigemöglichkeiten für das Pizza-Erkennungssystem, um Statusinformationen und Erkennungsergebnisse zu visualisieren. Es unterstützt sowohl RGB-LEDs als auch OLED-Displays und bietet eine flexible Abstraktionsschicht für verschiedene Hardware-Konfigurationen.

### Hauptkomponenten:

#### 10.1 LEDState Enum
```python
class LEDState(Enum):
    """Status-LED-Zustände"""
    OFF = "off"
    ON = "on"
    BLINK_SLOW = "blink_slow"  # 1Hz
    BLINK_FAST = "blink_fast"  # 2Hz
    PULSE = "pulse"            # Sanftes Pulsieren
```
- **Funktionalität**: Definiert verschiedene Anzeigemodi für Status-LEDs, die für unterschiedliche Systemzustände verwendet werden können.

#### 10.2 StatusDisplay (Basisklasse)
```python
class StatusDisplay:
    """Basisklasse für Status-Anzeigen (LED oder Display)"""
```
- **Funktionalität**: Abstrakte Basisklasse, die eine einheitliche Schnittstelle für verschiedene Anzeigetypen definiert.
- **Kernmethoden**:
  - `initialize()`: Initialisiert die Hardware.
  - `show_status(status, color, state)`: Zeigt einen Status mit angegebener Farbe und Zustand an.
  - `show_inference_result(result)`: Zeigt das Ergebnis einer Inferenz an.
  - `show_error(error_message)`: Zeigt einen Fehlerstatus an.
  - `clear()`: Löscht die Anzeige.
  - `close()`: Gibt Ressourcen frei.

#### 10.3 SimulatedRGBLED
```python
class SimulatedRGBLED(StatusDisplay):
    """Simulierte RGB-LED für Testzwecke."""
```
- **Funktionalität**: Implementiert eine simulierte RGB-LED für Entwicklungs- und Testzwecke, die Zustandsänderungen im Logger ausgibt.
- **Einsatz**: Wird verwendet, wenn keine echte Hardware verfügbar ist oder für Unit-Tests.

#### 10.4 RP2040RGBLED
```python
class RP2040RGBLED(StatusDisplay):
    """Hardware-Implementation für die RGB-LED auf dem RP2040"""
```
- **Funktionalität**: Steuert eine echte RGB-LED über die GPIO-Pins des RP2040.
- **Besondere Merkmale**:
  - Unterstützt sowohl gemeinsame Anode als auch gemeinsame Kathode LEDs
  - Implementiert verschiedene Blink- und Pulseffekte über einen separaten Thread
  - Verwendet PWM zur präzisen Farbsteuerung

#### 10.5 OLEDDisplay
```python
class OLEDDisplay(StatusDisplay):
    """Implementation für ein OLED-Display (SSD1306 oder ähnlich) über I2C."""
```
- **Funktionalität**: Steuert ein OLED-Display über I2C zur detaillierten Anzeige von Statusinformationen.
- **Besondere Merkmale**:
  - Zeigt Klassenname, Konfidenz und Zeitstempel an
  - Visualisiert Konfidenzwerte als Balkendiagramm
  - Teilt längere Texte automatisch in mehrere Zeilen auf

#### 10.6 Helfer-Funktionen
```python
def create_status_display(display_type="auto", **kwargs) -> StatusDisplay:
    """Erstellt eine Status-Anzeige passend zur Laufzeitumgebung"""
```
- **Funktionalität**: Factory-Funktion, die die passende Anzeigeimplementierung basierend auf der verfügbaren Hardware und den Benutzereinstellungen erstellt.
- **Modi**:
  - "auto": Erkennt automatisch die beste verfügbare Anzeigeoption
  - "rgb_led": Erstellt eine RP2040RGBLED-Instanz
  - "oled": Erstellt eine OLEDDisplay-Instanz
  - "simulated": Erstellt eine SimulatedRGBLED-Instanz für Tests

### Integration in das Gesamtsystem:

Das Statusanzeigesystem ist eine wichtige Komponente für die Benutzerschnittstelle des Pizza-Erkennungssystems. Es ermöglicht die Visualisierung von Systemzuständen und Erkennungsergebnissen ohne die Notwendigkeit eines externen Displays oder einer Verbindung zu einem Computer.

Die Flexibilität des Moduls erlaubt verschiedene Hardware-Konfigurationen:
- Einfache Implementierung mit einer RGB-LED für kostengünstige Systeme
- Detailliertere Anzeige mit einem OLED-Display für erweiterte Benutzerschnittstellen
- Simulierte Ausgabe für Entwicklung und Tests ohne physische Hardware

Die Farbcodierung und verschiedenen Anzeigemodi (dauerhaft, blinkend, pulsierend) ermöglichen eine intuitive Darstellung des Pizzazustands und der Systemsicherheit, selbst mit minimalen Anzeigemöglichkeiten.

## 11. utils.py und types.py

Diese Module bieten Hilfsfunktionen und Datenstrukturen für das Pizza-Erkennungssystem, die in verschiedenen Teilen des Projekts verwendet werden. Sie fördern die Wiederverwendung von Code und verbessern die Lesbarkeit und Wartbarkeit des Projekts.

### 11.1 types.py - Gemeinsame Datenstrukturen

Dieses Modul definiert grundlegende Datentypen, die in der gesamten Anwendung verwendet werden.

#### Kernstrukturen:
- **InferenceResult**: Datenklasse zur Speicherung von Inferenzergebnissen
  - `class_name`: Name der erkannten Klasse
  - `confidence`: Konfidenzwert (0-1)
  - `prediction`: Numerischer Klassenindex
  - `probabilities`: Dictionary mit Wahrscheinlichkeiten für alle Klassen

- **SystemConfig**: Konfigurationsparameter für das System
  - Hardware-Einstellungen
  - Erkennungsparameter
  - Energiemanagement-Einstellungen

- **PizzaClass**: Enum für verschiedene Pizzazustände
  - Definiert alle möglichen Pizzaklassen mit Beschreibungen und Standardfarben

### 11.2 utils.py - Allgemeine Hilfsfunktionen

Dieses Modul bietet verschiedene Hilfsfunktionen für häufig verwendete Operationen.

#### Bildverarbeitung:
- **Bildkonvertierung**: Funktionen zum Konvertieren zwischen verschiedenen Bildformaten
  - `convert_rgb565_to_rgb888(data, width, height)`: Konvertiert RGB565-Daten zu RGB888
  - `resize_image(image, target_size)`: Ändert die Größe eines Bildes unter Beibehaltung des Seitenverhältnisses

#### Dateisystem:
- **Dateiverwaltung**: Funktionen für den Umgang mit Dateien und Verzeichnissen
  - `ensure_directory(path)`: Erstellt ein Verzeichnis, falls es nicht existiert
  - `get_unique_filename(base_path, extension)`: Generiert einen eindeutigen Dateinamen

#### Modellverwaltung:
- **Modellfunktionen**: Funktionen zum Laden und Verwalten von Modellen
  - `load_model(model_path, device)`: Lädt ein Modell von einer Datei
  - `get_model_size(model)`: Berechnet die Größe eines Modells in MB

#### Konfiguration:
- **Konfigurationsverwaltung**: Funktionen zum Laden und Speichern von Konfigurationen
  - `load_config(config_file)`: Lädt Konfigurationsparameter aus einer Datei
  - `save_config(config, config_file)`: Speichert Konfigurationsparameter in einer Datei

#### Zeit und Performance:
- **Timing-Funktionen**: Hilfen zur Zeitmessung und Performance-Analyse
  - `measure_inference_time(model, input_tensor)`: Misst die Inferenzzeit eines Modells
  - `calculate_fps(time_elapsed, frame_count)`: Berechnet Frames pro Sekunde

#### Batteriemanagement:
- **Batteriefunktionen**: Hilfsfunktionen für Batterieberechnungen
  - `estimate_battery_life(capacity_mah, current_ma)`: Schätzt die Batterielebensdauer
  - `calculate_duty_cycle(active_time, sleep_time)`: Berechnet den Duty-Cycle

### Bedeutung für das Projekt:

Die `utils.py` und `types.py` Module bilden das Rückgrat für viele Funktionen im Pizza-Erkennungssystem. Sie bieten gut getestete, wiederverwendbare Komponenten, die die Entwicklung neuer Funktionen beschleunigen und die Konsistenz im gesamten Codebase fördern. Durch die Zentralisierung häufig verwendeter Funktionen wird die Wahrscheinlichkeit von Fehlern reduziert und die Wartbarkeit des Codes verbessert.

## 12. visualization.py

Dieses Modul implementiert umfangreiche Visualisierungsfunktionen für das Pizza-Erkennungssystem, die zur Darstellung von Trainingsergebnissen, Modellleistung, Ressourcenverbrauch und anderen Aspekten des Systems verwendet werden.

### Hauptkomponenten:

#### 12.1 Inferenz-Visualisierung
```python
def plot_inference_result(image, result, output_path=None)
```
- **Funktionalität**: Visualisiert ein einzelnes Inferenzergebnis mit dem analysierten Bild und einem Balkendiagramm der Klassenwahrscheinlichkeiten.
- **Parameter**:
  - `image`: Das ursprüngliche Bild als NumPy-Array
  - `result`: Ein InferenceResult-Objekt mit erkannter Klasse und Konfidenzwerten
  - `output_path`: Optionaler Pfad zum Speichern der Ausgabe (sonst wird sie angezeigt)

```python
def annotate_image(image, result, draw_confidence=True)
```
- **Funktionalität**: Fügt Erkennungsinformationen direkt zum Bild hinzu, z.B. einen farbigen Rahmen und Klassenbezeichnung.
- **Rückgabe**: Ein neues Bild mit den annotierten Informationen.

```python
def plot_inference_results(results, output_path=None)
```
- **Funktionalität**: Erstellt ein Balkendiagramm zum Vergleich mehrerer Inferenzergebnisse.

#### 12.2 Modellanalyse und -bewertung
```python
def plot_confusion_matrix(confusion_matrix, output_path=None)
```
- **Funktionalität**: Erstellt eine Heatmap-Visualisierung einer Konfusionsmatrix zur Bewertung der Klassifikationsleistung.
- **Parameter**:
  - `confusion_matrix`: NumPy-Array der Konfusionsmatrix
  - `output_path`: Optionaler Pfad zum Speichern der Ausgabe

```python
def plot_training_progress(epochs, losses, metrics, output_path=None)
```
- **Funktionalität**: Visualisiert den Trainingsverlauf mit Verlust- und Metrik-Kurven über die Epochen.
- **Parameter**:
  - `epochs`: Liste der Epochennummern
  - `losses`: Liste der Verlustwerte für jede Epoche
  - `metrics`: Dictionary mit Listen verschiedener Metriken (z.B. Genauigkeit)
  - `output_path`: Optionaler Pfad zum Speichern der Ausgabe

```python
def plot_metrics_comparison(metrics_list, output_path=None)
```
- **Funktionalität**: Vergleicht Leistungsmetriken verschiedener Modelle in einem Balkendiagramm.
- **Parameter**:
  - `metrics_list`: Liste von Tupeln (Modellname, ModelMetrics-Objekt)

```python
def visualize_model_architecture(model, input_shape, output_path=None)
```
- **Funktionalität**: Erzeugt eine graphische Darstellung der Modellarchitektur mithilfe von torchviz.
- **Parameter**:
  - `model`: PyTorch-Modell
  - `input_shape`: Form des Eingabetensors
  - `output_path`: Optionaler Pfad zum Speichern der Ausgabe

#### 12.3 Ressourcen- und Energieanalyse
```python
def plot_resource_usage(usage_history, output_path=None)
```
- **Funktionalität**: Visualisiert den zeitlichen Verlauf des Ressourcenverbrauchs (RAM, Flash, CPU, Leistung).
- **Parameter**:
  - `usage_history`: Liste von ResourceUsage-Objekten
  - `output_path`: Optionaler Pfad zum Speichern der Ausgabe

```python
def plot_power_profile(profile, output_path=None)
```
- **Funktionalität**: Stellt das Energieprofil einer Inferenz dar, einschließlich Durchschnitts- und Spitzenstromverbrauch.
- **Parameter**:
  - `profile`: PowerProfile-Objekt mit Energiedaten
  - `output_path`: Optionaler Pfad zum Speichern der Ausgabe

#### 12.4 Berichterstellung
```python
def create_report(model_metrics, resource_usage, power_profile, output_dir=None)
```
- **Funktionalität**: Erstellt einen umfassenden HTML-Bericht mit allen relevanten Visualisierungen und Metriken.
- **Parameter**:
  - `model_metrics`: ModelMetrics-Objekt mit Klassifikationsmetriken
  - `resource_usage`: Liste von ResourceUsage-Objekten für Ressourcennutzung
  - `power_profile`: PowerProfile-Objekt mit Energiedaten
  - `output_dir`: Optionaler Ausgabeverzeichnispfad

### Integration in das Gesamtsystem:

Das Visualisierungsmodul spielt eine zentrale Rolle bei der Analyse und Präsentation der Ergebnisse des Pizza-Erkennungssystems. Es bietet:

1. **Echtzeit-Feedback** für Entwickler während des Trainings und der Optimierung
2. **Validierungswerkzeuge** zur Bewertung der Modellleistung
3. **Ressourcenanalyse** für die Optimierung des Modells für die RP2040-Plattform
4. **Berichterstellung** für detaillierte Dokumentation der Systemleistung

### Verwendung in verschiedenen Projektphasen:

- **Entwicklungsphase**: Visualisierung des Trainingsverlaufs und der Architektur zur iterativen Verbesserung des Modells
- **Evaluierungsphase**: Erstellung von Konfusionsmatrizen und Metrikvergleichen zur Bewertung der Klassifikationsleistung
- **Optimierungsphase**: Analyse des Ressourcenverbrauchs zur Anpassung an die Zielplattform
- **Deployment-Phase**: Erstellung von Berichten zur Dokumentation der Systemleistung
- **Betriebsphase**: Visualisierung der Inferenzergebnisse zur Überprüfung der Erkennungsqualität

Das Modul unterstützt sowohl interaktive Visualisierungen für die Entwicklung als auch automatisierte Berichterstellung für kontinuierliche Integration und Überwachung des Produktionssystems.

## 13. Modellarchitekturen und Optimierung

Dieses Kapitel dokumentiert die verschiedenen Modellarchitekturen, die für das Pizza-Erkennungssystem auf dem RP2040 entwickelt wurden, sowie die angewandten Optimierungstechniken.

### 13.1 Modellarchitekturen

#### 13.1.1 MicroPizzaNet (Basis-Architektur)
```python
class MicroPizzaNet(nn.Module):
    """Ein ultrakompaktes CNN für die Pizza-Erkennung auf RP2040"""
```
- **Architektur**: Klassisches CNN mit ressourcenoptimierter Struktur
- **Hauptmerkmale**:
  - Reduzierte Anzahl von Faltungsschichten (3-4 Layer)
  - Kleine Filterzahlen (beginnend mit 16-32 Kanälen)
  - Frühe Auflösungsreduktion durch Striding statt Pooling
  - Kleine vollständig verbundene Schichten (128-256 Neuronen)
- **Speicherfußabdruck**: ~150-180KB (Float32), ~45-55KB (nach Int8-Quantisierung)
- **Inferenzzeit auf RP2040**: ~80-100ms

#### 13.1.2 MicroPizzaNetV2 (Inverted Residual)
```python
class MicroPizzaNetV2(nn.Module):
    """Verbesserte Version mit Inverted Residual Blocks"""
```
- **Architektur**: MobileNetV2-inspirierte Struktur mit Inverted Residual Blocks
- **Hauptmerkmale**:
  - Depthwise Separable Convolutions zur Parameterreduktion
  - Expansion-Reduction-Struktur in Residualblöcken
  - Verbindungen mit Skip-Connections
  - Effizientes Channel Shuffling
- **Speicherfußabdruck**: ~120-150KB (Float32), ~35-45KB (nach Int8-Quantisierung)
- **Inferenzzeit auf RP2040**: ~60-80ms

#### 13.1.3 MicroPizzaNetWithSE (Squeeze-and-Excitation)
```python
class MicroPizzaNetWithSE(nn.Module):
    """Integration von Squeeze-and-Excitation in das Basismodell"""
```
- **Architektur**: Basisarchitektur erweitert mit Squeeze-and-Excitation-Modulen
- **Hauptmerkmale**:
  - Channel-weise Aufmerksamkeitsmechanismen
  - Adaptives Gewichten von Kanälen basierend auf deren Wichtigkeit
  - Verbesserte Fokussierung auf relevante Bildmerkmale
- **Speicherfußabdruck**: ~160-190KB (Float32), ~50-60KB (nach Int8-Quantisierung)
- **Inferenzzeit auf RP2040**: ~90-110ms

### 13.2 Optimierungstechniken

#### 13.2.1 Quantisierungsbasierte Optimierung
- **Quantization-Aware Training (QAT)**:
  - Training mit simulierter Quantisierung für minimalen Genauigkeitsverlust
  - Explizites Modellieren der Quantisierungseffekte während des Trainings
  - Feinabstimmung der Quantisierungsparameter
- **Post-Training Quantization (PTQ)**:
  - Konvertierung des trainierten Float32-Modells zu Int8 nach dem Training
  - Kalibrierung mit repräsentativen Datensätzen
  - Verwendung dynamischer Quantisierungsbereiche für verschiedene Schichten

#### 13.2.2 Pruning und Strukturelle Optimierung
- **Channel Pruning**:
  - Entfernung unwichtiger Kanäle basierend auf L1-Norm oder anderen Kriterien
  - Schrittweise Pruning mit Feinabstimmung zwischen den Schritten
  - Zielgerichtetes Pruning für bestimmte Schichten
- **Strukturelle Optimierung**:
  - Ersetzen konventioneller Faltungen durch depthwise separable Faltungen
  - Verwendung effizienterer Aktivierungsfunktionen (z.B. ReLU6, Hard-Swish)
  - Optimierung der Tensorflüsse zur Minimierung des Arbeitsspeichers

#### 13.2.3 Knowledge Distillation
- **Teacher-Student-Training**:
  - Training eines kompakten Modells (Student) unter Anleitung eines größeren Modells (Teacher)
  - Übertragung von "weichem Wissen" durch Angleichung der Logits oder Feature-Maps
  - Kombinierte Verlustfunktion aus hartem (Labels) und weichem (Teacher) Wissen
- **Progressive Komprimierung**:
  - Schrittweise Verkleinerung und Optimierung des Modells
  - Verwendung mehrerer intermediärer Teacher-Modelle

#### 13.2.4 Hardware-spezifische Optimierungen
- **RP2040-spezifische Anpassungen**:
  - Vermeidung komplexer Operationen wie div, exp, log
  - Präferenz für Operationen, die gut auf ARM Cortex-M0+ abbildbar sind
  - Präzise Planung des Speicherlayouts zur Optimierung der Cachenutzung
- **Aktivierungsspeicher-Optimierung**:
  - In-Place-Operationen wo möglich
  - Wiederverwendung von Aktivierungstensoren

### 13.3 Leistungsvergleich der Modelle

| Modell | Größe (KB) | Genauigkeit (%) | Inferenzzeit (ms) | Stromverbrauch (mW) |
|--------|------------|-----------------|-------------------|---------------------|
| MicroPizzaNet | 48 (Int8) | 91.5 | 85 | 90 |
| MicroPizzaNetV2 | 40 (Int8) | 92.3 | 65 | 75 |
| MicroPizzaNetWithSE | 55 (Int8) | 93.8 | 95 | 100 |

Alle Modelle wurden so optimiert, dass sie innerhalb der Speicherbeschränkungen des RP2040 (264KB RAM, 2MB Flash) arbeiten können, wobei ein Kompromiss zwischen Genauigkeit, Geschwindigkeit und Energieeffizienz angestrebt wurde.

## 14. CI/CD-Pipeline und Automatisierung

Dieses Kapitel dokumentiert die kontinuierliche Integrations- und Bereitstellungspipeline (CI/CD) für das Pizza-Erkennungsprojekt, die eine effiziente Entwicklung, Tests und Deployment ermöglicht.

### 14.1 CI/CD-Architektur

Die CI/CD-Pipeline für das Pizza-Erkennungssystem ist darauf ausgelegt, eine konsistente und zuverlässige Entwicklung, Prüfung und Bereitstellung der Software auf der RP2040-Plattform zu gewährleisten. Sie umfasst:

1. **Quellcode-Management**: Git-basiertes Versionskontrollsystem
2. **Automatisierte Tests**: Unit-Tests, Integrationstests und Hardware-in-the-Loop-Tests
3. **Continuous Integration**: Automatisierte Builds und Tests bei jedem Commit
4. **Continuous Deployment**: Automatisierte Bereitstellung auf Testgeräten
5. **Leistungsüberwachung**: Kontinuierliche Überwachung der Modellleistung und Ressourcennutzung

### 14.2 Automatisierte Tests

#### 14.2.1 Unit-Tests
- **Python-seitige Tests**:
  - Tests für Datenaufbereitung und Augmentierung
  - Tests für Modellarchitektur und Training
  - Tests für Visualisierungs- und Analysefunktionen
- **C-seitige Tests**:
  - Tests für Modellimplementierung auf dem RP2040
  - Tests für Speicherverwaltung
  - Tests für Hardware-Abstraktionsschicht

#### 14.2.2 Integrationstests
- Tests für die Kommunikation zwischen Komponenten
- Tests für das Zusammenspiel von Kamera, Inferenz und Status-Anzeige
- Tests für die Energieverwaltung und Schlafmodi

#### 14.2.3 Hardware-in-the-Loop-Tests
- Automatisierte Tests auf echter RP2040-Hardware
- Simulierte Bildsequenzen zur Validierung der Erkennungsleistung
- Stresstests unter verschiedenen Umgebungsbedingungen (Temperatur, Beleuchtung)

### 14.3 Kontinuierliche Integration

Die kontinuierliche Integration wird durch einen automatisierten Workflow realisiert, der bei jedem Commit folgende Schritte ausführt:

1. **Code-Qualitätsprüfung**:
   - Statische Codeanalyse mit flake8 und pylint
   - Typprüfung mit mypy
   - Code-Formatierungsprüfung mit black

2. **Build**:
   - Python-Paketerstellung
   - Kompilierung der C/C++-Komponenten für RP2040
   - Erstellung der Firmware-Images

3. **Test-Suite-Ausführung**:
   - Ausführung aller Unit-Tests
   - Ausführung der Integrationstests
   - Simulationsbasierte Tests der Modellleistung

4. **Leistungsanalyse**:
   - Messung der Modellgröße und Inferenzzeit
   - Validierung gegen definierte Leistungsanforderungen
   - Erstellung von Leistungsberichten

### 14.4 Kontinuierliches Deployment

Nach erfolgreicher Integration wird die Software automatisch auf verschiedenen Umgebungen bereitgestellt:

1. **Entwicklungsumgebung**:
   - Deployment auf virtuellen Testumgebungen
   - Emulierte RP2040-Umgebungen für schnelles Feedback

2. **Test-Hardware**:
   - Automatisches Flashen von Testgeräten mit neuer Firmware
   - Durchführung von Hardware-in-the-Loop-Tests
   - Sammlung von Leistungsmetriken unter realen Bedingungen

3. **Produktionsbereitstellung**:
   - Erstellung signierter Firmware-Releases
   - Generierung der Dokumentation und Release Notes
   - Vorbereitung der Deploymentpakete für Feldgeräte

### 14.5 Monitoring und Feedback

Die Pipeline umfasst auch kontinuierliches Monitoring der Anwendung:

1. **Leistungsüberwachung**:
   - Erfassung von Inferenzzeiten und Genauigkeitsdaten
   - Überwachung des Ressourcenverbrauchs
   - Erkennung von Leistungsregressionen

2. **Feedback-Schleife**:
   - Sammlung von Fehlerereignissen
   - Automatisierte Fehlerberichte
   - Integration von Benutzer-Feedback

3. **Kontinuierliche Verbesserung**:
   - Automatische Identifizierung von Optimierungspotential
   - A/B-Tests für Modellvarianten
   - Regelmäßige Neutrainings mit erweiterten Datensätzen

Die CI/CD-Pipeline ist ein entscheidender Bestandteil des Entwicklungsprozesses und ermöglicht eine schnelle Iteration bei gleichzeitiger Aufrechterhaltung hoher Qualitätsstandards für das auf Ressourcen beschränkte Pizza-Erkennungssystem.