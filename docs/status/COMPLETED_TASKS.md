# Abgeschlossene Aufgaben - Pizza-Erkennungssystem

Dieses Dokument enthält alle abgeschlossenen Aufgaben des Pizza-Erkennungsprojekts mit Implementierungsdetails und Verweisen auf die relevanten Dateien. Es dient als Referenz und Nachschlagewerk, während der PROJECT_STATUS.txt den aktuellen Projektstatus und die noch zu erledigenden Aufgaben enthält.

Zuletzt aktualisiert: 2025-05-29

## DIFFUSION-2.1: Targeted Image Generation Pipeline ✅
**Ziel**: Entwicklung einer zielgerichteten Bildgenerierungs-Pipeline für spezifische Eigenschaften

**Implementierung**:
- Vollständige Pipeline für zielgerichtete Bildgenerierung implementiert
- 4 Beleuchtungskonditionen: overhead_harsh, side_dramatic, dim_ambient, backlit_rim
- 3 Verbrennungsgrade: slightly_burnt, moderately_burnt, severely_burnt
- Echtzeit-Eigenschaftsverifikation mit quantitativen Bewertungsalgorithmen
- Umfassendes Metadaten-System mit Generierungsparametern
- Qualitätsbewusste Generierung mit Retry-Mechanismen
- Speicher-optimierte Konfigurationsoptionen
- Standalone-Betrieb ohne bestehende Pipeline-Abhängigkeiten

**Relevante Dateien**:
- `src/augmentation/targeted_diffusion_pipeline.py` (Haupt-Pipeline, 1490 Zeilen)
- `test_targeted_pipeline.py` (Basis-Tests)
- `test_verification_algorithms.py` (Eigenschaftsverifikation)
- `diffusion_2_1_demo.py` (Umfassende Demonstration)
- `DIFFUSION_2_1_COMPLETE.md` (Vollständige Dokumentation)

**Technische Highlights**:
- Template-basierte Prompt-Generierung mit Variationen
- Schatten-Ratio und Helligkeitsanalyse für Beleuchtungsverifikation
- Farbbasierte Analyse für Verbrennungsgrad-Erkennung
- Konfigurierbare Verifikationsschwellen
- Umfassende Statistik-Verfolgung und Berichterstattung

## 1. Bild-Vorverarbeitung on-device ✅
**Ziel**: Automatische Beleuchtungs-Korrektur direkt auf dem RP2040

**Implementierung**:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) in C implementiert
- Leichtgewichtiger Algorithmus für RP2040 optimiert
- Vollständig in `test_image_preprocessing.py` implementiert
- C-Code in den Modell-Export-Dateien für den RP2040 integriert
- Temperatur und Arbeitsspeicher-Nutzung validiert

**Relevante Dateien**:
- `scripts/test_image_preprocessing.py`
- `models/rp2040_export/pizza_model.c` (enthält Vorverarbeitungscode)
- `models/rp2040_export/pizza_model.h`

## 2. Mehrbild-Entscheidung (Temporal Smoothing) ✅
**Ziel**: Verlässlicheres Ergebnis durch Abstimmung mehrerer Inferenz-Durchläufe

**Implementierung**:
- Mehrere aufeinanderfolgende Klassifikationen (n=5) implementiert
- Drei Strategien implementiert:
  - Mehrheitsvotum (Majoritätsentscheidung)
  - Gleitender Mittelwert
  - Exponentiell gewichteter Mittelwert
- Evaluation aller drei Strategien mit Vergleichsberichten

**Relevante Dateien**:
- `scripts/test_temporal_smoothing.py`
- `src/pizza_detector.py` (enthält die Smoothing-Logik)

## 3. Datensatz-Management & Label-Tool ✅
**Ziel**: Schnelles und konsistentes Nachlabeln neuer Pizza-Bilder

**Implementierung**:
- Umfassende GUI-Anwendung mit PyQt entwickelt
- Bildanzeige, Labelauswahl und JSON-Update-Funktionalität
- Integrierte KI-Vorschläge für effizienteres Labeln
- Git-basierte Versionierung des Datensatzes
- Batch-Labeling-Funktionalität

**Relevante Dateien**:
- `scripts/label_tool.py`
- `data/class_definitions.json`

## 4. Architektur-Hyperparameter-Suche ✅
**Ziel**: Optimales Trade-off aus Genauigkeit, Speicher & Tempo

**Implementierung**:
- Grid-Search über verschiedene Architekturparameter:
  - Depth-Multiplikatoren
  - Anzahl der Blöcke
  - Kanalbreiten
- Automatisiertes Training und Evaluation
- Umfassende Metrik-Erfassung und Visualisierung
- Export aller Ergebnisse in CSV und als Diagramme

**Relevante Dateien**:
- `scripts/hyperparameter_search.py`
- `hyperparameter_search.log`
- `models/evaluation_report.json`

## 5. Vergleich alternativer Tiny-CNNs ✅
**Ziel**: Bewertung von MCUNet & reduzierter MobileNet-Variante

**Implementierung**:
- Integration mehrerer leichtgewichtiger CNN-Architekturen
- Vergleich von MicroPizzaNet mit MCUNet und MobileNetV2-Tiny
- Messung von Inferenzzeit, Modellgröße und Genauigkeit
- Erstellung eines detaillierten Vergleichsberichts

**Relevante Dateien**:
- `scripts/compare_tiny_cnns.py`
- `tiny_cnn_comparison.log`
- `output/model_comparisons/tiny_cnn_comparison_report.html`

## 6. Offline-Visualisierung mit Grad-CAM ✅
**Ziel**: Heatmaps für falsch klassifizierte Bilder

**Implementierung**:
- Grad-CAM-Implementierung für MicroPizzaNet
- Generierung von Heatmaps für alle Testbilder
- Besonderer Fokus auf falsch klassifizierte Bilder
- Erzeugung von HTML- und PDF-Berichten mit Bild, Heatmap und Erklärungen

**Relevante Dateien**:
- `scripts/visualize_gradcam.py`
- `gradcam_visualization.log`
- `models/visualizations/gradcam_report.html`

## 7. CI/CD-Pipeline für Modell & Firmware ✅
**Ziel**: Automatisierte End-to-End-Updates

**Implementierung**:
- GitHub Actions Workflow eingerichtet
- Automatisierter Prozess:
  - Modeltraining
  - Quantisierung
  - C-Code-Generierung
  - Firmware-Build
  - Tests
- Benachrichtigungssystem (Slack/E-Mail) für Erfolg/Fehler

**Relevante Dateien**:
- `.github/workflows/model_pipeline.yml`
- `docs/ci_cd_pipeline.md`

## 8. On-Device Performance-Logging ⚠️ (Teilweise implementiert)
**Ziel**: Flaschenhälse sichtbar machen

**Implementierung**:
- Messung von Inferenz-Zyklen und Peak-RAM mit Hardware-Timern
- Analyse-Skript zur Auswertung der Performance-Logs
- UART-Logging implementiert, SD-Karten-Logging noch ausstehend

**Relevante Dateien**:
- `scripts/analyze_performance_logs.py`
- `pizza_training_detailed.log`

## 9. Einfache Statusanzeige ✅
**Ziel**: Direkte Rückmeldung am Gerät

**Implementierung**:
- Unterstützung für RGB-LED und OLED-Display
- Verschiedene visuelle Effekte basierend auf Erkennungsergebnissen
- Anzeige von Klassen (basic vs. burnt) mit Farben/Text
- Vollständig dokumentierte Schaltpläne und Beispielcode

**Relevante Dateien**:
- `scripts/demo_status_display.py`
- `src/status_display.py`
- `docs/hardware-documentation.html`

## 10. Automatisierte Test-Suite ✅
**Ziel**: Kontinuierliche Qualitätssicherung

**Implementierung**:
- Umfangreiche Testbilder in `data/test/`
- Unit-Tests für alle Klassifikationsfunktionen
- Automatisierte Test-Suite mit pytest
- Generierung von Coverage- und Accuracy-Berichten
- Mindestgenauigkeitsschwellen für Tests definiert

**Relevante Dateien**:
- `scripts/automated_test_suite.py`
- `scripts/test_pizza_classification.py`
- `scripts/run_pizza_tests.py`
- `pytest.ini`

## 11. CNN-Optimierungen & Parametertuning (Teilweise implementiert)
**Ziel**: Genauigkeit und Effizienz von MicroPizzaNet steigern

### Abgeschlossene Optimierungen ✅

#### Invertierte Restblöcke ✅
- MobileNetV2-Style Inverted Residual Blocks implementiert
- Shortcut-Verbindungen integriert
- In `MicroPizzaNetV2` implementiert und evaluiert

**Relevante Dateien**:
- `scripts/compare_inverted_residual.py`
- `inverted_residual_comparison.log`

#### Kanal-Aufmerksamkeit (Squeeze-and-Excitation) ✅
- SE-Module mit verschiedenen Ratio-Werten (4 und 8) implementiert
- Evaluation mit verschiedenen Konfigurationen
- Vollständige Berichte in CSV, Excel und HTML

**Relevante Dateien**:
- `scripts/compare_se_models.py`
- `se_comparison.log`
- `output/se_comparison/`

#### Hard-Swish-Aktivierung ✅
- ReLU-Ersatz durch Hard-Swish in kritischen Layers
- Messung der Auswirkungen auf Genauigkeit & Inferenzzeit
- Vollständige Vergleichsberichte

**Relevante Dateien**:
- `scripts/compare_hard_swish.py`
- `hard_swish_comparison.log`
- `output/hard_swish_comparison/`

#### Knowledge Distillation ✅
- Student-Modell mit Teacher-Netzwerk (MicroPizzaNetV2) trainiert
- Vollständige Ergebnisse und Vergleichsberichte

**Relevante Dateien**:
- `scripts/knowledge_distillation.py`
- `knowledge_distillation.log`
- `output/knowledge_distillation/`

#### Quantisierungs-bewusstes Training ✅
- QAT-Training mit Round-to-Nearest-Even-Simulation
- Automatischer Vergleich zwischen Standard-, QAT- und quantisiertem Modell

**Relevante Dateien**:
- `scripts/quantization_aware_training.py`
- `quantization_aware_training.log`

#### Neural Architecture Search (NAS)-Basis ✅
- Grid-Search über Depth-Multiplier und Kanalbreiten
- Automatische Suche nach optimaler Konfiguration

**Relevante Dateien**:
- `scripts/hyperparameter_search.py`
- `hyperparameter_search.log`

## Datenaufbereitung und Augmentation

### DATEN-3.1: Standard-Augmentierungs-Pipeline ✅
**Ziel**: Eine standard Augmentierungs-Pipeline für das Training des Pizza-Erkennungsmodells definieren und implementieren.

**Implementierung**:
- Standard-Augmentierungspipeline mit konfigurierbaren Parametern und Wahrscheinlichkeiten implementiert
- Verschiedene Augmentierungstechniken integriert (Geometrisch, Farbanpassungen, Rauschen, Pizza-spezifisch)
- Drei Intensitätsstufen (niedrig, mittel, hoch) für verschiedene Trainingszenarien
- Integration in den Trainingsprozess über die PyTorch DataLoader-Schnittstelle
- Ausführliche Dokumentation der Parameter und Verwendung

**Relevante Dateien**:
- `scripts/standard_augmentation.py` - Hauptimplementierung der Augmentierungspipeline
- `scripts/train_with_augmentation.py` - Beispiel für Integration ins Training
- `docs/standard_augmentation_pipeline.md` - Detaillierte Dokumentation
- `docs/completed_tasks/DATEN-3.1.md` - Spezifische Task-Dokumentation

Weitere Details: [DATEN-3.1 Dokumentation](/docs/completed_tasks/DATEN-3.1.md)

### Verbleibende Optimierungen ⏳
- Gewichts-Pruning & Clustering (in Arbeit, pruning_clustering.log vorhanden)
- Dynamische Inferenz (Early Exit) (in Arbeit, early_exit_evaluation.log vorhanden)

## 7. Hardware-Optimierungen ✅
### MODELL-3.1: Integration von CMSIS-NN ✅
**Ziel**: Integration der CMSIS-NN-Bibliothek für beschleunigte neuronale Netzwerkoperationen

**Implementierung**:
- CMSIS-NN Funktionen für kritische Operationen integriert (Convolution, Pooling, etc.)
- Performance-Verbesserung um Faktor 2.17x verifiziert
- RAM-Reduktion um 6.3 KB (10.8%) bei leichter Erhöhung des Flash-Verbrauchs

**Hauptdateien**:
- [models/rp2040_export/pizza_model_cmsis.c](/models/rp2040_export/pizza_model_cmsis.c)
- [scripts/verify_cmsis_nn.py](/scripts/verify_cmsis_nn.py)
- [output/performance/cmsis_nn_impact.json](/output/performance/cmsis_nn_impact.json)
- [docs/cmsis_nn_verification.md](/docs/cmsis_nn_verification.md)

**Ergebnis**: Die CMSIS-NN Integration übertrifft die geforderte Performance-Steigerung von 1.5x deutlich und erreicht 2.17x bei gleichzeitiger RAM-Reduktion.

## DIFFUSION-3.1: Dataset Balancing Strategy Using Generated Images ✅
**Ziel**: Definiere eine Strategie für Datensatz-Ausgleich mittels generierter Bilder

**Implementierung**:
- Umfassende Analyse des aktuellen Datensatz-Ungleichgewichts (56 Bilder, 2 Hauptklassen)
- Strategische Planung für ausgewogene Verteilung (300 Bilder, 6 ausgeglichene Klassen)
- Detaillierte Prompt-Spezifikationen für alle 6 Pizza-Klassen
- Automatisierte Ausführungs-Pipeline mit vollständiger Integration
- Qualitätssicherung und Eigenschaftsverifikation für alle generierten Bilder
- Konfigurierbare Parameter für Beleuchtung und Verbrennungsgrade
- Umfassende Risikominderung und Validierungsprotokoll

**Strategische Ziele**:
- **Zielverteilung**: 50 Bilder pro Klasse (minimum 40 für 80% Balance)
- **Zu generierende Bilder**: 244 zusätzliche synthetische Bilder
- **Beleuchtungsverteilung**: 25% pro Bedingung (overhead_harsh, side_dramatic, dim_ambient, backlit_rim)
- **Verbrennungsgrad-Verteilung**: 40% leicht, 35% mäßig, 25% stark verbrannt
- **Qualitätsstandards**: Professionelle Food-Fotografie mit Eigenschaftsverifikation ≥0.6

**Relevante Dateien**:
- `DIFFUSION_3_1_STRATEGY.md` (Vollständige Strategiedokumentation)
- `diffusion_balance_config.json` (Maschinenlesbare Konfiguration)
- `balance_with_diffusion.py` (Automatisierte Ausführungsscript)
- `DIFFUSION_3_1_COMPLETE.md` (Abschlussbericht)

**Implementierungsplan**:
1. **Phase 1 - Kritische Klassen** (192 Bilder): combined, mixed, progression, segment
2. **Phase 2 - Hohe Priorität** (24 Bilder): burnt mit Verbrennungsgrad-Variationen
3. **Phase 3 - Mittlere Priorität** (20 Bilder): basic Klassen-Vervollständigung

**Technische Highlights**:
- 32 automatisierte Generierungsaufgaben mit intelligenter Prioritätszuweisung
- Vollständige Integration mit DIFFUSION-2.1 Pipeline
- Trockenlauffunktionalität für Planungsvalidierung
- Umfassende Metadaten-Verfolgung und Berichterstattung
- GPU-Speicher-Optimierung und Batch-Verarbeitung

**Validierung**:
- Automatisches Script erfolgreich getestet (Trockenlauf: 32 Aufgaben, 232 Bilder)
- Konfigurationsparameter validiert und Pipeline-Integration bestätigt
- Qualitätssicherungsmaßnahmen definiert und implementiert
- Ausführungsbereit für sofortige Implementierung

## Projektorganisation (Abgeschlossen am 2025-05-26)
**TASK:** Projektorganisation und Aufräumarbeiten
**STATUS:** ✅ FERTIG
**DATUM:** 2025-05-26

### Implementierte Maßnahmen:
1. ✅ Projektstruktur gemäß README.md reorganisiert
2. ✅ Skripte in separate Verzeichnisse sortiert
3. ✅ Logdateien in output/logs verschoben
4. ✅ Dokumentationsdateien in docs/ organisiert
5. ✅ Konfigurationsdateien in config/ zentralisiert
6. ✅ Symlinks für wichtige Statusdateien im Hauptverzeichnis erstellt
7. ✅ Statusdateien in docs/status/ organisiert

### Ergebnisse:
- Aufgeräumte Projektstruktur mit klarer Ordnerhierarchie
- Verbesserte Auffindbarkeit von Dateien
- Konsistente Organisation gemäß Projektdokumentation
- Zentralisierte Dokumentation und Status-Tracking
(Ursprünglich dokumentiert in PROJECT_STATUS.txt)

## HWEMU-2.2 Adaptive Clock Frequency Adjustment Logic (Abgeschlossen am 2025-05-25)
**TASK:** Adaptive Clock Frequency Adjustment Logic
**STATUS:** ✅ FERTIG - Alle 7 Tests bestanden
**DATUM:** 2025-05-25

### Implementierte Features:
1. ✅ Temperaturschwellen-basierte Taktfrequenz-Anpassung:
   - 25°C → 133 MHz (max performance)
   - 45°C → 100 MHz (balanced mode)
   - 65°C → 75 MHz (conservative mode)
   - 80°C → 48 MHz (emergency mode)
2. ✅ Emergency Thermal Protection bei kritischen Temperaturen (≥75°C)
3. ✅ Hysterese-Verhalten zur Vermeidung von Oszillation
4. ✅ Systemregister-Abfragen zur Verifikation der Frequenzänderungen
5. ✅ Enable/Disable-Funktionalität für adaptives Clock Management
6. ✅ Temperatur-Spike-Injektion für erweiterte Testzwecke
7. ✅ Umfassende Logging- und Monitoring-Funktionen

### Gefixte Bugs (HWEMU-2.2):
- **Bug #1:** Incorrekte Temperatur-zu-Frequenz-Mapping in `_determine_target_frequency()`
  - Fixed: Line 984 - Korrekte Rückgabe von `max` frequency für niedrige Temperaturen
  - Fixed: Line 978 - Korrekte Aktivierung des thermal protection für hohe Temperaturen
- **Bug #2:** Test-Infrastruktur trigger fehlende adaptive clock updates
  - Fixed: Neue `set_temperature_for_testing()` Methoden hinzugefügt
- **Bug #3:** Temperatur-Spike-Injektion funktionierte nicht korrekt
  - Fixed: Erweiterte Temperaturlimits und direkte current_temperature Updates
(Ursprünglich dokumentiert in PROJECT_STATUS.txt)

## Test-Status: Behobene Probleme (Stand 2025-05-26)
Folgende Probleme wurden im Rahmen der Tests behoben (Details ursprünglich in PROJECT_STATUS.txt):
- `_replace()`-Methode zur `ResourceUsage`-Dataclass hinzugefügt
- Korrektur der Speicherberechnungen in der `MemoryEstimator`-Klasse
- Validierung für negative Speicherallokationen implementiert
- `ModelMetrics`-Klasse und Konfusionsmatrix-Format korrigiert (NumPy-Array)
- `enter_sleep_mode()` und `wake_up()` in der `RP2040Emulator`-Klasse implementiert
- Ressourcenprüfungen im Emulator für Flash und RAM verbessert
- Fehler im Sleep-Mode-Management beseitigt (RAM-Wiederherstellung)
- Visualisierungsfunktionen für neues Konfusionsmatrix-Format angepasst

## Speicheroptimierungs-Aufgaben (Abgeschlossen vor 2025-05-29)
(Ursprünglich dokumentiert in aufgaben.txt)

### [x] SPEICHER-1.1: Framebuffer-Simulationsgenauigkeit verifizieren
**Beschreibung:** Überprüfe und korrigiere bei Bedarf die Simulation des Kamera-Framebuffer-RAM-Bedarfs im RP2040-Emulator (EMU-01). Stelle sicher, dass die simulierte Nutzung exakt mit der erwarteten Nutzung auf Hardware übereinstimmt.
**Aufgabe für Agent:**
Referenziere Code/Simulation für Framebuffer-RAM-Berechnung.
Wenn Hardware-Messungen verfügbar sind (siehe Checklist "RP2040-Hardware-Integration"), vergleiche den simulierten Wert mit dem gemessenen Wert.
Dokumentiere den Unterschied.
Wenn die Abweichung > 5% ist, analysiere die Ursache in der Simulation und korrigiere sie.
**Kriterium "Fertig":** Die Abweichung zwischen simuliertem und (falls verfügbar) gemessenem Framebuffer-RAM-Bedarf ist kleiner oder gleich 5%. Die Logik der Berechnung in der Simulation ist nachvollziehbar und dokumentiert.

### [x] SPEICHER-1.2: Tensor-Arena-Schätzgenauigkeit verifizieren
**Beschreibung:** Überprüfe und korrigiere bei Bedarf die Schätzung der Tensor-Arena-Größe für das quantisierte Modell im RP2040-Emulator (EMU-02). Stelle sicher, dass die Schätzung den tatsächlich benötigten RAM für die Modellinferenz korrekt widerspiegelt.
**Aufgabe für Agent:**
Führe das Skript scripts/test_pizza_classification.py oder ein spezifisches Emulator-Skript aus, das die Modellinferenz mit dem quantisierten Modell lädt und durchführt.
Extrahiere die gemeldete oder gemessene Tensor-Arena-Größe aus dem Emulator-Log/Output.
Vergleiche diesen Wert mit dem in der Speicherbedarfsschätzung verwendeten Wert.
Dokumentiere den Unterschied.
Wenn die Abweichung > 5% ist, analysiere die Ursache in der Schätzung und korrigiere sie.
**Kriterium "Fertig":** Die Abweichung zwischen geschätzter und im Emulator gemessener Tensor-Arena-Sch Größe ist kleiner oder gleich 5%. Die Schätzlogik ist nachvollziehbar und dokumentiert.

### [x] SPEICHER-1.3: Detaillierte RAM-Nutzungsanalyse durchführen
**Beschreibung:** Ermittle den detaillierten RAM-Verbrauch aller relevanten Komponenten (Tensor Arena, Framebuffer, Zwischenpuffer für Vorverarbeitung/Ausgabe, Stack, Heap, globale Variablen, statische Puffer etc.) für den kritischsten Betriebsfall (z.B. eine einzelne Inferenz inklusive Vorverarbeitung).
**Aufgabe für Agent:**
Nutze das Performance-Logging des Emulators oder spezifische Debugging-Features, um RAM-Profile während eines Inferenzlaufs zu erstellen.
Analysiere die Log-Dateien oder den Output, um die Nutzung der verschiedenen RAM-Bereiche zu identifizieren.
Aggregiere die Ergebnisse und berechne die Gesamt-RAM-Nutzung.
Generiere einen Bericht (Text/JSON), der die Aufteilung der RAM-Nutzung zeigt und den Gesamtwert angibt.
**Kriterium "Fertig":** Ein strukturierter Bericht (output/ram_analysis/ram_usage_report.json oder ähnlich) existiert, der den RAM-Verbrauch der Hauptkomponenten auflistet und die Gesamt-RAM-Nutzung für einen Inferenzzyklus dokumentiert.

### [x] SPEICHER-2.1: Strukturbasiertes Pruning implementieren/anwenden
**Beschreibung:** Implementiere oder integriere Tools für strukturbasiertes Pruning und wende es auf das MicroPizzaNetV2-Modell an, um nicht benötigte Kanäle oder Filter zu entfernen.
**Aufgabe für Agent:**
Referenziere das Skript oder Tool für strukturbasiertes Pruning (z.B. scripts/pruning_tool.py).
Führe das Pruning-Skript mit vordefinierten Parametern (z.B. Ziel-Sparsity) aus.
Das Skript sollte ein neues, gepruntes Modell erzeugen.
**Kriterium "Fertig":** Das Skript wurde erfolgreich ausgeführt. Ein neues Modell-Datei (models/pruned_model.tflite oder ähnlich) wurde erstellt. Der Log des Skripts bestätigt den angewendeten Pruning-Grad.

### [x] SPEICHER-2.2: Genauigkeit des geprunten Modells evaluieren
**Beschreibung:** Evaluiere die Klassifikationsgenauigkeit des geprunten Modells auf dem Test-Datensatz.
**Aufgabe für Agent:**
Führe das Standard-Evaluierungsskript (scripts/run_pizza_tests.py oder ähnlich) für das geprunte Modell aus.
Sammle die Genauigkeitsmetriken (Accuracy, F1-Score etc.) aus dem Testbericht.
**Kriterium "Fertig":** Das Test-Skript wurde erfolgreich ausgeführt. Ein Evaluierungsbericht für das geprunte Modell (output/evaluation/pruned_model_evaluation.json) existiert und enthält die Genauigkeitsmetriken.

### [x] SPEICHER-2.3: RAM-Bedarf des geprunten Modells evaluieren
**Beschreibung:** Ermittle den RAM-Bedarf (Tensor Arena) des geprunten und quantisierten Modells.
**Aufgabe für Agent:**
Quantisiere das geprunte Modell (falls noch nicht geschehen).
Führe das Skript zur Tensor-Arena-Schätzung (siehe SPEICHER-1.2) für das geprunte und quantisierte Modell aus.
Extrahiere den geschätzten/gemessenen RAM-Bedarf.
**Kriterium "Fertig":** Der RAM-Bedarf des geprunten und quantisierten Modells ist dokumentiert.