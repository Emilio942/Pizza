# Manuelle Qualitätskontrolle für augmentierte/generierte Bilder

Dieses Dokument beschreibt den Prozess zur manuellen Überprüfung und Qualitätssicherung der augmentierten und KI-generierten Bilder im Pizza-Erkennungsprojekt.

## Übersicht

Der Qualitätskontrollprozess besteht aus drei Hauptschritten:

1. **Stichprobenauswahl**: Zufällige Auswahl von Bildern aus dem Datensatz
2. **Visuelle Inspektion**: Bewertung der Bildqualität durch manuelle Überprüfung
3. **Bereinigung**: Entfernung von Bildern niedriger Qualität aus dem Datensatz

## Voraussetzungen

- Python 3.6 oder höher
- Pillow (PIL) Bibliothek
- Webbrowser

## Verwendung des Qualitätskontroll-Tools

### 1. Starten des Tools

Führen Sie das bereitgestellte Shell-Skript aus:

```bash
./run_quality_control.sh
```

Alternativ können Sie eines der Python-Skripte direkt mit spezifischen Parametern aufrufen:

```bash
python scripts/image_quality_check.py --samples 50 --output output/quality_control
```

oder

```bash
python scripts/image_quality_control.py --samples 50 --output output/quality_control
```

Optionen:
- `--samples`: Anzahl der zufälligen Stichproben pro Klasse (Standard: 50)
- `--image-dirs`: Verzeichnisse mit den zu prüfenden Bildern (Standard: augmented_pizza)
- `--output`: Ausgabeverzeichnis für den HTML-Report (Standard: output/quality_control)
- `--check-existing`: Überprüft vorhandene Kontrolldaten, zeigt nur unverarbeitete Bilder an

### 2. Visuelle Inspektion durchführen

Nach dem Start des Tools wird ein HTML-Report in Ihrem Standardbrowser geöffnet. Dieser zeigt zufällig ausgewählte Bilder aus dem Datensatz an, organisiert nach Klassen.

#### Kriterien für die Bildqualitätsbewertung:

Bei der Inspektion sollten Sie folgende Kriterien beachten:

- **Plausibilität**: Stellt das Bild eine erkennbare Pizza dar?
- **Artefakte**: Gibt es künstliche Artefakte, Verzerrungen oder Fehler?
- **Erkennbarkeit**: Ist der Pizza-Zustand (basic, burnt, etc.) klar erkennbar?
- **Vielfalt**: Stellt das Bild einen nützlichen Anwendungsfall dar, der im Datensatz noch nicht ausreichend repräsentiert ist?

#### Markieren von Bildern zur Löschung:

1. Klicken Sie auf ein Bild, um es für die Löschung zu markieren (es wird mit einem roten Rahmen und einem "Löschen"-Badge versehen)
2. Klicken Sie erneut, um die Markierung aufzuheben
3. Verwenden Sie die Filteroptionen, um nur bestimmte Klassen oder nur markierte Bilder anzuzeigen
4. Nutzen Sie die "Alle sichtbaren Bilder markieren" Funktion für schnellere Bearbeitung

### 3. Speichern der Markierungen und Löschen der Bilder

1. Klicken Sie auf "Markierte Bilder speichern", um Ihre Auswahl zu speichern
2. Eine Bestätigungsseite wird angezeigt
3. Führen Sie den angegebenen Befehl aus, um die markierten Bilder zu löschen:

```bash
python scripts/image_quality_check.py --process-marked --output output/quality_control
```

oder

```bash
python scripts/image_quality_control.py --process-marked --output output/quality_control
```

Ein Löschbericht wird im Ausgabeverzeichnis gespeichert, der alle gelöschten Bilder dokumentiert.

## Bewertungskriterien für KI-generierte Bilder

Bei KI-generierten Bildern (falls DATEN-3.3 durchgeführt wurde) sollten zusätzlich folgende Aspekte bewertet werden:

- **Anatomische Korrektheit**: Weist die Pizza unrealistische oder verzerrte Formen auf?
- **Texturkonsistenz**: Sind Textur und Oberflächenbeschaffenheit realistisch?
- **Belag-Details**: Sind die Beläge natürlich platziert und realistisch dargestellt?
- **Schatten/Beleuchtung**: Sind Schatten und Beleuchtung physikalisch plausibel?
- **Farbgebung**: Erscheinen die Farben natürlich und konsistent?

## Beispiele für zu löschende Bilder:

- Bilder mit offensichtlichen Artefakten oder Verzerrungen
- Extrem unscharfe oder übersteuerte Bilder
- Stark verrauschte oder korrupte Bilder 
- Bei generierten Bildern: Bilder mit anatomisch unmöglichen Merkmalen
- Bei generierten Bildern: Bilder mit Text oder unrealistischen Objekten

## Abschluss des Prozesses

Nach Abschluss des Qualitätskontrollprozesses sollte ein gereinigter Datensatz vorliegen, der nur Bilder mit ausreichender Qualität enthält. Die gelöschten Bilder sind im Löschbericht dokumentiert.
