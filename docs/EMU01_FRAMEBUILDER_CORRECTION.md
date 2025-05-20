# EMU-01 Framebuilder Korrektur

**Datum:** 13. Mai 2025
**Autor:** AI-Assistent
**Status:** Implementiert

## Zusammenfassung

Die EMU-01 Framebuilder Korrektur implementiert eine präzise Simulation des Kamera-Framebuffers im RAM für das RP2040 Pizza-Erkennungssystem. Dieser Fix behebt eine kritische Unterschätzung des tatsächlichen RAM-Bedarfs in der vorherigen Implementierung des Emulators.

## Problem

Der ursprüngliche Emulator berücksichtigte in den Speicherberechnungen nur den RAM-Bedarf des Modells selbst (Tensor Arena) und einen festen System-Overhead. Der für den Kamera-Framebuffer benötigte Speicher wurde jedoch nicht in die Berechnung einbezogen, was zu einer signifikanten Unterschätzung des tatsächlichen RAM-Bedarfs führte:

- Ein 320x240 Graustufen-Bild (1 Byte pro Pixel) benötigt ~75 KB RAM
- Ein 320x240 RGB565-Bild (2 Bytes pro Pixel) benötigt ~150 KB RAM
- Ein 320x240 RGB888-Bild (3 Bytes pro Pixel) benötigt ~225 KB RAM

Bei einem RP2040 mit 264 KB RAM konnte dies zu Speicherüberlauf führen, wenn die Emulation fälschlicherweise anzeigte, dass genügend Speicher vorhanden sei.

### Fallbeispiel

**Vor der Korrektur:**
- Modell-RAM: 90 KB
- System-Overhead: 40 KB
- Gesamt-RAM-Bedarf: 130 KB (< 264 KB, erscheint sicher)

**Nach der Korrektur:**
- Modell-RAM: 90 KB
- System-Overhead: 40 KB
- Kamera-Framebuffer (RGB565): 150 KB
- Gesamt-RAM-Bedarf: 280 KB (> 264 KB, tatsächlich nicht möglich!)

## Lösung

Die Lösung besteht aus zwei Hauptkomponenten:

1. **Präziser Framebuffer-Simulator**:
   - Implementiert eine korrekte Speicherrepräsentation des Framebuffers
   - Unterstützt alle relevanten Pixelformate (RGB888, RGB565, Grayscale, YUV422)
   - Berücksichtigt Speicherausrichtung und Padding für optimale ARM-Prozessorleistung
   - Implementiert Schutzmechanismen gegen Speicherüberlauf

2. **Integration in den Emulator**:
   - Berücksichtigt Framebuffer-Größe bei RAM-Berechnungen
   - Aktualisiert RAM-Berechnungen bei Änderungen des Kameraformats
   - Erweitert die Ressourcenvalidierung und Speicherstatistiken
   - Bietet detaillierte Logging- und Debugging-Funktionen

## Implementierungsdetails

### FrameBuffer Klasse

Die `FrameBuffer`-Klasse in `src/emulation/frame_buffer.py` simuliert den Kamera-Framebuffer mit:

- Speicherausrichtung auf 4-Byte-Grenzen für optimale ARM-Prozessorleistung
- Unterstützung für verschiedene Pixelformate
- Synchronisationsmechanismen für Schreibvorgänge
- Umfangreiche Statistik- und Debugging-Funktionen
- Speicherüberlaufschutz

### Änderungen am RP2040Emulator

- Berücksichtigt Framebuffer-Größe in der RAM-Nutzungsberechnung
- Aktualisiert die Speichervalidierung beim Laden von Firmware
- Bietet Methoden zum Ändern des Kameraformats mit entsprechender Aktualisierung der Speicherberechnungen
- Liefert detaillierte Speicherstatistiken inklusive Framebuffer-Informationen

## Technische Spezifikationen

### Pixelformate

- **RGB888**: 3 Bytes pro Pixel (24-bit), höchste Qualität
- **RGB565**: 2 Bytes pro Pixel (16-bit), guter Kompromiss zwischen Qualität und Speicherbedarf
- **GRAYSCALE**: 1 Byte pro Pixel (8-bit), minimaler Speicherbedarf
- **YUV422**: 2 Bytes pro Pixel (16-bit), gute Farbwiedergabe bei reduziertem Speicherbedarf

### Speicherauswirkungen

| Format    | Auflösung | Bytes/Pixel | Unausgerichtete Größe | Ausgerichtete Größe | % des RP2040 RAM |
|-----------|-----------|-------------|----------------------|---------------------|------------------|
| RGB888    | 320x240   | 3           | ~225 KB              | ~230 KB             | ~87%             |
| RGB565    | 320x240   | 2           | ~150 KB              | ~150 KB             | ~57%             |
| Grayscale | 320x240   | 1           | ~75 KB               | ~76 KB              | ~29%             |
| YUV422    | 320x240   | 2           | ~150 KB              | ~150 KB             | ~57%             |

### Ausrichtung und Padding

Die Frame-Buffer-Implementierung richtet jede Zeile auf 4-Byte-Grenzen aus, wie es für ARM-Prozessoren wie den RP2040 typisch ist. Dies führt zu einem leichten Overhead durch Padding bei Bildbreiten, die nicht durch die Byte-pro-Pixel-Anzahl teilbar sind.

## Validierung

Die Implementierung wurde mit umfangreichen Tests validiert:

1. `test_frame_buffer.py`: Testet die grundlegende Funktion der Framebuffer-Klasse
2. `test_framebuffer_ram.py`: Testet die Integration des Framebuffers in den Emulator

## Verwendung und Dokumentation

### Demo-Skript

Das Skript `scripts/framebuffer_demo.py` demonstriert die Auswirkungen verschiedener Framebuffer-Konfigurationen auf den RAM-Verbrauch:

```sh
# Alle Demos ausführen
python scripts/framebuffer_demo.py --all

# Nur Pixelformat-Demo ausführen
python scripts/framebuffer_demo.py --format

# Nur Auflösungs-Demo ausführen
python scripts/framebuffer_demo.py --resolution

# Nur Firmware-Größen-Demo ausführen
python scripts/framebuffer_demo.py --firmware

# Nur Vergleich vor/nach der Korrektur
python scripts/framebuffer_demo.py --compare
```

## Empfehlungen

1. **Verwende Grayscale oder RGB565 statt RGB888**, wenn möglich, um den Speicherbedarf zu reduzieren
2. **Reduziere die Kameraauflösung auf das notwendige Minimum** für die Anwendung
3. **Berücksichtige immer den Framebuffer-Speicherbedarf** bei der Planung von RP2040-Anwendungen
4. **Teste mit realistischen Datengrößen** um Speicherprobleme frühzeitig zu erkennen

## Zusammenfassung

Die EMU-01 Framebuilder Korrektur sorgt für eine präzise Simulation des Kamera-Framebuffers im RAM und verbessert dadurch signifikant die Genauigkeit der Ressourcenschätzungen für das RP2040 Pizza-Erkennungssystem. Diese Korrektur verhindert potentielle Speicherüberläufe und Systemabstürze auf der tatsächlichen Hardware.
