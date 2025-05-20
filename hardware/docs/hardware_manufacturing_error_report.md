# Fehlerbericht: Hardware-Fertigungslogik (pcb_export.py)

Datum: 2025-05-09

## 1. Einleitung

Dieser Bericht dokumentiert konzeptionelle Fehler und spezifische Probleme, die in der aktuellen Hardware-Fertigungslogik, insbesondere im Skript `hardware/manufacturing/pcb_export.py`, identifiziert wurden. Die Analyse konzentriert sich auf die Generierung von Fertigungsdaten für Leiterplatten (PCBs) basierend auf einer SVG-Datei.

## 2. Wichtige konzeptionelle Schwachstellen

### 2.1. SVG als primäre PCB-Designquelle

Die Verwendung einer SVG-Datei als alleinige Grundlage für die PCB-Fertigungsdaten ist ein fundamentaler konzeptioneller Fehler. SVGs sind Grafikformate und entbehren der notwendigen detaillierten elektrischen, geometrischen und fertigungstechnischen Informationen, die für eine professionelle PCB-Herstellung erforderlich sind. Dazu gehören:

*   Netzlisten (Verbindungen zwischen Bauteilen)
*   Exakter Lagenaufbau (Anzahl und Art der Kupferschichten, dielektrische Materialien)
*   Präzise Padstacks (Definition von Pads, Vias, Bohrungen für jedes Bauteil)
*   Genaue Bohrdaten (Position, Durchmesser, Typ – durchkontaktiert/nicht durchkontaktiert)
*   Korrekte Footprints (standardisierte Landeflächen für Bauteile)

Der Versuch, diese kritischen Informationen aus einer SVG-Grafik zu rekonstruieren, ist inhärent unzuverlässig und führt zu fehlerhaften Fertigungsdaten.

### 2.2. Stark vereinfachte Fallback-Generierung (ohne KiCad)

Wenn die KiCad-Bibliotheken nicht verfügbar sind, greift das Skript auf eine stark vereinfachte Generierung von Gerber- und Bohrdateien zurück:

*   **Gerber-Dateien:** Die im Fallback-Modus generierten Gerber-Dateien sind extrem rudimentär. Sie scheinen lediglich einen Platinenumriss und einfache rechteckige "Pads" (durch das Flashen einer 1x1mm Apertur D11 an den Komponentenpositionen) zu erzeugen. Dies ist für die Fertigung einer komplexen 8-Lagen-Platine völlig unzureichend. Es fehlen:
    *   Leiterbahnen
    *   Vias (Durchkontaktierungen zwischen Layern)
    *   Korrekte Pad-Formen und -größen für die spezifischen Bauteile
    *   Kupferflächen (z.B. für Masseflächen)
    *   Lötstoppmasken- und Pastenmasken-Öffnungen, die präzise auf die Pads abgestimmt sind.
*   **Bohrdatei (`.TXT`):** Die generierte Bohrdatei enthält nur sehr wenige Bohrungen (scheinbar nur Montagelöcher und einige wenige für "RP2040" oder "FLASH"). Eine typische Platine benötigt jedoch eine Vielzahl von Vias und bauteilspezifischen Bohrungen für bedrahtete Bauteile oder Befestigungen.

### 2.3. Fehleranfällige Komponentenextraktion aus SVG

Die Methode zur Identifizierung von Komponenten basiert auf der Suche nach Text in der Nähe von Rechtecken innerhalb der SVG-Datei. Dieser Ansatz ist nicht robust:

*   Die Positionierung von Text relativ zu grafischen Elementen in einer SVG ist nicht standardisiert genug für eine präzise Bauteilplatzierung.
*   Die exakte Erfassung von Bauteilnamen und insbesondere deren Drehung ist auf diese Weise kaum zuverlässig möglich.

### 2.4. Fehlende oder falsche Bauteildrehung

Die Drehung von Bauteilen wird beim Parsen der SVG standardmäßig auf 0 Grad gesetzt und scheint im weiteren Verlauf nicht korrekt aktualisiert zu werden. Dieser Wert wird dann direkt in die CPL-Datei (Centroid Placement List) übernommen. Die meisten SMD-Bauteile auf einer Leiterplatte haben jedoch spezifische, von 0 Grad abweichende Drehungen, die für die automatische Bestückung kritisch sind.

### 2.5. Kritischer Fehler in der KiCad-Footprint-Erstellung (wenn KiCad verfügbar ist)

Selbst wenn die KiCad-Python-Module (`pcbnew`) verfügbar sind und das Skript versucht, eine `.kicad_pcb`-Datei zu erstellen, werden den Footprints **keine Pads** hinzugefügt. Die Funktionen `_create_qfn_footprint`, `_create_soic_footprint` und `_create_smd_footprint` sind im Wesentlichen leer und führen keine Operationen aus, die dem `pcbnew.FOOTPRINT` Objekt Pads hinzufügen würden.

*   **Konsequenz:** Die mit `pcbnew` erzeugte `.kicad_pcb`-Datei würde zwar Footprints an den aus der SVG abgeleiteten Positionen enthalten, diese Footprints hätten jedoch keine Kupfer-Pads.
*   **Folgefehler:** Gerber-Dateien, die aus einer solchen unvollständigen KiCad-Datei generiert würden, wären ebenfalls unbrauchbar, da die für das Löten der Bauteile notwendigen Kupfer-Pads fehlen würden.
*   Ironischerweise ist die textbasierte Fallback-Vorlage (`_create_kicad_pcb_template`) in diesem Punkt vollständiger, da sie zumindest generische Pads für die platzierten Module definiert.

## 3. Spezifische Probleme in `hardware/manufacturing/pcb_export.py`

### 3.1. Pfad zur SVG-Datei

*   Das Skript erwartet die SVG-Datei unter `PROJECT_ROOT / "pcb-layout-updated.svg"`. Basierend auf der Projektstruktur (`PROJECT_ROOT` ist `/home/emilio/Documents/ai/pizza`) wäre dies `/home/emilio/Documents/ai/pizza/pcb-layout-updated.svg`.
*   Laut der bereitgestellten Verzeichnisstruktur befindet sich die relevante SVG-Datei jedoch unter `docs/pcb-layout-updated.svg`.
*   **Konsequenz:** Das Skript wird die SVG-Datei nicht finden und die Verarbeitung fehlschlagen, es sei denn, die Datei wird manuell an den erwarteten Ort kopiert.

### 3.2. Wiederholte Bohrkoordinaten in der Fallback-Bohrdatei

*   Die Logik zur Erstellung der Bohrdatei im Fallback-Modus (ohne KiCad) kann zu doppelten Koordinaten führen. Dies geschieht, wenn mehrere aus der SVG geparste Komponenten (z.B. "RP2040", "FLASH") zufällig die gleichen (oder nach Rundung sehr ähnliche) Mittelpunktkoordinaten haben.
*   Die Beispieldatei `hardware/manufacturing/gerber/PizzaBoard-RP2040.TXT` zeigt dieses Problem mit mehrfach vorkommenden `X020000Y020000`-Einträgen. Jedes Bohrloch sollte in einer Excellon-Datei üblicherweise nur einmal mit seinen Koordinaten definiert werden.

### 3.3. Unklare Skalierung in der KiCad-PCB-Vorlage (`_create_kicad_pcb_template`)

*   In der Funktion `_create_kicad_pcb_template` werden die X- und Y-Koordinaten der Komponenten durch 4 geteilt (`x = comp["x"] / 4`, `y = comp["y"] / 4`).
*   Der Grund für diese spezifische Skalierung ist aus dem Code nicht ersichtlich und könnte zu Inkonsistenzen führen, wenn die Einheiten und Skalierungsfaktoren nicht durchgängig korrekt gehandhabt werden (SVG-Koordinaten vs. KiCad-interne Einheiten vs. Gerber-Einheiten).

## 4. Empfehlungen

### 4.1. Umstellung auf ein professionelles EDA-Tool

Für ein zuverlässiges PCB-Design und die Generierung korrekter, vollständiger Fertigungsdaten ist die **dringende Empfehlung**, ein etabliertes EDA (Electronic Design Automation)-Programm zu verwenden. Beispiele hierfür sind:

*   KiCad (Open Source, leistungsstark)
*   Autodesk Eagle
*   Altium Designer

Diese Werkzeuge sind speziell für das PCB-Design entwickelt worden und verwalten alle notwendigen Informationen (Schaltplan, Layout, Bauteilbibliotheken, Designregeln) integriert. Der Export von Gerber-, Bohr-, BOM- und CPL-Dateien sollte direkt aus dem EDA-Tool erfolgen.

### 4.2. Überarbeitung der KiCad-Integration (falls beibehalten)

Sollte der Ansatz, eine `.kicad_pcb`-Datei programmatisch zu erstellen, weiterhin verfolgt werden (was nicht empfohlen wird), müssen die folgenden kritischen Punkte adressiert werden:

*   **Pads zu Footprints hinzufügen:** Die Funktionen `_create_qfn_footprint`, `_create_soic_footprint`, `_create_smd_footprint` müssen so implementiert werden, dass sie den `pcbnew.FOOTPRINT`-Objekten tatsächlich Pads (`pcbnew.PAD`) mit korrekten Eigenschaften (Form, Größe, Layer, Bohrung etc.) hinzufügen.
*   **Verwendung von KiCad-Bibliotheks-Footprints:** Anstatt Footprints von Grund auf neu zu erstellen, sollten idealerweise Footprints aus den Standard-KiCad-Bibliotheken oder benutzerdefinierten Bibliotheken geladen und platziert werden. Dies stellt sicher, dass die Footprints industrieüblichen Standards entsprechen.
*   **Korrekte Layer-Zuweisung:** Sicherstellen, dass alle Elemente (Pads, Leiterbahnen, Texte, Grafiken) den korrekten Layern in KiCad zugewiesen werden.

### 4.3. Validierung der generierten Dateien

Unabhängig von der Methode der Generierung müssen **alle** Fertigungsdateien (Gerber, Bohrdaten) vor dem Absenden an einen Hersteller **immer** gründlich mit einem Gerber-Viewer (z.B. der in KiCad integrierte Viewer, `gerbv`, oder Online-Viewer von PCB-Herstellern) und idealerweise mit CAM (Computer-Aided Manufacturing)-Software überprüft werden. Dies hilft, Fehler wie fehlende Layer, falsche Skalierungen, Kurzschlüsse oder offene Verbindungen frühzeitig zu erkennen.

## 5. Fazit

Die aktuelle Methode zur Generierung von PCB-Fertigungsdaten weist schwerwiegende konzeptionelle Mängel auf, die eine erfolgreiche und zuverlässige Fertigung der Leiterplatte unwahrscheinlich machen. Eine grundlegende Überarbeitung des Design- und Exportprozesses unter Verwendung professioneller EDA-Werkzeuge wird dringend empfohlen.
