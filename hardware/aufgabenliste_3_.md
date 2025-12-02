Phase 1 – Projektstatus ermitteln

EDA-Daten inventarisieren

Suche alle relevanten Dateien im Projektverzeichnis:

.kicad_pcb, .kicad_sch, .kicad_pro, Symbol-/Footprint-Libs.

Ergebnis: JSON mit Dateipfaden, Größen, Änderungsdatum.

Versionen & Abhängigkeiten prüfen

KiCad-Version ermitteln.

Python-Version und Libraries prüfen (z. B. pcbnew, Gerber-Parser).

Letzten Fertigungsstatus auslesen

Prüfen, ob in manufacturing/JLCPCB_UPLOAD/ ein ZIP existiert.

Wenn ja: Dateien extrahieren und Struktur (Gerber/BOM/CPL) analysieren.

Phase 2 – Analyse & Validierung

DRC/ERC ausführen

Automatisiert DRC/ERC-Reports erzeugen.

Ergebnisse als maschinenlesbares JSON speichern.

BOM/CPL-Check

BOM auf LCSC-Felder, Werte, Footprints prüfen.

CPL auf Rotationen und Pin-1-Markierung prüfen.

Gerber-Dateien validieren

Layernamen, Bohrdaten, Boardgröße, Masken auf JLCPCB-Konformität prüfen.

Phase 3 – Artefakt-Erzeugung

Fertigungsbundle erstellen

Gerber, BOM, CPL in definierten Pfad exportieren.

ZIP-Datei gemäß JLCPCB-Spezifikation erzeugen.

Upload-Vorprüfung

ZIP-Struktur noch einmal validieren (Layer-Mapping, Dateinamen).

Prüfbericht speichern.

Phase 4 – Dokumentation

Projektstatus zusammenfassen

JSON-Report über alle Prüfschritte erzeugen:

Dateien, Fehler, Warnungen, finale ZIP-Pfade.

Änderungen und offene Punkte in Markdown-Changelog festhalten.