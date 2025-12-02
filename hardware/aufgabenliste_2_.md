Aufgabenliste – „DFM-Check & Fertigungsfreigabe“
Phase 1 – Analyse des aktuellen Status

EDA-Projekt öffnen

 Sicherstellen, dass alle KiCad-Dateien vorliegen (.kicad_pcb, .kicad_sch, Libraries).

 Prüfen, ob Version von KiCad ≥ 7.x.

ERC & DRC ausführen

 ERC-Report erzeugen, Fehler/Warnungen dokumentieren.

 DRC-Report erzeugen, Fehler/Warnungen dokumentieren.

 Offene Punkte bewerten: Welche müssen zwingend behoben werden?

BOM- und CPL-Dateien prüfen

 Fehlen Felder wie LCSC Part Number, Value, Footprint, Rotation?

 Vergleich gegen JLCPCB-Anforderungen.

Phase 2 – Export & Artefakt-Erzeugung

Automatisierten Export testen

 pcb_export.py ausführen → erzeugte Dateien überprüfen.

 Falls Fehler → pcb_export_corrected.py oder manuelle Anpassung.

Gerber-Viewer-Check

 Prüfen: Layerbelegung, Bohrungen, Fiducials, Ausrichtung.

JLCPCB-Simulation (Smoke-Test)

 Gerber/BOM/CPL in JLCPCB-Webinterface hochladen.

 Fehlerbericht herunterladen (falls vorhanden).

 Screenshots machen und in docs/ speichern.

Phase 3 – Korrektur & Finalisierung

Fehlerliste analysieren und beheben

 Alle DRC/ERC-Fehler beseitigen.

 BOM/CPL fehlende Felder ergänzen oder korrigieren.

 Thermische Simulation & Anpassungen durchführen (optional).

Finale Export-Iteration

 Finales Upload-Bundle erzeugen.

 Letzten Upload-Test durchführen.

Definition of Done prüfen

 JLCPCB akzeptiert Upload ohne Fehler.

 Bauteilorientierung und Rotation geprüft.

 Änderungen dokumentiert (docs/hardware_changelog.md).