# Projektstart – Hardware (PizzaBoard‑RP2040)

Kurzfassung: Startleitfaden und Qualitätskriterien für das Hardware-Teilprojekt (EDA, Fertigung, DFM/DFT) mit klaren Meilensteinen und „Schnellstart“ zur Artefakt-Erzeugung.

## Zielbild
- Ein stabil fertiges PizzaBoard‑RP2040 mit sauberem Sim‑to‑Real‑Übergang: thermisch stabil, sicher im Betrieb, mit automatisierter Fertigungs-Pipeline (Gerber/BOM/CPL) für JLCPCB.

## Erfolgskennzahlen (KPIs)
- Fertigung: JLCPCB‑Paket vollständig und importierbar (Gerber, BOM, CPL) ohne manuelle Nacharbeit oder DRC/DFM‑Fehler.
- Elektrik: Versorgung stabil (3V3 Rail, Spannungsabfälle im Rahmen), GPIO‑Mapping konsistent mit Dokumentation.
- Thermik: Temperaturanstieg unter typischer Last < X °C, Sim‑to‑Real Abweichung < Y %.
- Qualität: ERC/DRC clean, Fiducials korrekt, Bestückungsorientierung eindeutig.

## Umfang / Nicht‑Umfang
- In Scope: KiCad‑Projektpflege, DRC/DFM, Fertigungs‑Exports (Gerber/BOM/CPL), Upload‑Smoke‑Tests bei JLCPCB, Hardware‑relevante Stücklistenpflege.
- Out of Scope: Firmware‑Features jenseits Bring‑up, Produktionslogistik, mechanische Integration außerhalb der PCB‑Referenzabmessungen.

## Artefakte & Struktur (Repo‑Bezug)
- EDA: `eda/PizzaBoard-RP2040.kicad_*`
- Fertigung/Exports: `manufacturing/` (Gerber, BOM, CPL, Upload‑Bundle)
- Upload‑Outputs: `manufacturing/JLCPCB_UPLOAD/`
- Begleit‑Docs: `docs/` (z. B. `component_selection_gpio.md`, Fehlerberichte)

## Toolchain & Setup
- KiCad ≥ 7.x (DRC/ERC, Plotten der Gerber)
- Python ≥ 3.12 (für Export‑Automatisierung unter `manufacturing/`)
- Optional: Gerber‑Viewer (z. B. KiCad GerbView) zur visuellen QS

## Schnellstart (Erste Ergebnisse in 5 Minuten)
1) EDA Projekt öffnen: `eda/PizzaBoard-RP2040.kicad_pro`
2) DRC/ERC in KiCad ausführen und Abweichungen dokumentieren.
3) Fertigungsdaten erzeugen (z. B. Skript unter `manufacturing/` verwenden):
   - Gerber, BOM, CPL nach `manufacturing/JLCPCB_UPLOAD/` schreiben lassen.
4) Upload‑Smoke‑Test bei JLCPCB (Web‑UI): Paket importieren, Layerzuordnung, Bohrungen, Bestückungspositionen prüfen.

Hinweis: Im Ordner `manufacturing/` liegen Skripte wie `pcb_export.py` und `pcb_export_corrected.py`, sowie aktuelle Artefakte in `manufacturing/JLCPCB_UPLOAD/`.

## Qualitätssicherung (Quality Gates)
- ERC/DRC: Keine Fehler, nur begründete Warnungen (mit Vermerk).
- Gerber‑Check: Alle Layer korrekt (GTL, GBL, GTS/GBS, GTO/GBO, GKO, Bohrdaten), Polarity/Masken plausibel.
- BOM/CPL: Referenzbezeichner, Werte, Footprints, Rotationen stimmen; Feldnamen kompatibel mit JLCPCB.
- Upload‑Check: JLCPCB‑Import ohne Mapping‑Fehler; visuelle Inspektion von Pads, Ausrichtung, Fiducials.
- Dokumentation: Änderungen mit Kurznotiz in `docs/` festhalten.

## Meilensteine
- M1: ERC/DRC clean, initiales JLCPCB‑Paket erzeugt und importiert.
- M2: BOM/CPL verifiziert (Orientierung/Rotation), Fiducials bestätigt.
- M3: Thermische Annahmen mit Simulation gegengeprüft, Anpassungen in Layout/BoM erledigt.
- M4: Finales Upload‑Bundle freigegeben (Versionstag, Änderungsprotokoll).

## Risiken & Annahmen
- Bauteilverfügbarkeit/Alternativen (JLCPCB LCSC‑Stock); Mitigation: Second‑Source in BOM hinterlegen.
- Rotations-/Ausrichtungsfehler in CPL; Mitigation: Test‑Bestückung/3D‑Vorschau prüfen.
- Thermik unterschätzt; Mitigation: Kupferflächen/VIAs, Messpunkte, ggf. Heatsink‑Option vorsehen.

## Verantwortlichkeiten
- EDA/Layout
- BOM/Bestückung (Library‑Felder, LCSC‑Links)
- Fertigungs‑Exports & Upload‑Prüfung
- Dokumentation & Änderungsmanagement

## Arbeitsweise
- Kleine, nachvollziehbare Änderungen; DRC/ERC vor Commit.
- Export‑Artefakte versionieren (nur finale Bundles), Zwischendateien vermeiden.
- Fehler/Abweichungen in `docs/hardware_manufacturing_error_report.md` protokollieren.

## Definition of Done (Hardware)
- ERC/DRC clean, Gerber‑Viewer‑Check bestanden.
- JLCPCB‑Import ohne Fehler, Bauteilorientierung/CPL geprüft.
- Vollständige Artefakte: Gerber, BOM, CPL, README/Changelog.

## Nächste Schritte (konkret)
- [ ] DRC/ERC laufen lassen und Findings in `docs/` notieren.
- [ ] BOM‑Felder (LCSC, Hersteller‑PN) auditieren, fehlende Felder ergänzen.
- [ ] CPL Rotationen stichprobenartig prüfen (Polarität/Pin‑1‑Markierung).
- [ ] JLCPCB‑Smoke‑Upload durchführen, Screenshots/Notizen ablegen.
