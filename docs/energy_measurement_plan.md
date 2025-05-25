# Energiemesspunkte und Messmethoden für RP2040 Pizza-Erkennungssystem

## Übersicht

Dieses Dokument definiert die notwendigen Messpunkte und Messmethoden zur Verifikation des Energieverbrauchs des RP2040-basierten Pizza-Erkennungssystems. Die Messungen sind erforderlich, um die Simulation zu validieren und die Batterielebensdauer zu optimieren.

## Hardware-Systemarchitektur

### Hauptkomponenten
- **RP2040 Mikrocontroller**: Dual-Core ARM Cortex-M0+, 133MHz
- **OV2640 Kamerasensor**: 320x240 Pixel, RGB-Ausgabe
- **Stromversorgung**: Li-Ion Batterie (RCR123A/16340), 3.7V nominal, 1500mAh
- **Spannungsregler**: Buck-Boost DC/DC Wandler (U3) für stabile 3.3V
- **Schutzschaltung**: P-Kanal MOSFET Verpolungsschutz (Q1)

### Stromversorgungsarchitektur
```
BT1 (3.0-4.2V) → SW1 → Q1 (Schutz) → U3 (Buck-Boost) → 3.3V Rail
                                                      ↓
                                              RP2040 + OV2640
```

## Messpunkte Definition

### 1. Primäre Messpunkte (Kritisch)

#### MP1: Gesamtsystem-Stromverbrauch
- **Position**: Zwischen Batterie (BT1) und Hauptschalter (SW1)
- **Zweck**: Messung des gesamten Systemstromverbrauchs
- **Erwarteter Bereich**: 0.5mA (Sleep) bis 200mA (Peak Active)
- **Messmethode**: Präzisions-Shunt-Widerstand (1-10mΩ) + Differenzverstärker

#### MP2: Buck-Boost Eingang
- **Position**: Zwischen Verpolungsschutz (Q1) und Buck-Boost Eingang (U3 VIN)
- **Zweck**: Bewertung der Schutzschaltung und Eingangsstrom zum Regler
- **Erwarteter Bereich**: 0.5mA bis 200mA
- **Messmethode**: Shunt-Widerstand (10mΩ) + Präzisions-Op-Amp

#### MP3: Buck-Boost Ausgang (3.3V Rail)
- **Position**: Buck-Boost Ausgang (U3 VOUT) vor Lastverteilung
- **Zweck**: Messung der 3.3V-Rail-Belastung und Regler-Effizienz
- **Erwarteter Bereich**: 0.3mA bis 150mA
- **Messmethode**: Shunt-Widerstand (10mΩ) + instrumentierter Verstärker

### 2. Sekundäre Messpunkte (Detailanalyse)

#### MP4: RP2040 MCU Versorgung
- **Position**: 3.3V-Rail zu RP2040 VDD-Pins
- **Zweck**: Isolierung des MCU-Stromverbrauchs
- **Erwarteter Bereich**: 0.2mA bis 120mA
- **Messmethode**: Kleiner Shunt-Widerstand (5mΩ) + Präzisions-ADC

#### MP5: OV2640 Kamera Versorgung
- **Position**: 3.3V-Rail zu OV2640 VCC
- **Zweck**: Kamera-spezifischer Stromverbrauch
- **Erwarteter Bereich**: 0.1mA bis 50mA
- **Messmethode**: Shunt-Widerstand (20mΩ) + Differenzverstärker

#### MP6: I/O und Peripherie
- **Position**: 3.3V-Rail zu GPIO-Versorgung und LEDs
- **Zweck**: Messung des I/O-Verbrauchs
- **Erwarteter Bereich**: 0.05mA bis 20mA
- **Messmethode**: Shunt-Widerstand (50mΩ) + Op-Amp

### 3. Spannungsüberwachungspunkte

#### VP1: Batteriespannung
- **Position**: Direkt an Batterieanschlüssen (BT1)
- **Zweck**: Batteriespannungsüberwachung und SOC-Bestimmung
- **Bereich**: 2.8V bis 4.2V
- **Messmethode**: Spannungsteiler + ADC oder Voltmeter

#### VP2: Regler-Eingangsspannung
- **Position**: Buck-Boost Eingang (U3 VIN)
- **Zweck**: Eingangsspannung zum Spannungsregler
- **Bereich**: 2.8V bis 4.2V
- **Messmethode**: Spannungsteiler + ADC

#### VP3: Geregelte 3.3V-Ausgangsspannung
- **Position**: Buck-Boost Ausgang (U3 VOUT)
- **Zweck**: Überwachung der Spannungsstabilität
- **Bereich**: 3.2V bis 3.4V
- **Messmethode**: Direktmessung mit Präzisions-ADC

## Empfohlene Messgeräte und Methoden

### Option 1: Präzisions-Power-Analyzer (Empfohlen)
- **Gerät**: Keysight N6705C oder ähnlicher DC Power Analyzer
- **Vorteile**: 
  - Hohe Genauigkeit (±0.02% Strom, ±0.01% Spannung)
  - Simultane Mehrkanal-Messung
  - Hochauflösende Zeitaufzeichnung
  - Integrierte Datenerfassung
- **Nachteile**: Hohe Kosten
- **Anwendung**: Alle Messpunkte MP1-MP6

### Option 2: Oszilloskop + Shunt-Widerstände
- **Gerät**: 4-Kanal-Oszilloskop (≥100MHz) + Präzisions-Shunt-Widerstände
- **Shunt-Spezifikationen**:
  - Widerstandswerte: 1mΩ - 50mΩ je nach Strombereich
  - Toleranz: ±0.1% oder besser
  - Temperaturkoeffizient: <25ppm/°C
  - Induktivität: <10nH
- **Vorteile**: Zeitaufgelöste Messungen, niedrigere Kosten
- **Nachteile**: Manuelle Auswertung erforderlich

### Option 3: Multimeter + Stromzangen (Basis-Messung)
- **Gerät**: Präzisions-Multimeter (6½-stellig) + AC/DC-Stromzangen
- **Anwendung**: Statische Messungen und Durchschnittswerte
- **Vorteile**: Kostengünstig, einfache Handhabung
- **Nachteile**: Keine zeitaufgelösten Messungen

### Option 4: Embedded Power-Monitoring (In-System)
- **Komponenten**: INA219/INA226 Current-Sense-ICs
- **Integration**: Direkt auf der PCB montiert
- **Vorteile**: Kontinuierliche Überwachung, keine externen Geräte
- **Nachteile**: Zusätzliche Hardware erforderlich, begrenzte Genauigkeit

## Messmethodik

### Shunt-Widerstand Dimensionierung

#### Strommessung über Shunt-Widerstand
```
I = V_shunt / R_shunt
P_loss = I² × R_shunt
```

#### Empfohlene Shunt-Werte:
- **MP1 (Gesamtsystem)**: 10mΩ (Max. Verlust: 400µW bei 200mA)
- **MP2/MP3 (Buck-Boost)**: 10mΩ 
- **MP4 (RP2040)**: 5mΩ (Max. Verlust: 72µW bei 120mA)
- **MP5 (Kamera)**: 20mΩ (Max. Verlust: 50µW bei 50mA)
- **MP6 (I/O)**: 50mΩ (Max. Verlust: 20µW bei 20mA)

### Verstärker-Konfiguration

#### Instrumentierter Verstärker für Präzisionsmessungen:
- **IC-Empfehlung**: INA826, AD8422, oder ähnlich
- **Verstärkung**: 100x - 1000x je nach Strombereich
- **Bandbreite**: ≥1MHz für dynamische Messungen
- **Offset**: <10µV

### Zeitaufgelöste Messstrategie

#### Betriebsmodi-Identifikation:
1. **Sleep-Modus**: Strom <1mA, Dauer: Sekunden bis Minuten
2. **Wake-Up**: Stromanstieg >10mA in <10ms
3. **Initialisierung**: 10-50mA, Dauer: 100-500ms
4. **Kamera-Aktivierung**: +40mA zusätzlich, Dauer: 50-200ms
5. **Inferenz**: Peak bis 180mA, Dauer: 20-50ms
6. **Idle/Standby**: 5-15mA zwischen Operationen

#### Sampling-Anforderungen:
- **Minimale Abtastrate**: 10kHz (Nyquist-Kriterium für schnelle Übergänge)
- **Empfohlene Abtastrate**: 100kHz für detaillierte Analyse
- **Messdauer**: Mindestens 1 vollständiger Duty-Cycle (typisch 10-60 Sekunden)

## Kalibrierung und Validierung

### Kalibrierungsverfahren:
1. **Shunt-Widerstand-Verifikation**: Präzisions-LCR-Meter
2. **Verstärker-Offset-Korrektur**: Nullstrom-Messung
3. **Systemlinearität**: Bekannte Stromquellen (1mA, 10mA, 100mA)
4. **Temperaturkompensation**: Messungen bei 0°C, 25°C, 50°C

### Validierungstests:
1. **Statische Verifikation**: Vergleich mit kalibrierten Referenzgeräten
2. **Dynamische Verifikation**: Künstliche Lastzyklen
3. **Langzeit-Stabilität**: 24h-Messungen mit bekannter Last

## Erwartete Messergebnisse

### Stromverbrauch-Profile (basierend auf Simulation):

| Betriebsmodus | Erwarteter Strom | Dauer | Häufigkeit |
|---------------|------------------|--------|------------|
| Deep Sleep | 0.5 ± 0.1 mA | 30-300s | 90% der Zeit |
| Wake-Up | 5-15 mA | 10-50ms | Jeder Zyklus |
| Kamera Init | 15-40 mA | 100-200ms | Jeder Zyklus |
| Bildaufnahme | 40-60 mA | 50-100ms | Jeder Zyklus |
| Inferenz | 150-180 mA | 20-50ms | Jeder Zyklus |
| Datenlogger | 10-20 mA | 10-100ms | Nach Inferenz |

### Power Supply Effizienz:
- **Buck-Boost Effizienz**: >85% bei 10mA, >90% bei 100mA
- **Gesamtsystem-Effizienz**: >80% unter Last

### Batterielebensdauer-Verifikation:
- **Ziel**: >200 Stunden bei 1 Inferenz/Minute
- **Berechnung**: 1500mAh / I_avg = Laufzeit

## Durchführungsplan

### Phase 1: Hardware-Vorbereitung (1-2 Tage)
1. Identifikation der Messpunkte auf der PCB
2. Installation von Shunt-Widerständen (falls nicht vorhanden)
3. Verkabelung der Messpunkte
4. Aufbau der Messverstärker-Schaltung

### Phase 2: Statische Messungen (1 Tag)
1. Spannungsmessungen an allen Punkten
2. Statische Strommessungen in jedem Betriebsmodus
3. Effizienz-Messungen des Buck-Boost-Reglers

### Phase 3: Dynamische Messungen (2-3 Tage)
1. Zeitaufgelöste Stromprofile während typischer Zyklen
2. Peak-Strom-Messungen während Inferenz
3. Wake-Up-/Sleep-Übergangsanalyse
4. Langzeit-Power-Profiling (24h)

### Phase 4: Validierung und Dokumentation (1 Tag)
1. Vergleich mit Simulationsergebnissen
2. Identifikation von Optimierungspotenzialen
3. Erstellung des Messberichts

## Sicherheitshinweise

### Elektrische Sicherheit:
- Maximale Betriebsspannung: 5V DC
- Schutz vor Verpolung durch P-MOSFET bereits vorhanden
- ESD-Schutz beim Handling der PCB beachten

### Messgenauigkeit:
- Thermische Effekte: 30-minütige Einlaufzeit vor Messungen
- EMV-Schutz: Geschirmte Kabel für Signalleitungen verwenden
- Masse-Schleifen vermeiden: Single-Point-Ground-Konzept

## Dokumentationsanforderungen

### Messdokumentation:
1. **Rohdaten**: Alle Messungen in CSV/Excel-Format
2. **Kalibrierungszertifikate**: Für alle verwendeten Messgeräte
3. **Messaufbau-Fotos**: Dokumentation der physischen Konfiguration
4. **Analyse-Scripts**: Python/MATLAB-Code zur Datenauswertung

### Ergebnisbericht:
1. **Zusammenfassung**: Vergleich Simulation vs. Messung
2. **Detailanalyse**: Stromprofile für jeden Betriebsmodus
3. **Optimierungsempfehlungen**: Basierend auf Messergebnissen
4. **Batterielebensdauer-Vorhersage**: Aktualisierte Modelle

## Anhang

### A.1: Komponentenspezifikationen
- RP2040: ARM Cortex-M0+, 2x133MHz, 264KB SRAM, 2MB Flash
- OV2640: CMOS-Sensor, 2MP, 320x240@30fps bei reduzierter Leistung
- Buck-Boost: Eingangsspannung 2.8-4.2V, Ausgangsspannung 3.3V±3%

### A.2: Simulationsergebnisse (Referenz)
- Durchschnittlicher Stromverbrauch: 6.85mA (Duty-Cycle 90% Sleep)
- Batterielebensdauer: 9.1 Tage bei CR123A (1500mAh)
- Peak-Verbrauch: 180mA während Inferenz
- Sleep-Verbrauch: 0.5mA

### A.3: PCB-Layout Considerations
- Shunt-Widerstand-Platzierung: Möglichst nah an Versorgungspins
- Kelvin-Anschluss für Präzisionsmessungen empfohlen
- Thermische Entkopplung der Messelektronik von Leistungskomponenten

---
**Dokument-Version**: 1.0  
**Erstellt**: 2025-05-24  
**Autor**: Pizza Detection Team  
**Status**: Zur Umsetzung freigegeben
