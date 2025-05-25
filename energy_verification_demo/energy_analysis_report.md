# Energiemessdaten-Analyse

**Erstellt am:** 2025-05-24 12:07:24
**Eingabedatei:** test_energy_data.csv

## Zusammenfassung

- **Messdauer:** 120.0 Sekunden
- **Analysierte Messpunkte:** 6
- **Erkannte Betriebsmodi:** 6

## MP1: Gesamtsystem

### Allgemeine Kennzahlen

- **Durchschnittliche Leistung:** 3.90 mW
- **Spitzenleistung:** 647.30 mW
- **Messdauer:** 120.0 Sekunden

### Betriebsmodi-Analyse

| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |
|-------|-----------|------------|-------------|----------------|---------------|
| Deep Sleep | 117.8 | 98.2 | 0.50 | 0.73 | 0.0164 |
| Wake-Up | 0.2 | 0.2 | 9.69 | 11.88 | 0.0006 |
| Camera Init | 0.8 | 0.7 | 25.01 | 30.82 | 0.0056 |
| Image Capture | 0.4 | 0.3 | 50.19 | 58.08 | 0.0056 |
| Inference | 0.2 | 0.2 | 153.82 | 196.15 | 0.0092 |
| Data Logging | 0.6 | 0.5 | 13.50 | 17.76 | 0.0021 |

### Geschätzte Batterielebensdauer

| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |
|-------------|-----------------|----------------------|--------------------|
| CR123A | 1500 | 1267.8 | 52.8 |
| AA_Alkaline | 2500 | 2112.9 | 88.0 |
| 18650_LiIon | 3400 | 2873.6 | 119.7 |
| LiPo_500mAh | 500 | 422.6 | 17.6 |

---

## MP2: Buck-Boost Eingang

### Allgemeine Kennzahlen

- **Durchschnittliche Leistung:** 3.90 mW
- **Spitzenleistung:** 647.30 mW
- **Messdauer:** 120.0 Sekunden

### Betriebsmodi-Analyse

| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |
|-------|-----------|------------|-------------|----------------|---------------|
| Deep Sleep | 117.8 | 98.2 | 0.50 | 0.73 | 0.0164 |
| Wake-Up | 0.2 | 0.2 | 9.69 | 11.88 | 0.0006 |
| Camera Init | 0.8 | 0.7 | 25.01 | 30.82 | 0.0056 |
| Image Capture | 0.4 | 0.3 | 50.19 | 58.08 | 0.0056 |
| Inference | 0.2 | 0.2 | 153.82 | 196.15 | 0.0092 |
| Data Logging | 0.6 | 0.5 | 13.50 | 17.76 | 0.0021 |

### Geschätzte Batterielebensdauer

| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |
|-------------|-----------------|----------------------|--------------------|
| CR123A | 1500 | 1267.8 | 52.8 |
| AA_Alkaline | 2500 | 2112.9 | 88.0 |
| 18650_LiIon | 3400 | 2873.6 | 119.7 |
| LiPo_500mAh | 500 | 422.6 | 17.6 |

---

## MP3: Buck-Boost Ausgang (3.3V)

### Allgemeine Kennzahlen

- **Durchschnittliche Leistung:** 3.90 mW
- **Spitzenleistung:** 647.30 mW
- **Messdauer:** 120.0 Sekunden

### Betriebsmodi-Analyse

| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |
|-------|-----------|------------|-------------|----------------|---------------|
| Deep Sleep | 117.8 | 98.2 | 0.50 | 0.73 | 0.0164 |
| Wake-Up | 0.2 | 0.2 | 9.69 | 11.88 | 0.0006 |
| Camera Init | 0.8 | 0.7 | 25.01 | 30.82 | 0.0056 |
| Image Capture | 0.4 | 0.3 | 50.19 | 58.08 | 0.0056 |
| Inference | 0.2 | 0.2 | 153.82 | 196.15 | 0.0092 |
| Data Logging | 0.6 | 0.5 | 13.50 | 17.76 | 0.0021 |

### Geschätzte Batterielebensdauer

| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |
|-------------|-----------------|----------------------|--------------------|
| CR123A | 1500 | 1267.8 | 52.8 |
| AA_Alkaline | 2500 | 2112.9 | 88.0 |
| 18650_LiIon | 3400 | 2873.6 | 119.7 |
| LiPo_500mAh | 500 | 422.6 | 17.6 |

---

## MP4: RP2040 MCU

### Allgemeine Kennzahlen

- **Durchschnittliche Leistung:** 3.90 mW
- **Spitzenleistung:** 647.30 mW
- **Messdauer:** 120.0 Sekunden

### Betriebsmodi-Analyse

| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |
|-------|-----------|------------|-------------|----------------|---------------|
| Deep Sleep | 117.8 | 98.2 | 0.50 | 0.73 | 0.0164 |
| Wake-Up | 0.2 | 0.2 | 9.69 | 11.88 | 0.0006 |
| Camera Init | 0.8 | 0.7 | 25.01 | 30.82 | 0.0056 |
| Image Capture | 0.4 | 0.3 | 50.19 | 58.08 | 0.0056 |
| Inference | 0.2 | 0.2 | 153.82 | 196.15 | 0.0092 |
| Data Logging | 0.6 | 0.5 | 13.50 | 17.76 | 0.0021 |

### Geschätzte Batterielebensdauer

| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |
|-------------|-----------------|----------------------|--------------------|
| CR123A | 1500 | 1267.8 | 52.8 |
| AA_Alkaline | 2500 | 2112.9 | 88.0 |
| 18650_LiIon | 3400 | 2873.6 | 119.7 |
| LiPo_500mAh | 500 | 422.6 | 17.6 |

---

## MP5: OV2640 Kamera

### Allgemeine Kennzahlen

- **Durchschnittliche Leistung:** 3.90 mW
- **Spitzenleistung:** 647.30 mW
- **Messdauer:** 120.0 Sekunden

### Betriebsmodi-Analyse

| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |
|-------|-----------|------------|-------------|----------------|---------------|
| Deep Sleep | 117.8 | 98.2 | 0.50 | 0.73 | 0.0164 |
| Wake-Up | 0.2 | 0.2 | 9.69 | 11.88 | 0.0006 |
| Camera Init | 0.8 | 0.7 | 25.01 | 30.82 | 0.0056 |
| Image Capture | 0.4 | 0.3 | 50.19 | 58.08 | 0.0056 |
| Inference | 0.2 | 0.2 | 153.82 | 196.15 | 0.0092 |
| Data Logging | 0.6 | 0.5 | 13.50 | 17.76 | 0.0021 |

### Geschätzte Batterielebensdauer

| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |
|-------------|-----------------|----------------------|--------------------|
| CR123A | 1500 | 1267.8 | 52.8 |
| AA_Alkaline | 2500 | 2112.9 | 88.0 |
| 18650_LiIon | 3400 | 2873.6 | 119.7 |
| LiPo_500mAh | 500 | 422.6 | 17.6 |

---

## MP6: I/O und Peripherie

### Allgemeine Kennzahlen

- **Durchschnittliche Leistung:** 3.90 mW
- **Spitzenleistung:** 647.30 mW
- **Messdauer:** 120.0 Sekunden

### Betriebsmodi-Analyse

| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |
|-------|-----------|------------|-------------|----------------|---------------|
| Deep Sleep | 117.8 | 98.2 | 0.50 | 0.73 | 0.0164 |
| Wake-Up | 0.2 | 0.2 | 9.69 | 11.88 | 0.0006 |
| Camera Init | 0.8 | 0.7 | 25.01 | 30.82 | 0.0056 |
| Image Capture | 0.4 | 0.3 | 50.19 | 58.08 | 0.0056 |
| Inference | 0.2 | 0.2 | 153.82 | 196.15 | 0.0092 |
| Data Logging | 0.6 | 0.5 | 13.50 | 17.76 | 0.0021 |

### Geschätzte Batterielebensdauer

| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |
|-------------|-----------------|----------------------|--------------------|
| CR123A | 1500 | 1267.8 | 52.8 |
| AA_Alkaline | 2500 | 2112.9 | 88.0 |
| 18650_LiIon | 3400 | 2873.6 | 119.7 |
| LiPo_500mAh | 500 | 422.6 | 17.6 |

---

## Empfohlene Optimierungen

Basierend auf der Analyse werden folgende Optimierungen empfohlen:


## Visualisierungen

Die folgenden Diagramme wurden erstellt:

- `energy_analysis_MP1.png` - Detailanalyse MP1
- `energy_analysis_MP2.png` - Detailanalyse MP2
- `energy_analysis_MP3.png` - Detailanalyse MP3
- `energy_analysis_MP4.png` - Detailanalyse MP4
- `energy_analysis_MP5.png` - Detailanalyse MP5
- `energy_analysis_MP6.png` - Detailanalyse MP6
- `energy_analysis_summary.png` - Gesamtübersicht aller Messpunkte
