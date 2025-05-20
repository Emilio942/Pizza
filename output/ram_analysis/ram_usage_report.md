# RAM-Nutzungsanalyse für Pizza-Detektions-System

## Übersicht

**Datum:** 2025-05-19 06:35:39

**Gesamtnutzung:** 170.6 KB / 264.0 KB (64.6%)

**Verfügbarer RAM:** 93.4 KB

**Erfüllt Anforderungen:** Ja (Ziel: max 204 KB)

## Komponentenübersicht

| Komponente | Größe (KB) | Anteil (%) |
|------------|------------|------------|
| Tensor Arena | 10.8 | 6.3 |
| Framebuffer | 76.8 | 45.0 |
| System Overhead | 40.0 | 23.4 |
| Vorverarbeitungspuffer | 27.0 | 15.8 |
| Stack | 8.0 | 4.7 |
| Heap | 5.0 | 2.9 |
| Statische Puffer | 3.0 | 1.8 |

## Empfehlungen

- **Framebuffer**: Verwenden Sie eine niedrigere Auflösung oder ein einfacheres Pixelformat (z.B. Graustufen statt RGB).
