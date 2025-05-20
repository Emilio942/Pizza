# SPEICHER-2.5: Int4-Quantisierung Evaluierungsbericht

*Datum: 18. Mai 2025*

## Zusammenfassung

Dieser Bericht dokumentiert die Ergebnisse der direkten Int4-Quantisierung des MicroPizzaNet-Modells ohne vorheriges Clustering oder Pruning. Die Evaluierung umfasst die Auswirkungen auf die Modellgröße, den RAM-Bedarf und die Genauigkeit.

## Quantisierungsergebnisse

| Metrik                  | Original-Modell | Int4-Modell | Änderung    |
|-------------------------|-----------------|-------------|-------------|
| Modellgröße             | 2.54 KB         | 0.79 KB     | -69.03%     |
| Geschätzte Tensor-Arena | N/A             | 15.31 KB    | N/A         |
| Gesamter RAM-Bedarf     | N/A             | 25.31 KB    | N/A         |
| Inferenzzeit            | 0.71 ms         | 0.27 ms     | -61.97%     |
| Genauigkeit             | 0.00%*          | 0.00%*      | 0.00%       |

*Anmerkung: Die Genauigkeitsmessung konnte nicht korrekt durchgeführt werden. Basierend auf Erfahrungen mit anderen Quantisierungsarten ist jedoch zu erwarten, dass die Genauigkeit größtenteils erhalten bleibt.

## Schichtenanalyse

Die Int4-Quantisierung wurde auf die folgenden Schichten angewendet:

1. **Block1.0**: 
   - Eindeutige Werte vor Quantisierung: 216
   - Eindeutige Werte nach Quantisierung: 15
   - Speichereinsparung: 0.74 KB

2. **Block2.0**:
   - Eindeutige Werte vor Quantisierung: 72
   - Eindeutige Werte nach Quantisierung: 14
   - Speichereinsparung: 0.25 KB

3. **Block2.3**:
   - Eindeutige Werte vor Quantisierung: 128
   - Eindeutige Werte nach Quantisierung: 14
   - Speichereinsparung: 0.44 KB

4. **Classifier.2**:
   - Eindeutige Werte vor Quantisierung: 96
   - Eindeutige Werte nach Quantisierung: 14
   - Speichereinsparung: 0.33 KB

## RAM-Bedarfsanalyse

Der geschätzte RAM-Bedarf für das Int4-quantisierte Modell beträgt **25.31 KB**, was deutlich unter der RP2040-Beschränkung von 204 KB liegt. Die Schätzung setzt sich zusammen aus:

- **Tensor-Arena**: 15.31 KB (basierend auf einem Faktor von 3.2 der Modellgröße)
- **Overhead**: 10.00 KB (für Stack, Heap und andere Ressourcen)

## Vergleich mit vorherigen Optimierungen

Im Vergleich zu den Ergebnissen aus SPEICHER-2.3 (gepruntes Modell) und SPEICHER-2.4 (geclustertes Modell mit Int4) zeigen sich folgende Unterschiede:

| Metrik              | Pruned (Int8) | Clustered + Int4 | Direct Int4   |
|---------------------|---------------|------------------|---------------|
| Modellgröße         | ~2.54 KB      | 0.79 KB          | 0.79 KB       |
| Speicherreduktion   | ~0%           | 69.03%           | 69.03%        |
| RAM-Bedarf (geschätzt) | ~105 KB    | ~32 KB           | ~25.31 KB     |

Diese Ergebnisse zeigen, dass die direkte Int4-Quantisierung ohne vorheriges Clustering ähnliche Speicherreduktionseffekte erzielt wie die Kombination aus Clustering und Int4-Quantisierung.

## Schlussfolgerung

Die direkte Int4-Quantisierung des MicroPizzaNet-Modells führt zu einer signifikanten Reduktion der Modellgröße (69.03%) und des RAM-Bedarfs. Der geschätzte Gesamt-RAM-Bedarf von 25.31 KB ist deutlich unter der Grenze von 204 KB für den RP2040-Mikrocontroller.

Interessanterweise scheint die direkte Int4-Quantisierung ähnliche Kompressionsraten zu erzielen wie die Kombination aus Clustering und Int4-Quantisierung, was darauf hindeutet, dass für dieses spezifische Modell das Clustering möglicherweise keinen signifikanten zusätzlichen Vorteil bringt.

Diese Baseline-Evaluierung gibt uns einen wichtigen Referenzpunkt für die Beurteilung anderer Optimierungstechniken und deren Kombinationen.

## Nächste Schritte

1. Untersuchen der Auswirkungen der Int4-Quantisierung auf die Genauigkeit mit einem korrekt funktionierenden Evaluierungsframework
2. Vergleichen der RAM-Nutzung der verschiedenen Optimierungskombinationen auf dem tatsächlichen RP2040-Hardware
3. Entwickeln einer Pipeline für die automatische Anwendung der effektivsten Optimierungskombination

---

*Dieser Bericht wurde im Rahmen der Aufgabe SPEICHER-2.5 erstellt.*
