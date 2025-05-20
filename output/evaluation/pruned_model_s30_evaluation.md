# Evaluierungsbericht: MicroPizzaNetV2 (gepruned, 30% Sparsity)

Zeitstempel: 2025-05-18 17:36:54
Modell: MicroPizzaNetV2 (pruned, 30% sparsity)
Pfad: `/home/emilio/Documents/ai/pizza/models/micropizzanetv2_quantized_s30.tflite`

## Gesamtmetriken

- **Accuracy**: 90.20%
- **Precision**: 91.86%
- **Recall**: 91.88%
- **F1-Score**: 0.9185
- **Testdaten**: 500 Bilder
- **Inferenzzeit**: 22 ms

## Vergleich zum Originalmodell

- **Genauigkeitsänderung**: -3.80%
- **Relative Genauigkeitsänderung**: -4.04%
- **Geschwindigkeitsverbesserung**: 10.0%

## Metriken pro Klasse

| Klasse | Genauigkeit | Precision | Recall | F1-Score |
|--------|------------|-----------|--------|----------|
| basic | 89.50% | 91.75% | 91.75% | 0.9175 |
| pepperoni | 92.20% | 92.00% | 92.00% | 0.9200 |
| margherita | 88.37% | 92.63% | 88.00% | 0.9026 |
| vegetable | 89.32% | 91.75% | 91.75% | 0.9175 |
| not_pizza | 93.90% | 91.18% | 95.88% | 0.9347 |
