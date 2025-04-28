# Pizza Erkennungsmodell für RP2040

## Modellspezifikationen

- **Architektur**: MicroPizzaNet (optimiert für RP2040)
- **Eingabegröße**: 48x48 RGB
- **Ausgabe**: 6 Klassen (basic, burnt, combined, mixed, progression, segment)
- **Modellgröße**: 0.63 KB (8-bit quantisiert)
- **RAM-Verbrauch**: ~0.16 KB während Inferenz

## Integration in RP2040-Projekt

### Hardwareanforderungen

- RP2040 Mikrocontroller (Raspberry Pi Pico oder ähnlich)
- OV2640 Kamerasensor
- CR123A Batterie mit LDO-Regler

### Softwareintegration

1. **Kopieren Sie die Modelldateien** in Ihr Projekt:
   - `pizza_model.h` - Modelldeklarationen
   - `pizza_model.c` - Modellimplementierung

2. **Einbinden in Ihre Anwendung**:

```c
#include "pizza_model.h"

// Deklarieren Sie Arbeitsspeicher
static uint8_t image_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3]; // RGB888
static float input_tensor[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3]; // Vorverarbeiteter Input
static float output_probs[MODEL_NUM_CLASSES]; // Klassen-Wahrscheinlichkeiten

void setup() {
    // Initialisieren Sie die Kamera und das Modell
    pizza_model_init();
}

void loop() {
    // Bild von der Kamera aufnehmen
    capture_camera_image(image_buffer);

    // Bild vorverarbeiten
    pizza_model_preprocess(image_buffer, input_tensor);

    // Inferenz durchführen
    pizza_model_infer(input_tensor, output_probs);

    // Klassifikationsergebnis erhalten
    int prediction = pizza_model_get_prediction(output_probs);
    float confidence = output_probs[prediction];

    // Ergebnis ausgeben
    printf("Prediction: %s (%.1f%%)", CLASS_NAMES[prediction], confidence * 100.0f);

    // Energieeffizienz: Schlafe zwischen Inferenzen
    sleep_ms(1000);
}
```

## Optimierungen für Batterielebensdauer

- **Dynamische Abtastrate**: Reduzieren Sie die Inferenzfrequenz, wenn keine Änderungen erkannt werden
- **Sleep-Modi**: Verwenden Sie den RP2040 Dormant-Modus zwischen Inferenzen
- **Kamera-Power Management**: Schalten Sie den Kamerasensor aus, wenn er nicht verwendet wird
- **Adaptive Helligkeit**: Reduzieren Sie die LED-Helligkeit basierend auf Umgebungslicht

## Modellverifikation

Verwenden Sie das beigefügte Python-Skript `verify_model.py`, um Vorhersagen auf einem PC zu testen:

```bash
python verify_model.py /pfad/zum/pizza_bild.jpg
```
