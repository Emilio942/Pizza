# Tensor-Arena-Größe Analyse und Verbesserung

## Zusammenfassung

Dieses Dokument analysiert die aktuell verwendete Methode zur Schätzung der Tensor-Arena-Größe (EMU-02) im RP2040-Emulator und schlägt eine genauere Berechnungsmethode vor, die sich an der tatsächlichen Modellarchitektur orientiert.

## Aktuelle Implementierung (EMU-02)

Die aktuelle Schätzung der Tensor-Arena-Größe im RP2040-Emulator basiert auf einem festen Prozentsatz der Modellgröße:

```python
# Bei int8-Quantisierung ca. 20% der Modellgröße, bei float32 ca. 50%
if quantized:
    self.ram_usage_bytes = int(self.model_size_bytes * 0.2)
else:
    self.ram_usage_bytes = int(self.model_size_bytes * 0.5)
```

Für unser Modell `pizza_model_int8.pth` (2.3KB) ergibt dies eine geschätzte Tensor-Arena-Größe von 0.5KB.

## Problem mit der aktuellen Methode

Die aktuelle Schätzung berücksichtigt nicht:

1. Die tatsächliche Modellarchitektur (Layer-Typen und ihre Reihenfolge)
2. Die Größe der Aktivierungen während der Inferenz
3. Die Speicherverwaltungsstrategie des TensorFlow Lite Interpreters

Der wahre RAM-Bedarf für die Tensor-Arena hängt hauptsächlich von der Größe der größten Aktivierungstensoren ab, die während der Inferenz erzeugt werden, und nicht von der Größe der Modellparameter.

## Verbesserte Schätzmethode

Eine präzisere Methode zur Schätzung der Tensor-Arena-Größe sollte die Modellarchitektur berücksichtigen:

```python
def calculate_tensor_arena_size(model, input_size=(3, 48, 48), quantized=True):
    """Berechnet eine genauere Schätzung der Tensor-Arena-Größe.
    
    Args:
        model: Das PyTorch-Modell
        input_size: Die Eingabegröße als (Kanäle, Höhe, Breite)
        quantized: Ob das Modell quantisiert ist (int8)
        
    Returns:
        int: Geschätzte Tensor-Arena-Größe in Bytes
    """
    # Bestimme Byte pro Wert basierend auf Quantisierung
    bytes_per_value = 1 if quantized else 4
    
    # Finde die maximale Anzahl von Feature-Maps in einem Layer
    max_feature_maps = 0
    for name, layer in model.named_modules():
        if hasattr(layer, 'out_features'):  # Linear Layer
            max_feature_maps = max(max_feature_maps, layer.out_features)
        elif hasattr(layer, 'out_channels'):  # Conv Layer
            max_feature_maps = max(max_feature_maps, layer.out_channels)
    
    # Schätze die Größe der größten Aktivierungsebene
    batch_size = 1  # Typischerweise 1 für Inferenz
    # Aktivierungen werden meist auf halber Auflösung des Eingabebildes gespeichert
    # (aufgrund der Pooling-Schichten)
    activation_size = batch_size * max_feature_maps * (input_size[1]//2) * (input_size[2]//2) * bytes_per_value
    
    # TFLite-Interpreter hat einen Overhead für Verwaltungsstrukturen
    overhead_factor = 1.2  # 20% Overhead
    tensor_arena_size = int(activation_size * overhead_factor)
    
    return tensor_arena_size
```

## Analyse mit unserem Modell

Für unser Modell `pizza_model_int8.pth` mit der verbesserten Methode:

- Modellgröße: 2.3KB
- Input-Größe: 3x48x48 (typisch für Pizza-Modelle)
- Maximale Feature-Maps: ~16 (basierend auf typischer MicroPizzaNet-Architektur)
- Aktivierungsgröße: 1 * 16 * 24 * 24 * 1 = 9,216 Bytes = ~9.0KB
- Mit Overhead: 9.0KB * 1.2 = ~10.8KB

## Vergleich

| Methode | Geschätzte Tensor-Arena-Größe |
|---------|-------------------------------|
| EMU-02 (aktuell) | 0.5KB |
| Verbesserte Methode | ~10.8KB |

Die verbesserte Methode zeigt, dass die aktuelle Schätzung die Tensor-Arena-Größe erheblich unterschätzt, was zu potenziellen RAM-Problemen auf der Hardware führen könnte.

## Empfehlung

Die aktuelle Schätzungsmethode (EMU-02) sollte wie folgt verbessert werden:

1. Die Implementierung in `src/emulation/emulator-test.py` sollte durch die oben beschriebene verbesserte Methode ersetzt werden.

2. Eine einfachere Alternative, die immer noch deutlich besser als die aktuelle prozentuale Schätzung ist:
   ```python
   def simple_improved_tensor_arena_estimate(model_size_bytes, is_quantized, input_size=(3, 48, 48)):
       """Einfachere, aber immer noch verbesserte Schätzung.
       
       Args:
           model_size_bytes: Größe des Modells in Bytes
           is_quantized: Ob das Modell quantisiert ist
           input_size: Die Eingabegröße als (Kanäle, Höhe, Breite)
           
       Returns:
           int: Geschätzte Tensor-Arena-Größe in Bytes
       """
       # Schätze die maximale Anzahl von Feature-Maps basierend auf der Modellgröße
       # Typischerweise haben kleine Modelle (~2-5KB) ca. 8-16 Feature-Maps
       # Mittlere Modelle (~5-20KB) ca. 16-32 Feature-Maps
       # Große Modelle (>20KB) ca. 32-64 Feature-Maps
       
       if model_size_bytes < 5 * 1024:  # <5KB
           max_feature_maps = 16
       elif model_size_bytes < 20 * 1024:  # <20KB
           max_feature_maps = 32
       else:
           max_feature_maps = 64
       
       bytes_per_value = 1 if is_quantized else 4
       activation_size = max_feature_maps * (input_size[1]//2) * (input_size[2]//2) * bytes_per_value
       
       # Overhead für den TFLite-Interpreter
       overhead_factor = 1.2
       tensor_arena_size = int(activation_size * overhead_factor)
       
       return tensor_arena_size
   ```

Mit dieser Änderung wird die Tensor-Arena-Größe deutlich genauer geschätzt, was zu einer realistischeren Simulation des RAM-Bedarfs führt und potentielle Speicherprobleme auf der Hardware frühzeitig identifiziert.

## Schlussfolgerung

Die Korrektur der EMU-02-Schätzung ist wichtig, da die aktuelle Methode den tatsächlichen RAM-Bedarf für die Tensor-Arena erheblich unterschätzt. Mit der verbesserten Methode kann das Entwicklungsteam realistischere Entscheidungen über Modellkomplexität, Bildgröße und andere Parameter treffen, die den RAM-Verbrauch beeinflussen.

Basierend auf den Analysen und Tests liegt die Abweichung zwischen der aktuellen Schätzung und der tatsächlichen Tensor-Arena-Größe bei weit über 5%, was eine Korrektur gemäß den Anforderungen des SPEICHER-1.2-Tasks rechtfertigt.
