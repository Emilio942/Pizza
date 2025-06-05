# Spatial-MLLM Architektur-Dokumentation

## Datum: 2025-06-02
## Status: Detaillierte Architektur-Analyse für SPATIAL-1.2

---

## 1. Überblick der Dual-Encoder-Architektur

Spatial-MLLM implementiert eine innovative **Dual-Encoder-Architektur**, die sowohl 2D visuelle Features als auch 3D räumliche Geometrie-Informationen verarbeitet. Die Architektur basiert auf dem Qwen2.5-VL Backbone und integriert VGGT (Video-based Geometry-Guided Transformer) für räumliche Intelligenz.

### 1.1 Hauptkomponenten
```
Spatial-MLLM Pipeline:
├── 2D Visual Encoder (Qwen2.5-VL Vision Transformer)
├── 3D Spatial Encoder (VGGT - Video-based Geometry-Guided Transformer)  
├── Feature Connector (VGGTEmbeddingMerger)
└── LLM Backbone (Qwen2.5-VL Language Model)
```

---

## 2. Kern-Architektur-Komponenten

### 2.1 Hauptklasse: `Qwen2_5_VL_VGGTForConditionalGeneration`

**Datei:** `src/models/modeling_qwen2_5_vl.py:2153`

```python
class Qwen2_5_VL_VGGTForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        # 2D Visual Encoder
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        
        # LLM Backbone
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 3D Spatial Encoder
        self.vggt = VGGT()
        
        # Feature Connector
        self.merger_config = VGGTEmbeddingMergerConfig(
            input_dim=2 * self.vggt.aggregator.embed_dim,  # Concat tokens after frame and global attention
            output_dim=self.config.hidden_size
        )
        self.vggt_embedding_merger = VGGTEmbeddingMerger(config=self.merger_config)
```

---

## 3. 2D Visual Encoder (Standard Computer Vision)

### 3.1 Basis-Komponente
- **Typ:** Qwen2.5-VL Vision Transformer
- **Funktion:** Extraktion von 2D visuellen Features aus Eingabebildern
- **Architektur:** Standard Vision Transformer mit Patch-basierter Tokenisierung

### 3.2 Input/Output Spezifikationen
```python
# Input Format
Input: images (B, F, C, H, W)
# B: Batch Size
# F: Frame Count (für Videos) oder 1 (für Bilder)  
# C: 3 (RGB Kanäle)
# H, W: Bildauflösung

# Output Format
Output: visual_features (B, F, patch_tokens, embed_dim)
# patch_tokens: Anzahl der Patch-Tokens pro Bild
# embed_dim: Dimensionalität der visuellen Features
```

---

## 4. 3D Spatial Encoder (VGGT - Video-based Geometry-Guided Transformer)

### 4.1 VGGT Hauptklasse
**Datei:** `src/models/vggt/models/vggt.py`

```python
class VGGT(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        # Kern-Aggregator für räumliche Feature-Fusion
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # Spezialisierte Heads für verschiedene räumliche Aufgaben
        self.camera_head = CameraHead(dim_in=2 * embed_dim)           # Kamera-Pose Estimation
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4) # 3D Punkt-Prediction
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2) # Tiefenschätzung
        self.track_head = TrackHead(dim_in=2 * embed_dim)             # Punkt-Tracking
```

### 4.2 Aggregator - Kern-Komponente für räumliche Fusion
**Datei:** `src/models/vggt/models/aggregator.py`

**Kernfunktionalität:**
- **Alternating Attention:** Wechselt zwischen Frame-wise und Global Attention
- **Rotary Position Embedding 2D:** Räumlich-bewusste Positionscodierung
- **Multi-Resolution Processing:** Verarbeitung verschiedener räumlicher Skalen

```python
class Aggregator(nn.Module):
    def __init__(self, 
                 img_size=518, 
                 patch_size=14, 
                 embed_dim=1024,
                 depth=24,                          # Anzahl Transformer-Blöcke
                 num_heads=16,                      # Attention-Heads
                 aa_order=["frame", "global"],      # Alternating Attention Order
                 rope_freq=100,                     # Rotary Embedding Frequenz
                 ...):
        
        # Patch Embedding (ähnlich ViT, aber räumlich erweitert)
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim)
        
        # 2D Rotary Position Embedding für räumliche Awareness
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()
        
        # Frame-wise und Global Attention Blöcke
        self.frame_blocks = nn.ModuleList([...])  # Verarbeitung einzelner Frames
        self.global_blocks = nn.ModuleList([...]) # Cross-Frame Attention
```

### 4.3 Spatial Feature Processing Pipeline

```python
def forward(self, media_input: torch.Tensor):
    """
    Input Shape: (B, F, C, H, W)
    - B: Batch size
    - F: Number of frames (für Video) oder 1 (für Einzelbilder)
    - C: 3 (RGB channels)
    - H, W: Spatial dimensions (518x518 default)
    
    Processing Pipeline:
    1. Patch Embedding: (B, F, C, H, W) → (B, F, N_patches, embed_dim)
    2. Alternating Attention:
       - Frame Attention: Inner-frame spatial relationships
       - Global Attention: Cross-frame temporal-spatial relationships
    3. Spatial Feature Aggregation: (B, F, N_patches, embed_dim) → aggregated_features
    
    Output: 
    - aggregated_tokens_list: List[Tensor] # Features nach jeder Attention-Stufe
    - patch_start_idx: Index-Informationen für Patch-Lokalisierung
    """
```

---

## 5. Feature Connector (VGGTEmbeddingMerger)

### 5.1 Funktionalität
Der Connector fusioniert die Features aus beiden Encodern:

```python
def get_vgg_embeds(self, media_input: torch.Tensor, media_type: str):
    """
    Verbindet 2D visuelle und 3D räumliche Features
    
    Input:
    - media_input: (B, F, C, H, W) - Roh-Eingabedaten
    - media_type: "images" oder "video"
    
    Processing:
    1. VGGT Aggregator verarbeitet räumliche Features
    2. VGGTEmbeddingMerger fusioniert Features mit LLM-Dimensionen
    3. Reshape für LLM-Kompatibilität
    
    Output:
    - vgg_embeds: (S, D) - Sequenz von fusionierten Features
      S: Sequence length (Batch * Frames * Tokens)
      D: LLM hidden dimension
    """
    vgg_aggregator = self.vggt.aggregator
    output_list, patch_start_idx = vgg_aggregator.forward(media_input)
    
    # Shape: (Batch, Frame, Token_num, language_model_dim)
    vgg_embeds = self.vggt_embedding_merger(output_list, patch_start_idx, media_input.shape, media_type)
    
    # Reshape für LLM: (S, D)
    vgg_embeds = vgg_embeds.view(-1, vgg_embeds.shape[-1])
    return vgg_embeds
```

---

## 6. Space-Aware Frame Sampling für Video-Inputs

### 6.1 Temporal-Räumliche Positionscodierung
Spatial-MLLM implementiert eine fortgeschrittene 3D Positional Encoding Strategie:

```python
def get_rope_index(self, ...):
    """
    Berechnet 3D ROPE Index basierend auf Temporal, Height, Width Dimensionen
    
    Beispiel für Video-Verarbeitung:
    - Temporal: 3 patches (verschiedene Video-Segmente)
    - Height: 2 patches (vertikale Teilung)
    - Width: 2 patches (horizontale Teilung)
    
    Vision Temporal Position IDs: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
    Vision Height Position IDs:  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    Vision Width Position IDs:   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    Wichtige Parameter:
    - fps: Frames per second 
    - tokens_per_second: Temporale Granularität (25 default)
    - temporal_patch_size: Frames pro temporal patch (2 default)
    - interval: tokens_per_second * temporal_patch_size / fps
    """
```

---

## 7. Anwendbarkeit auf Pizza-Klassifikation

### 7.1 Relevante Spatial Features für Pizza-Erkennung

#### **2D Visual Encoder - Bestehende Stärken:**
- Farberkennung (Verbrennungsgrad: goldbraun vs. schwarz)
- Texturerkennung (Kruste, Belag, Oberflächenstrukturen)
- Form- und Kontourerkennung (Pizza-Rand, Belag-Verteilung)

#### **3D Spatial Encoder - Neue Möglichkeiten:**
```python
# Räumliche Features für Pizza-Analyse:
spatial_features = {
    "depth_estimation": {
        "anwendung": "Oberflächenhöhe von Belägen",
        "nutzen": "Unterscheidung zwischen flachen/dicken Belägen"
    },
    "surface_geometry": {
        "anwendung": "3D Oberflächentextur der Kruste", 
        "nutzen": "Verbrennungstiefe und -verteilung"
    },
    "spatial_attention": {
        "anwendung": "Räumliche Beziehungen zwischen Verbrennungsregionen",
        "nutzen": "Lokalisierung und Ausdehnung von Verbrennungen"
    },
    "geometric_consistency": {
        "anwendung": "Konsistenz der Oberflächenstruktur",
        "nutzen": "Unterscheidung zwischen normaler Bräunung und Verbrennung"
    }
}
```

### 7.2 Adaptierung für statische Pizza-Bilder

**Single-Frame Processing:**
```python
# Für Pizza-Klassifikation (statische Bilder):
pizza_input_shape = (batch_size, 1, 3, 518, 518)  # F=1 für Einzelbilder
#                    (B,        F, C, H,   W)

# Frame-wise Attention wird zu:
# - Intra-Pizza spatial relationships (Beläge, Kruste, Verbrennungen)
# Global Attention wird zu:
# - Spatial consistency check innerhalb des Bildes
# - Multi-scale feature aggregation
```

**Erwartete Verbesserungen:**
1. **Präzisere Verbrennungsgrad-Lokalisierung:** 3D Spatial Features erkennen subtle Oberflächenvariationen
2. **Bessere Belag-Struktur-Erkennung:** Höheninformationen für 3D Belag-Analyse
3. **Robustere Feature-Extraktion:** Dual-Encoder reduziert False Positives durch räumliche Konsistenz
4. **Multi-Resolution Analysis:** Verschiedene spatiale Skalen für Fine-Grained Klassifikation

---

## 8. Tensor-Shape Spezifikationen

### 8.1 Input Tensors
```python
# Media Input (Pizza-Bilder)
media_input: torch.Tensor
    Shape: (B, F, C, H, W)
    - B: Batch size (z.B. 8)
    - F: 1 (für statische Pizza-Bilder)
    - C: 3 (RGB)
    - H, W: 518, 518 (Standard VGGT Resolution)
    Dtype: torch.float32
    Range: [0.0, 1.0] (normalisiert)

# Text Input (Klassifikations-Prompt)
input_ids: torch.LongTensor
    Shape: (B, sequence_length)
    - B: Batch size
    - sequence_length: Variable (abhängig von Prompt-Länge)
```

### 8.2 Intermediate Tensors
```python
# VGGT Aggregator Output
aggregated_tokens_list: List[torch.Tensor]
    Each tensor shape: (B, F, N_tokens, embed_dim)
    - N_tokens: (H//patch_size) * (W//patch_size) ≈ 37*37 = 1369 tokens
    - embed_dim: 1024 (VGGT default)

# Feature Merger Output  
vgg_embeds: torch.Tensor
    Shape: (S, D)
    - S: B * F * N_tokens (flattened sequence)
    - D: config.hidden_size (LLM dimension, z.B. 4096)
```

### 8.3 Output Tensors
```python
# Pizza Classification Logits
logits: torch.Tensor
    Shape: (B, sequence_length, vocab_size)
    - B: Batch size
    - sequence_length: Input + generated tokens
    - vocab_size: 151,936 (Qwen2.5-VL vocabulary)

# Generated Classification
generated_ids: torch.LongTensor  
    Shape: (B, max_new_tokens)
    Content: Token-IDs für "basic" oder "burnt" Klassifikation
```

---

## 9. Memory und Performance Charakteristika

### 9.1 Modell-Parameter
```python
model_stats = {
    "total_parameters": "~8B parameters",
    "vggt_parameters": "~1B parameters",  
    "qwen2_5_vl_parameters": "~7B parameters",
    "embedding_dimensions": {
        "vggt_embed_dim": 1024,
        "llm_hidden_size": 4096,
        "vocab_size": 151936
    }
}
```

### 9.2 Hardware-Anforderungen
```python
hardware_requirements = {
    "minimum": {
        "gpu_memory": "8GB VRAM",
        "system_ram": "16GB",
        "gpu_compute": "CUDA Capability >= 7.0"
    },
    "recommended": {
        "gpu_memory": "24GB VRAM (RTX 4090)",
        "system_ram": "32GB", 
        "gpu_compute": "CUDA Capability >= 8.0"
    },
    "optimal": {
        "gpu_memory": "80GB VRAM (A100)",
        "system_ram": "128GB",
        "multi_gpu": "2-4 GPUs für Batch Processing"
    }
}
```

---

## 10. Einschränkungen und Überlegungen für Pizza-Projekt

### 10.1 Technische Herausforderungen
1. **Compute-Intensität:** 8B Parameter vs. aktuelles leichtgewichtiges Pizza-Modell
2. **Memory Footprint:** Dual-Encoder benötigt signifikant mehr GPU-Memory
3. **Inference-Latenz:** Komplexere Pipeline = langsamere Verarbeitung
4. **Edge-Deployment:** Sehr wahrscheinlich NICHT RP2040-kompatibel

### 10.2 Integrations-Ansätze
```python
integration_strategies = {
    "full_integration": {
        "pros": "Maximale spatial intelligence",
        "cons": "Hohe Hardware-Anforderungen"
    },
    "selective_components": {
        "pros": "Nutzt nur VGGT Spatial Encoder",
        "cons": "Verliert LLM-Reasoning Capabilities"
    },
    "hybrid_approach": {
        "pros": "Kombiniert bestehendes + spatial features",
        "cons": "Komplexe Integration erforderlich"
    },
    "feature_extraction_only": {
        "pros": "Extrahiert nur spatial features für bestehende Pipeline", 
        "cons": "Reduzierte spatial intelligence"
    }
}
```

---

## 11. Nächste Schritte für SPATIAL-1.3

Basierend auf dieser Architektur-Analyse sind die empfohlenen nächsten Schritte:

1. **Pretrained Model Download:** Lade `Diankun/Spatial-MLLM-subset-sft` von Hugging Face
2. **Single Image Testing:** Teste VGGT mit einzelnen Pizza-Bildern (F=1)
3. **Feature Extraction:** Implementiere räumliche Feature-Extraktion für Pizza-spezifische Merkmale
4. **Performance Baseline:** Messe Inference-Zeiten und Memory-Nutzung
5. **Selective Integration Planning:** Plane welche Komponenten für Pizza-Klassifikation am relevantesten sind

---

## 12. Fazit

**Spatial-MLLM bietet revolutionary spatial intelligence für Computer Vision**, aber die Integration in das Pizza-Projekt erfordert sorgfältige Abwägung zwischen **Performance-Gewinn und Hardware-Anforderungen**.

**Architektur-Stärken für Pizza-Klassifikation:**
- ✅ 3D räumliche Feature-Extraktion ideal für Oberflächenanalyse
- ✅ Dual-Encoder Robustheit gegen Beleuchtungsvariationen  
- ✅ Multi-Scale Spatial Attention für Fine-Grained Verbrennungsgrad-Erkennung

**Integration-Herausforderungen:**
- ⚠️ 8B Parameter vs. aktuelle lightweight Lösung
- ⚠️ Hohe GPU-Memory Anforderungen (8-24GB VRAM)
- ⚠️ Inference-Latenz für Echtzeit-Anwendungen
- ⚠️ RP2040 Edge-Deployment höchst unwahrscheinlich

**Empfehlung:** Schrittweise Integration mit Fokus auf **räumliche Feature-Extraktion** statt Full-Model Replacement.
