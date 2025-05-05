/**
 * @file pizza_clahe.c
 * @brief Implementierung der CLAHE-Bildvorverarbeitung für RP2040
 * 
 * Diese Implementierung ist speziell für den ressourcenbeschränkten RP2040
 * optimiert und verwendet In-Place-Verarbeitung, wo möglich, um den
 * Speicherverbrauch zu minimieren. Die Implementierung ist ein vereinfachter
 * Histogramm-Ausgleich, der auch unter schwierigen Beleuchtungsbedingungen
 * funktioniert, aber deutlich weniger Speicher benötigt als die vollständige
 * OpenCV-Implementierung.
 */

#include "pizza_clahe.h"
#include <string.h>  // für memcpy

// Standardparameter
#define DEFAULT_CLIP_LIMIT 4
#define DEFAULT_GRID_SIZE 8
#define MAX_GRID_SIZE 16

// Makros für RGB zu Grauwert und zurück
#define RGB_TO_Y(r, g, b) ((uint8_t)(0.299f * (r) + 0.587f * (g) + 0.114f * (b)))
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

// Fehlercodes
#define CLAHE_SUCCESS 0
#define CLAHE_ERROR_INVALID_PARAM -1
#define CLAHE_ERROR_MEMORY -2

/**
 * @brief Initialisiert die CLAHE-Konfiguration mit Standardwerten
 */
void pizza_clahe_init(PizzaClaheConfig* config, uint16_t width, uint16_t height) {
    if (!config) return;
    
    config->clip_limit = DEFAULT_CLIP_LIMIT;
    config->grid_size = DEFAULT_GRID_SIZE;
    config->image_width = width;
    config->image_height = height;
}

/**
 * @brief Interne Hilfsfunktion: Erstellt das Histogramm für eine Bildregion
 */
static void create_histogram(
    const uint8_t* src,
    uint32_t* hist,
    uint16_t x_start, uint16_t y_start,
    uint16_t width, uint16_t height,
    uint16_t src_stride
) {
    // Histogramm zurücksetzen
    memset(hist, 0, 256 * sizeof(uint32_t));
    
    // Histogramm aus der Region berechnen
    for (uint16_t y = y_start; y < y_start + height; y++) {
        for (uint16_t x = x_start; x < x_start + width; x++) {
            uint16_t pixel_idx = y * src_stride + x;
            hist[src[pixel_idx]]++;
        }
    }
}

/**
 * @brief Interne Hilfsfunktion: Wendet Clip-Limit auf das Histogramm an
 */
static void apply_clip_limit(uint32_t* hist, uint8_t clip_limit, uint32_t num_pixels) {
    // Konvertiere clip_limit zu einem absoluten Wert, basierend auf der Anzahl der Pixel
    uint32_t abs_clip_limit = (uint32_t)clip_limit * num_pixels / 100;
    if (abs_clip_limit < 1) abs_clip_limit = 1;
    
    // Zähle, wie viele Pixel über dem Limit liegen
    uint32_t excess = 0;
    for (int i = 0; i < 256; i++) {
        if (hist[i] > abs_clip_limit) {
            excess += hist[i] - abs_clip_limit;
            hist[i] = abs_clip_limit;
        }
    }
    
    // Verteile die überschüssigen Pixel gleichmäßig
    uint32_t redistribution = excess / 256;
    uint32_t mod = excess % 256;
    
    for (int i = 0; i < 256; i++) {
        hist[i] += redistribution;
        
        // Verteile den Rest
        if (mod > 0) {
            hist[i]++;
            mod--;
        }
    }
}

/**
 * @brief Interne Hilfsfunktion: Erstellt eine Lookup-Tabelle (CDF) für das Histogramm
 */
static void create_lookup_table(const uint32_t* hist, uint8_t* lut, uint32_t num_pixels) {
    // Erstelle die kumulative Verteilung
    uint32_t cdf[256] = {0};
    cdf[0] = hist[0];
    
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + hist[i];
    }
    
    // Normalisiere die kumulative Verteilung auf [0, 255]
    for (int i = 0; i < 256; i++) {
        // Vermeide Division durch Null
        if (num_pixels > 0) {
            // Vermeidet Überlauf durch Verwendung von 32-Bit-Arithmetik
            lut[i] = (uint8_t)((cdf[i] * 255) / num_pixels);
        } else {
            lut[i] = (uint8_t)i;
        }
    }
}

/**
 * @brief Intern: CLAHE auf ein Graustufenbild anwenden
 */
static int apply_clahe_grayscale_internal(
    const uint8_t* src,
    uint8_t* dst,
    uint16_t width,
    uint16_t height,
    uint8_t clip_limit,
    uint8_t grid_size
) {
    // Validierung
    if (!src || !dst || width == 0 || height == 0 || grid_size == 0) {
        return CLAHE_ERROR_INVALID_PARAM;
    }
    
    // Begrenze Grid-Größe
    if (grid_size > MAX_GRID_SIZE) grid_size = MAX_GRID_SIZE;
    
    // Berechne Gittergrößen
    uint16_t grid_width = width / grid_size;
    uint16_t grid_height = height / grid_size;
    
    // Stelle sicher, dass die Gitter mindestens 1 Pixel groß sind
    if (grid_width < 1) grid_width = 1;
    if (grid_height < 1) grid_height = 1;
    
    // Arbeitsspeicher für Histogramme und Lookup-Tabellen
    uint32_t hist[256];
    uint8_t lut[256];
    
    // Verarbeite jedes Gitter
    for (uint8_t grid_y = 0; grid_y < grid_size; grid_y++) {
        for (uint8_t grid_x = 0; grid_x < grid_size; grid_x++) {
            // Berechne die Grenzen des aktuellen Gitters
            uint16_t x_start = grid_x * grid_width;
            uint16_t y_start = grid_y * grid_height;
            uint16_t x_end = (grid_x + 1) * grid_width;
            uint16_t y_end = (grid_y + 1) * grid_height;
            
            // Berücksichtige den Bildrand
            if (x_end > width) x_end = width;
            if (y_end > height) y_end = height;
            
            uint16_t region_width = x_end - x_start;
            uint16_t region_height = y_end - y_start;
            uint32_t num_pixels = region_width * region_height;
            
            // Erstelle Histogramm für das aktuelle Gitter
            create_histogram(src, hist, x_start, y_start, region_width, region_height, width);
            
            // Clip Limit anwenden
            apply_clip_limit(hist, clip_limit, num_pixels);
            
            // Lookup-Tabelle erstellen
            create_lookup_table(hist, lut, num_pixels);
            
            // Wende Lookup-Tabelle auf die Region an
            for (uint16_t y = y_start; y < y_end; y++) {
                for (uint16_t x = x_start; x < x_end; x++) {
                    uint32_t idx = y * width + x;
                    dst[idx] = lut[src[idx]];
                }
            }
        }
    }
    
    return CLAHE_SUCCESS;
}

/**
 * @brief Wendet CLAHE auf ein Graustufenbild an
 */
int pizza_clahe_apply_grayscale(uint8_t* input, uint8_t* output, const PizzaClaheConfig* config) {
    if (!input || !output || !config) {
        return CLAHE_ERROR_INVALID_PARAM;
    }
    
    return apply_clahe_grayscale_internal(
        input, output, 
        config->image_width, config->image_height,
        config->clip_limit, config->grid_size
    );
}

/**
 * @brief Hilfsfunktion: RGB zu YUV-Konvertierung (nur Y-Kanal)
 */
static uint8_t rgb_to_y(uint8_t r, uint8_t g, uint8_t b) {
    // Y = 0.299R + 0.587G + 0.114B
    // Fixed-point Arithmetik: (77*R + 150*G + 29*B) / 256
    return (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
}

/**
 * @brief Hilfsfunktion: YUV zu RGB-Konvertierung (nur Y ändert sich)
 */
static void y_to_rgb(uint8_t y_new, uint8_t r_old, uint8_t g_old, uint8_t b_old, 
                     uint8_t* r_new, uint8_t* g_new, uint8_t* b_new) {
    // Berechne den ursprünglichen Y-Wert
    uint8_t y_old = rgb_to_y(r_old, g_old, b_old);
    
    // Wenn der alte Y-Wert 0 ist, setze die RGB-Werte einfach auf den neuen Y-Wert
    if (y_old == 0) {
        *r_new = y_new;
        *g_new = y_new;
        *b_new = y_new;
        return;
    }
    
    // Ansonsten skaliere die RGB-Werte proportional
    float scale = (float)y_new / y_old;
    *r_new = CLAMP((int)(r_old * scale), 0, 255);
    *g_new = CLAMP((int)(g_old * scale), 0, 255);
    *b_new = CLAMP((int)(b_old * scale), 0, 255);
}

/**
 * @brief Wendet CLAHE auf ein RGB-Bild an, indem nur der Y-Kanal bearbeitet wird
 */
int pizza_clahe_apply_rgb(uint8_t* input, uint8_t* output, const PizzaClaheConfig* config) {
    if (!input || !output || !config) {
        return CLAHE_ERROR_INVALID_PARAM;
    }
    
    uint16_t width = config->image_width;
    uint16_t height = config->image_height;
    uint32_t num_pixels = width * height;
    
    // Temporärer Speicher für Y-Kanal
    uint8_t* y_channel = (uint8_t*)malloc(num_pixels);
    if (!y_channel) {
        return CLAHE_ERROR_MEMORY;
    }
    
    // Extrahiere Y-Kanal
    for (uint32_t i = 0; i < num_pixels; i++) {
        uint32_t idx = i * 3;
        y_channel[i] = rgb_to_y(input[idx], input[idx+1], input[idx+2]);
    }
    
    // Wende CLAHE auf Y-Kanal an
    int result = apply_clahe_grayscale_internal(
        y_channel, y_channel,
        width, height,
        config->clip_limit, config->grid_size
    );
    
    if (result != CLAHE_SUCCESS) {
        free(y_channel);
        return result;
    }
    
    // Wende korrigierten Y-Kanal auf das RGB-Bild an
    for (uint32_t i = 0; i < num_pixels; i++) {
        uint32_t idx = i * 3;
        uint8_t r_new, g_new, b_new;
        y_to_rgb(
            y_channel[i],
            input[idx], input[idx+1], input[idx+2],
            &r_new, &g_new, &b_new
        );
        
        output[idx] = r_new;
        output[idx+1] = g_new;
        output[idx+2] = b_new;
    }
    
    free(y_channel);
    return CLAHE_SUCCESS;
}

/**
 * @brief Berechnet den erforderlichen Arbeitsspeicher für die CLAHE-Verarbeitung
 */
uint32_t pizza_clahe_get_required_memory(const PizzaClaheConfig* config) {
    if (!config) return 0;
    
    // Grundlegender Speicherbedarf: Histogramm und Lookup-Tabelle
    uint32_t basic_memory = 256 * sizeof(uint32_t) + 256 * sizeof(uint8_t);
    
    // Zusätzlicher Speicher für RGB-Modus: Y-Kanal
    uint32_t rgb_memory = config->image_width * config->image_height;
    
    // Gebe den Maximalbedarf zurück (für RGB-Modus)
    return basic_memory + rgb_memory;
}

/**
 * @brief Gibt Speichernutzung und Verarbeitungszeit für CLAHE zurück
 */
int pizza_clahe_get_resource_usage(
    const PizzaClaheConfig* config,
    uint32_t* peak_ram_bytes,
    uint32_t* flash_usage_bytes,
    uint32_t* estimated_cycles
) {
    if (!config || !peak_ram_bytes || !flash_usage_bytes || !estimated_cycles) {
        return CLAHE_ERROR_INVALID_PARAM;
    }
    
    // Geschätzte Flash-Nutzung (Konstante für den Code)
    *flash_usage_bytes = 2048;  // ~2KB Code
    
    // Maximaler RAM-Verbrauch
    *peak_ram_bytes = pizza_clahe_get_required_memory(config);
    
    // Geschätzte Zyklen basierend auf Bildgröße und Grid-Größe
    uint32_t num_pixels = config->image_width * config->image_height;
    uint32_t grid_ops = config->grid_size * config->grid_size;
    
    // Grobe Schätzung: 20 Zyklen pro Pixel für Hauptoperationen und 
    // 1000 Zyklen pro Grid für Histogramm-Berechnungen
    *estimated_cycles = num_pixels * 20 + grid_ops * 1000;
    
    return CLAHE_SUCCESS;
}