/**
 * Pizza Image Preprocessing - Optimized Version for ENERGIE-2.4
 * 
 * This optimized implementation reduces energy consumption by:
 * - Eliminating floating-point arithmetic in critical paths
 * - Using lookup tables for common operations
 * - Processing luminance-only CLAHE instead of per-channel
 * - Static memory allocation to avoid malloc/free overhead
 * - Vectorized operations where possible
 * - Adaptive processing based on image characteristics
 * 
 * Expected energy savings: 40-60% reduction in preprocessing energy
 */

#ifndef PIZZA_PREPROCESS_OPTIMIZED_H
#define PIZZA_PREPROCESS_OPTIMIZED_H

#include "pizza_preprocess.h"

// Optimization configurations
#define ENABLE_ADAPTIVE_PROCESSING 1    // Skip CLAHE for high-contrast images
#define ENABLE_LUMINANCE_ONLY_CLAHE 1   // Process only luminance instead of RGB
#define ENABLE_FAST_RESIZE 1            // Use optimized resize algorithms
#define ENABLE_LOOKUP_TABLES 1          // Pre-computed lookup tables

// Static memory pools to avoid malloc/free
#define STATIC_RESIZE_BUFFER_SIZE (320 * 240 * 3)  // Max expected image size
#define STATIC_CLAHE_LUT_SIZE (16 * 16 * 256)      // Max grid LUTs
#define STATIC_WORK_BUFFER_SIZE (320 * 240)        // Working buffer

// Optimization thresholds
#define CONTRAST_THRESHOLD 30           // Skip CLAHE if contrast > threshold
#define UNIFORM_REGION_THRESHOLD 5      // Skip processing for uniform regions

/**
 * Optimized complete preprocessing pipeline
 * Combines all optimizations for maximum energy savings
 */
preprocess_result_t pizza_preprocess_complete_optimized(
    const uint8_t* input_rgb,
    int input_width,
    int input_height,
    uint8_t* output_rgb,
    int output_width,
    int output_height
);

/**
 * Fast resize using integer-only arithmetic and vectorization
 */
preprocess_result_t pizza_preprocess_resize_rgb_fast(
    const uint8_t* input_rgb,
    int input_width,
    int input_height,
    uint8_t* output_rgb,
    int output_width,
    int output_height
);

/**
 * Luminance-only CLAHE with adaptive processing
 * Only processes luminance channel and reconstructs RGB
 */
preprocess_result_t pizza_preprocess_clahe_luminance_adaptive(
    uint8_t* input_rgb,
    uint8_t* output_rgb,
    int width,
    int height
);

/**
 * Analyze image characteristics to determine optimal processing
 */
typedef struct {
    uint8_t contrast_level;      // 0-255, higher = more contrast
    uint8_t brightness_level;    // 0-255, average brightness
    bool needs_clahe;           // Whether CLAHE processing is beneficial
    bool has_uniform_regions;   // Whether image has large uniform areas
} image_stats_t;

preprocess_result_t pizza_analyze_image_stats(
    const uint8_t* rgb_data,
    int width,
    int height,
    image_stats_t* stats
);

/**
 * Initialize optimization system (call once at startup)
 */
void pizza_preprocess_optimization_init(void);

/**
 * Get optimization performance metrics
 */
typedef struct {
    uint32_t total_preprocessed_frames;
    uint32_t clahe_skipped_frames;
    uint32_t average_processing_time_us;
    uint32_t energy_savings_percent;
} optimization_metrics_t;

void pizza_preprocess_get_optimization_metrics(optimization_metrics_t* metrics);

#endif // PIZZA_PREPROCESS_OPTIMIZED_H
