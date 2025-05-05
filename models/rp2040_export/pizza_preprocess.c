/**
 * Pizza Image Preprocessing Implementation
 * Memory-efficient preprocessing algorithms for RP2040
 */

#include "pizza_preprocess.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Static buffer to avoid heap allocations
// Note: this limits the max grid size but reduces memory fragmentation
#define MAX_GRID_SIZE 16
#define MAX_HISTOGRAM_BINS 256

// Memory-efficient histogram for grayscale images (8-bit)
typedef struct {
    uint32_t bins[MAX_HISTOGRAM_BINS]; // Histogram bins
    float clip_limit;                  // Maximum bin value (for contrast limiting)
    int total_pixels;                  // Total number of pixels in region
} histogram_t;

// Local function prototypes
static void init_histogram(histogram_t *hist, int region_pixels, float clip_limit);
static void add_to_histogram(histogram_t *hist, uint8_t *img, int x_start, int y_start,
                            int width, int height, int img_width, int img_stride);
static void clip_histogram(histogram_t *hist);
static void create_lut(histogram_t *hist, uint8_t *lut);
static uint8_t interpolate_pixel(int x, int y, int width, int height, 
                               int grid_size, uint8_t *luts, uint8_t pixel_value);

// Static buffer for resize operations to avoid frequent malloc/free
// This is shared between functions and carefully managed to avoid conflicts
static uint8_t* s_resize_buffer = NULL;
static size_t s_resize_buffer_size = 0;

/**
 * Resizes an RGB image to a different resolution
 * Uses nearest neighbor or bilinear interpolation (faster for RP2040)
 */
preprocess_result_t pizza_preprocess_resize_rgb(
    const uint8_t* input_rgb,
    int input_width,
    int input_height,
    uint8_t* output_rgb,
    int output_width,
    int output_height,
    bool use_bilinear)
{
    if (!input_rgb || !output_rgb) {
        return PREPROCESS_ERROR_NULL_POINTER;
    }
    
    if (input_width <= 0 || input_height <= 0 || 
        output_width <= 0 || output_height <= 0) {
        return PREPROCESS_ERROR_INVALID_PARAMS;
    }
    
    // Scale factors as fixed point values for better performance
    // We use 8.8 fixed point (8 bits for integer, 8 bits for fraction)
    const int x_ratio = (int)((input_width << 8) / output_width);
    const int y_ratio = (int)((input_height << 8) / output_height);
    
    if (use_bilinear) {
        // Bilinear interpolation
        for (int y = 0; y < output_height; y++) {
            int y_in = ((y * y_ratio) >> 8);
            int y_diff = (y * y_ratio) & 0xFF; // Fractional part
            if (y_in >= input_height - 1) y_in = input_height - 2;
            
            for (int x = 0; x < output_width; x++) {
                int x_in = ((x * x_ratio) >> 8);
                int x_diff = (x * x_ratio) & 0xFF; // Fractional part
                if (x_in >= input_width - 1) x_in = input_width - 2;
                
                // Get the four neighboring pixels
                int index00 = ((y_in) * input_width + x_in) * 3;
                int index01 = index00 + 3; // pixel to the right
                int index10 = index00 + input_width * 3; // pixel below
                int index11 = index10 + 3; // pixel below and to the right
                
                // Calculate output pixel for each channel
                for (int c = 0; c < 3; c++) {
                    int p00 = input_rgb[index00 + c];
                    int p01 = input_rgb[index01 + c];
                    int p10 = input_rgb[index10 + c];
                    int p11 = input_rgb[index11 + c];
                    
                    // Fixed-point interpolation
                    int top = (p00 * (256 - x_diff) + p01 * x_diff) >> 8;
                    int bottom = (p10 * (256 - x_diff) + p11 * x_diff) >> 8;
                    int pixel = (top * (256 - y_diff) + bottom * y_diff) >> 8;
                    
                    // Write to output
                    output_rgb[(y * output_width + x) * 3 + c] = (uint8_t)pixel;
                }
            }
        }
    } else {
        // Nearest neighbor (faster but lower quality)
        for (int y = 0; y < output_height; y++) {
            int y_in = ((y * y_ratio) >> 8);
            if (y_in >= input_height) y_in = input_height - 1;
            
            for (int x = 0; x < output_width; x++) {
                int x_in = ((x * x_ratio) >> 8);
                if (x_in >= input_width) x_in = input_width - 1;
                
                // Copy RGB values
                int in_idx = (y_in * input_width + x_in) * 3;
                int out_idx = (y * output_width + x) * 3;
                
                output_rgb[out_idx]     = input_rgb[in_idx];
                output_rgb[out_idx + 1] = input_rgb[in_idx + 1];
                output_rgb[out_idx + 2] = input_rgb[in_idx + 2];
            }
        }
    }
    
    return PREPROCESS_OK;
}

/**
 * Converts RGB image to grayscale using luminance formula:
 * Y = 0.299*R + 0.587*G + 0.114*B
 */
preprocess_result_t pizza_preprocess_rgb_to_gray(
    const uint8_t* input_rgb,
    uint8_t* output_gray,
    int width,
    int height)
{
    if (!input_rgb || !output_gray) {
        return PREPROCESS_ERROR_NULL_POINTER;
    }
    
    if (width <= 0 || height <= 0) {
        return PREPROCESS_ERROR_INVALID_PARAMS;
    }
    
    // Fixed-point arithmetic for better performance on RP2040
    // We use 8.8 fixed-point: 8 bits for integer, 8 bits for fraction
    const int r_weight = 77;  // 0.299 * 256
    const int g_weight = 150; // 0.587 * 256
    const int b_weight = 29;  // 0.114 * 256
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index_rgb = (y * width + x) * 3;
            int index_gray = y * width + x;
            
            // Get RGB values
            int r = input_rgb[index_rgb];
            int g = input_rgb[index_rgb + 1];
            int b = input_rgb[index_rgb + 2];
            
            // Calculate grayscale value using fixed-point arithmetic
            // and round to nearest integer
            int gray = (r_weight * r + g_weight * g + b_weight * b + 128) >> 8;
            
            // Ensure value is in valid range 0-255
            output_gray[index_gray] = (uint8_t)(gray > 255 ? 255 : (gray < 0 ? 0 : gray));
        }
    }
    
    return PREPROCESS_OK;
}

/**
 * Complete image preprocessing: resize + enhance lighting
 * Combines all preprocessing steps needed for the pizza detection model
 */
preprocess_result_t pizza_preprocess_complete(
    const uint8_t* input_rgb,
    int input_width,
    int input_height,
    uint8_t* output_rgb,
    int output_width,
    int output_height)
{
    if (!input_rgb || !output_rgb) {
        return PREPROCESS_ERROR_NULL_POINTER;
    }
    
    if (input_width <= 0 || input_height <= 0 || 
        output_width <= 0 || output_height <= 0) {
        return PREPROCESS_ERROR_INVALID_PARAMS;
    }
    
    // Allocate a temporary buffer for the resize operation
    size_t resize_buffer_size = output_width * output_height * 3;
    uint8_t* resize_buffer;
    
    // Try to reuse existing buffer if possible to save memory
    bool allocated_buffer = false;
    if (s_resize_buffer != NULL && s_resize_buffer_size >= resize_buffer_size) {
        // Reuse existing buffer
        resize_buffer = s_resize_buffer;
    } else {
        // Allocate new buffer
        resize_buffer = (uint8_t*)malloc(resize_buffer_size);
        if (!resize_buffer) {
            return PREPROCESS_ERROR_MEMORY;
        }
        allocated_buffer = true;
        
        // Store for potential reuse
        if (s_resize_buffer != NULL) {
            free(s_resize_buffer);
        }
        s_resize_buffer = resize_buffer;
        s_resize_buffer_size = resize_buffer_size;
    }
    
    // Step 1: Resize image
    // Use nearest neighbor for speed when downsampling significantly
    bool use_bilinear = (output_width > input_width / 4) && (output_height > input_height / 4);
    
    preprocess_result_t result = pizza_preprocess_resize_rgb(
        input_rgb, input_width, input_height,
        resize_buffer, output_width, output_height,
        use_bilinear
    );
    
    if (result != PREPROCESS_OK) {
        if (allocated_buffer) {
            free(resize_buffer);
            s_resize_buffer = NULL;
            s_resize_buffer_size = 0;
        }
        return result;
    }
    
    // Step 2: Apply CLAHE for lighting enhancement
    result = pizza_preprocess_enhance_lighting(
        resize_buffer, output_rgb, 
        output_width, output_height
    );
    
    // Buffer is managed globally now, don't free it here
    
    return result;
}

/**
 * Performs contrast-limited adaptive histogram equalization on a grayscale image.
 * This implementation is optimized for low memory usage and efficiency on RP2040.
 */
preprocess_result_t pizza_preprocess_clahe_gray(
    uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    float clip_limit,
    int grid_size)
{
    if (!input || !output) {
        return PREPROCESS_ERROR_NULL_POINTER;
    }
    
    if (width <= 0 || height <= 0 || grid_size <= 0 || grid_size > MAX_GRID_SIZE) {
        return PREPROCESS_ERROR_INVALID_PARAMS;
    }
    
    // Allocate memory for grid of lookup tables
    uint8_t *grid_luts = (uint8_t*)malloc(grid_size * grid_size * 256 * sizeof(uint8_t));
    if (!grid_luts) {
        return PREPROCESS_ERROR_MEMORY;
    }
    
    // Initialize local variables
    int region_width = width / grid_size;
    int region_height = height / grid_size;
    histogram_t hist;
    
    // Create histogram and transformation function for each region
    for (int region_y = 0; region_y < grid_size; region_y++) {
        for (int region_x = 0; region_x < grid_size; region_x++) {
            // Calculate region boundaries
            int x_start = region_x * region_width;
            int y_start = region_y * region_height;
            int region_width_actual = (region_x == grid_size - 1) ? 
                                      (width - x_start) : region_width;
            int region_height_actual = (region_y == grid_size - 1) ? 
                                      (height - y_start) : region_height;
            
            // Initialize histogram
            int region_pixels = region_width_actual * region_height_actual;
            init_histogram(&hist, region_pixels, clip_limit);
            
            // Collect histogram for this region
            add_to_histogram(&hist, input, x_start, y_start, 
                            region_width_actual, region_height_actual, 
                            width, width);
            
            // Apply contrast limiting if needed
            if (clip_limit > 1.0f) {
                clip_histogram(&hist);
            }
            
            // Create lookup table for this region
            uint8_t *lut = grid_luts + (region_y * grid_size + region_x) * 256;
            create_lut(&hist, lut);
        }
    }
    
    // Apply transformation to image using bilinear interpolation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            uint8_t pixel_value = input[index];
            
            // Apply interpolation to determine output value
            output[index] = interpolate_pixel(x, y, width, height, 
                                           grid_size, grid_luts, pixel_value);
        }
    }
    
    // Free allocated memory
    free(grid_luts);
    
    return PREPROCESS_OK;
}

/**
 * Performs CLAHE on RGB image by processing each channel separately
 */
preprocess_result_t pizza_preprocess_clahe_rgb(
    uint8_t* input_rgb,
    uint8_t* output_rgb,
    int width,
    int height,
    float clip_limit,
    int grid_size)
{
    if (!input_rgb || !output_rgb) {
        return PREPROCESS_ERROR_NULL_POINTER;
    }
    
    if (width <= 0 || height <= 0) {
        return PREPROCESS_ERROR_INVALID_PARAMS;
    }
    
    // Allocate temporary buffers for each channel
    uint8_t *channel = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    if (!channel) {
        return PREPROCESS_ERROR_MEMORY;
    }
    
    // Process each channel separately
    for (int c = 0; c < 3; c++) {
        // Extract channel
        for (int i = 0; i < width * height; i++) {
            channel[i] = input_rgb[i * 3 + c];
        }
        
        // Apply CLAHE to this channel
        preprocess_result_t result = pizza_preprocess_clahe_gray(
            channel, channel, width, height, clip_limit, grid_size);
        
        if (result != PREPROCESS_OK) {
            free(channel);
            return result;
        }
        
        // Put processed channel back
        for (int i = 0; i < width * height; i++) {
            output_rgb[i * 3 + c] = channel[i];
        }
    }
    
    free(channel);
    return PREPROCESS_OK;
}

/**
 * Simplified interface using default parameters optimized for pizza recognition
 */
preprocess_result_t pizza_preprocess_enhance_lighting(
    uint8_t* input_rgb,
    uint8_t* output_rgb,
    int width,
    int height)
{
    // Use predefined parameters optimized for pizza detection
    return pizza_preprocess_clahe_rgb(
        input_rgb, 
        output_rgb, 
        width, 
        height, 
        CLAHE_CLIP_LIMIT, 
        CLAHE_GRID_SIZE
    );
}

/**
 * Analyze memory usage of preprocessing operations
 */
uint32_t pizza_preprocess_get_memory_usage(
    int image_width,
    int image_height,
    int target_width,
    int target_height)
{
    uint32_t total_bytes = 0;
    
    // Resize buffer (RGB)
    uint32_t resize_buffer = target_width * target_height * 3;
    total_bytes += resize_buffer;
    
    // CLAHE working memory
    uint32_t clahe_grid_luts = CLAHE_GRID_SIZE * CLAHE_GRID_SIZE * 256;
    uint32_t clahe_channel_buffer = target_width * target_height;
    total_bytes += clahe_grid_luts + clahe_channel_buffer;
    
    // Histogram structure
    total_bytes += sizeof(histogram_t);
    
    return total_bytes;
}

/* ---- Local helper functions ---- */

/**
 * Initialize histogram data structure
 */
static void init_histogram(histogram_t *hist, int region_pixels, float clip_limit) {
    memset(hist->bins, 0, sizeof(hist->bins));
    hist->total_pixels = region_pixels;
    
    // Calculate actual clip limit based on region size
    if (clip_limit > 1.0f) {
        // Convert relative clip limit to absolute bin limit
        float normalized_clip_limit = clip_limit * region_pixels / 256.0f;
        hist->clip_limit = normalized_clip_limit;
    } else {
        // No clipping
        hist->clip_limit = region_pixels;
    }
}

/**
 * Add pixel values from a region to the histogram
 */
static void add_to_histogram(histogram_t *hist, uint8_t *img, int x_start, int y_start,
                           int width, int height, int img_width, int img_stride) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = (y_start + y) * img_stride + (x_start + x);
            uint8_t pixel_value = img[pixel_index];
            hist->bins[pixel_value]++;
        }
    }
}

/**
 * Apply contrast limiting to histogram by clipping and redistributing excess
 */
static void clip_histogram(histogram_t *hist) {
    int excess = 0;
    uint32_t clip_limit = (uint32_t)hist->clip_limit;
    
    // Calculate total excess
    for (int i = 0; i < 256; i++) {
        if (hist->bins[i] > clip_limit) {
            excess += hist->bins[i] - clip_limit;
            hist->bins[i] = clip_limit;
        }
    }
    
    // Redistribute excess evenly
    uint32_t redistBatch = excess / 256;
    uint32_t residual = excess - redistBatch * 256;
    
    if (redistBatch > 0) {
        for (int i = 0; i < 256; i++) {
            hist->bins[i] += redistBatch;
        }
    }
    
    // Handle residual (add one to first 'residual' bins)
    if (residual > 0) {
        for (int i = 0; i < 256 && residual > 0; i++) {
            if (hist->bins[i] < clip_limit) {
                hist->bins[i]++;
                residual--;
            }
        }
    }
}

/**
 * Create lookup table from histogram
 */
static void create_lut(histogram_t *hist, uint8_t *lut) {
    float scale = 255.0f / hist->total_pixels;
    int sum = 0;
    
    // Create cumulative histogram and scale to 0-255
    for (int i = 0; i < 256; i++) {
        sum += hist->bins[i];
        float mapped_val = sum * scale;
        // Clip to valid range and round to nearest integer
        lut[i] = (uint8_t)(mapped_val > 255.0f ? 255 : (mapped_val < 0.0f ? 0 : (uint8_t)(mapped_val + 0.5f)));
    }
}

/**
 * Interpolate pixel value based on surrounding grid points
 */
static uint8_t interpolate_pixel(int x, int y, int width, int height, 
                               int grid_size, uint8_t *luts, uint8_t pixel_value) {
    float region_width = (float)width / grid_size;
    float region_height = (float)height / grid_size;
    
    // Calculate grid coordinates and interpolation factors
    float grid_x = x / region_width;
    float grid_y = y / region_height;
    
    int grid_x0 = (int)grid_x;
    int grid_y0 = (int)grid_y;
    int grid_x1 = (grid_x0 + 1 < grid_size) ? grid_x0 + 1 : grid_x0;
    int grid_y1 = (grid_y0 + 1 < grid_size) ? grid_y0 + 1 : grid_y0;
    
    float x_factor = grid_x - grid_x0;
    float y_factor = grid_y - grid_y0;
    
    // Get values from lookup tables at four surrounding grid points
    uint8_t val_00 = luts[(grid_y0 * grid_size + grid_x0) * 256 + pixel_value];
    uint8_t val_01 = luts[(grid_y0 * grid_size + grid_x1) * 256 + pixel_value];
    uint8_t val_10 = luts[(grid_y1 * grid_size + grid_x0) * 256 + pixel_value];
    uint8_t val_11 = luts[(grid_y1 * grid_size + grid_x1) * 256 + pixel_value];
    
    // Perform bilinear interpolation
    float val = (1 - y_factor) * ((1 - x_factor) * val_00 + x_factor * val_01) +
               y_factor * ((1 - x_factor) * val_10 + x_factor * val_11);
    
    // Round to nearest integer and clip to valid range
    return (uint8_t)(val + 0.5f);
}