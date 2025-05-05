/**
 * Auto-generated Pizza Detection Model Implementation
 * Optimized for RP2040 Microcontroller
 * Includes advanced image preprocessing and temporal smoothing
 */

#include "pizza_model.h"
#include "pizza_preprocess.h"
#include "pizza_temporal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Buffer for preprocessing
static uint8_t preprocess_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];

// Init flag to track whether model has been initialized
static bool model_initialized = false;

/**
 * Preprocesses an RGB image for model inference
 * Now includes automatic resizing and CLAHE lighting enhancement
 * 
 * @param input_rgb Input RGB888 image (HxWx3)
 * @param output_tensor Output preprocessed tensor (3xHxW)
 */
void pizza_model_preprocess(const uint8_t* input_rgb, float* output_tensor) {
    // Step 1: Apply complete preprocessing pipeline
    // This handles both resizing and lighting enhancement
    preprocess_result_t result = pizza_preprocess_complete(
        input_rgb,               // Input can be any size (e.g., 320x240)
        CAMERA_WIDTH,            // Input size
        CAMERA_HEIGHT,
        preprocess_buffer,       // Outputs 48x48 enhanced image
        MODEL_INPUT_WIDTH, 
        MODEL_INPUT_HEIGHT
    );
    
    // Choose input source based on preprocessing result
    const uint8_t* source = (result == PREPROCESS_OK) ? preprocess_buffer : input_rgb;
    
    // Step 2: Convert RGB image to normalized tensor
    for (int y = 0; y < MODEL_INPUT_HEIGHT; y++) {
        for (int x = 0; x < MODEL_INPUT_WIDTH; x++) {
            for (int c = 0; c < 3; c++) {
                int in_idx = (y * MODEL_INPUT_WIDTH + x) * 3 + c;
                int out_idx = c * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH + y * MODEL_INPUT_WIDTH + x;
                float pixel_value = (float)source[in_idx] / 255.0f;
                output_tensor[out_idx] = (pixel_value - MODEL_MEAN[c]) / MODEL_STD[c];
            }
        }
    }
}

/**
 * Initializes the model and temporal smoother
 */
void pizza_model_init(void) {
    // Prevent double initialization
    if (model_initialized) {
        return;
    }
    
    // Initialize temporal smoother with majority vote strategy
    // This is the most reliable strategy for pizza recognition
    ts_init(TS_STRATEGY_MAJORITY_VOTE, 0.7f);
    
    // Set initialization flag
    model_initialized = true;
}

/**
 * Performs model inference
 * @param input_tensor Preprocessed input tensor
 * @param output_probs Output class probabilities
 */
void pizza_model_infer(const float* input_tensor, float* output_probs) {
    // Actual model inference implementation would go here
    // This is where the model weights and computation would be used
    
    // Placeholder implementation - set all probs to 0 except one
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        output_probs[i] = 0.0f;
    }
    output_probs[0] = 1.0f; // Default to first class
    
    // Initialize model if needed
    if (!model_initialized) {
        pizza_model_init();
    }
    
    // Add this prediction to temporal smoother
    int raw_class = 0;
    float max_prob = output_probs[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {
        if (output_probs[i] > max_prob) {
            max_prob = output_probs[i];
            raw_class = i;
        }
    }
    
    // Add raw prediction to the temporal smoother
    ts_add_prediction(raw_class, max_prob, output_probs);
    
    // Get the smoothed prediction
    ts_result_t smoothed_result;
    if (ts_get_smoothed_prediction(&smoothed_result) == 0) {
        // Replace output with smoothed probabilities
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
            output_probs[i] = smoothed_result.probabilities[i];
        }
    }
}

/**
 * Gets the most likely class index from probabilities
 * @param probs Array of class probabilities
 * @return Index of the most likely class
 */
int pizza_model_get_prediction(const float* probs) {
    int max_idx = 0;
    float max_prob = probs[0];
    
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/**
 * Resets the temporal smoother
 * Call this when the scene changes significantly
 */
void pizza_model_reset_temporal(void) {
    ts_reset();
}
