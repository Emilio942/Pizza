/**
 * Auto-generated Pizza Detection Model Implementation
 * Optimized for RP2040 Microcontroller
 */

#include "pizza_model.h"
#include <string.h>
#include <math.h>

/**
 * Preprocesses an RGB image for model inference
 * @param input_rgb Input RGB888 image (HxWx3)
 * @param output_tensor Output preprocessed tensor (3xHxW)
 */
void pizza_model_preprocess(const uint8_t* input_rgb, float* output_tensor) {
    // Convert RGB image to normalized tensor
    for (int y = 0; y < MODEL_INPUT_HEIGHT; y++) {
        for (int x = 0; x < MODEL_INPUT_WIDTH; x++) {
            for (int c = 0; c < 3; c++) {
                int in_idx = (y * MODEL_INPUT_WIDTH + x) * 3 + c;
                int out_idx = c * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH + y * MODEL_INPUT_WIDTH + x;
                float pixel_value = (float)input_rgb[in_idx] / 255.0f;
                output_tensor[out_idx] = (pixel_value - MODEL_MEAN[c]) / MODEL_STD[c];
            }
        }
    }
}

/**
 * Initializes the model
 */
void pizza_model_init(void) {
    // Initialize any model parameters if needed
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
