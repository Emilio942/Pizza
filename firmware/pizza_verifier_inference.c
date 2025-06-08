
/*
 * Pizza Verifier Inference Function for RP2040
 * CMSIS-NN optimized implementation
 */

#include "arm_nnfunctions.h"
#include "pizza_verifier_weights.h"
#include <stdint.h>
#include <string.h>

// Buffer for intermediate activations
static int8_t activation_buffer[8192];

/**
 * Perform pizza quality verification
 * @param image_features: Input image features (normalized to int8)
 * @param prediction_features: Model prediction features
 * @param confidence_score: Confidence score (scaled to int8)
 * @param output_buffer: Output buffer for quality score
 * @return: 0 on success, negative on error
 */
int pizza_verifier_inference(
    const int8_t* image_features,
    const int8_t* prediction_features, 
    int8_t confidence_score,
    int8_t* output_buffer
) {
    // Input validation
    if (!image_features || !prediction_features || !output_buffer) {
        return -1;
    }
    
    // Combine input features
    int8_t combined_input[256];  // Adjust size based on actual model
    memcpy(combined_input, image_features, 128);
    memcpy(combined_input + 128, prediction_features, 127);
    combined_input[255] = confidence_score;
    
    // Layer 1: Fully connected
    arm_fully_connected_s8(
        combined_input,
        layer1_weights,
        LAYER1_SIZE,
        256,  // Input size
        LAYER1_ZERO_POINT,
        LAYER1_SCALE,
        activation_buffer
    );
    
    // Layer 2: ReLU activation
    arm_relu_s8(activation_buffer, LAYER1_SIZE);
    
    // Layer 3: Output layer
    arm_fully_connected_s8(
        activation_buffer,
        output_weights,
        1,  // Output size (quality score)
        LAYER1_SIZE,
        OUTPUT_ZERO_POINT,
        OUTPUT_SCALE,
        output_buffer
    );
    
    return 0;
}

/**
 * Convert quality score from int8 to float
 */
float pizza_verifier_score_to_float(int8_t score) {
    return (score + OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
}
