
/*
 * Pizza Verifier Usage Example for RP2040
 */

#include <stdio.h>
#include "pizza_verifier_weights.h"

// External inference function
extern int pizza_verifier_inference(
    const int8_t* image_features,
    const int8_t* prediction_features,
    int8_t confidence_score,
    int8_t* output_buffer
);

extern float pizza_verifier_score_to_float(int8_t score);

int main() {
    // Example input data (would come from actual pizza detection)
    int8_t image_features[128];  // Normalized image features
    int8_t prediction_features[127];  // Model prediction features
    int8_t confidence_score = 85;  // Confidence score (0-100 scaled to int8)
    
    // Initialize with example data
    for (int i = 0; i < 128; i++) {
        image_features[i] = (int8_t)(i % 127 - 64);  // Example features
    }
    
    for (int i = 0; i < 127; i++) {
        prediction_features[i] = (int8_t)(i % 63 - 32);  // Example features
    }
    
    // Perform inference
    int8_t quality_score_raw;
    int result = pizza_verifier_inference(
        image_features,
        prediction_features,
        confidence_score,
        &quality_score_raw
    );
    
    if (result == 0) {
        float quality_score = pizza_verifier_score_to_float(quality_score_raw);
        printf("Pizza quality score: %.3f\n", quality_score);
        
        if (quality_score > 0.8) {
            printf("High quality detection - safe to proceed\n");
        } else if (quality_score > 0.6) {
            printf("Medium quality detection - manual check recommended\n");
        } else {
            printf("Low quality detection - manual verification required\n");
        }
    } else {
        printf("Inference failed with error: %d\n", result);
    }
    
    return 0;
}
