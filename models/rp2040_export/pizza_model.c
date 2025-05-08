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
#include "pizza_model_cmsis.h"
#include "performance_logger.h"

// Buffer for preprocessing
static uint8_t preprocess_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];

// Buffer for quantized input
static q7_t quantized_input_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];

// Config flag for using CMSIS-NN optimization
static bool use_cmsis_optimization = false;

// Init flag to track whether model has been initialized
static bool model_initialized = false;

/**
 * Enables or disables CMSIS-NN hardware optimization
 * @param enable True to enable CMSIS-NN optimization, false to use standard implementation
 */
void pizza_model_set_hardware_optimization(bool enable) {
    use_cmsis_optimization = enable;
    
    // Initialize CMSIS-NN if enabled for the first time
    if (enable && model_initialized) {
        pizza_model_cmsis_init();
    }
}

/**
 * Returns whether hardware optimization is currently enabled
 * @return True if CMSIS-NN optimization is enabled
 */
bool pizza_model_is_hardware_optimized(void) {
    return use_cmsis_optimization;
}

/**
 * Selectively enables specific CMSIS-NN optimizations based on the
 * optimization level and layer type. This provides fine-grained control
 * over which operations are hardware-accelerated.
 * 
 * @param layer_type Type of layer (1=Conv, 2=Depthwise, 3=Pointwise, 4=FC)
 * @return True if this layer type should be hardware accelerated
 */
bool pizza_model_should_optimize_layer(int layer_type) {
    // If CMSIS-NN completely disabled, optimize nothing
    if (!use_cmsis_optimization) {
        return false;
    }

    // With optimization level 0, optimize nothing regardless of CMSIS flag
    if (PIZZA_CONV_OPTIMIZATION_LEVEL == 0) {
        return false;
    }
    
    // With optimization level 1, only optimize standard convolutions
    if (PIZZA_CONV_OPTIMIZATION_LEVEL == 1) {
        return (layer_type == 1);  // Only standard convolutions
    }
    
    // With optimization level 2, optimize conv and depthwise operations
    if (PIZZA_CONV_OPTIMIZATION_LEVEL == 2) {
        return (layer_type == 1 || layer_type == 2);  // Conv and depthwise
    }
    
    // With optimization level 3 (default), optimize everything
    return true;
}

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
    // Initialize model if needed
    if (!model_initialized) {
        pizza_model_init();
        if (use_cmsis_optimization) {
            pizza_model_cmsis_init();
        }
    }
    
    // Start performance measurement
    performance_logger_start_measurement();
    
    if (use_cmsis_optimization) {
        // Quantize input for CMSIS-NN
        pizza_model_quantize_input(
            input_tensor, 
            quantized_input_buffer,
            MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3
        );
        
        // Use CMSIS-NN optimized implementation
        pizza_model_cmsis_infer(quantized_input_buffer, output_probs);
    } else {
        // Standard implementation (placeholder)
        // In a real implementation, this would perform the neural network operations
        // using standard floating point math instead of CMSIS-NN
        
        // Set default values for the placeholder
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
            output_probs[i] = 0.0f;
        }
        output_probs[0] = 1.0f; // Default to first class
    }
    
    // Find predicted class
    int raw_class = 0;
    float max_prob = output_probs[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {
        if (output_probs[i] > max_prob) {
            max_prob = output_probs[i];
            raw_class = i;
        }
    }
    
    // End performance measurement
    performance_logger_end_measurement(raw_class, (uint8_t)(max_prob * 100));
    
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
