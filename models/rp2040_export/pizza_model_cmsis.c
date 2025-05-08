/**
 * Pizza Detection Model Implementation with CMSIS-NN Optimization
 * Optimized for RP2040 Microcontroller using ARM CMSIS-NN functions
 * for accelerated convolution operations
 */

#include "pizza_model.h"
#include "pizza_model_cmsis.h"
#include "pizza_temporal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Include CMSIS-NN headers
#include "arm_nnfunctions.h"

// Weights and biases for the neural network (would be loaded from the model file)
// These are placeholders and would be replaced with the actual quantized model parameters
static const q7_t conv1_weights[8*3*3*3];  // 8 output channels, 3 input channels, 3x3 kernel
static const q7_t conv1_bias[8];
static const q7_t dw_conv_weights[8*1*3*3]; // 8 channels (depthwise), 3x3 kernel
static const q7_t dw_conv_bias[8];
static const q7_t pw_conv_weights[16*8*1*1]; // 16 output channels, 8 input channels, 1x1 kernel
static const q7_t pw_conv_bias[16];
static const q7_t fc_weights[16*4];  // 16 input features, 4 output classes
static const q7_t fc_bias[4];

// Buffers for intermediate results
static q7_t buffer1[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 8];   // After first conv
static q7_t buffer2[MODEL_INPUT_WIDTH/4 * MODEL_INPUT_HEIGHT/4 * 8];  // After first pooling
static q7_t buffer3[MODEL_INPUT_WIDTH/4 * MODEL_INPUT_HEIGHT/4 * 8];  // After depthwise conv
static q7_t buffer4[MODEL_INPUT_WIDTH/4 * MODEL_INPUT_HEIGHT/4 * 16]; // After pointwise conv
static q7_t buffer5[MODEL_INPUT_WIDTH/8 * MODEL_INPUT_HEIGHT/8 * 16]; // After second pooling
static q7_t buffer6[16]; // After global pooling
static q7_t output_buffer[MODEL_NUM_CLASSES]; // Final output

// Temporary buffers for CMSIS-NN operations
static q15_t col_buffer[2048];  // Buffer for im2col operation
static q7_t  scratch_buffer[2048]; // Scratch buffer for CMSIS-NN

/**
 * Initializes the model with CMSIS-NN optimization
 */
void pizza_model_cmsis_init(void) {
    // Initialize buffers and parameters
    memset(buffer1, 0, sizeof(buffer1));
    memset(buffer2, 0, sizeof(buffer2));
    memset(buffer3, 0, sizeof(buffer3));
    memset(buffer4, 0, sizeof(buffer4));
    memset(buffer5, 0, sizeof(buffer5));
    memset(buffer6, 0, sizeof(buffer6));
    memset(output_buffer, 0, sizeof(output_buffer));
}

/**
 * Performs max pooling operation (2x2 with stride 2)
 */
static void max_pooling_2x2(const q7_t *input, q7_t *output, 
                           uint16_t input_width, uint16_t input_height, 
                           uint16_t channels) {
    // For each channel
    for (int c = 0; c < channels; c++) {
        // For each output pixel
        for (int y = 0; y < input_height/2; y++) {
            for (int x = 0; x < input_width/2; x++) {
                // Input coordinates
                int in_x = x * 2;
                int in_y = y * 2;
                
                // Find maximum in 2x2 window
                q7_t max_val = input[(in_y * input_width + in_x) * channels + c];
                q7_t val;
                
                val = input[(in_y * input_width + in_x + 1) * channels + c];
                if (val > max_val) max_val = val;
                
                val = input[((in_y + 1) * input_width + in_x) * channels + c];
                if (val > max_val) max_val = val;
                
                val = input[((in_y + 1) * input_width + in_x + 1) * channels + c];
                if (val > max_val) max_val = val;
                
                // Set output pixel
                output[(y * input_width/2 + x) * channels + c] = max_val;
            }
        }
    }
}

/**
 * Global average pooling implementation
 */
static void global_avg_pooling(const q7_t *input, q7_t *output,
                              uint16_t width, uint16_t height,
                              uint16_t channels) {
    const int size = width * height;
    
    for (int c = 0; c < channels; c++) {
        int32_t sum = 0;
        
        // Sum all pixels for this channel
        for (int i = 0; i < size; i++) {
            sum += input[i * channels + c];
        }
        
        // Average
        output[c] = (q7_t)(sum / size);
    }
}

/**
 * Quantizes float input to q7_t format for CMSIS-NN operations
 * @param input_float Input tensor in float format
 * @param output_q7 Output tensor in q7_t format
 * @param size Size of the tensor
 */
void pizza_model_quantize_input(const float* input_float, q7_t* output_q7, int size) {
    // Fixed point conversion scale factor
    // For q7_t, range is [-128, 127], we use 127 for full range
    const float scale = 127.0f;
    
    for (int i = 0; i < size; i++) {
        // Clamp to [-1.0, 1.0] range
        float clamped = input_float[i];
        if (clamped > 1.0f) clamped = 1.0f;
        if (clamped < -1.0f) clamped = -1.0f;
        
        // Convert to q7_t format
        output_q7[i] = (q7_t)(clamped * scale);
    }
}

/**
 * Performs model inference with CMSIS-NN optimization
 * @param input_tensor Preprocessed and quantized input tensor (q7_t)
 * @param output_probs Output class probabilities (float)
 */
void pizza_model_cmsis_infer(const q7_t* input_tensor, float* output_probs) {
    // Layer 1: Convolution with 3x3 kernel, 3->8 channels
    if (pizza_model_should_optimize_layer(1)) {
        // Optimized version with CMSIS-NN
        arm_convolve_HWC_q7_basic(
            input_tensor,
            MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 3,  // Input dimensions and channels
            conv1_weights,                          // Weights
            8,                                      // Output channels
            3, 3,                                   // Kernel dimensions
            1, 1,                                   // Padding
            2, 2,                                   // Stride
            conv1_bias,                             // Bias
            1,                                      // Bias shift
            7,                                      // Output shift
            buffer1,                                // Output buffer
            MODEL_INPUT_WIDTH/2, MODEL_INPUT_HEIGHT/2, // Output dimensions
            col_buffer,                             // Working buffer for im2col
            NULL                                    // No activation
        );
    } else {
        // Fallback to standard convolution if optimization is disabled for this layer
        // (Implementation omitted for brevity, would be a standard convolution loop)
        // This would use the standard convolution implementation
    }

    // Max pooling 2x2 with stride 2
    max_pooling_2x2(
        buffer1, 
        buffer2, 
        MODEL_INPUT_WIDTH/2, MODEL_INPUT_HEIGHT/2, 
        8
    );

    // Layer 2: Depthwise convolution with 3x3 kernel
    if (pizza_model_should_optimize_layer(2)) {
        // Optimized version with CMSIS-NN
        arm_depthwise_separable_conv_HWC_q7(
            buffer2,                                // Input
            MODEL_INPUT_WIDTH/4, MODEL_INPUT_HEIGHT/4, // Input dimensions
            8,                                      // Input channels
            dw_conv_weights,                        // Depthwise weights
            8,                                      // Output channels (same for depthwise)
            1, 1,                                   // Padding
            1, 1,                                   // Stride
            dw_conv_bias,                           // Bias
            1,                                      // Bias shift
            7,                                      // Output shift
            buffer3,                                // Output buffer
            MODEL_INPUT_WIDTH/4, MODEL_INPUT_HEIGHT/4, // Output dimensions
            col_buffer,                             // Working buffer
            scratch_buffer                          // Scratch buffer
        );
    } else {
        // Fallback to standard depthwise convolution if optimization is disabled
        // (Implementation omitted for brevity)
    }

    // Pointwise convolution (1x1) to expand channels 8->16
    if (pizza_model_should_optimize_layer(3)) {
        // Optimized version with CMSIS-NN
        arm_convolve_1x1_HWC_q7_fast(
            buffer3,                                // Input
            MODEL_INPUT_WIDTH/4, MODEL_INPUT_HEIGHT/4, // Input dimensions 
            8,                                      // Input channels
            pw_conv_weights,                        // Weights
            16,                                     // Output channels
            1, 1,                                   // Padding
            1, 1,                                   // Stride
            pw_conv_bias,                           // Bias
            1,                                      // Bias shift
            7,                                      // Output shift
            buffer4,                                // Output buffer
            MODEL_INPUT_WIDTH/4, MODEL_INPUT_HEIGHT/4, // Output dimensions
            col_buffer,                             // Working buffer
            NULL                                    // No activation
        );
    } else {
        // Fallback to standard 1x1 convolution if optimization is disabled
        // (Implementation omitted for brevity)
    }

    // Max pooling 2x2 with stride 2
    max_pooling_2x2(
        buffer4, 
        buffer5, 
        MODEL_INPUT_WIDTH/4, MODEL_INPUT_HEIGHT/4, 
        16
    );

    // Global average pooling
    global_avg_pooling(
        buffer5, 
        buffer6, 
        MODEL_INPUT_WIDTH/8, MODEL_INPUT_HEIGHT/8, 
        16
    );

    // Fully connected layer
    if (pizza_model_should_optimize_layer(4)) {
        // Optimized version with CMSIS-NN
        arm_fully_connected_q7(
            buffer6,                                // Input
            fc_weights,                             // Weights
            16,                                     // Input size
            MODEL_NUM_CLASSES,                      // Output size
            1,                                      // Bias shift
            7,                                      // Output shift
            fc_bias,                                // Bias
            output_buffer,                          // Output
            col_buffer                              // Working buffer
        );
    } else {
        // Fallback to standard fully connected if optimization is disabled
        // (Implementation omitted for brevity)
    }

    // Convert q7_t outputs to float probabilities using softmax
    q7_t max_val = output_buffer[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {
        if (output_buffer[i] > max_val) {
            max_val = output_buffer[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        output_probs[i] = expf((output_buffer[i] - max_val) / 128.0f);
        sum += output_probs[i];
    }

    // Normalize probabilities
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        output_probs[i] /= sum;
    }
}