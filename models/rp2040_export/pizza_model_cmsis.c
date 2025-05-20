/**
 * Pizza Detection Model Implementation with CMSIS-NN Optimization
 * Optimized for RP2040 Microcontroller using ARM CMSIS-NN functions
 * for accelerated convolution operations
 * 
 * This implementation provides optimized neural network operations for
 * the RP2040 microcontroller using the CMSIS-NN library. It replaces
 * standard operations with optimized variants that leverage ARM Cortex-M
 * architecture features for better performance.
 */

#include "pizza_model.h"
#include "pizza_model_cmsis.h"
#include "pizza_temporal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Include CMSIS-NN headers
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

// Include performance logging for benchmarking
#include "performance_logger.h"

// Define performance event types for logging
typedef enum {
    PERF_EVENT_CMSIS_INIT = 1,
    PERF_EVENT_CONV = 2,
    PERF_EVENT_DEPTHWISE_CONV = 3,
    PERF_EVENT_POINTWISE_CONV = 4,
    PERF_EVENT_MAX_POOL = 5,
    PERF_EVENT_GLOBAL_POOL = 6,
    PERF_EVENT_FULLY_CONNECTED = 7,
    PERF_EVENT_SOFTMAX = 8
} perf_event_type_t;

// Performance logging wrapper (in case original function is missing)
static void log_performance_event(perf_event_type_t event_type, uint16_t param1, uint16_t param2) {
    #ifdef PERFORMANCE_LOGGER_HAS_EVENTS
    performance_logger_record_event(event_type, param1, param2);
    #endif
}

// Weights and biases for the neural network (would be loaded from the model file)
// These are placeholders and would be replaced with the actual quantized model parameters
static const q7_t conv1_weights[8*3*3*3] = {0};  // 8 output channels, 3 input channels, 3x3 kernel
static const q7_t conv1_bias[8] = {0};
static const q7_t dw_conv_weights[8*1*3*3] = {0}; // 8 channels (depthwise), 3x3 kernel
static const q7_t dw_conv_bias[8] = {0};
static const q7_t pw_conv_weights[16*8*1*1] = {0}; // 16 output channels, 8 input channels, 1x1 kernel
static const q7_t pw_conv_bias[16] = {0};
static const q7_t fc_weights[16*MODEL_NUM_CLASSES] = {0};  // 16 input features, MODEL_NUM_CLASSES output classes
static const q7_t fc_bias[MODEL_NUM_CLASSES] = {0};

// Buffers for intermediate results
static q7_t buffer1[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 8];   // After first conv
static q7_t buffer2[MODEL_INPUT_WIDTH/2 * MODEL_INPUT_HEIGHT/2 * 8];  // After first pooling
static q7_t buffer3[MODEL_INPUT_WIDTH/2 * MODEL_INPUT_HEIGHT/2 * 8];  // After depthwise conv
static q7_t buffer4[MODEL_INPUT_WIDTH/2 * MODEL_INPUT_HEIGHT/2 * 16]; // After pointwise conv
static q7_t buffer5[MODEL_INPUT_WIDTH/4 * MODEL_INPUT_HEIGHT/4 * 16]; // After second pooling
static q7_t buffer6[16]; // After global pooling
static q7_t output_buffer[MODEL_NUM_CLASSES]; // Final output

// Temporary buffers for CMSIS-NN operations
// These buffers need to be large enough for the largest intermediate computations
// For the convolution operations in this model
static q15_t col_buffer[2 * 2 * 3 * 3 * 8]; // Size depends on the largest kernel and number of channels
static q7_t scratch_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 16]; // Scratch memory for operations

// Model state tracking
static bool model_initialized = false;

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
    
    // Initialize scratch buffers for CMSIS-NN operations
    memset(col_buffer, 0, sizeof(col_buffer));
    memset(scratch_buffer, 0, sizeof(scratch_buffer));
    
    // Mark the model as initialized
    model_initialized = true;
    
    // Log initialization of CMSIS-NN for performance tracking
    log_performance_event(PERF_EVENT_CMSIS_INIT, 1, 0);
}

/**
 * Performs max pooling operation (2x2 with stride 2)
 * This version uses CMSIS-NN optimized operations when available
 */
static void max_pooling_2x2(const q7_t *input, q7_t *output, 
                           uint16_t input_width, uint16_t input_height, 
                           uint16_t channels) {
    // Define pooling parameters for CMSIS-NN
    cmsis_nn_pool_params pool_params;
    pool_params.activation.min = -128;
    pool_params.activation.max = 127;
    pool_params.stride.w = 2;
    pool_params.stride.h = 2;
    pool_params.padding.w = 0;
    pool_params.padding.h = 0;
    
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.w = input_width;
    input_dims.h = input_height;
    input_dims.c = channels;
    
    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.w = input_width / 2;
    output_dims.h = input_height / 2;
    output_dims.c = channels;
    
    cmsis_nn_context ctx;
    ctx.buf = NULL;
    ctx.size = 0;
    
    // Use CMSIS-NN max pool function if available
    #if defined(ARM_MATH_DSP) || defined(ARM_MATH_MVEI)
        // CMSIS-NN implementation is available
        arm_max_pool_s8(&ctx, &pool_params, &input_dims, input, &output_dims, output);
    #else
        // Fallback to manual implementation for Cortex-M0
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
                    output[(y * (input_width/2) + x) * channels + c] = max_val;
                }
            }
        }
    #endif
    
    // Log performance for maxpool operation
    log_performance_event(PERF_EVENT_MAX_POOL, channels, input_width * input_height);
}
}

/**
 * Global average pooling implementation
 */
static void global_avg_pooling(const q7_t *input, q7_t *output,
                              uint16_t width, uint16_t height,
                              uint16_t channels) {
    const int size = width * height;
    
    // CMSIS-NN doesn't have a direct global average pooling function,
    // but we can use a combination of other CMSIS-NN operations if available
    
    #if defined(ARM_MATH_DSP) || defined(ARM_MATH_MVEI)
        // Use CMSIS-DSP for more efficient accumulation and averaging
        // Set up pooling parameters
        cmsis_nn_pool_params pool_params;
        pool_params.activation.min = -128;
        pool_params.activation.max = 127;
        pool_params.stride.w = width;
        pool_params.stride.h = height;
        pool_params.padding.w = 0;
        pool_params.padding.h = 0;
        
        cmsis_nn_dims input_dims;
        input_dims.n = 1;
        input_dims.w = width;
        input_dims.h = height;
        input_dims.c = channels;
        
        cmsis_nn_dims output_dims;
        output_dims.n = 1;
        output_dims.w = 1;
        output_dims.h = 1;
        output_dims.c = channels;
        
        cmsis_nn_context ctx;
        ctx.buf = NULL;
        ctx.size = 0;
        
        // Use CMSIS-NN avg pool function
        arm_avgpool_s8(&ctx, &pool_params, &input_dims, input, &output_dims, output);
    #else
        // Fallback implementation for platforms without DSP extension
        for (int c = 0; c < channels; c++) {
            int32_t sum = 0;
            
            // Sum all pixels for this channel
            for (int i = 0; i < size; i++) {
                sum += input[i * channels + c];
            }
            
            // Average
            output[c] = (q7_t)(sum / size);
        }
    #endif
    
    // Record performance event for tracking
    log_performance_event(PERF_EVENT_GLOBAL_POOL, channels, size);
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
        
        // Log performance for convolution operation
        log_performance_event(PERF_EVENT_CONV, 8, 3*3*3);
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
        
        // Log performance for depthwise convolution
        log_performance_event(PERF_EVENT_DEPTHWISE_CONV, 8, 3*3);
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
        
        // Log performance for pointwise convolution
        log_performance_event(PERF_EVENT_POINTWISE_CONV, 16, 8);
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
        
        // Log performance for fully connected layer
        log_performance_event(PERF_EVENT_FULLY_CONNECTED, MODEL_NUM_CLASSES, 16);
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
    
    // Log performance for softmax calculation
    log_performance_event(PERF_EVENT_SOFTMAX, MODEL_NUM_CLASSES, 0);
}