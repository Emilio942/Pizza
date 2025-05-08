/**
 * Pizza Model with CMSIS-NN optimization
 * Header file for hardware-accelerated neural network operations
 */

#ifndef PIZZA_MODEL_CMSIS_H
#define PIZZA_MODEL_CMSIS_H

#include <stdint.h>

// Type definitions from CMSIS-NN
typedef int8_t q7_t;
typedef int16_t q15_t;
typedef int32_t q31_t;

/**
 * Initializes the model with CMSIS-NN optimization
 */
void pizza_model_cmsis_init(void);

/**
 * Performs model inference with CMSIS-NN optimization
 * @param input_tensor Preprocessed and quantized input tensor (q7_t)
 * @param output_probs Output class probabilities (float)
 */
void pizza_model_cmsis_infer(const q7_t* input_tensor, float* output_probs);

/**
 * Quantizes float input to q7_t format for CMSIS-NN operations
 * @param input_float Input tensor in float format
 * @param output_q7 Output tensor in q7_t format
 * @param size Size of the tensor
 */
void pizza_model_quantize_input(const float* input_float, q7_t* output_q7, int size);

#endif // PIZZA_MODEL_CMSIS_H