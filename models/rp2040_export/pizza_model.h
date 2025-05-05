/**
 * Auto-generated Pizza Detection Model for RP2040
 * Model Size: 0.63 KB
 * Input Shape: 3x48x48
 * Classes: basic, burnt, combined, mixed, progression, segment
 */

#ifndef PIZZA_MODEL_H
#define PIZZA_MODEL_H

#include <stdint.h>
#include <stdbool.h>

// Model Configuration
#define MODEL_INPUT_WIDTH 48
#define MODEL_INPUT_HEIGHT 48
#define MODEL_INPUT_CHANNELS 3
#define MODEL_NUM_CLASSES 6

// Preprocessing Parameters
static const float MODEL_MEAN[3] = {0.479359f, 0.395730f, 0.324222f};
static const float MODEL_STD[3] = {0.234756f, 0.251777f, 0.263924f};

// Class Names
static const char* const CLASS_NAMES[MODEL_NUM_CLASSES] = {
    "basic",
    "burnt",
    "combined",
    "mixed",
    "progression",
    "segment"
};

// Model API
/**
 * Initializes the model and temporal smoother
 */
void pizza_model_init(void);

/**
 * Preprocesses an RGB image for model inference
 * Includes automatic lighting enhancement with CLAHE
 * 
 * @param input_rgb Input RGB888 image (HxWx3)
 * @param output_tensor Output preprocessed tensor (3xHxW)
 */
void pizza_model_preprocess(const uint8_t* input_rgb, float* output_tensor);

/**
 * Performs model inference with temporal smoothing
 * Combines multiple predictions for more stable results
 * 
 * @param input_tensor Preprocessed input tensor
 * @param output_probs Output class probabilities
 */
void pizza_model_infer(const float* input_tensor, float* output_probs);

/**
 * Gets the most likely class index from probabilities
 * 
 * @param probs Array of class probabilities
 * @return Index of the most likely class
 */
int pizza_model_get_prediction(const float* probs);

/**
 * Resets the temporal smoother
 * Call this when the scene changes significantly or when
 * starting a new detection sequence
 */
void pizza_model_reset_temporal(void);

#endif // PIZZA_MODEL_H
