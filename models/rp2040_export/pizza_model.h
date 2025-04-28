/**
 * Auto-generated Pizza Detection Model for RP2040
 * Model Size: 0.63 KB
 * Input Shape: 3x48x48
 * Classes: basic, burnt, combined, mixed, progression, segment
 */

#ifndef PIZZA_MODEL_H
#define PIZZA_MODEL_H

#include <stdint.h>

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

// Model Parameters
// Model API
void pizza_model_init(void);
void pizza_model_preprocess(const uint8_t* input_rgb, float* output_tensor);
void pizza_model_infer(const float* input_tensor, float* output_probs);
int pizza_model_get_prediction(const float* probs);

#endif // PIZZA_MODEL_H
