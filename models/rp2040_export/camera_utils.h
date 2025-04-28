/**
 * OV2640 camera utilities for RP2040 pizza detection
 */

#ifndef CAMERA_UTILS_H
#define CAMERA_UTILS_H

#include <stdint.h>

// Camera parameters
#define CAMERA_WIDTH 320
#define CAMERA_HEIGHT 240
#define CAMERA_BUFFER_SIZE (CAMERA_WIDTH * CAMERA_HEIGHT * 3) // RGB888

// Camera modes
typedef enum {
    CAMERA_MODE_NORMAL,
    CAMERA_MODE_LOW_POWER,
    CAMERA_MODE_STANDBY
} camera_mode_t;

// Camera functions
void camera_init(void);
void camera_set_mode(camera_mode_t mode);
int camera_capture_image(uint8_t* buffer);
void camera_set_brightness(uint8_t value);
void camera_set_contrast(uint8_t value);
void camera_power_off(void);
void camera_power_on(void);

#endif // CAMERA_UTILS_H
