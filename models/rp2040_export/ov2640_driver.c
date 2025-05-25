/**
 * OV2640 Camera Driver Implementation for RP2040 Pizza Detection
 * Provides complete initialization and control sequences for OV2640
 */

#include "ov2640_driver.h"
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include <string.h>

// I2C instance
#define I2C_PORT i2c0

// Static configuration tracking
static ov2640_config_t current_config;
static bool initialized = false;

// 48x48 pixel configuration registers for pizza detection
// These register sequences configure the OV2640 for optimal pizza detection
static const uint8_t ov2640_48x48_init[][2] = {
    // Bank selection and reset
    {OV2640_REG_BANK_SEL, OV2640_BANK_DSP},
    {0x2c, 0xff},
    {0x2e, 0xdf},
    
    // Bank 0 (Sensor) registers
    {OV2640_REG_BANK_SEL, OV2640_BANK_SENSOR},
    {0x32, 0x36},
    {0x0c, 0x36},
    {0x0d, 0x34},
    {0x0e, 0x05},
    {0x0f, 0xc5},
    {0x10, 0x20},
    {0x11, 0x01},
    {0x12, 0x02},
    {0x13, 0x28},
    {0x14, 0x32},
    {0x15, 0x30},
    
    // Bank 1 (DSP) registers for 48x48 resolution
    {OV2640_REG_BANK_SEL, OV2640_BANK_DSP},
    {0xc0, 0x64},
    {0xc1, 0x4b},
    {0x86, 0x35},
    {0x50, 0x92},
    {0x51, 0x01},
    {0x52, 0x01},
    {0x53, 0x00},
    {0x54, 0x00},
    {0x55, 0x88},
    {0x57, 0x00},
    {0x5a, 0x30},  // Output width 48 pixels
    {0x5b, 0x30},  // Output height 48 pixels
    {0x5c, 0x00},
    {0xd3, 0x04},
    
    // Color matrix and format settings for RGB565
    {0x7f, 0x00},
    {0xe0, 0x00},
    {0xe1, 0x00},
    {0xe5, 0x00},
    {0xd7, 0x00},
    {0xda, 0x00},
    {0xe0, 0x00},
    
    // End marker
    {0xff, 0xff}
};

// Quality settings for JPEG mode
static const uint8_t jpeg_quality_high[] = {
    0x44, 0x0c, 0x46, 0x14, 0x47, 0x0c
};

static const uint8_t jpeg_quality_low[] = {
    0x44, 0x32, 0x46, 0x3f, 0x47, 0x32
};

/**
 * Initialize GPIO pins for camera control
 */
void ov2640_gpio_init(void) {
    // Initialize control pins
    gpio_init(OV2640_PIN_PWDN);
    gpio_set_dir(OV2640_PIN_PWDN, GPIO_OUT);
    gpio_put(OV2640_PIN_PWDN, 1);  // Start in power down mode
    
    gpio_init(OV2640_PIN_RESET);
    gpio_set_dir(OV2640_PIN_RESET, GPIO_OUT);
    gpio_put(OV2640_PIN_RESET, 0);  // Start in reset mode
    
    // Initialize I2C pins
    gpio_set_function(OV2640_PIN_SDA, GPIO_FUNC_I2C);
    gpio_set_function(OV2640_PIN_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(OV2640_PIN_SDA);
    gpio_pull_up(OV2640_PIN_SCL);
    
    // Initialize I2C at 100kHz for SCCB compatibility
    i2c_init(I2C_PORT, 100 * 1000);
    
    // Initialize PWM for XCLK generation (20MHz)
    gpio_set_function(OV2640_PIN_XCLK, GPIO_FUNC_PWM);
    uint slice_num = pwm_gpio_to_slice_num(OV2640_PIN_XCLK);
    pwm_config config = pwm_get_default_config();
    
    // Configure PWM for 20MHz output (system clock / divisor)
    // Assuming 125MHz system clock: 125MHz / 6.25 = 20MHz
    pwm_config_set_clkdiv(&config, 6.25f);
    pwm_config_set_wrap(&config, 1);
    pwm_init(slice_num, &config, false);
    pwm_set_gpio_level(OV2640_PIN_XCLK, 1);
    
    // Initialize parallel data pins for PIO capture
    for (int pin = OV2640_PIN_D0; pin <= OV2640_PIN_D7; pin++) {
        gpio_init(pin);
        gpio_set_dir(pin, GPIO_IN);
    }
    
    gpio_init(OV2640_PIN_PCLK);
    gpio_set_dir(OV2640_PIN_PCLK, GPIO_IN);
    
    gpio_init(OV2640_PIN_VSYNC);
    gpio_set_dir(OV2640_PIN_VSYNC, GPIO_IN);
    
    gpio_init(OV2640_PIN_HSYNC);
    gpio_set_dir(OV2640_PIN_HSYNC, GPIO_IN);
}

/**
 * Set power down state
 */
void ov2640_set_pwdn(bool state) {
    gpio_put(OV2640_PIN_PWDN, state ? 1 : 0);
    if (!state) {
        sleep_us(OV2640_PWDN_DELAY_US);
    }
}

/**
 * Set reset state
 */
void ov2640_set_reset(bool state) {
    gpio_put(OV2640_PIN_RESET, state ? 0 : 1);  // Reset is active low
    if (!state) {
        sleep_us(OV2640_RESET_DELAY_US);
    }
}

/**
 * Enable/disable external clock
 */
void ov2640_set_xclk(bool enable) {
    uint slice_num = pwm_gpio_to_slice_num(OV2640_PIN_XCLK);
    pwm_set_enabled(slice_num, enable);
    if (enable) {
        sleep_us(OV2640_XCLK_DELAY_US);
    }
}

/**
 * Write to OV2640 register via I2C/SCCB
 */
bool ov2640_write_reg(uint8_t reg, uint8_t value) {
    uint8_t data[2] = {reg, value};
    int result = i2c_write_blocking(I2C_PORT, OV2640_I2C_ADDR, data, 2, false);
    
    if (result == 2) {
        sleep_us(OV2640_REG_DELAY_US);
        return true;
    }
    return false;
}

/**
 * Read from OV2640 register via I2C/SCCB
 */
bool ov2640_read_reg(uint8_t reg, uint8_t* value) {
    int result = i2c_write_blocking(I2C_PORT, OV2640_I2C_ADDR, &reg, 1, true);
    if (result != 1) {
        return false;
    }
    
    result = i2c_read_blocking(I2C_PORT, OV2640_I2C_ADDR, value, 1, false);
    return (result == 1);
}

/**
 * Select register bank
 */
bool ov2640_select_bank(uint8_t bank) {
    return ov2640_write_reg(OV2640_REG_BANK_SEL, bank);
}

/**
 * Check device ID to verify communication
 */
bool ov2640_check_id(void) {
    uint8_t pidh, pidl, midh, midl;
    
    // Select sensor bank
    if (!ov2640_select_bank(OV2640_BANK_SENSOR)) {
        return false;
    }
    
    // Read product and manufacturer IDs
    if (!ov2640_read_reg(OV2640_REG_PIDH, &pidh) ||
        !ov2640_read_reg(OV2640_REG_PIDL, &pidl) ||
        !ov2640_read_reg(OV2640_REG_MIDH, &midh) ||
        !ov2640_read_reg(OV2640_REG_MIDL, &midl)) {
        return false;
    }
    
    // Verify IDs match expected values
    return (pidh == OV2640_PID_H && pidl == OV2640_PID_L &&
            midh == OV2640_MID_H && midl == OV2640_MID_L);
}

/**
 * Perform software reset
 */
bool ov2640_reset(void) {
    // Hardware reset sequence
    ov2640_set_reset(true);
    sleep_ms(10);
    ov2640_set_reset(false);
    sleep_ms(10);
    
    // Software reset via register
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    if (!ov2640_write_reg(0x80, 0x80)) {
        return false;
    }
    
    sleep_ms(50);  // Wait for reset to complete
    return true;
}

/**
 * Set power down mode
 */
bool ov2640_set_power_down(bool enable) {
    ov2640_set_pwdn(enable);
    return true;
}

/**
 * Initialize OV2640 camera with 48x48 pixel configuration
 */
bool ov2640_init(void) {
    // Initialize GPIO and I2C
    ov2640_gpio_init();
    
    // Power up sequence
    ov2640_set_pwdn(true);   // Power down first
    ov2640_set_reset(true);  // Reset
    ov2640_set_xclk(true);   // Enable clock
    
    sleep_ms(10);
    
    ov2640_set_pwdn(false);  // Release power down
    sleep_ms(10);
    
    ov2640_set_reset(false); // Release reset
    sleep_ms(50);
    
    // Check if device is responding
    if (!ov2640_check_id()) {
        return false;
    }
    
    // Perform software reset
    if (!ov2640_reset()) {
        return false;
    }
    
    // Load 48x48 pixel configuration
    int i = 0;
    while (ov2640_48x48_init[i][0] != 0xff || ov2640_48x48_init[i][1] != 0xff) {
        if (!ov2640_write_reg(ov2640_48x48_init[i][0], ov2640_48x48_init[i][1])) {
            return false;
        }
        i++;
    }
    
    // Set default configuration
    current_config.format = OV2640_FORMAT_RGB565;
    current_config.size = OV2640_SIZE_96X96;  // Closest to 48x48
    current_config.quality = 12;
    current_config.brightness = 0;
    current_config.contrast = 0;
    current_config.saturation = 0;
    current_config.flip_horizontal = false;
    current_config.flip_vertical = false;
    
    initialized = true;
    return true;
}

/**
 * Configure camera with specific settings
 */
bool ov2640_configure(const ov2640_config_t* config) {
    if (!initialized || !config) {
        return false;
    }
    
    // Apply configuration
    if (!ov2640_set_format(config->format) ||
        !ov2640_set_size(config->size) ||
        !ov2640_set_quality(config->quality) ||
        !ov2640_set_brightness(config->brightness) ||
        !ov2640_set_contrast(config->contrast) ||
        !ov2640_set_saturation(config->saturation) ||
        !ov2640_set_flip(config->flip_horizontal, config->flip_vertical)) {
        return false;
    }
    
    // Update current configuration
    memcpy(&current_config, config, sizeof(ov2640_config_t));
    return true;
}

/**
 * Start frame capture
 */
bool ov2640_start_capture(void) {
    if (!initialized) {
        return false;
    }
    
    // Enable capture by clearing any previous capture state
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    // Trigger capture start
    return ov2640_write_reg(0x3c, 0x46);
}

/**
 * Set image format
 */
bool ov2640_set_format(ov2640_format_t format) {
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    switch (format) {
        case OV2640_FORMAT_JPEG:
            return ov2640_write_reg(0xDA, 0x00) &&
                   ov2640_write_reg(0xE0, 0x00);
        
        case OV2640_FORMAT_RGB565:
            return ov2640_write_reg(0xDA, 0x09) &&
                   ov2640_write_reg(0xE0, 0x00);
        
        case OV2640_FORMAT_UYVY:
            return ov2640_write_reg(0xDA, 0x08) &&
                   ov2640_write_reg(0xE0, 0x00);
        
        case OV2640_FORMAT_YUYV:
            return ov2640_write_reg(0xDA, 0x00) &&
                   ov2640_write_reg(0xE0, 0x04);
        
        default:
            return false;
    }
}

/**
 * Set image size - for pizza detection we use 48x48 equivalent
 */
bool ov2640_set_size(ov2640_size_t size) {
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    // For pizza detection, we always use 48x48 equivalent settings
    // This maps to the closest available resolution
    return ov2640_write_reg(OV2640_REG_ZMOW, 0x30) &&  // 48 pixels width
           ov2640_write_reg(OV2640_REG_ZMOH, 0x30);     // 48 pixels height
}

/**
 * Set JPEG quality (for JPEG format)
 */
bool ov2640_set_quality(uint8_t quality) {
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    // Quality setting for JPEG mode
    return ov2640_write_reg(OV2640_REG_QS, quality & 0x3F);
}

/**
 * Set brightness (-2 to +2)
 */
bool ov2640_set_brightness(int8_t brightness) {
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    uint8_t value = (brightness + 2) << 4;
    return ov2640_write_reg(0x9B, value);
}

/**
 * Set contrast (-2 to +2)
 */
bool ov2640_set_contrast(int8_t contrast) {
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    uint8_t value = 0x20 + (contrast * 8);
    return ov2640_write_reg(0x9C, value);
}

/**
 * Set saturation (-2 to +2)
 */
bool ov2640_set_saturation(int8_t saturation) {
    if (!ov2640_select_bank(OV2640_BANK_DSP)) {
        return false;
    }
    
    uint8_t value = 0x40 + (saturation * 16);
    return ov2640_write_reg(0x9D, value);
}

/**
 * Set image flip
 */
bool ov2640_set_flip(bool horizontal, bool vertical) {
    if (!ov2640_select_bank(OV2640_BANK_SENSOR)) {
        return false;
    }
    
    uint8_t value = 0x06;  // Base value
    if (horizontal) value |= 0x40;
    if (vertical) value |= 0x10;
    
    return ov2640_write_reg(0x04, value);
}

/**
 * Capture image to buffer (simplified for emulation)
 */
bool ov2640_capture_to_buffer(uint8_t* buffer, size_t buffer_size) {
    if (!initialized || !buffer) {
        return false;
    }
    
    // Start capture
    if (!ov2640_start_capture()) {
        return false;
    }
    
    // In real implementation, this would:
    // 1. Wait for VSYNC
    // 2. Capture pixel data via PIO DMA
    // 3. Process data according to format
    
    // For emulation, we return success
    return true;
}
