/**
 * OV2640 Camera Driver for RP2040 Pizza Detection
 * Provides low-level control and configuration for the OV2640 image sensor
 */

#ifndef OV2640_DRIVER_H
#define OV2640_DRIVER_H

#include <stdint.h>
#include <stdbool.h>

// OV2640 I2C/SCCB Address
#define OV2640_I2C_ADDR 0x30

// OV2640 Register Banks
#define OV2640_BANK_SENSOR 0x00
#define OV2640_BANK_DSP 0x01

// Bank selection register
#define OV2640_REG_BANK_SEL 0xFF

// Sensor Bank Registers (Bank 0)
#define OV2640_REG_GAIN 0x00
#define OV2640_REG_BLUE 0x01
#define OV2640_REG_RED 0x02
#define OV2640_REG_GREEN 0x03
#define OV2640_REG_PIDH 0x0A  // Product ID High
#define OV2640_REG_PIDL 0x0B  // Product ID Low
#define OV2640_REG_MIDH 0x1C  // Manufacturer ID High
#define OV2640_REG_MIDL 0x1D  // Manufacturer ID Low

// DSP Bank Registers (Bank 1)
#define OV2640_REG_R_DVP_SP 0x06  // DVP control
#define OV2640_REG_QS 0x44        // JPEG quality control
#define OV2640_REG_CTRLI 0x50     // Control register I
#define OV2640_REG_HSIZE 0x51     // Horizontal size
#define OV2640_REG_VSIZE 0x52     // Vertical size
#define OV2640_REG_XOFFL 0x53     // X offset low
#define OV2640_REG_YOFFL 0x54     // Y offset low
#define OV2640_REG_VHYX 0x55      // Offset control
#define OV2640_REG_DPRP 0x56      // DP control
#define OV2640_REG_TEST 0x57      // Test register
#define OV2640_REG_ZMOW 0x5A      // Output window width
#define OV2640_REG_ZMOH 0x5B      // Output window height
#define OV2640_REG_ZMHH 0x5C      // Output window H offset high
#define OV2640_REG_BPADDR 0x7C    // SDE indirect register access: address
#define OV2640_REG_BPDATA 0x7D    // SDE indirect register access: data
#define OV2640_REG_CTRL2 0x86     // Control register 2
#define OV2640_REG_CTRL3 0x87     // Control register 3
#define OV2640_REG_SIZEL 0x8C     // Image size adjust

// Product and Manufacturer IDs
#define OV2640_PID_H 0x26
#define OV2640_PID_L 0x42
#define OV2640_MID_H 0x7F
#define OV2640_MID_L 0xA2

// GPIO pins for camera control
#define OV2640_PIN_PWDN 15    // Power down pin
#define OV2640_PIN_RESET 14   // Reset pin
#define OV2640_PIN_XCLK 13    // External clock pin

// I2C pins
#define OV2640_PIN_SDA 16     // I2C data pin
#define OV2640_PIN_SCL 17     // I2C clock pin

// Parallel data pins (PIO)
#define OV2640_PIN_PCLK 18    // Pixel clock
#define OV2640_PIN_VSYNC 19   // Vertical sync
#define OV2640_PIN_HSYNC 20   // Horizontal sync
#define OV2640_PIN_D0 21      // Data bit 0
#define OV2640_PIN_D1 22      // Data bit 1
#define OV2640_PIN_D2 23      // Data bit 2
#define OV2640_PIN_D3 24      // Data bit 3
#define OV2640_PIN_D4 25      // Data bit 4
#define OV2640_PIN_D5 26      // Data bit 5
#define OV2640_PIN_D6 27      // Data bit 6
#define OV2640_PIN_D7 28      // Data bit 7

// Timing constants (in microseconds)
#define OV2640_PWDN_DELAY_US 1000     // Power down delay
#define OV2640_RESET_DELAY_US 1000    // Reset delay
#define OV2640_XCLK_DELAY_US 100      // Clock stabilization delay
#define OV2640_REG_DELAY_US 10        // Delay between register writes

// Image formats
typedef enum {
    OV2640_FORMAT_JPEG,
    OV2640_FORMAT_RGB565,
    OV2640_FORMAT_UYVY,
    OV2640_FORMAT_YUYV
} ov2640_format_t;

// Image sizes
typedef enum {
    OV2640_SIZE_96X96,    // 96x96
    OV2640_SIZE_QQVGA,    // 160x120
    OV2640_SIZE_QCIF,     // 176x144
    OV2640_SIZE_HQVGA,    // 240x176
    OV2640_SIZE_240X240,  // 240x240
    OV2640_SIZE_QVGA,     // 320x240
    OV2640_SIZE_CIF,      // 400x296
    OV2640_SIZE_HVGA,     // 480x320
    OV2640_SIZE_VGA,      // 640x480
    OV2640_SIZE_SVGA,     // 800x600
    OV2640_SIZE_XGA,      // 1024x768
    OV2640_SIZE_HD,       // 1280x720
    OV2640_SIZE_SXGA,     // 1280x1024
    OV2640_SIZE_UXGA      // 1600x1200
} ov2640_size_t;

// Camera configuration structure
typedef struct {
    ov2640_format_t format;
    ov2640_size_t size;
    uint8_t quality;      // JPEG quality (0-63, higher = better quality)
    uint8_t brightness;   // Brightness (-2 to +2)
    uint8_t contrast;     // Contrast (-2 to +2)
    uint8_t saturation;   // Saturation (-2 to +2)
    bool flip_horizontal;
    bool flip_vertical;
} ov2640_config_t;

// Function prototypes
bool ov2640_init(void);
bool ov2640_check_id(void);
bool ov2640_reset(void);
bool ov2640_set_power_down(bool enable);
bool ov2640_set_format(ov2640_format_t format);
bool ov2640_set_size(ov2640_size_t size);
bool ov2640_set_quality(uint8_t quality);
bool ov2640_set_brightness(int8_t brightness);
bool ov2640_set_contrast(int8_t contrast);
bool ov2640_set_saturation(int8_t saturation);
bool ov2640_set_flip(bool horizontal, bool vertical);
bool ov2640_configure(const ov2640_config_t* config);
bool ov2640_start_capture(void);
bool ov2640_capture_to_buffer(uint8_t* buffer, size_t buffer_size);

// Low-level I2C functions
bool ov2640_write_reg(uint8_t reg, uint8_t value);
bool ov2640_read_reg(uint8_t reg, uint8_t* value);
bool ov2640_select_bank(uint8_t bank);

// GPIO control functions
void ov2640_gpio_init(void);
void ov2640_set_pwdn(bool state);
void ov2640_set_reset(bool state);
void ov2640_set_xclk(bool enable);

#endif // OV2640_DRIVER_H
