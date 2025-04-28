/**
 * Battery monitoring utilities for RP2040 pizza detection
 */

#ifndef BATTERY_MONITOR_H
#define BATTERY_MONITOR_H

#include <stdint.h>

// Battery parameters
#define BATTERY_FULL_MV 3000
#define BATTERY_LOW_MV 2200
#define BATTERY_CRITICAL_MV 2000

// Battery status
typedef enum {
    BATTERY_STATUS_FULL,
    BATTERY_STATUS_GOOD,
    BATTERY_STATUS_LOW,
    BATTERY_STATUS_CRITICAL
} battery_status_t;

// Battery management functions
void battery_monitor_init(uint8_t adc_pin);
uint16_t battery_read_voltage(void);
battery_status_t battery_get_status(void);
float battery_get_percentage(void);
void battery_enable_low_power_mode(void);

#endif // BATTERY_MONITOR_H
