#include "sensor_utils.h"
#include "hardware/adc.h"

// ADC reference voltage is typically 3.3V. The ADC is 12-bit.
const float ADC_CONVERSION_FACTOR = 3.3f / (1 << 12);

void init_onboard_sensors() {
    adc_init();

    // Enable temperature sensor
    adc_set_temp_sensor_enabled(true);

    // Initialize GPIO for voltage sensing if a pin is configured
    // Note: This setup is a generic example.
    // You MUST verify ADC_INPUT and DIVIDER_FACTOR against docs/hardware-documentation.html
    // If VOLTAGE_SENSE_ADC_INPUT is for a pin that's actually used:
    // adc_gpio_init(VOLTAGE_SENSE_GPIO);

    // Make sure to select an input before the first read if you have multiple ADC sources.
}

float read_rp2040_temperature() {
    // Select ADC channel 4 for the temperature sensor
    adc_select_input(4);
    uint16_t raw_adc = adc_read();
    float adc_voltage = (float)raw_adc * ADC_CONVERSION_FACTOR;
    // Formula from RP2040 datasheet: Temperature = 27 - (ADC_voltage - 0.706) / 0.001721
    float temperature = 27.0f - (adc_voltage - 0.706f) / 0.001721f;
    return temperature;
}

float read_board_voltage() {
    // This function is a placeholder and needs to be adapted based on your
    // specific hardware setup (which pin, what voltage is being measured, divider).
    // Refer to docs/hardware-documentation.html.

    // Example: Reading from VOLTAGE_SENSE_ADC_INPUT defined in the header
    // Ensure adc_gpio_init(VOLTAGE_SENSE_GPIO) was called in init_onboard_sensors()
    // if VOLTAGE_SENSE_GPIO is actually connected and used.

    // If no voltage sensing is implemented or configured, return a default/error value.
    // For demonstration, let's assume VOLTAGE_SENSE_ADC_INPUT is configured.
    // adc_select_input(VOLTAGE_SENSE_ADC_INPUT);
    // uint16_t raw_adc = adc_read();
    // float adc_voltage_at_pin = (float)raw_adc * ADC_CONVERSION_FACTOR;
    // float actual_voltage = adc_voltage_at_pin * VOLTAGE_SENSE_DIVIDER_FACTOR;
    // return actual_voltage;

    // For now, returning a dummy value if not properly configured.
    // Replace with actual implementation based on your hardware.
    adc_select_input(VOLTAGE_SENSE_ADC_INPUT); // Select the configured ADC input
    uint16_t raw_adc = adc_read();
    float adc_voltage_at_pin = (float)raw_adc * ADC_CONVERSION_FACTOR;
    return adc_voltage_at_pin * VOLTAGE_SENSE_DIVIDER_FACTOR; // Apply divider factor
}

void get_peripheral_metrics(peripheral_metrics_t* metrics) {
    if (!metrics) return;

    metrics->temperature_celsius = read_rp2040_temperature();
    metrics->voltage_v = read_board_voltage();

    // Populate other metrics if they are part of this structure
    // metrics->inference_time_ms = get_last_inference_time();
    // metrics->ram_usage_kb = get_current_ram_usage();
}