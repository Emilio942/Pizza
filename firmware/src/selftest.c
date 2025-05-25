#include "selftest.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

// For peripheral register checks
#include "hardware/uart.h"
#include "hardware/gpio.h"
#include "hardware/structs/sio.h" // For sio_hw_t for GPIO checks
#include "hardware/structs/uart.h" // For uart_hw_t if checking UART registers directly

// Define a small RAM area for testing.
// Ensure this is not optimized out and is in a writable segment.
#define RAM_TEST_SIZE_BYTES 256 // Must be a multiple of 4 for word access
uint8_t ram_test_area[RAM_TEST_SIZE_BYTES] __attribute__((aligned(4)));

// --- RAM Test ---
selftest_status_t selftest_ram(void) {
    printf("Starting RAM test (size: %d bytes)...\n", RAM_TEST_SIZE_BYTES);
    volatile uint32_t* p_ram_32 = (uint32_t*)ram_test_area;
    size_t ram_test_size_words = RAM_TEST_SIZE_BYTES / 4;

    if (RAM_TEST_SIZE_BYTES % 4 != 0) {
        printf("RAM_TEST_SIZE_BYTES must be a multiple of 4 for this test.\n");
        return SELFTEST_FAIL;
    }

    uint32_t original_ram_content[ram_test_size_words];
    memcpy(original_ram_content, (void*)p_ram_32, RAM_TEST_SIZE_BYTES);

    // Test 1: Pattern 0x55555555
    for (size_t i = 0; i < ram_test_size_words; ++i) p_ram_32[i] = 0x55555555;
    for (size_t i = 0; i < ram_test_size_words; ++i) {
        if (p_ram_32[i] != 0x55555555) {
            printf("RAM test fail (pattern 0x55555555): word %zu, expected 0x%08lX, got 0x%08lX\n", i, (unsigned long)0x55555555, (unsigned long)p_ram_32[i]);
            memcpy((void*)p_ram_32, original_ram_content, RAM_TEST_SIZE_BYTES);
            return SELFTEST_FAIL;
        }
    }

    // Test 2: Pattern 0xAAAAAAAA
    for (size_t i = 0; i < ram_test_size_words; ++i) p_ram_32[i] = 0xAAAAAAAA;
    for (size_t i = 0; i < ram_test_size_words; ++i) {
        if (p_ram_32[i] != 0xAAAAAAAA) {
            printf("RAM test fail (pattern 0xAAAAAAAA): word %zu, expected 0x%08lX, got 0x%08lX\n", i, (unsigned long)0xAAAAAAAA, (unsigned long)p_ram_32[i]);
            memcpy((void*)p_ram_32, original_ram_content, RAM_TEST_SIZE_BYTES);
            return SELFTEST_FAIL;
        }
    }

    // Test 3: Address as data (simple version)
    for (size_t i = 0; i < ram_test_size_words; ++i) {
        p_ram_32[i] = (uint32_t)((uintptr_t)&p_ram_32[i]);
    }
    for (size_t i = 0; i < ram_test_size_words; ++i) {
        if (p_ram_32[i] != (uint32_t)((uintptr_t)&p_ram_32[i])) {
            printf("RAM test fail (address as data): word %zu, expected 0x%08lX, got 0x%08lX\n", i, (unsigned long)((uintptr_t)&p_ram_32[i]), (unsigned long)p_ram_32[i]);
            memcpy((void*)p_ram_32, original_ram_content, RAM_TEST_SIZE_BYTES);
            return SELFTEST_FAIL;
        }
    }

    memcpy((void*)p_ram_32, original_ram_content, RAM_TEST_SIZE_BYTES);
    printf("RAM test PASS.\n");
    return SELFTEST_PASS;
}

// --- Flash Read Test ---
// These constants must be defined in your application, e.g., in main.c
extern const uint32_t flash_test_value_1;
extern const char* const flash_test_string;
extern const uint32_t flash_test_value_2;

selftest_status_t selftest_flash_read(void) {
    printf("Starting Flash read test...\n");
    bool pass = true;

    if (flash_test_value_1 != 0xDEADBEEF) {
        printf("Flash test fail: flash_test_value_1. Expected 0xDEADBEEF, got 0x%08lX\n", (unsigned long)flash_test_value_1);
        pass = false;
    }

    const char* expected_string = "FlashSelfTestString";
    if (flash_test_string == NULL || strcmp(flash_test_string, expected_string) != 0) {
        printf("Flash test fail: flash_test_string. Expected '%s', got '%s'\n", expected_string, flash_test_string ? flash_test_string : "NULL");
        pass = false;
    }

    if (flash_test_value_2 != 0x12345678) {
        printf("Flash test fail: flash_test_value_2. Expected 0x12345678, got 0x%08lX\n", (unsigned long)flash_test_value_2);
        pass = false;
    }

    // For a more robust flash integrity check, consider implementing a CRC32 checksum
    // over the application binary and comparing it against a pre-calculated value.
    // This requires linker script setup and a build step to compute the CRC.

    if (pass) {
        printf("Flash read test PASS.\n");
        return SELFTEST_PASS;
    } else {
        return SELFTEST_FAIL;
    }
}

// --- Peripheral Register Test ---
// This is highly dependent on your board's initialization.
// Add checks for peripherals critical to your application.
#ifndef PICO_DEFAULT_LED_PIN
#define PICO_DEFAULT_LED_PIN 25 // Common default, adjust if different
#endif

selftest_status_t selftest_peripheral_registers(void) {
    printf("Starting Peripheral register test...\n");
    bool all_ok = true;

    // Example 1: Check LED GPIO (assuming it's initialized as output)
    // This assumes gpio_init(PICO_DEFAULT_LED_PIN) and gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT)
    // have been called prior to this self-test.
    uint32_t led_pin_mask = 1u << PICO_DEFAULT_LED_PIN;
    if (!((sio_hw->gpio_oe & led_pin_mask))) { // Check if Output Enable is set
        printf("Peripheral test FAIL: GPIO %d not configured as OUTPUT (OE check).\n", PICO_DEFAULT_LED_PIN);
        all_ok = false;
    } else {
         printf("Peripheral test: GPIO %d OE set (is OUTPUT).\n", PICO_DEFAULT_LED_PIN);
    }
    if (gpio_get_function(PICO_DEFAULT_LED_PIN) != GPIO_FUNC_SIO) {
        printf("Peripheral test FAIL: GPIO %d not set to SIO function. Current func: %d\n",
               PICO_DEFAULT_LED_PIN, gpio_get_function(PICO_DEFAULT_LED_PIN));
        all_ok = false;
    } else {
        printf("Peripheral test: GPIO %d function is SIO.\n", PICO_DEFAULT_LED_PIN);
    }

    // Example 2: Check UART0 (assuming it's initialized for logging)
    // uart_get_baudrate() can be used, or check IBRD/FBRD registers directly
    // if specific values are known after uart_init(uart0, baudrate).
    // For simplicity, we'll just check if UART0 is enabled.
    if (!(uart_get_hw(uart0)->cr & UART_UARTCR_UARTEN_BITS)) {
        printf("Peripheral test FAIL: UART0 not enabled.\n");
        all_ok = false;
    } else {
        printf("Peripheral test: UART0 is enabled.\n");
    }

    // Add more checks here for other critical peripherals (SPI, I2C, Timers, DMA etc.)
    // by reading their configuration registers and comparing to expected values
    // after your board_init() or equivalent has run.

    if (all_ok) {
        printf("Peripheral register test PASS.\n");
        return SELFTEST_PASS;
    } else {
        return SELFTEST_FAIL;
    }
}

// --- Main Self-Test Runner and Printer ---
void run_selftests(selftest_results_t* results) {
    if (!results) return;

    printf("\n--- Starting Self-Tests ---\n");

    results->ram_test_status = selftest_ram();
    results->flash_read_test_status = selftest_flash_read();
    results->peripheral_reg_test_status = selftest_peripheral_registers();

    printf("--- Self-Tests Complete ---\n");
    print_selftest_results(results);
}

void print_selftest_results(const selftest_results_t* results) {
    if (!results) return;
    const char* pass_str = "PASS";
    const char* fail_str = "FAIL";
    const char* not_run_str = "NOT RUN";

    printf("\n--- Self-Test Results Summary ---\n");
    printf("RAM Test:                     %s\n", results->ram_test_status == SELFTEST_PASS ? pass_str : (results->ram_test_status == SELFTEST_FAIL ? fail_str : not_run_str));
    printf("Flash Read Test:              %s\n", results->flash_read_test_status == SELFTEST_PASS ? pass_str : (results->flash_read_test_status == SELFTEST_FAIL ? fail_str : not_run_str));
    printf("Peripheral Register Test:     %s\n", results->peripheral_reg_test_status == SELFTEST_PASS ? pass_str : (results->peripheral_reg_test_status == SELFTEST_FAIL ? fail_str : not_run_str));
    printf("--- End of Summary ---\n\n");

    // Single line summary for easier parsing by automated tools/emulator
    printf("SELFTEST_SUMMARY:RAM=%d,FLASH=%d,PERIPH=%d\n",
           results->ram_test_status,
           results->flash_read_test_status,
           results->peripheral_reg_test_status);
}