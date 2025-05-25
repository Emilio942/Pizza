#ifndef SELFTEST_H
#define SELFTEST_H

#include <stdint.h>
#include <stdbool.h>

// Test result enumeration
typedef enum {
    SELFTEST_PASS = 0,
    SELFTEST_FAIL = 1,
    SELFTEST_NOT_RUN = 2
} selftest_status_t;

// Structure to hold results for each test
typedef struct {
    selftest_status_t ram_test_status;
    selftest_status_t flash_read_test_status;
    selftest_status_t peripheral_reg_test_status;
    // Add more tests as needed
} selftest_results_t;

// Individual test functions
selftest_status_t selftest_ram(void);
selftest_status_t selftest_flash_read(void);
selftest_status_t selftest_peripheral_registers(void);

void run_selftests(selftest_results_t* results);
void print_selftest_results(const selftest_results_t* results);

#endif // SELFTEST_H