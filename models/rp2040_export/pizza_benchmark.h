/**
 * Pizza Model CMSIS-NN Benchmarking Utility
 * Header file for performance measurement functions
 */

#ifndef PIZZA_BENCHMARK_H
#define PIZZA_BENCHMARK_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Pizza CMSIS-NN benchmark results
 */
typedef struct {
    uint32_t standard_avg_time_us;   // Average inference time without optimization (µs)
    uint32_t cmsis_avg_time_us;      // Average inference time with CMSIS-NN (µs)
    uint32_t standard_max_time_us;   // Maximum inference time without optimization (µs)
    uint32_t cmsis_max_time_us;      // Maximum inference time with CMSIS-NN (µs)
    float speedup_factor;            // Speed improvement factor
    uint32_t standard_ram_usage;     // RAM usage without optimization (bytes)
    uint32_t cmsis_ram_usage;        // RAM usage with CMSIS-NN (bytes)
} pizza_benchmark_results_t;

/**
 * Run inference benchmark with and without CMSIS-NN acceleration
 * and measure performance differences
 * 
 * @param results Pointer to store benchmark results
 * @return true if benchmark succeeded, false on error
 */
bool pizza_benchmark_run(pizza_benchmark_results_t* results);

#endif // PIZZA_BENCHMARK_H