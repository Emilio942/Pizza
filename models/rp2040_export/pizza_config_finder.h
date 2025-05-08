/**
 * Pizza Model Optimization Configuration Finder
 * Header file for the optimization configuration utility
 */

#ifndef PIZZA_CONFIG_FINDER_H
#define PIZZA_CONFIG_FINDER_H

/**
 * Finds the optimal hardware optimization configuration for this specific model and device
 * by testing different combinations of optimization levels and measuring performance.
 * 
 * This function runs tests with different CMSIS-NN optimization levels and provides
 * recommendations for the optimal configuration based on inference speed, RAM usage,
 * and estimated power consumption.
 */
void pizza_find_optimal_config(void);

#endif // PIZZA_CONFIG_FINDER_H