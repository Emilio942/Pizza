#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simplified scaling test for tensor arena calculation
model_sizes = [2048, 5120, 10240, 30720, 102400]  # In bytes

for is_quantized in [True, False]:
    print(f"\nModel Type: {'Quantized (INT8)' if is_quantized else 'Non-quantized (FLOAT32)'}")
    print("-" * 60)
    print(f"{'Model Size':<15} | {'Old Method':<15} | {'New Method':<15} | {'Difference':<15}")
    print("-" * 60)
    
    for size_bytes in model_sizes:
        # Old method (fixed percentages)
        if is_quantized:
            old_estimate = int(size_bytes * 0.2)
        else:
            old_estimate = int(size_bytes * 0.5)
        
        # New method (improved)
        input_size = (3, 96, 96)  # (channels, height, width)
        if size_bytes < 5 * 1024:  # <5KB
            max_feature_maps = 16
        elif size_bytes < 20 * 1024:  # <20KB
            max_feature_maps = 32
        else:
            max_feature_maps = 64
        
        bytes_per_value = 1 if is_quantized else 4
        activation_size = max_feature_maps * (input_size[1]//2) * (input_size[2]//2) * bytes_per_value
        overhead_factor = 1.2
        new_estimate = int(activation_size * overhead_factor)
        
        # Calculate difference
        diff_percentage = ((new_estimate - old_estimate) / old_estimate * 100) if old_estimate > 0 else float('inf')
        
        # Output
        print(f"{size_bytes/1024:.1f}KB{' ':>10} | "
              f"{old_estimate/1024:.1f}KB{' ':>9} | "
              f"{new_estimate/1024:.1f}KB{' ':>9} | "
              f"{diff_percentage:.1f}%{' ':>8}")

print("\n=== TEST COMPLETED ===")
