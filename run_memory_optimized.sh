#!/bin/bash
# Run the memory-optimized pizza dataset generator with settings optimized for NVIDIA RTX 3060
# This script tries different approaches in sequence until one works

echo "Pizza Dataset Generation - Memory-Optimized for NVIDIA RTX 3060"
echo "=============================================================="
echo
echo "This script will try multiple approaches to generate the pizza dataset"
echo "working around the 12GB VRAM limitation of the RTX 3060."
echo

# Make the script executable
chmod +x memory_optimized_generator.py

# Try with the smallest, most efficient model (sd-food) at 512x512 resolution
echo "Approach 1: Using sd-food model at 512x512 resolution with memory optimizations"
python memory_optimized_generator.py \
  --preset small_diverse \
  --model sd-food \
  --image_size 512 \
  --batch_size 1 \
  --offload_to_cpu \
  --expand_segments \
  --output_dir data/synthetic/attempt1

# If the first approach fails, try with Kandinsky which might be smaller
if [ $? -ne 0 ]; then
  echo
  echo "Approach 2: Using kandinsky model at 512x512 resolution with memory optimizations"
  python memory_optimized_generator.py \
    --preset small_diverse \
    --model kandinsky \
    --image_size 512 \
    --batch_size 1 \
    --offload_to_cpu \
    --expand_segments \
    --output_dir data/synthetic/attempt2
fi

# If second approach fails, try with SDXL-Turbo but with very small size
if [ $? -ne 0 ]; then
  echo
  echo "Approach 3: Using sdxl-turbo with minimal image size and extreme memory optimizations"
  python memory_optimized_generator.py \
    --preset small_diverse \
    --model sdxl-turbo \
    --image_size 256 \
    --batch_size 1 \
    --offload_to_cpu \
    --expand_segments \
    --output_dir data/synthetic/attempt3
fi

# Nuclear option: Use CPU (will be very slow but should work)
if [ $? -ne 0 ]; then
  echo
  echo "Approach 4: Using CPU fallback mode (very slow but guaranteed to work)"
  python memory_optimized_generator.py \
    --preset small_diverse \
    --model sd-food \
    --image_size 512 \
    --use_cpu \
    --output_dir data/synthetic/attempt4
fi

# Check final result
if [ $? -eq 0 ]; then
  echo
  echo "Success! Dataset generation completed."
  echo "You can now use this dataset for training your pizza recognition model."
else
  echo
  echo "All approaches failed. Please check the logs for more information."
  echo "Consider freeing up more GPU memory or running on a machine with more VRAM."
fi
