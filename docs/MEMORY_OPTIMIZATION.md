# Pizza Dataset Generation - Memory Optimization Guide

This guide provides solutions for running the Stable Diffusion-based pizza dataset generation system on GPUs with limited VRAM, specifically targeting the NVIDIA GeForce RTX 3060 with 12GB memory.

## Quick Start

The easiest way to generate the dataset on limited VRAM is to run:

```bash
chmod +x run_memory_optimized.sh
./run_memory_optimized.sh
```

This script will try multiple approaches in sequence until one works.

## Memory Optimization Options

If you want more control, you can run the memory-optimized generator directly:

```bash
python memory_optimized_generator.py --preset small_diverse --model sd-food --image_size 512 --batch_size 1 --offload_to_cpu --expand_segments
```

### Key Parameters to Reduce Memory Usage

1. **Model Selection**
   - `--model sd-food` - Uses a smaller SD-based model instead of SDXL
   - `--model kandinsky` - Alternative diffusion model that might use less memory
   - `--model sdxl-turbo` - Faster SDXL variant that might use slightly less memory

2. **Image Size**
   - `--image_size 512` - Generate 512x512 images instead of 1024x1024
   - `--image_size 256` - For extreme memory constraints (lower quality)

3. **Batch Size**
   - `--batch_size 1` - Process one image at a time (minimum memory usage)

4. **Memory Optimizations**
   - `--offload_to_cpu` - Moves model components to CPU when not in use
   - `--expand_segments` - Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   - `--use_cpu` - Last resort: Run on CPU (very slow but guaranteed to work)

5. **Dataset Size**
   - `--preset small_diverse` - Generates only 120 images (smallest preset)

## Advanced Memory Optimization Techniques

If you're still facing memory issues, here are additional techniques:

### 1. Clear VRAM Before Running

```bash
# Clear CUDA cache in Python
python -c "import torch; torch.cuda.empty_cache()"

# Kill any GPU processes that might be using memory
sudo fuser -v /dev/nvidia*
```

### 2. Monitor GPU Memory Usage

Install nvidia-smi monitoring in a separate terminal:

```bash
watch -n 1 nvidia-smi
```

### 3. System-Level Optimizations

```bash
# Limit Python memory growth
export MALLOC_TRIM_THRESHOLD_=100000

# Disable CUDA Graph capturing which can use extra memory
export CUDA_DISABLE_GRAPH_CAPTURE=1
```

### 4. Use Swap Space as Backup

Increase swap space to help handle memory pressure:

```bash
# Add 8GB of swap (adjust path as needed)
sudo dd if=/dev/zero of=/swapfile bs=1G count=8
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Troubleshooting

### If you still get "CUDA out of memory" errors:

1. Try reducing the image size further: `--image_size 128`
2. Use CPU mode: `--use_cpu`
3. Try a smaller model from Hugging Face by modifying the MODEL_PRESETS in the generator code

### Other common issues:

- **Slow generation on CPU**: This is normal, diffusion models are compute-intensive
- **Image quality issues with small sizes**: You can upscale later with Super Resolution
- **Failed downloads**: Check your internet connection or use `--resume_download`

## Memory Consumption Reference

Approximate VRAM usage for different configurations:

| Model     | Image Size | Batch Size | VRAM Usage |
|-----------|------------|------------|------------|
| sdxl      | 1024       | 4          | >12GB      |
| sdxl      | 1024       | 1          | ~11GB      |
| sdxl      | 512        | 1          | ~10GB      |
| sdxl-turbo| 512        | 1          | ~9GB       |
| sd-food   | 512        | 1          | ~7GB       |
| kandinsky | 512        | 1          | ~6GB       |
| sd-food   | 256        | 1          | ~4GB       |

## Need More Help?

If you continue to experience issues, please:

1. Check the log files (`memory_optimized_generator.log` and `diffusion_control.log`)
2. Try running on a cloud GPU instance with more VRAM
3. Consider using a pre-generated dataset if available
