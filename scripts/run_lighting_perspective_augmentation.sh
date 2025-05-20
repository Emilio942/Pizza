#!/bin/zsh

# Directory containing pizza images
INPUT_DIR="/home/emilio/Documents/ai/pizza/augmented_pizza/raw"

# Output directory for augmented images
OUTPUT_DIR="/home/emilio/Documents/ai/pizza/augmented_pizza/lighting_perspective"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the augmentation script with lighting and perspective augmentations
echo "Running augmentations with lighting and perspective effects..."
python3 /home/emilio/Documents/ai/pizza/scripts/augment_dataset.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-per-image 5 \
  --aug-types lighting,perspective \
  --preview \
  --save-stats

echo "Done!"
