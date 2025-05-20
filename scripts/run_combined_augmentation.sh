#!/bin/zsh

# Directory containing pizza images
INPUT_DIR="/home/emilio/Documents/ai/pizza/augmented_pizza/raw"

# Output directory for augmented images 
OUTPUT_DIR="/home/emilio/Documents/ai/pizza/augmented_pizza/combined_lighting_perspective"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Add the combined augmentation type by creating a temporary modified version of augment_dataset.py
TEMP_SCRIPT="/home/emilio/Documents/ai/pizza/scripts/augment_dataset_with_combined.py"

# Create the temporary script with the combined augmentation type added
cat /home/emilio/Documents/ai/pizza/scripts/augment_dataset.py | sed 's/perspective,all/perspective,combined,all/g' > "$TEMP_SCRIPT"

# Add the combined augmentation type functionality
cat >> "$TEMP_SCRIPT" << 'EOF'

# Add the combined augmentation type
        elif aug_type == 'combined' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            augmentation_functions[aug_type] = lambda img, d=device: apply_combined_light_perspective_augmentation(img, d)
EOF

# Make the temporary script executable
chmod +x "$TEMP_SCRIPT"

# Run the augmentation script with the combined augmentations
echo "Running augmentations with combined lighting and perspective effects..."
python3 "$TEMP_SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-per-image 5 \
  --aug-types combined \
  --preview \
  --save-stats

# Clean up
echo "Cleaning up..."
rm "$TEMP_SCRIPT"

echo "Done!"
