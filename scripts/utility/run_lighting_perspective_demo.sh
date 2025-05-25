#!/bin/zsh

# Find a pizza image to test with
PIZZA_IMG=$(find /home/emilio/Documents/ai/pizza/augmented_pizza/raw -type f -name "*.jpg" | head -n 1)

if [ -z "$PIZZA_IMG" ]; then
  echo "No pizza image found in the raw directory. Looking in other directories..."
  PIZZA_IMG=$(find /home/emilio/Documents/ai/pizza/augmented_pizza -type f -name "*.jpg" | head -n 1)
fi

if [ -z "$PIZZA_IMG" ]; then
  echo "No pizza image found for testing."
  exit 1
fi

echo "Using pizza image: $PIZZA_IMG"

# Create output directory if it doesn't exist
OUTPUT_DIR="/home/emilio/Documents/ai/pizza/augmented_pizza/lighting_perspective_test"
mkdir -p "$OUTPUT_DIR"

# Run the demo script
python3 /home/emilio/Documents/ai/pizza/scripts/demo_lighting_perspective.py --image "$PIZZA_IMG" --output-dir "$OUTPUT_DIR"

echo "Done!"
