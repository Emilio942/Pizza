#!/bin/bash
# Generate and integrate diffusion model images for underrepresented classes
# and difficult-to-capture scenarios in the pizza dataset

echo "Pizza Diffusion Model Integration"
echo "================================="
echo

# Make script executable
chmod +x scripts/integrate_diffusion_images.py

# First, analyze the dataset to identify underrepresented classes
echo "Step 1: Analyzing dataset distribution..."
python scripts/integrate_diffusion_images.py --analyze-dataset

# Ask user if they want to continue with generation
read -p "Continue with image generation? (y/n): " continue_gen

if [[ $continue_gen != "y" && $continue_gen != "Y" ]]; then
  echo "Exiting without generating images."
  exit 0
fi

# Ask which specific scenario to generate
echo
echo "Select a generation scenario:"
echo "1) Basic diverse set (all classes, few images)"
echo "2) Underrepresented classes (focus on classes with fewer images)"
echo "3) Special lighting conditions (harsh shadows, overhead lighting)"
echo "4) Specific burn patterns (edge burns, spotty burns)"
echo "5) Progression sequences (raw to burnt transition states)"
read -p "Enter your choice (1-5): " scenario

# Define model and parameters based on available GPU memory
# Default to memory-optimized settings for NVIDIA RTX 3060
MODEL="sd-food"
IMAGE_SIZE=512
BATCH_SIZE=1

case $scenario in
  1)
    echo "Generating basic diverse set..."
    python scripts/integrate_diffusion_images.py --generate --preset small_diverse --model $MODEL --image-size $IMAGE_SIZE
    ;;
  2)
    echo "Generating for underrepresented classes..."
    python scripts/integrate_diffusion_images.py --generate --preset training_focused --model $MODEL --image-size $IMAGE_SIZE
    ;;
  3)
    echo "Generating special lighting conditions..."
    # For lighting conditions, we use training_focused but will process the images differently
    python scripts/integrate_diffusion_images.py --generate --preset training_focused --model $MODEL --image-size $IMAGE_SIZE
    ;;
  4)
    echo "Generating specific burn patterns..."
    python scripts/integrate_diffusion_images.py --generate --preset burn_patterns --model $MODEL --image-size $IMAGE_SIZE
    ;;
  5)
    echo "Generating progression sequences..."
    python scripts/integrate_diffusion_images.py --generate --preset progression_heavy --model $MODEL --image-size $IMAGE_SIZE
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

# Organize generated images into class directories
echo
echo "Step 3: Organizing generated images..."
python scripts/integrate_diffusion_images.py --organize-images

# Create quality control report
echo
echo "Step 4: Creating quality control report..."
python scripts/integrate_diffusion_images.py --quality-report

echo
echo "Quality control report created. Please review the images before proceeding."
echo "Look for:  1) Image quality issues  2) Correct classification  3) Diverse representation"
echo

# Ask user if they want to continue with integration
read -p "Continue with dataset integration? (y/n): " continue_int

if [[ $continue_int != "y" && $continue_int != "Y" ]]; then
  echo "Exiting without integrating images. You can integrate them later with:"
  echo "  python scripts/integrate_diffusion_images.py --integrate"
  exit 0
fi

# Integrate images into main dataset
echo
echo "Step 5: Integrating images into main dataset..."
python scripts/integrate_diffusion_images.py --integrate

echo
echo "Process complete! The diffusion-generated images have been integrated into the dataset."
echo "You can now train a model using the enhanced dataset."
