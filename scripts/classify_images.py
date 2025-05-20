#!/usr/bin/env python3
"""
Script for classifying pizza images using a trained model.
This script loads a pre-trained pizza classification model and uses it
to classify one or more images, displaying the results.
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
import glob
import cv2
from torchvision import transforms

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.pizza_detector import MicroPizzaNet
from src.constants import INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, DEFAULT_CLASSES
from src.utils.types import InferenceResult

# Define colors for visualization
CLASS_COLORS = {
    'basic': (0, 255, 0),       # Green
    'burnt': (255, 0, 0),       # Red
    'combined': (255, 165, 0),  # Orange
    'mixed': (128, 0, 128),     # Purple
    'progression': (0, 0, 255), # Blue
    'segment': (255, 255, 0)    # Yellow
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Classify pizza images using a trained model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=os.path.join(project_root, 'models', 'pizza_model.pth'),
        help='Path to the trained model file (.pth)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image or directory of images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output directory for saving results'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json', 'image', 'all'],
        default='all',
        help='Output format for classification results'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display results on screen'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use for inference (cpu or cuda)'
    )
    
    return parser.parse_args()

def load_model(model_path, num_classes=len(DEFAULT_CLASSES), device=None):
    """Load the pre-trained model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create model
    model = MicroPizzaNet(num_classes=num_classes)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    return model, device

def preprocess_image(image_path, img_size=INPUT_SIZE):
    """Preprocess image for inference."""
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None
    
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    # Transform image
    img_tensor = transform(img)
    
    return img_tensor, img

def get_images_from_path(input_path):
    """Get list of image paths from input path (file or directory)."""
    if os.path.isfile(input_path):
        # Single file
        return [input_path]
    elif os.path.isdir(input_path):
        # Directory - find all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(input_path, f'*{ext.upper()}')))
        return sorted(image_files)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []

def perform_inference(model, img_tensor, device):
    """Perform inference with the model."""
    # Prepare image batch for inference
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    # Disable gradients for inference
    with torch.no_grad():
        start_time = time.time()
        
        # Perform prediction
        outputs = model(input_batch)
        
        # Calculate probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class
        _, predicted = torch.max(outputs, 1)
        
        inference_time = (time.time() - start_time) * 1000  # ms
    
    # Convert to Python types
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item()
    
    # Create dictionary for all class probabilities
    class_probs = {DEFAULT_CLASSES[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    # Create InferenceResult object
    result = InferenceResult(
        class_name=DEFAULT_CLASSES[predicted_class],
        confidence=confidence,
        probabilities=class_probs,
        prediction=predicted_class
    )
    
    return result, inference_time

def annotate_image(image, result, draw_confidence=True):
    """Add classification results to an image."""
    # Convert PIL image to NumPy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    annotated = image.copy()
    
    # Frame color based on class
    color = CLASS_COLORS[result.class_name]
    
    # Draw frame
    height, width = image.shape[:2]
    thickness = max(2, int(min(height, width) / 200))
    cv2.rectangle(
        annotated,
        (0, 0),
        (width-1, height-1),
        color,
        thickness
    )
    
    # Text size based on image size
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(font_scale * 2))
    
    # Class name
    text = result.class_name
    if draw_confidence:
        text += f' ({result.confidence:.1%})'
    
    # Text background
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (text_width + 10, text_height + 10),
        color,
        -1  # Filled
    )
    
    # Text
    cv2.putText(
        annotated,
        text,
        (5, text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White
        font_thickness
    )
    
    return annotated

def save_result_as_image(annotated_image, output_path):
    """Save annotated image to file."""
    try:
        # Convert from RGB to BGR for OpenCV
        if isinstance(annotated_image, np.ndarray) and annotated_image.ndim == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def display_image(image, title=None):
    """Display image with matplotlib."""
    plt.figure(figsize=(10, 8))
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model, device = load_model(args.model, device=device)
    
    # Get image paths
    image_paths = get_images_from_path(args.input)
    
    if not image_paths:
        print("No images found. Exiting.")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        
    # Create a list to store results for JSON output
    all_results = []
    
    # Process each image
    for image_path in image_paths:
        img_tensor, img = preprocess_image(image_path)
        
        if img_tensor is None:
            print(f"Skipping {image_path} due to preprocessing error")
            continue
        
        # Perform inference
        result, inference_time = perform_inference(model, img_tensor, device)
        
        # Get base filename
        filename = os.path.basename(image_path)
        
        # Print text result
        if args.format in ['text', 'all']:
            print(f"\nImage: {filename}")
            print(f"Classification: {result.class_name} with {result.confidence:.2%} confidence")
            print(f"Inference time: {inference_time:.2f} ms")
            print("Class probabilities:")
            for class_name, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.2%}")
        
        # Store result for JSON output
        result_dict = {
            'filename': filename,
            'path': image_path,
            'class': result.class_name,
            'confidence': result.confidence,
            'probabilities': result.probabilities,
            'inference_time_ms': inference_time
        }
        all_results.append(result_dict)
        
        # Create annotated image
        annotated = annotate_image(img, result)
        
        # Save image result
        if args.format in ['image', 'all'] and args.output:
            output_filename = f"{os.path.splitext(filename)[0]}_classified.jpg"
            output_path = os.path.join(args.output, output_filename)
            save_result_as_image(annotated, output_path)
            print(f"Saved annotated image to: {output_path}")
        
        # Display result
        if not args.no_display:
            display_image(annotated, f"{filename} - {result.class_name} ({result.confidence:.2%})")
    
    # Save JSON output
    if args.format in ['json', 'all'] and args.output:
        json_path = os.path.join(args.output, 'classification_results.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON results to: {json_path}")
    
    print(f"\nClassification complete: {len(all_results)} images processed")

if __name__ == "__main__":
    main()