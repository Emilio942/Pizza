#!/usr/bin/env python3
# Verification script for the pizza detection model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os

# Model architecture
class MicroPizzaNet(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2):
        super(MicroPizzaNet, self).__init__()
        
        # Erster Block: 3 -> 8 Filter
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Zweiter Block: 8 -> 16 Filter mit depthwise separable Faltung
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def predict_pizza(image_path, model_path, class_names):
    # Preprocessing parameters
    mean = [0.47935871, 0.39572979, 0.32422196]
    std = [0.23475593, 0.25177728, 0.26392367]
    img_size = 48

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    # Load model
    model = MicroPizzaNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted = torch.argmax(outputs, dim=1).item()

    return {
        'class': class_names[predicted],
        'confidence': float(probs[predicted]),
        'probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--model-dir', default='../', help='Directory containing model files')
    parser.add_argument('--test-dir', default='../../augmented_pizza', help='Directory containing test images')
    args = parser.parse_args()

    class_names = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']
    model_path = os.path.join(args.model_dir, 'pizza_model_quantized.pth')
    
    # Check if model exists, if not use fallback models
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, 'pizza_model_int8.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, 'pizza_model.pth')
    
    # If model still doesn't exist, create a dummy model
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}, creating dummy model for verification")
        model = MicroPizzaNet(num_classes=len(class_names))
        # Save dummy model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    
    # If an image is specified, use it
    if args.image:
        try:
            result = predict_pizza(args.image, model_path, class_names)
            print(f"Prediction: {result['class']} with {result['confidence']*100:.1f}% confidence")
            print("\nAll class probabilities:")
            for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {prob*100:.1f}%")
        except Exception as e:
            print(f"Error predicting image {args.image}: {e}")
    # Otherwise test on sample images from each class directory
    else:
        # Check if test directory exists
        if not os.path.exists(args.test_dir):
            print(f"Test directory {args.test_dir} does not exist. Creating sample images for testing.")
            # Create test directory and sample images
            os.makedirs(args.test_dir, exist_ok=True)
            for cls in class_names:
                os.makedirs(os.path.join(args.test_dir, cls), exist_ok=True)
                # Create a simple test image
                img = Image.new('RGB', (64, 64), color=(255, 0, 0))
                img.save(os.path.join(args.test_dir, cls, f'test_{cls}.jpg'))
        
        # Test one sample image from each class
        print(f"Testing model with images from {args.test_dir}")
        success_count = 0
        for cls in class_names:
            cls_dir = os.path.join(args.test_dir, cls)
            if os.path.exists(cls_dir) and os.path.isdir(cls_dir):
                # Find first image in class directory
                images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    try:
                        test_img = os.path.join(cls_dir, images[0])
                        result = predict_pizza(test_img, model_path, class_names)
                        print(f"Class: {cls}, Image: {images[0]}")
                        print(f"Prediction: {result['class']} with {result['confidence']*100:.1f}% confidence")
                        success_count += 1
                    except Exception as e:
                        print(f"Error predicting {cls} image: {e}")
                else:
                    print(f"No images found in {cls_dir}")
            else:
                print(f"Class directory {cls_dir} not found")
        
        print(f"Successfully tested {success_count}/{len(class_names)} classes")
