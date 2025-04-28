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
    mean = [0.47935871 0.39572979 0.32422196]
    std = [0.23475593 0.25177728 0.26392367]
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
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', default='../pizza_model_int8.pth', help='Path to model file')
    args = parser.parse_args()

    class_names = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']

    result = predict_pizza(args.image, args.model, class_names)
    print(f"Prediction: {result['class']} with {result['confidence']*100:.1f}% confidence")
    print("\nAll class probabilities:")
    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob*100:.1f}%")
