#!/usr/bin/env python3
"""
Improved Prompting Strategies for Spatial-MLLM Pizza Classification
Testing different prompting approaches to overcome the model's bias toward option D
"""

import os
import sys
import torch
import time
import json
import random
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

def load_spatial_model(device="cuda"):
    """Load the Spatial-MLLM model for pizza classification"""
    model_name = "Diankun/Spatial-MLLM-subset-sft"
    
    print(f"🔄 Loading Spatial-MLLM model...")
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, processor

def classify_pizza_open_ended(model, tokenizer, processor, image_path, device="cuda"):
    """Open-ended pizza classification without multiple choice"""
    
    image = Image.open(image_path).convert('RGB')
    
    prompt = """Look at this pizza image and describe its cooking state. Is it undercooked (raw/basic), perfectly cooked (ready), overcooked (burnt), or has mixed cooking levels? Respond with a single word: basic, ready, burnt, or mixed."""
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,  # Add some randomness
            pad_token_id=tokenizer.eos_token_id,
        )
    
    if hasattr(outputs, 'sequences'):
        full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    else:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

def classify_pizza_yes_no(model, tokenizer, processor, image_path, pizza_class, device="cuda"):
    """Binary classification approach - ask specific yes/no questions"""
    
    image = Image.open(image_path).convert('RGB')
    
    class_descriptions = {
        "basic": "undercooked with pale dough and unmelted cheese",
        "ready": "perfectly cooked with golden-brown crust and melted cheese", 
        "burnt": "overcooked with dark or charred areas",
        "mixed": "having mixed cooking levels with some areas cooked and others not"
    }
    
    prompt = f"""Look at this pizza image. Is this pizza {class_descriptions[pizza_class]}? Answer YES or NO."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    if hasattr(outputs, 'sequences'):
        full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    else:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

def classify_pizza_few_shot(model, tokenizer, processor, image_path, device="cuda"):
    """Few-shot prompting with examples"""
    
    image = Image.open(image_path).convert('RGB')
    
    prompt = """Here are examples of pizza cooking states:

Example 1: A pizza with pale, uncooked dough and unmelted cheese → basic
Example 2: A pizza with golden-brown crust and bubbly melted cheese → ready  
Example 3: A pizza with black charred edges and dark spots → burnt
Example 4: A pizza with some areas cooked and others undercooked → mixed

Now look at this pizza image. What cooking state is it? Answer with one word: basic, ready, burnt, or mixed."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    if hasattr(outputs, 'sequences'):
        full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    else:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

def classify_pizza_randomized_options(model, tokenizer, processor, image_path, device="cuda"):
    """Multiple choice with randomized option order to reduce positional bias"""
    
    image = Image.open(image_path).convert('RGB')
    
    # Randomize the order of options
    options = [
        ("basic", "Raw or undercooked pizza with pale dough and unmelted cheese"),
        ("burnt", "Overcooked pizza with dark/black areas and charred edges"),
        ("mixed", "Pizza with mixed cooking levels (some areas cooked, others not)"), 
        ("ready", "Perfectly cooked pizza with golden-brown crust and melted cheese")
    ]
    
    random.shuffle(options)
    letters = ['A', 'B', 'C', 'D']
    
    prompt = "Look at this pizza image and analyze its cooking state.\n\nWhich cooking state best describes this pizza?\n\n"
    option_map = {}
    
    for i, (class_name, description) in enumerate(options):
        prompt += f"{letters[i]}) {class_name} - {description}\n"
        option_map[letters[i]] = class_name
    
    prompt += "\nAnswer:"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    if hasattr(outputs, 'sequences'):
        full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    else:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        response = full_response.strip()
    
    return response, option_map

def parse_response(response, method="open_ended", option_map=None):
    """Parse model response to extract predicted class"""
    response_lower = response.lower().strip()
    
    if method == "open_ended" or method == "few_shot":
        # Look for direct class names
        if "basic" in response_lower:
            return "basic"
        elif "ready" in response_lower:
            return "ready" 
        elif "burnt" in response_lower:
            return "burnt"
        elif "mixed" in response_lower:
            return "mixed"
        else:
            return "unknown"
    
    elif method == "yes_no":
        if "yes" in response_lower:
            return "yes"
        elif "no" in response_lower:
            return "no"
        else:
            return "unknown"
    
    elif method == "randomized" and option_map:
        # Look for A/B/C/D answers
        for letter in ['A', 'B', 'C', 'D']:
            if f"<answer>{letter.lower()}</answer>" in response_lower or f"answer: {letter.lower()}" in response_lower or response_lower.strip().startswith(letter.lower()):
                return option_map.get(letter, "unknown")
        return "unknown"
    
    return "unknown"

def test_single_image_all_methods(model, tokenizer, processor, image_path, true_class, device="cuda"):
    """Test a single image with all prompting methods"""
    
    print(f"\n🖼️  Testing image: {os.path.basename(image_path)}")
    print(f"   True class: {true_class}")
    print("-" * 50)
    
    results = {}
    
    # Method 1: Open-ended
    print("1️⃣  Open-ended classification...")
    try:
        start_time = time.time()
        response = classify_pizza_open_ended(model, tokenizer, processor, image_path, device)
        inference_time = time.time() - start_time
        predicted = parse_response(response, "open_ended")
        
        results["open_ended"] = {
            "response": response,
            "predicted": predicted,
            "correct": predicted == true_class,
            "inference_time": inference_time
        }
        
        print(f"   Response: {response}")
        print(f"   Predicted: {predicted} {'✅' if predicted == true_class else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["open_ended"] = {"error": str(e)}
    
    # Method 2: Few-shot
    print("\n2️⃣  Few-shot classification...")
    try:
        start_time = time.time()
        response = classify_pizza_few_shot(model, tokenizer, processor, image_path, device)
        inference_time = time.time() - start_time
        predicted = parse_response(response, "few_shot")
        
        results["few_shot"] = {
            "response": response,
            "predicted": predicted,
            "correct": predicted == true_class,
            "inference_time": inference_time
        }
        
        print(f"   Response: {response}")
        print(f"   Predicted: {predicted} {'✅' if predicted == true_class else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["few_shot"] = {"error": str(e)}
    
    # Method 3: Randomized multiple choice
    print("\n3️⃣  Randomized multiple choice...")
    try:
        start_time = time.time()
        response, option_map = classify_pizza_randomized_options(model, tokenizer, processor, image_path, device)
        inference_time = time.time() - start_time
        predicted = parse_response(response, "randomized", option_map)
        
        results["randomized"] = {
            "response": response,
            "predicted": predicted,
            "correct": predicted == true_class,
            "inference_time": inference_time,
            "option_map": option_map
        }
        
        print(f"   Response: {response}")
        print(f"   Option map: {option_map}")
        print(f"   Predicted: {predicted} {'✅' if predicted == true_class else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["randomized"] = {"error": str(e)}
    
    # Method 4: Binary classification for true class
    print(f"\n4️⃣  Binary classification (Is it {true_class}?)...")
    try:
        start_time = time.time()
        response = classify_pizza_yes_no(model, tokenizer, processor, image_path, true_class, device)
        inference_time = time.time() - start_time
        predicted = parse_response(response, "yes_no")
        
        results["binary"] = {
            "response": response,
            "predicted": predicted,
            "correct": predicted == "yes",  # Should say "yes" if classification is correct
            "inference_time": inference_time,
            "question_class": true_class
        }
        
        print(f"   Response: {response}")
        print(f"   Predicted: {predicted} {'✅' if predicted == 'yes' else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["binary"] = {"error": str(e)}
    
    return results

def main():
    """Main testing function"""
    
    print("🍕 Spatial-MLLM Improved Prompting Strategy Test")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data_dir = "/home/emilio/Documents/ai/pizza/data/test"
    output_dir = "/home/emilio/Documents/ai/pizza/output/spatial_improved_prompting"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔧 Device: {device}")
    print(f"📂 Test data: {test_data_dir}")
    print(f"💾 Output: {output_dir}")
    
    # Load model
    try:
        model, tokenizer, processor = load_spatial_model(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test on sample images from each class
    pizza_classes = ["basic", "burnt", "mixed", "combined"]
    all_results = {}
    
    for pizza_class in pizza_classes:
        print(f"\n\n📂 Testing {pizza_class.upper()} class")
        print("=" * 40)
        
        class_dir = os.path.join(test_data_dir, pizza_class)
        if not os.path.exists(class_dir):
            print(f"   ⚠️  Directory not found: {class_dir}")
            continue
        
        # Get sample images 
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"   ⚠️  No images found in {class_dir}")
            continue
        
        # Test on 2 random images per class
        sample_size = min(2, len(image_files))
        sample_images = random.sample(image_files, sample_size)
        
        class_results = []
        
        for img_file in sample_images:
            img_path = os.path.join(class_dir, img_file)
            
            # Map "combined" to "ready" for model understanding
            test_class = "ready" if pizza_class == "combined" else pizza_class
            
            result = test_single_image_all_methods(
                model, tokenizer, processor, img_path, test_class, device
            )
            
            result["image"] = img_file
            result["true_class"] = pizza_class
            class_results.append(result)
        
        all_results[pizza_class] = class_results
    
    # Save results
    results_file = os.path.join(output_dir, "improved_prompting_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n✅ Results saved to: {results_file}")
    
    # Print summary
    print("\n📊 SUMMARY OF PROMPTING STRATEGIES")
    print("=" * 60)
    
    methods = ["open_ended", "few_shot", "randomized", "binary"]
    method_names = ["Open-ended", "Few-shot", "Randomized MC", "Binary"]
    
    for i, method in enumerate(methods):
        print(f"\n{method_names[i]}:")
        correct = 0
        total = 0
        
        for pizza_class, class_results in all_results.items():
            for result in class_results:
                if method in result and "error" not in result[method]:
                    total += 1
                    if result[method]["correct"]:
                        correct += 1
        
        if total > 0:
            accuracy = correct / total * 100
            print(f"   Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        else:
            print(f"   No valid results")

if __name__ == "__main__":
    main()
