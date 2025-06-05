#!/usr/bin/env python3
"""
Spatial-MLLM Pizza Classification Test
Comprehensive testing of the Spatial-MLLM model for pizza cooking state classification
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

def classify_pizza_image(model, tokenizer, processor, image_path, device="cuda"):
    """Classify a single pizza image using the Spatial-MLLM model"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create multiple-choice prompt to match model's training format
    prompt = """Look at this pizza image and analyze its cooking state.

Which cooking state best describes this pizza?

A) basic - Raw or undercooked pizza with pale dough and unmelted cheese
B) burnt - Overcooked pizza with dark/black areas and charred edges  
C) mixed - Pizza with mixed cooking levels (some areas cooked, others not)
D) ready - Perfectly cooked pizza with golden-brown crust and melted cheese

Answer:"""
    
    # Process inputs using Qwen VL chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Process with processor
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Shorter since we expect A/B/C/D answer
            do_sample=False,  # Use greedy decoding for consistency
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    if hasattr(outputs, 'sequences'):
        full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    else:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the input prompt)
    if "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

def evaluate_pizza_classification(model, tokenizer, processor, test_data_dir, device="cuda"):
    """Evaluate pizza classification on test dataset"""
    
    print(f"\n🔍 Evaluating Pizza Classification Performance")
    print("=" * 60)
    
    # Define pizza classes
    pizza_classes = ["basic", "burnt", "mixed", "combined"]
    results = {}
    
    for pizza_class in pizza_classes:
        print(f"\n📂 Testing {pizza_class.upper()} class...")
        
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
        
        # Test on random sample (max 5 images per class)
        sample_size = min(5, len(image_files))
        sample_images = random.sample(image_files, sample_size)
        
        class_results = []
        
        for i, img_file in enumerate(sample_images):
            img_path = os.path.join(class_dir, img_file)
            print(f"   🖼️  Testing image {i+1}/{sample_size}: {img_file}")
            
            start_time = time.time()
            
            try:
                response = classify_pizza_image(
                    model, tokenizer, processor, img_path, device
                )
                
                inference_time = time.time() - start_time
                
                # Extract predicted class from response
                response_lower = response.lower()
                predicted_class = "unknown"
                
                # Look for A/B/C/D answers and answer tags
                if "<answer>a</answer>" in response_lower or "answer: a" in response_lower or response_lower.strip().startswith("a"):
                    predicted_class = "basic"
                elif "<answer>b</answer>" in response_lower or "answer: b" in response_lower or response_lower.strip().startswith("b"):
                    predicted_class = "burnt"
                elif "<answer>c</answer>" in response_lower or "answer: c" in response_lower or response_lower.strip().startswith("c"):
                    predicted_class = "mixed"
                elif "<answer>d</answer>" in response_lower or "answer: d" in response_lower or response_lower.strip().startswith("d"):
                    predicted_class = "combined"
                # Fallback to keyword matching
                elif "basic" in response_lower:
                    predicted_class = "basic"
                elif "burnt" in response_lower:
                    predicted_class = "burnt"
                elif "mixed" in response_lower:
                    predicted_class = "mixed"
                elif "ready" in response_lower:
                    predicted_class = "combined"
                
                # Map "combined" to "ready" for comparison (both refer to well-cooked pizza)
                if pizza_class == "combined" and predicted_class == "ready":
                    predicted_class = "combined"
                
                # Check if prediction is correct
                is_correct = predicted_class == pizza_class
                
                result = {
                    "image": img_file,
                    "true_class": pizza_class,
                    "predicted_class": predicted_class,
                    "correct": is_correct,
                    "inference_time": inference_time,
                    "full_response": response
                }
                
                class_results.append(result)
                
                status = "✅" if is_correct else "❌"
                print(f"     {status} Predicted: {predicted_class} | Time: {inference_time:.2f}s")
                print(f"     Response: {response[:100]}...")
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                result = {
                    "image": img_file,
                    "true_class": pizza_class,
                    "predicted_class": "error",
                    "correct": False,
                    "inference_time": 0,
                    "error": str(e)
                }
                class_results.append(result)
        
        # Calculate class accuracy
        correct_predictions = sum(1 for r in class_results if r["correct"])
        class_accuracy = correct_predictions / len(class_results) if class_results else 0
        avg_inference_time = np.mean([r["inference_time"] for r in class_results]) if class_results else 0
        
        results[pizza_class] = {
            "accuracy": class_accuracy,
            "correct": correct_predictions,
            "total": len(class_results),
            "avg_inference_time": avg_inference_time,
            "results": class_results
        }
        
        print(f"   📊 Class Accuracy: {class_accuracy:.2%} ({correct_predictions}/{len(class_results)})")
        print(f"   ⏱️  Avg Inference Time: {avg_inference_time:.2f}s")
    
    # Calculate overall accuracy
    total_correct = sum(r["correct"] for class_results in results.values() for r in class_results["results"])
    total_tested = sum(len(class_results["results"]) for class_results in results.values())
    overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"📊 OVERALL EVALUATION RESULTS")
    print(f"=" * 60)
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_tested})")
    print(f"Classes Tested: {len(results)}")
    
    # Per-class summary
    for class_name, class_data in results.items():
        print(f"{class_name.capitalize()}: {class_data['accuracy']:.2%} ({class_data['correct']}/{class_data['total']})")
    
    return results

def main():
    """Main function to run pizza classification evaluation"""
    
    print("🍕 SPATIAL-MLLM PIZZA CLASSIFICATION EVALUATION")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_dir = "/home/emilio/Documents/ai/pizza/data/test"
    output_dir = "/home/emilio/Documents/ai/pizza/output/spatial_pizza_evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Test Data: {test_data_dir}")
    print(f"Output: {output_dir}")
    
    # Load model
    try:
        model, tokenizer, processor = load_spatial_model(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Run evaluation
    try:
        results = evaluate_pizza_classification(
            model, tokenizer, processor, test_data_dir, device
        )
        
        # Save results
        results_file = os.path.join(output_dir, "pizza_classification_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        # Success if overall accuracy > 60%
        total_correct = sum(r["correct"] for class_results in results.values() for r in class_results["results"])
        total_tested = sum(len(class_results["results"]) for class_results in results.values())
        overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
        
        if overall_accuracy >= 0.6:
            print(f"\n🎉 Evaluation PASSED! Accuracy: {overall_accuracy:.2%}")
            return 0
        else:
            print(f"\n⚠️  Evaluation needs improvement. Accuracy: {overall_accuracy:.2%}")
            return 1
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
