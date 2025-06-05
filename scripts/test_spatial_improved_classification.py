#!/usr/bin/env python3
"""
Improved Pizza Classification Strategy for Spatial-MLLM
Based on bias investigation findings, implement better prompting approaches
"""

import os
import torch
import time
import json
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

def load_model():
    """Load the Spatial-MLLM model"""
    model_name = "Diankun/Spatial-MLLM-subset-sft"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    return model, tokenizer, processor, device

def classify_pizza_binary_cascade(model, tokenizer, processor, image_path, device):
    """Use binary cascade approach - ask specific yes/no questions in sequence"""
    
    image = Image.open(image_path).convert('RGB')
    
    # Step 1: Is it cooked at all?
    prompt1 = """Look at this pizza image. Is this pizza cooked (has melted cheese and some browning) or raw/uncooked (pale dough, unmelted cheese)? Answer COOKED or RAW."""
    
    response1 = run_inference(model, tokenizer, processor, image, prompt1, device, max_tokens=15)
    
    if "raw" in response1.lower() or "uncooked" in response1.lower():
        return "basic", response1
    
    # Step 2: If cooked, is it overcooked?
    prompt2 = """Look at this pizza image. Is this pizza burnt or overcooked (dark/black areas, charred edges)? Answer YES or NO."""
    
    response2 = run_inference(model, tokenizer, processor, image, prompt2, device, max_tokens=10)
    
    if "yes" in response2.lower():
        return "burnt", response2
    
    # Step 3: If not burnt, is cooking even?
    prompt3 = """Look at this pizza image. Does this pizza have uneven cooking (some areas more cooked than others, mixed browning)? Answer YES or NO."""
    
    response3 = run_inference(model, tokenizer, processor, image, prompt3, device, max_tokens=10)
    
    if "yes" in response3.lower():
        return "mixed", response3
    
    # Default: well-cooked
    return "ready", response3

def classify_pizza_descriptive(model, tokenizer, processor, image_path, device):
    """Use descriptive approach focused on visual cues"""
    
    image = Image.open(image_path).convert('RGB')
    
    prompt = """Look at this pizza image carefully. Describe the cooking state by examining:
1. Crust color: Is it pale (undercooked), golden-brown (perfect), or dark/black (burnt)?
2. Cheese state: Is it unmelted (raw), bubbly (perfect), or dark/charred (burnt)?
3. Overall appearance: Even cooking or mixed areas?

Based on these observations, classify as: BASIC (undercooked), READY (perfect), BURNT (overcooked), or MIXED (uneven)."""
    
    response = run_inference(model, tokenizer, processor, image, prompt, device, max_tokens=50)
    
    # Parse response for classification
    response_lower = response.lower()
    if "basic" in response_lower:
        return "basic", response
    elif "burnt" in response_lower:
        return "burnt", response
    elif "mixed" in response_lower:
        return "mixed", response
    elif "ready" in response_lower or "perfect" in response_lower:
        return "ready", response
    else:
        # Fallback to keyword analysis
        if any(word in response_lower for word in ["pale", "uncooked", "raw", "undercooked"]):
            return "basic", response
        elif any(word in response_lower for word in ["burnt", "charred", "dark", "black", "overcooked"]):
            return "burnt", response
        elif any(word in response_lower for word in ["mixed", "uneven", "some", "partial"]):
            return "mixed", response
        else:
            return "ready", response

def classify_pizza_improved_mc(model, tokenizer, processor, image_path, device):
    """Improved multiple choice with randomized order and better descriptions"""
    
    image = Image.open(image_path).convert('RGB')
    
    # Define options with more specific descriptions
    options = [
        ("basic", "Undercooked - Pale white/yellow dough, cheese not fully melted"),
        ("ready", "Well-cooked - Golden brown crust, cheese melted and bubbly"),
        ("burnt", "Overcooked - Dark brown/black areas, charred spots"),
        ("mixed", "Uneven cooking - Some areas cooked well, others undercooked")
    ]
    
    # Randomize order
    random.shuffle(options)
    letters = ['A', 'B', 'C', 'D']
    
    prompt = """Examine this pizza image closely and focus on the visual cooking indicators.

Which cooking state matches what you observe?

"""
    
    option_map = {}
    for i, (class_name, description) in enumerate(options):
        prompt += f"{letters[i]}) {class_name.upper()} - {description}\n"
        option_map[letters[i]] = class_name
    
    prompt += "\nChoose the letter that best matches: A, B, C, or D"
    
    response = run_inference(model, tokenizer, processor, image, prompt, device, max_tokens=20)
    
    # Parse response
    response_upper = response.upper()
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_upper:
            predicted_class = option_map.get(letter, "unknown")
            return predicted_class, response
    
    return "unknown", response

def run_inference(model, tokenizer, processor, image, prompt, device, max_tokens=20):
    """Run inference with the given prompt"""
    
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
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more consistent responses
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

def evaluate_improved_strategies(model, tokenizer, processor, test_data_dir, device):
    """Evaluate all improved strategies on test dataset"""
    
    print(f"\n🔍 Evaluating Improved Classification Strategies")
    print("=" * 60)
    
    # Define pizza classes - map combined to ready for model
    pizza_classes = {
        "basic": "basic",
        "burnt": "burnt", 
        "mixed": "mixed",
        "combined": "ready"  # Map combined to ready for model understanding
    }
    
    strategies = [
        ("binary_cascade", "Binary Cascade"),
        ("descriptive", "Descriptive Analysis"),
        ("improved_mc", "Improved Multiple Choice")
    ]
    
    all_results = {}
    
    for strategy_id, strategy_name in strategies:
        print(f"\n🧪 Testing Strategy: {strategy_name}")
        print("-" * 40)
        
        strategy_results = {}
        total_correct = 0
        total_tested = 0
        
        for dir_class, model_class in pizza_classes.items():
            print(f"\n📂 Testing {dir_class.upper()} class...")
            
            class_dir = os.path.join(test_data_dir, dir_class)
            if not os.path.exists(class_dir):
                print(f"   ⚠️  Directory not found: {class_dir}")
                continue
            
            # Get sample images
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"   ⚠️  No images found in {class_dir}")
                continue
            
            # Test on 3 random images per class
            sample_size = min(3, len(image_files))
            sample_images = random.sample(image_files, sample_size)
            
            class_results = []
            class_correct = 0
            
            for i, img_file in enumerate(sample_images):
                img_path = os.path.join(class_dir, img_file)
                print(f"   🖼️  Testing {i+1}/{sample_size}: {img_file}")
                
                start_time = time.time()
                
                try:
                    if strategy_id == "binary_cascade":
                        predicted_class, response = classify_pizza_binary_cascade(
                            model, tokenizer, processor, img_path, device
                        )
                    elif strategy_id == "descriptive":
                        predicted_class, response = classify_pizza_descriptive(
                            model, tokenizer, processor, img_path, device
                        )
                    elif strategy_id == "improved_mc":
                        predicted_class, response = classify_pizza_improved_mc(
                            model, tokenizer, processor, img_path, device
                        )
                    
                    inference_time = time.time() - start_time
                    
                    # Check if prediction is correct
                    is_correct = predicted_class == model_class
                    
                    if is_correct:
                        class_correct += 1
                        total_correct += 1
                    
                    total_tested += 1
                    
                    result = {
                        "image": img_file,
                        "true_class": dir_class,
                        "model_expected": model_class,
                        "predicted_class": predicted_class,
                        "correct": is_correct,
                        "inference_time": inference_time,
                        "response": response
                    }
                    
                    class_results.append(result)
                    
                    status = "✅" if is_correct else "❌"
                    print(f"      {status} Predicted: {predicted_class} (Expected: {model_class})")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    continue
            
            class_accuracy = class_correct / len(class_results) if class_results else 0
            strategy_results[dir_class] = {
                "accuracy": class_accuracy,
                "correct": class_correct,
                "total": len(class_results),
                "results": class_results
            }
            
            print(f"   📊 Class accuracy: {class_correct}/{len(class_results)} ({class_accuracy*100:.1f}%)")
        
        overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
        strategy_results["overall"] = {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_tested
        }
        
        all_results[strategy_id] = {
            "name": strategy_name,
            "results": strategy_results
        }
        
        print(f"\n📊 {strategy_name} Overall: {total_correct}/{total_tested} ({overall_accuracy*100:.1f}%)")
    
    return all_results

def main():
    """Main evaluation function"""
    
    print("🍕 Spatial-MLLM Improved Pizza Classification")
    print("=" * 60)
    
    # Setup
    test_data_dir = "/home/emilio/Documents/ai/pizza/data/test"
    output_dir = "/home/emilio/Documents/ai/pizza/output/spatial_improved_classification"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Test data: {test_data_dir}")
    print(f"💾 Output: {output_dir}")
    
    # Load model
    try:
        model, tokenizer, processor, device = load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run evaluation
    results = evaluate_improved_strategies(model, tokenizer, processor, test_data_dir, device)
    
    # Save results
    results_file = os.path.join(output_dir, "improved_classification_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Print summary
    print("\n📊 STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    
    for strategy_id, strategy_data in results.items():
        strategy_name = strategy_data["name"]
        overall = strategy_data["results"]["overall"]
        accuracy = overall["accuracy"] * 100
        
        print(f"\n{strategy_name}:")
        print(f"   Overall accuracy: {overall['correct']}/{overall['total']} ({accuracy:.1f}%)")
        
        # Show per-class results
        for class_name in ["basic", "burnt", "mixed", "combined"]:
            if class_name in strategy_data["results"]:
                class_data = strategy_data["results"][class_name]
                class_acc = class_data["accuracy"] * 100
                print(f"   {class_name}: {class_data['correct']}/{class_data['total']} ({class_acc:.1f}%)")

if __name__ == "__main__":
    main()
