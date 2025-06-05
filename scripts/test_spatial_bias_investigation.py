#!/usr/bin/env python3
"""
Targeted test for Spatial-MLLM prompting bias investigation
Focus on testing the model's response patterns with different prompt formats
"""

import os
import torch
import time
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

def test_multiple_choice_variants(model, tokenizer, processor, image, device):
    """Test different multiple choice formats to see if position bias exists"""
    
    print("\n📊 Testing Multiple Choice Position Bias")
    print("-" * 50)
    
    # Test 1: Original order (A=basic, B=burnt, C=mixed, D=ready)
    print("1️⃣  Original order (A=basic, B=burnt, C=mixed, D=ready):")
    prompt1 = """Look at this pizza image and analyze its cooking state.

Which cooking state best describes this pizza?

A) basic - Raw or undercooked pizza with pale dough and unmelted cheese
B) burnt - Overcooked pizza with dark/black areas and charred edges  
C) mixed - Pizza with mixed cooking levels (some areas cooked, others not)
D) ready - Perfectly cooked pizza with golden-brown crust and melted cheese

Answer:"""
    
    response1 = run_inference(model, tokenizer, processor, image, prompt1, device)
    print(f"   Response: {response1}")
    
    # Test 2: Reversed order (A=ready, B=mixed, C=burnt, D=basic)
    print("\n2️⃣  Reversed order (A=ready, B=mixed, C=burnt, D=basic):")
    prompt2 = """Look at this pizza image and analyze its cooking state.

Which cooking state best describes this pizza?

A) ready - Perfectly cooked pizza with golden-brown crust and melted cheese
B) mixed - Pizza with mixed cooking levels (some areas cooked, others not)
C) burnt - Overcooked pizza with dark/black areas and charred edges
D) basic - Raw or undercooked pizza with pale dough and unmelted cheese

Answer:"""
    
    response2 = run_inference(model, tokenizer, processor, image, prompt2, device)
    print(f"   Response: {response2}")
    
    # Test 3: Random order 1 (A=burnt, B=ready, C=basic, D=mixed)
    print("\n3️⃣  Random order 1 (A=burnt, B=ready, C=basic, D=mixed):")
    prompt3 = """Look at this pizza image and analyze its cooking state.

Which cooking state best describes this pizza?

A) burnt - Overcooked pizza with dark/black areas and charred edges
B) ready - Perfectly cooked pizza with golden-brown crust and melted cheese
C) basic - Raw or undercooked pizza with pale dough and unmelted cheese
D) mixed - Pizza with mixed cooking levels (some areas cooked, others not)

Answer:"""
    
    response3 = run_inference(model, tokenizer, processor, image, prompt3, device)
    print(f"   Response: {response3}")

def test_format_variations(model, tokenizer, processor, image, device):
    """Test different answer format expectations"""
    
    print("\n📝 Testing Answer Format Variations")
    print("-" * 50)
    
    # Test 1: Numbers instead of letters
    print("1️⃣  Using numbers (1/2/3/4):")
    prompt1 = """Look at this pizza image and analyze its cooking state.

Which cooking state best describes this pizza?

1) basic - Raw or undercooked pizza with pale dough and unmelted cheese
2) burnt - Overcooked pizza with dark/black areas and charred edges  
3) mixed - Pizza with mixed cooking levels (some areas cooked, others not)
4) ready - Perfectly cooked pizza with golden-brown crust and melted cheese

Answer:"""
    
    response1 = run_inference(model, tokenizer, processor, image, prompt1, device)
    print(f"   Response: {response1}")
    
    # Test 2: No explicit options, just description
    print("\n2️⃣  Descriptive without options:")
    prompt2 = """Look at this pizza image. Describe the cooking state of this pizza. Choose from: basic (undercooked), burnt (overcooked), mixed (uneven cooking), or ready (perfectly cooked)."""
    
    response2 = run_inference(model, tokenizer, processor, image, prompt2, device)
    print(f"   Response: {response2}")
    
    # Test 3: Very simple question
    print("\n3️⃣  Simple binary question:")
    prompt3 = """Look at this pizza. Is it cooked? Answer YES or NO."""
    
    response3 = run_inference(model, tokenizer, processor, image, prompt3, device)
    print(f"   Response: {response3}")

def test_image_descriptions(model, tokenizer, processor, image, device):
    """Test general image description capabilities"""
    
    print("\n🖼️  Testing General Image Description")
    print("-" * 50)
    
    # Test 1: General description
    print("1️⃣  General description:")
    prompt1 = """Describe what you see in this image."""
    
    response1 = run_inference(model, tokenizer, processor, image, prompt1, device, max_tokens=50)
    print(f"   Response: {response1}")
    
    # Test 2: Specific pizza features
    print("\n2️⃣  Pizza-specific description:")
    prompt2 = """Look at this pizza image. Describe the color of the crust, the state of the cheese, and any toppings you can see."""
    
    response2 = run_inference(model, tokenizer, processor, image, prompt2, device, max_tokens=50)
    print(f"   Response: {response2}")

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

def main():
    """Main test function"""
    
    print("🔍 Spatial-MLLM Bias Investigation")
    print("=" * 50)
    
    # Load model
    try:
        model, tokenizer, processor, device = load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load test image
    test_image_dir = "/home/emilio/Documents/ai/pizza/data/test/basic"
    if not os.path.exists(test_image_dir):
        print(f"❌ Test directory not found: {test_image_dir}")
        return
    
    image_files = [f for f in os.listdir(test_image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {test_image_dir}")
        return
    
    test_image_path = os.path.join(test_image_dir, image_files[0])
    image = Image.open(test_image_path).convert('RGB')
    
    print(f"🖼️  Test image: {image_files[0]} (from basic class)")
    print(f"    Image size: {image.size}")
    
    # Run tests
    test_multiple_choice_variants(model, tokenizer, processor, image, device)
    test_format_variations(model, tokenizer, processor, image, device)
    test_image_descriptions(model, tokenizer, processor, image, device)
    
    print("\n✅ Bias investigation completed!")
    print("\nKey findings will help identify the root cause of the D-bias issue.")

if __name__ == "__main__":
    main()
