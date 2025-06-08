\
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch.quantization
import torch.multiprocessing
from pathlib import Path # Added import
import logging # Added import
import torchvision.transforms as T # Added import

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path("/home/emilio/Documents/ai/pizza")
DATA_DIR = BASE_DIR / "output/verification_data/prepared_dataset"
MODEL_OUTPUT_DIR = BASE_DIR / "models/verification"
TRAIN_FILE = DATA_DIR / "pizza_verifier_train.json"
VALID_FILE = DATA_DIR / "pizza_verifier_validation.json"

# --- Configuration ---
PRETRAINED_MODEL_NAME = 'prajjwal1/bert-tiny' # Smaller transformer
IMAGE_FEATURE_DIM = 256  # Placeholder: Actual dimension from MicroPizzaNet feature extractor
MAX_LEN = 64 # Max length for tokenizer (shorter for smaller inputs)
BATCH_SIZE = 8
EPOCHS = 20 # Increased epochs for potentially complex task
LEARNING_RATE = 3e-5 # Adjusted LR
FOOD_SAFETY_PENALTY_WEIGHT = 2.0 # Increased penalty weight
QAT_ENABLED = True # Enable Quantization-Aware Training

# Placeholder for MicroPizzaNet feature extractor
# In a real scenario, this would use the actual MicroPizzaNet model's feature extraction part.
class MicroPizzaNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # This is a dummy feature extractor. Replace with actual MicroPizzaNet layers.
        # Example: Use a few conv layers and pooling to simulate feature extraction.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Example: 224x224 -> 112x112
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112x112 -> 56x56
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, output_dim) # Project to IMAGE_FEATURE_DIM
        logger.info(f"Initialized DUMMY MicroPizzaNetFeatureExtractor with output_dim={output_dim}")

    def forward(self, x_image):
        x = self.pool1(self.relu1(self.conv1(x_image)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

# 1. Dataset and DataLoader
class PizzaVerifierDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len, image_transforms, feature_extractor_model, device):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_transforms = image_transforms
        self.feature_extractor = feature_extractor_model.to(device)
        self.feature_extractor.eval() # Feature extractor should be in eval mode
        self.device = device

        with open(json_file, 'r') as f:
            raw_data_wrapper = json.load(f)

        # Attempt to find the actual list of records
        # Common keys for the list of records could be 'records', 'data', 'samples', etc.
        # Or, it might be the only other key apart from 'split_info'
        data_iterable = None
        if isinstance(raw_data_wrapper, dict):
            possible_keys = [k for k in raw_data_wrapper.keys() if k != 'split_info']
            if len(possible_keys) == 1 and isinstance(raw_data_wrapper[possible_keys[0]], list):
                data_iterable = raw_data_wrapper[possible_keys[0]]
            elif 'records' in raw_data_wrapper and isinstance(raw_data_wrapper['records'], list):
                data_iterable = raw_data_wrapper['records']
            elif 'data' in raw_data_wrapper and isinstance(raw_data_wrapper['data'], list):
                data_iterable = raw_data_wrapper['data']
            # Add more potential keys if necessary
        
        if data_iterable is None:
            # Fallback: if the structure is just a list directly (e.g. if split_info was removed)
            if isinstance(raw_data_wrapper, list):
                data_iterable = raw_data_wrapper
            else:
                raise ValueError(f"Could not find the list of records in {json_file}. " \
                                 f"Found keys: {list(raw_data_wrapper.keys()) if isinstance(raw_data_wrapper, dict) else 'Not a dict'}")

        for item in data_iterable:
            # Ensure item is a dictionary before proceeding
            if not isinstance(item, dict):
                print(f"Skipping non-dictionary item: {item}")
                continue
            
            # Check if required keys exist
            if 'model_prediction' not in item or 'confidence_score' not in item or \
               'pizza_image_path' not in item or 'quality_score' not in item or \
               'food_safety_critical' not in item:
                print(f"Skipping item with missing keys: {item}")
                continue

            text_input = f"Prediction: {item['model_prediction']}. Confidence: {item['confidence_score']:.3f}"
            
            self.data.append({
                "text_input": text_input,
                "image_path": BASE_DIR / item["pizza_image_path"], # Assuming path is relative to project root
                "quality_score": float(item["quality_score"]),
                "food_safety_critical": bool(item["food_safety_critical"]),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer.encode_plus(
            item["text_input"],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Image features
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            processed_image = self.image_transforms(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.feature_extractor(processed_image).squeeze(0).cpu() # (IMAGE_FEATURE_DIM)
        except FileNotFoundError:
            logger.warning(f"Image not found: {item['image_path']}. Using zero features.")
            image_features = torch.zeros(IMAGE_FEATURE_DIM)
        except Exception as e:
            logger.error(f"Error processing image {item['image_path']}: {e}. Using zero features.")
            image_features = torch.zeros(IMAGE_FEATURE_DIM)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image_features': image_features,
            'quality_score': torch.tensor(item['quality_score'], dtype=torch.float),
            'food_safety_critical': torch.tensor(item['food_safety_critical'], dtype=torch.float)
        }

# 2. Model Definition
class PizzaVerifierMultimodal(nn.Module):
    def __init__(self, pretrained_text_model_name, image_feature_dim, num_labels=1):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(pretrained_text_model_name)
        text_original_hidden_size = self.text_model.config.hidden_size
        
        # Project text features to a smaller dimension if needed, or use directly
        # For bert-tiny (hidden_size=128), direct use is fine.
        self.text_feature_dim = text_original_hidden_size 
        
        combined_input_dim = self.text_feature_dim + image_feature_dim
        
        # Simple MLP combiner
        self.combiner = nn.Sequential(
            nn.Linear(combined_input_dim, combined_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2), # Increased dropout
            nn.Linear(combined_input_dim // 2, num_labels),
            nn.Sigmoid() # Output quality score between 0 and 1
        )
        logger.info(f"Initialized PizzaVerifierMultimodal with text_feature_dim={self.text_feature_dim}, image_feature_dim={image_feature_dim}")

    def forward(self, input_ids, attention_mask, image_features):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token's hidden state
        text_cls_hidden_state = text_outputs.last_hidden_state[:, 0] 
        
        combined_features = torch.cat((text_cls_hidden_state, image_features), dim=1)
        quality_pred = self.combiner(combined_features)
        return quality_pred

# 3. Loss Function
class MultiObjectiveLoss(nn.Module):
    def __init__(self, food_safety_penalty_weight=1.0):
        super().__init__()
        self.food_safety_penalty_weight = food_safety_penalty_weight
        logger.info(f"Initialized MultiObjectiveLoss with penalty_weight={food_safety_penalty_weight}")

    def forward(self, predictions, targets_quality, targets_food_safety_critical):
        # Ensure predictions are squeezed if they are (batch, 1)
        predictions = predictions.squeeze()
        
        # Base MSE loss for quality score
        individual_mse = nn.functional.mse_loss(predictions, targets_quality, reduction='none')
        
        # Apply penalty weights
        # Weight is 1 for normal, 1 + penalty_weight for food safety critical items
        weights = 1.0 + (targets_food_safety_critical * self.food_safety_penalty_weight)
        
        weighted_mse_loss = (individual_mse * weights).mean()
        return weighted_mse_loss

# 4. Training Function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_features = batch['image_features'].to(device)
        targets_quality = batch['quality_score'].to(device)
        targets_fsc = batch['food_safety_critical'].to(device)
        
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask, image_features)
        loss = loss_fn(predictions, targets_quality, targets_fsc)
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        if batch_idx % 10 == 0: # Log progress
             logger.debug(f"Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")
            
    return total_loss / len(data_loader)

# 5. Evaluation Function
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions_all = []
    targets_all = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_features = batch['image_features'].to(device)
            targets_quality = batch['quality_score'].to(device)
            targets_fsc = batch['food_safety_critical'].to(device)

            predictions = model(input_ids, attention_mask, image_features)
            loss = loss_fn(predictions, targets_quality, targets_fsc)
            total_loss += loss.item()
            
            predictions_all.extend(predictions.squeeze().cpu().tolist())
            targets_all.extend(targets_quality.cpu().tolist())
            
    avg_loss = total_loss / len(data_loader)
    correlation = 0.0
    if len(predictions_all) > 1 and len(targets_all) > 1 and not (np.std(predictions_all) == 0 or np.std(targets_all) == 0) :
        correlation = np.corrcoef(predictions_all, targets_all)[0, 1]
    
    logger.info(f"Validation Loss: {avg_loss:.4f}, Correlation: {correlation:.4f}")
    return avg_loss, correlation

# Main training script
def main():
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    
    # Image transforms - adjust as per MicroPizzaNet's expected input
    image_transforms = T.Compose([
        T.Resize((224, 224)), # Example size, adjust to MicroPizzaNet
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])

    # Initialize the (dummy) feature extractor
    # In a real scenario, load pre-trained MicroPizzaNet and adapt it.
    micropizzanet_feature_extractor = MicroPizzaNetFeatureExtractor(output_dim=IMAGE_FEATURE_DIM)

    train_dataset = PizzaVerifierDataset(TRAIN_FILE, tokenizer, MAX_LEN, image_transforms, micropizzanet_feature_extractor, device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    valid_dataset = PizzaVerifierDataset(VALID_FILE, tokenizer, MAX_LEN, image_transforms, micropizzanet_feature_extractor, device)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    model = PizzaVerifierMultimodal(PRETRAINED_MODEL_NAME, IMAGE_FEATURE_DIM).to(device)
    
    if QAT_ENABLED:
        model.train() 
        # For 'fbgemm', float16 is not supported. Ensure model is float32.
        # model.float() # If it was on another dtype
        # For ARM targets, 'qnnpack' is the backend. 'fbgemm' is common for server-side.
        # Using 'fbgemm' for QAT prep is fine, actual backend choice matters at conversion.
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') 
        logger.info(f"Preparing model for QAT with qconfig: {model.qconfig}")
        
        if hasattr(model, 'text_model') and model.text_model is not None:
            for module_name, module in model.text_model.named_modules():
                if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                    module.qconfig = torch.quantization.float_qparams_weight_only_qconfig
                    logger.info(f"Applied float_qparams_weight_only_qconfig to {type(module).__name__}: {module_name}")
        
        # It's important to fuse modules BEFORE preparing for QAT if applicable,
        # though for BERT-tiny, explicit fusion might not be standard or necessary
        # unless specific patterns are known to benefit.
        # Example: torch.quantization.fuse_modules(model, [['conv', 'relu']], inplace=True)

        torch.quantization.prepare_qat(model.train(), inplace=True)
        logger.info("Model prepared for QAT.")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MultiObjectiveLoss(food_safety_penalty_weight=FOOD_SAFETY_PENALTY_WEIGHT)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    logger.info("Starting training...")
    best_val_corr = -1.0

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        logger.info(f"Training Loss: {train_loss:.4f}")
        
        val_loss, val_corr = eval_model(model, valid_loader, loss_fn, device)
        if val_corr > best_val_corr :
            best_val_corr = val_corr
            logger.info(f"New best validation correlation: {best_val_corr:.4f}")
            # Save the best model (non-quantized version during QAT, or standard if QAT disabled)
            if QAT_ENABLED:
                 # During QAT, we save the QAT-prepared model. Final conversion is after all epochs.
                 pass # Or save intermediate QAT model state_dict if needed
            else:
                model_save_path = MODEL_OUTPUT_DIR / "pizza_verifier_model_best.pth"
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Saved best standard model to {model_save_path}")

    logger.info("Training finished.")

    final_model_to_save = model
    if QAT_ENABLED:
        logger.info("Converting QAT model to quantized version...")
        model.eval() 
        model_cpu = model.to("cpu") # Quantization conversion typically done on CPU
        # Ensure all submodules that have qconfig are in eval mode for conversion
        for module in model_cpu.modules():
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                module.eval()

        model_quantized_and_converted = torch.quantization.convert(model_cpu, inplace=False)
        logger.info("Model converted to quantized version.")
        final_model_to_save = model_quantized_and_converted
        model_save_path = MODEL_OUTPUT_DIR / "pizza_verifier_model_quantized.pth"
    else:
        model_save_path = MODEL_OUTPUT_DIR / "pizza_verifier_model_final.pth"

    torch.save(final_model_to_save.state_dict(), model_save_path)
    logger.info(f"Final model saved to {model_save_path}")

    logger.info("Script completed. Key considerations for next steps:")
    logger.info("1. MicroPizzaNet Integration: Replace the DUMMY MicroPizzaNetFeatureExtractor with the actual feature extraction logic from your MicroPizzaNet model. This includes using correct image preprocessing and ensuring IMAGE_FEATURE_DIM matches.")
    logger.info("2. RP2040 Export: The saved .pth file (especially the quantized one) is the input for the model export pipeline (e.g., to TFLite Micro -> C array). Ensure the quantization scheme ('fbgemm' or 'qnnpack' for qconfig) and operations are compatible with CMSIS-NN and the RP2040 toolchain.")
    logger.info("3. Hyperparameter Tuning: Adjust EPOCHS, LEARNING_RATE, BATCH_SIZE, penalty weights, and model architecture (e.g. layer sizes in combiner) for optimal performance.")
    logger.info("4. Evaluation Metrics: Expand evaluation with more pizza-specific metrics as required for Aufgabe 2.3.")

if __name__ == '__main__':
    # Attempt to set multiprocessing start method to 'spawn' for CUDA compatibility
    # This can help avoid certain CUDA + multiprocessing issues on some systems.
    if torch.cuda.is_available():
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        # Only set if not already 'spawn' and if it's possible to set.
        # force=True will override if it was already set by user or another library.
        if current_start_method != 'spawn':
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
                print("Successfully set multiprocessing start method to 'spawn'.")
            except RuntimeError as e:
                # This can happen if the context has already been started with a different method.
                print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Current method: {current_start_method}")
    
    main()
