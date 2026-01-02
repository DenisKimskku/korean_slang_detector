#!/usr/bin/env python3
# eval2.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from datetime import datetime
import re
from tqdm import tqdm
import random

# ------------------------- Configuration Parameters ------------------------- #
TEST_DATA_DIR = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/test2/modified5'
MODEL_DIR = '/home/minseok/forensic/models_seq_class'
LOG_DIR = 'evaluation_logs_rb5'
MODEL_NAME = "klue/roberta-base"  # Must match the model used during training
BATCH_SIZE = 32
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------- Logging Setup ------------------------- #
log_filename = os.path.join(LOG_DIR, f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("Starting evaluation of Sequence Classification models...")

# ------------------------- Reproducibility Setup ------------------------- #
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ------------------------- Define the Dataset Class ------------------------- #
class MessengerDataset(Dataset):
    def __init__(self, input_dir, context_window=2):
        """
        Initializes the dataset by loading all JSON files and preparing samples with context.
        Each sample consists of concatenated context sentences and the label of the target sentence.
        """
        self.samples = []
        json_files = [file for file in os.listdir(input_dir) if file.endswith('.json')]
        
        for file in tqdm(json_files, desc="Loading JSON files"):
            file_path = os.path.join(input_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from file {file}: {e}")
                    continue  # Skip files with JSON errors

            if not isinstance(data, list):
                logging.warning(f"Unexpected JSON structure in file {file}. Expected a list.")
                continue

            for entry in data:
                utterances = entry.get('utterance', [])
                total_utterances = len(utterances)
                for i in range(total_utterances):
                    # Determine the window boundaries
                    start = max(i - context_window, 0)
                    end = min(i + context_window + 1, total_utterances)
                    
                    # Extract context sentences
                    context_sentences = [utterances[j]['original_form'] for j in range(start, end)]
                    context_text = ' '.join(context_sentences)
                    
                    # Get the label of the target sentence
                    label = utterances[i].get('label', 0)
                    
                    self.samples.append((context_text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_text, label = self.samples[idx]
        return context_text, label

# ------------------------- Define Evaluation Function ------------------------- #
def evaluate(model, dataloader, criterion, tokenizer):
    """
    Evaluates the model on the provided dataloader.
    Returns average loss, accuracy, and confusion matrix components.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            texts, labels = batch
            labels = labels.float().to(DEVICE)

            # Tokenize the input texts
            encoded_inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(DEVICE)

            # Forward pass
            outputs = model(**encoded_inputs)
            logits = outputs.logits.squeeze(-1)  # For binary classification
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Predictions
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Compute confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        # Handle cases where one of the classes is missing
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(set(all_labels)) == 1:
            if all_labels[0] == 0:
                tn = len(all_labels)
            else:
                tp = len(all_labels)
        elif len(set(all_preds)) == 1:
            if all_preds[0] == 0:
                tn = sum(1 for label in all_labels if label == 0)
                fp = sum(1 for label in all_labels if label == 1)
            else:
                tp = sum(1 for label in all_labels if label == 1)
                fn = sum(1 for label in all_labels if label == 0)
    
    return avg_loss, accuracy, tn, fp, fn, tp

# ------------------------- Main Evaluation Pipeline ------------------------- #
def main():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Initialize the Dataset with the same context windows used during training
    # Assuming that models were trained with context windows 1, 2, 3
    context_windows = [1, 2]
    learning_rates = [2e-5, 3e-5, 5e-5]

    # Iterate over all combinations of hyperparameters to evaluate corresponding models
    for context_window in context_windows:
        for learning_rate in learning_rates:
            model_filename = f'best_model_seq_cw{context_window}_lr{learning_rate}.pth'
            model_path = os.path.join(MODEL_DIR, model_filename)

            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_filename} not found in {MODEL_DIR}. Skipping.")
                continue

            logging.info(f"\nEvaluating Model: {model_filename}")
            logging.info(f"Hyperparameters - Context Window: {context_window}, Learning Rate: {learning_rate}")

            # Initialize the Dataset with current context_window
            dataset = MessengerDataset(TEST_DATA_DIR, context_window=context_window)
            total_size = len(dataset)
            if total_size == 0:
                logging.warning(f"No data found in {TEST_DATA_DIR} for context window {context_window}. Skipping.")
                continue

            # Split the dataset into validation and test sets
            train_size = int(total_size * (1 - VALIDATION_SPLIT - TEST_SPLIT))
            val_size = int(total_size * VALIDATION_SPLIT)
            test_size = total_size - val_size - train_size

            _, _, test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(RANDOM_SEED)
            )

            logging.info(f"Test Dataset split: {val_size} validation, {test_size} test samples.")

            # Initialize DataLoader for test set
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

            logging.info(f"Test Dataset: {len(test_dataset)} samples.")

            # Initialize the Model
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)

            # Evaluate the model
            test_loss, test_accuracy, tn, fp, fn, tp = evaluate(model, test_loader, criterion, tokenizer)

            logging.info(f"Test Loss: {test_loss:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
            logging.info(f"Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    logging.info("\nEvaluation of all models completed.")

if __name__ == "__main__":
    main()
