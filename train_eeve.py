import os
import json
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from datetime import datetime

#debug
import sys

# ------------------------- Configuration Parameters ------------------------- #
INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
VOCAB_CSV_PATH = 'vocab.csv'  # Path to vocab.csv if needed for future extensions #Not used here
MODEL_NAME = "yanolja/EEVE-Korean-2.8B-v1.0"  # Changed to a RoBERTa model
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATES = [2e-5, 3e-5, 5e-5]  # Three learning rates for hyperparameter search
CONTEXT_WINDOWS = [1, 2, 3]  # Context window sizes for hyperparameter search
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
RANDOM_SEED = 42
LOG_DIR = 'eeve_3b'
MODEL_DIR = 'models_eeve_3b'

DATA_USAGE_RATIO = 0.01  # Use 1% of the dataset for testing

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------- Logging Setup ------------------------- #
log_filename = os.path.join(LOG_DIR, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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

logging.info("Starting hyperparameter search for Sequence Classification...")

# ------------------------- Reproducibility Setup ------------------------- #
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ------------------------- Device Configuration ------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float
logging.info(f"Using device: {device} with precision: {precision_dtype}")


# ------------------------- Initialize Tokenizer and Model ------------------------- #
print("Model name: ", MODEL_NAME)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # Note: AutoModelForSequenceClassification includes a classification head
# # num_labels=1 implies regression; for binary classification, num_labels=1 with BCE loss is acceptable
# # Alternatively, set num_labels=2 for explicit binary classification
# model_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
# model_base.config.pad_token_id = tokenizer.pad_token_id

# # Ensure the tokenizer has a padding token
# print("Tokenizer padding token: ", tokenizer.pad_token)

# # Ensure the tokenizer has a padding token
# if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add custom PAD token
#     tokenizer.pad_token = '[PAD]'
#     model_base.resize_token_embeddings(len(tokenizer))  # Resize the model’s embeddings
#     model_base.config.pad_token_id = tokenizer.pad_token_id


# # Set pad_token_id in model config
# model_base.config.pad_token_id = tokenizer.pad_token_id

# # Log the padding token details for verification
# print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
# logging.info(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")


# model_base = model_base.to(device=device, dtype=precision_dtype)


class EarlyStopping:
    def __init__(self, patience=3, delta=0, save_best=True):
        """
        Early stopping to stop training when validation loss is not improving.

        :param patience: Number of epochs to wait for improvement before stopping.
        :param delta: Minimum change to qualify as an improvement.
        :param save_best: Whether to save the best model based on validation performance.
        """
        self.patience = patience
        self.delta = delta
        self.best_val_accuracy = None
        self.best_epoch = 0
        self.counter = 0
        self.save_best = save_best
        self.best_model_weights = None

    def __call__(self, val_accuracy, model, epoch):
        """
        Call this function after each epoch to check for early stopping criteria.

        :param val_accuracy: Validation accuracy in the current epoch.
        :param model: The model being trained.
        :param epoch: The current epoch.
        :return: True if training should stop, otherwise False.
        """
        if self.best_val_accuracy is None:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch
            return False

        if val_accuracy - self.best_val_accuracy > self.delta:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch
            self.counter = 0
            if self.save_best:
                self.best_model_weights = model.state_dict()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False



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
        
        # Tokenize the context text
        encoded_input = tokenizer(
            context_text,
            padding='max_length',  # You can remove this and rely on the collator's padding
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Return the tokenized input as a dictionary and the label
        return {key: val.squeeze(0) for key, val in encoded_input.items()}, label


# ------------------------- Define Training Function ------------------------- #
# def train(model, dataloader, optimizer, criterion, scheduler=None):
#     model.train()
#     total_loss = 0

#     scaler = torch.cuda.amp.GradScaler()  # Enable gradient scaling for AMP
#     for batch in tqdm(dataloader, desc="Training", leave=False):
#         texts, labels = batch
#         labels = labels.float().to(device)

#         # Tokenize the input texts
#         encoded_inputs = tokenizer(
#             texts,
#             padding='max_length',  # Explicit padding to max length
#             truncation=True,
#             return_tensors='pt',
#             max_length=512
#         ).to(device)


#         optimizer.zero_grad()
#         #print pad token of model
#         # print("Model pad token: ", model.config.pad_token_id)
#         # #print pad token of tokenizer
#         # print("Tokenizer pad token: ", tokenizer.pad_token_id)
#         # sys.exit()
#         # AMP forward pass and loss computation
#         with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
#             outputs = model(**encoded_inputs)
#             logits = outputs.logits.squeeze(-1)
#             loss = criterion(logits, labels)

#         scaler.scale(loss).backward()  # Scale loss for backward pass
#         scaler.step(optimizer)  # Optimizer step with scaler
#         scaler.update()  # Update the scaler

#         if scheduler:
#             scheduler.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

ACCUMULATION_STEPS = 4  # Simulate a larger batch size by accumulating gradients over 4 steps

def train(model, dataloader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler for AMP

    optimizer.zero_grad()  # Clear gradients before starting

    for step, (encoded_inputs, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Move inputs and labels to the appropriate device
        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].to(device)
        labels = labels.float().to(device)

        with torch.cuda.amp.autocast(dtype=torch.float16):  # Enable AMP for mixed precision
            outputs = model(**encoded_inputs)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels) / ACCUMULATION_STEPS  # Scale loss for accumulation

        scaler.scale(loss).backward()  # Backward pass with scaled loss

        # Step optimizer only after accumulating gradients for ACCUMULATION_STEPS
        if (step + 1) % ACCUMULATION_STEPS == 0 or step == len(dataloader) - 1:
            scaler.step(optimizer)  # Apply the optimizer step
            scaler.update()  # Update the scaler
            optimizer.zero_grad()  # Clear accumulated gradients

            if scheduler:
                scheduler.step()  # Update learning rate

        total_loss += loss.item() * ACCUMULATION_STEPS

    avg_loss = total_loss / len(dataloader)
    return avg_loss



# ------------------------- Define Evaluation Function ------------------------- #
def evaluate(model, dataloader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking
        for encoded_inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move inputs and labels to the appropriate device
            for key in encoded_inputs:
                encoded_inputs[key] = encoded_inputs[key].to(device)
            labels = labels.float().to(device)

            # Forward pass with mixed precision using AMP
            with torch.cuda.amp.autocast(dtype=torch.float16):  # Ensure mixed precision
                outputs = model(**encoded_inputs)
                logits = outputs.logits.squeeze(-1)  # Remove unnecessary dimensions
                loss = criterion(logits, labels)  # Calculate loss

            total_loss += loss.item()  # Accumulate loss

            # Convert logits to predictions
            preds = torch.sigmoid(logits)  # Apply sigmoid for binary classification
            preds = (preds > 0.5).long().cpu().numpy()  # Threshold the probabilities
            all_preds.extend(preds)  # Collect predictions
            all_labels.extend(labels.cpu().numpy())  # Collect true labels

    avg_loss = total_loss / len(dataloader)  # Compute average loss
    accuracy = accuracy_score(all_labels, all_preds)  # Compute accuracy

    # Compute confusion matrix components
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0  # Handle empty confusion matrix case

    return avg_loss, accuracy, tn, fp, fn, tp


# ------------------------- Main Training Pipeline with Hyperparameter Search ------------------------- #
def main():
    # Hyperparameter search space
    learning_rates = LEARNING_RATES
    context_windows = CONTEXT_WINDOWS

    # Iterate over all combinations of hyperparameters
    for context_window in context_windows:
        for learning_rate in learning_rates:
            logging.info(f"\nStarting hyperparameter combination: Context Window={context_window}, Learning Rate={learning_rate}, Dataset Ratio={DATA_USAGE_RATIO}")
            if DATA_USAGE_RATIO < 1:
                logging.info(f"Using {DATA_USAGE_RATIO * 100}% of the dataset for training. This is intended for testing purposes.")
            
            dataset = MessengerDataset(INPUT_DIRECTORY, context_window=context_window)
            # total_size = len(dataset)
            # Sample a small subset of the dataset based on DATA_USAGE_RATIO
            total_size = len(dataset)
            sample_size = int(total_size * DATA_USAGE_RATIO)

            # Ensure sample size is at least 1 to avoid empty subsets
            if sample_size < 1:
                sample_size = 1

            # Randomly select a subset of indices
            sampled_indices = random.sample(range(total_size), sample_size)

            # Create a subset of the dataset using the sampled indices
            dataset = torch.utils.data.Subset(dataset, sampled_indices)

            # Update total_size after sampling
            total_size = len(dataset)

            # train_size = int(total_size * (1 - VALIDATION_SPLIT - TEST_SPLIT))
            # val_size = int(total_size * VALIDATION_SPLIT)
            # test_size = total_size - train_size - val_size
            train_size = int(total_size * (1 - VALIDATION_SPLIT - TEST_SPLIT))
            val_size = int(total_size * VALIDATION_SPLIT)
            test_size = total_size - train_size - val_size

            # Split the dataset into training, validation, and test sets
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(RANDOM_SEED)
            )

            # Initialize DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

            # Initialize the Model
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
            model.config.pad_token_id = tokenizer.pad_token_id
            model.gradient_checkpointing_enable()

            # Define Loss Function and Optimizer
            criterion = nn.BCEWithLogitsLoss().to(device)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            #use sgd
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)

            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )

            early_stopping = EarlyStopping(patience=3, del