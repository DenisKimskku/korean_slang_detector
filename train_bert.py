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

# ------------------------- Configuration Parameters ------------------------- #
INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
VOCAB_CSV_PATH = 'vocab.csv'  # Path to vocab.csv if needed for future extensions
MODEL_NAME = "klue/bert-base"  # Changed to a RoBERTa model
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATES = [2e-5, 3e-5, 5e-5]  # Three learning rates for hyperparameter search
CONTEXT_WINDOWS = [1, 2, 3]  # Context window sizes for hyperparameter search
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
RANDOM_SEED = 42
LOG_DIR = 'bert_base'
MODEL_DIR = 'models_bert_base'

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
logging.info(f"Using device: {device}")

# ------------------------- Initialize Tokenizer and Model ------------------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Note: AutoModelForSequenceClassification includes a classification head
# num_labels=1 implies regression; for binary classification, num_labels=1 with BCE loss is acceptable
# Alternatively, set num_labels=2 for explicit binary classification
model_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

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

# ------------------------- Define Training Function ------------------------- #
def train(model, dataloader, optimizer, criterion, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts, labels = batch
        labels = labels.float().to(device)

        # Tokenize the input texts
        encoded_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512  # Ensure sequences are within model's max length
        ).to(device)

        # Forward pass
        outputs = model(**encoded_inputs)
        logits = outputs.logits.squeeze(-1)  # Changed from squeeze() to squeeze(-1)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ------------------------- Define Evaluation Function ------------------------- #
def evaluate(model, dataloader, criterion):
    """
    Evaluates the model on the validation or test set.
    Returns average loss, accuracy, and confusion matrix components.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            texts, labels = batch
            labels = labels.float().to(device)

            # Tokenize the input texts
            encoded_inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(device)

            # Forward pass
            outputs = model(**encoded_inputs)
            logits = outputs.logits.squeeze(-1)  # Changed from squeeze() to squeeze(-1)
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

# ------------------------- Main Training Pipeline with Hyperparameter Search ------------------------- #
def main():
    # Define hyperparameter search space
    learning_rates = LEARNING_RATES
    context_windows = CONTEXT_WINDOWS

    # Iterate over all combinations of hyperparameters
    for context_window in context_windows:
        for learning_rate in learning_rates:
            logging.info(f"\nStarting hyperparameter combination: Context Window={context_window}, Learning Rate={learning_rate}")

            # Initialize the Dataset with current context_window
            dataset = MessengerDataset(INPUT_DIRECTORY, context_window=context_window)
            total_size = len(dataset)
            train_size = int(total_size * (1 - VALIDATION_SPLIT - TEST_SPLIT))
            val_size = int(total_size * VALIDATION_SPLIT)
            test_size = total_size - train_size - val_size

            # Split the dataset into training, validation, and test sets
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(RANDOM_SEED)
            )

            logging.info(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples.")

            # Initialize DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            # Initialize the Model
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

            # Define Optimizer, Scheduler, and Loss Function
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )
            criterion = nn.BCEWithLogitsLoss()

            # Variables to track the best model
            best_val_accuracy = 0
            best_epoch = 0

            # Training Loop
            for epoch in range(1, EPOCHS + 1):
                logging.info(f"\nEpoch {epoch}/{EPOCHS} for Context Window={context_window}, Learning Rate={learning_rate}")

                train_loss = train(model, train_loader, optimizer, criterion, scheduler)
                val_loss, val_accuracy, tn, fp, fn, tp = evaluate(model, val_loader, criterion)

                logging.info(f"Training Loss: {train_loss:.4f}")
                logging.info(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy*100:.2f}%")
                logging.info(f"Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

                # Save the model if it has the best validation accuracy so far
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch
                    model_save_path = os.path.join(MODEL_DIR, f'best_model_seq_cw{context_window}_lr{learning_rate}.pth')
                    torch.save(model.state_dict(), model_save_path)
                    logging.info(f"New best model saved at epoch {epoch} with validation accuracy {val_accuracy*100:.2f}%.")

            logging.info(f"Finished training for Context Window={context_window}, Learning Rate={learning_rate}.")
            logging.info(f"Best Validation Accuracy: {best_val_accuracy*100:.2f}% at epoch {best_epoch}.")

            # Load the best model for testing
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'best_model_seq_cw{context_window}_lr{learning_rate}.pth')))
            test_loss, test_accuracy, tn, fp, fn, tp = evaluate(model, test_loader, criterion)
            logging.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}%")
            logging.info(f"Test Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

            # Optionally, you can log or store the test results for further analysis

    logging.info("\nHyperparameter search complete for Sequence Classification.")

if __name__ == "__main__":
    main()
