import os
import json
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from datetime import datetime

# Configuration Parameters
INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
VOCAB_CSV_PATH = 'vocab.csv'  # Path to vocab.csv if needed for future extensions
MODEL_NAME = "upskyy/e5-large-korean"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATES = [1e-3, 5e-4, 1e-4]  # Three learning rates for hyperparameter search
CONTEXT_WINDOWS = [1, 2, 3]  # Context window sizes for hyperparameter search
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
RANDOM_SEED = 2024
LOG_DIR = 'logs'
MODEL_DIR = 'models'

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set up logging
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

logging.info("Starting hyperparameter search...")

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Initialize Tokenizer and Embedding Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
embedding_model.eval()  # Set to evaluation mode
for param in embedding_model.parameters():
    param.requires_grad = False  # Freeze embedding model

# Define the Dataset Class
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

# Define the Classification Model
class ClassificationModel(nn.Module):
    def __init__(self, embedding_model, hidden_size=512):
        """
        Initializes the classification model with a pre-trained embedding model and a classification head.
        """
        super(ClassificationModel, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_model.config.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)  # Binary classification
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass that computes embeddings and outputs logits.
        """
        with torch.no_grad():
            outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean Pooling
            embeddings = self.mean_pooling(outputs, attention_mask)
        
        logits = self.classifier(embeddings)
        return logits.squeeze()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Performs mean pooling on the token embeddings.
        """
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_length, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

# Define Training Function
def train(model, dataloader, optimizer, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        texts, labels = batch
        labels = labels.float().to(device)

        # Tokenize the input texts
        encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Define Evaluation Function
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
            encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
            input_ids = encoded_inputs['input_ids']
            attention_mask = encoded_inputs['attention_mask']

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Predictions
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    return avg_loss, accuracy, tn, fp, fn, tp

# Main Training Pipeline with Hyperparameter Search
def main():
    # Initialize the full Dataset (will be re-initialized for different context windows)
    full_dataset = MessengerDataset(INPUT_DIRECTORY, context_window=CONTEXT_WINDOWS[1])  # Default context_window
    total_size = len(full_dataset)
    train_size = int(total_size * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(total_size * VALIDATION_SPLIT)
    test_size = total_size - train_size - val_size

    # Iterate over all combinations of hyperparameters
    for context_window in CONTEXT_WINDOWS:
        for learning_rate in LEARNING_RATES:
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

            # Initialize DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            logging.info(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples.")

            # Initialize the Classification Model
            model = ClassificationModel(embedding_model).to(device)

            # Define Optimizer and Loss Function
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            # Variables to track the best model
            best_val_accuracy = 0
            best_epoch = 0

            # Training Loop
            for epoch in range(1, EPOCHS + 1):
                logging.info(f"\nEpoch {epoch}/{EPOCHS} for Context Window={context_window}, Learning Rate={learning_rate}")

                train_loss = train(model, train_loader, optimizer, criterion)
                val_loss, val_accuracy, tn, fp, fn, tp = evaluate(model, val_loader, criterion)

                logging.info(f"Training Loss: {train_loss:.4f}")
                logging.info(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy*100:.2f}%")
                logging.info(f"Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

                # Save the model if it has the best validation accuracy so far
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch
                    model_save_path = os.path.join(MODEL_DIR, f'best_model_cw{context_window}_lr{learning_rate}.pth')
                    torch.save(model.state_dict(), model_save_path)
                    logging.info(f"New best model saved at epoch {epoch} with validation accuracy {val_accuracy*100:.2f}%.")

            logging.info(f"Finished training for Context Window={context_window}, Learning Rate={learning_rate}.")
            logging.info(f"Best Validation Accuracy: {best_val_accuracy*100:.2f}% at epoch {best_epoch}.")

            # Load the best model for testing
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'best_model_cw{context_window}_lr{learning_rate}.pth')))
            test_loss, test_accuracy, tn, fp, fn, tp = evaluate(model, test_loader, criterion)
            logging.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}%")
            logging.info(f"Test Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

            # Optionally, you can log or store the test results for further analysis

    logging.info("\nHyperparameter search complete.")

if __name__ == "__main__":
    main()
