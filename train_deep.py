import os
import json
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

# Configuration Parameters
INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
CONTEXT_WINDOW = 2
MODEL_NAME = "klue/roberta-base"  # Changed to a RoBERTa model
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5  # Adjusted learning rate for fine-tuning
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize Tokenizer and Model for Sequence Classification
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)


# Define the Dataset Class
class MessengerDataset(Dataset):
    def __init__(self, input_dir, context_window=2):
        """
        Initializes the dataset by loading all JSON files and preparing samples with context.
        Each sample consists of concatenated context sentences and the label of the target sentence.
        """
        self.samples = []
        json_files = [file for file in os.listdir(input_dir) if file.endswith('.json')]
        
        for file in json_files:
            file_path = os.path.join(input_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file}: {e}")
                    continue  # Skip files with JSON errors

            if not isinstance(data, list):
                print(f"Unexpected JSON structure in file {file}. Expected a list.")
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
        logits = outputs.logits.squeeze()
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
def evaluate(model, dataloader, criterion):
    """
    Evaluates the model on the validation set.
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
            logits = outputs.logits.squeeze()
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Predictions
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy
def main():
    # Initialize the Dataset
    dataset = MessengerDataset(INPUT_DIRECTORY, context_window=CONTEXT_WINDOW)
    total_size = len(dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Define Optimizer, Scheduler, and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_val_accuracy = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train(model, train_loader, optimizer, criterion, scheduler)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_classification_model.pth')
            print("Best model saved.")

    print("\nTraining complete. Best Validation Accuracy: {:.2f}%".format(best_val_accuracy * 100))

if __name__ == "__main__":
    main()