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

# -------------------- Changed splits to 85:5:10 -------------------- #
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.05
TEST_SPLIT = 0.10

RANDOM_SEED = 42
LOG_DIR = 'eeve_3b'
MODEL_DIR = 'models_eeve_3b'
DATA_USAGE_RATIO = 1.0  # Use 100% of the dataset by default

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


# ------------------------- Initialize Tokenizer ------------------------- #
print("Model name: ", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Return the tokenized input as a dictionary and the label
        return {key: val.squeeze(0) for key, val in encoded_input.items()}, label


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

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**encoded_inputs)
                logits = outputs.logits.squeeze(-1)  # Remove unnecessary dimension
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Convert logits to predictions
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long().cpu().numpy()  # threshold
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0

    # Compute confusion matrix components
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    return avg_loss, accuracy, tn, fp, fn, tp


# ------------------------- Modified Train Function with Early Stopping ------------------------- #
def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler=None, 
    epochs=5, 
    eval_every=100,
    patience=3
):
    """
    Train the model for 'epochs' epochs but evaluate on 'val_loader' every
    'eval_every' steps. If validation loss does not improve for 'patience'
    consecutive evaluations, stop early.
    """
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    # For logging
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
        for step, (encoded_inputs, labels) in enumerate(progress_bar):
            # Move inputs and labels to device
            for key in encoded_inputs:
                encoded_inputs[key] = encoded_inputs[key].to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**encoded_inputs)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Evaluate every 'eval_every' steps
            if global_step % eval_every == 0:
                val_loss, val_accuracy, tn, fp, fn, tp = evaluate(model, val_loader, criterion)
                logging.info(
                    f"Step {global_step}: Val Loss={val_loss:.4f}, "
                    f"Val Acc={val_accuracy*100:.2f}%, "
                    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
                )

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(
                            f"Early stopping triggered at epoch {epoch}, step {global_step}. "
                            f"Val loss has not improved for {patience_counter} evaluations."
                        )
                        return  # Stop training completely

            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"avg_train_loss": avg_loss})

        # End of epoch logging
        logging.info(f"Epoch {epoch} completed. Average Training Loss: {avg_loss:.4f}")

    # If we finish all epochs without triggering early stopping:
    logging.info("Training completed without early stopping.")
    return


# ------------------------- Main Training Pipeline with Hyperparameter Search ------------------------- #
def main():
    # Hyperparameter search space
    learning_rates = LEARNING_RATES
    context_windows = CONTEXT_WINDOWS

    overall_best_val_accuracy = 0
    overall_best_model_weights = None
    overall_best_hyperparams = None

    for context_window in context_windows:
        for learning_rate in learning_rates:
            logging.info(
                f"\nStarting hyperparameter combination: "
                f"Context Window={context_window}, Learning Rate={learning_rate}, Dataset Ratio={DATA_USAGE_RATIO}"
            )
            if DATA_USAGE_RATIO < 1:
                logging.info(f"Using {DATA_USAGE_RATIO * 100}% of the dataset for training (for testing).")

            # Create the full dataset
            dataset = MessengerDataset(INPUT_DIRECTORY, context_window=context_window)
            total_size = len(dataset)

            # Optionally sample a subset
            sample_size = int(total_size * DATA_USAGE_RATIO)
            sample_size = max(1, sample_size)
            sampled_indices = random.sample(range(total_size), sample_size)
            dataset = torch.utils.data.Subset(dataset, sampled_indices)
            total_size = len(dataset)

            # -------------------- 85:5:10 Split -------------------- #
            train_size = int(total_size * TRAIN_SPLIT)
            val_size = int(total_size * VAL_SPLIT)
            test_size = total_size - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(RANDOM_SEED)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
            model.config.pad_token_id = tokenizer.pad_token_id

            # Enable gradient checkpointing if desired (large models)
            model.gradient_checkpointing_enable()

            # Define optimizer, scheduler, loss
            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )

            # -------------------- Train with Early Stopping inside train_model -------------------- #
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=EPOCHS,
                eval_every=100,     # Evaluate every 100 steps
                patience=3          # Early stop if no improvement after 3 evaluations
            )

            # -------------------- Evaluate on validation one final time after training -------------------- #
            val_loss, val_accuracy, tn, fp, fn, tp = evaluate(model, val_loader, criterion)
            logging.info(
                f"Final Validation: Loss={val_loss:.4f}, "
                f"Accuracy={val_accuracy*100:.2f}%, "
                f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
            )

            # Save model for current hyperparameter combination
            model_save_path = os.path.join(MODEL_DIR, f'model_cw{context_window}_lr{learning_rate}.pth')
            torch.save(model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)
            logging.info(f"Model saved at {model_save_path}")

            # Check if this is overall best
            if val_accuracy > overall_best_val_accuracy:
                overall_best_val_accuracy = val_accuracy
                overall_best_model_weights = model.state_dict()
                overall_best_hyperparams = (context_window, learning_rate)
                logging.info(f"New overall best model with Validation Accuracy {val_accuracy*100:.2f}%")

    # -------------------- After Hyperparameter Search: Test the Overall Best -------------------- #
    if overall_best_model_weights:
        cw, lr = overall_best_hyperparams
        best_model_save_path = os.path.join(MODEL_DIR, f'best_model_cw{cw}_lr{lr}.pth')
        torch.save(overall_best_model_weights, best_model_save_path, _use_new_zipfile_serialization=False)
        logging.info(f"\nOverall best model saved: Context Window={cw}, Learning Rate={lr}. "
                     f"Val Accuracy={overall_best_val_accuracy*100:.2f}%")

        # Rebuild the same dataset splits to test the best model
        dataset = MessengerDataset(INPUT_DIRECTORY, context_window=cw)
        total_size = len(dataset)
        sample_size = int(total_size * DATA_USAGE_RATIO)
        sample_size = max(1, sample_size)
        sampled_indices = random.sample(range(total_size), sample_size)
        dataset = torch.utils.data.Subset(dataset, sampled_indices)
        total_size = len(dataset)

        train_size = int(total_size * TRAIN_SPLIT)
        val_size = int(total_size * VAL_SPLIT)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
        best_model.config.pad_token_id = tokenizer.pad_token_id
        best_model.load_state_dict(overall_best_model_weights)

        test_loss, test_accuracy, tn, fp, fn, tp = evaluate(best_model, test_loader, nn.BCEWithLogitsLoss().to(device))
        logging.info(
            f"\nTest Results for the best model [cw={cw}, lr={lr}]: "
            f"Loss={test_loss:.4f}, Accuracy={test_accuracy*100:.2f}%, "
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )

        # Save test results
        test_results_path = os.path.join(MODEL_DIR, f'test_results_cw{cw}_lr{lr}.txt')
        with open(test_results_path, 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")

    logging.info("\nHyperparameter search complete for Sequence Classification.")


if __name__ == "__main__":
    main()
