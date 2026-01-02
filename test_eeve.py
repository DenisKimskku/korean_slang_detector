import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------- Configuration ------------------------- #
MODEL_NAME = "yanolja/EEVE-Korean-2.8B-v1.0"
MODEL_PATH = "/home/minseok/forensic/models_eeve_3b/model_cw1_lr0.0001.pth"
TEST_DATA_PATH = "/home/minseok/forensic/test_gpt.json"  # Update if needed
BATCH_SIZE = 1
CONTEXT_WINDOW = 1  # as indicated by the model file naming
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- Define the Test Dataset ------------------------- #
class MessengerTestDataset(Dataset):
    """
    This dataset loads the JSON test file which contains entries in the following format:
    {
        "id": "...",
        "utterance": [
            {"original_form": "...", "label": 0 or 1},
            ...
        ]
    }
    For each utterance, we form a context window (target utterance plus surrounding utterances) 
    and return the tokenized text along with its label.
    """
    def __init__(self, json_file, context_window=1):
        self.samples = []
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # data is expected to be a list of entries
        for entry in data:
            utterances = entry.get('utterance', [])
            total = len(utterances)
            for i in range(total):
                start = max(i - context_window, 0)
                end = min(i + context_window + 1, total)
                # Concatenate context utterances into one string
                context_text = ' '.join([utterances[j]['original_form'] for j in range(start, end)])
                label = utterances[i].get('label', 0)
                self.samples.append((context_text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        # Tokenize the text; note: padding to max_length for fixed size tensors
        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # Squeeze out batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding, label

# ------------------------- Evaluation Function ------------------------- #
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for encoded_inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move inputs to device
            for key in encoded_inputs:
                encoded_inputs[key] = encoded_inputs[key].to(DEVICE)
            labels = torch.tensor(labels).float().to(DEVICE)
            
            # Use autocast for mixed precision (float16) if on GPU
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**encoded_inputs)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return avg_loss, accuracy, tn, fp, fn, tp

# ------------------------- Main Test Script ------------------------- #
if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
    
    # Load saved model weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # Prepare the test dataset and dataloader
    test_dataset = MessengerTestDataset(TEST_DATA_PATH, context_window=CONTEXT_WINDOW)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Define loss criterion (BCEWithLogitsLoss for binary classification)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    
    # Evaluate on test set
    test_loss, test_accuracy, tn, fp, fn, tp = evaluate(model, test_loader, criterion)
    
    # Print out test performance
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
