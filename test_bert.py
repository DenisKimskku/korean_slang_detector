import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------- Configuration ------------------------- #
MODEL_PATH = "/home/minseok/forensic/models_bert_base/best_model_seq_cw1_lr2e-05.pth"
TEST_DATA_PATH = "/home/minseok/forensic/test_gpt.json"
MODEL_NAME = "klue/bert-base"
BATCH_SIZE = 32
CONTEXT_WINDOW = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- Define the Dataset Class ------------------------- #
class MessengerTestDataset(Dataset):
    def __init__(self, json_file, context_window=1):
        """
        Loads a single JSON file in the expected format and creates samples with context.
        Each sample is a tuple of (context_text, label).
        """
        self.samples = []
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Data is expected to be a list of entries.
        for entry in data:
            utterances = entry.get('utterance', [])
            total_utterances = len(utterances)
            for i in range(total_utterances):
                start = max(i - context_window, 0)
                end = min(i + context_window + 1, total_utterances)
                # Concatenate context utterances into one string
                context_text = ' '.join([utterances[j]['original_form'] for j in range(start, end)])
                label = utterances[i].get('label', 0)
                self.samples.append((context_text, label))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# ------------------------- Evaluation Function ------------------------- #
def evaluate(model, dataloader, criterion, tokenizer):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating"):
            labels = torch.tensor(labels).float().to(DEVICE)
            encoded_inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(DEVICE)
            
            outputs = model(**encoded_inputs)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(all_labels)
    print(all_preds)
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(set(all_labels)) == 1:
            if all_labels[0] == 0:
                tn = len(all_labels)
            else:
                tp = len(all_labels)
    return avg_loss, accuracy, tn, fp, fn, tp

# ------------------------- Main Evaluation Script ------------------------- #
def main():
    token = os.environ.get("HF_TOKEN")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, token=token).to(DEVICE)
    
    # Load the saved model weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # Create test dataset and dataloader
    test_dataset = MessengerTestDataset(TEST_DATA_PATH, context_window=CONTEXT_WINDOW)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Define the loss function (BCEWithLogitsLoss for binary classification)
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate the model on the test set
    test_loss, test_accuracy, tn, fp, fn, tp = evaluate(model, test_loader, criterion, tokenizer)
    
    # Print performance metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    main()
