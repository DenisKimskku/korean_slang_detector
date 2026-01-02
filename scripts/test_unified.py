import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix

class MessengerTestDataset(Dataset):
    def __init__(self, json_file, context_window=1):
        self.samples = []
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            data = [data] # Handle single object JSON

        for entry in data:
            utterances = entry.get('utterance', [])
            total_utterances = len(utterances)
            for i in range(total_utterances):
                start = max(i - context_window, 0)
                end = min(i + context_window + 1, total_utterances)
                context_text = ' '.join([utterances[j]['original_form'] for j in range(start, end)])
                label = utterances[i].get('label', 0)
                self.samples.append((context_text, label))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def evaluate(model, dataloader, criterion, tokenizer, device, use_fp16=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating"):
            labels = torch.tensor(labels).float().to(device)
            encoded_inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(device)
            
            if use_fp16 and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(**encoded_inputs)
                    logits = outputs.logits.squeeze(-1)
                    loss = criterion(logits, labels)
            else:
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
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return avg_loss, accuracy, tn, fp, fn, tp

def main():
    parser = argparse.ArgumentParser(description="Unified Test Script")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved .pth model file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--context_window", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
    
    print(f"Loading model weights from {args.model_path}...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_dataset = MessengerTestDataset(args.test_data, context_window=args.context_window)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    
    test_loss, test_accuracy, tn, fp, fn, tp = evaluate(model, test_loader, criterion, tokenizer, device, args.fp16)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    main()
