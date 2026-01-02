import os
import json
import random
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from datetime import datetime

# ------------------------- Dataset Class -------------------------
class MessengerDataset(Dataset):
    def __init__(self, input_dir, context_window=2):
        self.samples = []
        if os.path.isdir(input_dir):
            files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
        else:
            files = [input_dir]

        for file_path in tqdm(files, desc="Loading JSON files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from {file_path}: {e}")
                    continue

            if not isinstance(data, list):
                logging.warning(f"Unexpected JSON structure in {file_path}. Expected a list.")
                continue

            for entry in data:
                utterances = entry.get('utterance', [])
                total_utterances = len(utterances)
                for i in range(total_utterances):
                    start = max(i - context_window, 0)
                    end = min(i + context_window + 1, total_utterances)
                    context_sentences = [utterances[j]['original_form'] for j in range(start, end)]
                    context_text = ' '.join(context_sentences)
                    label = utterances[i].get('label', 0)
                    self.samples.append((context_text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ------------------------- Loss Functions -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------- Training & Eval Functions -------------------------
def train(model, dataloader, optimizer, criterion, tokenizer, device, scheduler=None):
    model.train()
    total_loss = 0
    for texts, labels in tqdm(dataloader, desc="Training", leave=False):
        labels = labels.float().to(device)
        encoded_inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt', max_length=512
        ).to(device)

        outputs = model(**encoded_inputs)
        logits = outputs.logits.squeeze(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = labels.float().to(device)
            encoded_inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(device)

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
        if len(set(all_labels)) == 1:
            if all_labels[0] == 0: tn = len(all_labels)
            else: tp = len(all_labels)
            
    return avg_loss, accuracy, tn, fp, fn, tp

# ------------------------- Main Pipeline -------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for Korean Slang Detection")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing training JSON data")
    parser.add_argument("--model_name", type=str, default="klue/bert-base", help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"], help="Loss function")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[2e-5, 3e-5, 5e-5], help="List of learning rates")
    parser.add_argument("--context_windows", type=int, nargs="+", default=[1, 2, 3], help="List of context window sizes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(args.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Config: {vars(args)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Hyperparameter Search
    for cw in args.context_windows:
        for lr in args.learning_rates:
            logging.info(f"\n--- Experiment: Context Window={cw}, Learning Rate={lr} ---")
            
            dataset = MessengerDataset(args.input_dir, context_window=cw)
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
            )

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            total_steps = len(train_loader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
            
            if args.loss == 'focal':
                criterion = FocalLoss()
            else:
                criterion = nn.BCEWithLogitsLoss()

            best_acc = 0
            for epoch in range(1, args.epochs + 1):
                train_loss = train(model, train_loader, optimizer, criterion, tokenizer, device, scheduler)
                val_loss, val_acc, tn, fp, fn, tp = evaluate(model, val_loader, criterion, tokenizer, device)

                logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    sanitized_model_name = args.model_name.replace("/", "_")
                    save_path = os.path.join(args.output_dir, f'best_model_{sanitized_model_name}_cw{cw}_lr{lr}.pth')
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Saved best model to {save_path}")

if __name__ == "__main__":
    main()
