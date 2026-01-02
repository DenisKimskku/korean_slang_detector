import os
import json
import random
import numpy as np
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Configuration ------------------------- #
class Config:
    # Data paths
    INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
    
    # Model parameters
    MODEL_NAME = "klue/roberta-base"  # RoBERTa for better context understanding
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2
    EPOCHS = 20
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Data splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Sliding window parameters
    WINDOW_SIZE = 10  # Number of messages in context
    STRIDE = 5  # Window stride
    MAX_LENGTH = 512
    
    # Architecture parameters
    USE_ATTENTION_POOLING = True
    HIDDEN_DROPOUT = 0.3
    ATTENTION_DROPOUT = 0.1
    
    # Training parameters
    EARLY_STOPPING_PATIENCE = 5
    GRADIENT_CLIP_VAL = 1.0
    LABEL_SMOOTHING = 0.1
    
    # Loss parameters
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Augmentation
    USE_AUGMENTATION = True
    AUGMENT_PROB = 0.3
    
    # Output paths
    LOG_DIR = 'logs_pure_lm_attn'
    MODEL_DIR = 'models_pure_lm_attn'
    CHECKPOINT_DIR = 'checkpoints_pure_lm_attn'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 42

# Create directories
for dir_path in [Config.LOG_DIR, Config.MODEL_DIR, Config.CHECKPOINT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ------------------------- Logging Setup ------------------------- #
def setup_logging():
    log_filename = os.path.join(
        Config.LOG_DIR, 
        f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ------------------------- Set Random Seeds ------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(Config.RANDOM_SEED)

# ------------------------- Data Augmentation ------------------------- #
class ConversationAugmenter:
    """Simple augmentation techniques for conversation data"""
    
    def __init__(self, augment_prob=0.3):
        self.augment_prob = augment_prob
        self.augmentation_methods = [
            self.shuffle_within_window,
            self.drop_messages,
            self.duplicate_messages,
            self.mask_tokens
        ]
    
    def augment(self, messages: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Apply random augmentation to messages"""
        if random.random() > self.augment_prob:
            return messages, labels
        
        method = random.choice(self.augmentation_methods)
        return method(messages.copy(), labels.copy())
    
    def shuffle_within_window(self, messages: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Shuffle messages within small windows to maintain some coherence"""
        window_size = 3
        for i in range(0, len(messages) - window_size + 1, window_size):
            window_end = min(i + window_size, len(messages))
            indices = list(range(i, window_end))
            random.shuffle(indices)
            
            # Reorder messages and labels
            window_messages = [messages[j] for j in indices]
            window_labels = [labels[j] for j in indices]
            
            for j, idx in enumerate(range(i, window_end)):
                messages[idx] = window_messages[j]
                labels[idx] = window_labels[j]
        
        return messages, labels
    
    def drop_messages(self, messages: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Randomly drop some messages"""
        if len(messages) <= 3:
            return messages, labels
        
        drop_rate = 0.2
        keep_indices = [i for i in range(len(messages)) if random.random() > drop_rate]
        
        if len(keep_indices) < 3:  # Ensure minimum messages
            return messages, labels
        
        messages = [messages[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]
        
        return messages, labels
    
    def duplicate_messages(self, messages: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Duplicate some messages (simulating repetition)"""
        if len(messages) >= 15:  # Don't make too long
            return messages, labels
        
        dup_rate = 0.2
        new_messages = []
        new_labels = []
        
        for msg, label in zip(messages, labels):
            new_messages.append(msg)
            new_labels.append(label)
            if random.random() < dup_rate:
                new_messages.append(msg)
                new_labels.append(label)
        
        return new_messages, new_labels
    
    def mask_tokens(self, messages: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Mask random tokens in messages"""
        mask_rate = 0.15
        masked_messages = []
        
        for msg in messages:
            words = msg.split()
            masked_words = []
            for word in words:
                if random.random() < mask_rate:
                    masked_words.append('[MASK]')
                else:
                    masked_words.append(word)
            masked_messages.append(' '.join(masked_words))
        
        return masked_messages, labels

# ------------------------- Dataset Class ------------------------- #
class SlidingWindowConversationDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        tokenizer,
        window_size: int = 10,
        stride: int = 5,
        max_length: int = 512,
        augmenter=None,
        is_training: bool = True
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.max_length = max_length
        self.augmenter = augmenter
        self.is_training = is_training

        # make sure "[MSG]" is a recognized token
        if "[MSG]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": ["[MSG]"]
            })
        self.msg_token_id = self.tokenizer.convert_tokens_to_ids("[MSG]")

        self.samples = []
        self._load_data(input_dir)
        self._log_statistics()

    def _load_data(self, input_dir: str):
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        for file in tqdm(json_files, desc="Loading conversations"):
            path = os.path.join(input_dir, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for conv in data:
                    utts = conv.get('utterance', [])
                    if len(utts) >= self.window_size:
                        self._create_windows(utts, conv.get('id', 'unknown'))
            except Exception as e:
                # you already have logger
                print(f"Error loading {file}: {e}")

    def _create_windows(self, utterances, conv_id):
        for start in range(0, len(utterances) - self.window_size + 1, self.stride):
            end = start + self.window_size
            msgs = [u['original_form'] for u in utterances[start:end]]
            labs = [u.get('label',0) for u in utterances[start:end]]

            # optional augmentation (be careful: best if it preserves count)
            if self.is_training and self.augmenter:
                prob = 0.7 if any(labs) else 0.2
                if random.random() < prob:
                    msgs, labs = self.augmenter.augment(msgs, labs)

            # build conversation text with [MSG] at each message start
            parts = []
            for i,(m,l) in enumerate(zip(msgs,labs)):
                sp = f"[SPEAKER_{i%2}]"
                parts.append(f"[MSG] {sp} {m}")
            conv_text = " [SEP] ".join(parts)

            win_label = 1 if any(labs) else 0
            anomaly_pos = [i for i,l in enumerate(labs) if l==1]

            # build the target distribution over messages
            tgt = np.zeros(self.window_size, dtype=float)
            if win_label==1:
                for p in anomaly_pos:
                    if p < self.window_size:
                        tgt[p] = 1.0/len(anomaly_pos)

            self.samples.append({
                "text": conv_text,
                "label": win_label,
                "msg_target": tgt,                # <-- new
                "num_messages": len(msgs),
                "conv_id": conv_id,
            })

    def _log_statistics(self):
        labs = [s['label'] for s in self.samples]
        cnt = Counter(labs)
        print(f" Total windows: {len(labs)}")
        print(f"  anomalies: {cnt[1]} ({cnt[1]/len(labs)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)        # [L]
        attn_mask = enc['attention_mask'].squeeze(0)   # [L]

        # find all [MSG] positions in this token sequence
        pos = (input_ids == self.msg_token_id).nonzero(as_tuple=False).view(-1)
        # pad or truncate to exactly window_size
        if pos.numel() > self.window_size:
            pos = pos[:self.window_size]
        elif pos.numel() < self.window_size:
            pad = torch.zeros(self.window_size - pos.numel(), dtype=torch.long)
            pos = torch.cat([pos, pad], dim=0)

        return {
            "input_ids":    input_ids,
            "attention_mask": attn_mask,
            "label":         torch.tensor(s['label'], dtype=torch.long),
            "num_messages":  torch.tensor(s['num_messages'], dtype=torch.long),
            "msg_target":    torch.tensor(s['msg_target'], dtype=torch.float),  # [W]
            "msg_positions": pos,  # [W]
        }

# ------------------------- Model Architecture ------------------------- #
class ConversationAnomalyDetector(nn.Module):
    """Pure transformer-based model for anomaly detection"""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        hidden_dropout: float = 0.3,
        attention_dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        super().__init__()
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout)
        
        # Attention pooling layer (if enabled)
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_weights = nn.Linear(hidden_size, 1)
            self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of classification head"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pooling strategy
        if self.use_attention_pooling:
            # Attention-based pooling
            attention_scores = self.attention_weights(sequence_output).squeeze(-1)  # (batch_size, seq_len)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
            attention_weights = self.attention_dropout(attention_weights)
            
            # Weighted sum of hidden states
            pooled_output = torch.bmm(
                attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
                sequence_output  # (batch_size, seq_len, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
        else:
            # Use CLS token
            pooled_output = sequence_output[:, 0]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits, outputs.attentions

# ------------------------- Loss Functions ------------------------- #
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = -(targets_one_hot * F.log_softmax(inputs, dim=-1)).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------- Training ------------------------- #
class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config,
        device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        
        self.best_val_score = 0
        self.patience_counter = 0
        self.global_step = 0
    
    def train_epoch(self):
        """Train for one epoch with robust attention‐supervision."""
        self.model.train()
        total_loss = 0.0
        predictions, labels = [], []

        pbar = tqdm(self.train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            batch_labels   = batch['label'].to(self.device)
            msg_positions  = batch['msg_positions'].to(self.device)  # [B, W]
            msg_target     = batch['msg_target'].to(self.device)     # [B, W]

            # 1) forward
            logits, attn_weights = self.model(input_ids, attention_mask)

            # 2) window‐level loss
            loss_win = self.criterion(logits, batch_labels)

            # 3) attention‐supervision only on positives
            mask_pos = (batch_labels == 1)
            if mask_pos.any():
                attn_per_msg = attn_weights.gather(1, msg_positions)                # [B, W]
                msg_dist     = attn_per_msg / (attn_per_msg.sum(dim=1, keepdim=True) + 1e-12)

                dist_pos   = msg_dist[mask_pos]    # [Npos, W]
                target_pos = msg_target[mask_pos]  # [Npos, W]

                # avoid log(0)
                dist_pos = dist_pos + 1e-8
                dist_pos = dist_pos / dist_pos.sum(dim=1, keepdim=True)

                # supervised‐attention loss (cross‐entropy form)
                attn_loss = - (target_pos * dist_pos.log()).sum(dim=1).mean()

                loss = loss_win + self.model.attn_supervision_weight * attn_loss
            else:
                attn_loss = None
                loss = loss_win

            # 4) backprop / accumulation
            loss = loss / self.config.ACCUMULATION_STEPS
            loss.backward()
            if (step + 1) % self.config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VAL
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # 5) track metrics
            total_loss += loss.item() * self.config.ACCUMULATION_STEPS
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

            pbar.set_postfix({
                'win_loss':  f"{loss_win.item():.4f}",
                'attn_loss': f"{attn_loss.item() if attn_loss is not None else 0:.4f}",
                'lr':        f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        # 6) epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )

        return {
            'loss':      avg_loss,
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f1':        f1
        }

    
    
    def evaluate(self, dataloader=None):
        """Evaluate the model"""
        if dataloader is None:
            dataloader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        probabilities = []
        all_attentions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                logits, attentions = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, batch_labels)
                
                total_loss += loss.item()
                probs = F.softmax(logits, dim=-1)
                probabilities.extend(probs[:, 1].cpu().numpy())
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                
                # Store attention for analysis (only first few)
                if len(all_attentions) < 5:
                    all_attentions.append(attentions[-1].cpu())  # Last layer attention
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        # Additional metrics for imbalanced data
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # ROC-AUC and Average Precision if we have both classes
        if len(set(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, probabilities)
            metrics['ap'] = average_precision_score(labels, probabilities)
        
        # Confusion matrix
        if len(set(labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        else:
            tn = fp = fn = tp = 0
            if labels[0] == 0:
                tn = len(labels)
            else:
                tp = len(labels)
        
        metrics['confusion_matrix'] = (tn, fp, fn, tp)
        
        return metrics, all_attentions
    
    def save_checkpoint(self, metrics, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch}_f1_{metrics["f1"]:.4f}.pt'
        )
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train metrics: {train_metrics}")
            
            # Evaluate
            val_metrics, _ = self.evaluate()
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Log confusion matrix
            tn, fp, fn, tp = val_metrics['confusion_matrix']
            logger.info(f"Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
            
            # Check for improvement
            current_score = val_metrics['f1']
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(val_metrics, epoch)
                
                best_model_path = os.path.join(
                    self.config.MODEL_DIR,
                    'best_model.pt'
                )
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"New best model saved with F1: {current_score:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
        return self.best_val_score

# ------------------------- Analysis Functions ------------------------- #
def analyze_model_predictions(model, test_loader, tokenizer, device, num_examples=5):
    """Analyze what the model learned"""
    model.eval()
    
    examples = {
        'true_positives': [],
        'false_positives': [],
        'false_negatives': [],
        'true_negatives': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            logits, attentions = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1).cpu()
            
            for i in range(len(preds)):
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                confidence = probs[i, preds[i]].cpu().item()
                
                example = {
                    'text': text,
                    'predicted': preds[i].item(),
                    'actual': labels[i].item(),
                    'confidence': confidence,
                    'attention': attentions[-1][i].mean(dim=0).cpu()  # Average attention heads
                }
                
                if preds[i] == 1 and labels[i] == 1:
                    examples['true_positives'].append(example)
                elif preds[i] == 1 and labels[i] == 0:
                    examples['false_positives'].append(example)
                elif preds[i] == 0 and labels[i] == 1:
                    examples['false_negatives'].append(example)
                else:
                    examples['true_negatives'].append(example)
            
            # Stop if we have enough examples
            if all(len(v) >= num_examples for v in examples.values()):
                break
    
    # Log analysis
    logger.info("\n" + "="*80)
    logger.info("MODEL PREDICTION ANALYSIS")
    logger.info("="*80)
    
    for category, category_examples in examples.items():
        logger.info(f"\n{category.upper().replace('_', ' ')} ({len(category_examples)} total):")
        for i, ex in enumerate(category_examples[:num_examples]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Text: {ex['text'][:300]}...")
            logger.info(f"Confidence: {ex['confidence']:.3f}")

# ------------------------- Main Pipeline ------------------------- #
def main():
    logger.info("Starting Pure Language Model Anomaly Detection Pipeline")
    logger.info(f"Device: {Config.DEVICE}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    # after loading tokenizer…
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[MSG]"]
    })

    # Initialize augmenter
    augmenter = ConversationAugmenter(augment_prob=Config.AUGMENT_PROB) if Config.USE_AUGMENTATION else None
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = SlidingWindowConversationDataset(
        Config.INPUT_DIRECTORY,
        tokenizer,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        max_length=Config.MAX_LENGTH,
        augmenter=augmenter,
        is_training=True
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * Config.TRAIN_SPLIT)
    val_size = int(total_size * Config.VAL_SPLIT)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    msg_tok_id = tokenizer.convert_tokens_to_ids("[MSG]")

    model = ConversationAnomalyDetector(
        Config.MODEL_NAME,
        msg_token_id=msg_tok_id,
        attn_supervision_weight=0.5,
        hidden_dropout=Config.HIDDEN_DROPOUT,
        attention_dropout=Config.ATTENTION_DROPOUT,
        use_attention_pooling=Config.USE_ATTENTION_POOLING
    ).to(Config.DEVICE)
    
    # Resize token embeddings for special tokens
    model.transformer.resize_token_embeddings(len(tokenizer))
    
    # Calculate class weights for imbalanced data
    train_labels = [full_dataset.samples[i]['label'] for i in train_dataset.indices]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)
    logger.info(f"Class weights: {class_weights}")
    
    # Initialize loss function
    if Config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=Config.FOCAL_ALPHA,
            gamma=Config.FOCAL_GAMMA,
            label_smoothing=Config.LABEL_SMOOTHING
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING
        )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * Config.EPOCHS // Config.ACCUMULATION_STEPS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=Config,
        device=Config.DEVICE
    )
    
    # Train model
    best_f1 = trainer.train()
    
    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(Config.MODEL_DIR, 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path))
    
    # Final test evaluation
    logger.info("Evaluating on test set...")
    test_metrics, test_attentions = trainer.evaluate(test_loader)
    
    # Log final results
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
    
    if 'auc' in test_metrics:
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test Average Precision: {test_metrics['ap']:.4f}")
    
    tn, fp, fn, tp = test_metrics['confusion_matrix']
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {tn:6d}")
    logger.info(f"  False Positives: {fp:6d}")
    logger.info(f"  False Negatives: {fn:6d}")
    logger.info(f"  True Positives:  {tp:6d}")
    
    # Calculate additional metrics
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)
        logger.info(f"\nSensitivity (True Positive Rate): {sensitivity:.4f}")
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)
        logger.info(f"Specificity (True Negative Rate): {specificity:.4f}")
    
    # Analyze model predictions
    logger.info("\nAnalyzing model predictions...")
    analyze_model_predictions(model, test_loader, tokenizer, Config.DEVICE)
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'best_validation_f1': best_f1,
        'config': {k: v for k, v in vars(Config).items() if not k.startswith('_')},
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    results_path = os.path.join(
        Config.LOG_DIR,
        f'final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info("Pipeline completed successfully!")

# ------------------------- Inference Functions ------------------------- #
def load_model_for_inference(model_path: str, config: Config):
    """Load trained model for inference"""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    special_tokens = {
        'additional_special_tokens': [
            '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model
    model = ConversationAnomalyDetector(
        config.MODEL_NAME,
        num_labels=2,
        hidden_dropout=config.HIDDEN_DROPOUT,
        attention_dropout=config.ATTENTION_DROPOUT,
        use_attention_pooling=config.USE_ATTENTION_POOLING
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    
    return model, tokenizer

def predict_conversation(
    model,
    tokenizer,
    messages: List[str],
    device: torch.device,
    window_size: int = 10,
    stride: int = 5
) -> List[Dict]:
    """Predict anomalies in a conversation"""
    model.eval()
    predictions = []
    
    # Create sliding windows
    for start_idx in range(0, len(messages) - window_size + 1, stride):
        end_idx = min(start_idx + window_size, len(messages))
        window_messages = messages[start_idx:end_idx]
        
        # Create conversation text
        conversation_parts = []
        for i, msg in enumerate(window_messages):
            speaker = f"[SPEAKER_{i % 2}]"
            conversation_parts.append(f"{speaker} {msg}")
        
        conversation_text = " [SEP] ".join(conversation_parts)
        
        # Tokenize
        encoding = tokenizer(
            conversation_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            logits, attentions = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, pred].item()
        
        predictions.append({
            'window_start': start_idx,
            'window_end': end_idx,
            'prediction': pred,
            'confidence': confidence,
            'anomaly_probability': probs[0, 1].item()
        })
    
    return predictions

# ------------------------- Example Usage ------------------------- #
def example_usage():
    """Example of how to use the trained model"""
    # Load model
    model_path = os.path.join(Config.MODEL_DIR, 'best_model.pt')
    model, tokenizer = load_model_for_inference(model_path, Config)
    
    # Example conversation
    example_messages = [
        "안녕하세요! 오늘 날씨 좋네요",
        "네, 정말 좋아요. 뭐 하고 계세요?",
        "그냥 집에서 쉬고 있어요",
        "저도 그래요. 주말이라 좋네요",
        "혹시 그거 있어?",  # Suspicious transition
        "뭐요?",
        "어제 말한 그거",  # Vague reference
        "아 그거요? 준비됐어요",
        "언제 가능해?",  # Meeting arrangement
        "오늘 저녁쯤?"
    ]
    
    # Get predictions
    predictions = predict_conversation(
        model,
        tokenizer,
        example_messages,
        Config.DEVICE,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE
    )
    
    # Display results
    print("\nAnomaly Detection Results:")
    print("-" * 60)
    for pred in predictions:
        status = "ANOMALOUS" if pred['prediction'] == 1 else "NORMAL"
        print(f"Window [{pred['window_start']}:{pred['window_end']}]: "
              f"{status} (confidence: {pred['confidence']:.3f}, "
              f"anomaly prob: {pred['anomaly_probability']:.3f})")

if __name__ == "__main__":
    main()
    
    # Uncomment to run example usage
    # example_usage()