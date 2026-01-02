"""
Evaluate trained models (bert_base, roberta_base, roberta_large) on new_data.
Uses the same evaluation logic from train.py but adapted for the new dataset.
"""

import os
import json
import random
import numpy as np
import argparse
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Configuration ------------------------- #
class Config:
    def __init__(self, model_type: str):
        # Model mappings
        MODEL_CONFIGS = {
            'bert_base': {
                'name': 'klue/bert-base',
                'model_path': '/home/minseok/forensic/bert_base/models_pure_lm_attn/best_model.pt'
            },
            'roberta_base': {
                'name': 'klue/roberta-base',
                'model_path': '/home/minseok/forensic/roberta_base/models_pure_lm_attn/best_model.pt'
            },
            'roberta_large': {
                'name': 'klue/roberta-large',
                'model_path': '/home/minseok/forensic/roberta_large/models_pure_lm_attn/best_model.pt'
            },
            'electra_base': {
                'name': 'monologg/koelectra-base-v3-discriminator',
                'model_path': '/home/minseok/forensic/electra_base/models_electra/best_model.pt'
            },
            'distillbert_base': {
                'name': 'bongsoo/mdistilbertV3.1',
                'model_path': '/home/minseok/forensic/distillbert_base/models_distilbert/best_model.pt'
            }
        }

        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Model type must be one of {list(MODEL_CONFIGS.keys())}")

        # Data path
        self.INPUT_FILE = '/home/minseok/forensic/new_data_evaluation/preprocessed/all_conversations.json'

        # Model parameters
        self.MODEL_TYPE = model_type
        self.MODEL_NAME = MODEL_CONFIGS[model_type]['name']
        self.MODEL_PATH = MODEL_CONFIGS[model_type]['model_path']
        self.BATCH_SIZE = 16

        # Context parameters
        self.MAX_CONTEXT_LENGTH = 512
        self.CONTEXT_WINDOW = 5

        # Output paths
        self.LOG_DIR = f'/home/minseok/forensic/new_data_evaluation/logs_{model_type}'
        self.RESULTS_DIR = f'/home/minseok/forensic/new_data_evaluation/results_{model_type}'

        # Device
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.RANDOM_SEED = 42

# ------------------------- Logging Setup ------------------------- #
def setup_logging(config):
    for dir_path in [config.LOG_DIR, config.RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    log_filename = os.path.join(
        config.LOG_DIR,
        f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------------- Dataset Class ------------------------- #
class SimpleEvaluationDataset(Dataset):
    """Simplified dataset for evaluation without feature extraction"""

    def __init__(
        self,
        data_file: str,
        tokenizer,
        context_window: int = 5,
        max_length: int = 512,
        messengers: List[str] = None
    ):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.max_length = max_length
        self.messengers = messengers

        self.samples = []
        self.load_data(data_file)
        self._log_statistics()

    def load_data(self, data_file: str):
        """Load preprocessed JSON data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        # Filter by messenger if specified
        if self.messengers:
            conversations = [c for c in conversations if c.get('source') in self.messengers]
            print(f"Filtered to {len(conversations)} conversations from: {', '.join(self.messengers)}")

        for conversation in tqdm(conversations, desc="Loading conversations"):
            utterances = conversation.get('utterance', [])
            conv_id = conversation.get('id', 'unknown')
            self._process_conversation(utterances, conv_id)

    def _process_conversation(self, utterances: List[Dict], conv_id: str):
        """Process a single conversation with context"""
        for i, utt in enumerate(utterances):
            label = utt.get('label', 0)

            # Get context
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(utterances), i + self.context_window + 1)

            # Build context with position information
            context_messages = []

            for j in range(start_idx, end_idx):
                msg = utterances[j]['original_form']

                if j == i:
                    msg = f"[CUR] {msg} [/CUR]"

                context_messages.append(msg)

            context_text = " [SEP] ".join(context_messages)

            sample = {
                'text': context_text,
                'label': label,
                'conv_id': conv_id,
                'position': i
            }

            self.samples.append(sample)

    def _log_statistics(self):
        labels = [s['label'] for s in self.samples]
        label_counts = Counter(labels)

        print(f"Dataset Statistics:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Class distribution: {dict(label_counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Note: No additional features for pure language model evaluation
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'conv_id': sample['conv_id'],
            'position': sample['position']
        }

# ------------------------- Model Architecture (Same as legacy_msg.py) ------------------------- #
class PureLMModel(nn.Module):
    """Pure language model with attention pooling (matches legacy_msg.py architecture)"""

    def __init__(self, model_name: str, num_labels: int = 2, dropout_rate: float = 0.3,
                 attention_dropout: float = 0.1, use_attention_pooling: bool = True):
        super().__init__()

        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Attention pooling layer (if enabled)
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attn_scoring = nn.Linear(hidden_size, 1)  # Named 'attn_scoring' to match saved model
            self.attention_dropout = nn.Dropout(attention_dropout)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
            attention_scores = self.attn_scoring(sequence_output).squeeze(-1)  # (batch_size, seq_len)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attn_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
            attn_weights_dropout = self.attention_dropout(attn_weights)

            # Weighted sum of hidden states
            pooled_output = torch.bmm(
                attn_weights_dropout.unsqueeze(1),  # (batch_size, 1, seq_len)
                sequence_output  # (batch_size, seq_len, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
        else:
            # Use CLS token
            pooled_output = sequence_output[:, 0]
            attn_weights = None

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)

        return logits, attn_weights

# ------------------------- Evaluation Function ------------------------- #
def evaluate_model(model, dataloader, device, logger):
    """Evaluate the model on given dataloader"""
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    detailed_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            logits, _ = model(input_ids, attention_mask)

            probs = F.softmax(logits, dim=-1)
            batch_probs = probs[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_labels_np = batch_labels.cpu().numpy()

            probabilities.extend(batch_probs)
            predictions.extend(batch_preds)
            labels.extend(batch_labels_np)

            # Store detailed predictions
            for i in range(len(batch_preds)):
                detailed_predictions.append({
                    'conv_id': batch['conv_id'][i],
                    'position': int(batch['position'][i]),
                    'predicted_label': int(batch_preds[i]),
                    'actual_label': int(batch_labels_np[i]),
                    'drug_probability': float(batch_probs[i]),
                    'confidence': float(probs[i, batch_preds[i]])
                })

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_samples': len(labels),
        'positive_samples': sum(labels),
        'negative_samples': len(labels) - sum(labels),
        'predicted_positive': sum(predictions),
        'predicted_negative': len(predictions) - sum(predictions)
    }

    # Additional metrics
    if len(set(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, probabilities)
        metrics['ap'] = average_precision_score(labels, probabilities)
    else:
        metrics['auc'] = 0.5
        metrics['ap'] = sum(labels) / len(labels)

    # Confusion matrix
    if len(set(labels)) > 1:
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            if labels[0] == 0:
                tn, fp, fn, tp = len(labels), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(labels)
    else:
        if labels[0] == 0:
            tn, fp, fn, tp = len(labels), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(labels)

    metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

    return metrics, detailed_predictions

# ------------------------- Main Function ------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on new dataset')
    parser.add_argument('--model', type=str, required=True,
                       choices=['bert_base', 'roberta_base', 'roberta_large', 'electra_base', 'distillbert_base'],
                       help='Model type to evaluate')
    parser.add_argument('--messenger', type=str, nargs='+',
                       choices=['band', 'facebook', 'instagram', 'nateon', 'all'],
                       default=['all'],
                       help='Messenger platforms to evaluate (default: all)')

    args = parser.parse_args()

    # Parse messenger argument
    if 'all' in args.messenger:
        messengers = None
        messenger_suffix = 'all'
    else:
        messengers = args.messenger
        messenger_suffix = '_'.join(messengers)

    # Initialize config
    config = Config(args.model)

    # Update output paths to include messenger suffix if filtering
    if messengers:
        config.LOG_DIR = f'{config.LOG_DIR}_{messenger_suffix}'
        config.RESULTS_DIR = f'{config.RESULTS_DIR}_{messenger_suffix}'

    # Setup logging
    logger = setup_logging(config)
    set_seed(config.RANDOM_SEED)

    messenger_info = ', '.join(messengers) if messengers else 'all messengers'
    logger.info(f"Starting evaluation of {args.model} on new dataset")
    logger.info(f"Messengers: {messenger_info}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model path: {config.MODEL_PATH}")
    logger.info(f"Data path: {config.INPUT_FILE}")

    try:
        # Check if files exist
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")
        if not os.path.exists(config.INPUT_FILE):
            raise FileNotFoundError(f"Data file not found: {config.INPUT_FILE}")

        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Add special tokens to tokenizer FIRST (before model initialization)
        # These must match the tokens used during training
        special_tokens = {
            'additional_special_tokens': [
                '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["[MSG]"]
        })
        logger.info(f"Added special tokens to tokenizer (total vocab: {len(tokenizer)})")

        # Initialize model
        logger.info("Initializing model...")
        model = PureLMModel(config.MODEL_NAME)
        model.to(config.DEVICE)

        # Resize embeddings to match tokenizer (includes special tokens)
        model.transformer.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

        # NOW load trained weights (embedding sizes will match)
        logger.info(f"Loading model weights from {config.MODEL_PATH}...")
        state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        # Use strict=False to ignore position_ids buffer (auto-generated, not trainable)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully!")

        # Load dataset
        logger.info("Loading dataset...")
        dataset = SimpleEvaluationDataset(
            config.INPUT_FILE,
            tokenizer,
            context_window=config.CONTEXT_WINDOW,
            max_length=config.MAX_CONTEXT_LENGTH,
            messengers=messengers
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Evaluate
        logger.info("Starting evaluation...")
        metrics, detailed_predictions = evaluate_model(model, dataloader, config.DEVICE, logger)

        # Log results
        logger.info("\n" + "="*80)
        logger.info(f"{args.model.upper()} EVALUATION RESULTS ON NEW DATA")
        logger.info("="*80)
        logger.info(f"Total Samples: {metrics['total_samples']}")
        logger.info(f"Actual Positive: {metrics['positive_samples']} ({metrics['positive_samples']/metrics['total_samples']*100:.2f}%)")
        logger.info(f"Actual Negative: {metrics['negative_samples']} ({metrics['negative_samples']/metrics['total_samples']*100:.2f}%)")
        logger.info(f"Predicted Positive: {metrics['predicted_positive']} ({metrics['predicted_positive']/metrics['total_samples']*100:.2f}%)")
        logger.info(f"Predicted Negative: {metrics['predicted_negative']} ({metrics['predicted_negative']/metrics['total_samples']*100:.2f}%)")
        logger.info(f"\nAccuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Average Precision: {metrics['ap']:.4f}")

        cm = metrics['confusion_matrix']
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm['tn']:6d}  FP: {cm['fp']:6d}")
        logger.info(f"  FN: {cm['fn']:6d}  TP: {cm['tp']:6d}")

        # Calculate specificity (important for clean data)
        if cm['tn'] + cm['fp'] > 0:
            specificity = cm['tn'] / (cm['tn'] + cm['fp'])
            logger.info(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
            logger.info(f"False Positive Rate: {1 - specificity:.4f}")

        # Save results
        results = {
            'model_type': args.model,
            'model_name': config.MODEL_NAME,
            'model_path': config.MODEL_PATH,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'max_length': config.MAX_CONTEXT_LENGTH,
                'context_window': config.CONTEXT_WINDOW,
                'random_seed': config.RANDOM_SEED,
            }
        }

        results_path = os.path.join(
            config.RESULTS_DIR,
            f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # Save detailed predictions
        predictions_path = os.path.join(
            config.RESULTS_DIR,
            f'detailed_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {results_path}")
        logger.info(f"Detailed predictions saved to: {predictions_path}")
        logger.info("Evaluation completed successfully!")

        # Print summary
        print(f"\n{'='*60}")
        print(f"MODEL: {args.model}")
        print(f"ACCURACY: {metrics['accuracy']:.4f}")
        print(f"PRECISION: {metrics['precision']:.4f}")
        print(f"RECALL: {metrics['recall']:.4f}")
        print(f"F1-SCORE: {metrics['f1']:.4f}")
        print(f"SPECIFICITY: {specificity:.4f}" if cm['tn'] + cm['fp'] > 0 else "SPECIFICITY: N/A")
        print(f"FALSE_POSITIVE_RATE: {1-specificity:.4f}" if cm['tn'] + cm['fp'] > 0 else "FALSE_POSITIVE_RATE: N/A")
        print(f"RESULTS_DIR: {config.RESULTS_DIR}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
