"""
Evaluate pretrained models (without fine-tuning) on new_data.
This provides a baseline for comparison with the fine-tuned models.
"""

import os
import json
import random
import numpy as np
import argparse
from collections import Counter
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
            'bert_base': 'klue/bert-base',
            'roberta_base': 'klue/roberta-base',
            'roberta_large': 'klue/roberta-large',
            'electra_base': 'monologg/koelectra-base-v3-discriminator',
            'distillbert_base': 'bongsoo/mdistilbertV3.1'
        }

        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Model type must be one of {list(MODEL_CONFIGS.keys())}")

        # Data path
        self.INPUT_FILE = '/home/minseok/forensic/new_data_evaluation/preprocessed/all_conversations.json'

        # Model parameters
        self.MODEL_TYPE = model_type
        self.MODEL_NAME = MODEL_CONFIGS[model_type]
        self.BATCH_SIZE = 16

        # Sliding window parameters
        self.WINDOW_SIZE = 10
        self.STRIDE = 5
        self.MAX_LENGTH = 512

        # Output paths
        self.LOG_DIR = f'/home/minseok/forensic/new_data_evaluation/logs_plain_{model_type}'
        self.RESULTS_DIR = f'/home/minseok/forensic/new_data_evaluation/results_plain_{model_type}'

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
class SimpleConversationDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        tokenizer,
        window_size: int = 10,
        stride: int = 5,
        max_length: int = 512,
        messengers: List[str] = None
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.max_length = max_length
        self.messengers = messengers

        self.samples = []
        self._load_data(data_file)
        self._log_statistics()

    def _load_data(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        # Filter by messenger if specified
        if self.messengers:
            conversations = [c for c in conversations if c.get('source') in self.messengers]
            print(f"Filtered to {len(conversations)} conversations from: {', '.join(self.messengers)}")

        for conv in tqdm(conversations, desc="Loading conversations"):
            utts = conv.get('utterance', [])
            if len(utts) >= self.window_size:
                self._create_windows(utts, conv.get('id', 'unknown'))

    def _create_windows(self, utterances, conv_id):
        for start in range(0, len(utterances) - self.window_size + 1, self.stride):
            end = start + self.window_size
            msgs = [u['original_form'] for u in utterances[start:end]]
            labs = [u.get('label', 0) for u in utterances[start:end]]

            # Create conversation text with Korean instruction
            conversation_text = self._create_korean_prompt(msgs)
            window_label = 1 if any(labs) else 0

            self.samples.append({
                "text": conversation_text,
                "label": window_label,
                "conv_id": conv_id,
                "start_idx": start,
                "end_idx": end,
                "original_messages": msgs,
                "original_labels": labs,
            })

    def _create_korean_prompt(self, messages: List[str]) -> str:
        """Create Korean prompt for drug-related classification"""
        # Join messages with speaker indicators
        conversation = ""
        for i, msg in enumerate(messages):
            speaker = "화자A" if i % 2 == 0 else "화자B"
            conversation += f"{speaker}: {msg}\n"

        # Add Korean instruction
        prompt = f"""다음 대화를 읽고 마약 관련 내용이 포함되어 있는지 판단하세요.

대화:
{conversation}

질문: 이 대화에 마약 관련 내용이 포함되어 있습니까?
답변: """

        return prompt

    def _log_statistics(self):
        labs = [s['label'] for s in self.samples]
        cnt = Counter(labs)
        print(f"Total windows: {len(labs)}")
        print(f"  Normal: {cnt[0]} ({cnt[0]/len(labs)*100:.1f}%)")
        print(f"  Drug-related: {cnt[1]} ({cnt[1]/len(labs)*100:.1f}%)")

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
        input_ids = enc['input_ids'].squeeze(0)
        attn_mask = enc['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "label": torch.tensor(s['label'], dtype=torch.long),
            "conv_id": s['conv_id'],
            "start_idx": s['start_idx'],
            "end_idx": s['end_idx'],
            "original_messages": s['original_messages'],
            "original_labels": s['original_labels'],
        }

# ------------------------- Custom Collate Function ------------------------- #
def custom_collate_fn(batch):
    """Custom collate function to handle mixed data types"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    conv_ids = [item['conv_id'] for item in batch]
    start_indices = [item['start_idx'] for item in batch]
    end_indices = [item['end_idx'] for item in batch]
    original_messages = [item['original_messages'] for item in batch]
    original_labels = [item['original_labels'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'conv_id': conv_ids,
        'start_idx': start_indices,
        'end_idx': end_indices,
        'original_messages': original_messages,
        'original_labels': original_labels,
    }

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

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
                    'start_idx': batch['start_idx'][i],
                    'end_idx': batch['end_idx'][i],
                    'predicted_label': int(batch_preds[i]),
                    'actual_label': int(batch_labels_np[i]),
                    'drug_probability': float(batch_probs[i]),
                    'confidence': float(probs[i, batch_preds[i]]),
                    'original_messages': batch['original_messages'][i],
                    'original_labels': batch['original_labels'][i],
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
    parser = argparse.ArgumentParser(description='Evaluate pretrained models (no fine-tuning) on new dataset')
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
    logger.info(f"Starting PLAIN evaluation of {args.model} on new dataset")
    logger.info(f"Messengers: {messenger_info}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME} (pretrained, not fine-tuned)")
    logger.info(f"Data path: {config.INPUT_FILE}")

    try:
        # Check if data file exists
        if not os.path.exists(config.INPUT_FILE):
            raise FileNotFoundError(f"Data file not found: {config.INPUT_FILE}")

        # Initialize tokenizer and model
        logger.info("Loading pretrained model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=2
        ).to(config.DEVICE)

        logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Load dataset
        logger.info("Loading dataset...")
        dataset = SimpleConversationDataset(
            config.INPUT_FILE,
            tokenizer,
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE,
            max_length=config.MAX_LENGTH,
            messengers=messengers
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        # Evaluate
        logger.info("Starting evaluation...")
        metrics, detailed_predictions = evaluate_model(model, dataloader, config.DEVICE, logger)

        # Log results
        logger.info("\n" + "="*80)
        logger.info(f"PLAIN {args.model.upper()} EVALUATION RESULTS ON NEW DATA")
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

        # Calculate specificity
        specificity = None
        if cm['tn'] + cm['fp'] > 0:
            specificity = cm['tn'] / (cm['tn'] + cm['fp'])
            logger.info(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
            logger.info(f"False Positive Rate: {1 - specificity:.4f}")

        # Save results
        results = {
            'model_type': args.model,
            'model_name': config.MODEL_NAME,
            'evaluation_type': 'plain (pretrained, no fine-tuning)',
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'max_length': config.MAX_LENGTH,
                'window_size': config.WINDOW_SIZE,
                'stride': config.STRIDE,
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
        print(f"MODEL: {args.model} (PLAIN - NO FINE-TUNING)")
        print(f"ACCURACY: {metrics['accuracy']:.4f}")
        print(f"PRECISION: {metrics['precision']:.4f}")
        print(f"RECALL: {metrics['recall']:.4f}")
        print(f"F1-SCORE: {metrics['f1']:.4f}")
        if specificity is not None:
            print(f"SPECIFICITY: {specificity:.4f}")
            print(f"FALSE_POSITIVE_RATE: {1-specificity:.4f}")
        print(f"RESULTS_DIR: {config.RESULTS_DIR}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
