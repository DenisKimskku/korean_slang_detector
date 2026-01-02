import os
import json
import random
import numpy as np
import argparse
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Configuration ------------------------- #
class Config:
    def __init__(self, model_name: str, output_suffix: str):
        # Data paths
        self.INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
        
        # Model parameters
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = 16
        
        # Data splits
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        
        # Sliding window parameters
        self.WINDOW_SIZE = 10  # Number of messages in context
        self.STRIDE = 5  # Window stride
        self.MAX_LENGTH = 512
        
        # Output paths
        self.LOG_DIR = f'logs_{output_suffix}'
        self.RESULTS_DIR = f'results_{output_suffix}'
        self.PREDICTIONS_DIR = f'predictions_{output_suffix}'
        
        # Device
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.RANDOM_SEED = 42

# ------------------------- Logging Setup ------------------------- #
def setup_logging(config):
    # Create directories
    for dir_path in [config.LOG_DIR, config.RESULTS_DIR, config.PREDICTIONS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    log_filename = os.path.join(
        config.LOG_DIR, 
        f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # Clear any existing handlers
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

# ------------------------- Custom Collate Function ------------------------- #
def custom_collate_fn(batch):
    """Custom collate function to handle mixed data types"""
    # Extract tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Extract non-tensor data
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
        input_dir: str,
        tokenizer,
        window_size: int = 10,
        stride: int = 5,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.max_length = max_length
        
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
                print(f"Error loading {file}: {e}")

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

# ------------------------- Evaluation Function ------------------------- #
def evaluate_model(model, dataloader, device, config, logger, save_predictions=True):
    """Evaluate the model on given dataloader"""
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    detailed_predictions = []
    
    logger.info(f"Starting evaluation with {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]  # First element is usually logits
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim=-1)
            batch_probs = probs[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_labels_np = batch_labels.cpu().numpy()
            
            probabilities.extend(batch_probs)
            predictions.extend(batch_preds)
            labels.extend(batch_labels_np)
            
            # Store detailed predictions for analysis
            if save_predictions:
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
    
    # Save detailed predictions
    if save_predictions:
        pred_file = os.path.join(
            config.PREDICTIONS_DIR,
            f'detailed_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed predictions saved to: {pred_file}")
    
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
    }
    
    # Additional metrics for imbalanced data
    if len(set(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, probabilities)
        metrics['ap'] = average_precision_score(labels, probabilities)
    else:
        metrics['auc'] = 0.5  # Default value when only one class
        metrics['ap'] = sum(labels) / len(labels)  # Proportion of positive class
    
    # Confusion matrix
    if len(set(labels)) > 1:
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case where only one class is present
            if labels[0] == 0:
                tn, fp, fn, tp = len(labels), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(labels)
    else:
        if labels[0] == 0:
            tn, fp, fn, tp = len(labels), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(labels)
    
    metrics['confusion_matrix'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    
    return metrics, detailed_predictions

# ------------------------- Analysis Functions ------------------------- #
def analyze_predictions(model, test_loader, tokenizer, device, logger, num_examples=3):
    """Analyze model predictions and show examples"""
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]  # First element is usually logits
            else:
                logits = outputs
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
                }
                
                if preds[i] == 1 and labels[i] == 1:
                    examples['true_positives'].append(example)
                elif preds[i] == 1 and labels[i] == 0:
                    examples['false_positives'].append(example)
                elif preds[i] == 0 and labels[i] == 1:
                    examples['false_negatives'].append(example)
                else:
                    examples['true_negatives'].append(example)
            
            # Stop if we have enough examples for all categories
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
            # Show only the conversation part, not the full prompt
            conv_start = ex['text'].find('대화:')
            conv_end = ex['text'].find('\n\n질문:')
            if conv_start != -1 and conv_end != -1:
                conversation = ex['text'][conv_start:conv_end].replace('대화:\n', '')
                logger.info(f"Conversation:\n{conversation}")
            else:
                logger.info(f"Text: {ex['text'][:300]}...")
            logger.info(f"Predicted: {ex['predicted']} | Actual: {ex['actual']}")
            logger.info(f"Confidence: {ex['confidence']:.3f}")

# ------------------------- Main Evaluation Pipeline ------------------------- #
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate KLUE models for drug detection')
    parser.add_argument('--model', type=str, required=True,
                       choices=['klue/roberta-small', 'klue/roberta-base', 
                               'klue/roberta-large', 'klue/bert-base'],
                       help='Model name to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Create model-specific suffix
    model_suffix = args.model.replace('klue/', '').replace('-', '_')
    
    # Initialize config
    config = Config(args.model, model_suffix)
    config.BATCH_SIZE = args.batch_size
    config.MAX_LENGTH = args.max_length
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    logger.info(f"Starting Drug Detection Evaluation with {args.model}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Output directories: {config.LOG_DIR}, {config.RESULTS_DIR}, {config.PREDICTIONS_DIR}")
    
    try:
        # Initialize tokenizer and model
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2
        ).to(config.DEVICE)
        
        logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Load dataset
        logger.info("Loading dataset...")
        full_dataset = SimpleConversationDataset(
            config.INPUT_DIRECTORY,
            tokenizer,
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE,
            max_length=config.MAX_LENGTH,
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * config.TRAIN_SPLIT)
        val_size = int(total_size * config.VAL_SPLIT)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )
        
        logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Create test data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics, detailed_predictions = evaluate_model(
            model, test_loader, config.DEVICE, config, logger
        )
        
        # Log results
        logger.info("\n" + "="*80)
        logger.info(f"{args.model.upper()} EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test Average Precision: {test_metrics['ap']:.4f}")
        
        cm = test_metrics['confusion_matrix']
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {cm['tn']:6d}")
        logger.info(f"  False Positives: {cm['fp']:6d}")
        logger.info(f"  False Negatives: {cm['fn']:6d}")
        logger.info(f"  True Positives:  {cm['tp']:6d}")
        
        # Calculate additional metrics
        if cm['tp'] + cm['fn'] > 0:
            sensitivity = cm['tp'] / (cm['tp'] + cm['fn'])
            logger.info(f"\nSensitivity (TPR): {sensitivity:.4f}")
        
        if cm['tn'] + cm['fp'] > 0:
            specificity = cm['tn'] / (cm['tn'] + cm['fp'])
            logger.info(f"Specificity (TNR): {specificity:.4f}")
        
        # Analyze model predictions
        logger.info("\nAnalyzing model predictions...")
        analyze_predictions(model, test_loader, tokenizer, config.DEVICE, logger)
        
        # Save results
        results = {
            'model_name': args.model,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_metrics': test_metrics,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'max_length': config.MAX_LENGTH,
                'window_size': config.WINDOW_SIZE,
                'stride': config.STRIDE,
                'random_seed': config.RANDOM_SEED,
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            'dataset_info': {
                'total_samples': total_size,
                'test_samples': test_size,
                'positive_samples': test_metrics['positive_samples'],
                'negative_samples': test_metrics['negative_samples'],
            }
        }
        
        results_path = os.path.join(
            config.RESULTS_DIR,
            f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\nResults saved to: {results_path}")
        logger.info("Evaluation completed successfully!")
        
        # Print summary for shell script
        print(f"\n{'='*60}")
        print(f"MODEL: {args.model}")
        print(f"ACCURACY: {test_metrics['accuracy']:.4f}")
        print(f"PRECISION: {test_metrics['precision']:.4f}")
        print(f"RECALL: {test_metrics['recall']:.4f}")
        print(f"F1-SCORE: {test_metrics['f1']:.4f}")
        print(f"AUC: {test_metrics['auc']:.4f}")
        print(f"RESULTS_DIR: {config.RESULTS_DIR}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()