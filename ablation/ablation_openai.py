import os
import json
import random
import asyncio
import aiohttp
import numpy as np
import argparse
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

import logging
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Configuration ------------------------- #
class Config:
    def __init__(self, model_name: str = "gpt-4o"):
        # Data paths
        self.INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
        # self.INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/test/modified5'
        self.CONFIG_PATH = '/home/minseok/PoisonedRAG/model_configs/gpt4_config.json'
        
        # Model parameters
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = 16  # Process 16 conversations at once
        self.MAX_TOKENS = 1000
        self.TEMPERATURE = 0.1  # Low temperature for consistent responses
        
        # Data splits
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        
        # Sliding window parameters
        self.WINDOW_SIZE = 10
        self.STRIDE = 5
        
        # API parameters
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2  # seconds
        self.REQUEST_DELAY = 0.1  # Delay between requests to avoid rate limiting
        
        # Output paths
        self.LOG_DIR = f'logs_openai_{model_name.replace("-", "_")}'
        self.RESULTS_DIR = f'results_openai_{model_name.replace("-", "_")}'
        self.PREDICTIONS_DIR = f'predictions_openai_{model_name.replace("-", "_")}'
        
        self.RANDOM_SEED = 42

# ------------------------- Utility Functions ------------------------- #
def load_json(file_path: str) -> dict:
    """Load JSON configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")

def setup_logging(config):
    """Setup logging configuration"""
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

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

# ------------------------- OpenAI API Client ------------------------- #
class OpenAIClient:
    def __init__(self, config_path: str, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.config = load_json(config_path)
        
        # Extract API key
        api_keys = self.config['api_key_info']['api_keys']
        self.api_key = ''.join(api_keys) if isinstance(api_keys, list) else api_keys
        
        # API endpoint
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests
        
    def create_batch_prompt(self, conversation_windows: List[Dict]) -> str:
        """Create a batch prompt for multiple conversation windows"""
        
        batch_prompt = """다음은 여러 개의 대화 창입니다. 각 대화를 읽고 마약 관련 내용이 포함되어 있는지 판단하세요.

각 대화에 대해 다음 형식으로 답변해주세요:
대화 [번호]: [예/아니오]

"""
        
        for i, window in enumerate(conversation_windows, 1):
            messages = window['messages']
            conversation_text = ""
            for j, msg in enumerate(messages):
                speaker = "화자A" if j % 2 == 0 else "화자B"
                conversation_text += f"{speaker}: {msg}\n"
            
            batch_prompt += f"""대화 {i}:
{conversation_text}

"""
        
        batch_prompt += f"""질문: 위의 {len(conversation_windows)}개 대화 중 마약 관련 내용이 포함된 대화는 어느 것입니까?

답변 형식:
대화 1: [예/아니오]
대화 2: [예/아니오]
...
대화 {len(conversation_windows)}: [예/아니오]

답변:"""
        print(batch_prompt)
        return batch_prompt
    
    async def make_api_request(self, session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make async API request to OpenAI"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 한국어 대화에서 마약 관련 내용을 탐지하는 전문가입니다. 정확하고 일관성 있게 답변해주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                async with session.post(self.api_url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        logging.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logging.error(f"API request failed with status {response.status}: {error_text}")
                        if attempt == max_retries - 1:
                            return None
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logging.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    def parse_batch_response(self, response: str, num_windows: int) -> List[int]:
        """Parse batch response and extract predictions"""
        predictions = []
        
        lines = response.strip().split('\n')
        for i in range(1, num_windows + 1):
            found = False
            for line in lines:
                if f"대화 {i}:" in line:
                    if "예" in line:
                        predictions.append(1)
                    elif "아니오" in line:
                        predictions.append(0)
                    else:
                        predictions.append(0)  # Default to negative
                    found = True
                    break
            
            if not found:
                predictions.append(0)  # Default to negative if not found
        
        return predictions

# ------------------------- Dataset Class ------------------------- #
class ConversationDataset:
    def __init__(self, input_dir: str, window_size: int = 10, stride: int = 5):
        self.window_size = window_size
        self.stride = stride
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
            
            window_label = 1 if any(labs) else 0
            
            self.samples.append({
                "messages": msgs,
                "label": window_label,
                "conv_id": conv_id,
                "start_idx": start,
                "end_idx": end,
                "original_labels": labs,
            })
    
    def _log_statistics(self):
        labs = [s['label'] for s in self.samples]
        cnt = Counter(labs)
        print(f"Total windows: {len(labs)}")
        print(f"  Normal: {cnt[0]} ({cnt[0]/len(labs)*100:.1f}%)")
        print(f"  Drug-related: {cnt[1]} ({cnt[1]/len(labs)*100:.1f}%)")
    
    def get_test_split(self, test_split: float = 0.15, random_seed: int = 42):
        """Get test split of the dataset"""
        random.seed(random_seed)
        shuffled_samples = self.samples.copy()
        random.shuffle(shuffled_samples)
        
        test_size = int(len(shuffled_samples) * test_split)
        return shuffled_samples[-test_size:]

# ------------------------- Evaluation Function ------------------------- #
async def evaluate_with_openai(openai_client: OpenAIClient, test_samples: List[Dict], 
                              batch_size: int, logger, config) -> Tuple[Dict, List[Dict]]:
    """Evaluate samples using OpenAI API with batching"""
    
    all_predictions = []
    all_labels = []
    detailed_predictions = []
    
    # Create batches
    batches = [test_samples[i:i + batch_size] for i in range(0, len(test_samples), batch_size)]
    
    logger.info(f"Processing {len(test_samples)} samples in {len(batches)} batches of size {batch_size}")
    
    async with aiohttp.ClientSession() as session:
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            try:
                # Create batch prompt
                prompt = openai_client.create_batch_prompt(batch)
                
                # Make API request
                response = await openai_client.make_api_request(session, prompt)
                print(response)
                if response is None:
                    logger.error(f"Failed to get response for batch {batch_idx}")
                    # Use default predictions (all negative)
                    batch_predictions = [0] * len(batch)
                else:
                    # Parse response
                    batch_predictions = openai_client.parse_batch_response(response, len(batch))
                
                # Ensure we have the right number of predictions
                if len(batch_predictions) != len(batch):
                    logger.warning(f"Prediction count mismatch in batch {batch_idx}. Expected {len(batch)}, got {len(batch_predictions)}")
                    batch_predictions = batch_predictions[:len(batch)] + [0] * (len(batch) - len(batch_predictions))
                
                # Store results
                print(f"Batch {batch_idx + 1}/{len(batches)}: {batch_predictions}")
                for i, (sample, pred) in enumerate(zip(batch, batch_predictions)):
                    all_predictions.append(pred)
                    all_labels.append(sample['label'])
                    
                    detailed_predictions.append({
                        'conv_id': sample['conv_id'],
                        'start_idx': sample['start_idx'],
                        'end_idx': sample['end_idx'],
                        'predicted_label': int(pred),
                        'actual_label': int(sample['label']),
                        'original_messages': sample['messages'],
                        'original_labels': sample['original_labels'],
                        'batch_idx': batch_idx,
                        'api_response': response if i == 0 else None,  # Store response only for first item in batch
                    })
                
                # Small delay between batches
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Add default predictions for failed batch
                for sample in batch:
                    all_predictions.append(0)
                    all_labels.append(sample['label'])
                    detailed_predictions.append({
                        'conv_id': sample['conv_id'],
                        'start_idx': sample['start_idx'],
                        'end_idx': sample['end_idx'],
                        'predicted_label': 0,
                        'actual_label': int(sample['label']),
                        'original_messages': sample['messages'],
                        'original_labels': sample['original_labels'],
                        'batch_idx': batch_idx,
                        'error': str(e),
                    })
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_samples': len(all_labels),
        'positive_samples': sum(all_labels),
        'negative_samples': len(all_labels) - sum(all_labels),
    }
    
    # Additional metrics
    if len(set(all_labels)) > 1:
        # Convert predictions to probabilities (0 or 1) for AUC calculation
        pred_probs = [float(p) for p in all_predictions]
        metrics['auc'] = roc_auc_score(all_labels, pred_probs)
        metrics['ap'] = average_precision_score(all_labels, pred_probs)
    else:
        metrics['auc'] = 0.5
        metrics['ap'] = sum(all_labels) / len(all_labels)
    
    # Confusion matrix
    if len(set(all_labels)) > 1:
        cm = confusion_matrix(all_labels, all_predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            if all_labels[0] == 0:
                tn, fp, fn, tp = len(all_labels), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(all_labels)
    else:
        if all_labels[0] == 0:
            tn, fp, fn, tp = len(all_labels), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(all_labels)
    
    metrics['confusion_matrix'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    
    return metrics, detailed_predictions

# ------------------------- Analysis Functions ------------------------- #
def analyze_predictions(detailed_predictions: List[Dict], logger, num_examples: int = 3):
    """Analyze predictions and show examples"""
    
    examples = {
        'true_positives': [],
        'false_positives': [],
        'false_negatives': [],
        'true_negatives': []
    }
    
    for pred in detailed_predictions:
        predicted = pred['predicted_label']
        actual = pred['actual_label']
        
        if predicted == 1 and actual == 1:
            examples['true_positives'].append(pred)
        elif predicted == 1 and actual == 0:
            examples['false_positives'].append(pred)
        elif predicted == 0 and actual == 1:
            examples['false_negatives'].append(pred)
        else:
            examples['true_negatives'].append(pred)
    
    # Log analysis
    logger.info("\n" + "="*80)
    logger.info("OPENAI MODEL PREDICTION ANALYSIS")
    logger.info("="*80)
    
    for category, category_examples in examples.items():
        logger.info(f"\n{category.upper().replace('_', ' ')} ({len(category_examples)} total):")
        for i, ex in enumerate(category_examples[:num_examples]):
            logger.info(f"\nExample {i+1}:")
            logger.info("Original conversation:")
            for j, msg in enumerate(ex['original_messages']):
                speaker = "화자A" if j % 2 == 0 else "화자B"
                logger.info(f"  {speaker}: {msg}")
            logger.info(f"Predicted: {ex['predicted_label']} | Actual: {ex['actual_label']}")
            if 'api_response' in ex and ex['api_response']:
                logger.info(f"API Response snippet: {ex['api_response'][:200]}...")

# ------------------------- Main Function ------------------------- #
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate OpenAI GPT models for drug detection')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
                       help='OpenAI model name to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for API requests')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config(args.model)
    config.BATCH_SIZE = args.batch_size
    
    # Setup logging
    logger = setup_logging(config)
    set_seed(config.RANDOM_SEED)
    
    logger.info(f"Starting OpenAI Drug Detection Evaluation with {args.model}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Output directories: {config.LOG_DIR}, {config.RESULTS_DIR}, {config.PREDICTIONS_DIR}")
    
    try:
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        openai_client = OpenAIClient(config.CONFIG_PATH, args.model)
        logger.info(f"Using model: {args.model}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = ConversationDataset(
            config.INPUT_DIRECTORY,
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE
        )
        
        # Get test split
        test_samples = dataset.get_test_split(args.test_split, config.RANDOM_SEED)
        logger.info(f"Test samples: {len(test_samples)}")
        
        # Evaluate with OpenAI
        logger.info("Starting evaluation with OpenAI API...")
        start_time = time.time()
        
        metrics, detailed_predictions = await evaluate_with_openai(
            openai_client, test_samples, config.BATCH_SIZE, logger, config
        )
        
        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Log results
        logger.info("\n" + "="*80)
        logger.info(f"OPENAI {args.model.upper()} EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1']:.4f}")
        logger.info(f"Test AUC: {metrics['auc']:.4f}")
        logger.info(f"Test Average Precision: {metrics['ap']:.4f}")
        
        cm = metrics['confusion_matrix']
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
        
        logger.info(f"\nEvaluation time: {evaluation_time:.2f} seconds")
        logger.info(f"Average time per sample: {evaluation_time/len(test_samples):.3f} seconds")
        
        # Analyze predictions
        logger.info("\nAnalyzing predictions...")
        analyze_predictions(detailed_predictions, logger)
        
        # Save detailed predictions
        pred_file = os.path.join(
            config.PREDICTIONS_DIR,
            f'detailed_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed predictions saved to: {pred_file}")
        
        # Save results
        results = {
            'model_name': args.model,
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_time_seconds': evaluation_time,
            'test_metrics': metrics,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'window_size': config.WINDOW_SIZE,
                'stride': config.STRIDE,
                'test_split': args.test_split,
                'random_seed': config.RANDOM_SEED,
            },
            'dataset_info': {
                'total_samples': len(dataset.samples),
                'test_samples': len(test_samples),
                'positive_samples': metrics['positive_samples'],
                'negative_samples': metrics['negative_samples'],
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
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MODEL: {args.model}")
        print(f"ACCURACY: {metrics['accuracy']:.4f}")
        print(f"PRECISION: {metrics['precision']:.4f}")
        print(f"RECALL: {metrics['recall']:.4f}")
        print(f"F1-SCORE: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"EVALUATION_TIME: {evaluation_time:.2f}s")
        print(f"RESULTS_DIR: {config.RESULTS_DIR}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())