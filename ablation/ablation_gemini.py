import os
import json
import random
import time
import re
import argparse
import numpy as np
from collections import Counter, deque
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import concurrent.futures
import threading

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

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
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # Data paths
        self.INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/test/modified5'
        self.CONFIG_PATH = '/home/minseok/PoisonedRAG/model_configs/palm2_config.json'
        
        # Model parameters
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = 16  # Process 16 conversations at once
        self.MAX_TOKENS = 2048
        self.TEMPERATURE = 0.1  # Low temperature for consistent responses
        
        # Data splits
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        
        # Sliding window parameters
        self.WINDOW_SIZE = 10
        self.STRIDE = 5
        
        # Rate limiting settings
        self.INITIAL_RPM = 30
        self.MIN_RPM = 10
        self.MAX_RPM = 60
        self.MAX_WORKERS = 1
        
        # Retry settings
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        
        # Output paths
        self.LOG_DIR = f'logs_gemini_{model_name.replace("-", "_").replace(".", "_")}'
        self.RESULTS_DIR = f'results_gemini_{model_name.replace("-", "_").replace(".", "_")}'
        self.PREDICTIONS_DIR = f'predictions_gemini_{model_name.replace("-", "_").replace(".", "_")}'
        
        self.RANDOM_SEED = 42

# ------------------------- Utility Functions ------------------------- #
def load_json(path: str) -> dict:
    """Load JSON configuration file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}")

def save_json(data, path):
    """Save data to JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_json_text(raw: str) -> str:
    """Strip triple-backtick fences or any leading/trailing noise"""
    txt = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", txt, flags=re.DOTALL)
    return m.group(1).strip() if m else txt

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

# ------------------------- Rate Limiting ------------------------- #
class AdaptiveRateLimiter:
    """Manages API request timing with adaptive backoff based on success/failure"""
    
    def __init__(self, 
                 initial_requests_per_minute=30, 
                 min_requests_per_minute=10,
                 max_requests_per_minute=60,
                 window_size=10):
        self.current_rpm = initial_requests_per_minute
        self.min_rpm = min_requests_per_minute
        self.max_rpm = max_requests_per_minute
        self.window_size = window_size
        self.recent_results = deque(maxlen=window_size)
        self.last_request_time = 0
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Wait the appropriate amount of time before the next request"""
        with self.lock:
            current_rpm = self.current_rpm
            wait_time = 60.0 / current_rpm
            
            # Add small jitter (±10%) to avoid all threads hitting exactly at once
            jitter = random.uniform(-0.1, 0.1) * wait_time
            wait_time += jitter
            
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < wait_time:
                sleep_time = wait_time - time_since_last
                time.sleep(sleep_time)
                
            self.last_request_time = time.time()
        
    def report_success(self):
        """Report a successful API call"""
        with self.lock:
            self.recent_results.append(True)
            self._adjust_rate()
        
    def report_failure(self, is_rate_limit=False):
        """Report a failed API call, with flag if it was due to rate limiting"""
        with self.lock:
            self.recent_results.append(False)
            
            # Immediate backoff for rate limit errors
            if is_rate_limit:
                self.current_rpm = max(self.min_rpm, self.current_rpm * 0.7)  # Reduce by 30%
            
            self._adjust_rate()
        
    def _adjust_rate(self):
        """Adjust the rate based on recent success/failure history"""
        if len(self.recent_results) < self.window_size:
            return  # Not enough data yet
                
        success_rate = sum(1 for r in self.recent_results if r) / len(self.recent_results)
        
        if success_rate > 0.95:  # >95% success
            # Gradually increase rate
            self.current_rpm = min(self.max_rpm, self.current_rpm * 1.05)
        elif success_rate < 0.8:  # <80% success
            # More aggressively decrease rate
            self.current_rpm = max(self.min_rpm, self.current_rpm * 0.9)

# ------------------------- Gemini API Client ------------------------- #
class GeminiClient:
    def __init__(self, config_path: str, model_name: str = "gemini-2.0-flash", config: Config = None):
        self.model_name = model_name
        self.config_data = load_json(config_path)
        self.config = config
        
        # Extract API key
        api_keys = self.config_data['api_key_info']['api_keys']
        api_key = ''.join(api_keys) if isinstance(api_keys, list) else api_keys
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            initial_requests_per_minute=config.INITIAL_RPM,
            min_requests_per_minute=config.MIN_RPM,
            max_requests_per_minute=config.MAX_RPM
        ) if config else AdaptiveRateLimiter()
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = genai.GenerationConfig(
            temperature=config.TEMPERATURE if config else 0.1,
            max_output_tokens=config.MAX_TOKENS if config else 2048,
            candidate_count=1,
        )
        
        # Safety settings
        self.safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
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
        
        return batch_prompt
    
    def call_gemini(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make API call to Gemini with retries"""
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Make API call
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                self.rate_limiter.report_success()
                return response.text.strip()
                
            except ResourceExhausted as e:
                self.rate_limiter.report_failure(is_rate_limit=True)
                if attempt == max_retries - 1:
                    logging.error(f"Rate limit exceeded after {max_retries} attempts: {e}")
                    return None
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                logging.warning(f"Rate limit hit, waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
            except ServiceUnavailable as e:
                self.rate_limiter.report_failure(is_rate_limit=False)
                if attempt == max_retries - 1:
                    logging.error(f"Service unavailable after {max_retries} attempts: {e}")
                    return None
                wait_time = 2 ** attempt
                logging.warning(f"Service unavailable, waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
            except Exception as e:
                self.rate_limiter.report_failure(is_rate_limit=False)
                logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None
    
    def parse_batch_response(self, response: str, num_windows: int) -> List[int]:
        """Parse batch response and extract predictions"""
        predictions = []
        
        if not response:
            return [0] * num_windows
        
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
def evaluate_with_gemini(gemini_client: GeminiClient, test_samples: List[Dict], 
                         batch_size: int, logger, config) -> Tuple[Dict, List[Dict]]:
    """Evaluate samples using Gemini API with batching"""
    
    all_predictions = []
    all_labels = []
    detailed_predictions = []
    
    # Create batches
    batches = [test_samples[i:i + batch_size] for i in range(0, len(test_samples), batch_size)]
    
    logger.info(f"Processing {len(test_samples)} samples in {len(batches)} batches of size {batch_size}")
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        try:
            # Create batch prompt
            prompt = gemini_client.create_batch_prompt(batch)
            
            # Make API request
            response = gemini_client.call_gemini(prompt, config.MAX_RETRIES)
            print(response)
            if response is None:
                logger.error(f"Failed to get response for batch {batch_idx}")
                # Use default predictions (all negative)
                batch_predictions = [0] * len(batch)
            else:
                # Parse response
                batch_predictions = gemini_client.parse_batch_response(response, len(batch))
            
            # Ensure we have the right number of predictions
            if len(batch_predictions) != len(batch):
                logger.warning(f"Prediction count mismatch in batch {batch_idx}. Expected {len(batch)}, got {len(batch_predictions)}")
                batch_predictions = batch_predictions[:len(batch)] + [0] * (len(batch) - len(batch_predictions))
            
            # Store results
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
            time.sleep(0.5)
            
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
    logger.info("GEMINI MODEL PREDICTION ANALYSIS")
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
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Gemini models for drug detection')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro',
                       choices=['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'],
                       help='Gemini model name to evaluate')
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
    
    logger.info(f"Starting Gemini Drug Detection Evaluation with {args.model}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Output directories: {config.LOG_DIR}, {config.RESULTS_DIR}, {config.PREDICTIONS_DIR}")
    
    try:
        # Initialize Gemini client
        logger.info("Initializing Gemini client...")
        gemini_client = GeminiClient(config.CONFIG_PATH, args.model, config)
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
        
        # Evaluate with Gemini
        logger.info("Starting evaluation with Gemini API...")
        start_time = time.time()
        
        metrics, detailed_predictions = evaluate_with_gemini(
            gemini_client, test_samples, config.BATCH_SIZE, logger, config
        )
        
        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Log results
        logger.info("\n" + "="*80)
        logger.info(f"GEMINI {args.model.upper()} EVALUATION RESULTS")
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
        save_json(detailed_predictions, pred_file)
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
                'temperature': config.TEMPERATURE,
                'max_tokens': config.MAX_TOKENS,
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
        
        save_json(results, results_path)
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
    main()