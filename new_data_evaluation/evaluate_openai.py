"""
Evaluate OpenAI GPT models on new_data.
Adapted from ablation/ablation_openai.py for the new dataset.
"""

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
        self.INPUT_FILE = '/home/minseok/forensic/new_data_evaluation/preprocessed/all_conversations.json'
        self.CONFIG_PATH = '/home/minseok/PoisonedRAG/model_configs/gpt4_config.json'

        # Model parameters
        self.MODEL_NAME = model_name
        self.BATCH_SIZE = 16
        self.MAX_TOKENS = 1000
        self.TEMPERATURE = 0.1

        # Sliding window parameters
        self.WINDOW_SIZE = 10
        self.STRIDE = 5

        # API parameters
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        self.REQUEST_DELAY = 0.1

        # Output paths
        self.LOG_DIR = f'/home/minseok/forensic/new_data_evaluation/logs_openai_{model_name.replace("-", "_")}'
        self.RESULTS_DIR = f'/home/minseok/forensic/new_data_evaluation/results_openai_{model_name.replace("-", "_")}'

        self.RANDOM_SEED = 42

# ------------------------- Utility Functions ------------------------- #
def load_json(file_path: str) -> dict:
    """Load JSON configuration file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_logging(config):
    """Setup logging configuration"""
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
        self.min_request_interval = 0.1

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
                        predictions.append(0)
                    found = True
                    break

            if not found:
                predictions.append(0)

        return predictions

# ------------------------- Dataset Class ------------------------- #
class ConversationDataset:
    def __init__(self, data_file: str, window_size: int = 10, stride: int = 5, messengers: List[str] = None):
        self.window_size = window_size
        self.stride = stride
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

# ------------------------- Evaluation Function ------------------------- #
async def evaluate_with_openai(openai_client: OpenAIClient, samples: List[Dict],
                              batch_size: int, logger, config) -> Tuple[Dict, List[Dict]]:
    """Evaluate samples using OpenAI API with batching"""

    all_predictions = []
    all_labels = []
    detailed_predictions = []

    # Create batches
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

    logger.info(f"Processing {len(samples)} samples in {len(batches)} batches of size {batch_size}")

    async with aiohttp.ClientSession() as session:
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            try:
                # Create batch prompt
                prompt = openai_client.create_batch_prompt(batch)

                # Make API request
                response = await openai_client.make_api_request(session, prompt)

                if response is None:
                    logger.error(f"Failed to get response for batch {batch_idx}")
                    batch_predictions = [0] * len(batch)
                else:
                    batch_predictions = openai_client.parse_batch_response(response, len(batch))

                # Ensure correct number of predictions
                if len(batch_predictions) != len(batch):
                    logger.warning(f"Prediction count mismatch in batch {batch_idx}")
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
                        'api_response': response if i == 0 else None,
                    })

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
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
        'predicted_positive': sum(all_predictions),
        'predicted_negative': len(all_predictions) - sum(all_predictions)
    }

    # Additional metrics
    if len(set(all_labels)) > 1:
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

    metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

    return metrics, detailed_predictions

# ------------------------- Main Function ------------------------- #
async def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenAI GPT models on new dataset')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
                       help='OpenAI model name to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for API requests')
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
    config.BATCH_SIZE = args.batch_size

    # Update output paths to include messenger suffix if filtering
    if messengers:
        config.LOG_DIR = f'{config.LOG_DIR}_{messenger_suffix}'
        config.RESULTS_DIR = f'{config.RESULTS_DIR}_{messenger_suffix}'

    # Setup logging
    logger = setup_logging(config)
    set_seed(config.RANDOM_SEED)

    messenger_info = ', '.join(messengers) if messengers else 'all messengers'
    logger.info(f"Starting OpenAI evaluation with {args.model} on new dataset")
    logger.info(f"Messengers: {messenger_info}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Output directories: {config.LOG_DIR}, {config.RESULTS_DIR}")

    try:
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        openai_client = OpenAIClient(config.CONFIG_PATH, args.model)
        logger.info(f"Using model: {args.model}")

        # Load dataset
        logger.info("Loading dataset...")
        dataset = ConversationDataset(
            config.INPUT_FILE,
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE,
            messengers=messengers
        )

        logger.info(f"Total samples: {len(dataset.samples)}")

        # Evaluate with OpenAI
        logger.info("Starting evaluation with OpenAI API...")
        start_time = time.time()

        metrics, detailed_predictions = await evaluate_with_openai(
            openai_client, dataset.samples, config.BATCH_SIZE, logger, config
        )

        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")

        # Log results
        logger.info("\n" + "="*80)
        logger.info(f"OPENAI {args.model.upper()} EVALUATION RESULTS ON NEW DATA")
        logger.info("="*80)
        logger.info(f"Total Samples: {metrics['total_samples']}")
        logger.info(f"Actual Positive: {metrics['positive_samples']}")
        logger.info(f"Predicted Positive: {metrics['predicted_positive']}")
        logger.info(f"\nAccuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")

        cm = metrics['confusion_matrix']
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm['tn']:6d}  FP: {cm['fp']:6d}")
        logger.info(f"  FN: {cm['fn']:6d}  TP: {cm['tp']:6d}")

        specificity = None
        if cm['tn'] + cm['fp'] > 0:
            specificity = cm['tn'] / (cm['tn'] + cm['fp'])
            logger.info(f"\nSpecificity: {specificity:.4f}")
            logger.info(f"False Positive Rate: {1 - specificity:.4f}")

        # Save results
        results = {
            'model_name': args.model,
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_time_seconds': evaluation_time,
            'metrics': metrics,
            'config': {
                'batch_size': config.BATCH_SIZE,
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

        logger.info(f"\nResults saved to: {results_path}")

        predictions_path = os.path.join(
            config.RESULTS_DIR,
            f'detailed_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed predictions saved to: {predictions_path}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"MODEL: {args.model}")
        print(f"ACCURACY: {metrics['accuracy']:.4f}")
        print(f"F1-SCORE: {metrics['f1']:.4f}")
        if specificity is not None:
            print(f"SPECIFICITY: {specificity:.4f}")
        print(f"EVALUATION_TIME: {evaluation_time:.2f}s")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    asyncio.run(main())
