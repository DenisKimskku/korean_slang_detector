import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional, Set
import unicodedata

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
    BertModel,
    BertTokenizer
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
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Configuration Parameters ------------------------- #
class Config:
    # Data paths
    INPUT_DIRECTORY = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
    POISON_CSV_PATH = '/home/minseok/forensic/poison.csv'
    
    # Model parameters
    MODEL_NAME = "klue/bert-base"  # Korean BERT
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2
    EPOCHS = 15
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Data splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Context parameters
    MAX_CONTEXT_LENGTH = 512
    CONTEXT_WINDOW = 5  # Increased for better context
    
    # Feature parameters
    USE_KEYWORD_FEATURES = True
    USE_PATTERN_FEATURES = True
    USE_LINGUISTIC_FEATURES = True
    KEYWORD_WEIGHT = 2.0  # Weight for keyword matching features
    
    # Training parameters
    EARLY_STOPPING_PATIENCE = 4
    GRADIENT_CLIP_VAL = 1.0
    
    # Loss parameters
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Augmentation parameters
    USE_AUGMENTATION = True
    AUGMENT_POSITIVE_RATIO = 3
    
    # Output paths
    LOG_DIR = 'logs_generalized'
    MODEL_DIR = 'models_generalized'
    CHECKPOINT_DIR = 'checkpoints_generalized'
    
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

# ------------------------- Drug Keyword Manager ------------------------- #
class DrugKeywordManager:
    """Manages drug-related keywords and their variations"""
    
    def __init__(self, csv_path: str):
        self.keywords = self._load_keywords(csv_path)
        self.all_variations = self._build_variation_set()
        self.patterns = self._build_patterns()
        logger.info(f"Loaded {len(self.keywords)} drug keywords with {len(self.all_variations)} total variations")
    
    def _load_keywords(self, csv_path: str) -> Dict[str, List[str]]:
        """Load keywords from CSV"""
        df = pd.read_csv(csv_path, encoding='utf-8')
        keywords = {}
        
        for _, row in df.iterrows():
            original = row['원본단어']
            variations = [row[col] for col in df.columns[1:] if pd.notna(row[col])]
            keywords[original] = variations
        
        return keywords
    
    def _build_variation_set(self) -> Set[str]:
        """Build a set of all variations for fast lookup"""
        all_vars = set()
        for original, variations in self.keywords.items():
            all_vars.add(original.lower())
            all_vars.update(v.lower() for v in variations)
        return all_vars
    
    def _build_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for flexible matching"""
        patterns = []
        
        # Pattern for alphanumeric mixing (e.g., 'ᄋais', 'a이ᄋs')
        patterns.append(re.compile(r'[a-zA-Z]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+|[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+[a-zA-Z]+'))
        
        # Pattern for repeated characters (e.g., 'ppong', 'ttong')
        patterns.append(re.compile(r'(.)\1{2,}'))
        
        # Pattern for spaced variations (e.g., '공 포 의')
        patterns.append(re.compile(r'(\S)\s+(\S)'))
        
        return patterns
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract keyword-based features from text"""
        text_lower = text.lower()
        features = {
            'exact_match_count': 0,
            'partial_match_count': 0,
            'pattern_match_count': 0,
            'keyword_density': 0,
            'max_keyword_length': 0,
            'has_english_korean_mix': 0,
            'has_repeated_chars': 0,
            'suspicious_spacing': 0
        }
        
        # Check exact matches
        matches = []
        for var in self.all_variations:
            if var in text_lower:
                features['exact_match_count'] += 1
                matches.append(var)
                features['max_keyword_length'] = max(features['max_keyword_length'], len(var))
        
        # Check partial matches with edit distance
        words = text_lower.split()
        for word in words:
            for var in self.all_variations:
                if self._is_similar(word, var, threshold=0.8):
                    features['partial_match_count'] += 1
        
        # Pattern-based features
        if self.patterns[0].search(text):  # English-Korean mix
            features['has_english_korean_mix'] = 1
            features['pattern_match_count'] += 1
        
        if self.patterns[1].search(text):  # Repeated characters
            features['has_repeated_chars'] = 1
            features['pattern_match_count'] += 1
        
        if self.patterns[2].search(text):  # Suspicious spacing
            features['suspicious_spacing'] = 1
            features['pattern_match_count'] += 1
        
        # Keyword density
        if len(words) > 0:
            features['keyword_density'] = features['exact_match_count'] / len(words)
        
        return features
    
    def _is_similar(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """Check if two words are similar using edit distance ratio"""
        if len(word1) < 2 or len(word2) < 2:
            return False
        
        # Simple Levenshtein distance ratio
        distance = self._levenshtein_distance(word1, word2)
        max_len = max(len(word1), len(word2))
        ratio = 1 - (distance / max_len)
        
        return ratio >= threshold
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

# ------------------------- Linguistic Feature Extractor ------------------------- #
class LinguisticFeatureExtractor:
    """Extract linguistic features that might indicate drug-related conversations"""
    
    def __init__(self):
        # Suspicious patterns in drug conversations
        self.code_patterns = [
            r'\b\d+\s*(개|알|정|병|봉지|그램|g|gram)\b',  # Quantity patterns
            r'\b(가격|얼마|원|만원|천원)\b',  # Price-related
            r'\b(어디|장소|만나|거래|교환)\b',  # Location/meeting
            r'\b(조심|비밀|몰래|쉿|ㅂㅁ)\b',  # Secrecy
            r'\b(효과|기분|느낌|약발)\b',  # Effects
            r'[0-9]{2,}',  # Numbers (prices, quantities)
        ]
        
        self.compiled_patterns = [re.compile(p) for p in self.code_patterns]
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'unique_word_ratio': 0,
            'numeric_ratio': 0,
            'special_char_ratio': 0,
            'english_ratio': 0,
            'pattern_matches': 0,
            'has_quantity_pattern': 0,
            'has_price_pattern': 0,
            'has_location_pattern': 0,
            'has_secrecy_pattern': 0,
            'short_message_ratio': 0
        }
        
        # Basic statistics
        words = text.split()
        if words:
            features['unique_word_ratio'] = len(set(words)) / len(words)
        
        # Character type ratios
        if len(text) > 0:
            features['numeric_ratio'] = sum(c.isdigit() for c in text) / len(text)
            features['special_char_ratio'] = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
            features['english_ratio'] = sum(c.isalpha() and ord(c) < 128 for c in text) / len(text)
        
        # Pattern matching
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                features['pattern_matches'] += 1
                if i == 0:
                    features['has_quantity_pattern'] = 1
                elif i == 1:
                    features['has_price_pattern'] = 1
                elif i == 2:
                    features['has_location_pattern'] = 1
                elif i == 3:
                    features['has_secrecy_pattern'] = 1
        
        # Short message ratio (drug conversations often have short, coded messages)
        if features['word_count'] > 0:
            features['short_message_ratio'] = sum(1 for w in words if len(w) <= 3) / features['word_count']
        
        return features

# ------------------------- Enhanced Dataset Class ------------------------- #
class DrugConversationDataset(Dataset):
    def __init__(
        self, 
        input_dir: str,
        tokenizer,
        keyword_manager: DrugKeywordManager,
        linguistic_extractor: LinguisticFeatureExtractor,
        context_window: int = 5,
        max_length: int = 512,
        augment_positive: bool = False,
        is_training: bool = True
    ):
        self.tokenizer = tokenizer
        self.keyword_manager = keyword_manager
        self.linguistic_extractor = linguistic_extractor
        self.context_window = context_window
        self.max_length = max_length
        self.augment_positive = augment_positive
        self.is_training = is_training
        
        self.samples = []
        self.load_data(input_dir)
        
        # Analyze and log statistics
        self._analyze_dataset()
    
    def load_data(self, input_dir: str):
        """Load and preprocess data with context"""
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        for file in tqdm(json_files, desc="Loading data files"):
            file_path = os.path.join(input_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    continue
                
                for conversation in data:
                    utterances = conversation.get('utterance', [])
                    self._process_conversation(utterances, conversation.get('id', 'unknown'))
                    
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
    
    def _process_conversation(self, utterances: List[Dict], conv_id: str):
        """Process a single conversation with context"""
        for i, utt in enumerate(utterances):
            label = utt.get('label', 0)
            
            # Get context
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(utterances), i + self.context_window + 1)
            
            # Build context with position information
            context_messages = []
            relative_positions = []
            
            for j in range(start_idx, end_idx):
                msg = utterances[j]['original_form']
                rel_pos = j - i
                
                if j == i:
                    msg = f"[CUR] {msg} [/CUR]"
                
                context_messages.append(msg)
                relative_positions.append(rel_pos)
            
            context_text = " [SEP] ".join(context_messages)
            
            # Extract features
            target_text = utterances[i]['original_form']
            keyword_features = self.keyword_manager.extract_features(target_text)
            linguistic_features = self.linguistic_extractor.extract_features(target_text)
            
            # Also extract context features
            full_context = " ".join([utterances[j]['original_form'] for j in range(start_idx, end_idx)])
            context_keyword_features = self.keyword_manager.extract_features(full_context)
            context_linguistic_features = self.linguistic_extractor.extract_features(full_context)
            
            # Combine all features
            all_features = {
                **{f'target_{k}': v for k, v in keyword_features.items()},
                **{f'target_{k}': v for k, v in linguistic_features.items()},
                **{f'context_{k}': v for k, v in context_keyword_features.items()},
                **{f'context_{k}': v for k, v in context_linguistic_features.items()},
                'context_window_size': end_idx - start_idx
            }
            
            sample = {
                'text': context_text,
                'label': label,
                'features': all_features,
                'conv_id': conv_id,
                'position': i
            }
            
            self.samples.append(sample)
            
            # Augment positive samples
            if label == 1 and self.augment_positive and self.is_training:
                for _ in range(Config.AUGMENT_POSITIVE_RATIO):
                    aug_sample = self._augment_sample(sample, utterances, i, start_idx, end_idx)
                    self.samples.append(aug_sample)
    
    def _augment_sample(self, original_sample: Dict, utterances: List[Dict], 
                       target_idx: int, start_idx: int, end_idx: int) -> Dict:
        """Augment a positive sample"""
        # Simple augmentation strategies
        aug_messages = []
        
        for j in range(start_idx, end_idx):
            msg = utterances[j]['original_form']
            
            if j == target_idx:
                # Apply augmentation to target message
                if random.random() < 0.5:
                    msg = self._add_typos(msg)
                if random.random() < 0.3:
                    msg = self._add_spaces(msg)
                msg = f"[CUR] {msg} [/CUR]"
            
            aug_messages.append(msg)
        
        aug_text = " [SEP] ".join(aug_messages)
        
        # Recalculate features for augmented text
        aug_target = self._add_typos(utterances[target_idx]['original_form'])
        keyword_features = self.keyword_manager.extract_features(aug_target)
        linguistic_features = self.linguistic_extractor.extract_features(aug_target)
        
        aug_sample = original_sample.copy()
        aug_sample['text'] = aug_text
        aug_sample['features'] = {
            **{f'target_{k}': v for k, v in keyword_features.items()},
            **{f'target_{k}': v for k, v in linguistic_features.items()},
            **{f'context_{k}': v for k, v in original_sample['features'].items() if k.startswith('context_')},
            'context_window_size': original_sample['features']['context_window_size']
        }
        
        return aug_sample
    
    def _add_typos(self, text: str, prob: float = 0.1) -> str:
        """Add random typos"""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < prob:
                if random.random() < 0.5 and i > 0:
                    # Swap with previous
                    chars[i], chars[i-1] = chars[i-1], chars[i]
                else:
                    # Duplicate
                    chars[i] = chars[i] * 2
        return ''.join(chars)
    
    def _add_spaces(self, text: str, prob: float = 0.1) -> str:
        """Add random spaces"""
        chars = list(text)
        result = []
        for char in chars:
            result.append(char)
            if random.random() < prob:
                result.append(' ')
        return ''.join(result)
    
    def _analyze_dataset(self):
        """Analyze dataset statistics"""
        labels = [s['label'] for s in self.samples]
        label_counts = Counter(labels)
        
        # Feature statistics for positive samples
        positive_features = [s['features'] for s in self.samples if s['label'] == 1]
        
        if positive_features:
            avg_keyword_matches = np.mean([f['target_exact_match_count'] for f in positive_features])
            avg_pattern_matches = np.mean([f['target_pattern_matches'] for f in positive_features])
            
            logger.info(f"Dataset Statistics:")
            logger.info(f"  Total samples: {len(self.samples)}")
            logger.info(f"  Class distribution: {dict(label_counts)}")
            logger.info(f"  Positive ratio: {label_counts[1] / len(self.samples):.2%}")
            logger.info(f"  Avg keyword matches in positive samples: {avg_keyword_matches:.2f}")
            logger.info(f"  Avg pattern matches in positive samples: {avg_pattern_matches:.2f}")
    
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
        
        # Convert features to tensor
        feature_values = list(sample['features'].values())
        features_tensor = torch.tensor(feature_values, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'features': features_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

# ------------------------- Model Architecture ------------------------- #
class GeneralizableDrugDetectionModel(nn.Module):
    """Model that combines BERT with hand-crafted features for better generalization"""
    
    def __init__(self, model_name: str, num_features: int, num_labels: int = 2, 
                 dropout_rate: float = 0.3, feature_hidden_dim: int = 128):
        super().__init__()
        
        # BERT for text understanding
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Feature processing network
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, feature_hidden_dim),
            nn.BatchNorm1d(feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_hidden_dim, feature_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism for BERT outputs
        self.attention = nn.MultiheadAttention(
            bert_hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Fusion layer
        fusion_input_dim = bert_hidden_size * 2 + feature_hidden_dim // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, bert_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bert_hidden_size // 2, num_labels)
        )
        
        # Gate mechanism for feature importance
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize non-pretrained weights"""
        for module in [self.feature_processor, self.fusion_layer, self.classifier, self.feature_gate]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, features):
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            sequence_output,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pool BERT representations
        cls_output = attn_output[:, 0]  # CLS token
        mean_output = (attn_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        
        # Process hand-crafted features
        processed_features = self.feature_processor(features)
        
        # Apply feature gating
        feature_gate = self.feature_gate(processed_features)
        gated_features = processed_features * feature_gate
        
        # Combine all representations
        combined = torch.cat([cls_output, mean_output, gated_features], dim=-1)
        
        # Fusion and classification
        fused = self.fusion_layer(combined)
        logits = self.classifier(fused)
        
        return logits, attn_weights, feature_gate

# ------------------------- Focal Loss ------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------- Training and Evaluation ------------------------- #
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 criterion, config, device):
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
        
        # For tracking feature importance
        self.feature_importance_tracker = defaultdict(list)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch['features'].to(self.device)
            batch_labels = batch['label'].to(self.device)
            
            # Forward pass
            logits, _, feature_gates = self.model(input_ids, attention_mask, features)
            loss = self.criterion(logits, batch_labels)
            
            # Track feature importance
            self.feature_importance_tracker['gates'].append(feature_gates.mean().item())
            
            # Gradient accumulation
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
            
            # Metrics
            total_loss += loss.item() * self.config.ACCUMULATION_STEPS
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.ACCUMULATION_STEPS:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        probabilities = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                logits, attn_weights, _ = self.model(input_ids, attention_mask, features)
                loss = self.criterion(logits, batch_labels)
                
                total_loss += loss.item()
                probs = F.softmax(logits, dim=-1)
                probabilities.extend(probs[:, 1].cpu().numpy())
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                
                # Store attention weights for analysis
                if len(attention_weights_list) < 10:  # Store first 10 for analysis
                    attention_weights_list.append(attn_weights.cpu())
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels, probabilities)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics, attention_weights_list
    
    def save_checkpoint(self, metrics, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'feature_importance': dict(self.feature_importance_tracker)
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
def analyze_predictions(model, test_loader, keyword_manager, device, num_examples=10):
    """Analyze model predictions to understand what it learned"""
    model.eval()
    
    false_positives = []
    false_negatives = []
    true_positives = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label']
            
            logits, attn_weights, feature_gates = model(input_ids, attention_mask, features)
            preds = torch.argmax(logits, dim=-1).cpu()
            
            # Collect examples
            for i in range(len(preds)):
                if preds[i] == 1 and labels[i] == 0:  # False positive
                    false_positives.append({
                        'text': tokenizer.decode(input_ids[i], skip_special_tokens=True),
                        'features': features[i].cpu(),
                        'attention': attn_weights[i].cpu(),
                        'feature_gate': feature_gates[i].cpu()
                    })
                elif preds[i] == 0 and labels[i] == 1:  # False negative
                    false_negatives.append({
                        'text': tokenizer.decode(input_ids[i], skip_special_tokens=True),
                        'features': features[i].cpu(),
                        'attention': attn_weights[i].cpu(),
                        'feature_gate': feature_gates[i].cpu()
                    })
                elif preds[i] == 1 and labels[i] == 1:  # True positive
                    true_positives.append({
                        'text': tokenizer.decode(input_ids[i], skip_special_tokens=True),
                        'features': features[i].cpu(),
                        'attention': attn_weights[i].cpu(),
                        'feature_gate': feature_gates[i].cpu()
                    })
    
    # Log analysis
    logger.info("\n" + "="*80)
    logger.info("PREDICTION ANALYSIS")
    logger.info("="*80)
    
    # True Positives
    logger.info(f"\nTrue Positives (correctly identified drug conversations): {len(true_positives)}")
    for i, example in enumerate(true_positives[:num_examples]):
        logger.info(f"\nTP Example {i+1}:")
        logger.info(f"Text: {example['text'][:200]}...")
        logger.info(f"Feature gate value: {example['feature_gate'].item():.3f}")
    
    # False Positives
    logger.info(f"\nFalse Positives (incorrectly flagged as drug-related): {len(false_positives)}")
    for i, example in enumerate(false_positives[:num_examples]):
        logger.info(f"\nFP Example {i+1}:")
        logger.info(f"Text: {example['text'][:200]}...")
        logger.info(f"Feature gate value: {example['feature_gate'].item():.3f}")
    
    # False Negatives
    logger.info(f"\nFalse Negatives (missed drug conversations): {len(false_negatives)}")
    for i, example in enumerate(false_negatives[:num_examples]):
        logger.info(f"\nFN Example {i+1}:")
        logger.info(f"Text: {example['text'][:200]}...")
        logger.info(f"Feature gate value: {example['feature_gate'].item():.3f}")

# ------------------------- Main Pipeline ------------------------- #
def main():
    logger.info("Starting Generalizable Drug Conversation Detection Pipeline")
    logger.info(f"Device: {Config.DEVICE}")
    
    # Initialize components
    keyword_manager = DrugKeywordManager(Config.POISON_CSV_PATH)
    linguistic_extractor = LinguisticFeatureExtractor()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['[CUR]', '[/CUR]', '[SEP]']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = DrugConversationDataset(
        Config.INPUT_DIRECTORY,
        tokenizer,
        keyword_manager,
        linguistic_extractor,
        context_window=Config.CONTEXT_WINDOW,
        max_length=Config.MAX_CONTEXT_LENGTH,
        augment_positive=Config.USE_AUGMENTATION,
        is_training=True
    )
    
    # Get number of features from first sample
    num_features = len(full_dataset[0]['features'])
    logger.info(f"Number of features: {num_features}")
    
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
    model = GeneralizableDrugDetectionModel(
        Config.MODEL_NAME,
        num_features=num_features
    )
    model.to(Config.DEVICE)
    
    # Resize token embeddings
    model.bert.resize_token_embeddings(len(tokenizer))
    
    # Calculate class weights
    train_labels = [full_dataset.samples[i]['label'] for i in train_dataset.indices]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)
    logger.info(f"Class weights: {class_weights}")
    
    # Initialize loss
    if Config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=Config.FOCAL_ALPHA,
            gamma=Config.FOCAL_GAMMA
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
    test_metrics, _ = trainer.evaluate(test_loader)
    
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
    logger.info(f"Confusion Matrix:")
    logger.info(f"  TN: {tn:6d}  FP: {fp:6d}")
    logger.info(f"  FN: {fn:6d}  TP: {tp:6d}")
    
    # Analyze predictions
    logger.info("\nAnalyzing model predictions...")
    analyze_predictions(model, test_loader, keyword_manager, Config.DEVICE)
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_f1': best_f1,
        'config': vars(Config),
        'feature_importance': dict(trainer.feature_importance_tracker)
    }
    
    results_path = os.path.join(
        Config.LOG_DIR,
        f'final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
    
    def _calculate_metrics(self, predictions, labels, probabilities=None):
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Add AUC and AP if we have probabilities
        if probabilities is not None and len(set(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, probabilities)
            metrics['ap'] = average_precision_score(labels, probabilities)
        
        # Confusion matrix
        if len(set(labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        else:
            # Handle single class case
            tn = fp = fn = tp = 0
            if labels[0] == 0:
                tn = len(labels)
            else:
                tp = len(labels)
        
        metrics['confusion_matrix'] = (tn, fp, fn, tp)
        
        return metrics