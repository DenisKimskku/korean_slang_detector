"""
Robustness Evaluation for Drug Slang Detection Models
Compares naive (base) vs fine-tuned classification models

Tests:
1. Out-of-Distribution (OOD) Detection - confidence on clean vs noisy data
2. Adversarial Robustness - resilience to perturbations
3. Embedding Quality - representation learning
4. Calibration - probability output quality
5. Prediction Stability - consistency under noise
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# ========================== Configuration ==========================
DATASET_PATH = '/home/minseok/rag_security/KorQuAD_v1.0_dev_cleansed.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 384
RANDOM_SEED = 42

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Model configurations
MODEL_CONFIGS = {
    'bert_base': {
        'name': 'klue/bert-base',
        'checkpoint': '/home/minseok/forensic/bert_base/models_pure_lm_attn/best_model.pt'
    },
    'distillbert_base': {
        'name': 'monologg/distilkobert',
        'checkpoint': '/home/minseok/forensic/distillbert_base/models_distilbert/best_model.pt'
    },
    'electra_base': {
        'name': 'monologg/koelectra-base-v3-discriminator',
        'checkpoint': '/home/minseok/forensic/electra_base/models_electra/best_model.pt'
    },
    'roberta_base': {
        'name': 'klue/roberta-base',
        'checkpoint': '/home/minseok/forensic/roberta_base/models_pure_lm_attn/best_model.pt'
    },
    'roberta_large': {
        'name': 'klue/roberta-large',
        'checkpoint': '/home/minseok/forensic/roberta_large/models_pure_lm_attn/best_model.pt'
    }
}

# ========================== Model Architecture ==========================

class ClassificationModel(nn.Module):
    """Classification model matching the training architecture"""
    def __init__(self, base_model, hidden_size=512):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size

        # Match the training architecture from train_roberta_large_v1.py
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        return logits.squeeze(dim=1), cls_output  # Return logits and embeddings

# ========================== Data Loading ==========================

def load_korquad_data(file_path, max_samples=None):
    """Load KorQuAD dataset - clean Korean text (OOD for drug slang models)"""
    print(f"Loading KorQuAD dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    for item in data['data']:
        for qa in item['qas']:
            # Use questions and contexts as clean text samples
            texts.append(qa['question'])
            texts.append(qa['context'])
            if qa['answers']:
                texts.append(qa['answers'][0])

    # Remove duplicates and filter empty
    texts = list(set([t.strip() for t in texts if t.strip()]))

    if max_samples:
        texts = texts[:max_samples]

    print(f"Loaded {len(texts)} unique text samples")
    return texts

# ========================== Perturbation Functions ==========================

def char_swap_perturbation(text, prob=0.1):
    """Swap adjacent characters randomly"""
    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < prob:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)

def char_delete_perturbation(text, prob=0.1):
    """Delete characters randomly"""
    chars = [c for c in text if random.random() > prob]
    return ''.join(chars) if chars else text

def char_insert_perturbation(text, prob=0.1):
    """Insert random characters"""
    korean_chars = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'
    chars = list(text)
    result = []
    for c in chars:
        result.append(c)
        if random.random() < prob:
            result.append(random.choice(korean_chars))
    return ''.join(result)

def word_shuffle_perturbation(text, prob=0.3):
    """Shuffle words randomly"""
    words = text.split()
    if len(words) <= 1:
        return text

    indices = list(range(len(words)))
    if random.random() < prob:
        random.shuffle(indices)

    return ' '.join([words[i] for i in indices])

# ========================== Robustness Metrics ==========================

def calculate_prediction_confidence(model, tokenizer, texts, device, desc="Calculating"):
    """Calculate prediction confidence statistics"""
    model.eval()
    confidences = []
    predictions = []
    entropies = []

    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)

            # Get predictions
            logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
            prob = torch.sigmoid(logits).item()

            # Calculate metrics
            confidences.append(prob)
            predictions.append(1 if prob > 0.5 else 0)

            # Calculate entropy (uncertainty)
            p = max(min(prob, 0.9999), 0.0001)  # Avoid log(0)
            entropy = -p * np.log(p) - (1-p) * np.log(1-p)
            entropies.append(entropy)

    return {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'mean_entropy': np.mean(entropies),
        'positive_ratio': np.mean(predictions),
        'confidence_distribution': confidences
    }

def calculate_adversarial_robustness(model, tokenizer, texts, device, num_perturbations=5):
    """Test robustness against adversarial perturbations"""
    model.eval()

    perturbation_functions = [
        ('char_swap', char_swap_perturbation),
        ('char_delete', char_delete_perturbation),
        ('char_insert', char_insert_perturbation),
        ('word_shuffle', word_shuffle_perturbation)
    ]

    results = defaultdict(list)

    with torch.no_grad():
        for text in tqdm(texts[:500], desc="Adversarial Robustness"):  # Use subset for speed
            # Get original prediction
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)

            original_logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
            original_prob = torch.sigmoid(original_logits).item()
            original_pred = 1 if original_prob > 0.5 else 0

            # Test each perturbation type
            for pert_name, pert_func in perturbation_functions:
                flip_count = 0
                conf_changes = []

                for _ in range(num_perturbations):
                    # Apply perturbation
                    perturbed_text = pert_func(text)

                    # Get perturbed prediction
                    inputs = tokenizer(
                        perturbed_text,
                        max_length=MAX_LENGTH,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    ).to(device)

                    perturbed_logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
                    perturbed_prob = torch.sigmoid(perturbed_logits).item()
                    perturbed_pred = 1 if perturbed_prob > 0.5 else 0

                    # Check if prediction flipped
                    if original_pred != perturbed_pred:
                        flip_count += 1

                    # Calculate confidence change
                    conf_changes.append(abs(original_prob - perturbed_prob))

                results[f'{pert_name}_flip_rate'].append(flip_count / num_perturbations)
                results[f'{pert_name}_conf_change'].append(np.mean(conf_changes))

    # Aggregate results
    aggregated = {}
    for key, values in results.items():
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)

    return aggregated

def calculate_embedding_quality(model, tokenizer, texts, device):
    """Analyze embedding space quality"""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for text in tqdm(texts[:1000], desc="Embedding Analysis"):  # Use subset
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)

            _, embedding = model(inputs['input_ids'], inputs['attention_mask'])
            embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)

    # Calculate statistics
    mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
    std_norm = np.std(np.linalg.norm(embeddings, axis=1))

    # Calculate pairwise cosine similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    similarities = np.dot(normalized, normalized.T)

    # Get upper triangle (exclude diagonal)
    n = similarities.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_sims = similarities[upper_tri_indices]

    return {
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'mean_similarity': np.mean(upper_tri_sims),
        'std_similarity': np.std(upper_tri_sims),
        'embedding_dimension': embeddings.shape[1]
    }

def calculate_calibration_error(model, tokenizer, texts, device, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    model.eval()
    confidences = []
    predictions = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Calibration Analysis"):
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)

            logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
            prob = torch.sigmoid(logits).item()

            confidences.append(prob)
            predictions.append(1 if prob > 0.5 else 0)

    confidences = np.array(confidences)
    predictions = np.array(predictions)

    # Since we don't have true labels for KorQuAD (OOD data),
    # we'll measure confidence distribution properties instead
    return {
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences)),
        'confidence_std': float(np.std(confidences)),
        'overconfident_ratio': float(np.mean(confidences > 0.9)),
        'underconfident_ratio': float(np.mean(confidences < 0.1)),
        'uncertain_ratio': float(np.mean((confidences > 0.3) & (confidences < 0.7)))
    }

# ========================== Model Loading ==========================

def load_classification_model(model_name, checkpoint_path, device):
    """Load classification model with proper architecture"""
    print(f"Loading model: {model_name}")

    # Load base model
    base_model = AutoModel.from_pretrained(MODEL_CONFIGS[model_name]['name'], trust_remote_code=True)

    # Create classification model
    model = ClassificationModel(base_model)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load with strict=False to handle any mismatches
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("No checkpoint loaded - using base model")

    model = model.to(device)
    model.eval()
    return model

# ========================== Main Evaluation ==========================

def evaluate_model_robustness(model_name, config, texts, device):
    """Comprehensive robustness evaluation"""
    print(f"\n{'='*70}")
    print(f"Robustness Evaluation: {model_name}")
    print(f"{'='*70}")

    results = {
        'model': model_name,
        'naive': {},
        'finetuned': {}
    }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)

    # Evaluate NAIVE model
    print(f"\n--- NAIVE Model ---")
    try:
        naive_model = load_classification_model(model_name, None, device)

        print("\n1. Confidence Analysis (OOD Detection)")
        conf_results = calculate_prediction_confidence(naive_model, tokenizer, texts, device, "Naive Confidence")

        print("\n2. Adversarial Robustness")
        adv_results = calculate_adversarial_robustness(naive_model, tokenizer, texts, device)

        print("\n3. Embedding Quality")
        emb_results = calculate_embedding_quality(naive_model, tokenizer, texts, device)

        print("\n4. Calibration Analysis")
        cal_results = calculate_calibration_error(naive_model, tokenizer, texts, device)

        results['naive'] = {
            'confidence': conf_results,
            'adversarial': adv_results,
            'embedding': emb_results,
            'calibration': cal_results
        }

        del naive_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating naive model: {e}")
        import traceback
        traceback.print_exc()

    # Evaluate FINE-TUNED model
    print(f"\n--- FINE-TUNED Model ---")
    try:
        finetuned_model = load_classification_model(model_name, config['checkpoint'], device)

        print("\n1. Confidence Analysis (OOD Detection)")
        conf_results = calculate_prediction_confidence(finetuned_model, tokenizer, texts, device, "Fine-tuned Confidence")

        print("\n2. Adversarial Robustness")
        adv_results = calculate_adversarial_robustness(finetuned_model, tokenizer, texts, device)

        print("\n3. Embedding Quality")
        emb_results = calculate_embedding_quality(finetuned_model, tokenizer, texts, device)

        print("\n4. Calibration Analysis")
        cal_results = calculate_calibration_error(finetuned_model, tokenizer, texts, device)

        results['finetuned'] = {
            'confidence': conf_results,
            'adversarial': adv_results,
            'embedding': emb_results,
            'calibration': cal_results
        }

        del finetuned_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating fine-tuned model: {e}")
        import traceback
        traceback.print_exc()

    return results

# ========================== Results Presentation ==========================

def print_robustness_comparison(all_results):
    """Print comprehensive robustness comparison"""
    print("\n" + "="*100)
    print(" " * 30 + "ROBUSTNESS EVALUATION RESULTS")
    print("="*100)

    for result in all_results:
        model = result['model']
        naive = result.get('naive', {})
        finetuned = result.get('finetuned', {})

        print(f"\n{'='*100}")
        print(f"Model: {model}")
        print(f"{'='*100}")

        # 1. OOD Detection (Confidence on clean text)
        print(f"\n{'--- OOD Detection (Lower confidence = Better OOD detection) ---':^100}")
        print(f"{'Metric':<40} {'Naive':<25} {'Fine-tuned':<25} {'Improvement':<10}")
        print("-"*100)

        if 'confidence' in naive and 'confidence' in finetuned:
            metrics = ['mean_confidence', 'std_confidence', 'mean_entropy', 'positive_ratio']
            for metric in metrics:
                n_val = naive['confidence'].get(metric, 0)
                f_val = finetuned['confidence'].get(metric, 0)

                if metric == 'mean_entropy':
                    improvement = "✓" if f_val > n_val else "✗"
                else:
                    improvement = "✓" if abs(f_val - n_val) < abs(n_val) else "✗"

                print(f"{metric:<40} {n_val:<25.4f} {f_val:<25.4f} {improvement:<10}")

        # 2. Adversarial Robustness
        print(f"\n{'--- Adversarial Robustness (Lower flip rate = Better) ---':^100}")
        print(f"{'Perturbation Type':<40} {'Naive Flip Rate':<25} {'FT Flip Rate':<25} {'Improvement':<10}")
        print("-"*100)

        if 'adversarial' in naive and 'adversarial' in finetuned:
            pert_types = ['char_swap', 'char_delete', 'char_insert', 'word_shuffle']
            for pert in pert_types:
                key = f'{pert}_flip_rate_mean'
                n_val = naive['adversarial'].get(key, 0)
                f_val = finetuned['adversarial'].get(key, 0)
                improvement = "✓" if f_val < n_val else "✗"
                print(f"{pert:<40} {n_val:<25.4f} {f_val:<25.4f} {improvement:<10}")

        # 3. Embedding Quality
        print(f"\n{'--- Embedding Quality ---':^100}")
        print(f"{'Metric':<40} {'Naive':<25} {'Fine-tuned':<25} {'Note':<10}")
        print("-"*100)

        if 'embedding' in naive and 'embedding' in finetuned:
            metrics = ['mean_norm', 'std_norm', 'mean_similarity', 'std_similarity']
            for metric in metrics:
                n_val = naive['embedding'].get(metric, 0)
                f_val = finetuned['embedding'].get(metric, 0)
                print(f"{metric:<40} {n_val:<25.4f} {f_val:<25.4f} {'-':<10}")

        # 4. Calibration
        print(f"\n{'--- Calibration (Confidence Distribution) ---':^100}")
        print(f"{'Metric':<40} {'Naive':<25} {'Fine-tuned':<25} {'Note':<10}")
        print("-"*100)

        if 'calibration' in naive and 'calibration' in finetuned:
            metrics = ['mean_confidence', 'confidence_std', 'overconfident_ratio', 'uncertain_ratio']
            for metric in metrics:
                n_val = naive['calibration'].get(metric, 0)
                f_val = finetuned['calibration'].get(metric, 0)
                print(f"{metric:<40} {n_val:<25.4f} {f_val:<25.4f} {'-':<10}")

    print("\n" + "="*100)
    print("INTERPRETATION GUIDE:")
    print("="*100)
    print("1. OOD Detection: Lower confidence on clean data = Better at detecting out-of-distribution")
    print("2. Adversarial Robustness: Lower flip rate = More robust to perturbations")
    print("3. Embedding Quality: Higher norm variance may indicate better feature learning")
    print("4. Calibration: Lower overconfident_ratio = Better calibrated model")
    print("="*100)

# ========================== Main ==========================

def main():
    print("="*100)
    print(" " * 25 + "ROBUSTNESS EVALUATION FOR CLASSIFICATION MODELS")
    print("="*100)
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Task: Drug Slang Detection (Binary Classification)")
    print(f"Evaluation: Robustness on Out-of-Distribution (Clean Korean) Data")
    print()

    # Load data
    texts = load_korquad_data(DATASET_PATH)

    # Ask for sample size
    print("\n" + "="*100)
    print("How many samples would you like to use for evaluation?")
    print("  - 500-1000:  Quick evaluation (~10-20 minutes per model)")
    print("  - 2000-3000: Moderate evaluation (~30-40 minutes per model)")
    print("  - 5000+:     Comprehensive evaluation (1+ hours per model)")
    print("="*100)

    while True:
        try:
            user_input = input("Enter number of samples [default: 1000]: ").strip()
            max_samples = int(user_input) if user_input else 1000

            if max_samples > len(texts):
                print(f"Warning: Requested {max_samples} but only {len(texts)} available.")
                max_samples = len(texts)

            break
        except ValueError:
            print("Please enter a valid number!")

    eval_texts = texts[:max_samples]
    print(f"\nEvaluating on {len(eval_texts)} samples...")

    # Evaluate all models
    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        if not os.path.exists(config['checkpoint']):
            print(f"\nWarning: Checkpoint not found for {model_name}")
            continue

        try:
            result = evaluate_model_robustness(model_name, config, eval_texts, device)
            all_results.append(result)
        except Exception as e:
            print(f"\nFailed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison
    print_robustness_comparison(all_results)

    # Save results
    output_file = '/home/minseok/forensic/robustness_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "="*100)
    print("Evaluation Complete!")
    print("="*100)

if __name__ == "__main__":
    main()
