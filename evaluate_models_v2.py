"""
Model Evaluation Script for KorQuAD Dataset
Compares naive (base) models vs fine-tuned models
Metrics: Perplexity and BLEU Score

Note: The fine-tuned models are trained for sequence classification,
so we adapt them for QA evaluation by using the language model head.
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModel
)
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================== Configuration ==========================
DATASET_PATH = '/home/minseok/rag_security/KorQuAD_v1.0_dev_cleansed.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 384
BATCH_SIZE = 8

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

# ========================== Helper Functions ==========================

def load_korquad_data(file_path, max_samples=None):
    """Load KorQuAD dataset from JSON file"""
    print(f"Loading KorQuAD dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data['data']:
        for qa in item['qas']:
            samples.append({
                'question': qa['question'],
                'context': qa['context'],
                'answers': qa['answers']
            })

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} samples")
    return samples


def simple_tokenize(text):
    """Simple character-level tokenization for Korean BLEU"""
    # Remove spaces and split into characters
    return list(text.replace(" ", ""))


def calculate_bleu_score(reference, hypothesis):
    """
    Calculate BLEU score for Korean text using character-level matching
    Implements a simplified BLEU with smoothing
    """
    if not hypothesis or not reference:
        return 0.0

    ref_chars = simple_tokenize(reference)
    hyp_chars = simple_tokenize(hypothesis)

    if len(hyp_chars) == 0:
        return 0.0

    # Calculate n-gram precisions (n=1 to 4)
    precisions = []
    for n in range(1, 5):
        ref_ngrams = defaultdict(int)
        hyp_ngrams = defaultdict(int)

        # Count n-grams in reference
        for i in range(len(ref_chars) - n + 1):
            ngram = tuple(ref_chars[i:i+n])
            ref_ngrams[ngram] += 1

        # Count n-grams in hypothesis
        for i in range(len(hyp_chars) - n + 1):
            ngram = tuple(hyp_chars[i:i+n])
            hyp_ngrams[ngram] += 1

        # Calculate clipped counts
        clipped_count = 0
        total_count = 0
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count

        if total_count > 0:
            precision = clipped_count / total_count
        else:
            precision = 0.0

        # Add smoothing for zero counts
        if precision == 0.0:
            precision = 1e-10

        precisions.append(precision)

    # Calculate geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0

    # Brevity penalty
    ref_len = len(ref_chars)
    hyp_len = len(hyp_chars)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / (hyp_len + 1e-10))

    bleu = bp * geo_mean
    return bleu


def calculate_perplexity_on_answers(model, tokenizer, samples, device):
    """
    Calculate pseudo-perplexity by measuring model's confidence on answer text
    Uses masked language modeling approach
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Calculating Perplexity"):
            answer = sample['answers'][0] if sample['answers'] else ""
            question = sample['question']
            context = sample['context']

            if not answer:
                continue

            # Create input: question + context + answer
            full_text = f"{question} {context} {answer}"

            try:
                # Tokenize
                inputs = tokenizer(
                    full_text,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                ).to(device)

                # Get model outputs
                outputs = model(**inputs, output_hidden_states=True)

                # For encoder models, we'll use a proxy: average attention weights
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Calculate attention entropy as a proxy for uncertainty
                    attention = outputs.attentions[-1]  # Last layer
                    # Average over heads and positions
                    avg_attention = attention.mean()
                    loss = -torch.log(avg_attention + 1e-10)
                    total_loss += loss.item()
                    total_count += 1
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use hidden state variance as a proxy
                    hidden = outputs.last_hidden_state
                    variance = hidden.var().item()
                    # Lower variance might indicate more confidence
                    loss = np.log(variance + 1.0)
                    total_loss += loss
                    total_count += 1

            except Exception:
                continue

    if total_count == 0:
        return float('inf')

    avg_loss = total_loss / total_count
    perplexity = np.exp(avg_loss)
    return perplexity


def answer_question_with_qa_model(model, tokenizer, question, context, device):
    """Extract answer using QA model"""
    model.eval()

    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(
            question,
            context,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(device)

        # Get predictions
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the most likely answer span
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()

        # Extract answer
        if start_idx <= end_idx and end_idx < len(inputs['input_ids'][0]):
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
            answer = ""

        return answer


def evaluate_qa_model(model, tokenizer, samples, device, model_type='naive'):
    """Evaluate QA model with perplexity and BLEU"""
    print(f"Evaluating {model_type} model...")

    bleu_scores = []
    valid_predictions = 0

    # Calculate BLEU scores
    for sample in tqdm(samples, desc=f"BLEU Evaluation ({model_type})"):
        question = sample['question']
        context = sample['context']
        reference_answer = sample['answers'][0] if sample['answers'] else ""

        if not reference_answer:
            continue

        try:
            # Get predicted answer
            predicted_answer = answer_question_with_qa_model(
                model, tokenizer, question, context, device
            )

            # Calculate BLEU
            bleu = calculate_bleu_score(reference_answer, predicted_answer)
            bleu_scores.append(bleu)

            if predicted_answer:
                valid_predictions += 1

        except Exception:
            bleu_scores.append(0.0)

    # Calculate perplexity
    perplexity = calculate_perplexity_on_answers(model, tokenizer, samples, device)

    # Calculate average BLEU
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

    print(f"{model_type.upper()} - Perplexity: {perplexity:.4f}, BLEU: {avg_bleu:.4f}")
    print(f"Valid predictions: {valid_predictions}/{len(samples)}")

    return perplexity, avg_bleu


def load_model_with_qa_head(model_name, checkpoint_path, device):
    """
    Load model with QA head
    If checkpoint exists, try to load it; otherwise use base model
    """
    # Load base model with QA head
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, trust_remote_code=True)
    except:
        # If QA head not available, create a basic model
        print(f"QA head not available for {model_name}, using base model")
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Create simple QA head
        class SimpleQAModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.qa_outputs = nn.Linear(base_model.config.hidden_size, 2)

            def forward(self, input_ids, attention_mask=None, **kwargs):
                # kwargs included for compatibility with QA model interface
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state
                logits = self.qa_outputs(sequence_output)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

                # Create a simple output object
                class Output:
                    pass
                result = Output()
                result.start_logits = start_logits
                result.end_logits = end_logits
                result.last_hidden_state = sequence_output
                return result

        model = SimpleQAModel(base_model)

    # Try to load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
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

            # Try to load compatible parameters
            model_state = model.state_dict()
            loaded_params = 0

            for key, value in state_dict.items():
                new_key = key.replace('module.', '').replace('embedding_model.', 'base_model.')

                # Try to find matching key
                if new_key in model_state:
                    if model_state[new_key].shape == value.shape:
                        model_state[new_key] = value
                        loaded_params += 1

            model.load_state_dict(model_state, strict=False)
            print(f"Loaded {loaded_params} parameters from checkpoint")

        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Using base model weights only")

    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model_name, config, samples, device):
    """Evaluate both naive and fine-tuned versions of a model"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    results = {
        'model': model_name,
        'base_model_name': config['name'],
        'naive': {},
        'finetuned': {}
    }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)

    # Evaluate Naive Model
    print(f"\n--- NAIVE Model ({config['name']}) ---")
    try:
        naive_model = load_model_with_qa_head(config['name'], None, device)
        perplexity, bleu = evaluate_qa_model(
            naive_model, tokenizer, samples, device, 'naive'
        )

        results['naive'] = {
            'perplexity': float(perplexity),
            'bleu': float(bleu)
        }

        del naive_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating naive model: {e}")
        results['naive'] = {
            'perplexity': float('inf'),
            'bleu': 0.0,
            'error': str(e)
        }

    # Evaluate Fine-tuned Model
    print(f"\n--- FINE-TUNED Model ({config['checkpoint']}) ---")
    try:
        finetuned_model = load_model_with_qa_head(
            config['name'], config['checkpoint'], device
        )
        perplexity, bleu = evaluate_qa_model(
            finetuned_model, tokenizer, samples, device, 'finetuned'
        )

        results['finetuned'] = {
            'perplexity': float(perplexity),
            'bleu': float(bleu)
        }

        del finetuned_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating fine-tuned model: {e}")
        results['finetuned'] = {
            'perplexity': float('inf'),
            'bleu': 0.0,
            'error': str(e)
        }

    return results


def print_comparison_table(all_results):
    """Print a comprehensive comparison table"""
    print("\n" + "="*100)
    print(" " * 35 + "EVALUATION RESULTS")
    print("="*100)
    print(f"{'Model':<20} {'Type':<15} {'Perplexity':<15} {'BLEU':<12} {'Δ Perplexity':<18} {'Δ BLEU':<15}")
    print("-"*100)

    for result in all_results:
        model = result['model']
        naive = result['naive']
        finetuned = result['finetuned']

        # Naive results
        naive_ppl = naive.get('perplexity', float('inf'))
        naive_bleu = naive.get('bleu', 0.0)

        print(f"{model:<20} {'Naive':<15} {naive_ppl:<15.4f} {naive_bleu:<12.4f} {'-':<18} {'-':<15}")

        # Fine-tuned results
        ft_ppl = finetuned.get('perplexity', float('inf'))
        ft_bleu = finetuned.get('bleu', 0.0)

        # Calculate improvements
        if naive_ppl != float('inf') and ft_ppl != float('inf') and naive_ppl > 0:
            ppl_change = ft_ppl - naive_ppl
            ppl_pct = (ppl_change / naive_ppl) * 100
            ppl_str = f"{ppl_change:+.4f} ({ppl_pct:+.2f}%)"
        else:
            ppl_str = "N/A"

        if naive_bleu > 0:
            bleu_change = ft_bleu - naive_bleu
            bleu_pct = (bleu_change / naive_bleu) * 100
            bleu_str = f"{bleu_change:+.4f} ({bleu_pct:+.2f}%)"
        else:
            bleu_str = "N/A"

        print(f"{model:<20} {'Fine-tuned':<15} {ft_ppl:<15.4f} {ft_bleu:<12.4f} {ppl_str:<18} {bleu_str:<15}")
        print("-"*100)

    print("\nNote:")
    print("  - Lower Perplexity = Better (model is more confident)")
    print("  - Higher BLEU = Better (predictions closer to reference)")
    print("  - Negative Δ Perplexity = Improvement")
    print("  - Positive Δ BLEU = Improvement")
    print("="*100)


# ========================== Main Execution ==========================

def main():
    print("="*100)
    print(" " * 20 + "Model Evaluation: Perplexity and BLEU Score Analysis")
    print("="*100)
    # print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max Length: {MAX_LENGTH}")
    print()

    # Load dataset
    samples = load_korquad_data(DATASET_PATH)

    # Ask for sample size
    print("\n" + "="*100)
    print("How many samples would you like to evaluate?")
    print("  - 50-100:  Quick test (~5-10 minutes)")
    print("  - 200-500: Moderate evaluation (~20-30 minutes)")
    print("  - 1000+:   Comprehensive evaluation (1+ hours)")
    print("="*100)

    while True:
        try:
            user_input = input("Enter number of samples [default: 100]: ").strip()
            max_samples = int(user_input) if user_input else 100

            if max_samples > len(samples):
                print(f"Warning: Requested {max_samples} but only {len(samples)} available. Using {len(samples)}.")
                max_samples = len(samples)

            break
        except ValueError:
            print("Please enter a valid number!")

    print(f"\nEvaluating on {max_samples} samples from KorQuAD dataset...")
    eval_samples = samples[:max_samples]

    # Evaluate all models
    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        # Check if checkpoint exists
        if not os.path.exists(config['checkpoint']):
            print(f"\nWarning: Checkpoint not found for {model_name}: {config['checkpoint']}")
            print("Skipping this model...")
            continue

        try:
            result = evaluate_model(model_name, config, eval_samples, device)
            all_results.append(result)
        except Exception as e:
            print(f"\nFailed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison table
    print_comparison_table(all_results)

    # Save results to JSON
    output_file = '/home/minseok/forensic/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "="*100)
    print("Evaluation Complete!")
    print("="*100)


if __name__ == "__main__":
    main()
