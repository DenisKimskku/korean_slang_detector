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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================== Configuration ==========================
DATASET_PATH = '/home/minseok/rag_security/KorQuAD_v1.0_dev_cleansed.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 512
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


def calculate_perplexity(model, tokenizer, samples, device, model_type='naive'):
    """
    Calculate perplexity for language models
    For QA task, we measure perplexity on the answer generation
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    print(f"Calculating perplexity for {model_type} model...")

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"Perplexity ({model_type})"):
            # Create input: question + context
            text = f"질문: {sample['question']} 문맥: {sample['context']}"
            target = sample['answers'][0] if sample['answers'] else ""

            if not target:
                continue

            # Tokenize
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            target_tokens = tokenizer(
                target,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids'].to(device)

            if target_tokens.size(1) == 0:
                continue

            try:
                # Get model outputs
                if hasattr(model, 'forward'):
                    outputs = model(**inputs)

                    if hasattr(outputs, 'last_hidden_state'):
                        # For encoder models, use the last hidden state
                        hidden_states = outputs.last_hidden_state
                        # Use the first token's representation as a proxy
                        logits = hidden_states[:, 0, :]

                        # Calculate cross-entropy with target tokens
                        # This is an approximation for encoder-only models
                        vocab_size = tokenizer.vocab_size
                        if logits.size(-1) != vocab_size:
                            # Skip if dimensions don't match
                            continue
                    else:
                        continue

                # Simple approximation: calculate average loss
                loss = criterion(logits.view(-1), target_tokens.view(-1))
                total_loss += loss.item()
                total_tokens += target_tokens.numel()

            except Exception as e:
                # Skip problematic samples
                continue

    if total_tokens == 0:
        return float('inf')

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


def calculate_perplexity_qa(model, tokenizer, samples, device, model_type='naive'):
    """
    Calculate perplexity using QA model approach
    Measures how well the model assigns probability to correct answers
    """
    model.eval()
    total_nll = 0  # negative log-likelihood
    count = 0

    print(f"Calculating QA perplexity for {model_type} model...")

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"QA Perplexity ({model_type})"):
            question = sample['question']
            context = sample['context']
            answers = sample['answers']

            if not answers or not answers[0]:
                continue

            answer_text = answers[0]

            # Tokenize input
            encoding = tokenizer(
                question,
                context,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            try:
                # Get model predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # Find answer position in context
                answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                context_tokens = encoding['input_ids'][0].tolist()

                # Find the answer span in the tokenized context
                answer_start = -1
                for i in range(len(context_tokens) - len(answer_tokens) + 1):
                    if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
                        answer_start = i
                        break

                if answer_start != -1 and answer_start + len(answer_tokens) <= MAX_LENGTH:
                    answer_end = answer_start + len(answer_tokens) - 1

                    # Calculate negative log-likelihood
                    start_nll = -torch.log_softmax(start_logits, dim=1)[0, answer_start]
                    end_nll = -torch.log_softmax(end_logits, dim=1)[0, answer_end]

                    total_nll += (start_nll.item() + end_nll.item())
                    count += 1

            except Exception as e:
                continue

    if count == 0:
        return float('inf')

    # Perplexity = exp(average negative log-likelihood)
    perplexity = np.exp(total_nll / count)
    return perplexity


def calculate_bleu(model, tokenizer, samples, device, model_type='naive'):
    """Calculate BLEU score for QA predictions"""
    model.eval()
    bleu_scores = []
    smooth = SmoothingFunction()

    print(f"Calculating BLEU score for {model_type} model...")

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"BLEU ({model_type})"):
            question = sample['question']
            context = sample['context']
            reference_answers = sample['answers']

            if not reference_answers:
                continue

            # Tokenize input
            encoding = tokenizer(
                question,
                context,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            try:
                # Get model predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # Get predicted span
                start_idx = torch.argmax(start_logits, dim=1).item()
                end_idx = torch.argmax(end_logits, dim=1).item()

                if start_idx <= end_idx:
                    predicted_tokens = input_ids[0][start_idx:end_idx+1]
                    predicted_answer = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                else:
                    predicted_answer = ""

                # Calculate BLEU score
                if predicted_answer:
                    # Tokenize for BLEU calculation (character-level for Korean)
                    pred_chars = list(predicted_answer.replace(" ", ""))
                    ref_chars_list = [list(ref.replace(" ", "")) for ref in reference_answers]

                    # Calculate BLEU score with smoothing
                    bleu = sentence_bleu(
                        ref_chars_list,
                        pred_chars,
                        smoothing_function=smooth.method1
                    )
                    bleu_scores.append(bleu)
                else:
                    bleu_scores.append(0.0)

            except Exception as e:
                bleu_scores.append(0.0)
                continue

    if not bleu_scores:
        return 0.0

    avg_bleu = np.mean(bleu_scores)
    return avg_bleu


def load_finetuned_qa_model(base_model_name, checkpoint_path, device):
    """
    Load fine-tuned model checkpoint
    Assumes the checkpoint contains a QA head or can be adapted
    """
    try:
        # Load the base QA model architecture
        model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Try to load state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Filter out incompatible keys
            model_state = model.state_dict()
            filtered_state_dict = {}

            for key, value in state_dict.items():
                # Remove 'module.' prefix if present
                new_key = key.replace('module.', '')

                # Check if key exists in model and shapes match
                if new_key in model_state and model_state[new_key].shape == value.shape:
                    filtered_state_dict[new_key] = value

            # Load filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded {len(filtered_state_dict)} parameters from checkpoint")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using base model")

        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        print("Falling back to base model...")
        model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)
        model = model.to(device)
        model.eval()
        return model


def evaluate_model(model_name, config, samples, device, max_samples=100):
    """Evaluate both naive and fine-tuned models"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    results = {
        'model': model_name,
        'naive': {},
        'finetuned': {}
    }

    # Use subset for faster evaluation
    eval_samples = samples[:max_samples]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['name'])

    # ============ Evaluate Naive (Base) Model ============
    print(f"\n--- Loading NAIVE model: {config['name']} ---")
    try:
        naive_model = AutoModelForQuestionAnswering.from_pretrained(config['name'])
        naive_model = naive_model.to(device)
        naive_model.eval()

        # Calculate metrics
        perplexity = calculate_perplexity_qa(naive_model, tokenizer, eval_samples, device, 'naive')
        bleu = calculate_bleu(naive_model, tokenizer, eval_samples, device, 'naive')

        results['naive'] = {
            'perplexity': perplexity,
            'bleu': bleu
        }

        print(f"Naive Model - Perplexity: {perplexity:.4f}, BLEU: {bleu:.4f}")

        # Clean up
        del naive_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating naive model: {e}")
        results['naive'] = {
            'perplexity': float('inf'),
            'bleu': 0.0,
            'error': str(e)
        }

    # ============ Evaluate Fine-tuned Model ============
    print(f"\n--- Loading FINE-TUNED model from {config['checkpoint']} ---")
    try:
        finetuned_model = load_finetuned_qa_model(
            config['name'],
            config['checkpoint'],
            device
        )

        # Calculate metrics
        perplexity = calculate_perplexity_qa(finetuned_model, tokenizer, eval_samples, device, 'finetuned')
        bleu = calculate_bleu(finetuned_model, tokenizer, eval_samples, device, 'finetuned')

        results['finetuned'] = {
            'perplexity': perplexity,
            'bleu': bleu
        }

        print(f"Fine-tuned Model - Perplexity: {perplexity:.4f}, BLEU: {bleu:.4f}")

        # Clean up
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


def print_summary(all_results):
    """Print summary comparison table"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Type':<12} {'Perplexity':<15} {'BLEU':<10} {'Performance':<15}")
    print("-"*80)

    for result in all_results:
        model = result['model']

        # Naive model
        naive = result['naive']
        naive_ppl = naive.get('perplexity', float('inf'))
        naive_bleu = naive.get('bleu', 0.0)

        print(f"{model:<20} {'Naive':<12} {naive_ppl:<15.4f} {naive_bleu:<10.4f} {'Baseline':<15}")

        # Fine-tuned model
        ft = result['finetuned']
        ft_ppl = ft.get('perplexity', float('inf'))
        ft_bleu = ft.get('bleu', 0.0)

        # Calculate improvement
        if naive_ppl != float('inf') and ft_ppl != float('inf'):
            ppl_improvement = ((naive_ppl - ft_ppl) / naive_ppl) * 100
            ppl_str = f"{ppl_improvement:+.2f}%"
        else:
            ppl_str = "N/A"

        if naive_bleu > 0:
            bleu_improvement = ((ft_bleu - naive_bleu) / naive_bleu) * 100
            bleu_str = f"{bleu_improvement:+.2f}%"
        else:
            bleu_str = "N/A"

        performance = f"PPL:{ppl_str} BLEU:{bleu_str}"

        print(f"{model:<20} {'Fine-tuned':<12} {ft_ppl:<15.4f} {ft_bleu:<10.4f} {performance:<15}")
        print("-"*80)

    print("\nNote: Lower perplexity is better, higher BLEU is better")
    print("="*80)


# ========================== Main Execution ==========================

def main():
    print("="*80)
    print("Model Evaluation: Perplexity and BLEU Score Comparison")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")
    print()

    # Load dataset
    samples = load_korquad_data(DATASET_PATH)

    # Ask user for number of samples to evaluate
    print("\nHow many samples would you like to evaluate?")
    print("(Smaller number = faster evaluation, recommended: 50-200)")
    try:
        max_samples = int(input("Enter number (or press Enter for 100): ") or "100")
    except:
        max_samples = 100

    print(f"\nEvaluating on {max_samples} samples...")

    # Evaluate all models
    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        try:
            result = evaluate_model(model_name, config, samples, DEVICE, max_samples)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")
            continue

    # Print summary
    print_summary(all_results)

    # Save results to JSON
    output_file = 'model_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
