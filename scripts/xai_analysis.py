#!/usr/bin/env python
"""
XAI Analysis for Drug Slang Detection Models
Performs SHAP and Saliency analysis on trained models
Compares label 0 (non-slang) vs label 1 (slang)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

# HTML generation
from jinja2 import Template

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ========================== Configuration ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128
BATCH_SIZE = 16

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

# Data path
DATA_PATH = '/home/minseok/forensic/drug_slang_conversations.json'
OUTPUT_DIR = '/home/minseok/forensic/xai_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seaborn color palette
palette = sns.color_palette("muted", 8)

# ========================== Data Loading ==========================

def load_test_data(file_path, max_conversations=None):
    """Load test data from JSON file"""
    print(f"Loading test data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_conversations:
        data = data[:max_conversations]

    print(f"Loaded {len(data)} conversations")
    return data

# ========================== Model Loading ==========================

def load_model(model_name, config, device):
    """Load fine-tuned model for sequence classification"""
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)

    # Load model with classification head
    model_config = AutoConfig.from_pretrained(config['name'], num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        config['name'],
        config=model_config,
        trust_remote_code=True
    )

    # Load checkpoint
    if os.path.exists(config['checkpoint']):
        print(f"Loading checkpoint from {config['checkpoint']}")
        checkpoint = torch.load(config['checkpoint'], map_location=device)

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

        # Clean up state dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '').replace('embedding_model.', '')
            new_state_dict[new_key] = value

        # Load state dict
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint successfully")

    model = model.to(device)
    model.eval()

    return model, tokenizer

# ========================== Saliency Computation ==========================

def compute_saliency(model, tokenizer, text, device):
    """
    Compute saliency map using gradient-based method (similar to saliency4.py)
    Returns saliency scores for each token
    """
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)

    # Get embeddings and make them leaf variables
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(inputs['input_ids'])

    # Detach and clone to make it a leaf variable, then require grad
    embeddings = embeddings.detach().clone()
    embeddings.requires_grad = True

    # Forward pass with custom embeddings
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=inputs['attention_mask']
    )

    # Get prediction
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1)

    # Backward pass on predicted class
    model.zero_grad()
    logits[0, pred_class].backward()

    # Compute saliency as L2 norm of gradients
    saliency = embeddings.grad.norm(dim=-1).squeeze(0).cpu().detach().numpy()

    # Get tokens (excluding padding)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_mask = inputs['attention_mask'][0].cpu().numpy()

    # Filter to actual tokens (non-padding)
    actual_length = int(attention_mask.sum())
    tokens = tokens[:actual_length]
    saliency = saliency[:actual_length]

    return saliency, tokens, pred_class.item()

# ========================== SHAP Computation ==========================

def compute_shap(model, tokenizer, texts, device, max_samples=100):
    """
    Compute SHAP values for text classification using a simplified approach
    """
    print("Computing SHAP values...")

    # Sample texts if too many
    if len(texts) > max_samples:
        sample_indices = np.random.choice(len(texts), max_samples, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts

    # Compute token-level SHAP-like values using gradient-based approximation
    # This is a simplified version that works better with transformers
    all_shap_values = []

    for text in tqdm(sample_texts, desc="Computing SHAP"):
        try:
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)

            # Get embeddings and make them leaf variables
            with torch.no_grad():
                embeddings = model.get_input_embeddings()(inputs['input_ids'])

            # Detach and clone to make it a leaf variable
            embeddings = embeddings.detach().clone()
            embeddings.requires_grad = True

            # Forward pass
            outputs = model(
                inputs_embeds=embeddings,
                attention_mask=inputs['attention_mask']
            )

            # Get gradients for both classes
            logits = outputs.logits

            # For class 1 (slang)
            model.zero_grad()
            logits[0, 1].backward()
            grad_class1 = embeddings.grad.clone()

            # Compute importance as gradient * embedding (approximation of SHAP)
            importance = (grad_class1 * embeddings.detach()).sum(dim=-1).abs().squeeze(0)

            # Get actual tokens (non-padding)
            attention_mask = inputs['attention_mask'][0].cpu().numpy()
            actual_length = int(attention_mask.sum())

            # Store importance values for actual tokens
            token_importance = importance[:actual_length].cpu().detach().numpy()
            all_shap_values.append(token_importance)

        except Exception as e:
            print(f"Error computing SHAP for text: {e}")
            continue

    return all_shap_values

# ========================== Analysis Functions ==========================

def analyze_model(model_name, config, data, device, load_checkpoint=True, model_type='trained'):
    """
    Perform comprehensive XAI analysis on a single model
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name} ({model_type})")
    print(f"{'='*70}")

    # Load model
    if load_checkpoint:
        model, tokenizer = load_model(model_name, config, device)
    else:
        # Load naive base model without checkpoint
        print(f"Loading naive base model: {config['name']}")
        tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(config['name'], num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            config['name'],
            config=model_config,
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()

    # Collect data by label
    label_0_texts = []  # Non-slang
    label_1_texts = []  # Slang
    label_0_saliency = []
    label_1_saliency = []
    conversation_results = []

    # Process each conversation
    for conv in tqdm(data, desc=f"Processing {model_name}"):
        conv_id = conv['id']
        conv_utterances = []

        for utt in conv['utterance']:
            text = utt['original_form']
            label = utt['label']

            try:
                # Compute saliency
                saliency, tokens, pred = compute_saliency(model, tokenizer, text, device)

                # Normalize saliency to 0-10 scale
                if saliency.max() > saliency.min():
                    saliency_normalized = (saliency - saliency.min()) / (saliency.max() - saliency.min()) * 10
                else:
                    saliency_normalized = saliency

                # Store by label
                if label == 0:
                    label_0_texts.append(text)
                    label_0_saliency.extend(saliency_normalized)
                else:
                    label_1_texts.append(text)
                    label_1_saliency.extend(saliency_normalized)

                # Store conversation results
                conv_utterances.append({
                    'text': text,
                    'label': label,
                    'prediction': pred,
                    'tokens': tokens,
                    'saliency': saliency_normalized.tolist()
                })

            except Exception as e:
                print(f"Error processing utterance: {e}")
                continue

        conversation_results.append({
            'conv_id': conv_id,
            'utterances': conv_utterances
        })

    # Compute SHAP values - sample from each label separately to ensure balanced representation
    max_samples_per_label = 50

    # Sample label 0 texts
    if len(label_0_texts) > max_samples_per_label:
        sample_0_indices = np.random.choice(len(label_0_texts), max_samples_per_label, replace=False)
        sample_0_texts = [label_0_texts[i] for i in sample_0_indices]
    else:
        sample_0_texts = label_0_texts

    # Sample label 1 texts
    if len(label_1_texts) > max_samples_per_label:
        sample_1_indices = np.random.choice(len(label_1_texts), max_samples_per_label, replace=False)
        sample_1_texts = [label_1_texts[i] for i in sample_1_indices]
    else:
        sample_1_texts = label_1_texts

    # Compute SHAP for each label
    print(f"Computing SHAP for {len(sample_0_texts)} label_0 texts and {len(sample_1_texts)} label_1 texts...")
    shap_values_0 = compute_shap(model, tokenizer, sample_0_texts, device, max_samples=len(sample_0_texts))
    shap_values_1 = compute_shap(model, tokenizer, sample_1_texts, device, max_samples=len(sample_1_texts))

    # Extract SHAP values by label
    label_0_shap = []
    label_1_shap = []

    for shap_val in shap_values_0:
        label_0_shap.extend(shap_val)

    for shap_val in shap_values_1:
        label_1_shap.extend(shap_val)

    results = {
        'model_name': model_name,
        'model_type': model_type,
        'label_0_saliency': np.array(label_0_saliency),
        'label_1_saliency': np.array(label_1_saliency),
        'label_0_shap': np.array(label_0_shap) if label_0_shap else np.array([]),
        'label_1_shap': np.array(label_1_shap) if label_1_shap else np.array([]),
        'conversations': conversation_results,
        'shap_values_0': shap_values_0,
        'shap_values_1': shap_values_1
    }

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results

# ========================== Visualization ==========================

def plot_kde_comparison(label_0_values, label_1_values, title, xlabel, output_path):
    """
    Create KDE distribution plot comparing label 0 vs label 1
    """
    plt.figure(figsize=(10, 6))

    # Plot KDE
    sns.kdeplot(label_0_values, color=palette[0], label="Non-Slang (Label 0)", fill=True, alpha=0.6)
    sns.kdeplot(label_1_values, color=palette[3], label="Slang (Label 1)", fill=True, alpha=0.6)

    plt.xlabel(xlabel, fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlim(left=-0.05)

    # Add statistics text
    mean_0 = np.mean(label_0_values)
    mean_1 = np.mean(label_1_values)
    std_0 = np.std(label_0_values)
    std_1 = np.std(label_1_values)

    stats_text = f"Label 0: μ={mean_0:.3f}, σ={std_0:.3f}\nLabel 1: μ={mean_1:.3f}, σ={std_1:.3f}\nDifference: {mean_1-mean_0:.3f}"
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format ticks
    plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")

def create_html_visualization(conversation_data, model_name, output_path):
    """
    Create HTML visualization for a single conversation
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>XAI Analysis - {{ model_name }} - {{ conv_id }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            h2 {
                color: #555;
                margin-top: 30px;
            }
            .utterance {
                margin: 20px 0;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ddd;
            }
            .label-0 {
                background-color: #e3f2fd;
                border-left-color: #2196F3;
            }
            .label-1 {
                background-color: #fff3e0;
                border-left-color: #FF9800;
            }
            .tokens {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-top: 10px;
            }
            .token {
                padding: 5px 10px;
                border-radius: 3px;
                font-family: monospace;
                font-size: 14px;
            }
            .info {
                margin: 10px 0;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .label-badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            .label-0-badge {
                background-color: #2196F3;
                color: white;
            }
            .label-1-badge {
                background-color: #FF9800;
                color: white;
            }
            .pred-correct {
                color: #4CAF50;
                font-weight: bold;
            }
            .pred-incorrect {
                color: #f44336;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>XAI Analysis: {{ model_name }}</h1>
            <h2>Conversation ID: {{ conv_id }}</h2>

            {% for utt in utterances %}
            <div class="utterance label-{{ utt.label }}">
                <div class="info">
                    <strong>Text:</strong> {{ utt.text }}<br>
                    <strong>True Label:</strong>
                    <span class="label-badge label-{{ utt.label }}-badge">
                        {{ "Slang" if utt.label == 1 else "Non-Slang" }}
                    </span>
                    <strong>Prediction:</strong>
                    <span class="{% if utt.label == utt.prediction %}pred-correct{% else %}pred-incorrect{% endif %}">
                        {{ "Slang" if utt.prediction == 1 else "Non-Slang" }}
                    </span>
                </div>

                <div class="tokens">
                    {% for token, sal in zip(utt.tokens, utt.saliency) %}
                    <span class="token" style="background-color: rgba(255, 0, 0, {{ sal / 10 }});">
                        {{ token }}
                    </span>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    html_content = template.render(
        model_name=model_name,
        conv_id=conversation_data['conv_id'],
        utterances=conversation_data['utterances'],
        zip=zip
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Saved HTML: {output_path}")

# ========================== Main Execution ==========================

def main():
    print("="*70)
    print("XAI Analysis for Drug Slang Detection Models")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load data
    # Set to None to load all conversations, or specify a number for testing
    data = load_test_data(DATA_PATH, max_conversations=None)  # Load all conversations

    # Analyze each model (both naive and trained)
    all_results = {}

    for model_name, config in MODEL_CONFIGS.items():
        if not os.path.exists(config['checkpoint']):
            print(f"Warning: Checkpoint not found for {model_name}")
            continue

        try:
            # Analyze NAIVE model
            print(f"\n{'#'*70}")
            print(f"# Analyzing NAIVE MODEL: {model_name}")
            print(f"{'#'*70}")
            naive_results = analyze_model(model_name, config, data, device, load_checkpoint=False, model_type='naive')
            all_results[f"{model_name}_naive"] = naive_results

            # Analyze TRAINED model
            print(f"\n{'#'*70}")
            print(f"# Analyzing TRAINED MODEL: {model_name}")
            print(f"{'#'*70}")
            trained_results = analyze_model(model_name, config, data, device, load_checkpoint=True, model_type='trained')
            all_results[f"{model_name}_trained"] = trained_results

            # Save individual results for both naive and trained
            for results_key, results in [(f"{model_name}_naive", naive_results), (f"{model_name}_trained", trained_results)]:
                # Process both naive and trained results
                base_model_name = model_name
                model_type = results['model_type']

                # Create output directory for this model
                model_output_dir = os.path.join(OUTPUT_DIR, base_model_name, model_type)
                os.makedirs(model_output_dir, exist_ok=True)

                # Plot saliency KDE
                if len(results['label_0_saliency']) > 0 and len(results['label_1_saliency']) > 0:
                    plot_kde_comparison(
                        results['label_0_saliency'],
                        results['label_1_saliency'],
                        f"Saliency Distribution - {base_model_name} ({model_type})",
                        "Saliency Value",
                        os.path.join(model_output_dir, 'saliency_kde.png')
                    )

                # Plot SHAP KDE
                if len(results['label_0_shap']) > 0 and len(results['label_1_shap']) > 0:
                    plot_kde_comparison(
                        results['label_0_shap'],
                        results['label_1_shap'],
                        f"SHAP Distribution - {base_model_name} ({model_type})",
                        "SHAP Value (Absolute)",
                        os.path.join(model_output_dir, 'shap_kde.png')
                    )

                # Generate HTML visualizations for each conversation
                html_dir = os.path.join(model_output_dir, 'conversations')
                os.makedirs(html_dir, exist_ok=True)

                # Generate HTML for all conversations (no limit)
                print(f"Generating HTML for {len(results['conversations'])} conversations...")
                for conv_data in results['conversations']:
                    html_path = os.path.join(html_dir, f"{conv_data['conv_id']}.html")
                    create_html_visualization(conv_data, f"{base_model_name} ({model_type})", html_path)

                # Save statistics
                stats = {
                    'model_name': base_model_name,
                    'model_type': model_type,
                    'label_0_count': len(results['label_0_saliency']),
                    'label_1_count': len(results['label_1_saliency']),
                    'label_0_saliency_mean': float(np.mean(results['label_0_saliency'])) if len(results['label_0_saliency']) > 0 else 0.0,
                    'label_1_saliency_mean': float(np.mean(results['label_1_saliency'])) if len(results['label_1_saliency']) > 0 else 0.0,
                    'label_0_saliency_std': float(np.std(results['label_0_saliency'])) if len(results['label_0_saliency']) > 0 else 0.0,
                    'label_1_saliency_std': float(np.std(results['label_1_saliency'])) if len(results['label_1_saliency']) > 0 else 0.0,
                    'label_0_shap_mean': float(np.mean(results['label_0_shap'])) if len(results['label_0_shap']) > 0 else 0.0,
                    'label_1_shap_mean': float(np.mean(results['label_1_shap'])) if len(results['label_1_shap']) > 0 else 0.0,
                    'label_0_shap_std': float(np.std(results['label_0_shap'])) if len(results['label_0_shap']) > 0 else 0.0,
                    'label_1_shap_std': float(np.std(results['label_1_shap'])) if len(results['label_1_shap']) > 0 else 0.0
                }

                stats_path = os.path.join(model_output_dir, 'statistics.json')
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

                print(f"Completed analysis for {results_key}")
                print(f"Statistics: {stats}")
                print()

        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary comparison plots
    print("\nCreating summary comparison plots...")

    # Group results by base model
    model_groups = {}
    for full_name, results in all_results.items():
        base_name = full_name.replace('_naive', '').replace('_trained', '')
        if base_name not in model_groups:
            model_groups[base_name] = {}
        model_type = results['model_type']
        model_groups[base_name][model_type] = results

    # Create comparison plots for naive vs trained
    for base_name, models in model_groups.items():
        if 'naive' in models and 'trained' in models:
            _, axes = plt.subplots(2, 2, figsize=(16, 12))

            naive = models['naive']
            trained = models['trained']

            # Saliency - Label 0
            ax = axes[0, 0]
            sns.kdeplot(naive['label_0_saliency'], color=palette[0], label='Naive', fill=True, alpha=0.6, ax=ax)
            sns.kdeplot(trained['label_0_saliency'], color=palette[3], label='Trained', fill=True, alpha=0.6, ax=ax)
            ax.set_xlabel('Saliency Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{base_name} - Saliency (Non-Slang)', fontsize=14, fontweight='bold')
            # Add statistics
            n_mean = np.mean(naive['label_0_saliency'])
            t_mean = np.mean(trained['label_0_saliency'])
            change = t_mean - n_mean
            stats_text = f"Naive: μ={n_mean:.3f}\nTrained: μ={t_mean:.3f}\nChange: {change:+.3f}"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Saliency - Label 1
            ax = axes[0, 1]
            sns.kdeplot(naive['label_1_saliency'], color=palette[0], label='Naive', fill=True, alpha=0.6, ax=ax)
            sns.kdeplot(trained['label_1_saliency'], color=palette[3], label='Trained', fill=True, alpha=0.6, ax=ax)
            ax.set_xlabel('Saliency Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{base_name} - Saliency (Slang)', fontsize=14, fontweight='bold')
            # Add statistics
            n_mean = np.mean(naive['label_1_saliency'])
            t_mean = np.mean(trained['label_1_saliency'])
            change = t_mean - n_mean
            stats_text = f"Naive: μ={n_mean:.3f}\nTrained: μ={t_mean:.3f}\nChange: {change:+.3f}"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.legend()
            ax.grid(True, alpha=0.3)

            # SHAP - Label 0
            ax = axes[1, 0]
            if len(naive['label_0_shap']) > 0 and len(trained['label_0_shap']) > 0:
                sns.kdeplot(naive['label_0_shap'], color=palette[0], label='Naive', fill=True, alpha=0.6, ax=ax)
                sns.kdeplot(trained['label_0_shap'], color=palette[3], label='Trained', fill=True, alpha=0.6, ax=ax)
                # Add statistics
                n_mean = np.mean(naive['label_0_shap'])
                t_mean = np.mean(trained['label_0_shap'])
                change = t_mean - n_mean
                stats_text = f"Naive: μ={n_mean:.4f}\nTrained: μ={t_mean:.4f}\nChange: {change:+.4f}"
                ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{base_name} - SHAP (Non-Slang)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # SHAP - Label 1
            ax = axes[1, 1]
            if len(naive['label_1_shap']) > 0 and len(trained['label_1_shap']) > 0:
                sns.kdeplot(naive['label_1_shap'], color=palette[0], label='Naive', fill=True, alpha=0.6, ax=ax)
                sns.kdeplot(trained['label_1_shap'], color=palette[3], label='Trained', fill=True, alpha=0.6, ax=ax)
                # Add statistics
                n_mean = np.mean(naive['label_1_shap'])
                t_mean = np.mean(trained['label_1_shap'])
                change = t_mean - n_mean
                stats_text = f"Naive: μ={n_mean:.4f}\nTrained: μ={t_mean:.4f}\nChange: {change:+.4f}"
                ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{base_name} - SHAP (Slang)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}_naive_vs_trained.png'), dpi=1200, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison plot: {base_name}_naive_vs_trained.png")

    # Overall summary histogram
    num_models = len(all_results)

    if num_models == 0:
        print("\nWarning: No models were successfully analyzed. Skipping summary plots.")
        print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")
        print("="*70)
        return

    _, axes = plt.subplots(num_models, 2, figsize=(16, 4 * num_models))
    if num_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, results) in enumerate(all_results.items()):
        # Saliency
        ax = axes[idx, 0] if num_models > 1 else axes[0]
        ax.hist(results['label_0_saliency'], bins=50, alpha=0.6, label='Non-Slang', color=palette[0])
        ax.hist(results['label_1_saliency'], bins=50, alpha=0.6, label='Slang', color=palette[3])
        ax.set_xlabel('Saliency Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Saliency', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # SHAP
        ax = axes[idx, 1] if num_models > 1 else axes[1]
        ax.hist(results['label_0_shap'], bins=50, alpha=0.6, label='Non-Slang', color=palette[0])
        ax.hist(results['label_1_shap'], bins=50, alpha=0.6, label='Slang', color=palette[3])
        ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - SHAP', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_all_models.png'), dpi=1200, bbox_inches='tight')
    plt.close()

    print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
