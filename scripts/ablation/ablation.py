import os
import json
import argparse
import itertools
import pandas as pd
from datetime import datetime
import torch
import numpy as np

from legacy_msg import (  # adjust if the filename or import path differs
    Config,
    SlidingWindowConversationDataset,
    ConversationAnomalyDetector,
    FocalLoss,
    Trainer,
    ConversationAugmenter,
    set_seed,
    analyze_model_predictions,
)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F

def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    extra = ['[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]', '[MSG]']
    tokenizer.add_special_tokens({'additional_special_tokens': extra})
    return tokenizer

def get_criterion(use_focal, class_weights, label_smoothing, alpha, gamma):
    if use_focal:
        return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

def prepare_model(tokenizer, use_attention_pooling):
    model = ConversationAnomalyDetector(
        model_name=Config.MODEL_NAME,
        hidden_dropout=Config.HIDDEN_DROPOUT,
        attention_dropout=Config.ATTENTION_DROPOUT,
        use_attention_pooling=use_attention_pooling
    ).to(Config.DEVICE)
    model.transformer.resize_token_embeddings(len(tokenizer))
    
    # If you need these attributes, set them after initialization
    model.msg_token_id = tokenizer.convert_tokens_to_ids("[MSG]")
    model.attn_supervision_weight = 0.5
    
    return model

def run_experiment(exp_name, settings, output_dir):
    set_seed(Config.RANDOM_SEED)

    tokenizer = build_tokenizer()
    augmenter = ConversationAugmenter(augment_prob=Config.AUGMENT_PROB) if settings['augmentation'] else None

    full_dataset = SlidingWindowConversationDataset(
        Config.INPUT_DIRECTORY,
        tokenizer,
        window_size=settings.get('window_size', Config.WINDOW_SIZE),
        stride=settings.get('stride', Config.STRIDE),
        max_length=Config.MAX_LENGTH,
        augmenter=augmenter,
        is_training=True
    )

    total_size = len(full_dataset)
    train_size = int(total_size * Config.TRAIN_SPLIT)
    val_size = int(total_size * Config.VAL_SPLIT)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = prepare_model(
        tokenizer,
        use_attention_pooling=settings['use_attention_pooling']
    )

    # class weights
    train_labels = [full_dataset.samples[i]['label'] for i in train_dataset.indices]
    class_weights_np = compute_class_weight(
        'balanced', classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(Config.DEVICE)

    criterion = get_criterion(
        use_focal=settings['use_focal_loss'],
        class_weights=class_weights,
        label_smoothing=settings.get('label_smoothing', Config.LABEL_SMOOTHING),
        alpha=settings.get('focal_alpha', Config.FOCAL_ALPHA),
        gamma=settings.get('focal_gamma', Config.FOCAL_GAMMA),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    Config.EPOCHS = 5
    total_steps = len(train_loader) * Config.EPOCHS // Config.ACCUMULATION_STEPS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

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

    best_val_f1 = trainer.train()
    test_metrics, _ = trainer.evaluate(test_loader)

    result = {
        'experiment': exp_name,
        'settings': settings,
        'best_val_f1': best_val_f1,
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics.get('auc'),
        'test_ap': test_metrics.get('ap'),
        'confusion_matrix': test_metrics['confusion_matrix'],
        'class_weights': class_weights_np.tolist(),
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{exp_name.replace(' ', '_')}_{int(datetime.now().timestamp())}.json"
    with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result

def main():
    parser = argparse.ArgumentParser(description="Ablation runner with window/stride included.")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="output folder")
    parser.add_argument("--quick", action="store_true", help="run a small subset")
    args = parser.parse_args()

    # Ablation grid: keep attention supervision weight fixed
    grid = {
        'use_attention_pooling': [True, False],
        'use_focal_loss': [True, False],
        'augmentation': [True, False],
        'window_size': [10],  # ablate
        'stride': [5],        # ablate
    }

    combos = list(itertools.product(
        grid['use_attention_pooling'],
        grid['use_focal_loss'],
        grid['augmentation'],
        grid['window_size'],
        grid['stride'],
    ))

    if args.quick:
        combos = combos[:4]  # small subset for sanity

    all_results = []
    for uap, ufl, aug, ws, st in combos:
        name = f"attnPool_{uap}_focal_{ufl}_aug_{aug}_ws_{ws}_st_{st}"
        settings = {
            'use_attention_pooling': uap,
            'use_focal_loss': ufl,
            'augmentation': aug,
            'window_size': ws,
            'stride': st,
            # keep label_smoothing, focal params at default unless you want to vary
        }
        print(f"\n=== Running {name} ===")
        try:
            res = run_experiment(name, settings, args.output_dir)
            all_results.append(res)
        except Exception as e:
            print(f"Experiment {name} failed: {e}")

    # Aggregate summary
    df = pd.json_normalize(all_results)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nWritten summary to {summary_path}")

if __name__ == "__main__":
    main()
