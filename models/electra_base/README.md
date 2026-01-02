# ELECTRA Base Training

## Model Information

- **Model Name**: `monologg/koelectra-base-v3-discriminator`
- **Architecture**: ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
- **Language**: Korean
- **Task**: Drug-related conversation anomaly detection

## Directory Structure

```
electra_base/
├── train.py                    # Main training script
├── logs_electra/               # Training logs (created during training)
├── models_electra/             # Saved models (created during training)
└── checkpoints_electra/        # Training checkpoints (created during training)
```

## Key Differences from BERT/RoBERTa

ELECTRA uses a unique training approach:
- **Replaced Token Detection**: Instead of masked language modeling, ELECTRA uses a generator-discriminator architecture
- **More efficient**: Learns from all tokens, not just masked ones
- **Better sample efficiency**: Achieves better performance with less training data
- **Korean-optimized**: `koelectra-base-v3` is specifically trained for Korean language

## Training

### Prerequisites

1. Activate the `ms` conda environment:
```bash
conda activate ms
```

2. Ensure training data is available at:
```
/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified/
```

### Run Training

```bash
cd /home/minseok/forensic/electra_base
python train.py
```

### Training Configuration

- **Batch Size**: 16
- **Gradient Accumulation**: 2 steps
- **Epochs**: 20 (with early stopping)
- **Learning Rate**: 2e-5
- **Window Size**: 10 messages
- **Stride**: 5 messages
- **Max Length**: 512 tokens

### Key Features

1. **Sliding Window**: Processes conversations in overlapping windows
2. **Attention Supervision**: Guides model to focus on anomalous messages
3. **Focal Loss**: Handles class imbalance
4. **Data Augmentation**: Shuffling, dropping, duplicating messages
5. **Special Tokens**: [MSG], [SPEAKER_0], [SPEAKER_1], [SEP], [MASK]

## Expected Training Time

- **GPU**: ~4-5 hours on NVIDIA A100/V100
- **CPU**: Not recommended (20+ hours)

## Output

After training completes, you'll find:

1. **Best Model**: `models_electra/best_model.pt`
2. **Training Logs**: `logs_electra/training_log_YYYYMMDD_HHMMSS.log`
3. **Checkpoints**: `checkpoints_electra/checkpoint_epoch_*_f1_*.pt`
4. **Final Results**: `logs_electra/final_results_YYYYMMDD_HHMMSS.json`

## Model Architecture

```
ConversationAnomalyDetector
├── ELECTRA Transformer (monologg/koelectra-base-v3-discriminator)
├── Attention Pooling Layer
└── Classification Head
    ├── Linear(768 → 768)
    ├── LayerNorm + ReLU + Dropout
    ├── Linear(768 → 384)
    ├── ReLU + Dropout
    └── Linear(384 → 2)
```

## Monitoring Training

```bash
# Watch training progress
tail -f logs_electra/training_log_*.log

# Check GPU usage
nvidia-smi -l 1
```

## Evaluation

After training, the model will be automatically evaluated on the test set. Results include:
- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- Confusion Matrix
- Example predictions

## Troubleshooting

### Out of Memory
```python
# Edit train.py, line 45
BATCH_SIZE = 8  # Reduce from 16
```

### Slow Training
```python
# Edit train.py, line 47
EPOCHS = 10  # Reduce from 20
```

### Model Not Converging
```python
# Edit train.py, line 48
LEARNING_RATE = 5e-5  # Increase from 2e-5
```

## Using the Trained Model

```python
from train import load_model_for_inference, predict_conversation, Config

# Load model
model, tokenizer = load_model_for_inference(
    'models_electra/best_model.pt',
    Config
)

# Predict on new conversation
messages = ["안녕하세요", "뭐하세요?", "그거 있어요?"]
predictions = predict_conversation(
    model, tokenizer, messages, Config.DEVICE
)

print(predictions)
```

## Comparison with Other Models

| Model | Parameters | Training Time | Training Method | Korean Support |
|-------|-----------|---------------|-----------------|----------------|
| BERT | ~110M | ~6 hours | MLM | Via KLUE |
| RoBERTa | ~125M | ~7 hours | MLM | Via KLUE |
| **ELECTRA** | **~110M** | **~4-5 hours** | **RTD** | **Native (KoELECTRA)** |
| DistilBERT | ~66M | ~5 hours | Distillation | Multilingual |

**RTD** = Replaced Token Detection (ELECTRA's training method)
**MLM** = Masked Language Modeling (BERT/RoBERTa's training method)

## Advantages of ELECTRA

### For Korean Language
- **Native Korean model**: Trained specifically on Korean corpus
- **Better Korean understanding**: Optimized for Korean morphology and syntax
- **Proven performance**: Widely used in Korean NLP tasks

### For Training Efficiency
- **Sample efficient**: Learns from all tokens, not just 15% masked tokens
- **Faster convergence**: Typically requires less training time
- **Better low-resource performance**: Works well even with limited data

### For Production
- **Similar size to BERT**: ~110M parameters
- **Comparable inference speed**: No significant overhead
- **Well-tested**: Battle-tested in many Korean NLP applications

## Known Advantages Over ALBERT

1. **No tokenizer issues**: ELECTRA uses WordPiece (like BERT), not SentencePiece
2. **Better Korean support**: Native Korean training vs. multilingual ALBERT
3. **More stable**: Well-established model with extensive use in production
4. **Better documentation**: Extensive Korean NLP community support

## References

- ELECTRA Paper: https://arxiv.org/abs/2003.10555
- KoELECTRA (monologg): https://github.com/monologg/KoELECTRA
- HuggingFace Model: https://huggingface.co/monologg/koelectra-base-v3-discriminator
