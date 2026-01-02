# ELECTRA and DistilBERT Integration

## Overview

Successfully integrated two additional transformer models into the evaluation pipeline:
1. **ELECTRA** (`monologg/koelectra-base-v3-discriminator`) - Korean ELECTRA model
2. **DistilBERT** (`bongsoo/mdistilbertV3.1`) - Multilingual DistilBERT model

**Note**: Originally planned to use ALBERT (`kykim/albert-kor-base`), but switched to ELECTRA due to tokenizer compatibility issues with the ALBERT model.

## What Was Done

### 1. Training Infrastructure Created

Two new training directories were set up with complete training scripts:

#### `/home/minseok/forensic/electra_base/`
- `train.py` - Training script for ELECTRA model
- `README.md` - Comprehensive training guide
- Output directories (created during training):
  - `models_electra/` - Saved models
  - `logs_electra/` - Training logs
  - `checkpoints_electra/` - Training checkpoints

#### `/home/minseok/forensic/distillbert_base/`
- `train.py` - Training script for DistilBERT model
- `README.md` - Comprehensive training guide
- Output directories (created during training):
  - `models_distilbert/` - Saved models
  - `logs_distilbert/` - Training logs
  - `checkpoints_distilbert/` - Training checkpoints

### 2. Evaluation Pipeline Integration

Updated all evaluation scripts to support ELECTRA and DistilBERT:

#### Modified Files

**evaluate_models.py** (lines 52-59):
- Added `electra_base` configuration pointing to `/home/minseok/forensic/electra_base/models_electra/best_model.pt`
- Added `distillbert_base` configuration pointing to `/home/minseok/forensic/distillbert_base/models_distilbert/best_model.pt`

**evaluate_plain.py** (lines 42-43):
- Added `electra_base: 'monologg/koelectra-base-v3-discriminator'`
- Added `distillbert_base: 'bongsoo/mdistilbertV3.1'`

**Shell Scripts** (all updated from 9 steps to 13 steps):
- `run_all_evaluations.sh` - Main evaluation script (15 steps total)
- `run_all_evaluations_nateon.sh` - NateOn-specific evaluations (13 steps)
- `run_all_evaluations_band.sh` - Band-specific evaluations (13 steps)
- `run_all_evaluations_facebook.sh` - Facebook-specific evaluations (13 steps)
- `run_all_evaluations_instagram.sh` - Instagram-specific evaluations (13 steps)

#### New Evaluation Steps

Each messenger-specific script now includes:

**Fine-tuned Model Evaluations (Steps 2-6):**
- Step 2: bert_base
- Step 3: roberta_base
- Step 4: roberta_large
- Step 5: **electra_base** (NEW)
- Step 6: **distillbert_base** (NEW)

**Plain Model Evaluations (Steps 7-11):**
- Step 7: plain bert_base
- Step 8: plain roberta_base
- Step 9: plain roberta_large
- Step 10: **plain electra_base** (NEW)
- Step 11: **plain distillbert_base** (NEW)

**API Evaluations (Steps 12-13):**
- Step 12: Gemini
- Step 13: OpenAI

## Training the New Models

### Prerequisites

```bash
# Activate ms conda environment
conda activate ms

# Ensure training data is available
ls /home/minseok/forensic/NIKL_MESSENGER_v2.0/modified/
```

### Training ELECTRA

```bash
cd /home/minseok/forensic/electra_base
python train.py
```

**Expected Training Time:** ~4-5 hours on NVIDIA A100/V100

**Key Features:**
- Model: `monologg/koelectra-base-v3-discriminator`
- Parameters: ~110M
- Architecture: Replaced Token Detection (RTD)
- Training config: 20 epochs, batch size 16, lr 2e-5
- **Korean-native**: Specifically trained for Korean language

### Training DistilBERT

```bash
cd /home/minseok/forensic/distillbert_base
python train.py
```

**Expected Training Time:** ~5-6 hours on NVIDIA A100/V100

**Key Features:**
- Model: `bongsoo/mdistilbertV3.1`
- Parameters: ~66M (40% smaller than BERT)
- Architecture: 6 transformer layers (vs 12 in BERT)
- Training config: 20 epochs, batch size 16, lr 2e-5

### Monitoring Training

```bash
# Watch ELECTRA training
tail -f /home/minseok/forensic/electra_base/logs_electra/training_log_*.log

# Watch DistilBERT training
tail -f /home/minseok/forensic/distillbert_base/logs_distilbert/training_log_*.log

# Check GPU usage
nvidia-smi -l 1
```

## Running Evaluations

### After Training Completes

Once both models are trained, you can evaluate them on the new data:

#### Option 1: Single Messenger (Testing)

```bash
cd /home/minseok/forensic/new_data_evaluation

# Test on NateOn (smallest dataset)
./run_all_evaluations_nateon.sh
```

This will automatically:
1. Preprocess NateOn data (using `forensic` environment)
2. Evaluate all 5 fine-tuned models: bert_base, roberta_base, roberta_large, **electra_base**, **distillbert_base**
3. Evaluate all 5 plain models: plain bert_base, plain roberta_base, plain roberta_large, **plain electra_base**, **plain distillbert_base**
4. Evaluate Gemini API
5. Evaluate OpenAI API

#### Option 2: All Messengers

```bash
cd /home/minseok/forensic/new_data_evaluation

# Run complete evaluation pipeline
./run_all_evaluations.sh
```

**Estimated Time:** ~4-5 hours for all messengers

### Expected Output Directories

After evaluation completes, results will be saved in:

```
results_electra_base/                    # Fine-tuned ELECTRA results
results_electra_base_nateon/             # Fine-tuned ELECTRA results (NateOn only)
results_electra_base_band/               # Fine-tuned ELECTRA results (Band only)
results_electra_base_facebook/           # Fine-tuned ELECTRA results (Facebook only)
results_electra_base_instagram/          # Fine-tuned ELECTRA results (Instagram only)

results_distillbert_base/                # Fine-tuned DistilBERT results
results_distillbert_base_nateon/         # Fine-tuned DistilBERT results (NateOn only)
results_distillbert_base_band/           # Fine-tuned DistilBERT results (Band only)
results_distillbert_base_facebook/       # Fine-tuned DistilBERT results (Facebook only)
results_distillbert_base_instagram/      # Fine-tuned DistilBERT results (Instagram only)

results_plain_electra_base/              # Plain ELECTRA results
results_plain_electra_base_nateon/       # Plain ELECTRA results (NateOn only)
results_plain_electra_base_band/         # Plain ELECTRA results (Band only)
results_plain_electra_base_facebook/     # Plain ELECTRA results (Facebook only)
results_plain_electra_base_instagram/    # Plain ELECTRA results (Instagram only)

results_plain_distillbert_base/          # Plain DistilBERT results
results_plain_distillbert_base_nateon/   # Plain DistilBERT results (NateOn only)
results_plain_distillbert_base_band/     # Plain DistilBERT results (Band only)
results_plain_distillbert_base_facebook/ # Plain DistilBERT results (Facebook only)
results_plain_distillbert_base_instagram/# Plain DistilBERT results (Instagram only)
```

Each directory contains:
- `evaluation_results_YYYYMMDD_HHMMSS.json` - Metrics (accuracy, F1, precision, recall, AUC-ROC)
- `detailed_predictions_YYYYMMDD_HHMMSS.json` - Per-sample predictions with probabilities

## Model Comparison

| Model | Parameters | Layers | Training Method | Korean Support | Key Advantage |
|-------|-----------|--------|-----------------|----------------|---------------|
| BERT | ~110M | 12 | MLM | Via KLUE | Standard baseline |
| RoBERTa | ~125M | 12 | MLM | Via KLUE | Optimized BERT |
| **ELECTRA** | **~110M** | **12** | **RTD** | **Native (KoELECTRA)** | **Korean-optimized, sample efficient** |
| **DistilBERT** | **~66M** | **6** | **Distillation** | **Multilingual** | **Faster inference (1.6x)** |

**RTD** = Replaced Token Detection (ELECTRA's training method)
**MLM** = Masked Language Modeling (BERT/RoBERTa's training method)

### ELECTRA Advantages
- **Korean-native**: Trained specifically on Korean corpus
- **Sample efficient**: Learns from all tokens via discriminative task
- **No tokenizer issues**: Uses WordPiece tokenizer (like BERT)
- **Well-tested**: Widely used in Korean NLP production systems
- **Proven performance**: State-of-the-art results on Korean benchmarks

### DistilBERT Advantages
- **Faster inference**: 60% faster than BERT
- **Smaller size**: 40% fewer parameters
- **Good performance**: Retains 97% of BERT's performance
- **Multilingual**: Supports multiple languages including Korean

## Why ELECTRA Instead of ALBERT?

**Original Plan**: Use ALBERT (`kykim/albert-kor-base`) for its parameter efficiency

**Issue Encountered**:
```python
TypeError: not a string
```
- ALBERT uses SentencePiece tokenizer which had compatibility issues
- The vocab file path was not being correctly passed as a string
- Multiple workarounds attempted (use_fast=False, cache clearing) but issue persisted

**Solution**: Switch to ELECTRA (`monologg/koelectra-base-v3-discriminator`)

**Benefits of the Switch**:
1. ✓ No tokenizer issues (uses WordPiece like BERT)
2. ✓ Better Korean support (KoELECTRA is the de facto standard for Korean)
3. ✓ More stable and well-tested
4. ✓ Similar training efficiency to ALBERT
5. ✓ Extensive Korean NLP community support

## Troubleshooting

### If Training Fails

#### Out of Memory
```bash
# Edit train.py in either electra_base/ or distillbert_base/
# Change BATCH_SIZE from 16 to 8
BATCH_SIZE = 8  # line 45
```

#### Model Not Converging
```bash
# Edit train.py
# Increase learning rate
LEARNING_RATE = 5e-5  # line 48 (from 2e-5)
```

### If Evaluation Fails

#### Model Not Found
- Ensure training completed successfully
- Check that `best_model.pt` exists in the respective `models_*` directory

#### Out of Memory During Evaluation
```bash
# Edit the shell script
# Change batch_size from 16 to 8
python evaluate_models.py --model electra_base --messenger nateon --batch_size 8
```

## Next Steps

1. **Train both models:**
   ```bash
   # Train ELECTRA (4-5 hours)
   cd /home/minseok/forensic/electra_base && conda activate ms && python train.py

   # Train DistilBERT (5-6 hours)
   cd /home/minseok/forensic/distillbert_base && conda activate ms && python train.py
   ```

2. **Run evaluations:**
   ```bash
   # Test on smallest dataset first
   cd /home/minseok/forensic/new_data_evaluation
   ./run_all_evaluations_nateon.sh
   ```

3. **Compare results:**
   - Compare ELECTRA vs BERT: Korean language handling, accuracy
   - Compare DistilBERT vs BERT: Inference speed, memory usage, accuracy
   - Analyze trade-offs between model size and performance

## Files Modified

### Training Files Created
- `/home/minseok/forensic/electra_base/train.py`
- `/home/minseok/forensic/electra_base/README.md`
- `/home/minseok/forensic/distillbert_base/train.py`
- `/home/minseok/forensic/distillbert_base/README.md`

### Evaluation Files Modified
- `/home/minseok/forensic/new_data_evaluation/evaluate_models.py` (lines 52-59)
- `/home/minseok/forensic/new_data_evaluation/evaluate_plain.py` (lines 42-43)
- `/home/minseok/forensic/new_data_evaluation/run_all_evaluations.sh`
- `/home/minseok/forensic/new_data_evaluation/run_all_evaluations_nateon.sh`
- `/home/minseok/forensic/new_data_evaluation/run_all_evaluations_band.sh`
- `/home/minseok/forensic/new_data_evaluation/run_all_evaluations_facebook.sh`
- `/home/minseok/forensic/new_data_evaluation/run_all_evaluations_instagram.sh`

## Documentation
- See `QUICK_START.md` for general evaluation instructions
- See `electra_base/README.md` for ELECTRA-specific details
- See `distillbert_base/README.md` for DistilBERT-specific details

## References

### ELECTRA
- Paper: https://arxiv.org/abs/2003.10555
- KoELECTRA: https://github.com/monologg/KoELECTRA
- Model: https://huggingface.co/monologg/koelectra-base-v3-discriminator

### DistilBERT
- Paper: https://arxiv.org/abs/1910.01108
- Model: https://huggingface.co/bongsoo/mdistilbertV3.1
