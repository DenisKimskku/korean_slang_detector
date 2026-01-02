# Environment Setup for New Data Evaluation

## Overview

The evaluation pipeline uses **two different conda environments** for different stages:

1. **`forensic` environment** - For preprocessing (Kkma/konlpy)
2. **`ms` environment** - For model evaluation

## Environment Usage

### 1. Preprocessing Stage (`forensic` environment)

**Used for**: `preprocess_new_data.py`

**Why**: Requires Kkma (Korean morphological analyzer) which has specific Java/JVM dependencies that work properly in the `forensic` environment.

**Activation**:
```bash
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate forensic
python preprocess_new_data.py --messenger nateon
```

**Dependencies**:
- `konlpy` with Kkma
- `sklearn` (TfidfVectorizer)
- `pandas`
- `tqdm`

### 2. Evaluation Stage (`ms` environment)

**Used for**: All evaluation scripts
- `evaluate_models.py` (fine-tuned BERT/RoBERTa models)
- `evaluate_plain.py` (pretrained models)
- `evaluate_gemini.py` (Gemini API)
- `evaluate_openai.py` (OpenAI API)
- `generate_summary_report.py`

**Why**: Contains PyTorch, transformers, and other ML libraries needed for model evaluation.

**Activation**:
```bash
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms
python evaluate_models.py --model bert_base --messenger nateon
```

**Dependencies**:
- `torch`
- `transformers`
- `google-generativeai`
- `openai`
- `scikit-learn`
- `tqdm`

## Shell Scripts Workflow

All evaluation shell scripts follow this two-stage pattern:

### Stage 1: Preprocessing (forensic environment)
```bash
# Step 1: Preprocess data (requires forensic conda environment for Kkma)
echo "[Step 1/9] Preprocessing data with TF-IDF drug keyword substitution..."
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate forensic 2>/dev/null
python preprocess_new_data.py --messenger $MESSENGER
PREPROCESS_STATUS=$?
conda deactivate 2>/dev/null

if [ $PREPROCESS_STATUS -eq 0 ]; then
    echo "✓ Preprocessing completed successfully"
else
    echo "✗ Preprocessing failed"
    exit 1
fi
echo ""
```

### Stage 2: Evaluation (ms environment)
```bash
# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

# Step 2-9: All evaluation steps run in ms environment
echo "[Step 2/9] Evaluating fine-tuned bert_base..."
python evaluate_models.py --model bert_base --messenger $MESSENGER

echo "[Step 3/9] Evaluating fine-tuned roberta_base..."
python evaluate_models.py --model roberta_base --messenger $MESSENGER

# ... etc
```

## Updated Shell Scripts

All shell scripts now use this two-environment pattern:

### Main Script
- ✅ `run_all_evaluations.sh`
  - Stage 1: `forensic` for preprocessing ALL messengers
  - Stage 2: `ms` for evaluating all models

### Messenger-Specific Scripts
- ✅ `run_all_evaluations_nateon.sh`
  - Stage 1: `forensic` for preprocessing NateOn
  - Stage 2: `ms` for evaluation

- ✅ `run_all_evaluations_band.sh`
  - Stage 1: `forensic` for preprocessing Band
  - Stage 2: `ms` for evaluation

- ✅ `run_all_evaluations_facebook.sh`
  - Stage 1: `forensic` for preprocessing Facebook
  - Stage 2: `ms` for evaluation

- ✅ `run_all_evaluations_instagram.sh`
  - Stage 1: `forensic` for preprocessing Instagram
  - Stage 2: `ms` for evaluation

## Manual Usage

### Option 1: Use Shell Scripts (Recommended)
```bash
# Everything is handled automatically
./run_all_evaluations_nateon.sh
```

### Option 2: Manual Step-by-Step

```bash
# Step 1: Preprocessing with forensic environment
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate forensic
python preprocess_new_data.py --messenger nateon
conda deactivate

# Step 2: Evaluation with ms environment
conda activate ms
python evaluate_models.py --model bert_base --messenger nateon
python evaluate_models.py --model roberta_base --messenger nateon
python evaluate_models.py --model roberta_large --messenger nateon
python evaluate_plain.py --model bert_base --messenger nateon
python evaluate_plain.py --model roberta_base --messenger nateon
python evaluate_plain.py --model roberta_large --messenger nateon
python evaluate_gemini.py --model gemini-2.0-flash --messenger nateon --batch_size 16
python evaluate_openai.py --model gpt-4o --messenger nateon --batch_size 16
conda deactivate
```

## Troubleshooting

### Issue: Kkma not working during preprocessing
**Solution**: Make sure you're using the `forensic` environment:
```bash
conda activate forensic
python -c "from konlpy.tag import Kkma; k = Kkma(); print(k.nouns('테스트'))"
```

### Issue: PyTorch/transformers errors during evaluation
**Solution**: Make sure you're using the `ms` environment:
```bash
conda activate ms
python -c "import torch; import transformers; print('OK')"
```

### Issue: Wrong environment activated
**Solution**: Check which environment is active:
```bash
conda info --envs
# The active environment will have a * next to it

# Or check programmatically
echo $CONDA_DEFAULT_ENV
```

## Summary

| Stage | Environment | Scripts | Purpose |
|-------|-------------|---------|---------|
| **Preprocessing** | `forensic` | `preprocess_new_data.py` | TF-IDF + Kkma morphological analysis |
| **Evaluation** | `ms` | `evaluate_*.py`, `generate_summary_report.py` | Model inference and metrics |

This two-environment setup ensures:
- ✅ Kkma works properly with correct Java/JVM dependencies
- ✅ PyTorch and transformers work in a clean environment
- ✅ No dependency conflicts between konlpy and ML libraries
- ✅ Automatic environment switching in shell scripts
