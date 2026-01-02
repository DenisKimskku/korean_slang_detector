# Bug Fixes Applied

## Issue: Model Architecture Mismatch

### Problem
When evaluating fine-tuned models (bert_base, roberta_base, roberta_large), there was a state_dict size mismatch error:

```
size mismatch for classifier.0.weight: copying a param with shape torch.Size([768, 768]) 
from checkpoint, the shape in current model is torch.Size([768, 1536]).
```

### Root Cause
The model architecture in `evaluate_models.py` didn't match the architecture used during training (found in `legacy_msg.py`). The evaluation script was using:
- **Wrong:** Concatenated CLS + mean pooling → `[768 * 2 = 1536]` input to classifier
- **Correct:** Single attention-weighted pooling → `[768]` input to classifier

### Solution
Updated `PureLMModel` class in `evaluate_models.py` to match the `ConversationAnomalyDetector` architecture from `legacy_msg.py`:

1. **Changed pooling strategy:**
   - Old: Concatenated CLS and mean outputs
   - New: Attention-weighted pooling (or CLS only)

2. **Fixed classifier input:**
   - Old: `nn.Linear(hidden_size * 2, hidden_size)`  # 1536 → 768
   - New: `nn.Linear(hidden_size, hidden_size)`      # 768 → 768

3. **Updated attribute names:**
   - Old: `self.bert`
   - New: `self.transformer`

### Files Modified
- `/home/minseok/forensic/new_data_evaluation/evaluate_models.py`
  - Line 212-293: Complete model architecture rewrite
  - Line 460: Changed `model.bert` to `model.transformer`

## Issue: Embedding Size Mismatch

### Problem
After fixing the architecture, a new error appeared:
```
size mismatch for transformer.embeddings.word_embeddings.weight:
copying a param with shape torch.Size([32003, 768]) from checkpoint,
the shape in current model is torch.Size([32000, 768]).
```

### Root Cause
Two issues:
1. **Wrong special tokens:** Used `[CUR]`, `[/CUR]`, `[SEP]` but training used `[SEP]`, `[SPEAKER_0]`, `[SPEAKER_1]`, `[MASK]`, `[MSG]`
2. **Wrong order of operations:** Added special tokens AFTER loading model weights, causing size mismatch

### Solution
1. **Used correct special tokens** from training script (`legacy_msg.py`):
   - `[SEP]`, `[SPEAKER_0]`, `[SPEAKER_1]`, `[MASK]` (first call)
   - `[MSG]` (second call)
   - Only 3 new tokens added (BERT already has `[SEP]` and `[MASK]`)
   - Total vocabulary: 32003 (32000 base + 3 new)

2. **Reordered operations:**
   - Old order: Initialize model → Load weights → Add tokens
   - New order: Add tokens to tokenizer → Initialize model → Resize embeddings → Load weights

3. **Used `strict=False`** to ignore missing `position_ids` buffer (auto-generated)

### Files Modified
- `/home/minseok/forensic/new_data_evaluation/evaluate_models.py`
  - Lines 429-456: Reordered tokenizer/model initialization and special token handling

## Issue: Gemini Model Name Inconsistency

### Problem
Different Gemini model names used across files:
- Python default: `gemini-2.0-flash-exp`
- Shell scripts: `gemini-2.0-flash` (after user update)

### Solution
Standardized to `gemini-2.0-flash` across all files:
- Updated `evaluate_gemini.py` default model name
- All shell scripts now use consistent model name

### Files Modified
- `/home/minseok/forensic/new_data_evaluation/evaluate_gemini.py`
- All `run_all_evaluations*.sh` files

## Testing Recommendations

Before running full evaluation:

1. **Test with smallest dataset:**
   ```bash
   ./run_all_evaluations_nateon.sh
   ```

2. **Verify model loading:**
   - Check that no state_dict mismatch errors occur
   - Confirm model initializes correctly

3. **Check outputs:**
   - Ensure results are saved to correct directories
   - Verify metrics are calculated properly

## Expected Behavior

After fixes:
- ✅ Models load without state_dict errors
- ✅ Evaluation runs successfully on all messengers
- ✅ Results saved with appropriate naming (e.g., `results_bert_base_nateon/`)
- ✅ Consistent Gemini model usage across scripts

## Test Results

Successfully tested bert_base model on NateOn messenger (2025-11-12):
- ✅ Model loaded without errors (32003 token vocabulary)
- ✅ Evaluated 25,660 samples
- ✅ Results saved to `results_bert_base_nateon/`
- ✅ Accuracy: 90.75%
- ✅ No state_dict or embedding size errors

**Ready to run full evaluation pipeline on all messengers!**

## Environment Configuration

### Two-Stage Environment Setup

The evaluation pipeline now uses **two different conda environments**:

1. **Preprocessing Stage**: `forensic` environment
   - Used for: `preprocess_new_data.py`
   - Why: Kkma morphological analyzer requires specific Java/JVM setup
   - Dependencies: konlpy, sklearn, pandas

2. **Evaluation Stage**: `ms` environment
   - Used for: All `evaluate_*.py` scripts
   - Why: PyTorch, transformers, and API clients
   - Dependencies: torch, transformers, google-generativeai, openai

### Shell Script Flow

All shell scripts now follow this pattern:
```bash
# Stage 1: Preprocessing (forensic env)
conda activate forensic
python preprocess_new_data.py --messenger nateon
conda deactivate

# Stage 2: Evaluation (ms env)
conda activate ms
python evaluate_models.py --model bert_base --messenger nateon
python evaluate_plain.py --model bert_base --messenger nateon
# ... etc
```

### Updated Shell Scripts
- ✅ `run_all_evaluations.sh` - Two-stage environment switching
- ✅ `run_all_evaluations_nateon.sh` - Two-stage environment switching
- ✅ `run_all_evaluations_band.sh` - Two-stage environment switching
- ✅ `run_all_evaluations_facebook.sh` - Two-stage environment switching
- ✅ `run_all_evaluations_instagram.sh` - Two-stage environment switching

See `ENVIRONMENT_SETUP.md` for detailed documentation.
