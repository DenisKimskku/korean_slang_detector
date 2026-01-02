# Quick Start Guide - New Data Evaluation

## Running Evaluations

### Option 1: Single Messenger (Recommended for Testing)

```bash
cd /home/minseok/forensic/new_data_evaluation

# Test on smallest dataset (NateOn - ~900 conversations)
./run_all_evaluations_nateon.sh

# Or test on other messengers
./run_all_evaluations_band.sh       # ~1,000 conversations
./run_all_evaluations_facebook.sh   # ~4,500 conversations
./run_all_evaluations_instagram.sh  # ~3,000 conversations
```

### Option 2: All Messengers

```bash
cd /home/minseok/forensic/new_data_evaluation

# Run complete evaluation pipeline on all messengers
./run_all_evaluations.sh
```

## What Happens When You Run a Script?

Each script performs these steps automatically:

### Stage 1: Preprocessing (forensic environment)
- Activates `forensic` conda environment
- Runs `preprocess_new_data.py` with TF-IDF drug keyword substitution
- For each input conversation, generates:
  - 1 plain version (all label=0)
  - 1 drug-substituted version (with TF-IDF replacements, label=1 where replaced)
- Saves to `preprocessed/all_conversations.json`
- Deactivates forensic environment

### Stage 2: Evaluation (ms environment)
- Activates `ms` conda environment
- Evaluates **3 fine-tuned models**:
  - bert_base
  - roberta_base
  - roberta_large
- Evaluates **3 pretrained models** (no fine-tuning):
  - bert_base (plain)
  - roberta_base (plain)
  - roberta_large (plain)
- Evaluates **Gemini API**: gemini-2.0-flash
- Evaluates **OpenAI API**: gpt-4o
- (Optional) Generates summary report

## Expected Output

Results are saved in directories with messenger suffix:

```
results_bert_base_nateon/
  ├── evaluation_results_YYYYMMDD_HHMMSS.json
  └── detailed_predictions_YYYYMMDD_HHMMSS.json

results_roberta_base_nateon/
  └── ...

results_plain_bert_base_nateon/
  └── ...

results_gemini_2.0-flash_nateon/
  └── ...

results_openai_gpt-4o_nateon/
  └── ...
```

## Estimated Runtime

| Messenger | Files | Conversations | Est. Time |
|-----------|-------|---------------|-----------|
| NateOn | 1,606 | 3,212 | ~30 min |
| Band | ~1,000 | ~2,000 | ~20 min |
| Instagram | ~3,000 | ~6,000 | ~60 min |
| Facebook | ~4,500 | ~9,000 | ~90 min |
| **ALL** | ~10,106 | ~20,212 | ~3-4 hours |

*Preprocessing: ~5-10 min per messenger*
*Fine-tuned models: ~5-10 min each*
*API models: Depends on rate limits*

## Environment Requirements

### Automatic (via shell scripts)
No manual setup needed! Scripts handle environment switching:
- `forensic` for preprocessing
- `ms` for evaluation

### Manual (if running Python scripts directly)

**Preprocessing:**
```bash
conda activate forensic
python preprocess_new_data.py --messenger nateon
```

**Evaluation:**
```bash
conda activate ms
python evaluate_models.py --model bert_base --messenger nateon
```

## Monitoring Progress

### During Preprocessing
You'll see:
```
Initializing Korean morphological analyzer (Kkma)...
Kkma initialized successfully

Processing 1606 files from nateon...
Processing nateon:  50%|█████     | 803/1606 [01:10<01:00, 13.09it/s]
```

### During Evaluation
You'll see:
```
Loading tokenizer...
Added special tokens to tokenizer (total vocab: 32003)
Initializing model...
Resized model embeddings to 32003 tokens
Loading model weights from /home/minseok/forensic/bert_base/...
Model loaded successfully!

Evaluating:  50%|█████     | 800/1604 [01:30<01:30,  8.90it/s]
```

## Checking Results

```bash
# View evaluation metrics
cat results_bert_base_nateon/evaluation_results_*.json | python -m json.tool

# Count results by model type
ls -d results_*_nateon/ | wc -l

# Check preprocessing output
python -c "
import json
data = json.load(open('preprocessed/all_conversations.json'))
print(f'Total conversations: {len(data)}')
print(f'Plain: {len([c for c in data if c[\"type\"]==\"plain\"])}')
print(f'Drug substituted: {len([c for c in data if c[\"type\"]==\"drug_substituted\"])}')
"
```

## Troubleshooting

### Issue: Preprocessing fails with Kkma error
```bash
# Test Kkma in forensic environment
conda activate forensic
python -c "from konlpy.tag import Kkma; k = Kkma(); print('OK')"
```

### Issue: Model loading errors
```bash
# Check ms environment
conda activate ms
python -c "import torch; import transformers; print('OK')"
```

### Issue: Out of memory
```bash
# Reduce batch size in shell scripts
# Edit the script and change: --batch_size 16  →  --batch_size 8
```

### Issue: API rate limits (Gemini/OpenAI)
The scripts will continue even if API evaluations fail. Check the logs for specific errors.

## Next Steps After Evaluation

1. **Check metrics**: Review JSON files in `results_*/` directories
2. **Compare models**: Look at accuracy, precision, recall, F1 across models
3. **Analyze by type**: Compare performance on plain vs drug-substituted conversations
4. **Generate visualizations**: Use the detailed predictions for further analysis

## Files Generated

| File | Description |
|------|-------------|
| `preprocessed/all_conversations.json` | Preprocessed conversations (plain + drug-substituted) |
| `results_*/evaluation_results_*.json` | Metrics (accuracy, F1, etc.) |
| `results_*/detailed_predictions_*.json` | Per-sample predictions with probabilities |
| `summary_report.json` | (Optional) Aggregated results across all models |

## Quick Commands

```bash
# Run smallest test
./run_all_evaluations_nateon.sh

# Check if preprocessing worked
ls -lh preprocessed/all_conversations.json

# Count result directories
ls -d results_*/ | wc -l

# View latest results
find results_*/ -name "evaluation_results_*.json" -exec cat {} \; | python -m json.tool
```

## Documentation

- `FIXES.md` - Bug fixes and solutions applied
- `PREPROCESSING_README.md` - Detailed preprocessing documentation
- `ENVIRONMENT_SETUP.md` - Environment configuration details
- `README.md` - Original project overview
