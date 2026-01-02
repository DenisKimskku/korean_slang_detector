# New Data Evaluation Pipeline

This directory contains scripts to evaluate multiple models on the new clean dataset.

## Overview

The evaluation pipeline tests the following models on clean conversation data (no drug-related content):

### Fine-tuned Models
1. **BERT-base** - Fine-tuned BERT model from `/home/minseok/forensic/bert_base/models_pure_lm_attn/`
2. **RoBERTa-base** - Fine-tuned RoBERTa model from `/home/minseok/forensic/roberta_base/models_pure_lm_attn/`
3. **RoBERTa-large** - Fine-tuned RoBERTa model from `/home/minseok/forensic/roberta_large/models_pure_lm_attn/`

### Baseline Models (No Fine-tuning)
4. **BERT-base (plain)** - Pretrained BERT without fine-tuning
5. **RoBERTa-base (plain)** - Pretrained RoBERTa without fine-tuning
6. **RoBERTa-large (plain)** - Pretrained RoBERTa without fine-tuning

### API Models
7. **Gemini** - Using gemini-2.0-flash-exp model
8. **OpenAI GPT-4o** - Using gpt-4o model

## Directory Structure

```
new_data_evaluation/
├── preprocess_new_data.py         # Convert .txt files to JSON format
├── evaluate_models.py              # Evaluate fine-tuned models
├── evaluate_plain.py               # Evaluate pretrained models
├── evaluate_gemini.py              # Evaluate Gemini API
├── evaluate_openai.py              # Evaluate OpenAI API
├── generate_summary_report.py      # Generate comparison report
├── run_all_evaluations.sh          # Master script to run everything
├── README.md                       # This file
├── preprocessed/                   # Preprocessed JSON data
├── results_*/                      # Result directories for each model
└── logs_*/                         # Log directories for each model
```

## Quick Start

### Run All Evaluations (Recommended)

To run the complete evaluation pipeline:

```bash
cd /home/minseok/forensic/new_data_evaluation
./run_all_evaluations.sh
```

This will:
1. Preprocess the new_data from .txt to JSON format
2. Evaluate all 3 fine-tuned models
3. Evaluate all 3 pretrained (plain) models
4. Evaluate Gemini model via API
5. Evaluate OpenAI GPT-4o via API
6. Generate a summary comparison report

**Estimated time:**
- Preprocessing: ~5 minutes
- Fine-tuned model evaluations: ~15 minutes each
- Plain model evaluations: ~15 minutes each
- API model evaluations: ~30-60 minutes each (depending on rate limits)
- **Total:** ~2-3 hours

### Run Individual Steps

#### 1. Preprocessing Only

```bash
python preprocess_new_data.py
```

Input: `/home/minseok/forensic/new_data/` (band/, facebook/, instagram/, nateon/)
Output: `preprocessed/all_conversations.json`

#### 2. Evaluate Specific Fine-tuned Model

```bash
# BERT-base
python evaluate_models.py --model bert_base

# RoBERTa-base
python evaluate_models.py --model roberta_base

# RoBERTa-large
python evaluate_models.py --model roberta_large
```

#### 3. Evaluate Specific Plain Model

```bash
# BERT-base (no fine-tuning)
python evaluate_plain.py --model bert_base

# RoBERTa-base (no fine-tuning)
python evaluate_plain.py --model roberta_base

# RoBERTa-large (no fine-tuning)
python evaluate_plain.py --model roberta_large
```

#### 4. Evaluate Gemini

```bash
# Default: gemini-2.0-flash-exp with batch_size=16
python evaluate_gemini.py --model gemini-2.0-flash-exp --batch_size 16

# Alternative models
python evaluate_gemini.py --model gemini-1.5-pro --batch_size 16
python evaluate_gemini.py --model gemini-1.5-flash --batch_size 16
```

#### 5. Evaluate OpenAI GPT

```bash
# Default: gpt-4o with batch_size=16
python evaluate_openai.py --model gpt-4o --batch_size 16

# Alternative models
python evaluate_openai.py --model gpt-4o-mini --batch_size 16
python evaluate_openai.py --model gpt-4-turbo --batch_size 16
```

#### 6. Generate Summary Report

```bash
python generate_summary_report.py
```

Output: `summary_report.json`

## Results

### Output Files

Each evaluation creates:
- **Results JSON** - `results_{model_name}/evaluation_results_*.json`
  - Contains metrics: accuracy, precision, recall, F1, AUC
  - Confusion matrix
  - Configuration used

- **Detailed Predictions** - `results_{model_name}/detailed_predictions_*.json`
  - Per-sample predictions
  - Confidence scores
  - Original messages

- **Logs** - `logs_{model_name}/evaluation_log_*.log`
  - Detailed execution logs
  - Error messages

### Summary Report

The `summary_report.json` provides:
- Side-by-side comparison of all models
- Key metrics for each model
- Best performing models by different criteria

### Interpretation

Since the new_data contains **clean conversations** (no drug-related content):

**Expected Results:**
- **High Specificity** - Models should correctly identify conversations as clean
- **Low False Positive Rate (FPR)** - Few false alarms
- **High True Negative (TN) count** - Most samples correctly classified as negative

**Key Metrics to Watch:**
1. **Specificity** = TN / (TN + FP) - Should be high (~0.95+)
2. **False Positive Rate** = 1 - Specificity - Should be low (~0.05-)
3. **Accuracy** - Overall correctness
4. **Predicted Positive Rate** - How many samples were flagged (should be low for clean data)

## Configuration

### Model Paths
- BERT-base: `/home/minseok/forensic/bert_base/models_pure_lm_attn/best_model.pt`
- RoBERTa-base: `/home/minseok/forensic/roberta_base/models_pure_lm_attn/best_model.pt`
- RoBERTa-large: `/home/minseok/forensic/roberta_large/models_pure_lm_attn/best_model.pt`

### Data Path
- Input: `/home/minseok/forensic/new_data/`
- Preprocessed: `./preprocessed/all_conversations.json`

### API Configuration
- Gemini: `/home/minseok/PoisonedRAG/model_configs/palm2_config.json`
- OpenAI: `/home/minseok/PoisonedRAG/model_configs/gpt4_config.json`

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Check that model paths exist
   - Ensure models were trained and saved correctly

2. **Preprocessed data not found**
   - Run `python preprocess_new_data.py` first

3. **API rate limits**
   - Reduce batch_size: `--batch_size 8`
   - Gemini and OpenAI scripts include retry logic and rate limiting

4. **Out of memory (GPU)**
   - Reduce batch_size in scripts
   - For plain models: `python evaluate_plain.py --model bert_base` (no --batch_size needed, uses config)
   - For fine-tuned models: Edit Config class in script

5. **CUDA out of memory**
   - Models will automatically fall back to CPU if CUDA unavailable
   - For large models (roberta_large), may need GPU with >16GB memory

### Logs

Check logs for detailed error messages:
```bash
# Latest log for bert_base
ls -lt logs_bert_base/

# View log
cat logs_bert_base/evaluation_log_*.log
```

## Customization

### Modify Evaluation Parameters

Edit the Config class in each script:
- `BATCH_SIZE` - Batch size for evaluation
- `MAX_CONTEXT_LENGTH` - Maximum sequence length
- `CONTEXT_WINDOW` - Number of context messages
- `WINDOW_SIZE` - Sliding window size for API models
- `STRIDE` - Stride for sliding window

### Add New Models

1. For transformer models:
   - Add model configuration to `evaluate_models.py` or `evaluate_plain.py`

2. For API models:
   - Create new script based on `evaluate_gemini.py` or `evaluate_openai.py`
   - Update `run_all_evaluations.sh`

## Performance Notes

### Speed
- Fine-tuned models: ~1000-2000 samples/minute (GPU)
- Plain models: ~1000-2000 samples/minute (GPU)
- Gemini: ~100-300 samples/minute (API rate limits)
- OpenAI: ~100-300 samples/minute (API rate limits)

### Accuracy vs Speed Trade-off
- **roberta_large**: Best accuracy, slowest
- **roberta_base**: Good balance
- **bert_base**: Fastest, lower accuracy
- **API models**: Very slow, quality varies

## Citation

If using this code, please cite:
```
[Your paper/project information]
```

## Contact

For questions or issues:
- Check logs first
- Review this README
- Contact: [Your contact information]

---

## Messenger Filtering (NEW)

All evaluation scripts now support filtering by messenger platform using the `--messenger` flag.

### Available Options
- `band` - Band messenger conversations
- `facebook` - Facebook messenger conversations  
- `instagram` - Instagram messenger conversations
- `nateon` - NateOn messenger conversations
- `all` - All messengers (default)

### Examples

```bash
# Preprocess only Band data
python preprocess_new_data.py --messenger band

# Evaluate BERT-base on Facebook only
python evaluate_models.py --model bert_base --messenger facebook

# Evaluate on multiple messengers
python evaluate_models.py --model roberta_base --messenger band facebook

# API models with messenger filtering
python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger instagram --batch_size 16
python evaluate_openai.py --model gpt-4o --messenger nateon --batch_size 16
```

For detailed examples and use cases, see [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md).

### Output Directory Naming

When using `--messenger` flag, results are saved with messenger suffix:
- Single messenger: `results_bert_base_band/`
- Multiple messengers: `results_bert_base_band_facebook/`
- All messengers: `results_bert_base/` (default)

