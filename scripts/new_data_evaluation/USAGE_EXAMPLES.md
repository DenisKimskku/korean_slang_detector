# Usage Examples - Messenger-Specific Evaluations

All evaluation scripts now support filtering by messenger platform using the `--messenger` flag.

## Available Messengers

- `band` - Band messenger conversations
- `facebook` - Facebook messenger conversations
- `instagram` - Instagram messenger conversations
- `nateon` - NateOn messenger conversations
- `all` - All messengers (default)

## Preprocessing Examples

### Preprocess All Messengers (Default)
```bash
python preprocess_new_data.py
# or explicitly
python preprocess_new_data.py --messenger all
```

### Preprocess Single Messenger
```bash
# Band only
python preprocess_new_data.py --messenger band

# Facebook only
python preprocess_new_data.py --messenger facebook

# Instagram only
python preprocess_new_data.py --messenger instagram

# NateOn only
python preprocess_new_data.py --messenger nateon
```

### Preprocess Multiple Messengers
```bash
# Band and Facebook
python preprocess_new_data.py --messenger band facebook

# Instagram and NateOn
python preprocess_new_data.py --messenger instagram nateon
```

## Fine-tuned Model Evaluation Examples

### Evaluate on All Messengers
```bash
python evaluate_models.py --model bert_base
python evaluate_models.py --model roberta_base
python evaluate_models.py --model roberta_large
```

### Evaluate on Single Messenger
```bash
# BERT-base on Band only
python evaluate_models.py --model bert_base --messenger band

# RoBERTa-base on Facebook only
python evaluate_models.py --model roberta_base --messenger facebook

# RoBERTa-large on Instagram only
python evaluate_models.py --model roberta_large --messenger instagram
```

### Evaluate on Multiple Messengers
```bash
# BERT-base on Band and Facebook
python evaluate_models.py --model bert_base --messenger band facebook

# RoBERTa-large on Instagram and NateOn
python evaluate_models.py --model roberta_large --messenger instagram nateon
```

## Plain Model Evaluation Examples

### Evaluate Pretrained Models on Specific Messengers
```bash
# BERT-base (no fine-tuning) on Band
python evaluate_plain.py --model bert_base --messenger band

# RoBERTa-base (no fine-tuning) on Facebook and Instagram
python evaluate_plain.py --model roberta_base --messenger facebook instagram

# RoBERTa-large (no fine-tuning) on all messengers
python evaluate_plain.py --model roberta_large --messenger all
```

## Gemini API Evaluation Examples

### Evaluate Gemini on Specific Messengers
```bash
# Gemini on Band only
python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger band

# Gemini on Facebook and Instagram
python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger facebook instagram --batch_size 16

# Gemini on all messengers (default)
python evaluate_gemini.py --model gemini-2.0-flash-exp
```

## OpenAI API Evaluation Examples

### Evaluate GPT models on Specific Messengers
```bash
# GPT-4o on Band only
python evaluate_openai.py --model gpt-4o --messenger band

# GPT-4o on Instagram and NateOn
python evaluate_openai.py --model gpt-4o --messenger instagram nateon --batch_size 16

# GPT-4o on all messengers (default)
python evaluate_openai.py --model gpt-4o
```

## Output Directory Structure

When using `--messenger` flag, results are saved to separate directories:

### All Messengers (Default)
```
results_bert_base/
logs_bert_base/
```

### Single Messenger
```
results_bert_base_band/
logs_bert_base_band/

results_roberta_base_facebook/
logs_roberta_base_facebook/
```

### Multiple Messengers
```
results_bert_base_band_facebook/
logs_bert_base_band_facebook/

results_roberta_large_instagram_nateon/
logs_roberta_large_instagram_nateon/
```

## Complete Workflow Example

### Scenario: Compare model performance on Band messenger only

```bash
# Step 1: Preprocess Band data
python preprocess_new_data.py --messenger band

# Step 2: Evaluate all fine-tuned models on Band
python evaluate_models.py --model bert_base --messenger band
python evaluate_models.py --model roberta_base --messenger band
python evaluate_models.py --model roberta_large --messenger band

# Step 3: Evaluate plain models on Band (for comparison)
python evaluate_plain.py --model bert_base --messenger band
python evaluate_plain.py --model roberta_base --messenger band
python evaluate_plain.py --model roberta_large --messenger band

# Step 4: Evaluate API models on Band
python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger band --batch_size 16
python evaluate_openai.py --model gpt-4o --messenger band --batch_size 16

# Step 5: Generate summary report
python generate_summary_report.py
```

### Scenario: Compare different messengers for single model

```bash
# Preprocess all messengers
python preprocess_new_data.py

# Evaluate BERT-base on each messenger separately
python evaluate_models.py --model bert_base --messenger band
python evaluate_models.py --model bert_base --messenger facebook
python evaluate_models.py --model bert_base --messenger instagram
python evaluate_models.py --model bert_base --messenger nateon

# Compare results in separate directories:
# - results_bert_base_band/
# - results_bert_base_facebook/
# - results_bert_base_instagram/
# - results_bert_base_nateon/
```

## Performance Considerations

### Memory Usage by Messenger

Approximate number of conversations (may vary):
- Band: ~1,000 conversations
- Facebook: ~4,500 conversations (largest)
- Instagram: ~3,000 conversations
- NateOn: ~900 conversations

**Recommendation:** If you have limited GPU memory, evaluate messengers separately:
```bash
# Instead of all at once
python evaluate_models.py --model roberta_large

# Run separately
python evaluate_models.py --model roberta_large --messenger band
python evaluate_models.py --model roberta_large --messenger facebook
python evaluate_models.py --model roberta_large --messenger instagram
python evaluate_models.py --model roberta_large --messenger nateon
```

### API Cost Optimization

For API models (Gemini/OpenAI), evaluating specific messengers can reduce costs:

```bash
# Test on smallest messenger first (Band)
python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger band

# If results look good, run on larger messengers
python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger facebook
```

## Parallel Execution

You can run multiple evaluations in parallel on different messengers:

```bash
# Terminal 1: Band
python evaluate_models.py --model bert_base --messenger band &

# Terminal 2: Facebook
python evaluate_models.py --model bert_base --messenger facebook &

# Terminal 3: Instagram
python evaluate_models.py --model bert_base --messenger instagram &

# Terminal 4: NateOn
python evaluate_models.py --model bert_base --messenger nateon &

wait  # Wait for all background jobs
```

## Tips

1. **Start Small**: Test with a single messenger (e.g., `band`) to verify everything works before running on all data.

2. **Incremental Evaluation**: Evaluate one messenger at a time if you need to stop/resume work.

3. **Comparison Studies**: Use messenger filtering to study how models perform on different conversation styles.

4. **Debug Faster**: When debugging, use a small messenger like `band` or `nateon` for faster iterations.

5. **Resource Management**: For limited GPU memory, process messengers separately rather than all at once.
