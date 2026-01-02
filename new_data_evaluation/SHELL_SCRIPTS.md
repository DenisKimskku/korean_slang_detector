# Shell Script Reference

This directory contains multiple shell scripts for running evaluations on different messenger platforms.

## Available Scripts

### 1. `run_all_evaluations.sh` - ALL Messengers (Default)
Runs complete evaluation pipeline on **all messengers** (band, facebook, instagram, nateon).

```bash
./run_all_evaluations.sh
```

**Use when:** You want to evaluate on the complete dataset.

**Estimated time:** 2-3 hours (depending on GPU and API rate limits)

---

### 2. `run_all_evaluations_band.sh` - Band Messenger Only
Runs complete evaluation pipeline on **Band messenger only**.

```bash
./run_all_evaluations_band.sh
```

**Dataset size:** ~1,000 conversations (smallest)

**Use when:**
- Testing/debugging (fastest)
- Band-specific analysis
- Limited resources

**Estimated time:** 15-30 minutes

---

### 3. `run_all_evaluations_facebook.sh` - Facebook Messenger Only
Runs complete evaluation pipeline on **Facebook messenger only**.

```bash
./run_all_evaluations_facebook.sh
```

**Dataset size:** ~4,500 conversations (largest)

**Use when:**
- Facebook-specific analysis
- Have sufficient time and resources

**Estimated time:** 1-2 hours

---

### 4. `run_all_evaluations_instagram.sh` - Instagram Messenger Only
Runs complete evaluation pipeline on **Instagram messenger only**.

```bash
./run_all_evaluations_instagram.sh
```

**Dataset size:** ~3,000 conversations (medium-large)

**Use when:**
- Instagram-specific analysis
- Medium-sized dataset testing

**Estimated time:** 45-90 minutes

---

### 5. `run_all_evaluations_nateon.sh` - NateOn Messenger Only
Runs complete evaluation pipeline on **NateOn messenger only**.

```bash
./run_all_evaluations_nateon.sh
```

**Dataset size:** ~900 conversations (smallest)

**Use when:**
- Testing/debugging (very fast)
- NateOn-specific analysis
- Limited resources

**Estimated time:** 15-25 minutes

---

## What Each Script Does

All scripts follow the same pipeline:

1. **Preprocessing** - Convert .txt files to JSON format
2. **Fine-tuned Model Evaluation** - bert_base, roberta_base, roberta_large
3. **Plain Model Evaluation** - Pretrained models without fine-tuning
4. **Gemini Evaluation** - API-based evaluation with gemini-2.0-flash-exp
5. **OpenAI Evaluation** - API-based evaluation with gpt-4o

## Output Directory Structure

### All Messengers (run_all_evaluations.sh)
```
results_bert_base/
results_roberta_base/
results_roberta_large/
results_plain_bert_base/
results_plain_roberta_base/
results_plain_roberta_large/
results_gemini_*/
results_openai_*/
```

### Specific Messenger (e.g., run_all_evaluations_band.sh)
```
results_bert_base_band/
results_roberta_base_band/
results_roberta_large_band/
results_plain_bert_base_band/
results_plain_roberta_base_band/
results_plain_roberta_large_band/
results_gemini_*_band/
results_openai_*_band/
```

## Parallel Execution

You can run multiple messenger scripts in parallel:

```bash
# Run in separate terminals
./run_all_evaluations_band.sh      # Terminal 1
./run_all_evaluations_facebook.sh  # Terminal 2
./run_all_evaluations_instagram.sh # Terminal 3
./run_all_evaluations_nateon.sh    # Terminal 4
```

Or in background:

```bash
./run_all_evaluations_band.sh > band.log 2>&1 &
./run_all_evaluations_facebook.sh > facebook.log 2>&1 &
./run_all_evaluations_instagram.sh > instagram.log 2>&1 &
./run_all_evaluations_nateon.sh > nateon.log 2>&1 &

# Monitor progress
tail -f band.log
tail -f facebook.log
```

## Recommendations

### For Quick Testing
```bash
# Start with smallest dataset
./run_all_evaluations_nateon.sh
```

### For Production Runs
```bash
# Run on complete dataset
./run_all_evaluations.sh
```

### For Messenger Comparison
```bash
# Run each messenger separately to compare
./run_all_evaluations_band.sh
./run_all_evaluations_facebook.sh
./run_all_evaluations_instagram.sh
./run_all_evaluations_nateon.sh
```

### For Resource-Constrained Environments
```bash
# Run smallest messengers first
./run_all_evaluations_nateon.sh  # ~900 conversations
./run_all_evaluations_band.sh    # ~1,000 conversations
```

### For API Cost Management
```bash
# Test API models on small dataset first
./run_all_evaluations_nateon.sh  # Lowest API costs
# If results look good, run on larger datasets
./run_all_evaluations.sh         # Full dataset
```

## Error Handling

All scripts use `set -e` which means they will stop on first error. However:
- Preprocessing errors will stop the entire pipeline
- Individual model evaluation errors are logged but don't stop the pipeline
- You can check logs in `logs_*` directories for details

## Resuming After Interruption

If a script is interrupted, you can continue manually:

```bash
# If preprocessing completed but evaluations didn't
python evaluate_models.py --model bert_base --messenger band
python evaluate_models.py --model roberta_base --messenger band
# ... continue with remaining steps
```

## Customization

To modify what models are evaluated, edit the scripts:
- Comment out steps you don't need
- Change model names (e.g., `gemini-1.5-pro` instead of `gemini-2.0-flash-exp`)
- Adjust batch sizes for API models

Example:
```bash
# Edit run_all_evaluations_band.sh
# Comment out Gemini evaluation:
# echo "[Step 8/9] Evaluating Gemini on Band..."
# python evaluate_gemini.py --model gemini-2.0-flash-exp --messenger band --batch_size 16
```

## Performance Tips

1. **GPU Memory:** If you get out-of-memory errors, run smaller messengers or reduce batch sizes
2. **API Rate Limits:** Gemini/OpenAI scripts have built-in rate limiting and retry logic
3. **Time Estimation:**
   - NateOn/Band: 15-30 minutes each
   - Instagram: 45-90 minutes
   - Facebook: 1-2 hours
   - All messengers: 2-3 hours total

4. **Parallel Execution:** Run different messengers on different GPUs if available
5. **Sequential Execution:** If GPU memory is limited, run scripts one at a time
