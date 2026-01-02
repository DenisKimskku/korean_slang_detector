# Model Evaluation Scripts

This directory contains scripts to evaluate the performance gap between naive (base) models and fine-tuned models using the KorQuAD v1.0 dataset.

## Files Created

1. **evaluate_models.py** - First version with direct QA model approach
2. **evaluate_models_v2.py** - Improved version with better checkpoint handling (RECOMMENDED)
3. **check_requirements.py** - Verify all required packages are installed

## Models Evaluated

The following Korean language models are evaluated:

- **bert_base**: klue/bert-base
- **distillbert_base**: monologg/distilkobert
- **electra_base**: monologg/koelectra-base-v3-discriminator
- **roberta_base**: klue/roberta-base
- **roberta_large**: klue/roberta-large

## Metrics

### Perplexity
- Measures how well the model predicts the text
- **Lower is better**
- Indicates model confidence and language understanding
- Negative change (Δ) means improvement

### BLEU Score
- Measures overlap between predicted and reference answers
- **Higher is better** (range: 0.0 to 1.0)
- Character-level matching for Korean text
- Positive change (Δ) means improvement

## Usage

### Step 1: Check Requirements

```bash
python check_requirements.py
```

This will verify that you have all necessary packages installed.

### Step 2: Run Evaluation (Recommended)

```bash
python evaluate_models_v2.py
```

The script will:
1. Load the KorQuAD dataset from `/home/minseok/rag_security/KorQuAD_v1.0_dev_cleansed.json`
2. Ask you how many samples to evaluate (recommended: 100-200 for quick results)
3. For each model:
   - Load the naive (base) model
   - Load the fine-tuned model from checkpoint
   - Calculate Perplexity and BLEU scores
   - Compare performance
4. Display a comprehensive comparison table
5. Save results to `evaluation_results.json`

### Step 3: Review Results

The script outputs a comparison table showing:
- Model name
- Perplexity (lower is better)
- BLEU score (higher is better)
- Performance improvements (Δ Perplexity and Δ BLEU)

Results are also saved to `evaluation_results.json` for further analysis.

## Expected Output Format

```
==================================================================================================
                                   EVALUATION RESULTS
==================================================================================================
Model                Type            Perplexity      BLEU         Δ Perplexity       Δ BLEU
--------------------------------------------------------------------------------------------------
bert_base            Naive           12.3456         0.2345       -                  -
bert_base            Fine-tuned      10.2345         0.3456       -2.1111 (-17.11%)  +0.1111 (+47.38%)
--------------------------------------------------------------------------------------------------
...
```

## Customization

### Adjust Sample Size

Edit the script or respond to the prompt to change the number of samples evaluated:
- **50-100**: Quick test (~5-10 minutes)
- **200-500**: Moderate evaluation (~20-30 minutes)
- **1000+**: Comprehensive evaluation (1+ hours)

### Modify Model Configurations

Edit the `MODEL_CONFIGS` dictionary in the script:

```python
MODEL_CONFIGS = {
    'your_model_name': {
        'name': 'huggingface/model-name',
        'checkpoint': '/path/to/best_model.pt'
    }
}
```

### Adjust Parameters

In the script header, you can modify:
- `MAX_LENGTH`: Maximum sequence length (default: 384)
- `BATCH_SIZE`: Batch size for evaluation (default: 8)
- `DEVICE`: Computation device (auto-detected: cuda/cpu)

## Understanding the Results

### Good Performance Indicators

For **fine-tuned** models compared to **naive** models:
- ✓ **Lower perplexity** → Model is more confident and better at language understanding
- ✓ **Higher BLEU score** → Model generates answers closer to reference answers
- ✓ **Negative Δ Perplexity** → Perplexity decreased (improved)
- ✓ **Positive Δ BLEU** → BLEU increased (improved)

### Example Interpretation

```
bert_base Fine-tuned: Perplexity 8.5, BLEU 0.42
bert_base Naive:      Perplexity 12.3, BLEU 0.28
```

This shows:
- Perplexity decreased by ~31% (good!)
- BLEU increased by ~50% (good!)
- Fine-tuning significantly improved performance

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:
1. Reduce the number of samples
2. Reduce `BATCH_SIZE` in the script
3. Reduce `MAX_LENGTH` in the script

### Missing Checkpoints

If a checkpoint file is not found:
- Verify the path in `MODEL_CONFIGS`
- The script will skip models with missing checkpoints
- Check that `best_model.pt` exists in the specified directory

### Import Errors

Run `check_requirements.py` to ensure all packages are installed:
```bash
pip install torch transformers numpy tqdm
```

## Notes

- The fine-tuned models in this directory were trained on sequence classification tasks
- The evaluation adapts them for question-answering evaluation on KorQuAD
- Results are saved in JSON format for further analysis
- GPU is recommended but CPU will work (slower)

## Dataset Information

- **Dataset**: KorQuAD v1.0 (Korean Question Answering Dataset)
- **Location**: `/home/minseok/rag_security/KorQuAD_v1.0_dev_cleansed.json`
- **Format**: Question-Answer pairs with context
- **Language**: Korean

## Contact & Support

If you encounter any issues:
1. Check the error messages carefully
2. Verify dataset and checkpoint paths
3. Review the requirements
4. Check GPU memory availability
