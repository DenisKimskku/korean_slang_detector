# Quick Start Guide - XAI Analysis

## Ready to Run! ✅

All dependencies are installed and the script is ready to execute.

## Three Ways to Run

### Option 1: Using the launcher script (Easiest)
```bash
cd /home/minseok/forensic
./run_xai_analysis.sh
```

### Option 2: Direct Python execution
```bash
conda activate forensic
cd /home/minseok/forensic
python xai_analysis.py
```

### Option 3: Using conda run
```bash
conda run -n forensic python /home/minseok/forensic/xai_analysis.py
```

## What Will Happen

The script will:
1. **Load test data** from `test_gpt.json` (50 conversations)
2. **For each of 5 models:**
   - Load NAIVE (base) model
   - Compute SHAP values (100 samples)
   - Compute saliency maps (all utterances)
   - Generate visualizations
   - Load TRAINED (fine-tuned) model
   - Repeat analysis
   - Create comparison plots

3. **Generate outputs:**
   - KDE plots comparing label 0 vs label 1
   - KDE plots comparing naive vs trained
   - HTML visualizations per conversation
   - Statistics JSON files
   - Summary comparison plots

## Models Analyzed

1. ✅ bert_base
2. ✅ distillbert_base
3. ✅ electra_base
4. ✅ roberta_base
5. ✅ roberta_large

## Expected Runtime

- **Per model:** ~30-60 minutes (with GPU)
- **All 5 models:** ~2.5-5 hours
- **Memory usage:** ~8-16 GB GPU RAM

## Output Structure

```
xai_results/
├── bert_base/
│   ├── naive/
│   │   ├── saliency_kde.png          ← Label 0 vs Label 1 saliency
│   │   ├── shap_kde.png              ← Label 0 vs Label 1 SHAP
│   │   ├── statistics.json           ← Numerical statistics
│   │   └── conversations/            ← HTML visualizations
│   │       ├── MDRW1900000001.html   ← Interactive saliency map
│   │       └── ...
│   └── trained/
│       └── (same structure)
├── distillbert_base/
├── electra_base/
├── roberta_base/
├── roberta_large/
├── bert_base_naive_vs_trained.png    ← 2x2 comparison plot
├── distillbert_base_naive_vs_trained.png
├── electra_base_naive_vs_trained.png
├── roberta_base_naive_vs_trained.png
├── roberta_large_naive_vs_trained.png
└── summary_all_models.png             ← All models overview
```

## Customization Options

Edit `xai_analysis.py` to adjust:

```python
# Line 58: Number of conversations to analyze
max_conversations = 50  # Reduce for faster testing

# Line 128: SHAP sample size
max_samples = 100  # Reduce if running out of memory

# Line 48: Output directory
OUTPUT_DIR = '/home/minseok/forensic/xai_results'

# Line 51: Input data
DATA_PATH = '/home/minseok/forensic/test_gpt.json'
```

## Quick Test Run

To test with just 1 model and 10 conversations:

```python
# Edit xai_analysis.py line 529:
data = load_test_data(DATA_PATH, max_conversations=10)

# Edit lines 534-552 to comment out models you don't want:
for model_name, config in MODEL_CONFIGS.items():
    if model_name != 'bert_base':  # Only run bert_base
        continue
    # ... rest of code
```

## What to Look For in Results

### 1. Saliency KDE Plots
- **Good:** Clear separation between label 0 (blue) and label 1 (orange)
- **Bad:** Overlapping distributions

### 2. Naive vs Trained Comparison
- **Good:** Trained model shows:
  - Higher saliency on drug terms for label 1
  - More focused attention (less spread)
  - Better separation between labels
- **Bad:** No difference between naive and trained

### 3. HTML Visualizations
- Open in browser to see token-level importance
- Red intensity = saliency strength
- Check if drug terms are highlighted in slang utterances

### 4. Statistics JSON
```json
{
  "label_1_saliency_mean": 5.67,  // Higher = more attention to slang
  "label_0_saliency_mean": 3.45,  // Lower = less attention to non-slang
  // Bigger difference = better model
}
```

## Troubleshooting

### Out of Memory
```python
# Reduce conversations
max_conversations = 20

# Reduce SHAP samples
max_samples = 50

# Or use CPU instead of GPU
device = torch.device('cpu')
```

### Too Slow
- Use GPU (CUDA is detected and working!)
- Reduce max_conversations
- Comment out unwanted models

### Import Errors
```bash
conda activate forensic
pip install -r requirements_xai.txt
```

## Next Steps After Analysis

1. **Open HTML files** to visually inspect saliency
2. **Compare KDE plots** across models
3. **Read statistics.json** for quantitative comparison
4. **Check naive_vs_trained.png** to see training impact
5. **Use results** for your paper/analysis

## Getting Help

- Check `XAI_README.md` for detailed documentation
- Review `xai_analysis.py` code comments
- Test with small subset first (10 conversations, 1 model)

---

## Ready? Let's Go!

```bash
./run_xai_analysis.sh
```

Monitor progress in terminal. Results will be saved to `xai_results/` directory.

Good luck! 🚀
