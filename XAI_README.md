# XAI Analysis for Drug Slang Detection Models

## Overview

This script performs comprehensive Explainable AI (XAI) analysis on drug slang detection models using:
1. **SHAP (Shapley Additive Explanations)** - Feature importance analysis
2. **Saliency Maps** - Gradient-based token importance visualization

The analysis compares:
- **Naive (base) models** vs **Trained (fine-tuned) models**
- **Label 0 (Non-slang)** vs **Label 1 (Slang)** utterances

## Features

### 1. SHAP Analysis
- Computes SHAP-like values using gradient-based approximation
- Uses gradient × embedding method (similar to Integrated Gradients)
- Shows which words contribute most to the model's decision
- Compares importance distributions between labels and model types
- Note: Uses gradient-based approximation instead of true Shapley values for better compatibility with transformers

### 2. Saliency Analysis
- Gradient-based visualization of token importance
- Shows which tokens the model "focuses on" when making predictions
- Normalized to 0-10 scale for easy comparison

### 3. Visualizations

#### KDE Distribution Plots
- Compare saliency/SHAP distributions between:
  - Non-slang (Label 0) vs Slang (Label 1)
  - Naive model vs Trained model

#### HTML Visualizations
- Per-conversation HTML files showing:
  - Token-level saliency heatmaps (red intensity = importance)
  - True labels vs predictions
  - Interactive visualizations

#### Comparison Plots
- 2x2 grid comparing naive vs trained for each model:
  - Saliency (Non-slang)
  - Saliency (Slang)
  - SHAP (Non-slang)
  - SHAP (Slang)

## Installation

### Required Packages
```bash
pip install torch transformers shap numpy matplotlib seaborn pandas jinja2 tqdm
```

## Usage

### Basic Usage
```bash
python xai_analysis.py
```

### Configuration

Edit these variables in the script:

```python
# Number of conversations to analyze
max_conversations = 50  # Reduce for faster testing

# Output directory
OUTPUT_DIR = '/home/minseok/forensic/xai_results'

# Data path
DATA_PATH = '/home/minseok/forensic/test_gpt.json'

# Max samples for SHAP (SHAP is computationally expensive)
max_samples = 100
```

## Output Structure

```
xai_results/
├── bert_base/
│   ├── naive/
│   │   ├── saliency_kde.png
│   │   ├── shap_kde.png
│   │   ├── statistics.json
│   │   └── conversations/
│   │       ├── MDRW1900000001.html
│   │       ├── MDRW1900000002.html
│   │       └── ...
│   └── trained/
│       ├── saliency_kde.png
│       ├── shap_kde.png
│       ├── statistics.json
│       └── conversations/
│           └── ...
├── distillbert_base/
│   └── ...
├── electra_base/
│   └── ...
├── roberta_base/
│   └── ...
├── roberta_large/
│   └── ...
├── bert_base_naive_vs_trained.png
├── distillbert_base_naive_vs_trained.png
├── electra_base_naive_vs_trained.png
├── roberta_base_naive_vs_trained.png
├── roberta_large_naive_vs_trained.png
└── summary_all_models.png
```

## Understanding the Results

### Saliency Maps
- **Higher values** (brighter red in HTML) = more important for the prediction
- **Label 0 (Non-slang)**: Expected to have lower saliency on drug terms
- **Label 1 (Slang)**: Expected to have higher saliency on drug terms

### SHAP Values
- **Absolute SHAP values** measure feature importance
- Higher values = token contributes more to the classification
- Positive/negative indicates direction of contribution

### KDE Plots
- Show the **distribution** of saliency/SHAP values
- Compare how the model's attention differs between:
  - Non-slang vs Slang utterances
  - Naive vs Trained models
- **Expected behavior**: Trained models should show clearer separation between label 0 and label 1

### Naive vs Trained Comparison
- **Naive model**: Base pretrained model without fine-tuning
- **Trained model**: Fine-tuned on drug slang detection task
- **Expected difference**: Trained model should:
  - Focus more on drug-related terms for slang utterances
  - Show clearer distinction between slang and non-slang

## Statistics JSON

Each model's statistics file contains:
```json
{
  "model_name": "bert_base",
  "model_type": "trained",
  "label_0_count": 1234,
  "label_1_count": 567,
  "label_0_saliency_mean": 3.45,
  "label_1_saliency_mean": 5.67,
  "label_0_saliency_std": 1.23,
  "label_1_saliency_std": 1.89,
  "label_0_shap_mean": 0.123,
  "label_1_shap_mean": 0.234,
  "label_0_shap_std": 0.056,
  "label_1_shap_std": 0.078
}
```

## HTML Visualizations

Open any HTML file in a browser to see:
- **Original text** with true label
- **Prediction** (correct/incorrect highlighted)
- **Token heatmap** (red intensity = saliency)
- Color coding:
  - Blue = Non-slang utterance
  - Orange = Slang utterance

## Performance Notes

- **SHAP computation** is expensive - limited to 100 samples by default
- **Saliency computation** is faster - runs on all data
- For faster testing, reduce `max_conversations`
- GPU recommended for faster processing
- Each model analysis takes approximately 10-30 minutes

## Interpreting Results

### What to look for:

1. **Well-trained models should show:**
   - Higher saliency on drug slang terms for label 1
   - Lower, more distributed saliency for label 0
   - Clear separation in KDE plots

2. **Naive vs Trained comparison:**
   - Trained models should focus more on relevant terms
   - Naive models may have random or unfocused attention

3. **HTML visualizations:**
   - Check if models highlight the right words
   - Verify that drug terms get high saliency in slang utterances
   - Look for prediction errors and their saliency patterns

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_conversations`
- Reduce `max_samples` for SHAP
- Reduce `BATCH_SIZE`
- Use CPU instead: `device = torch.device('cpu')`

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Model Checkpoint Not Found
- Verify checkpoint paths in `MODEL_CONFIGS`
- Check that models were trained and saved

## Citation

Based on gradient-based saliency methods from:
- Saliency reference: `/home/minseok/binshot_attk/saliency4.py`
- SHAP library: https://github.com/slundberg/shap
