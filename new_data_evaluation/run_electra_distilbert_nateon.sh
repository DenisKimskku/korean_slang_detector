#!/bin/bash

# Evaluation script for ELECTRA and DistilBERT only (NateOn messenger)
# This script will run evaluations on NateOn data for just the two new models

set -e  # Exit on error

MESSENGER="nateon"

echo "========================================"
echo "ELECTRA & DistilBERT - NateOn Evaluation"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Check if preprocessing is needed
PREPROCESSED_FILE="preprocessed/all_conversations.json"

if [ -f "$PREPROCESSED_FILE" ]; then
    echo "[Step 1/5] Preprocessed data already exists at $PREPROCESSED_FILE"
    echo "✓ Skipping preprocessing step"
    echo ""
else
    echo "[Step 1/5] Preprocessing NateOn data with TF-IDF drug keyword substitution..."
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
fi

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

# Step 2-3: Evaluate fine-tuned models
echo "[Step 2/5] Evaluating fine-tuned electra_base on NateOn..."
python evaluate_models.py --model electra_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ electra_base evaluation completed"
else
    echo "✗ electra_base evaluation failed"
fi
echo ""

echo "[Step 3/5] Evaluating fine-tuned distillbert_base on NateOn..."
python evaluate_models.py --model distillbert_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ distillbert_base evaluation completed"
else
    echo "✗ distillbert_base evaluation failed"
fi
echo ""

# Step 4-5: Evaluate plain (pretrained) models
echo "[Step 4/5] Evaluating plain electra_base on NateOn..."
python evaluate_plain.py --model electra_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain electra_base evaluation completed"
else
    echo "✗ Plain electra_base evaluation failed"
fi
echo ""

echo "[Step 5/5] Evaluating plain distillbert_base on NateOn..."
python evaluate_plain.py --model distillbert_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain distillbert_base evaluation completed"
else
    echo "✗ Plain distillbert_base evaluation failed"
fi
echo ""

echo "========================================"
echo "NateOn ELECTRA & DistilBERT Completed!"
echo "========================================"
echo ""
echo "Results are saved in directories with '_nateon' suffix:"
echo "  - Fine-tuned models: results_{electra_base,distillbert_base}_nateon/"
echo "  - Plain models: results_plain_{electra_base,distillbert_base}_nateon/"
echo ""
