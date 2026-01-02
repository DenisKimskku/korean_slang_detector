#!/bin/bash

# Evaluation script for Instagram messenger only
# This script will run all evaluations on Instagram data

set -e  # Exit on error

MESSENGER="instagram"

echo "========================================"
echo "Instagram Messenger Evaluation Pipeline"
echo "========================================"
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Check if preprocessing is needed
PREPROCESSED_FILE="preprocessed/all_conversations.json"

if [ -f "$PREPROCESSED_FILE" ]; then
    echo "[Step 1/13] Preprocessed data already exists at $PREPROCESSED_FILE"
    echo "✓ Skipping preprocessing step"
    echo ""
else
    echo "[Step 1/13] Preprocessing data with TF-IDF drug keyword substitution..."
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

# Step 2-6: Evaluate fine-tuned models
echo "[Step 2/13] Evaluating fine-tuned bert_base on Instagram..."
python evaluate_models.py --model bert_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ bert_base evaluation completed"
else
    echo "✗ bert_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 3/13] Evaluating fine-tuned roberta_base on Instagram..."
python evaluate_models.py --model roberta_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ roberta_base evaluation completed"
else
    echo "✗ roberta_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 4/13] Evaluating fine-tuned roberta_large on Instagram..."
python evaluate_models.py --model roberta_large --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ roberta_large evaluation completed"
else
    echo "✗ roberta_large evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 5/13] Evaluating fine-tuned electra_base on Instagram..."
python evaluate_models.py --model electra_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ electra_base evaluation completed"
else
    echo "✗ electra_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 6/13] Evaluating fine-tuned distillbert_base on Instagram..."
python evaluate_models.py --model distillbert_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ distillbert_base evaluation completed"
else
    echo "✗ distillbert_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

# Step 7-11: Evaluate plain models
echo "[Step 7/13] Evaluating plain bert_base on Instagram..."
python evaluate_plain.py --model bert_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain bert_base evaluation completed"
else
    echo "✗ Plain bert_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 8/13] Evaluating plain roberta_base on Instagram..."
python evaluate_plain.py --model roberta_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain roberta_base evaluation completed"
else
    echo "✗ Plain roberta_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 9/13] Evaluating plain roberta_large on Instagram..."
python evaluate_plain.py --model roberta_large --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain roberta_large evaluation completed"
else
    echo "✗ Plain roberta_large evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 10/13] Evaluating plain electra_base on Instagram..."
python evaluate_plain.py --model electra_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain electra_base evaluation completed"
else
    echo "✗ Plain electra_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "[Step 11/13] Evaluating plain distillbert_base on Instagram..."
python evaluate_plain.py --model distillbert_base --messenger $MESSENGER
if [ $? -eq 0 ]; then
    echo "✓ Plain distillbert_base evaluation completed"
else
    echo "✗ Plain distillbert_base evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

# Step 12: Evaluate Gemini
echo "[Step 12/13] Evaluating Gemini on Instagram..."
python evaluate_gemini.py --model gemini-2.0-flash --messenger $MESSENGER --batch_size 16
if [ $? -eq 0 ]; then
    echo "✓ Gemini evaluation completed"
else
    echo "✗ Gemini evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

# Step 13: Evaluate OpenAI
echo "[Step 13/13] Evaluating OpenAI on Instagram..."
python evaluate_openai.py --model gpt-4o --messenger $MESSENGER --batch_size 16
if [ $? -eq 0 ]; then
    echo "✓ OpenAI evaluation completed"
else
    echo "✗ OpenAI evaluation failed"
fi
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null

echo "========================================"
echo "Instagram Evaluation Pipeline Completed!"
echo "========================================"
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null
echo "Results are saved in directories with '_instagram' suffix:"
echo "  - Fine-tuned models: results_*_instagram/"
echo "  - Plain models: results_plain_*_instagram/"
echo "  - Gemini results: results_gemini_*_instagram/"
echo "  - OpenAI results: results_openai_*_instagram/"
echo ""

# Activate ms environment for evaluation steps
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate ms 2>/dev/null
