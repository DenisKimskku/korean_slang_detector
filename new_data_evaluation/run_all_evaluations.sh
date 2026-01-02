#!/bin/bash

# Master script to run all evaluations on new_data (ALL MESSENGERS)
# This script will:
# 1. Preprocess the data (all messengers: band, facebook, instagram, nateon)
# 2. Evaluate the three fine-tuned models (bert_base, roberta_base, roberta_large)
# 3. Evaluate plain (pretrained) models
# 4. Evaluate Gemini model
# 5. Evaluate OpenAI model
#
# For messenger-specific evaluations, use:
#   - run_all_evaluations_band.sh
#   - run_all_evaluations_facebook.sh
#   - run_all_evaluations_instagram.sh
#   - run_all_evaluations_nateon.sh

set -e  # Exit on error

echo "========================================"
echo "New Data Evaluation Pipeline (ALL MESSENGERS)"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Check if preprocessing is needed
PREPROCESSED_FILE="preprocessed/all_conversations.json"

if [ -f "$PREPROCESSED_FILE" ]; then
    echo "[Step 1/15] Preprocessed data already exists at $PREPROCESSED_FILE"
    echo "✓ Skipping preprocessing step"
    echo ""
else
    echo "[Step 1/15] Preprocessing new_data with TF-IDF drug keyword substitution..."
    source /home/minseok/anaconda3/etc/profile.d/conda.sh
    conda activate forensic 2>/dev/null
    python preprocess_new_data.py
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
echo "[Step 2/15] Evaluating fine-tuned bert_base model..."
python evaluate_models.py --model bert_base
if [ $? -eq 0 ]; then
    echo "✓ bert_base evaluation completed"
else
    echo "✗ bert_base evaluation failed"
fi
echo ""

echo "[Step 3/15] Evaluating fine-tuned roberta_base model..."
python evaluate_models.py --model roberta_base
if [ $? -eq 0 ]; then
    echo "✓ roberta_base evaluation completed"
else
    echo "✗ roberta_base evaluation failed"
fi
echo ""

echo "[Step 4/15] Evaluating fine-tuned roberta_large model..."
python evaluate_models.py --model roberta_large
if [ $? -eq 0 ]; then
    echo "✓ roberta_large evaluation completed"
else
    echo "✗ roberta_large evaluation failed"
fi
echo ""

echo "[Step 5/15] Evaluating fine-tuned electra_base model..."
python evaluate_models.py --model electra_base
if [ $? -eq 0 ]; then
    echo "✓ electra_base evaluation completed"
else
    echo "✗ electra_base evaluation failed"
fi
echo ""

echo "[Step 6/15] Evaluating fine-tuned distillbert_base model..."
python evaluate_models.py --model distillbert_base
if [ $? -eq 0 ]; then
    echo "✓ distillbert_base evaluation completed"
else
    echo "✗ distillbert_base evaluation failed"
fi
echo ""

# Step 7-11: Evaluate plain (pretrained) models
echo "[Step 7/15] Evaluating plain bert_base model (no fine-tuning)..."
python evaluate_plain.py --model bert_base
if [ $? -eq 0 ]; then
    echo "✓ Plain bert_base evaluation completed"
else
    echo "✗ Plain bert_base evaluation failed"
fi
echo ""

echo "[Step 8/15] Evaluating plain roberta_base model (no fine-tuning)..."
python evaluate_plain.py --model roberta_base
if [ $? -eq 0 ]; then
    echo "✓ Plain roberta_base evaluation completed"
else
    echo "✗ Plain roberta_base evaluation failed"
fi
echo ""

echo "[Step 9/15] Evaluating plain roberta_large model (no fine-tuning)..."
python evaluate_plain.py --model roberta_large
if [ $? -eq 0 ]; then
    echo "✓ Plain roberta_large evaluation completed"
else
    echo "✗ Plain roberta_large evaluation failed"
fi
echo ""

echo "[Step 10/15] Evaluating plain electra_base model (no fine-tuning)..."
python evaluate_plain.py --model electra_base
if [ $? -eq 0 ]; then
    echo "✓ Plain electra_base evaluation completed"
else
    echo "✗ Plain electra_base evaluation failed"
fi
echo ""

echo "[Step 11/15] Evaluating plain distillbert_base model (no fine-tuning)..."
python evaluate_plain.py --model distillbert_base
if [ $? -eq 0 ]; then
    echo "✓ Plain distillbert_base evaluation completed"
else
    echo "✗ Plain distillbert_base evaluation failed"
fi
echo ""

# Step 12: Evaluate Gemini model
echo "[Step 12/15] Evaluating Gemini model (gemini-2.0-flash)..."
python evaluate_gemini.py --model gemini-2.0-flash --batch_size 16
if [ $? -eq 0 ]; then
    echo "✓ Gemini evaluation completed"
else
    echo "✗ Gemini evaluation failed"
fi
echo ""

# Step 13: Evaluate OpenAI model
echo "[Step 13/15] Evaluating OpenAI model (gpt-4o)..."
python evaluate_openai.py --model gpt-4o --batch_size 16
if [ $? -eq 0 ]; then
    echo "✓ OpenAI evaluation completed"
else
    echo "✗ OpenAI evaluation failed"
fi
echo ""

# Step 14: Generate summary report
echo "[Step 14/15] Generating summary report..."
python generate_summary_report.py
if [ $? -eq 0 ]; then
    echo "✓ Summary report generated"
else
    echo "⚠ Summary report generation failed (results are still available)"
fi
echo ""

echo "========================================"
echo "Evaluation Pipeline Completed!"
echo "========================================"
echo ""
echo "Results are saved in the following directories:"
echo "  - Preprocessed data: preprocessed/"
echo "  - Fine-tuned models: results_{bert_base,roberta_base,roberta_large,electra_base,distillbert_base}/"
echo "  - Plain models: results_plain_{bert_base,roberta_base,roberta_large,electra_base,distillbert_base}/"
echo "  - Gemini results: results_gemini_*/"
echo "  - OpenAI results: results_openai_*/"
echo ""
echo "Check the summary report in: summary_report.json"
