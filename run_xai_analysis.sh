#!/bin/bash

# XAI Analysis Launcher Script
# This script runs the XAI analysis in the forensic conda environment

echo "========================================================================"
echo "XAI Analysis for Drug Slang Detection Models"
echo "========================================================================"
echo ""
echo "Starting analysis..."
echo "This will analyze all 5 models (bert, distilbert, electra, roberta_base, roberta_large)"
echo "Both NAIVE and TRAINED versions will be analyzed"
echo ""
echo "Estimated time: 30-60 minutes per model (depends on GPU)"
echo "Total: 2.5-5 hours for all models"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Activate conda environment and run
conda run -n forensic python /home/minseok/forensic/xai_analysis.py

echo ""
echo "========================================================================"
echo "Analysis complete!"
echo "Results saved to: /home/minseok/forensic/xai_results/"
echo "========================================================================"
echo ""
echo "To view results:"
echo "  - Open HTML files in a browser for visualizations"
echo "  - Check PNG files for KDE distribution plots"
echo "  - Read statistics.json for numerical results"
