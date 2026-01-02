# Korean Slang Detector

This project is a drug slang detection system for Korean messenger conversations. It utilizes various Transformer-based models (BERT, RoBERTa, ELECTRA, etc.) to detect illicit drug-related slang in chat messages.

## Project Structure

The project is organized as follows:

- **`app/`**: Contains the main application and frontend.
  - `unified_demo.py`: The main FastAPI application serving the detection API.
  - `unified_frontend.html`: A simple HTML frontend for testing the API.
- **`scripts/`**: Contains utility and training scripts.
  - **`check_requirements.py`**: Verifies that all necessary Python packages are installed.
  - **`evaluate_models_v2.py`**: A comprehensive script for evaluating models using Perplexity and BLEU scores on the KorQuAD dataset.
  - **`xai_analysis.py`**: Performs Explainable AI (XAI) analysis to visualize model attention and feature importance.
  - **`run_xai_analysis.sh`**: A shell script to easily launch the XAI analysis.
  - **`train_*.py`**: Scripts for training different model architectures (e.g., `train_bert.py`, `train_roberta_large_v1.py`).
  - **`archive/`**: Contains older or redundant versions of scripts.
- **`data/`**: Stores datasets and configuration files.
  - `vocab.csv`, `poison.csv`, `noun.txt`: Data resources.
  - `test_gpt.json`, `test_claude.json`: Test datasets.
- **`models/`**: Stores model checkpoints, logs, and specific backend scripts for each architecture.
  - `bert_base/`, `roberta_base/`, `roberta_large/`, etc.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DenisKimskku/korean_slang_detector.git
    cd korean_slang_detector
    ```

2.  **Install Dependencies:**
    Run the requirements checker to see what is missing:
    ```bash
    python scripts/check_requirements.py
    ```
    Or install directly (ensure you have `torch`, `transformers`, `fastapi`, `uvicorn`, `shap`, `matplotlib`, `seaborn`, `pandas`, `jinja2`, `tqdm`):
    ```bash
    pip install torch transformers fastapi uvicorn shap matplotlib seaborn pandas jinja2 tqdm
    ```

## Usage

### Running the Demo Application

To start the Unified Drug Chat Detection API and Frontend:

1.  Navigate to the `app` directory:
    ```bash
    cd app
    ```
2.  Run the server:
    ```bash
    uvicorn unified_demo:app --reload --host 0.0.0.0 --port 8000
    ```
3.  Open your browser to `http://localhost:8000/unified_frontend.html` (or serve the HTML file separately if needed).

### Evaluation

To evaluate model performance (Perplexity and BLEU scores):

```bash
python scripts/evaluate_models_v2.py
```
*   This script compares "naive" (base) models against fine-tuned versions.
*   It generates an `evaluation_results.json` file.
*   **Metrics:**
    *   **Perplexity**: Lower is better (indicates better language understanding).
    *   **BLEU**: Higher is better (measures overlap with reference answers).

### Explainable AI (XAI) Analysis

To visualize and understand model decisions using SHAP and Saliency Maps:

1.  **Run the analysis:**
    ```bash
    ./scripts/run_xai_analysis.sh
    ```
    Or directly via Python:
    ```bash
    python scripts/xai_analysis.py
    ```

2.  **Output:**
    Results are saved in `xai_results/`. You will find:
    *   **KDE Plots**: Showing the distribution of attention (saliency) for Slang vs. Non-Slang.
    *   **HTML Visualizations**: Interactive per-conversation heatmaps showing which words triggered the model.
    *   **Comparison Plots**: Comparing Naive vs. Trained model focus.
    *   **Statistics JSON**: Quantitative data on model attention.

3.  **Interpretation:**
    *   **Good Performance**: Trained models should show higher saliency (red attention) on specific drug slang terms in "Label 1" (Slang) utterances compared to "Label 0" (Non-Slang).
    *   **Naive vs. Trained**: You should observe a clear shift where the trained model focuses more sharply on slang terms than the naive model.

## Models

The project supports and evaluates several Korean language models:
*   `klue/bert-base`
*   `klue/roberta-base`
*   `klue/roberta-large`
*   `monologg/distilkobert`
*   `monologg/koelectra-base-v3-discriminator`


## Citation


```bibtex
@article{ART003276332},
author={ 김민석 and 구형준 },
title={ LLM 기반 마약 은어 키워드 탐지 시스템 },
journal={ 정보보호학회논문지 },
issn={1598-3986},
year={2025},
number={6},
pages={1611 - 1625},
```

