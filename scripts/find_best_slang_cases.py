#!/usr/bin/env python
"""
Find the best cases of slang detection from XAI analysis
Analyzes saliency patterns to find where models correctly detect drug slang
"""

import json
import numpy as np
from pathlib import Path

def load_conversation_data(data_path):
    """Load the original conversation data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_best_cases(model_dir, data_path):
    """
    Find best slang detection cases based on:
    1. Correct predictions on slang utterances
    2. High saliency on drug-related terms
    3. Clear difference between slang and non-slang
    """

    # Load original data
    conversations = load_conversation_data(data_path)

    # Common drug slang terms to check
    drug_terms = [
        '픽업', '블루', '청포도', '핑크', '스위치', '배달', '물건', '떨',
        '대마', '마리화나', '히로뽕', '필로폰', '메스', '엑스터시', '코카인',
        '아이스', '얼음', '캔디', '케이', '고기', '허브', '쿠시', 'GHB', '물뽕'
    ]

    # Find statistics JSON
    stats_files = list(Path(model_dir).glob('**/statistics.json'))

    if not stats_files:
        print(f"No statistics found in {model_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Analyzing: {model_dir}")
    print(f"{'='*70}\n")

    # Load statistics
    for stats_file in stats_files:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        model_type = stats['model_type']
        print(f"{model_type.upper()} Model Statistics:")
        print(f"  Label 0 (non-slang): {stats['label_0_count']} utterances")
        print(f"    Saliency: μ={stats['label_0_saliency_mean']:.3f}, σ={stats['label_0_saliency_std']:.3f}")
        print(f"    SHAP:     μ={stats['label_0_shap_mean']:.4f}, σ={stats['label_0_shap_std']:.4f}")
        print(f"  Label 1 (slang):     {stats['label_1_count']} utterances")
        print(f"    Saliency: μ={stats['label_1_saliency_mean']:.3f}, σ={stats['label_1_saliency_std']:.3f}")
        print(f"    SHAP:     μ={stats['label_1_shap_mean']:.4f}, σ={stats['label_1_shap_std']:.4f}")

        # Calculate improvement metrics
        sal_diff = stats['label_1_saliency_mean'] - stats['label_0_saliency_mean']
        shap_diff = stats['label_1_shap_mean'] - stats['label_0_shap_mean']

        print(f"\n  Differences (Label 1 - Label 0):")
        print(f"    Saliency: {sal_diff:+.3f} {'✓ GOOD' if sal_diff > 0 else '✗ POOR'}")
        print(f"    SHAP:     {shap_diff:+.4f} {'✓ GOOD' if shap_diff > 0 else '✗ POOR'}")
        print()

def find_high_saliency_slang_examples(conversations, min_saliency=7.0):
    """
    Find specific utterances that are likely good examples
    Based on:
    - Label 1 (slang)
    - Contains drug-related terms
    """

    drug_terms = [
        '픽업', '블루', '청포도', '핑크', '스위치', '배달', '물건', '떨',
        '대마', '마리화나', '히로뽕', '필로폰', '메스', '엑스터시', '코카인',
        '아이스', '얼음', '캔디', '케이', '고기', '허브', '쿠시', 'GHB', '물뽕',
        '가루', '덱스', '똥', '작대기', '도리도리', '엠디엠에이', '엘에스디'
    ]

    good_examples = []

    for conv in conversations:
        conv_id = conv['id']
        for utt in conv['utterance']:
            text = utt['original_form']
            label = utt['label']

            # Look for slang utterances with drug terms
            if label == 1:
                contains_drug_term = any(term in text for term in drug_terms)
                if contains_drug_term:
                    good_examples.append({
                        'conv_id': conv_id,
                        'text': text,
                        'label': label,
                        'contains_drug_terms': True
                    })

    return good_examples

def main():
    """Main analysis"""

    # Paths
    data_path = '/home/minseok/forensic/drug_slang_conversations.json'
    xai_results_dir = '/home/minseok/forensic/xai_results'

    # Load conversations
    print("Loading conversation data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations")

    # Find good examples
    print("\nFinding best slang detection examples...")
    good_examples = find_high_saliency_slang_examples(conversations)

    print(f"\nFound {len(good_examples)} slang utterances with drug terms\n")

    # Show top examples
    print("="*70)
    print("TOP SLANG DETECTION CANDIDATES")
    print("="*70)
    print("\nThese utterances contain drug slang and are label 1:")
    print()

    for i, ex in enumerate(good_examples[:20], 1):
        print(f"{i}. Conversation: {ex['conv_id']}")
        print(f"   Text: {ex['text']}")
        print(f"   Label: {ex['label']} (slang)")
        print()

    # Analyze each model
    print("\n" + "="*70)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*70)

    model_dirs = [
        'bert_base',
        'electra_base',
        'roberta_base',
        'roberta_large'
    ]

    for model_name in model_dirs:
        model_dir = Path(xai_results_dir) / model_name
        if model_dir.exists():
            analyze_best_cases(model_dir, data_path)
        else:
            print(f"Skipping {model_name} (not found)")

    # Generate recommendation
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR BEST EXAMPLES")
    print("="*70)
    print("""
To find the BEST slang detection cases, look for HTML files where:

1. **High Saliency on Drug Terms**
   - Drug-related words (픽업, 블루, 물건, etc.) are highlighted in RED
   - Saliency values > 7.0 on these terms

2. **Correct Predictions**
   - True Label = 1 (Slang)
   - Prediction = 1 (Slang)
   - Green "correct" indicator

3. **Clear Difference from Non-Slang**
   - Slang utterances have focused attention on drug terms
   - Non-slang utterances have distributed/low attention

4. **Training Effect Visible**
   - Compare same conversation in naive vs trained folders
   - Trained model should show MORE focus on drug terms

BEST MODELS TO CHECK:
- Look at the statistics above
- Choose model with highest "Saliency Difference"
- Check trained/ folder for that model

TOP CONVERSATIONS TO EXAMINE:
    """)

    # List first few good examples
    for i, ex in enumerate(good_examples[:5], 1):
        print(f"{i}. {ex['conv_id']} - Contains: \"{ex['text'][:50]}...\"")

    print(f"\nCheck these HTML files in:")
    print(f"  {xai_results_dir}/[model_name]/trained/conversations/[conv_id].html")

if __name__ == "__main__":
    main()
