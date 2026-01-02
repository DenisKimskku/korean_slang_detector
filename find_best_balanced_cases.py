#!/usr/bin/env python
"""
Find TRULY good slang detection cases where model discriminates well
Not just detecting slang, but also correctly rejecting non-slang
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_conversation_quality(conv_data):
    """
    Score a conversation based on:
    - Correct predictions on both slang and non-slang
    - High saliency on drug terms for slang
    - Low saliency for non-slang
    """

    slang_correct = 0
    slang_total = 0
    non_slang_correct = 0
    non_slang_total = 0

    slang_saliency = []
    non_slang_saliency = []

    drug_terms = [
        '픽업', '블루', '청포도', '핑크', '스위치', '배달', '물건', '떨',
        '케이', '케', '고기', '허브', '쿠시', 'GHB', '물뽕', '가루', '덱스', '똥',
        '아이스', '얼음', '캔디', '작대기', '도리도리'
    ]

    for utt in conv_data['utterances']:
        label = utt['label']
        pred = utt['prediction']
        saliency_vals = utt['saliency']

        if label == 1:  # Slang
            slang_total += 1
            if pred == 1:
                slang_correct += 1
                slang_saliency.extend(saliency_vals)
        else:  # Non-slang
            non_slang_total += 1
            if pred == 0:
                non_slang_correct += 1
                non_slang_saliency.extend(saliency_vals)

    # Calculate scores
    slang_acc = slang_correct / slang_total if slang_total > 0 else 0
    non_slang_acc = non_slang_correct / non_slang_total if non_slang_total > 0 else 0

    # Balanced accuracy
    balanced_acc = (slang_acc + non_slang_acc) / 2

    # Saliency difference (slang should have higher)
    avg_slang_sal = np.mean(slang_saliency) if slang_saliency else 0
    avg_non_slang_sal = np.mean(non_slang_saliency) if non_slang_saliency else 0
    sal_diff = avg_slang_sal - avg_non_slang_sal

    return {
        'conv_id': conv_data['conv_id'],
        'slang_acc': slang_acc,
        'non_slang_acc': non_slang_acc,
        'balanced_acc': balanced_acc,
        'slang_correct': slang_correct,
        'slang_total': slang_total,
        'non_slang_correct': non_slang_correct,
        'non_slang_total': non_slang_total,
        'avg_slang_saliency': avg_slang_sal,
        'avg_non_slang_saliency': avg_non_slang_sal,
        'saliency_diff': sal_diff,
        'quality_score': balanced_acc * (1 + sal_diff)  # Higher is better
    }

def load_conversation_results(model_dir, model_type='trained'):
    """Load all conversation results from JSON stats"""

    # We need to parse HTML files to get predictions
    html_dir = Path(model_dir) / model_type / 'conversations'

    if not html_dir.exists():
        return []

    conversations = []

    for html_file in sorted(html_dir.glob('*.html')):
        conv_id = html_file.stem

        # Parse HTML to extract data
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Extract conversation data from HTML
            conv_data = parse_html_conversation(html_content, conv_id)
            if conv_data:
                conversations.append(conv_data)
        except Exception as e:
            print(f"Error parsing {html_file}: {e}")
            continue

    return conversations

def parse_html_conversation(html_content, conv_id):
    """Parse HTML to extract utterances, predictions, and saliency"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, 'html.parser')
    utterances = []

    for div in soup.find_all('div', class_='utterance'):
        # Get label
        label_badge = div.find('span', class_='label-badge')
        if 'label-1-badge' in label_badge.get('class', []):
            label = 1
        else:
            label = 0

        # Get prediction
        pred_span = div.find('span', class_=['pred-correct', 'pred-incorrect'])
        pred_text = pred_span.text.strip()
        prediction = 1 if 'Slang' in pred_text and 'Non-Slang' not in pred_text else 0

        # Get text
        text_elem = div.find('strong', string='Text:')
        if text_elem:
            text = text_elem.next_sibling.strip()
        else:
            text = ""

        # Get saliency values from token backgrounds
        tokens_div = div.find('div', class_='tokens')
        saliency = []
        tokens = []

        if tokens_div:
            for token_span in tokens_div.find_all('span', class_='token'):
                # Extract saliency from rgba background
                style = token_span.get('style', '')
                if 'rgba(255, 0, 0,' in style:
                    alpha_str = style.split('rgba(255, 0, 0,')[1].split(')')[0].strip()
                    try:
                        alpha = float(alpha_str)
                        saliency.append(alpha * 10)  # Scale to 0-10
                    except:
                        saliency.append(0)
                else:
                    saliency.append(0)

                tokens.append(token_span.text.strip())

        utterances.append({
            'text': text,
            'label': label,
            'prediction': prediction,
            'tokens': tokens,
            'saliency': saliency
        })

    return {
        'conv_id': conv_id,
        'utterances': utterances
    }

def main():
    """Find best balanced cases"""

    xai_results_dir = '/home/minseok/forensic/xai_results'

    print("="*70)
    print("FINDING BEST BALANCED SLANG DETECTION CASES")
    print("="*70)
    print("\nA good case must:")
    print("  ✓ Correctly predict SLANG as slang")
    print("  ✓ Correctly predict NON-SLANG as non-slang")
    print("  ✓ Show higher saliency on slang utterances")
    print()

    # Try to import BeautifulSoup
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Installing beautifulsoup4...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'beautifulsoup4', '-q'])
        from bs4 import BeautifulSoup

    model_dirs = [
        'bert_base',
        'electra_base',
        'roberta_base',
        'roberta_large'
    ]

    all_model_results = {}

    for model_name in model_dirs:
        model_dir = Path(xai_results_dir) / model_name
        if not model_dir.exists():
            continue

        print(f"\nAnalyzing {model_name}...")

        # Load trained model results
        conversations = load_conversation_results(model_dir, 'trained')

        if not conversations:
            print(f"  No conversations found")
            continue

        # Score each conversation
        scored_convs = []
        for conv in conversations:
            score = analyze_conversation_quality(conv)
            scored_convs.append(score)

        # Sort by quality score
        scored_convs.sort(key=lambda x: x['quality_score'], reverse=True)

        all_model_results[model_name] = scored_convs

        # Show top 5
        print(f"\n  Top 5 best conversations for {model_name}:")
        print(f"  {'Rank':<5} {'Conv ID':<20} {'Balanced Acc':<15} {'Slang':<12} {'Non-Slang':<12} {'Sal Diff':<10}")
        print(f"  {'-'*75}")

        for i, conv in enumerate(scored_convs[:5], 1):
            slang_str = f"{conv['slang_correct']}/{conv['slang_total']}"
            non_slang_str = f"{conv['non_slang_correct']}/{conv['non_slang_total']}"

            print(f"  {i:<5} {conv['conv_id']:<20} {conv['balanced_acc']:.3f}           "
                  f"{slang_str:<12} {non_slang_str:<12} {conv['saliency_diff']:+.2f}")

    # Find overall best
    print("\n" + "="*70)
    print("OVERALL BEST CASES ACROSS ALL MODELS")
    print("="*70)

    all_cases = []
    for model_name, convs in all_model_results.items():
        for conv in convs:
            all_cases.append({
                'model': model_name,
                **conv
            })

    # Sort by quality
    all_cases.sort(key=lambda x: x['quality_score'], reverse=True)

    print(f"\n{'Rank':<5} {'Model':<15} {'Conv ID':<20} {'Bal Acc':<10} {'Slang':<10} {'Non-Slang':<12} {'Sal Diff':<10}")
    print(f"{'-'*90}")

    for i, case in enumerate(all_cases[:20], 1):
        slang_str = f"{case['slang_correct']}/{case['slang_total']}"
        non_slang_str = f"{case['non_slang_correct']}/{case['non_slang_total']}"

        print(f"{i:<5} {case['model']:<15} {case['conv_id']:<20} {case['balanced_acc']:.3f}      "
              f"{slang_str:<10} {non_slang_str:<12} {case['saliency_diff']:+.2f}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if all_cases:
        best = all_cases[0]
        print(f"\n🏆 BEST OVERALL CASE:")
        print(f"  Model: {best['model']}")
        print(f"  Conversation: {best['conv_id']}")
        print(f"  Balanced Accuracy: {best['balanced_acc']:.1%}")
        print(f"  Slang Correct: {best['slang_correct']}/{best['slang_total']}")
        print(f"  Non-Slang Correct: {best['non_slang_correct']}/{best['non_slang_total']}")
        print(f"  Saliency Difference: {best['saliency_diff']:+.2f}")
        print(f"\n  Open in browser:")
        print(f"  {xai_results_dir}/{best['model']}/trained/conversations/{best['conv_id']}.html")

        # Find cases with perfect accuracy
        perfect_cases = [c for c in all_cases if c['balanced_acc'] == 1.0]
        if perfect_cases:
            print(f"\n✨ {len(perfect_cases)} conversations with PERFECT accuracy (100%)!")
            print(f"\nTop 5 perfect cases:")
            for i, case in enumerate(perfect_cases[:5], 1):
                print(f"  {i}. {case['model']}/{case['conv_id']} - Saliency diff: {case['saliency_diff']:+.2f}")

if __name__ == "__main__":
    main()
