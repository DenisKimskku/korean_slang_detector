#!/usr/bin/env python
"""
Find conversations with biggest training improvements:
- High increase in saliency for slang utterances (label 1)
- High decline in saliency for non-slang utterances (label 0)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from bs4 import BeautifulSoup
from collections import defaultdict

def parse_html_conversation(html_content, conv_id):
    """Parse HTML to extract saliency values"""
    soup = BeautifulSoup(html_content, 'html.parser')
    utterances = []

    for div in soup.find_all('div', class_='utterance'):
        label_badge = div.find('span', class_='label-badge')
        if 'label-1-badge' in label_badge.get('class', []):
            label = 1
        else:
            label = 0

        # Get saliency from token backgrounds
        tokens_div = div.find('div', class_='tokens')
        saliency = []

        if tokens_div:
            for token_span in tokens_div.find_all('span', class_='token'):
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

        utterances.append({
            'label': label,
            'saliency': saliency
        })

    return {'conv_id': conv_id, 'utterances': utterances}

def compute_improvement_score(naive_conv, trained_conv):
    """
    Compute improvement score:
    - Positive: slang utterances got MORE salient (good)
    - Negative: non-slang utterances got LESS salient (good)
    """

    # Collect saliency by label
    naive_slang_sal = []
    naive_non_slang_sal = []
    trained_slang_sal = []
    trained_non_slang_sal = []

    for utt in naive_conv['utterances']:
        if utt['label'] == 1:
            naive_slang_sal.extend(utt['saliency'])
        else:
            naive_non_slang_sal.extend(utt['saliency'])

    for utt in trained_conv['utterances']:
        if utt['label'] == 1:
            trained_slang_sal.extend(utt['saliency'])
        else:
            trained_non_slang_sal.extend(utt['saliency'])

    # Compute means
    naive_slang_mean = np.mean(naive_slang_sal) if naive_slang_sal else 0
    naive_non_slang_mean = np.mean(naive_non_slang_sal) if naive_non_slang_sal else 0
    trained_slang_mean = np.mean(trained_slang_sal) if trained_slang_sal else 0
    trained_non_slang_mean = np.mean(trained_non_slang_sal) if trained_non_slang_sal else 0

    # Compute changes
    slang_increase = trained_slang_mean - naive_slang_mean  # Want positive
    non_slang_decrease = naive_non_slang_mean - trained_non_slang_mean  # Want positive (decrease means trained is lower)

    # Combined improvement score (both positive is good)
    improvement_score = slang_increase + non_slang_decrease

    return {
        'conv_id': naive_conv['conv_id'],
        'naive_slang_mean': naive_slang_mean,
        'naive_non_slang_mean': naive_non_slang_mean,
        'trained_slang_mean': trained_slang_mean,
        'trained_non_slang_mean': trained_non_slang_mean,
        'slang_increase': slang_increase,
        'non_slang_decrease': non_slang_decrease,
        'improvement_score': improvement_score
    }

def main():
    """Find and visualize training improvements"""

    xai_dir = Path('/home/minseok/forensic/xai_results')

    model_dirs = [
        'bert_base',
        'electra_base',
        'roberta_base',
        'roberta_large'
    ]

    all_improvements = []

    print("Analyzing training improvements...")
    print()

    for model_name in model_dirs:
        model_dir = xai_dir / model_name
        if not model_dir.exists():
            continue

        naive_html_dir = model_dir / 'naive' / 'conversations'
        trained_html_dir = model_dir / 'trained' / 'conversations'

        if not naive_html_dir.exists() or not trained_html_dir.exists():
            continue

        print(f"Processing {model_name}...")

        # Process each conversation
        for trained_html in sorted(trained_html_dir.glob('*.html')):
            conv_id = trained_html.stem
            naive_html = naive_html_dir / f"{conv_id}.html"

            if not naive_html.exists():
                continue

            try:
                # Parse both versions
                with open(naive_html, 'r', encoding='utf-8') as f:
                    naive_conv = parse_html_conversation(f.read(), conv_id)

                with open(trained_html, 'r', encoding='utf-8') as f:
                    trained_conv = parse_html_conversation(f.read(), conv_id)

                # Compute improvement
                improvement = compute_improvement_score(naive_conv, trained_conv)
                improvement['model'] = model_name
                all_improvements.append(improvement)

            except Exception as e:
                print(f"Error processing {conv_id}: {e}")
                continue

    # Sort by improvement score
    all_improvements.sort(key=lambda x: x['improvement_score'], reverse=True)

    # Print top 20
    print("\n" + "="*100)
    print("TOP 20 CONVERSATIONS WITH BIGGEST TRAINING IMPROVEMENTS")
    print("="*100)
    print(f"{'Rank':<5} {'Model':<16} {'Conv ID':<20} {'Slang↑':<10} {'Non-Slang↓':<12} {'Score':<10}")
    print("-"*100)

    for i, imp in enumerate(all_improvements[:20], 1):
        print(f"{i:<5} {imp['model']:<16} {imp['conv_id']:<20} "
              f"{imp['slang_increase']:+.3f}     {imp['non_slang_decrease']:+.3f}      "
              f"{imp['improvement_score']:+.3f}")

    # Create visualization
    print("\nCreating visualization...")

    top_20 = all_improvements[:20]

    # Prepare data
    conv_labels = [f"{imp['model'][:4]}\n{imp['conv_id'][-4:]}" for imp in top_20]
    slang_increases = [imp['slang_increase'] for imp in top_20]
    non_slang_decreases = [imp['non_slang_decrease'] for imp in top_20]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # Plot 1: Slang increase (higher is better)
    ax = axes[0]
    colors_slang = ['green' if x > 0 else 'red' for x in slang_increases]
    bars = ax.bar(range(20), slang_increases, color=colors_slang, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Saliency Increase (Slang)', fontsize=14, fontweight='bold')
    ax.set_title('Top 20: Training Effect on SLANG Utterances\n(Green = Increase ✓, Red = Decrease ✗)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(range(20))
    ax.set_xticklabels(conv_labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, slang_increases)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # Plot 2: Non-slang decrease (higher is better - means trained is lower)
    ax = axes[1]
    colors_non_slang = ['green' if x > 0 else 'red' for x in non_slang_decreases]
    bars = ax.bar(range(20), non_slang_decreases, color=colors_non_slang, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Saliency Decrease (Non-Slang)', fontsize=14, fontweight='bold')
    ax.set_title('Top 20: Training Effect on NON-SLANG Utterances\n(Green = Decrease ✓, Red = Increase ✗)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(range(20))
    ax.set_xticklabels(conv_labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Model & Conversation ID', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, non_slang_decreases)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = '/home/minseok/forensic/xai_results/top20_training_improvements.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Create scatter plot showing both dimensions
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot all points
    for imp in all_improvements:
        if imp['model'] == 'bert_base':
            color, marker = 'blue', 'o'
        elif imp['model'] == 'electra_base':
            color, marker = 'red', 's'
        elif imp['model'] == 'roberta_base':
            color, marker = 'green', '^'
        else:  # roberta_large
            color, marker = 'purple', 'D'

        ax.scatter(imp['slang_increase'], imp['non_slang_decrease'],
                  c=color, marker=marker, s=50, alpha=0.3)

    # Highlight top 20
    for i, imp in enumerate(top_20, 1):
        if imp['model'] == 'bert_base':
            color, marker = 'blue', 'o'
        elif imp['model'] == 'electra_base':
            color, marker = 'red', 's'
        elif imp['model'] == 'roberta_base':
            color, marker = 'green', '^'
        else:
            color, marker = 'purple', 'D'

        ax.scatter(imp['slang_increase'], imp['non_slang_decrease'],
                  c=color, marker=marker, s=300, alpha=0.8, edgecolors='black', linewidths=2)
        ax.annotate(f"{i}", (imp['slang_increase'], imp['non_slang_decrease']),
                   fontsize=8, fontweight='bold', ha='center', va='center')

    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Add quadrant labels
    ax.text(0.95, 0.95, 'BEST\n(↑slang, ↓non-slang)', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.05, 0.05, 'WORST\n(↓slang, ↑non-slang)', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='left', va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax.set_xlabel('Slang Saliency Increase (Trained - Naive)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Non-Slang Saliency Decrease (Naive - Trained)', fontsize=14, fontweight='bold')
    ax.set_title('Training Effect: Saliency Changes for Slang vs Non-Slang\nTop 20 Highlighted with Numbers',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='BERT'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='ELECTRA'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='RoBERTa-Base'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=10, label='RoBERTa-Large')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

    plt.tight_layout()
    scatter_path = '/home/minseok/forensic/xai_results/training_improvement_scatter.png'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {scatter_path}")

    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    for model_name in model_dirs:
        model_imps = [imp for imp in all_improvements if imp['model'] == model_name]
        if not model_imps:
            continue

        avg_slang_inc = np.mean([imp['slang_increase'] for imp in model_imps])
        avg_non_slang_dec = np.mean([imp['non_slang_decrease'] for imp in model_imps])
        avg_score = np.mean([imp['improvement_score'] for imp in model_imps])

        print(f"\n{model_name}:")
        print(f"  Avg Slang Increase:        {avg_slang_inc:+.3f}")
        print(f"  Avg Non-Slang Decrease:    {avg_non_slang_dec:+.3f}")
        print(f"  Avg Improvement Score:     {avg_score:+.3f}")

    print("\n" + "="*100)
    print("HTML FILES FOR TOP 20:")
    print("="*100)
    for i, imp in enumerate(top_20, 1):
        print(f"{i:2}. {imp['model']}/{imp['conv_id']}")
        print(f"    Naive:   /home/minseok/forensic/xai_results/{imp['model']}/naive/conversations/{imp['conv_id']}.html")
        print(f"    Trained: /home/minseok/forensic/xai_results/{imp['model']}/trained/conversations/{imp['conv_id']}.html")
        print()

if __name__ == "__main__":
    main()
