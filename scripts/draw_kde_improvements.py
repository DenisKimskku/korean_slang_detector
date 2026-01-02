#!/usr/bin/env python
"""
Create KDE plots for top 20 training improvement cases
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from bs4 import BeautifulSoup

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
    """Compute improvement score"""
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
    slang_increase = trained_slang_mean - naive_slang_mean
    non_slang_decrease = naive_non_slang_mean - trained_non_slang_mean
    improvement_score = slang_increase + non_slang_decrease

    return {
        'conv_id': naive_conv['conv_id'],
        'naive_slang_sal': naive_slang_sal,
        'naive_non_slang_sal': naive_non_slang_sal,
        'trained_slang_sal': trained_slang_sal,
        'trained_non_slang_sal': trained_non_slang_sal,
        'improvement_score': improvement_score,
        'slang_increase': slang_increase,
        'non_slang_decrease': non_slang_decrease
    }

def main():
    """Create KDE plots for top 20 improvements"""

    xai_dir = Path('/home/minseok/forensic/xai_results')

    model_dirs = [
        'bert_base',
        'electra_base',
        'roberta_base',
        'roberta_large'
    ]

    all_improvements = []

    print("Analyzing training improvements...")

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
                continue

    # Sort by improvement score
    all_improvements.sort(key=lambda x: x['improvement_score'], reverse=True)

    # Get top 20
    top_20 = all_improvements[:20]

    print(f"\nCreating KDE plots for top 20 conversations...")

    # Create KDE plots - 4x5 grid (20 subplots)
    fig, axes = plt.subplots(5, 4, figsize=(24, 30))
    axes = axes.flatten()

    palette = sns.color_palette("muted", 8)

    for idx, imp in enumerate(top_20):
        ax = axes[idx]

        # Collect all saliency values
        naive_slang = np.array(imp['naive_slang_sal'])
        naive_non_slang = np.array(imp['naive_non_slang_sal'])
        trained_slang = np.array(imp['trained_slang_sal'])
        trained_non_slang = np.array(imp['trained_non_slang_sal'])

        # Plot KDE for Slang
        if len(naive_slang) > 1:
            sns.kdeplot(naive_slang, color=palette[0], label='Naive Slang',
                       fill=True, alpha=0.3, ax=ax, linewidth=2)
        if len(trained_slang) > 1:
            sns.kdeplot(trained_slang, color=palette[3], label='Trained Slang',
                       fill=True, alpha=0.3, ax=ax, linewidth=2)

        # Plot KDE for Non-Slang with dashed lines
        if len(naive_non_slang) > 1:
            sns.kdeplot(naive_non_slang, color=palette[0], label='Naive Non-Slang',
                       fill=False, alpha=0.5, ax=ax, linestyle='--', linewidth=2)
        if len(trained_non_slang) > 1:
            sns.kdeplot(trained_non_slang, color=palette[3], label='Trained Non-Slang',
                       fill=False, alpha=0.5, ax=ax, linestyle='--', linewidth=2)

        # Add statistics text
        stats_text = (f"Slang: {imp['slang_increase']:+.2f}\n"
                     f"Non-Slang: {imp['non_slang_decrease']:+.2f}\n"
                     f"Score: {imp['improvement_score']:+.2f}")

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontweight='bold')

        ax.set_xlabel('Saliency Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'#{idx+1}: {imp["model"]}\n{imp["conv_id"]}',
                    fontsize=12, fontweight='bold')
        ax.set_xlim(left=-0.5, right=10.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Top 20 Training Improvements: Saliency Distribution Changes\n'
                'Solid Lines = Slang | Dashed Lines = Non-Slang',
                fontsize=20, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    output_path = '/home/minseok/forensic/xai_results/top20_kde_improvements.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Create aggregated KDE plot (combining all top 20)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Aggregate all saliency values
    all_naive_slang = np.concatenate([imp['naive_slang_sal'] for imp in top_20 if len(imp['naive_slang_sal']) > 0])
    all_trained_slang = np.concatenate([imp['trained_slang_sal'] for imp in top_20 if len(imp['trained_slang_sal']) > 0])
    all_naive_non_slang = np.concatenate([imp['naive_non_slang_sal'] for imp in top_20 if len(imp['naive_non_slang_sal']) > 0])
    all_trained_non_slang = np.concatenate([imp['trained_non_slang_sal'] for imp in top_20 if len(imp['trained_non_slang_sal']) > 0])

    # Plot 1: Slang utterances - Naive vs Trained
    ax = axes[0]
    sns.kdeplot(all_naive_slang, color=palette[0], label='Naive',
               fill=True, alpha=0.4, ax=ax, linewidth=3)
    sns.kdeplot(all_trained_slang, color=palette[3], label='Trained',
               fill=True, alpha=0.4, ax=ax, linewidth=3)

    # Add statistics
    naive_mean = np.mean(all_naive_slang)
    trained_mean = np.mean(all_trained_slang)
    diff = trained_mean - naive_mean

    stats_text = f"Naive: μ={naive_mean:.2f}\nTrained: μ={trained_mean:.2f}\nIncrease: {diff:+.2f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=14, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           fontweight='bold')

    ax.set_xlabel('Saliency Value', fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax.set_title('SLANG Utterances (Top 20 Combined)\nTraining Effect: Higher Saliency ✓',
                fontsize=18, fontweight='bold')
    ax.set_xlim(left=-0.5)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Non-Slang utterances - Naive vs Trained
    ax = axes[1]
    sns.kdeplot(all_naive_non_slang, color=palette[0], label='Naive',
               fill=True, alpha=0.4, ax=ax, linewidth=3)
    sns.kdeplot(all_trained_non_slang, color=palette[3], label='Trained',
               fill=True, alpha=0.4, ax=ax, linewidth=3)

    # Add statistics
    naive_mean = np.mean(all_naive_non_slang)
    trained_mean = np.mean(all_trained_non_slang)
    diff = naive_mean - trained_mean

    stats_text = f"Naive: μ={naive_mean:.2f}\nTrained: μ={trained_mean:.2f}\nDecrease: {diff:+.2f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=14, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           fontweight='bold')

    ax.set_xlabel('Saliency Value', fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax.set_title('NON-SLANG Utterances (Top 20 Combined)\nTraining Effect: Lower Saliency ✓',
                fontsize=18, fontweight='bold')
    ax.set_xlim(left=-0.5)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    aggregate_path = '/home/minseok/forensic/xai_results/top20_aggregate_kde.png'
    plt.savefig(aggregate_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {aggregate_path}")
    plt.close()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nGenerated plots:")
    print(f"  1. Individual KDE plots: {output_path}")
    print(f"  2. Aggregated KDE plot:  {aggregate_path}")

if __name__ == "__main__":
    main()
