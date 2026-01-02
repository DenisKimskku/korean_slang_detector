#!/usr/bin/env python
"""
Create individual KDE plots for top 20 conversations
Each plot shows naive vs trained (no slang/non-slang distinction)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from bs4 import BeautifulSoup
import os

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
    # Collect all saliency values
    naive_all_sal = []
    trained_all_sal = []

    # Also collect by label for statistics
    naive_slang_sal = []
    naive_non_slang_sal = []
    trained_slang_sal = []
    trained_non_slang_sal = []

    for utt in naive_conv['utterances']:
        naive_all_sal.extend(utt['saliency'])
        if utt['label'] == 1:
            naive_slang_sal.extend(utt['saliency'])
        else:
            naive_non_slang_sal.extend(utt['saliency'])

    for utt in trained_conv['utterances']:
        trained_all_sal.extend(utt['saliency'])
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
        'naive_all_sal': naive_all_sal,
        'trained_all_sal': trained_all_sal,
        'naive_slang_mean': naive_slang_mean,
        'naive_non_slang_mean': naive_non_slang_mean,
        'trained_slang_mean': trained_slang_mean,
        'trained_non_slang_mean': trained_non_slang_mean,
        'slang_increase': slang_increase,
        'non_slang_decrease': non_slang_decrease,
        'improvement_score': improvement_score
    }

def main():
    """Create individual KDE plots for top 20"""

    xai_dir = Path('/home/minseok/forensic/xai_results')
    output_dir = Path('/home/minseok/forensic/xai_results/top20_individual_kdes')
    output_dir.mkdir(exist_ok=True)

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

    print(f"\nCreating individual KDE plots for top 20 conversations...")

    palette = sns.color_palette("muted", 8)

    for rank, imp in enumerate(top_20, 1):
        # Create individual plot
        fig, ax = plt.subplots(figsize=(12, 8))

        naive_sal = np.array(imp['naive_all_sal'])
        trained_sal = np.array(imp['trained_all_sal'])

        # Plot KDE
        if len(naive_sal) > 1:
            sns.kdeplot(naive_sal, color=palette[0], label='Naive Model',
                       fill=True, alpha=0.5, ax=ax, linewidth=3)
        if len(trained_sal) > 1:
            sns.kdeplot(trained_sal, color=palette[3], label='Trained Model',
                       fill=True, alpha=0.5, ax=ax, linewidth=3)

        # Calculate statistics
        naive_mean = np.mean(naive_sal) if len(naive_sal) > 0 else 0
        naive_std = np.std(naive_sal) if len(naive_sal) > 0 else 0
        trained_mean = np.mean(trained_sal) if len(trained_sal) > 0 else 0
        trained_std = np.std(trained_sal) if len(trained_sal) > 0 else 0
        overall_change = trained_mean - naive_mean

        # Add detailed statistics text box
        stats_text = (
            f"Naive Model:\n"
            f"  μ = {naive_mean:.3f}, σ = {naive_std:.3f}\n"
            f"\n"
            f"Trained Model:\n"
            f"  μ = {trained_mean:.3f}, σ = {trained_std:.3f}\n"
            f"\n"
            f"Change: {overall_change:+.3f}\n"
            f"\n"
            f"Breakdown:\n"
            f"  Slang: {imp['slang_increase']:+.3f}\n"
            f"  Non-Slang: {imp['non_slang_decrease']:+.3f}\n"
            f"  Score: {imp['improvement_score']:+.3f}"
        )

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=13, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontweight='bold', family='monospace')

        ax.set_xlabel('Saliency Value', fontsize=16, fontweight='bold')
        ax.set_ylabel('Density', fontsize=16, fontweight='bold')
        ax.set_title(f'Rank #{rank}: {imp["model"]} - {imp["conv_id"]}\n'
                    f'Training Effect on Saliency Distribution',
                    fontsize=18, fontweight='bold', pad=20)

        ax.set_xlim(left=-0.5, right=10.5)

        # Format ticks
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')

        ax.legend(fontsize=16, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with rank and model name
        filename = f"rank{rank:02d}_{imp['model']}_{imp['conv_id']}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")

    # Create index HTML file for easy viewing
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Top 20 Training Improvements - KDE Plots</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        .plot-item {
            border: 2px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #fafafa;
        }
        .plot-item img {
            width: 100%;
            border-radius: 3px;
        }
        .plot-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Top 20 Training Improvements - Individual KDE Plots</h1>
        <p>Each plot compares the saliency distribution between naive (blue) and trained (red/orange) models.</p>

        <div class="plot-grid">
"""

    for rank, imp in enumerate(top_20, 1):
        filename = f"rank{rank:02d}_{imp['model']}_{imp['conv_id']}.png"
        html_content += f"""
            <div class="plot-item">
                <div class="plot-title">Rank #{rank}: {imp['model']} - {imp['conv_id']}</div>
                <img src="{filename}" alt="Rank {rank}">
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    index_path = output_dir / 'index.html'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n{'='*80}")
    print(f"DONE! Created {len(top_20)} individual KDE plots")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Index HTML: {index_path}")
    print(f"\nOpen index.html in a browser to view all plots!")

if __name__ == "__main__":
    main()
