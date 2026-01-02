"""
Generate a summary report comparing all evaluation results.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List

def find_latest_result_file(results_dir: str) -> str:
    """Find the most recent evaluation results file in a directory"""
    if not os.path.exists(results_dir):
        return None

    json_files = glob.glob(os.path.join(results_dir, 'evaluation_results_*.json'))
    if not json_files:
        return None

    # Get the most recent file
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

def load_result(results_dir: str) -> Dict:
    """Load the latest result from a directory"""
    result_file = find_latest_result_file(results_dir)
    if not result_file:
        return None

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return None

def extract_metrics(result: Dict) -> Dict:
    """Extract key metrics from a result"""
    if not result:
        return None

    metrics = result.get('metrics', {})
    cm = metrics.get('confusion_matrix', {})

    extracted = {
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
        'auc': metrics.get('auc', 0),
        'total_samples': metrics.get('total_samples', 0),
        'predicted_positive': metrics.get('predicted_positive', 0),
        'tn': cm.get('tn', 0),
        'fp': cm.get('fp', 0),
        'fn': cm.get('fn', 0),
        'tp': cm.get('tp', 0)
    }

    # Calculate specificity and FPR
    if extracted['tn'] + extracted['fp'] > 0:
        extracted['specificity'] = extracted['tn'] / (extracted['tn'] + extracted['fp'])
        extracted['false_positive_rate'] = 1 - extracted['specificity']
    else:
        extracted['specificity'] = None
        extracted['false_positive_rate'] = None

    return extracted

def main():
    base_dir = '/home/minseok/forensic/new_data_evaluation'

    print("Generating summary report...")
    print("="*60)

    # Define all models to evaluate
    models = [
        # Fine-tuned models
        {'name': 'BERT-base (fine-tuned)', 'dir': 'results_bert_base', 'type': 'fine-tuned'},
        {'name': 'RoBERTa-base (fine-tuned)', 'dir': 'results_roberta_base', 'type': 'fine-tuned'},
        {'name': 'RoBERTa-large (fine-tuned)', 'dir': 'results_roberta_large', 'type': 'fine-tuned'},

        # Plain models
        {'name': 'BERT-base (plain)', 'dir': 'results_plain_bert_base', 'type': 'plain'},
        {'name': 'RoBERTa-base (plain)', 'dir': 'results_plain_roberta_base', 'type': 'plain'},
        {'name': 'RoBERTa-large (plain)', 'dir': 'results_plain_roberta_large', 'type': 'plain'},

        # API models
        {'name': 'Gemini-2.0-Flash', 'dir': 'results_gemini_gemini_2_0_flash_exp', 'type': 'api'},
        {'name': 'GPT-4o', 'dir': 'results_openai_gpt_4o', 'type': 'api'},
    ]

    summary = {
        'report_timestamp': datetime.now().isoformat(),
        'models': []
    }

    # Collect results for each model
    for model_info in models:
        model_name = model_info['name']
        results_dir = os.path.join(base_dir, model_info['dir'])

        print(f"\nLoading results for {model_name}...")

        result = load_result(results_dir)
        if result:
            metrics = extract_metrics(result)
            if metrics:
                model_summary = {
                    'name': model_name,
                    'type': model_info['type'],
                    'metrics': metrics
                }
                summary['models'].append(model_summary)
                print(f"  ✓ Loaded (Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f})")
            else:
                print(f"  ✗ Could not extract metrics")
        else:
            print(f"  ✗ No results found in {results_dir}")

    # Save summary report
    output_file = os.path.join(base_dir, 'summary_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Summary report saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print comparison table
    print("COMPARISON TABLE")
    print("="*100)
    print(f"{'Model':<30} {'Type':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPR':<10}")
    print("-"*100)

    for model in summary['models']:
        name = model['name']
        model_type = model['type']
        m = model['metrics']
        fpr = f"{m['false_positive_rate']:.4f}" if m['false_positive_rate'] is not None else "N/A"

        print(f"{name:<30} {model_type:<12} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {fpr:<10}")

    print("="*100)
    print("\nNote: FPR = False Positive Rate (lower is better for clean data)")
    print("      Since the new_data is clean (no drug content), we expect:")
    print("      - High specificity (low FPR)")
    print("      - Low false positives")
    print("")

    # Find best models
    if summary['models']:
        # Best by F1
        best_f1 = max(summary['models'], key=lambda x: x['metrics']['f1'])
        print(f"Best F1 Score: {best_f1['name']} ({best_f1['metrics']['f1']:.4f})")

        # Best by accuracy
        best_acc = max(summary['models'], key=lambda x: x['metrics']['accuracy'])
        print(f"Best Accuracy: {best_acc['name']} ({best_acc['metrics']['accuracy']:.4f})")

        # Best specificity (lowest FPR)
        models_with_fpr = [m for m in summary['models'] if m['metrics']['false_positive_rate'] is not None]
        if models_with_fpr:
            best_spec = min(models_with_fpr, key=lambda x: x['metrics']['false_positive_rate'])
            print(f"Best Specificity (Lowest FPR): {best_spec['name']} (FPR: {best_spec['metrics']['false_positive_rate']:.4f})")

    print("\nReport generation completed!")

if __name__ == "__main__":
    main()
