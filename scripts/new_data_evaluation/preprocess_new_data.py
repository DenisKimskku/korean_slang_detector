"""
Preprocess new_data from .txt format to JSON format expected by evaluation scripts.
This version includes TF-IDF based drug keyword substitution matching the training data generation.

Generates TWO versions:
1. Plain (unchanged) conversations - all label=0
2. Drug keyword substituted conversations - with TF-IDF replacement
"""

import os
import json
import random
import pandas as pd
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from typing import List, Dict

def parse_txt_file(file_path: str) -> List[Dict]:
    """
    Parse a .txt file with conversation format:
    1 : message
    2 : message
    3 : message

    Returns a list of utterances with speaker and message.
    """
    utterances = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse format: "speaker_id : message"
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker_id = parts[0].strip()
                message = parts[1].strip()

                utterances.append({
                    'speaker_id': speaker_id,
                    'original_form': message,
                    'label': 0  # Default label
                })

    return utterances

# Global Kkma instance (initialize once to avoid Java/JVM issues)
_kkma = None

def get_kkma():
    """Get or initialize the global Kkma instance."""
    global _kkma
    if _kkma is None:
        try:
            _kkma = Kkma()
        except Exception:
            # If Kkma fails, return None
            _kkma = None
    return _kkma

def extract_nouns(texts: List[str]) -> List[List[str]]:
    """
    Extract nouns from a list of texts using Kkma.
    """
    kkma = get_kkma()
    if kkma is None:
        # If Kkma is not available, return empty lists
        return [[] for _ in texts]

    nouns_list = []
    for text in texts:
        try:
            nouns = kkma.nouns(text)
            nouns_list.append(nouns)
        except Exception:
            # If morphological analysis fails, use empty list
            nouns_list.append([])
    return nouns_list

def compute_top_n_tfidf(nouns_list: List[List[str]], top_n: int = 3) -> List[str]:
    """
    Compute TF-IDF scores for the nouns and return the top_n important nouns.
    """
    # Filter out empty noun lists
    nouns_list = [nouns for nouns in nouns_list if nouns]

    if not nouns_list:
        return []

    # Join nouns back into strings for TF-IDF
    documents = [' '.join(nouns) for nouns in nouns_list]

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        tfidf_dict = dict(zip(vocab, tfidf_scores))

        # Get the top_n nouns with the highest TF-IDF scores
        sorted_nouns = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
        top_nouns = [noun for noun, score in sorted_nouns[:top_n]]
        return top_nouns
    except Exception as e:
        # If TF-IDF computation fails, return empty list
        return []

def load_vocab(csv_path: str) -> List[str]:
    """
    Load vocab.csv and return a list of vocab words.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"vocab.csv not found at path: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'vocab' not in df.columns:
        raise KeyError("Column 'vocab' not found in vocab.csv.")
    vocab_words = df['vocab'].dropna().tolist()
    return vocab_words

def apply_drug_keyword_substitution(utterances: List[Dict], vocab_words: List[str]) -> List[Dict]:
    """
    Apply TF-IDF based drug keyword substitution to utterances.
    Returns a new list of utterances with drug keywords substituted.
    """
    # Extract texts for TF-IDF analysis
    texts = [utt['original_form'] for utt in utterances]

    # Extract nouns
    nouns_list = extract_nouns(texts)

    # Compute top-3 TF-IDF nouns
    top_nouns = compute_top_n_tfidf(nouns_list, top_n=3)

    # If we don't have enough top nouns or vocab words, return original utterances with label=0
    if len(top_nouns) < 3 or len(vocab_words) < 3:
        return [{'speaker_id': utt['speaker_id'], 'original_form': utt['original_form'], 'label': 0}
                for utt in utterances]

    # Select unique replacement nouns
    replacement_nouns = random.sample(vocab_words, k=3)
    noun_mapping = dict(zip(top_nouns, replacement_nouns))

    # Apply replacements
    modified_utterances = []
    for utt in utterances:
        original_text = utt['original_form']
        modified_text = original_text
        label = 0

        # Check and replace each target noun
        for target_noun, replacement_noun in noun_mapping.items():
            if target_noun in modified_text:
                modified_text = modified_text.replace(target_noun, replacement_noun)
                label = 1

        modified_utterances.append({
            'speaker_id': utt['speaker_id'],
            'original_form': modified_text,
            'label': label
        })

    return modified_utterances

def convert_directory_to_json(input_base_dir: str, output_dir: str, vocab_words: List[str],
                              messengers: List[str] = None):
    """
    Convert all .txt files in subdirectories to JSON format.
    Generates BOTH plain and drug keyword substituted versions.

    Directory structure:
    input_base_dir/
        band/
            file1.txt
            file2.txt
        facebook/
            file1.txt
        instagram/
            file1.txt
        nateon/
            file1.txt

    Args:
        input_base_dir: Base directory containing messenger subdirectories
        output_dir: Output directory for JSON file
        vocab_words: List of vocabulary words for drug keyword substitution
        messengers: List of messengers to process (e.g., ['band', 'facebook']).
                    If None, processes all messengers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all subdirectories
    all_subdirs = ['band', 'facebook', 'instagram', 'nateon']
    subdirs = messengers if messengers else all_subdirs

    all_conversations = []
    file_count = 0

    for subdir in subdirs:
        subdir_path = os.path.join(input_base_dir, subdir)

        if not os.path.exists(subdir_path):
            print(f"Warning: Directory {subdir_path} not found, skipping...")
            continue

        txt_files = [f for f in os.listdir(subdir_path) if f.endswith('.txt')]
        print(f"\nProcessing {len(txt_files)} files from {subdir}...")

        for txt_file in tqdm(txt_files, desc=f"Processing {subdir}"):
            file_path = os.path.join(subdir_path, txt_file)

            try:
                # Parse the original file
                utterances = parse_txt_file(file_path)

                if utterances:
                    base_id = f"{subdir}_{txt_file.replace('.txt', '')}"

                    # 1. Create PLAIN (unchanged) conversation - all label=0
                    plain_conversation = {
                        'id': base_id,
                        'source': subdir,
                        'filename': txt_file,
                        'type': 'plain',
                        'utterance': [
                            {
                                'speaker_id': utt['speaker_id'],
                                'original_form': utt['original_form'],
                                'label': 0
                            } for utt in utterances
                        ]
                    }
                    all_conversations.append(plain_conversation)

                    # 2. Create DRUG KEYWORD SUBSTITUTED conversation
                    try:
                        drug_substituted_utterances = apply_drug_keyword_substitution(utterances, vocab_words)
                    except Exception as drug_error:
                        # If drug substitution fails, create version with all label=0
                        drug_substituted_utterances = [
                            {
                                'speaker_id': utt['speaker_id'],
                                'original_form': utt['original_form'],
                                'label': 0
                            } for utt in utterances
                        ]

                    drug_conversation = {
                        'id': f"{base_id}_drug_substituted",
                        'source': subdir,
                        'filename': txt_file,
                        'type': 'drug_substituted',
                        'utterance': drug_substituted_utterances
                    }
                    all_conversations.append(drug_conversation)

                    file_count += 1

            except Exception as e:
                print(f"\nError processing {file_path}: {e}")

    # Save all conversations to a single JSON file
    output_file = os.path.join(output_dir, 'all_conversations.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Preprocessing completed!")
    print(f"Total conversations generated: {len(all_conversations)}")
    print(f"  - Plain conversations: {len([c for c in all_conversations if c['type'] == 'plain'])}")
    print(f"  - Drug substituted conversations: {len([c for c in all_conversations if c['type'] == 'drug_substituted'])}")
    print(f"Total files processed: {file_count}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")

    # Print statistics
    total_utterances = sum(len(conv['utterance']) for conv in all_conversations)
    total_labels_1 = sum(sum(1 for utt in conv['utterance'] if utt['label'] == 1)
                        for conv in all_conversations)
    by_source = {}
    for conv in all_conversations:
        source = conv['source']
        conv_type = conv['type']
        key = f"{source}_{conv_type}"
        by_source[key] = by_source.get(key, 0) + 1

    print(f"\nStatistics:")
    print(f"  Total utterances: {total_utterances}")
    print(f"  Total utterances with drug keywords (label=1): {total_labels_1}")
    print(f"  Average utterances per conversation: {total_utterances / len(all_conversations):.2f}")
    print(f"\nConversations by source and type:")
    for key in sorted(by_source.keys()):
        print(f"  {key}: {by_source[key]} conversations")

    return output_file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess new_data from .txt to JSON format with TF-IDF drug keyword substitution')
    parser.add_argument('--messenger', type=str, nargs='+',
                       choices=['band', 'facebook', 'instagram', 'nateon', 'all'],
                       default=['all'],
                       help='Messenger platforms to process (default: all)')
    parser.add_argument('--vocab-path', type=str,
                       default='/home/minseok/forensic/vocab.csv',
                       help='Path to vocab.csv file (default: /home/minseok/forensic/vocab.csv)')

    args = parser.parse_args()

    INPUT_DIR = '/home/minseok/forensic/new_data'
    OUTPUT_DIR = '/home/minseok/forensic/new_data_evaluation/preprocessed'

    # Parse messenger argument
    if 'all' in args.messenger:
        messengers = None  # Process all messengers
        messenger_str = "all messengers"
    else:
        messengers = args.messenger
        messenger_str = ", ".join(messengers)

    print("Starting preprocessing of new_data...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Vocab file: {args.vocab_path}")
    print(f"Processing: {messenger_str}")
    print()

    # Load vocabulary
    print("Loading vocabulary from vocab.csv...")
    vocab_words = load_vocab(args.vocab_path)
    print(f"Loaded {len(vocab_words)} vocabulary words")
    print()

    # Initialize Kkma once at the beginning
    print("Initializing Korean morphological analyzer (Kkma)...")
    try:
        kkma = get_kkma()
        if kkma is not None:
            print("Kkma initialized successfully")
        else:
            print("Warning: Kkma initialization failed - will proceed without drug keyword substitution")
    except Exception as e:
        print(f"Warning: Kkma initialization failed: {e}")
        print("Will proceed without drug keyword substitution")
    print()

    output_file = convert_directory_to_json(INPUT_DIR, OUTPUT_DIR, vocab_words, messengers)

    print("\nPreprocessing complete!")
