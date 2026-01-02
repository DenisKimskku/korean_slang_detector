import os
import json
import random
import pandas as pd
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def load_all_jsons(directory):
    """
    Load all JSON files from the specified directory.
    Returns a list of tuples containing (filename, data).
    """
    json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No JSON files found in the specified directory.")
    
    all_data = []
    for file in json_files:
        json_path = os.path.join(directory, file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append((file, data))
    return all_data

def extract_original_forms(data):
    """
    Extract 'original_form' from each utterance in all documents.
    """
    original_forms = []
    if 'document' in data:
        for doc in data['document']:
            if 'utterance' in doc:
                for utterance in doc['utterance']:
                    original_form = utterance.get('original_form', '').strip()
                    # Skip empty strings or placeholders like '{emoji}'
                    if original_form and original_form != '{emoji}':
                        original_forms.append(original_form)
    else:
        raise KeyError("'document' key not found in JSON data.")
    return original_forms

def extract_nouns(texts, kkma):
    """
    Extract nouns from a list of texts using Kkma.
    """
    nouns_list = [kkma.nouns(text) for text in texts]
    return nouns_list

def compute_top_n_tfidf(nouns_list, top_n=3):
    """
    Compute TF-IDF scores for the nouns and return the top_n important nouns.
    """
    # Join nouns back into strings for TF-IDF
    documents = [' '.join(nouns) for nouns in nouns_list]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(vocab, tfidf_scores))
    
    # Get the top_n nouns with the highest TF-IDF scores
    sorted_nouns = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
    top_nouns = [noun for noun, score in sorted_nouns[:top_n]]
    return top_nouns

def load_poison_csv(csv_path):
    """
    Load poison.csv and return a DataFrame with '원본단어' and its variations.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"poison.csv not found at path: {csv_path}")
    df = pd.read_csv(csv_path)
    required_columns = ['원본단어', '변형어1', '변형어2', '변형어3', '변형어4']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in poison.csv.")
    # Drop rows with missing '원본단어'
    df = df.dropna(subset=['원본단어'])
    return df

def replace_nouns_in_text(text, noun_mapping):
    """
    Replace target nouns in the text based on the noun_mapping dictionary.
    """
    for target, replacement in noun_mapping.items():
        text = text.replace(target, replacement)
    return text

def create_output_structure(data, noun_mapping):
    """
    Create the desired output structure:
    [
        {
            "id": <id>,
            "utterance": [
                {
                    "original_form": <modified or original original_form>,
                    "label": <1 or 0>
                },
                ...
            ]
        }
    ]
    """
    output = []
    if 'id' not in data:
        raise KeyError("'id' key not found in JSON data.")
    
    top_id = data['id']
    utterance_list = []

    if 'document' in data:
        for doc in data['document']:
            if 'utterance' in doc:
                for utterance in doc['utterance']:
                    original_form = utterance.get('original_form', '').strip()
                    if original_form and original_form != '{emoji}':
                        modified_form = original_form
                        label = 0
                        # Check and replace each target noun
                        for target_noun, replacement_noun in noun_mapping.items():
                            if target_noun in modified_form:
                                modified_form = modified_form.replace(target_noun, replacement_noun)
                                # Only set label to 1 if replacement occurs
                                label = 1
                        utterance_entry = {
                            "original_form": modified_form,
                            "label": label
                        }
                        utterance_list.append(utterance_entry)
    else:
        raise KeyError("'document' key not found in JSON data.")
    
    output_entry = {
        "id": top_id,
        "utterance": utterance_list
    }
    output.append(output_entry)
    return output

def main():
    # Define input and base output directories
    input_directory = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/국립국어원 메신저 말뭉치(버전 2.0)'
    base_output_directory = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/test2'  # Changed to 'test2' folder

    # Define modified subdirectories
    modified_folders = {
        'modified1': 'modified1',  # Replace with '원본단어'
        'modified2': 'modified2',  # Replace with '변형어1'
        'modified3': 'modified3',  # Replace with '변형어2'
        'modified4': 'modified4',  # Replace with '변형어3'
        'modified5': 'modified5'   # Replace with '변형어4'
    }

    # Ensure all modified directories exist
    for folder in modified_folders.values():
        os.makedirs(os.path.join(base_output_directory, folder), exist_ok=True)
    
    # Load all JSON files
    all_json_data = load_all_jsons(input_directory)
    
    # Load poison.csv
    poison_csv_path = '/home/minseok/forensic/poison.csv'  # Updated path
    poison_df = load_poison_csv(poison_csv_path)
    
    # Create lists of replacement words for each modification
    replacements = {
        'modified1': poison_df['원본단어'].dropna().tolist(),
        'modified2': poison_df['변형어1'].dropna().tolist(),
        'modified3': poison_df['변형어2'].dropna().tolist(),
        'modified4': poison_df['변형어3'].dropna().tolist(),
        'modified5': poison_df['변형어4'].dropna().tolist()
    }
    
    # Initialize Kkma once to improve performance
    kkma = Kkma()
    
    # Initialize tqdm progress bar
    for filename, data in tqdm(all_json_data, desc="Processing JSON files", unit="file"):
        try:
            # Step 1: Extract 'original_form' texts
            texts = extract_original_forms(data)
            
            # Step 2: Extract nouns
            nouns_list = extract_nouns(texts, kkma)
            
            # Step 3: Compute top-3 TF-IDF nouns
            top_nouns = compute_top_n_tfidf(nouns_list, top_n=3)
            
            if not top_nouns:
                # If no nouns found, skip replacement
                continue
            
            # Step 4: For each modification, create noun_mapping and apply replacements
            for mod_key, replacement_list in replacements.items():
                # Skip modification if replacement list is empty
                if not replacement_list:
                    continue
                
                noun_mapping = {}
                for noun in top_nouns:
                    replacement_word = random.choice(replacement_list)
                    noun_mapping[noun] = replacement_word
                
                # Create the output structure with replacements and labels
                output_data = create_output_structure(data, noun_mapping)
                
                # Define output path
                output_path = os.path.join(base_output_directory, modified_folders[mod_key], filename)
                
                # Save the modified JSON to the respective modified folder
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        except Exception as e:
            # Handle exceptions for individual files without stopping the entire process
            print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    main()
