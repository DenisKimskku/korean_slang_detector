import os
import json
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

def extract_nouns(texts):
    """
    Extract nouns from a list of texts using Kkma.
    """
    kkma = Kkma()
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
        },
        ...
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
    base_output_directory = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/test'  # Changed to 'test' folder

    # Define modified subdirectories
    modified_folders = {
        'modified1': 'modified1',  # 원본단어
        'modified2': 'modified2',  # 변형어1
        'modified3': 'modified3',  # 변형어2
        'modified4': 'modified4',  # 변형어3
        'modified5': 'modified5'   # 변형어4
    }

    # Ensure all modified directories exist
    for folder in modified_folders.values():
        os.makedirs(os.path.join(base_output_directory, folder), exist_ok=True)
    
    # Load all JSON files
    all_json_data = load_all_jsons(input_directory)
    
    # Load poison.csv
    poison_csv_path = '/home/minseok/forensic/poison.csv'  # Updated path
    poison_df = load_poison_csv(poison_csv_path)
    
    # Create noun mappings for each modified folder
    # modified1: original words (no replacement)
    noun_mapping_modified1 = dict(zip(poison_df['원본단어'], poison_df['원본단어']))
    # modified2 to modified5: 변형어1 to 변형어4
    noun_mapping_modified = {}
    for i in range(1, 5):
        column = f'변형어{i}'
        noun_mapping_modified[f'modified{i+1}'] = dict(zip(poison_df['원본단어'], poison_df[column]))
    
    # Initialize tqdm progress bar
    for filename, data in tqdm(all_json_data, desc="Processing JSON files", unit="file"):
        try:
            # Step 1: Extract 'original_form' texts
            texts = extract_original_forms(data)
            
            # Step 2: Extract nouns
            nouns_list = extract_nouns(texts)
            
            # Step 3: Compute top-3 TF-IDF nouns
            top_nouns = compute_top_n_tfidf(nouns_list, top_n=3)
            
            # Filter noun_mapping to include only top_nouns
            # This ensures only the top nouns are replaced
            # Create separate mappings for each modified folder
            noun_mappings = {}
            # modified1: original words (no replacement)
            noun_mappings['modified1'] = {noun: noun for noun in top_nouns if noun in noun_mapping_modified1}
            # modified2 to modified5
            for key in noun_mapping_modified:
                noun_mappings[key] = {noun: noun_mapping_modified[key][noun] for noun in top_nouns if noun in noun_mapping_modified[key]}
            
            # Iterate through each modified folder and create modified JSON
            for folder_key, mapping in noun_mappings.items():
                # Skip if there are no nouns to replace
                if not mapping:
                    continue
                
                # Create output structure with replacements and labels
                output_data = create_output_structure(data, mapping)
                
                # Define output path
                output_path = os.path.join(base_output_directory, folder_key, filename)
                
                # Save the modified JSON to the respective modified folder
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            # Additionally, handle modified5 separately if needed
            # Since 'modified5' corresponds to 변형어4
            # Already handled in the loop above
            
        except Exception as e:
            # Handle exceptions for individual files without stopping the entire process
            print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    main()
