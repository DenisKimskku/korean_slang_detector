import csv
import re

def convert_noun_to_csv(input_text):
    # Split the text into lines
    lines = input_text.split('\n')
    
    # Initialize list to store vocab entries
    vocab_entries = []
    
    # Regular expression to match Korean word followed by /NNG or /NNP
    pattern = r'^(.+?)/(NNG|NNP)'
    
    for line in lines:
        # Skip empty lines and comment lines
        if not line.strip() or line.strip().startswith('//'):
            continue
            
        # Try to match the pattern
        match = re.match(pattern, line.strip())
        if match:
            word = match.group(1)  # The Korean word
            word_type = match.group(2)  # NNG or NNP
            vocab_entries.append([word, word_type])
    
    # Write to CSV
    with open('vocab.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['vocab', 'type'])
        # Write entries
        writer.writerows(vocab_entries)

#load text from /home/minseok/forensic/noun.txt and do the conversion
with open('/home/minseok/forensic/noun.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()
    convert_noun_to_csv(input_text)