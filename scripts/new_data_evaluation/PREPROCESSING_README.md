# New Data Preprocessing with TF-IDF Drug Keyword Substitution

## Overview

The `preprocess_new_data.py` script now includes TF-IDF based drug keyword substitution, matching the training data generation process from `data_generator.py`.

## Key Features

### 1. Dual Output Generation
For each input conversation, the script generates TWO versions:

- **Plain Version**: Unchanged conversations with all labels = 0
- **Drug Substituted Version**: With TF-IDF based keyword replacement

**Example**: If there are 1,606 conversations in the input:
- Output: 3,212 conversations (1,606 plain + 1,606 drug_substituted)

### 2. TF-IDF Drug Keyword Substitution Process

The script follows the same process as the original training data generation:

1. **Extract Nouns**: Uses Kkma morphological analyzer to extract nouns from all messages
2. **Compute TF-IDF Scores**: Calculates TF-IDF scores to identify the top-3 most important nouns
3. **Replace Keywords**: Replaces the top-3 nouns with random drug-related vocabulary words from `vocab.csv`
4. **Label Messages**:
   - Messages with replacements: `label = 1`
   - Unchanged messages: `label = 0`

### 3. Output Format

Each conversation in the output JSON has:

```json
{
  "id": "nateon_NATEON_40_08",
  "source": "nateon",
  "filename": "NATEON_40_08.txt",
  "type": "plain",  // or "drug_substituted"
  "utterance": [
    {
      "speaker_id": "1",
      "original_form": "안녕하세요",
      "label": 0
    }
  ]
}
```

## Requirements

### Conda Environment
The script requires the `forensic` conda environment for Kkma to work properly:

```bash
source /home/minseok/anaconda3/etc/profile.d/conda.sh
conda activate forensic
python preprocess_new_data.py --messenger nateon
```

### Dependencies
- `konlpy` with Kkma (Korean morphological analyzer)
- `sklearn` (TfidfVectorizer)
- `pandas` (for vocab.csv)
- `tqdm` (progress bars)

## Usage

### Command Line

```bash
# Preprocess all messengers
conda activate forensic
python preprocess_new_data.py

# Preprocess specific messenger(s)
python preprocess_new_data.py --messenger nateon
python preprocess_new_data.py --messenger band facebook
python preprocess_new_data.py --messenger all

# Custom vocab file
python preprocess_new_data.py --vocab-path /path/to/vocab.csv
```

### Via Shell Scripts

All evaluation shell scripts automatically activate the forensic environment for preprocessing:

```bash
# Preprocesses NateOn data with TF-IDF substitution, then evaluates
./run_all_evaluations_nateon.sh

# Preprocesses all messengers, then evaluates
./run_all_evaluations.sh
```

## Example Output Statistics

For NateOn messenger:
- Input: 1,606 conversation files
- Output: 3,212 conversations
  - Plain conversations: 1,606 (all label=0)
  - Drug substituted: 1,606 (mixed label=0 and label=1)
- Total utterances: 51,320
- Utterances with drug keywords (label=1): 13,479 (~26%)

## Example Conversation

**Plain Version** (all label=0):
```
1. [CLEAN] 뭐 하는 게 맞는 걸까?
2. [CLEAN] 음 그건 개인 자유 아닐까 형
3. [CLEAN] 아 진짜 하는 게 맞는 걸까 나도 궁금 형
```

**Drug Substituted Version** (with TF-IDF replacement):
```
1. [🔴 DRUG] 월사금을 하는 게 맞는 걸까?
2. [🔴 DRUG] 음 그건 개인 자유 아닐까 양안장
3. [🔴 DRUG] 아 진짜 하는 게 맞는 걸까 나도 궁금 양안장
```

In the drug substituted version:
- "뭐" → "월사금" (drug vocab word)
- "형" → "양안장" (drug vocab word)
- These messages are marked with `label=1`

## Error Handling

The script includes robust error handling:

1. **Kkma Initialization Failure**: If Kkma fails to initialize, the script continues but creates drug_substituted versions with all label=0
2. **TF-IDF Computation Failure**: Falls back to creating drug_substituted versions without replacements (all label=0)
3. **Individual File Errors**: Logged but don't stop the entire process

## Files Modified

1. `preprocess_new_data.py` - Complete rewrite with TF-IDF support
2. `run_all_evaluations.sh` - Updated to use forensic conda environment
3. `run_all_evaluations_nateon.sh` - Updated for forensic environment
4. `run_all_evaluations_band.sh` - Updated for forensic environment
5. `run_all_evaluations_facebook.sh` - Updated for forensic environment
6. `run_all_evaluations_instagram.sh` - Updated for forensic environment

## Comparison with Training Data Generation

The preprocessing now matches the training data generation process:

| Feature | Training (`data_generator.py`) | Evaluation (`preprocess_new_data.py`) |
|---------|-------------------------------|--------------------------------------|
| Noun Extraction | ✅ Kkma | ✅ Kkma |
| TF-IDF Scoring | ✅ Top-3 nouns | ✅ Top-3 nouns |
| Vocab Replacement | ✅ Random 3 from vocab.csv | ✅ Random 3 from vocab.csv |
| Labeling | ✅ label=1 for replaced | ✅ label=1 for replaced |
| Output Format | ✅ JSON with id/utterance | ✅ JSON with id/utterance |
| **NEW: Dual Output** | ❌ Only substituted version | ✅ Both plain + substituted |

## Why Both Versions?

Generating both plain and drug-substituted versions allows us to:

1. **Evaluate on clean data**: Test how well models handle truly clean conversations
2. **Evaluate on synthetic drug data**: Test detection capability on artificially created drug-related conversations
3. **Compare performance**: See how models perform on both types
4. **Match training conditions**: The drug-substituted version matches how the training data was generated

## Next Steps

After preprocessing, the evaluation scripts will:
1. Load both plain and drug-substituted conversations
2. Run sliding window analysis
3. Evaluate models on both types separately
4. Generate comprehensive metrics

This ensures thorough testing of the models on both clean and synthetic drug-related data.
