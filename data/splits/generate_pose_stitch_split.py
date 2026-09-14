#!/usr/bin/env python3
"""
Generate fixed external CSV files with PoseStitch-style preprocessing.
Pipeline: lowercasing → contraction expansion → punctuation removal → number splitting → person names → <UNKNOWN> low-freq
Split: 90/5/5, seed=42, UID-mapped to original keypoints.
"""
import os, re, sys, json, random
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ═══ Configuration ═══
ISIGN_CSV = "/home/shoaib/phd/Datasets/iSign/iSign_v1.1.csv"
OUT_DIR = Path("data/pose_stitch_split")
KP_DIR = Path("/home/shoaib/phd/Datasets/iSign/misc/iSign_Keypoints")

# ═══ Contraction expansion ═══
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "it's": "it is", "that's": "that is",
    "he's": "he is", "she's": "she is", "i'm": "i am",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "let's": "let us", "ain't": "am not",
    "mightn't": "might not", "mustn't": "must not", "needn't": "need not",
    "shan't": "shall not", "there's": "there is", "here's": "here is",
    "who's": "who is", "what's": "what is", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
}

# Simple English stopwords for person name detection heuristic
STOPWORDS = set("""
a an the and or but in on at to for of with by from up about into through
during before after above below between out off over under again further
then once here there when where why how all both each few more most other
some such no nor not only own same so than too very s t will just don
should now is was are were been be have has had do does did can could
would may might must shall should will would
""".split())

def preprocess_text(text):
    """Apply PoseStitch-style text normalization."""
    text = text.strip()
    # 1. Lowercasing
    text = text.lower()
    # 2. Contraction expansion
    words = text.split()
    words = [CONTRACTIONS.get(w, w) for w in words]
    text = ' '.join(words)
    # 3. Punctuation removal (replace with spaces, keep only word characters + digits)
    text = re.sub(r'[^\w\s]', ' ', text)
    # 4. Split numbers into individual digits
    def split_num(m):
        return ' '.join(list(m.group(0)))
    text = re.sub(r'\d+', split_num, text)
    # 5. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_person_name(word, position_in_sentence):
    """Heuristic: treat capitalized non-stopwords as potential person names."""
    # We work on original text before lowercasing for person detection
    if len(word) <= 2:
        return False
    if word[0].isupper() and word[1:].islower() and not position_in_sentence == 0:
        if word.lower() not in STOPWORDS:
            return True
    return False


def main():
    print(f"Loading {ISIGN_CSV}...")
    df = pd.read_csv(ISIGN_CSV)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Build UID → keypoint lookup
    print(f"\nBuilding keypoint lookup from {KP_DIR}...")
    uid_to_kp = {}
    kp_count = 0
    for f in sorted(os.listdir(KP_DIR)):
        if f.endswith('.npy'):
            uid_to_kp[f.replace('.npy', '')] = f
            kp_count += 1
    print(f"  {kp_count} keypoint files found")

    # Filter rows with valid keypoints
    valid_rows = []
    missing_kp = 0
    for i, row in df.iterrows():
        uid = str(row.get('uid', ''))
        if uid in uid_to_kp:
            text = str(row.get('text', ''))
            if isinstance(text, str) and len(text.strip()) >= 3:
                valid_rows.append((uid, text))
        else:
            missing_kp += 1
    print(f"  Valid rows with keypoints: {len(valid_rows)}")
    print(f"  Missing keypoints: {missing_kp}")

    # Shuffle with seed=42
    random.shuffle(valid_rows)

    # Split 90/5/5
    n = len(valid_rows)
    train_end = int(n * 0.90)
    val_end = train_end + int(n * 0.05)
    train_uids = valid_rows[:train_end]
    val_uids = valid_rows[train_end:val_end]
    test_uids = valid_rows[val_end:]
    print(f"\nSplit (seed=42): train={len(train_uids)} val={len(val_uids)} test={len(test_uids)}")

    # ── Preprocess text ──
    print("\nPreprocessing text...")
    all_uid_text_processed = {}
    all_uids_original = {}
    for uid, text in train_uids + val_uids + test_uids:
        # Store original for reference
        all_uids_original[uid] = text
        # Apply preprocessing
        processed = preprocess_text(text)
        all_uid_text_processed[uid] = processed

    # ── Count word frequencies (for <UNKNOWN> replacement) ──
    print("Counting word frequencies...")
    word_counter = Counter()
    for uid, text in all_uid_text_processed.items():
        word_counter.update(text.split())
    print(f"  Unique words: {len(word_counter)}")
    print(f"  Words with freq=1: {sum(1 for k,v in word_counter.items() if v == 1)}")
    print(f"  Words with freq<3:  {sum(1 for k,v in word_counter.items() if v < 3)}")

    # Replace low-freq words with <UNKNOWN>
    print("Replacing low-freq words with <UNKNOWN>...")
    for uid in all_uid_text_processed:
        words = all_uid_text_processed[uid].split()
        words = [w if word_counter[w] >= 3 else 'unknown' for w in words]
        all_uid_text_processed[uid] = ' '.join(words)

    # ── Person name detection on ORIGINAL text, mark in processed text ──
    # Note: this is a heuristic since we don't have Indian_names.txt
    # We'll mark capitalized non-stopwords that appear in the original text
    # as <PERSON> in the processed output
    print("Detecting person names (heuristic)...")
    person_count = 0
    for uid in all_uid_text_processed:
        original = all_uids_original[uid]
        processed = all_uid_text_processed[uid]
        # Find capitalized words in original (not sentence-start)
        orig_words = original.split()
        proc_words = processed.split()
        if len(orig_words) != len(proc_words):
            continue  # Skip if contraction expansion changed word count
        for i, (ow, pw) in enumerate(zip(orig_words, proc_words)):
            if is_person_name(ow, i):
                proc_words[i] = 'person'
                person_count += 1
        all_uid_text_processed[uid] = ' '.join(proc_words)
    print(f"  Person name replacements: {person_count}")

    # ── Write CSV files ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {'train': train_uids, 'val': val_uids, 'test': test_uids}
    for split_name, uid_list in splits.items():
        rows = []
        for uid, _ in uid_list:
            processed_text = all_uid_text_processed[uid]
            original_text = all_uids_original[uid]
            kp_file = uid_to_kp.get(uid, '')
            rows.append({
                'uid': uid,
                'text': processed_text,
                'text_original': original_text,
                'kp_file': kp_file,
                'split': split_name,
            })
        out_df = pd.DataFrame(rows)
        out_path = OUT_DIR / f'isign_{split_name}_pose_stitch.csv'
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}: {len(out_df)} rows")

    # Write combined CSV
    all_rows = []
    for split_name, uid_list in splits.items():
        for uid, _ in uid_list:
            all_rows.append({
                'uid': uid,
                'text': all_uid_text_processed[uid],
                'text_original': all_uids_original[uid],
                'kp_file': uid_to_kp.get(uid, ''),
                'split': split_name,
            })
    combined = pd.DataFrame(all_rows)
    combined.to_csv(OUT_DIR / 'isign_all_pose_stitch.csv', index=False)

    # Write summary
    summary = {
        'total_samples': n,
        'train': len(train_uids),
        'val': len(val_uids),
        'test': len(test_uids),
        'preprocessing': ['lowercasing', 'contraction expansion', 'punctuation removal',
                         'number splitting', 'person name → person', 'low-freq → unknown'],
        'seed': 42,
        'split_ratios': '90/5/5',
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Show sample
    print(f"\n=== Sample preprocessed text ===")
    for uid, _ in train_uids[:5]:
        print(f"  UID: {uid}")
        print(f"    Original: {all_uids_original[uid][:120]}")
        print(f"    Processed: {all_uid_text_processed[uid][:120]}")
        print()

    print(f"\nDone. Files in {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
