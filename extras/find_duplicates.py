import numpy as np
import json
import os
from datasets import load_from_disk

# TODO: remove this
from transformers import AutoTokenizer
tokenizer_src = AutoTokenizer.from_pretrained("bert-base-german-cased", use_fast=True)
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
####

def load_training_data_stats():
    path = os.path.join(os.getenv("MTDATA"), "wmt24-eng-deu/train.stats.json")
    with open(path, 'r') as f:
        stats = json.load(f)
    return stats

def create_dataset_index_mapping():
    """
    Creates a mapping from sequential indices to dataset names with their start/end indices.
    
    Returns:
        dict: A dictionary mapping sequential index to (dataset_name, start_idx, end_idx)
    """
    stats = load_training_data_stats()
    
    # Get the selected counts (these are the actual counts used in the dataset)
    selected_counts = stats['counts']['selected']
    
    # Create the mapping
    index_mapping = {}
    current_idx = 0
    
    for dataset_name, count in selected_counts.items():
        start_idx = current_idx
        end_idx = current_idx + count
        index_mapping[len(index_mapping)] = (dataset_name, start_idx, end_idx)
        current_idx = end_idx
    
    return index_mapping

def get_dataset_slice_by_index(dataset_idx):
    """
    Get the dataset name and start/end indices for a given sequential index.
    
    Args:
        dataset_idx (int): Sequential index of the dataset
        
    Returns:
        tuple: (dataset_name, start_idx, end_idx) or None if index not found
    """
    mapping = create_dataset_index_mapping()
    return mapping.get(dataset_idx)

def find_exact_duplicates(dataset, start_idx=0, end_idx=None):
    # Slice the dataset according to the provided indices
    sliced_dataset = dataset[start_idx:end_idx]
    
    # Convert input_ids to numpy array
    token_sequences = np.array(sliced_dataset["input_ids"])
    texts = np.array(sliced_dataset["text"])  # Keep texts for display
    
    # Find unique sequences and their counts
    _, indices, counts = np.unique(token_sequences, axis=0, return_inverse=True, return_counts=True)
    
    # Find which sequences have duplicates (count > 1)
    duplicate_indices = np.where(counts[indices] > 1)[0]
    
    # Group duplicate indices
    groups = {}
    for idx in duplicate_indices:
        seq_tuple = tuple(token_sequences[idx])
        if seq_tuple not in groups:
            groups[seq_tuple] = []
        groups[seq_tuple].append(idx)
    
    # Print the duplicates
    print(f"Found {len(groups)} sequences with duplicates:")
    for seq, indices in groups.items():
        print(f"\nDuplicate indices: {indices}")
        print("Corresponding texts:")
        for idx in indices:
            print(f"[{idx}]: {texts[idx]}")
        
    return groups

def display_random_sentence_pairs(dataset_dict, num_pairs=10, start_idx=0, end_idx=None):
    # Slice the dataset according to the provided indices
    sliced_de = dataset_dict['train.de'][start_idx:end_idx]['text']
    sliced_en = dataset_dict['train.en'][start_idx:end_idx]['text']
    # Get random indices
    random_indices = np.random.choice(len(sliced_de), num_pairs, replace=False)
    de_sentences = [sliced_de[idx] for idx in random_indices]
    en_sentences = [sliced_en[idx] for idx in random_indices]
    
    # Display the pairs
    print(f"Displaying {num_pairs} random sentence pairs:")
    for i, idx in enumerate(random_indices):
        print(f"Sentence pair {i+1} (original index {idx}):")
        print(f"Source: {de_sentences[i]}")
        print(f"Target: {en_sentences[i]}")
        print("\n")

# Example usage
dataset_dict = load_from_disk("artifacts/saved_data/tokenized_data/wmt24_de_en")

# Example: Get dataset info for index 0
dataset_info = get_dataset_slice_by_index(0)
if dataset_info:
    dataset_name, start_idx, end_idx = dataset_info
    print(f"Dataset index 0: {dataset_name}")
    print(f"Start index: {start_idx}, End index: {end_idx}")
    print(f"Number of samples: {end_idx - start_idx}")

# Example: Show all available datasets
print("\nAll available datasets:")
mapping = create_dataset_index_mapping()
for idx, (name, start, end) in mapping.items():
    print(f"Index {idx}: {name} ({start:,} - {end:,}, {end-start:,} samples)")


import pdb; pdb.set_trace()
duplicates = find_exact_duplicates(dataset_dict['train.de'], start_idx=-1000)
display_random_sentence_pairs(dataset_dict, num_pairs=10, start_idx=-1000)