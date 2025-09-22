import os
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy import pad
import numpy as np
import matplotlib.pyplot as plt # TODO: remove this
from collections import defaultdict
import json

class DataProcessor:
    def __init__(self, tokenizer_src, tokenizer_tgt, config):
        super().__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_padding = config.model.max_padding
        self.language_pair = config.dataset.language_pair
    
    def preprocess_data(self, raw_data):
        '''
        Preprocess raw data sentence by sentence and save to disk.
        Returns: Dict of torch tensors for each split
        '''
        # self.plot_token_lengths(raw_data)
        preprocessed_dataset = {}
        BATCH_SIZE = 1000  # Adjust this based on your available memory
            
        for split in tqdm(raw_data.keys(), desc="Preprocessing dataset"):
            src_sentences = [src for src, _ in raw_data[split]]
            tgt_sentences = [tgt for _, tgt in raw_data[split]]
            
            # Initialize empty tensors for this split
            all_src_tokens = []
            all_tgt_tokens = []
            
            # Process in batches
            for i in tqdm(range(0, len(src_sentences), BATCH_SIZE), desc=f"Processing {split}", leave=False):
                src_batch = src_sentences[i:i + BATCH_SIZE]
                tgt_batch = tgt_sentences[i:i + BATCH_SIZE]
                
                tokenized_src = src_batch.map(self.tokenizer_src, batched=True)
                
                tokenized_tgt = self.tokenizer_tgt(
                    tgt_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_padding,
                    return_tensors="pt",
                )["input_ids"]
                
                all_src_tokens.append(tokenized_src)
                all_tgt_tokens.append(tokenized_tgt)
            
            # Concatenate all batches
            src_tokens = torch.cat(all_src_tokens, dim=0)
            tgt_tokens = torch.cat(all_tgt_tokens, dim=0)
            preprocessed_dataset[split] = torch.stack([src_tokens, tgt_tokens], dim=1)
            
            # Optional: Clear memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return preprocessed_dataset
    
    @staticmethod
    def split_data(data, split_ratio=(0.8, 0.1, 0.1), random_seed=None):
        '''
        Splits a given dataset into train, validation and test sets as
        determined by the specified split ratio
        TODO: deprecate this if split is done in get_data
        '''
        train_size, val_size, test_size = split_ratio
        train_data, val_and_test_data = train_test_split(
            data,
            train_size=train_size,
            random_state=random_seed
        )
        val_data, test_data = train_test_split(
            val_and_test_data,
            train_size=val_size/(val_size+test_size),
            random_state=random_seed
        )
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
        return train_data, val_data, test_data

    @staticmethod
    def limit_train_size(data, dataset_size, max_padding, part_start_index=None, part_end_index=None):
        """
        Limit the size of the training dataset and the max padding length.
        Optionally load a specific part of the dataset based on start/end indices.
        
        Args:
            data: Dict of torch tensors for each split
            dataset_size: Int, maximum size of the training dataset
            max_padding: Int, maximum padding length
            part_start_index: Optional start index for loading a specific part
            part_end_index: Optional end index for loading a specific part
        Returns:
            data: Dict of torch tensors for each split, with limited size and max padding
        """
        original_dataset_size, _, original_max_padding = data['train'].shape
        
        # Handle part-specific loading
        if part_start_index is not None and part_end_index is not None:
            # Validate indices
            if part_start_index < 0 or part_end_index > original_dataset_size:
                raise ValueError(f"Part indices {part_start_index} to {part_end_index} out of range for dataset size {original_dataset_size}")
            if part_start_index >= part_end_index:
                raise ValueError(f"Invalid part range: start {part_start_index} >= end {part_end_index}")
            
            # Load the specific part
            data['train'] = data['train'][part_start_index:part_end_index, :, :]
            print(f"Loaded part of dataset: indices {part_start_index:,} to {part_end_index:,} (size: {part_end_index - part_start_index:,})")
        else:
            # Original behavior: limit train size to dataset_size
            dataset_size = min(dataset_size, original_dataset_size)
            data['train'] = data['train'][:dataset_size, :, :]
            print(f"Limited dataset to first {dataset_size:,} samples")
        
        # limit training data to max_padding tokens
        max_padding = min(max_padding, original_max_padding)
        data['train'] = data['train'][:, :, :max_padding]
        
        return data