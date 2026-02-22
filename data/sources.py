import torch
import os
import os.path as op
import json
from datasets import load_dataset # huggingface datasets
from datasets import Dataset, DatasetDict, Features
from transformers import AutoTokenizer
import random
from datasets import load_from_disk, concatenate_datasets
from datasets.features.translation import Translation

# cosine_distances = squareform(pdist(x, metric='cosine')); near_duplicates = np.where(cosine_distances < threshold); near_duplicates = [(i, j) for i, j in zip(*near_duplicates) if i < j]

def get_indices_for_keys(desired_keys, stats_file="/ocean/projects/cis230090p/npawar/mtdata_cache_np/wmt24-eng-deu/train.stats.json"):
    """
    Reads the stats JSON file and returns a flattened list of all indices for the desired keys.
    
    Args:
        desired_keys: List of dataset keys (e.g., ["Tilde-ema-2016-deu-eng", "Tilde-rapid-2016-deu-eng"])
        stats_file: Path to the stats JSON file
    
    Returns:
        List of all indices from the merged ranges (e.g., [0, 1, 2, 3, 4, 5, 6] for ranges (0,2) and (3,6))
    """
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Get the counts from the "selected" section (or "total" if "selected" is not available)
    counts = stats.get("counts", {}).get("selected", stats.get("counts", {}).get("total", {}))
    
    # Build cumulative index ranges and collect all indices
    current_index = 0
    all_indices = []
    
    for key, count in counts.items():
        if key in desired_keys:
            start_idx = current_index
            end_idx = current_index + count - 1
            # Add all indices from start to end (inclusive)
            all_indices.extend(range(start_idx, end_idx + 1))
        current_index += count
    
    return all_indices

def custom_tokenize_tgt(dataset_dict): 
    return tokenizer_tgt(dataset_dict["text"], padding=True, truncation=True, max_length=70)

def custom_tokenize_src(dataset_dict): 
    return tokenizer_src(dataset_dict["text"], padding=True, truncation=True, max_length=70)

tokenizer_src = AutoTokenizer.from_pretrained("bert-base-german-cased", use_fast=True)
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)

# TODO LIST
# - figure out map with src and tgt together
# - figure out good saving method

# - keep track of original data if needed
# - use take for subsampling
# - see if we can find duplicates using filter


class DataSource:
    def __init__(self, config):
        self.config = config
        self.name = config.dataset.name
        self.language_pair = config.dataset.language_pair
        self.cache = config.dataset.cache
        self.dataset_size = config.dataset.size
        self.max_padding_train = config.model.max_padding_train
        self.max_padding_test = config.model.max_padding_test
        self.random_seed = config.training.random_seed
        self.size = config.dataset.size
        self.test_size = config.dataset.test_size

    def get_data(self):
        '''
        Returns the raw or preprocessed data for a given dataset.
        Optionally loads a specific part of the dataset based on start/end indices.
        '''
        if self.name == "wmt14":
            dataset_dict = load_dataset("wmt14", "-".join(self.language_pair), num_proc=min(os.cpu_count()-1, 8))
            dataset_dict = self._random_sample(dataset_dict, self.size)
            dataset_dict = self._overwrite_test_wmt14(dataset_dict)
            # x = dataset_dict.flatten().map(custom_tokenize_src, batched=True, batch_size=10000, num_proc=8)
            # dataset_dict = load_from_disk("artifacts/saved_data/tokenized_data/wmt14_de_en")    
            # dataset_dict["train"] = dataset_dict["train"].select(range(30000))
        elif self.name == "wmt24":
            dataset_dict = load_from_disk("artifacts/saved_data/tokenized_data/wmt24_de_en_5m")
            dataset_dict = self._reformat_wmt24(dataset_dict)
            dataset_dict = self._random_sample(dataset_dict, self.size + self.test_size)
            dataset_dict = self._overwrite_val_test_wmt24(dataset_dict)
            # dataset_dict.save_to_disk("artifacts/saved_data/tokenized_data/wmt24_de_en_reformatted_new")
        elif self.name == "m30k":
            dataset_dict = load_dataset("bentrevett/multi30k", num_proc=os.cpu_count()-1)
        elif self.name == "txt":
            with open(self.dataset_path, 'r') as f:
                file_text = f.readlines()
                src_sentences = [sentence_pair.split("|")[0].strip() for sentence_pair in file_text]
                tgt_sentences = [sentence_pair.split("|")[1].strip() for sentence_pair in file_text]
                raw_data = list(zip(src_sentences, tgt_sentences))
        else: 
            raise ValueError(f"Received {self.name}, available options: 'wmt14', 'wmt24', 'm30k', 'txt'")
        
        print(f"Selected {len(dataset_dict['train'])} samples from the training dataset")
        return dataset_dict

    def _build_and_tokenize_wmt24(self):
        root_dir = os.getenv("MTDATA")
        data_files = {
            "train.de": op.join(root_dir, "wmt24-eng-deu", "train.deu"), 
                      "train.en": op.join(root_dir, "wmt24-eng-deu", "train.eng"), 
                      "val.de": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt23test.de"), 
                      "val.en": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt23test.en"), 
                      "test.de": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt24test.de"), 
                      "test.en": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt24test.en")}
        dataset_dict = load_dataset("text", data_files=data_files)
        for key in dataset_dict.keys():
            if ".de" in key:
                dataset_dict[key] = dataset_dict[key].map(custom_tokenize_src, batched=True, num_proc=8, batch_size=100000)
            else:
                dataset_dict[key] = dataset_dict[key].map(custom_tokenize_tgt, batched=True, num_proc=8, batch_size=100000)
        dataset_dict.save_to_disk("artifacts/saved_data/tokenized_data/wmt24_de_en")
        pass

    def _overwrite_val_test_wmt24(self, dataset_dict):
        """
        Overwrite the validation set of wmt24 with the test set.
        test becomes old validation 
        """
        new_split = dataset_dict['train'].train_test_split(test_size=self.test_size, seed=self.random_seed)
        dataset_dict['train'] = new_split['train']
        dataset_dict['test'] = dataset_dict['validation'] # wmt24test hand translated becomes test
        dataset_dict['validation'] = new_split['test']
        return dataset_dict

    def _overwrite_test_wmt14(self, dataset_dict):
        """
        Make test set from wmt24 test set for wmt14
        """
        dataset_dict["test"] = load_from_disk("artifacts/saved_data/tokenized_data/wmt24_de_en_val_only") # wmt24test hand translated becomes test
        return dataset_dict

    def _reformat_wmt24(self, dataset_dict):
        """
        Convert wmt24 format to wmt14_debugging format using lazy operations.
        
        Input format:
        - train.de, train.en, val.de, val.en, test.de, test.en
        - Each has features: ['text', 'input_ids', 'token_type_ids', 'attention_mask']
        
        Output format:
        - train, validation, test
        - Each has feature: ['translation'] where translation is a dict with language codes as keys
        
        Uses lazy operations (Dataset.zip + Dataset.map) to avoid loading 300M+ sentences into memory.
        """
        src_lang, tgt_lang = self.language_pair[0], self.language_pair[1]
        
        # Define the mapping from wmt24 splits to wmt14 splits
        split_mapping = {
            'train': 'train',
            'val': 'validation',
            'test': 'test'
        }
        
        new_dataset_dict = {}
        
        for old_split, new_split in split_mapping.items():
            src_key = f"{old_split}.{src_lang}"
            tgt_key = f"{old_split}.{tgt_lang}"
            
            if src_key not in dataset_dict or tgt_key not in dataset_dict:
                print(f"Warning: Missing {src_key} or {tgt_key}, skipping {new_split}")
                continue
            
            src_dataset = dataset_dict[src_key]
            tgt_dataset = dataset_dict[tgt_key]
            
            # Get the minimum length to handle size mismatches
            min_len = min(len(src_dataset), len(tgt_dataset))
            if len(src_dataset) != len(tgt_dataset):
                print(f"Warning: Size mismatch in {old_split}: {src_key} has {len(src_dataset)} rows, {tgt_key} has {len(tgt_dataset)} rows. Using {min_len} rows.")
            
            # Truncate to minimum length (lazy operation)
            src_dataset = src_dataset.select(range(min_len))
            tgt_dataset = tgt_dataset.select(range(min_len))
            
            # Select only 'text' column from both datasets (lazy, reduces memory)
            src_text = src_dataset.select_columns(['text'])
            tgt_text = tgt_dataset.select_columns(['text'])
            
            # Use a generator to pair datasets lazily (truly lazy, no materialization)
            def translation_generator():
                """Generator that yields paired translation examples on demand."""
                for i in range(min_len):
                    yield {
                        'translation': {
                            src_lang: src_text[i]['text'],
                            tgt_lang: tgt_text[i]['text']
                        }
                    }
            
            # Create dataset from generator (lazy - only processes when accessed)
            translation_features = Features({
                'translation': Translation(languages=self.language_pair)
            })
            
            new_dataset = Dataset.from_generator(
                translation_generator,
                features=translation_features
            )
            
            new_dataset_dict[new_split] = new_dataset
        
        return DatasetDict(new_dataset_dict)

    def _random_sample(self, dataset_dict, desired_size):
        random.seed(self.random_seed)
        if desired_size < len(dataset_dict["train"]):
            random_indices = random.sample(range(len(dataset_dict["train"])), desired_size)
            dataset_dict["train"] = dataset_dict["train"].select(random_indices)
        else:
            self.config._update(['dataset.size'], len(dataset_dict['train']))
            print(f"Warning: Desired dataset size {desired_size} is greater than the dataset size {len(dataset_dict['train'])}, using the entire dataset")
        return dataset_dict