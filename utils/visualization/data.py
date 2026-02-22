import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from utils.config import Config
from data.sources import DataSource

class DataVisualizer:
    def __init__(self, config):
        self.config = config
        self.language_pair = config.dataset.language_pair
        tokenizer_map = {
            "en": "bert-base-cased",
            "de": "bert-base-german-cased"
        }
        self.tokenizer_src = AutoTokenizer.from_pretrained(tokenizer_map[self.language_pair[0]], use_fast=True)
        self.tokenizer_tgt = AutoTokenizer.from_pretrained(tokenizer_map[self.language_pair[1]], use_fast=True)
    
    def _convert_wmt24_format(self, dataset_dict):
        """Convert wmt24 DatasetDict format to expected format"""
        splits = {}
        for key in dataset_dict.keys():
            if "." in key:
                split, lang = key.split(".")
                if split not in splits:
                    splits[split] = {}
                splits[split][lang] = dataset_dict[key]['text']
        
        # Convert to list of (src, tgt) tuples
        result = {}
        for split, langs in splits.items():
            src_lang = self.language_pair[0]
            tgt_lang = self.language_pair[1]
            if src_lang in langs and tgt_lang in langs:
                result[split] = list(zip(langs[src_lang], langs[tgt_lang]))
        return result
        
    def analyze_token_lengths(self, raw_data):
        """Analyze token lengths using fast tokenizer"""
        cache_file = "all_lens.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
            
        all_lens = defaultdict(lambda: defaultdict(list))
        BATCH_SIZE = 100000
        
        for split in tqdm(raw_data.keys(), desc="Calculating token lengths"):
            src_sentences = [src for src, _ in raw_data[split]]
            tgt_sentences = [tgt for _, tgt in raw_data[split]]
            
            for lang_idx, (language, tokenizer) in enumerate(zip(self.language_pair, [self.tokenizer_src, self.tokenizer_tgt])):
                sentences = src_sentences if lang_idx == 0 else tgt_sentences
                
                for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc=f"Processing {language}", leave=False):
                    batch = sentences[i:i + BATCH_SIZE]
                    tokenized_batch = tokenizer(batch, padding=False, truncation=False)["input_ids"]
                    batch_lengths = [len(sent) for sent in tokenized_batch]
                    all_lens[language][split].extend(batch_lengths)
        
        json.dump(all_lens, open(cache_file, "w"), indent=4)
        return all_lens

    def plot_token_lengths(self, raw_data):
        # Convert wmt24 format if needed
        if isinstance(raw_data, dict) and any("." in k for k in raw_data.keys()):
            raw_data = self._convert_wmt24_format(raw_data)
        
        fig, axs = plt.subplots(nrows=len(raw_data.keys()), ncols=len(self.language_pair), 
                            figsize=(15, 5*len(raw_data.keys())), dpi=500)
        
        if len(self.language_pair) == 1:
            axs = np.array([axs]).reshape(-1, 1)

        all_lens = self.analyze_token_lengths(raw_data)

        global_max_len = 0
        for language in self.language_pair:
            for split in raw_data.keys():
                lens = all_lens[language][split]
                length_threshold = np.percentile(lens, 99.99)
                filtered_lens = [l for l in lens if l <= length_threshold]
                split_max_len = int(np.percentile(filtered_lens, 99.99))
                global_max_len = max(global_max_len, split_max_len)
        print(f"Global max length: {global_max_len}")

        for i, language in tqdm(enumerate(self.language_pair), desc="Plotting token lengths", position=0, leave=True):
            for j, split in enumerate(raw_data.keys()):
                lens = all_lens[language][split]
                length_threshold = np.percentile(lens, 99.99)
                filtered_lens = [l for l in lens if l <= length_threshold]
                
                quartiles_of_interest = [25, 50, 75, 90, 95, 99]
                quartiles = np.percentile(filtered_lens, quartiles_of_interest)
                mean = np.mean(filtered_lens)
                
                bins = np.linspace(0, global_max_len, 50)
                axs[j, i].hist(filtered_lens, bins=bins, alpha=0.7)
                axs[j, i].grid(True, alpha=0.3)
                axs[j, i].set_title(f"{language} - {split} (Total: {len(filtered_lens):,} sentences)")
                axs[j, i].set_xlabel("Token Length")
                axs[j, i].set_ylabel("Number of Sentences")
                axs[j, i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                
                values = [*quartiles, mean]
                labels = [f'{q}th percentile' for q in quartiles_of_interest] + ['Mean']
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
                
                sorted_indices = np.argsort(values)
                sorted_values = np.array(values)[sorted_indices]
                sorted_labels = np.array(labels)[sorted_indices]
                sorted_colors = np.array(colors)[sorted_indices]
                
                for label, value, color in zip(sorted_labels, sorted_values, sorted_colors):
                    axs[j, i].axvline(value, linestyle='--', label=f'{label}: {int(value)}', color=color)
                
                axs[j, i].legend(fontsize='small')
                print(f"\n{language} - {split}:")
                print(f"Total sentences: {len(filtered_lens):,}")
                print(f"Filtered out: {len(lens) - len(filtered_lens):,} sentences beyond {int(length_threshold)} tokens")

        plt.tight_layout()
        plt.suptitle(f"Token count statistics for '{self.language_pair}' language pair", fontsize=16, y=1.05, fontweight='bold')
        plt.savefig(f"artifacts/token_lengths_{'-'.join(self.language_pair)}.png")

if __name__ == "__main__":
    config = Config("configs/default.yaml")
    data_visualizer = DataVisualizer(config)
    raw_data = DataSource.get_data(
        config.dataset.name, 
        config.dataset.language_pair, 
        config.dataset.cache, 
        None, 
        config.dataset.size,
    )
    data_visualizer.plot_token_lengths(raw_data)