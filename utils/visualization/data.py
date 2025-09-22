import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class DataVisualizer:
    def __init__(self, config):
        self.config = config

    def analyze_token_lengths(self, raw_data):
        """Analyze token lengths using fast tokenizer and track long sentence pairs"""
        if os.path.exists("all_lens.json"):
            with open("all_lens.json", "r") as f:
                return json.load(f)
            
        all_lens = defaultdict(lambda: defaultdict(list))
        long_pairs = defaultdict(list)
        BATCH_SIZE = 100000  # Smaller batch size to manage memory
        
        for split in tqdm(raw_data.keys(), desc="Calculating token lengths"):
            # Get all sentence pairs for this split
            src_sentences = [src for src, _ in raw_data[split]]
            tgt_sentences = [tgt for _, tgt in raw_data[split]]
            
            # Process both languages
            for lang_idx, (language, tokenizer) in enumerate(zip(self.language_pair, [self.tokenizer_src, self.tokenizer_tgt])):
                sentences = src_sentences if lang_idx == 0 else tgt_sentences
                
                # Process in batches to manage memory
                for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc=f"Processing {language}", leave=False):
                    batch = sentences[i:i + BATCH_SIZE]
                    tokenized_batch = tokenizer(
                        batch,
                        padding=False,
                        truncation=False
                    )["input_ids"]
                    
                    # Process batch results
                    batch_lengths = [len(sent) for sent in tokenized_batch]
                    all_lens[language][split].extend(batch_lengths)
                    
                    # Track long sentences
                    for j, length in enumerate(batch_lengths):
                        if length > 200:
                            abs_idx = i + j  # Get absolute index in original list
                            long_pairs[split].append({
                                "index": abs_idx,
                                self.language_pair[0]: src_sentences[abs_idx],
                                self.language_pair[1]: tgt_sentences[abs_idx],
                                'length': length,
                                'language': language
                            })
        
        # Cache results
        json.dump(all_lens, open("all_lens.json", "w"), indent=4)
        json.dump(long_pairs, open("long_pairs.json", "w"), indent=4)
        return all_lens

    def plot_token_lengths(self, raw_data):
        fig, axs = plt.subplots(nrows=len(raw_data.keys()), ncols=len(self.language_pair), 
                            figsize=(15, 5*len(raw_data.keys())), dpi=500)
        
        # Make axs 2D if there's only one language pair
        if len(self.language_pair) == 1:
            axs = np.array([axs]).reshape(-1, 1)

        # Load or calculate token lengths
        all_lens = self.analyze_token_lengths(raw_data)

        # Plot the distributions
        # Calculate global max_len across all splits and languages
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
                
                # Remove extreme outliers (beyond 99.99 percentile)
                length_threshold = np.percentile(lens, 99.99)
                filtered_lens = [l for l in lens if l <= length_threshold]
                
                # Calculate statistics
                quartiles_of_interest = [25, 50, 75, 90, 95, 99]
                quartiles = np.percentile(filtered_lens, quartiles_of_interest)
                mean = np.mean(filtered_lens)
                
                # Plot histogram with better binning
                # max_len = int(np.percentile(filtered_lens, 99.99))
                bins = np.linspace(0, global_max_len, 50)
                
                counts, bins, _ = axs[j, i].hist(filtered_lens, bins=bins, alpha=0.7)
                axs[j, i].grid(True, alpha=0.3)
                axs[j, i].set_title(f"{language} - {split} (Total: {len(filtered_lens):,} sentences)")
                axs[j, i].set_xlabel("Token Length")
                axs[j, i].set_ylabel("Number of Sentences")
                
                # Format y-axis with comma separator
                axs[j, i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                
                # Add vertical lines for statistics
                values = [*quartiles, mean]
                labels = [f'{q}th percentile' for q in quartiles_of_interest] + ['Mean']
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
                
                # Sort by value while keeping labels aligned
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
