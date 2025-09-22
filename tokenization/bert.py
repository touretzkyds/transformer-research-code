from transformers import AutoTokenizer
from utils.config import Config

class HFProcessor:
    def __init__(self, config):
        self.config = config
        self._initialize_mapping()
        self.tokenizer_src = AutoTokenizer.from_pretrained(self.available_tokenizers[self.config.dataset.language_pair[0]], use_fast=True)
        self.tokenizer_tgt = AutoTokenizer.from_pretrained(self.available_tokenizers[self.config.dataset.language_pair[1]], use_fast=True)

    def tokenize_wmt14(self, dataset_dict):
        tokenized_src = self.tokenizer_src(dataset_dict["translation.de"], padding=True, truncation=True, max_length=60)
        tokenized_tgt = self.tokenizer_tgt(dataset_dict["translation.en"], padding=True, truncation=True, max_length=60)
        return {"tokenized_src": tokenized_src, "tokenized_tgt": tokenized_tgt}
        
    def tokenize_m30k(self, dataset_dict):
        tokenized_src = self.tokenizer_src(dataset_dict["de"], padding=True, truncation=True, max_length=60)
        tokenized_tgt = self.tokenizer_tgt(dataset_dict["en"], padding=True, truncation=True, max_length=60)
        return {"tokenized_src": tokenized_src, "tokenized_tgt": tokenized_tgt}
    
    def _initialize_mapping(self):
        language_pair = self.config.dataset.language_pair
        dataset_name = self.config.dataset.name
        self.available_tokenizers = {"en": "bert-base-cased",
                                    "de": "bert-base-german-cased"}

    def process(self, dataset_dict):
        if self.config.dataset.name == "wmt14":
            dataset_dict.map(self.tokenize_wmt14, batched=True, batch_size=10000, num_proc=8)
        elif self.config.dataset.name == "m30k":
            dataset_dict.map(self.tokenize_m30k, batched=True, batch_size=10000, num_proc=8)
        else:
            raise ValueError(f"Received {self.config.dataset.name}, available options: 'wmt14', 'm30k'")