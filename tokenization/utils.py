import os
import spacy
import torch
import os.path as op
from transformers import AutoTokenizer
from utils.config import Config

def build_tokenizers(config: Config):
    language_pair = config.dataset.language_pair
    tokenizer_type = config.tokenizer.type
    cache = config.tokenizer.cache  
    cache_path = op.join(config.logging.artifacts_dir, 
                         config.logging.subdirs.tokenizers, 
                         f"{tokenizer_type}_tokenizers.pt")
    if (cache and op.exists(cache_path)):
        tokenizer_src, tokenizer_tgt = torch.load(cache_path)
        print(f"Loaded saved tokenizers from {cache_path}")
    else:
        tokenizer_src, tokenizer_tgt = build_bert_tokenizers(language_pair)
        torch.save((tokenizer_src, tokenizer_tgt), cache_path)
        print(f"Saved tokenizers to {cache_path}")
    print(f"Src vocab size: {len(tokenizer_src.vocab)}")
    print(f"Tgt vocab size: {len(tokenizer_tgt.vocab)}")
    return tokenizer_src, tokenizer_tgt

def build_bert_tokenizers(language_pair):
    available_tokenizers = {"en": "dbmdz/bert-base-german-cased",
                            "de": "bert-base-german-cased"}
    tokenizer_src = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[0]], use_fast=True)
    tokenizer_tgt = AutoTokenizer.from_pretrained(available_tokenizers[language_pair[1]], use_fast=True)
    return tokenizer_src, tokenizer_tgt