from transformers import AutoTokenizer
from utils.config import Config

def build_tokenizers(config: Config):
    available_tokenizers = {"en": "dbmdz/bert-base-german-cased",
                            "de": "bert-base-german-cased"}
    tokenizer_src = AutoTokenizer.from_pretrained(available_tokenizers[config.dataset.language_pair[0]], use_fast=True)
    tokenizer_tgt = AutoTokenizer.from_pretrained(available_tokenizers[config.dataset.language_pair[1]], use_fast=True)
    print(f"Src vocab size: {len(tokenizer_src.vocab)}")
    print(f"Tgt vocab size: {len(tokenizer_tgt.vocab)}")
    return tokenizer_src, tokenizer_tgt