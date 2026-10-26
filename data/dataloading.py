import os
import torch
from functools import partial
from data.sources import DataSource
from data.sampler import MAX_TOKENS_PER_SENTENCE, TokenBudgetBatchSampler, token_lengths
from torch.utils.data import DataLoader
from utils.config import Config

class DataManager:
    def __init__(self, config: Config, tokenizer_src, tokenizer_tgt): # TODO: will these block us? if yes move down to load_dataloaders fn
        self.config = config
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def load_dataloaders(self):
        data_source = DataSource(self.config)
        dataset_dict = data_source.get_data()
        if self.config.training.dynamic_batching:
            lengths = token_lengths(dataset_dict['train'], self.tokenizer_src, self.tokenizer_tgt, self.config.dataset.language_pair)
            batch_sampler = TokenBudgetBatchSampler(lengths, self.config.training.random_seed)
            train_dataloader = DataLoader(dataset_dict['train'], batch_sampler=batch_sampler, collate_fn=partial(self.collate_fn, max_length=MAX_TOKENS_PER_SENTENCE))
        else:
            shuffle_generator = torch.Generator()
            shuffle_generator.manual_seed(self.config.training.random_seed)
            train_dataloader = DataLoader(dataset_dict['train'], batch_size=self.config.training.batch_size, shuffle=self.config.training.shuffle, collate_fn=self.collate_fn, generator=shuffle_generator)
        train_toy_dataloader = DataLoader(dataset_dict['train_sampled'], batch_size=self.config.training.batch_size, shuffle=False, collate_fn=self.collate_fn)
        val_dataloader = DataLoader(dataset_dict['validation'], batch_size=self.config.training.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_dataloader = DataLoader(dataset_dict['test'], batch_size=self.config.training.batch_size, shuffle=False, collate_fn=self.collate_fn)
        return {
            'train': train_dataloader,
            'train_toy': train_toy_dataloader,
            'val': val_dataloader,
            'test': test_dataloader
        }

    def collate_fn(self, sentence_pairs, max_length=None):
        max_length = max_length or self.config.model.max_padding
        tokenizer_src_outputs = self.tokenizer_src([sentence_pair["translation"][self.config.dataset.language_pair[0]] for sentence_pair in sentence_pairs], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokenizer_tgt_outputs = self.tokenizer_tgt([sentence_pair["translation"][self.config.dataset.language_pair[1]] for sentence_pair in sentence_pairs], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return Batch(tokenizer_src_outputs, tokenizer_tgt_outputs, self.tokenizer_tgt.pad_token_id)

class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, tokenizer_src_outputs, tokenizer_tgt_outputs, pad_token_id):
        self.src_tokens = tokenizer_src_outputs.input_ids
        self.src_mask = tokenizer_src_outputs.attention_mask.unsqueeze(-2) # (batch, 1, src_len): 0 at padding
        self.tgt_tokens = tokenizer_tgt_outputs.input_ids
        self.ntokens = tokenizer_tgt_outputs.attention_mask.sum()
        self.pad_token_id = pad_token_id
    
    def create_decoder_attention_mask(self, tgt_tokens, pad_id):
        pad_mask = (tgt_tokens != pad_id).unsqueeze(-2) # match tgt shifted right shape
        pad_mask_T = pad_mask.transpose(1,2)
        # Create causal mask: False above diagonal (to mask future tokens), True on/below diagonal
        # Original code used (upper_tri_matrix == 0) which gives False above diagonal
        upper_tri = torch.triu(torch.ones(pad_mask.shape[-1], pad_mask.shape[-1], device=tgt_tokens.device), diagonal=1)
        subseq_tokens_mask = (upper_tri == 0).bool().unsqueeze(0)  # False above diagonal, True on/below
        self.decoder_attention_mask = pad_mask & subseq_tokens_mask & pad_mask_T

    def shift_tgts(self):
        self.tgt_shifted_right = self.tgt_tokens[:, :-1] # everything except last pad token
        self.tgt_label = self.tgt_tokens[:, 1:] # everything except beginning of sentence token

    def to(self, device):
        self.src_tokens = self.src_tokens.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt_shifted_right = self.tgt_shifted_right.to(device)
        self.tgt_label = self.tgt_label.to(device)
        self.decoder_attention_mask = self.decoder_attention_mask.to(device)

    def offload(self):
        self.src_tokens = self.src_tokens.cpu()
        self.src_mask = self.src_mask.cpu()
        self.tgt_tokens = self.tgt_tokens.cpu()
        self.tgt_shifted_right = self.tgt_shifted_right.cpu()
        self.tgt_label = self.tgt_label.cpu()
        self.decoder_attention_mask = self.decoder_attention_mask.cpu()

    def prepare_for_training(self):
        self.shift_tgts() # align tgt for training
        self.create_decoder_attention_mask(self.tgt_shifted_right, self.pad_token_id)

    def prepare_for_inference(self, predictions_with_bos):
        self.shift_tgts() # align tgt for training # TODO remove as redundant
        self.create_decoder_attention_mask(predictions_with_bos, self.pad_token_id)