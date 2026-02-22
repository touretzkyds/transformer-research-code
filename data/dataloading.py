import os
import torch
from data.sources import DataSource
from torch.utils.data import DataLoader
from utils.config import Config
from dotted_dict import DottedDict
import random
# from data.preprocess import DataProcessor
# from datasets import DatasetDict
# from data.hf_dataset_processor import process_hf_dataset, collate_batch
# from datasets import DatasetDict
# from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import DataCollatorForSeq2Seq
# from model.full_model import TransformerModel

# class Batch:
#     """
#     Object for holding a batch of data with mask during training.
#     """
#     def __init__(self, pad_idx_tgt, src, tgt=None):
#         self.src = src
#         if tgt is not None:
#             self.tgt_shifted_right = tgt[:, :-1] # everything except last pad token
#             self.tgt_label = tgt[:, 1:] # everything except beginning of sentence token
#             self.decoder_attn_mask = self.make_decoder_attn_mask(
#                 self.tgt_shifted_right, pad_idx_tgt
#             )
#             self.ntokens = (self.tgt_label != pad_idx_tgt).sum()

#     @staticmethod
#     def make_decoder_attn_mask(tgt, pad_idx_tgt):
#         """
#         Create a mask to hide padding and future words.
#         TODO: explain multi mask creation for entire batch
#         """
#         pad_mask = (tgt != pad_idx_tgt).unsqueeze(-2)
#         pad_mask_T = pad_mask.transpose(1,2)
#         subseq_tokens_mask = Batch.get_subseq_tokens_mask(tgt)
#         decoder_attn_mask = pad_mask & subseq_tokens_mask & pad_mask_T
#         return decoder_attn_mask
    
#     @staticmethod
#     def get_subseq_tokens_mask(tgt):
#         """
#         Generate an upper triangular matrix to mask out subsequent positions
#         """
#         mask_shape = (1, tgt.size(-1), tgt.size(-1))
#         upper_tri_matrix = torch.triu(torch.ones(mask_shape), diagonal=1)
#         subseq_tokens_mask = (upper_tri_matrix == 0).type_as(tgt)
#         return subseq_tokens_mask
    
# class RuntimeDatasetOld(Dataset):
#     """Legacy PyTorch dataset for backward compatibility."""
#     def __init__(self, data, config):
#         self.data = data
#         self.config = config
#         self.length = data.shape[0]
#         self.device = config.hardware.device
#         self.pad_idx_tgt = config.extras.tokenizer_tgt.pad_token_id
    
#     def __getitem__(self, i):
#         '''
#         Return a tensor corresponding to a single pair of sentences
#         '''
#         tok_sentence_pair = self.data[i]
#         return tok_sentence_pair
    
#     def __len__(self):
#         '''
#         Return length of entire dataset
#         '''
#         return self.length
    
#     def collate_fn(self, raw_batch):
#         '''
#         Collate a batch from N preprocessed data samples, where N is 
#         the batch size specified in the dataloader.
#         '''
#         batch_tensor = torch.stack(raw_batch)
#         batch_src = batch_tensor[:,0,:].to(self.device)
#         batch_tgt = batch_tensor[:,1,:].to(self.device)
#         return Batch(self.pad_idx_tgt, batch_src, batch_tgt)

# training_args = Seq2SeqTrainingArguments(
#     output_dir="my_awesome_opus_books_model",
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=2,
#     predict_with_generate=True,
#     fp16=True, #change to bf16=True for XPU
#     push_to_hub=False,
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_books["train"],
#     eval_dataset=tokenized_books["test"],
#     processing_class=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

class DataManager:
    def __init__(self, config: Config, tokenizer_src, tokenizer_tgt): # TODO: will these block us? if yes move down to load_dataloaders fn
        self.config = config
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def load_dataloaders(self):
        data_source = DataSource(self.config)
        dataset_dict = data_source.get_data()
        random.seed(self.config.training.random_seed)
        train_dataloader = DataLoader(dataset_dict['train'], batch_size=self.config.training.batch_size, shuffle=self.config.training.shuffle, collate_fn=self.train_collate_fn)
        train_toy_dataloader = DataLoader(dataset_dict['train'].select(random.sample(range(len(dataset_dict['train'])), 5000)), batch_size=self.config.training.batch_size, shuffle=False, collate_fn=self.train_collate_fn)
        val_dataloader = DataLoader(dataset_dict['validation'], batch_size=self.config.training.batch_size, shuffle=False, collate_fn=self.train_collate_fn)
        test_dataloader = DataLoader(dataset_dict['test'], batch_size=self.config.training.batch_size, shuffle=False, collate_fn=self.train_collate_fn)
        return {
            'train': train_dataloader,
            'train_toy': train_toy_dataloader,
            'val': val_dataloader,
            'test': test_dataloader
        }

    def train_collate_fn(self, sentence_pairs):
        tokenizer_src_outputs = self.tokenizer_src([sentence_pair["translation"][self.config.dataset.language_pair[0]] for sentence_pair in sentence_pairs], return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_padding_train)
        tokenizer_tgt_outputs = self.tokenizer_tgt([sentence_pair["translation"][self.config.dataset.language_pair[1]] for sentence_pair in sentence_pairs], return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_padding_train)
        batch = Batch(tokenizer_src_outputs, tokenizer_tgt_outputs, self.tokenizer_tgt.pad_token_id)
        return batch

    def val_collate_fn(self, sentence_pairs):
        tokenizer_src_outputs = self.tokenizer_src([sentence_pair["translation"][self.config.dataset.language_pair[0]] for sentence_pair in sentence_pairs], return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_padding_test)
        tokenizer_tgt_outputs = self.tokenizer_tgt([sentence_pair["translation"][self.config.dataset.language_pair[1]] for sentence_pair in sentence_pairs], return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_padding_test)
        batch = Batch(tokenizer_src_outputs, tokenizer_tgt_outputs, self.tokenizer_tgt.pad_token_id)
        return batch

class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, tokenizer_src_outputs, tokenizer_tgt_outputs, pad_token_id):
        self.src_tokens = tokenizer_src_outputs.input_ids
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
        self.tgt_shifted_right = self.tgt_shifted_right.to(device)
        self.tgt_label = self.tgt_label.to(device)
        self.decoder_attention_mask = self.decoder_attention_mask.to(device)

    def offload(self):
        self.src_tokens = self.src_tokens.cpu()
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