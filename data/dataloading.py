import os
import torch
from data.sources import DataSource
from torch.utils.data import Dataset, DataLoader
from data.preprocess import DataProcessor
from utils.config import Config

class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, pad_idx_tgt, src, tgt=None):
        self.src = src
        # TODO: remove this HARVARD code line
        self.src_mask = (src != pad_idx_tgt).unsqueeze(-2)
        if tgt is not None:
            self.tgt_shifted_right = tgt[:, :-1] # everything except last pad token
            self.tgt_label = tgt[:, 1:] # everything except beginning of sentence token
            self.decoder_attn_mask = self.make_decoder_attn_mask(
                self.tgt_shifted_right, pad_idx_tgt
            )
            self.ntokens = (self.tgt_label != pad_idx_tgt).sum()

    @staticmethod
    def make_decoder_attn_mask(tgt, pad_idx_tgt):
        """
        Create a mask to hide padding and future words.
        TODO: explain multi mask creation for entire batch
        """
        pad_mask = (tgt != pad_idx_tgt).unsqueeze(-2)
        pad_mask_T = pad_mask.transpose(1,2)
        subseq_tokens_mask = Batch.get_subseq_tokens_mask(tgt)
        decoder_attn_mask = pad_mask & subseq_tokens_mask & pad_mask_T
        return decoder_attn_mask
    
    @staticmethod
    def get_subseq_tokens_mask(tgt):
        """
        Generate an upper triangular matrix to mask out subsequent positions
        """
        mask_shape = (1, tgt.size(-1), tgt.size(-1))
        upper_tri_matrix = torch.triu(torch.ones(mask_shape), diagonal=1)
        subseq_tokens_mask = (upper_tri_matrix == 0).type_as(tgt)
        return subseq_tokens_mask
    
class RuntimeDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.length = data.shape[0]
        self.device = config.hardware.device
        self.pad_idx_tgt = config.extras.tokenizer_tgt.pad_token_id
    
    def __getitem__(self, i):
        '''
        Return a tensor corresponding to a single pair of sentences
        '''
        tok_sentence_pair = self.data[i]
        return tok_sentence_pair
    
    def __len__(self):
        '''
        Return length of entire dataset
        '''
        return self.length
    
    def collate_fn(self, raw_batch):
        '''
        Collate a batch from N preprocessed data samples, where N is 
        the batch size specified in the dataloader.
        '''
        batch_tensor = torch.stack(raw_batch)
        batch_src = batch_tensor[:,0,:].to(self.device)
        batch_tgt = batch_tensor[:,1,:].to(self.device)
        return Batch(self.pad_idx_tgt, batch_src, batch_tgt)
    
        
def load_datasets(tokenizer_src, tokenizer_tgt, config: Config):
    '''
    A utility function that sources the preprocessed data, calls a split on 
    it, generates runtime dataset splits for training, validation and testing.
    '''
    print(f'Loading dataset {config.dataset.name}')
    data_processor = DataProcessor(tokenizer_src, tokenizer_tgt, config)
    print("Preprocessing data")
    
    # Get part indices from config if they exist
    part_start_index = getattr(config, 'part_start_index', None)
    part_end_index = getattr(config, 'part_end_index', None)
    
    preprocessed_data = DataSource.get_data(config.dataset.name, 
                                            config.dataset.language_pair, 
                                            config.dataset.cache, 
                                            data_processor,
                                            config.dataset.size,
                                            config.model.max_padding,
                                            config.training.random_seed,
                                            part_start_index,
                                            part_end_index)

    train_dataset = RuntimeDataset(preprocessed_data['train'], config)
    val_dataset = RuntimeDataset(preprocessed_data['val'], config)
    test_dataset = RuntimeDataset(preprocessed_data['test'], config)

    print(f"Number of sentence pairs: \n"
          f"Training: {train_dataset.length}\t"
          f"Validation: {val_dataset.length}\t"
          f"Test: {test_dataset.length}\t")

    return train_dataset, val_dataset, test_dataset

    
def load_dataloaders(tokenizer_src, tokenizer_tgt, config: Config):
    '''
    A utility function that takes runtime dataset splits and creates 
    corresponding train, validation and test dataloaders that consume the 
    dataset splits and batch them at runtime for the model.
    '''
    train_dataset, val_dataset, test_dataset = load_datasets(tokenizer_src, tokenizer_tgt, config)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=config.training.batch_size, 
                                  shuffle=config.training.shuffle,
                                  collate_fn=train_dataset.collate_fn,
                                  # num_workers=5,
                                  # persistent_workers=True,
                                  # pin_memory=True,
                                  # prefetch_factor=3,
                                  )
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=config.training.batch_size, 
                                shuffle=False,
                                collate_fn=val_dataset.collate_fn,
                                # num_workers=5,
                                # persistent_workers=True,
                                # pin_memory=True,
                                # prefetch_factor=3,
                                )
    
    test_dataloader = DataLoader(dataset=test_dataset, 
                                batch_size=config.training.batch_size, 
                                shuffle=False,
                                collate_fn=test_dataset.collate_fn)
    
    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
    return dataloaders
