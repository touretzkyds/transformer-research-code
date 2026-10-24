import sys
import site
_user_site = site.getusersitepackages()
if _user_site in sys.path:
    sys.path.remove(_user_site)

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
from multiprocessing import set_start_method
from torch.optim.lr_scheduler import LambdaLR
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# import project modules
from tokenization.utils import build_tokenizers
from data.dataloading import DataManager
from training.logging import TrainingLogger
from training.loss import LabelSmoothing, SimpleLossCompute
from training.utils import get_learning_rate
from inference.utils import BleuUtils
from model.utils import create_model
from utils.config import Config
from utils.device import resolve_device

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = create_model(config)
        self.tokenizer_src = config.extras.tokenizer_src
        self.tokenizer_tgt = config.extras.tokenizer_tgt
        self.pad_token_id_tgt = self.tokenizer_tgt.pad_token_id
        self.criterion = SimpleLossCompute(LabelSmoothing(config.model.tgt_vocab_size, self.pad_token_id_tgt, config.training.label_smoothing))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate.base, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = LambdaLR(optimizer = self.optimizer, lr_lambda = lambda step_num: get_learning_rate(step_num+1, config.model.d_model, warmup=config.training.learning_rate.warmup_steps))
        self.logger = TrainingLogger(config)
        self.data_mgr = DataManager(config, self.tokenizer_src, self.tokenizer_tgt)
        self.dataloaders = self.data_mgr.load_dataloaders()
        self.bleu_utils = BleuUtils(config, self.logger)
        self.device = resolve_device(config.hardware.device)
        self.accum_iter = config.training.accumulation_steps
        self.num_train_batches = len(self.dataloaders['train'])
        self.num_val_batches = len(self.dataloaders['val'])
        self.num_test_batches = len(self.dataloaders['test'])
        self.print_frequency = max(self.num_train_batches // 8, 1)
        self.updates_per_epoch = math.ceil(self.num_train_batches / self.accum_iter)
        self.max_updates = config.training.max_updates
        self.eval_every = config.training.eval_every
        self.update_step = 0

    def train(self):
        # every dataset size trains for the same number of weight updates, so small sets loop over more epochs
        self.run_t0 = time.time()
        self.eval_t0 = time.time()
        epoch = 0
        while self.update_step < self.max_updates:
            epoch += 1
            print(f"Epoch: {epoch}")
            self.run_train_epoch()
        elapsed = time.time() - self.run_t0
        self.logger.log_run_time(elapsed)
        print(f"Training complete | Total time: {elapsed/3600:.2f} h ({elapsed/60:.1f} min)")

    def evaluate(self):
        train_bleu = self.bleu_utils.evaluate_bleu(self.model, self.dataloaders['train_toy'], self.update_step, "train")
        val_loss, val_bleu = self.run_val_epoch()
        self.logger.log_metric("train_bleu", train_bleu, self.update_step)
        self.logger.log_metric("val_loss", val_loss, self.update_step)
        self.logger.log_metric("val_bleu", val_bleu, self.update_step)

        elapsed = time.time() - self.run_t0
        print(f"Update: {self.update_step}/{self.max_updates} | "
              f"Epoch: {self.update_step / self.updates_per_epoch:.2f} | "
              f"Train BLEU: {train_bleu:.2f} | "
              f"Validation: Loss: {val_loss:.3f}, BLEU: {val_bleu:.2f} | "
              f"Time since last eval: {(time.time() - self.eval_t0)/60:.2f} min | "
              f"Total: {elapsed/60:.2f} min")
        print("="*100)

        self.logger.save_model(self.model, self.update_step)
        self.logger.log_run_time(elapsed)
        self.logger.saveplot(self.updates_per_epoch, metric_names=['train_loss', 'val_loss'], title='Loss', plot_type='loss')
        self.logger.saveplot(self.updates_per_epoch, metric_names=['train_bleu', 'val_bleu'], title='BLEU', plot_type='bleu')
        self.model.train()
        self.eval_t0 = time.time()

    def run_train_epoch(self):
        self.model.train()
        window_loss = 0.0
        window_count = 0
        for i, batch in enumerate(self.dataloaders['train']):
            batch.prepare_for_training()
            output_logprobabilities = self.model.forward(batch.src_tokens.to(self.device), 
                                                batch.tgt_shifted_right.to(self.device),
                                                batch.decoder_attention_mask.to(self.device))
            # compute loss and BLEU score
            loss = self.criterion(output_logprobabilities, batch.tgt_label.to(self.device), batch.ntokens)
            # backpropagate and apply optimizer-based gradient descent 
            loss.backward()
            window_loss += loss.item()
            window_count += 1
            
            if i % self.accum_iter == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.update_step += 1
                self.logger.log_metric("train_loss", window_loss / window_count, self.update_step)
                window_loss = 0.0
                window_count = 0
                if self.update_step % self.eval_every == 0:
                    self.evaluate()
            
            # step learning rate (every batch, not just when optimizer steps)
            self.scheduler.step()
            
            # print metrics
            if i % self.print_frequency == 0:
                print(f"Batch: {i+1}/{self.num_train_batches} \t|\t"
                      f"Training loss: {loss.detach().cpu().numpy():.3f} \t|\t"
                      f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Clean up intermediate tensors
            del output_logprobabilities, loss
            batch.offload()
            if self.update_step == self.max_updates:
                break

    def run_val_epoch(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloaders['val']):
                batch.prepare_for_training()
                output_logprobabilities = self.model.forward(batch.src_tokens.to(self.device), 
                                                            batch.tgt_shifted_right.to(self.device),
                                                            batch.decoder_attention_mask.to(self.device))
                # compute loss
                loss = self.criterion(output_logprobabilities, batch.tgt_label.to(self.device), batch.ntokens)
                # accumulate loss
                epoch_loss += loss.detach().cpu().numpy()
                # Clean up intermediate tensors
                del output_logprobabilities, loss
                batch.offload()
                del batch
                # Periodically clear cache during validation for large datasets
                if torch.cuda.is_available() and self.num_val_batches > 100 and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        # average the metrics
        epoch_loss /= self.num_val_batches
        bleu = self.bleu_utils.evaluate_bleu(self.model, self.dataloaders['val'], self.update_step, "val")
        return epoch_loss, bleu

    def run_test_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.dataloaders['test']):
                batch.prepare_for_training()
                output_logprobabilities = self.model.forward(batch.src_tokens.to(self.device), 
                                                            batch.tgt_shifted_right.to(self.device),
                                                            batch.decoder_attention_mask.to(self.device))
                # compute loss
                loss = self.criterion(output_logprobabilities, batch.tgt_label.to(self.device), batch.ntokens)
                # accumulate loss
                epoch_loss += loss.detach().cpu().numpy()
                # Clean up intermediate tensors
                del output_logprobabilities, loss
                batch.offload()
                del batch
                # Periodically clear cache during validation for large datasets
                if torch.cuda.is_available() and self.num_test_batches > 100 and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        # average the metrics
        epoch_loss /= self.num_test_batches
        bleu = self.bleu_utils.evaluate_bleu(self.model, self.dataloaders['test'], epoch, "test")
        self.logger.log_metric("test_bleu", bleu, self.update_step)
        return epoch_loss, bleu
    
if __name__ == "__main__":
    config = Config("configs/default.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=config.training.max_updates)
    parser.add_argument("--eval_every", type=int, default=config.training.eval_every)
    parser.add_argument("--n_layers", type=int, default=config.model.n_layers)
    parser.add_argument("--language_pair", type=tuple, default=config.dataset.language_pair)
    parser.add_argument("--batch_size", type=int, default=config.training.batch_size)
    parser.add_argument("--max_padding", type=int, default=config.model.max_padding)
    parser.add_argument("--dataset_name", type=str, choices=["wmt14", "wmt24", "m30k", "toy", "txt"], default=config.dataset.name)
    parser.add_argument("--dataset_size", type=int, default=config.dataset.size)
    parser.add_argument("--random_seed", type=int, default=config.training.random_seed)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default=config.hardware.device)
    
    args = parser.parse_args()
    config.update_from_args(args)
    config.print()

    set_seed(config.training.random_seed)
    
    tokenizer_src, tokenizer_tgt = build_tokenizers(config)
    config.model.src_vocab_size = len(tokenizer_src.vocab)
    config.model.tgt_vocab_size = len(tokenizer_tgt.vocab)
    config.extras.tokenizer_src = tokenizer_src
    config.extras.tokenizer_tgt = tokenizer_tgt

    trainer = Trainer(config)
    model = trainer.model  # useful for user inspection
    trainer.train()
