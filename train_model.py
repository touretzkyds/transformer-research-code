import os
import time
import json
import torch
import torch.nn as nn
import argparse
from torch.optim.lr_scheduler import LambdaLR

from tokenization.utils import build_tokenizers
from data.dataloading import load_datasets, load_dataloaders

from training.logging import DirectoryCreator, TrainingLogger
from training.loss import LabelSmoothing, SimpleLossCompute
from training.utils import get_learning_rate
from inference.utils import BleuUtils
from model.utils import count_params, create_model, create_config
from utils.config import Config
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self._update_config_from_args()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._init_tokenizers()
        self._init_model()
        self._init_datasets()
        self._init_training_components()

    def _update_config_from_args(self):
        """Update config from command line arguments"""
        args = argparse.ArgumentParser()
        self.config.update_from_args(args)

    def _init_tokenizers(self):
        """Initialize tokenizers based on config"""
        self.tokenizer_src, self.tokenizer_tgt = build_tokenizers(self.config)
        self.config.model.src_vocab_size = len(self.tokenizer_src.vocab)
        self.config.model.tgt_vocab_size = len(self.tokenizer_tgt.vocab)
        self.config.extras.tokenizer_src = self.tokenizer_src
        self.config.extras.tokenizer_tgt = self.tokenizer_tgt

    def _init_model(self):
        """Initialize the transformer model"""
        if self.config.extras.harvard:
            from harvard import make_model
            self.model = make_model(
                self.config.model.src_vocab_size,
                self.config.model.tgt_vocab_size,
                self.config.model.N,
                self.config.model.d_model,
                self.config.model.d_ff,
                self.config.model.n_heads,
                self.config.model.dropout_prob
            ).to(self.device)
        else:
            self.model = create_model(self.config).to(self.device)

    def _init_datasets(self):
        """Initialize train/val/test datasets and dataloaders"""
        # datasets = load_datasets(
        #     self.config.dataset.name,
        #     self.config.dataset.language_pair,
        #     self.tokenizer_src,
        #     self.tokenizer_tgt,
        #     self.config.model.max_padding,
        #     device=self.device,
        #     cache=self.config.dataset.cache,
        #     random_seed=self.config.training.random_seed,
        #     dataset_size=self.config.dataset.size
        # )
        
        self.train_loader, self.val_loader, self.test_loader = load_dataloaders(
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config
        )

    def _init_training_components(self):
        # Logger
        self.logger = TrainingLogger(self.config)

        # Loss
        label_smoothing = LabelSmoothing(
            len(self.tokenizer_tgt.vocab),
            self.tokenizer_tgt.pad_token_id,
            0.1
        ).to(self.device)
        self.criterion = SimpleLossCompute(label_smoothing)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate.base,
            betas=(0.9, 0.98),
            eps=1e-9
        ).to(self.device)

        # Scheduler
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: get_learning_rate(
                step + 1,
                self.config.model.d_model,
                warmup=self.config.training.learning_rate.warmup_steps
            )
        ).to(self.device)

    def train(self):
        """Main training loop"""
        for epoch in range(1, self.config.training.epochs + 1):
            print(f"Epoch: {epoch}")
            start = time.time()

            # Training
            train_loss, train_bleu = self._run_epoch(
                self.train_loader,
                is_training=True,
                epoch=epoch
            )

            # Validation
            val_loss, val_bleu = self._run_epoch(
                self.val_loader,
                is_training=False
            )

            # Log metrics
            self._log_epoch_metrics(epoch, train_loss, train_bleu, val_loss, val_bleu, start)
            
            # Save model checkpoint

# class Trainer:
#     def __init__(self, config: Config):
#         self.config = config
#         self._update_config_from_args()
#         self.logger = TrainingLogger(config)



def train(dataloaders, model, criterion, 
          optimizer, scheduler, config):
    # initiate logger for saving metrics
    logger = TrainingLogger(config)
    # start training
    # TODO: introduce torch.autocast
    for epoch in range(1, config.training.epochs+1): # epochs are 1-indexed
        save_frequency = 2
        print(f"Epoch: {epoch}")
        # initialize timer
        start = time.time()
        # training
        train_loss, train_bleu = run_train_epoch(dataloaders['train'], 
                                                 model, 
                                                 criterion, 
                                                 optimizer, 
                                                 scheduler, 
                                                 config.training.accumulation_steps, 
                                                 logger,
                                                 epoch)
        # validation
        val_loss, val_bleu = run_val_epoch(dataloaders['val'], model, criterion)

        # accumulate loss history
        logger.log_metric('val_loss', val_loss, epoch)
        logger.log_metric('val_bleu', val_bleu, epoch)

        # print losses
        print(f"Epoch: {epoch} | "
              f"Training: Loss: {train_loss:.3f}, BLEU: {train_bleu:.3f} | "
              f"Validation: Loss: {val_loss:.3f}, BLEU: {val_bleu:.3f} | "
              f"Time taken: {1/60*(time.time() - start):.2f} min")
        print("="*100)

        # save model
        if epoch % save_frequency == 0:
            if HARVARD:
                save_path = f'{config.logging.model_dir}/N{config.model.N}/harvard/{config.dataset.name}/dataset_size_{config.dataset.size}/epoch_{epoch:02d}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            else:
                torch.save(model,
                           f'{config.logging.model_dir}/N{config.model.N}/{config.dataset.name}/dataset_size_{config.dataset.size}/epoch_{epoch:02d}.pt')

        # plot and save loss curves
        # log loss v/s weight updates
        logger.saveplot(epoch, 
                        metric_names=['train_loss', 'val_loss'], 
                        title='Loss', 
                        title_dict={k:config[k] for k in ['batch_size', 'N', 'dataset_size']}, 
                        plot_type='loss', 
                        xlabel='Weight Update',
                        )
        logger.saveplot(epoch,
                        metric_names=['train_bleu', 'val_bleu'], 
                        title='BLEU', 
                        title_dict={k:config[k] for k in ['batch_size', 'N', 'dataset_size']}, 
                        plot_type='bleu', 
                        xlabel='Weight Update',
                        )
    print("Training complete")

def run_train_epoch(train_dataloader, model, criterion, 
                    optimizer, scheduler, accum_iter, logger, epoch_num):
    # COMPUTE_BLEU_FREQ = 100
    # put model in training mode
    model.train()
    # iterate over the training data and compute losses
    epoch_loss = 0
    num_batches = len(train_dataloader)
    print_frequency = max(num_batches // 8, 1) # for printing progress
    for i, batch in enumerate(train_dataloader):
        if HARVARD:
            output_logprobabilities = model.forward(batch.src, 
                                                    batch.tgt_shifted_right,
                                                    batch.src_mask,
                                                    batch.decoder_attn_mask)
        else:
            output_logprobabilities = model.forward(batch.src, 
                                                    batch.tgt_shifted_right,
                                                    batch.decoder_attn_mask)
        # compute loss and BLEU score
        loss = criterion(output_logprobabilities, batch.tgt_label, batch.ntokens)
        # backpropagate and apply optimizer-based gradient descent 
        loss.backward()
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # log train loss after every weight update
            logger.log_metric("train_loss", loss.item(), i // accum_iter)
        
        # step learning rate
        scheduler.step()
        # accumulate loss
        epoch_loss += loss.detach().cpu().numpy()
        
        # print metrics
        if i % print_frequency == 0:
            print(f"Batch: {i+1}/{num_batches} \t|\t"
                  f"Training loss: {loss.detach().cpu().numpy():.3f} \t|\t"
                  f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        del batch
    # average the metrics
    epoch_loss /= num_batches
    # evaluate BLEU score on random batches for computational efficiency
    bleu = BleuUtils.evaluate_bleu_random_batches(model, train_dataloader, tokenizer_tgt, HARVARD)
    logger.log_metric("train_bleu", bleu, epoch_num)
    return epoch_loss, bleu

def run_val_epoch(val_dataloader, model, criterion):
    # put model in evaluation mode
    model.eval()
    num_batches = len(val_dataloader)
    # iterate over the validation data and compute losses
    epoch_loss = 0
    for batch in val_dataloader:
        if HARVARD:
            output_logprobabilities = model.forward(batch.src, 
                                                    batch.tgt_shifted_right,
                                                    batch.src_mask,
                                                    batch.decoder_attn_mask)
        else:
            output_logprobabilities = model.forward(batch.src, 
                                                    batch.tgt_shifted_right,
                                                    batch.decoder_attn_mask)
        # compute loss
        loss = criterion(output_logprobabilities, batch.tgt_label, batch.ntokens)
        # accumulate loss
        epoch_loss += loss.detach().cpu().numpy()
        del batch
    
    # average the metrics
    epoch_loss /= num_batches
    # evaluate BLEU score on random batches for computational efficiency
    bleu = BleuUtils.evaluate_bleu_random_batches(model, val_dataloader, tokenizer_tgt, HARVARD)
    return epoch_loss, bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--language_pair", type=tuple, default=("de", "en"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_padding", type=int, default=20)
    parser.add_argument("--dataset_name", type=str, choices=["wmt14", "m30k", "txt", "wmt24"], default="wmt14")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--dataset_size", type=int, default=5000000)
    parser.add_argument("--random_seed", type=int, default=40)
    parser.add_argument("--experiment_name", type=str, default="default experiment")
    parser.add_argument("--HARVARD", action="store_true")
    parser.add_argument("--tokenizer_type", type=str, choices=["spacy", "bert"])
    
    # parser.add_argument("--run_name", type=str, default="default run")
    args = parser.parse_args()
    config = Config("configs/default.yaml")
    config.update_from_args(args)
    config.print()

    # if missing, create directories required for saving artifacts
    DirectoryCreator.create_dirs(['saved_tokenizers', 
                                  'saved_data/raw_data',
                                  'saved_data/preprocessed_data',
                                  f'saved_models/N{args.N}/{args.dataset_name}/dataset_size_{args.dataset_size}', 
                                  f'loss_curves/N{args.N}']) # /{args.dataset_name}

    HARVARD = args.HARVARD
    print(f"HARVARD: {HARVARD}")
    
    # load tokenizers and vocabulary
    if HARVARD:
        tokenizer_type = "spacy"
    else:
        tokenizer_type = args.tokenizer_type
    tokenizer_src, tokenizer_tgt = build_tokenizers(config)
    config.model.src_vocab_size = len(tokenizer_src.vocab)
    config.model.tgt_vocab_size = len(tokenizer_tgt.vocab)
    config.extras.tokenizer_src = tokenizer_src
    config.extras.tokenizer_tgt = tokenizer_tgt

    if HARVARD:
        from harvard import EncoderDecoder, Encoder, Decoder, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, make_model, Generator, EncoderLayer, DecoderLayer, Sublayer, attention_fn
        from harvard import LayerNorm, Embeddings
        model = make_model(config.model.src_vocab_size, config.model.tgt_vocab_size, config.model.N, config.model.d_model, config.model.d_ff, config.model.n_heads, config.model.dropout_prob)
    else:
        # initialize model
        model = create_model(config)
    print(f"Model: \n{model}")

    dataloaders = load_dataloaders(tokenizer_src, tokenizer_tgt, config)  
    # create loss criterion, learning rate optimizer and scheduler
    label_smoothing = LabelSmoothing(config.model.tgt_vocab_size, 
                                     tokenizer_tgt.pad_token_id, 
                                     config.training.label_smoothing)
    criterion = SimpleLossCompute(label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config.training.learning_rate.base, 
                                 betas=(0.9, 0.98), 
                                 eps=1e-9)
    scheduler = LambdaLR(optimizer = optimizer, 
                         lr_lambda = lambda step_num: get_learning_rate(
                             step_num+1, 
                             config.model.d_model, 
                             warmup=config.training.learning_rate.warmup_steps))
    
    # train
    train(dataloaders, model, criterion, optimizer, scheduler, config)