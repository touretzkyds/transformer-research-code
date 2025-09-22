import json
import argparse
import torch
from tqdm import tqdm
from tokenization.utils import build_tokenizers
from data.dataloading import load_datasets, load_dataloaders
from training.logging import TranslationLogger
from inference.utils import greedy_decode, BleuUtils
import os

class Translator:
    def __init__(self, args, config_path):
        '''
        Initializes the Translator class by creating required directories 
        and loading the runtime configs
        '''
        # load model and training configurations saved from training run
        self.load_config(config_path, args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_config(self, filepath, args):
        '''
        Load a saved configuration json file as a dictionary to be used 
        for model loading and translation.
        '''
        try:
            with open(filepath, 'r') as fp:
                config = json.load(fp)
        except FileNotFoundError:
            print(f"Warning: Config file {filepath} not found. Creating default config.")
            config = self.create_default_config()
        
        # add information from args
        for k, v in args.__dict__.items():
            config[k] = v
        
        # Ensure required config attributes exist
        self.ensure_config_attributes(config)
        self.config = config
    
    def create_default_config(self):
        '''Create a default configuration with minimal required attributes'''
        return {
            'logging': {
                'model_dir': 'artifacts/saved_models',
                'artifacts_dir': 'artifacts'
            },
            'model': {
                'N': 6,
                'max_padding': 20
            },
            'dataset': {
                'name': 'wmt14',
                'size': 5000000
            },
            'translation': {
                'batch_limit': 100
            },
            'tokenizer': {
                'type': 'bert',
                'cache': True
            }
        }
    
    def ensure_config_attributes(self, config):
        '''Ensure all required configuration attributes exist with sensible defaults'''
        # Ensure nested dictionaries exist
        if 'logging' not in config:
            config['logging'] = {}
        if 'model' not in config:
            config['model'] = {}
        if 'dataset' not in config:
            config['dataset'] = {}
        if 'translation' not in config:
            config['translation'] = {}
        if 'tokenizer' not in config:
            config['tokenizer'] = {}
        
        # Set default values for required attributes
        config['logging'].setdefault('model_dir', 'artifacts/saved_models')
        config['logging'].setdefault('artifacts_dir', 'artifacts')
        config['model'].setdefault('N', 6)
        config['model'].setdefault('max_padding', 20)
        config['dataset'].setdefault('name', 'wmt14')
        config['dataset'].setdefault('size', 5000000)
        config['translation'].setdefault('batch_limit', 100)
        config['tokenizer'].setdefault('type', 'bert')
        config['tokenizer'].setdefault('cache', True)

    def prepare_tokenizers(self): 
        '''
        Load tokenizers and vocabularies
        '''
        try:
            if not hasattr(self.config, 'tokenizer') or not hasattr(self.config.tokenizer, 'type'):
                raise ValueError("Missing 'tokenizer.type' in configuration")
            if not hasattr(self.config, 'dataset') or not hasattr(self.config.dataset, 'language_pair'):
                raise ValueError("Missing 'dataset.language_pair' in configuration")
            
            tokenizer_src, tokenizer_tgt = build_tokenizers(self.config)
            self.tokenizer_src = tokenizer_src
            self.tokenizer_tgt = tokenizer_tgt
            
            print(f"Successfully loaded tokenizers: {self.config.tokenizer.type}")
            
        except Exception as e:
            print(f"Error preparing tokenizers: {e}")
            raise

    def prepare_model(self):
        try:
            # Ensure required config attributes exist
            if not hasattr(self.config, 'epoch'):
                raise ValueError("Missing 'epoch' in configuration")
            if not hasattr(self.config.model, 'N'):
                raise ValueError("Missing 'model.N' in configuration")
            if not hasattr(self.config.dataset, 'name'):
                raise ValueError("Missing 'dataset.name' in configuration")
            if not hasattr(self.config.dataset, 'size'):
                raise ValueError("Missing 'dataset.size' in configuration")
            if not hasattr(self.config.logging, 'model_dir'):
                raise ValueError("Missing 'logging.model_dir' in configuration")
            
            save_path = f"{self.config.logging.model_dir}/N{self.config.model.N}/{self.config.dataset.name}/dataset_size_{self.config.dataset.size}/epoch_{self.config.epoch:02d}.pt"
            
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"Model file not found: {save_path}")
                
            self.model = torch.load(save_path)
            
            if isinstance(self.model, torch.nn.DataParallel):
                print("Unwrapping DataParallel model")
                self.model = self.model.module
            
            self.model.to(self.device)
            print(f"Successfully loaded model from {save_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def prepare_dataloader(self, split):
        try:
            if not hasattr(self.config, 'translation') or not hasattr(self.config.translation, 'batch_limit'):
                raise ValueError("Missing 'translation.batch_limit' in configuration")
            
            dataloaders = load_dataloaders(self.tokenizer_src, self.tokenizer_tgt, self.config)
            
            if split not in dataloaders:
                available_splits = list(dataloaders.keys())
                raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
            
            # select dataloader to use for translation
            self.dataloader = dataloaders[split]
            self.dataloader = list(self.dataloader)[:self.config.translation.batch_limit]
            
            print(f"Successfully prepared dataloader for split '{split}' with {len(self.dataloader)} batches")
            
        except Exception as e:
            print(f"Error preparing dataloader: {e}")
            raise

    def translate_dataset(self):
        avg_bleu = 0
        pbar = tqdm(self.dataloader)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(pbar):
                pbar.set_description(f"Translating batch {batch_idx + 1} / {len(self.dataloader)}")
                bleu = self.translate_batch(self.model, batch, self.tokenizer_tgt, self.logger)
                avg_bleu += bleu
        avg_bleu /= len(self.dataloader)
        return avg_bleu
    
    def translate_batch(self, model, batch, tokenizer_tgt, logger=None):
        predictions = greedy_decode(model, batch, tokenizer_tgt)
        src_sentences = self.tokenizer_src.batch_decode(batch.src, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tgt_sentences = self.tokenizer_tgt.batch_decode(batch.tgt_label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predicted_sentences = self.tokenizer_tgt.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        bleu = BleuUtils.compute_batch_bleu(predicted_sentences, tgt_sentences)
        if logger:
            logger.log_sentence_batch(src_sentences, tgt_sentences, predicted_sentences, bleu)
        return bleu

    def validate_config(self):
        '''Validate that all required configuration attributes exist'''
        required_attrs = [
            'epoch',
            'model.N',
            'dataset.name',
            'dataset.size',
            'logging.model_dir',
            'translation.batch_limit',
            'tokenizer.type'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            try:
                # Navigate nested attributes
                value = self.config
                for key in attr.split('.'):
                    value = value[key]
            except (KeyError, AttributeError):
                missing_attrs.append(attr)
        
        if missing_attrs:
            raise ValueError(f"Missing required configuration attributes: {missing_attrs}")
        
        print("Configuration validation passed")
        return True

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch", type=int, default=1
                            ) # 1-indexed epoch number of saved model
        parser.add_argument("--num_examples", type=int, default=5)
        parser.add_argument("--N", type=int, default=6)
        parser.add_argument("--split", type=str, default='test', choices=['train', 'val', 'test'])
        parser.add_argument("--dataset_name", type=str, default='wmt14')
        parser.add_argument("--dataset_size", type=int, default=5000000)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--batch_limit", type=int, default=100)
        parser.add_argument("--tokenizer_type", type=str, default='bert', choices=['bert', 'spacy'])
        parser.add_argument("--random_seed", type=int, default=40)
        parser.add_argument("--max_padding", type=int, default=20)
        args = parser.parse_args()
        
        print("Starting translation pipeline...")
        print(f"Arguments: {vars(args)}")
        
        # initialize Translator class
        translator = Translator(args, 'artifacts/training_config.json')
        
        # validate configuration
        translator.validate_config()
        
        # initialize logger
        translator.logger = TranslationLogger(translator.config)
        
        # prepare fundamental blocks of translation pipeline
        print("Preparing tokenizers...")
        translator.prepare_tokenizers()
        
        print("Preparing dataloader...")
        translator.prepare_dataloader(split=args.split)
        
        print("Preparing model...")
        translator.prepare_model()
        
        # run translations  
        print("Starting translation...")
        bleu_score = translator.translate_dataset()
        print(f"Translation completed. Average BLEU score: {bleu_score:.4f}")
        
        # save translation results
        translator.logger.save_as_txt('artifacts/generated_translations/', 
                            title='Transformer translations',
                            title_dict={k:translator.config[k] for k in 
                                        ['N', 'epoch', 'num_examples', 
                                         'dataset_size', 'dataset_name']})
        
        print("Translation results saved successfully!")
        
    except Exception as e:
        print(f"Error in translation pipeline: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
