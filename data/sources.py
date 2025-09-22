import torch
import os
import os.path as op
from datasets import load_dataset # huggingface datasets
from datasets import Dataset, DatasetDict, Features, Value
from data.preprocess import DataProcessor
from transformers import AutoTokenizer
from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
from datasets import load_from_disk

# cosine_distances = squareform(pdist(x, metric='cosine')); near_duplicates = np.where(cosine_distances < threshold); near_duplicates = [(i, j) for i, j in zip(*near_duplicates) if i < j]
class DataSource:
    @classmethod
    def get_data(cls, name, language_pair, cache=False, preprocessor=None, dataset_size=5000000, max_padding=60, random_seed=None, part_start_index=None, part_end_index=None):
        '''
        Returns the raw or preprocessed data for a given dataset.
        Optionally loads a specific part of the dataset based on start/end indices.
        '''
        if preprocessor:
            data = cls._get_preprocessed_data(name, language_pair, cache, preprocessor, random_seed)
        else:
            data = cls._get_raw_data(name, language_pair, cache, random_seed)
        data = DataProcessor.limit_train_size(data, dataset_size, max_padding, part_start_index, part_end_index)
        return data

    @classmethod
    def _get_raw_data(cls, name, language_pair, cache, random_seed=None, dataset_path="artifacts/saved_data/de_en.txt"):
        # conditionally return saved data directly
        cache_path = f"artifacts/saved_data/raw_data/{name}.pt"
        if (cache and op.exists(cache_path)):
            raw_data = torch.load(cache_path)
            print(f"Loaded raw data from {cache_path}")
        else:
            raw_data = DataDownloader.download(name, language_pair, random_seed, dataset_path)
            torch.save(raw_data, cache_path) # TODO: save raw data using huggingface datasets
        return raw_data

    @classmethod
    def _get_preprocessed_data(cls, name, language_pair, cache, preprocessor: DataProcessor, random_seed=None, dataset_path="artifacts/saved_data/de_en.txt"):
        # conditionally return saved data directly
        cache_path = f"artifacts/saved_data/preprocessed_data/{name}.pt"
        if (cache and op.exists(cache_path)):
            preprocessed_data = torch.load(cache_path)
            print(f"Loaded preprocessed data from {cache_path}")
        else:
            raw_data = cls._get_raw_data(name, language_pair, cache, random_seed, dataset_path)
            preprocessed_data = preprocessor.preprocess_data(raw_data)
            torch.save(preprocessed_data, cache_path)
        return preprocessed_data

def custom_tokenize_tgt(dataset_dict): 
    return tokenizer_tgt(dataset_dict["text"], padding=True, truncation=True, max_length=70)

def custom_tokenize_src(dataset_dict): 
    return tokenizer_src(dataset_dict["text"], padding=True, truncation=True, max_length=70)

tokenizer_src = AutoTokenizer.from_pretrained("bert-base-german-cased", use_fast=True)
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)

# TODO LIST
# - figure out map with src and tgt together
# - figure out good saving method

# - keep track of original data if needed
# - use take for subsampling
# - see if we can find duplicates using filter

class DataDownloader:
    @staticmethod
    def download(name, language_pair, random_seed=None, dataset_path="artifacts/saved_data/de_en.txt"):
        # get raw data
        if name == "wmt14":
            dataset_dict = load_dataset("wmt14", "-".join(language_pair))#, num_proc=min(os.cpu_count()-1, 8))
            import pdb; pdb.set_trace()
            x = dataset_dict.flatten().map(custom_tokenize_src, batched=True, batch_size=10000, num_proc=8)
        elif name == "wmt24":
            dataset_dict = load_from_disk("artifacts/saved_data/tokenized_data/wmt24_de_en")
            # root_dir = os.getenv("MTDATA")
            # data_files = {"train.de": op.join(root_dir, "wmt24-eng-deu", "train.deu"), 
            #               "train.en": op.join(root_dir, "wmt24-eng-deu", "train.eng"), 
            #               "val.de": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt23test.de"), 
            #               "val.en": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt23test.en"), 
            #               "test.de": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt24test.de"), 
            #               "test.en": op.join(root_dir, "wmt24-eng-deu", "tests", "wmt24test.en")}
            # dataset_dict = load_dataset("text", data_files=data_files)
            # for key in dataset_dict.keys():
            #     if ".de" in key:
            #         dataset_dict[key] = dataset_dict[key].map(custom_tokenize_src, batched=True, num_proc=8, batch_size=100000)
            #     else:
            #         dataset_dict[key] = dataset_dict[key].map(custom_tokenize_tgt, batched=True, num_proc=8, batch_size=100000)
            # dataset_dict.save_to_disk("artifacts/saved_data/tokenized_data/wmt24_de_en")
        elif name == "m30k":
            dataset_dict = load_dataset("bentrevett/multi30k", num_proc=os.cpu_count()-1)
        elif name == "txt":
            with open(dataset_path, 'r') as f:
                file_text = f.readlines()
                src_sentences = [sentence_pair.split("|")[0].strip() for sentence_pair in file_text]
                tgt_sentences = [sentence_pair.split("|")[1].strip() for sentence_pair in file_text]
                raw_data = list(zip(src_sentences, tgt_sentences))
        else: 
            raise ValueError(f"Received {name}, available options: 'wmt14', 'm30k', 'txt'")
        
        """
        to move
        """
        from transformers import TrainingArguments
        from transformers import Trainer
        from model.full_model import TransformerModel

        model = TransformerModel(
            src_vocab_size=len(tokenizer_src.vocab),
            tgt_vocab_size=len(tokenizer_tgt.vocab),
            N=1,
            d_model=512,
            d_ff=2048,
            n_heads=2,
            dropout_prob=0.1,
        )

        training_args = TrainingArguments(
            output_dir="artifacts/saved_models/wmt24_de_en",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        import pdb; pdb.set_trace()
        return raw_data
