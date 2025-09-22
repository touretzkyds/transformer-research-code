from dotted_dict import DottedDict
import yaml
import json
import os
from typing import Any

class Config(DottedDict):
    """
    Configuration class that extends DottedDict with loading/saving functionality
    """
    def __init__(self, load_path: str):
        config_dict = yaml.safe_load(open(load_path, 'r'))
        super().__init__(config_dict)
        self.load_path = load_path
        self.build_arg_mapping()
        
    def update_from_args(self, args: Any) -> "Config":
        """Update config with command line arguments"""
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                config_path = self.arg_mapping.get(arg_name, f"extras.{arg_name}")
                self._update(config_path, arg_value)
        
    def save(self, save_path: str, format: str = "yaml"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if format == "yaml":
            with open(save_path, 'w') as f:
                yaml.dump(dict(self), f)
        elif format == "json":
            with open(save_path, 'w') as f:
                json.dump(dict(self), f)

    def print(self):
        print('-'*100)
        print(f"Config: {self}")
        print('-'*100)

    def _update(self, config_path: str | list[str], value: Any):
        """
        Update single or multiple config paths with a value
        """
        if isinstance(config_path, list): # update multiple paths
            for path in config_path:
                self._update(path, value)
        elif isinstance(config_path, str): # update single path
            parent, key = config_path.rsplit('.', 1)
            self[parent][key] = value
        else:
            raise ValueError(f"Invalid config path: {config_path}")
        
    def build_arg_mapping(self):
        self.arg_mapping = {
            'N': 'model.N',
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'dataset_name': 'dataset.name',
            'dataset_size': 'dataset.size',
            'max_padding': 'model.max_padding',
            'cache': ['dataset.cache', 'tokenizer.cache'],
            'tokenizer_type': 'tokenizer.type',
            'random_seed': 'training.random_seed'
        }