import json
import torch.nn as nn
from numerize.numerize import numerize as nu
# from model.full_model import TransformerModel
from model.full_model import TransformerModel
def create_model(config):
    model =  TransformerModel(config.model.src_vocab_size, 
                              config.model.tgt_vocab_size, 
                              config.model.N, 
                              config.model.d_model, 
                              config.model.d_ff, 
                              config.model.n_heads, 
                              config.model.dropout_prob)
    for layer in [model, 
                  model.input_embedding_layer, 
                  model.output_embedding_layer, 
                  model.input_positional_enc_layer, 
                  model.output_positional_enc_layer,
                  model.encoder_stack, 
                  model.decoder_stack, 
                  model.linear_and_softmax_layers]:
        count_params(layer)
    if config.hardware.data_parallel:
        model = nn.DataParallel(model) # enable running on multiple GPUs
    return model

def load_model(config, pretrained=False, model_path=None):
    model = create_model(config)
    if pretrained:
        model.load_state_dict(model_path)
    return model

def count_params(model):
    num_param_matrices, num_params, num_trainable_params = 0, 0, 0
    
    for p in model.parameters():
        num_param_matrices += 1
        num_params += p.numel()
        num_trainable_params += p.numel() * p.requires_grad

    print(f"{'-'*60}"
          f"\n{model.__class__.__name__}\n"
          f"Number of parameter matrices: {nu(num_param_matrices)}\n"
          f"Number of parameters: {nu(num_params)}\n"
          f"Number of trainable parameters: {nu(num_trainable_params)}")
    return num_param_matrices, num_params, num_trainable_params


def create_config(args, src_vocab_size, tgt_vocab_size):
    config = {
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "dataset_name": args.dataset_name,
        "language_pair": tuple(args.language_pair),
        "N": args.N,
        "batch_size": args.batch_size,
        "d_model": 512,
        "d_ff": 2048,
        "h": 8,
        "dropout_prob": 0.1,
        "num_epochs": args.epochs,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": args.max_padding,
        "warmup": 3000,
        "model_dir": f"artifacts/saved_models",
        "dataset_size": args.dataset_size,
        "experiment_name": args.experiment_name,
        "random_seed": args.random_seed,
    }
    # save config as a json file
    with open('artifacts/training_config.json', 'w') as fp:
        json.dump(config, fp)
    return config