import torch.nn as nn
from numerize.numerize import numerize as nu

from model.full_model import TransformerModel


def create_model(config):
    model = TransformerModel(
        config.model.src_vocab_size,
        config.model.tgt_vocab_size,
        config.model.N,
        config.model.d_model,
        config.model.d_ff,
        config.model.n_heads,
        config.model.dropout_prob,
    )
    for layer in [
        model,
        model.input_embedding_layer,
        model.output_embedding_layer,
        model.input_positional_enc_layer,
        model.output_positional_enc_layer,
        model.encoder_stack,
        model.decoder_stack,
        model.linear_and_softmax_layers,
    ]:
        count_params(layer)
    if config.hardware.data_parallel:
        model = nn.DataParallel(model)
    return model


def count_params(model):
    num_param_matrices, num_params, num_trainable_params = 0, 0, 0

    for p in model.parameters():
        num_param_matrices += 1
        num_params += p.numel()
        num_trainable_params += p.numel() * p.requires_grad

    print(
        f"{'-'*60}"
        f"\n{model.__class__.__name__}\n"
        f"Number of parameter matrices: {nu(num_param_matrices)}\n"
        f"Number of parameters: {nu(num_params)}\n"
        f"Number of trainable parameters: {nu(num_trainable_params)}"
    )
    return num_param_matrices, num_params, num_trainable_params
