import json
import torch.nn as nn
from numerize.numerize import numerize as nu
# from model.full_model import TransformerModel
from collections import OrderedDict
from model.full_model import TransformerModel, MultiHeadedAttentionModule
def create_model(config):
    model =  TransformerModel(config.model.src_vocab_size, 
                              config.model.tgt_vocab_size, 
                              config.model.N, 
                              config.model.d_model, 
                              config.model.d_ff, 
                              config.model.n_heads, 
                              config.model.dropout_prob)
    count_ops(model, seq_len=50)
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


def count_ops(model, seq_len=512):
    """
    Estimate multiply operations (per sample) for each layer to find bottlenecks.

    Counts two kinds of operations:
        nn.Linear projections:   seq_len * in_features * out_features
        Attention matmuls:       heads * seq_len^2 * d_k   (for both QK^T and attn@V)

    Attention scales quadratically with seq_len, so it dominates at long sequences.
    Embedding, LayerNorm, Dropout, Softmax, and ReLU are negligible and omitted.

    Output has two sections:
        1. Per-operation breakdown (every Linear projection and attention matmul)
        2. Per-module summary (ops summed within each Sublayer / top-level module)
    """
    ops_by_layer = []
    total_ops = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mults = seq_len * module.in_features * module.out_features
            ops_by_layer.append((name, "Linear", mults))
            total_ops += mults

        elif isinstance(module, MultiHeadedAttentionModule):
            h, d_k = module.h, module.d_k
            qk_ops = h * seq_len * seq_len * d_k
            av_ops = h * seq_len * seq_len * d_k
            ops_by_layer.append((name + " (QK^T)", "Attention", qk_ops))
            ops_by_layer.append((name + " (attn@V)", "Attention", av_ops))
            total_ops += qk_ops + av_ops

    # --- Per-operation breakdown ---
    print(f"\n{'='*90}")
    print(f"  {model.__class__.__name__} — Multiply ops per sample (seq_len={seq_len})")
    print(f"{'='*90}")
    print(f"  {'Layer':<50} {'Type':<12} {'Mult Ops':>12} {'%':>7}")
    print(f"  {'-'*81}")
    for name, op_type, ops in ops_by_layer:
        pct = 100.0 * ops / total_ops if total_ops > 0 else 0
        print(f"  {name:<50} {op_type:<12} {nu(ops):>12} {pct:>6.1f}%")
    print(f"  {'-'*81}")
    print(f"  {'TOTAL':<50} {'':12} {nu(total_ops):>12}")

    # --- Per-module summary ---
    ops_by_module = _aggregate_ops_by_module(ops_by_layer, total_ops)
    print(f"\n  {'Module summary':^81}")
    print(f"  {'-'*81}")
    print(f"  {'Module':<50} {'#Ops':<6} {'Mult Ops':>12} {'%':>7}")
    print(f"  {'-'*81}")
    for module_name, (count, module_ops) in ops_by_module.items():
        pct = 100.0 * module_ops / total_ops if total_ops > 0 else 0
        print(f"  {module_name:<50} {count:<6} {nu(module_ops):>12} {pct:>6.1f}%")
    print(f"  {'-'*81}")
    print(f"  {'TOTAL':<50} {'':6} {nu(total_ops):>12}")
    print(f"{'='*90}\n")

    return ops_by_layer, ops_by_module, total_ops


def _aggregate_ops_by_module(ops_by_layer, total_ops):
    """
    Group individual op entries by their parent module.

    Grouping strategy (first match wins):
        *.self_attn_sublayer.*   -> grouped under self_attn_sublayer
        *.cross_attn_sublayer.*  -> grouped under cross_attn_sublayer
        *.pos_ff_sublayer.*      -> grouped under pos_ff_sublayer
        *.linear_layer / other   -> grouped under their top-level module
    """
    ops_by_module = OrderedDict()

    for name, op_type, ops in ops_by_layer:
        parts = name.split(".")
        module_key = _find_parent_module(parts)

        if module_key not in ops_by_module:
            ops_by_module[module_key] = [0, 0]
        ops_by_module[module_key][0] += 1
        ops_by_module[module_key][1] += ops

    return ops_by_module


def _find_parent_module(name_parts):
    """
    Walk the dotted name and return the path up to and including the first
    sublayer-level component (self_attn_sublayer, cross_attn_sublayer,
    pos_ff_sublayer).  Falls back to the top-level module name.
    """
    sublayer_keywords = {"self_attn_sublayer", "cross_attn_sublayer", "pos_ff_sublayer"}
    path = []
    for part in name_parts:
        path.append(part)
        if part in sublayer_keywords:
            return ".".join(path)
    if "linear_and_softmax_layers" in name_parts:
        return "linear_and_softmax_layers"
    return name_parts[0] if name_parts else "unknown"


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