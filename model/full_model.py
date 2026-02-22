import math
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class TransformerModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048,
                 h=8, dropout_prob=0.1):
        '''
        This class assembles the transformer model from the individual submodules created,
        block by block as shown in Figure 1 of the paper.
           src_vocab_size: number of tokens in encoder's embedding dictionary
           tgt_vocab_size: number of tokens in decoder's embedding dictionary
           N: number of encoder/decoder layers
           d_model: embedding size
           d_ff: feedforward layer size
           h: number of attention heads
        '''
        super(TransformerModel, self).__init__()

        # Source (encoding) layers
        self.input_embedding_layer = EmbeddingLayer(src_vocab_size, d_model)
        self.input_positional_enc_layer = PositionalEncodingLayer(d_model, dropout_prob)
        self.encoder_stack = EncoderStack(h, d_model, d_ff, dropout_prob, N)

        # Target (decoding) layers
        self.output_embedding_layer = EmbeddingLayer(tgt_vocab_size, d_model)
        self.output_positional_enc_layer = PositionalEncodingLayer(d_model, dropout_prob)
        self.decoder_stack = DecoderStack(h, d_model, d_ff, dropout_prob, N)

        # linear and softmax layers
        self.linear_and_softmax_layers = LinearAndSoftmaxLayers(d_model, tgt_vocab_size)

        # Initialize parameters with Glorot / fan_avg.
        # This was important according to the paper's code TODO: verify this from the code
        for p in self.parameters():
            if p.dim() > 1: # presumably biases skipped TODO: verify this
                nn.init.xavier_uniform_(p)
        
    def encode(self, src):
        # convert source tokens to embeddings and add positional encoding
        src_embeddings = self.input_embedding_layer(src)
        src_embeddings_with_positions = self.input_positional_enc_layer(src_embeddings)
        # encode
        encoder_stack_output = self.encoder_stack(src_embeddings_with_positions)
        return encoder_stack_output

    def decode(self, tgt, memory, decoder_attn_mask):
        # convert target tokens to embeddings and add positional encoding
        tgt_embeddings = self.output_embedding_layer(tgt)
        tgt_embeddings_with_positions = self.output_positional_enc_layer(tgt_embeddings)
        # decode
        decoder_stack_output = self.decoder_stack(tgt_embeddings_with_positions, memory, decoder_attn_mask)
        return decoder_stack_output

    def forward(self, src, tgt, decoder_attn_mask):
        # pass source tokens through encoder
        encoder_stack_output = self.encode(src)
        # pass target tokens and encoder output through decoder
        decoder_stack_output = self.decode(tgt, encoder_stack_output, decoder_attn_mask)
        # project decoder output to transformer output vocabulary size
        output_logprobabilities = self.linear_and_softmax_layers(decoder_stack_output)
        return output_logprobabilities
    

class EmbeddingLayer(nn.Module): # TODO nn.Embedding from nn.Module
    def __init__(self, vocab, d_model):
        super(EmbeddingLayer, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.scale_factor = math.sqrt(d_model)

    def forward(self, token): # TODO: super.forward() or similar
        # get embedding vector
        embedding_vector = self.lookup_table(token)
        # scale the vector
        scaled_embedding_vector = embedding_vector * self.scale_factor
        return scaled_embedding_vector


class PositionalEncodingLayer(nn.Module):
    """
    Implement the PE function."""

    def __init__(self, d_model, dropout_prob, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register buffer as a module parameter, so it gets saved with the model,
        # but is not updated during backprop
        self.register_buffer("pe", pe)
        self.dropout_layer = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x_with_position = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout_layer(x_with_position)


class EncoderStack(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, h, d_model, d_ff, dropout_prob, N):
        super(EncoderStack, self).__init__()
        # create and stack encoder layers
        encoder_layer_list = []
        for i in range(N):
            encoder_layer_i = EncoderLayer(h, d_model, d_ff, dropout_prob)
            encoder_layer_list.append(encoder_layer_i)
        self.encoder_layers = nn.ModuleList(encoder_layer_list)
        # layer normalization
        self.norm_layer = LayerNorm(d_model)

    def forward(self, x):
        "Pass the input through each layer in turn."
        layer_input = x
        for layer in self.encoder_layers:
            # compute layer output
            layer_output = layer(layer_input)
            # this becomes input to next layer
            layer_input = layer_output
        normed_output = self.norm_layer(layer_output)
        return normed_output


class DecoderStack(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, h, d_model, d_ff, dropout_prob, N):
        super(DecoderStack, self).__init__()
        # create and stack decoder layers
        decoder_layer_list = []
        for i in range(N):
            decoder_layer_i = DecoderLayer(h, d_model, d_ff, dropout_prob)
            decoder_layer_list.append(decoder_layer_i)
        self.decoder_layers = nn.ModuleList(decoder_layer_list)
        # layer normalization
        self.norm_layer = LayerNorm(d_model)

    def forward(self, x, memory, decoder_attn_mask):
        layer_input = x
        for layer in self.decoder_layers:
            # compute layer output
            layer_output = layer(layer_input, memory, decoder_attn_mask)
            # this becomes input to next layer
            layer_input = layer_output
        normed_output = self.norm_layer(layer_output)
        return normed_output


class Sublayer(nn.Module):
    """
    Sublayer accepts a workhorse (either a self attention module, a cross attention module,
    or a position-wise feed forward module) as an argument and composes it with the following 
    operations:
    x + Dropout(Workhorse(LayerNorm(x)))
    """
    def __init__(self, sublayer_type, workhorse, size, dropout_prob):
        super(Sublayer, self).__init__()
        self.norm_layer = LayerNorm(size)
        self.sublayer_type = sublayer_type # "attention" or "feedforward"
        self.workhorse = workhorse  
        self.dropout_layer = nn.Dropout(p = dropout_prob)

    def forward(self, x, mask=None, memory=None): # x is representation / embedding
        normed_x = self.norm_layer(x)
        if self.sublayer_type == "attention":
            # attention sublayers
            if memory is None: # self attention or masked self attention
                workhorse_output = self.workhorse(query = normed_x, 
                                                  key = normed_x, 
                                                  value = normed_x,
                                                  attention_mask = mask)
            else: # cross attention
                workhorse_output = self.workhorse(query = normed_x, 
                                                  key = memory, 
                                                  value = memory)
        elif self.sublayer_type == "feedforward":
            # positionwise feedforward
            workhorse_output = self.workhorse(normed_x)
        else:
            raise ValueError("Invalid configuration for Sublayer forward method")

        # apply dropout
        dropout_output = self.dropout_layer(workhorse_output)

        # residual connection
        residual_output = x + dropout_output
        return residual_output


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)." # TODO: citation

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    """
    EncoderLayer is made up of one self-attention sublayer and
    one positionwise feed forward sublayer
    """

    def __init__(self, h, d_model, d_ff, dropout_prob):
        super(EncoderLayer, self).__init__()
        # Initialize self attention network
        self_attn_module = MultiHeadedAttentionModule(h, d_model, dropout_prob) # TODO: "module" or "workhorse"?
        # Create self attention sublayer:
        #   Wrap the self attention module in a Sublayer.
        #   This Sublayer likely handles residual connections and normalization.
        #   Encoder layers use the same input for query, key, and value (no separate memory input).
        self.self_attn_sublayer = Sublayer(sublayer_type="attention", 
                                           workhorse=self_attn_module,
                                           size=d_model, 
                                           dropout_prob=dropout_prob)
        # create feedforward sublayer
        pos_ff_module = PositionwiseFeedForwardNetwork(d_model, d_ff, dropout_prob) # TODO: "module" or "workhorse"?
        self.pos_ff_sublayer = Sublayer(sublayer_type="feedforward", 
                                        workhorse=pos_ff_module, 
                                        size=d_model, 
                                        dropout_prob=dropout_prob)

    def forward(self, x):
        self_attn_sublayer_output = self.self_attn_sublayer(x)
        pos_ff_sublayer_output = self.pos_ff_sublayer(self_attn_sublayer_output)
        return pos_ff_sublayer_output


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    The steps are similar, with an added cross attention component called
    'source attention' which performs attention on the encoder output
    """

    def __init__(self, h, d_model, d_ff, dropout_prob):
        super(DecoderLayer, self).__init__()

        # Create self attention sublayer:
        #   Create a closure referencing a self attention layer. TODO: update this doc
        #   The self attention module of the decoder layer uses a mask
        #   to prevent positions from attending to subsequent positions.
        self_attn_module = MultiHeadedAttentionModule(h, d_model, dropout_prob) # TODO: "module" or "workhorse"?
        self.self_attn_sublayer = Sublayer(sublayer_type="attention", 
                                           workhorse=self_attn_module,
                                           size=d_model, 
                                           dropout_prob=dropout_prob)

        # Create cross attention sublayer:TODO: update this doc
        #   Create a closure referencing a cross attention layer.
        #   "memory" indicates that decoder layer operates on the output embedding from the
        #   encoder stack as explained in section 3.2.3 of the paper.
        cross_attn_module = MultiHeadedAttentionModule(h, d_model, dropout_prob) # TODO: "module" or "workhorse"?
        self.cross_attn_sublayer = Sublayer(sublayer_type="attention", 
                                           workhorse=cross_attn_module,
                                           size=d_model, 
                                           dropout_prob=dropout_prob)

        # create feedforward sublayer
        pos_ff_module = PositionwiseFeedForwardNetwork(d_model, d_ff, dropout_prob) # TODO: "module" or "workhorse"?
        self.pos_ff_sublayer = Sublayer(sublayer_type="feedforward", 
                                        workhorse=pos_ff_module, 
                                        size=d_model, 
                                        dropout_prob=dropout_prob)


    def forward(self, x, memory, decoder_attn_mask):
        "Follow Figure 1 (right) for connections."
        # self attention
        self_attn_sublayer_output = self.self_attn_sublayer(x, 
                                                            mask = decoder_attn_mask)
        # cross attention
        cross_attn_sublayer_output = self.cross_attn_sublayer(self_attn_sublayer_output, 
                                                             memory = memory)
        # feedforward network
        pos_ff_sublayer_output = self.pos_ff_sublayer(x = cross_attn_sublayer_output)

        return pos_ff_sublayer_output


class MultiHeadedAttentionModule(nn.Module):
    """
    Generates a multiheaded attention network, which is 3 feedforward networks,
    for linear transformation on the query, key and value. These linearly 
    transformed vectors are then passed to the attention function which 
    performs scaled dot product attention on the vectors, returning the
    attended vector.
    """
    def __init__(self, h, d_model, dropout_prob=0.1):
        super().__init__()
        # Model dimension must be multiple of number of heads
        assert d_model % h == 0, f'dimension mismatch, d_model must be a multiple of h (got {d_model} and {h})'
        self.h = h # number of heads
        self.d_k = d_model // h # assume key size = value size

        # create linear layers for weights corresponding to q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attention_mask=None):
        # The attention mask is not used in the encoder, and hence is only
        # assigned a null value for the encoder
        if attention_mask is not None:
            attention_mask_tensor = attention_mask.unsqueeze(1)
        else:
            attention_mask_tensor = None
        batch_size = query.size(0)
        # The w_q, w_k, w_v matrices compute the sized d_k derived query/key/value
        # vectors for all h attention heads in one operation. We then 
        # partition this into separate vectors for each attention head 
        # (Paragraph 1, Section 3.2.2 of the paper).
        partition_across_attn_heads = lambda x : x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        derived_queries = partition_across_attn_heads(self.w_q(query))
        derived_keys = partition_across_attn_heads(self.w_k(key))
        derived_values = partition_across_attn_heads(self.w_v(value))
        # compute attention
        attention_outputs, attention_weightings = \
            self.attention_fn(derived_queries, derived_keys, derived_values, 
                              attention_mask_tensor)
        # save weightings only for visualization
        self.attention_weightings = attention_weightings
        # concatenate outputs of the h attention heads into one vector
        concatenated_attention_outputs = \
            attention_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # pass through final linear layer
        result = self.linear_layer(concatenated_attention_outputs)
        del query, key, value, derived_queries, derived_keys, derived_values
        return result
    
    def attention_fn(self, derived_queries, derived_keys, derived_values, 
                     mask=None):
        '''
        Compute scaled dot product attention based on equation (1) under 
        section 3.2.1 of the paper: Attention(Q, K, V) = Softmax(QK^T / √dk)V. 
        Additional to this equation, dropout is applied (@dst.cs.cmu.edu). 
        Steps:
            1. Compute alignment scores, ie. dot product QK^T
            2. Scale by square root of key size
            3. Optionally apply masking if computing attention within the decoder
            4. Apply dropout for regularization 
            4. Apply softmax to get an attention matrix of weightings
            5. Multiply attention matrix with values to get resulting attention outputs
        '''
        # key size
        d_k = derived_queries.size(-1)
        # equation (1) of paper
        scores = torch.matmul(derived_queries, derived_keys.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weightings = scores.softmax(dim=-1)
        # apply dropout
        attention_weightings = self.dropout_layer(attention_weightings)
        # compute attended 
        attention_outputs = torch.matmul(attention_weightings, derived_values)
        return attention_outputs, attention_weightings


class PositionwiseFeedForwardNetwork(nn.Module):
    """
    Implements a fully connected position-wise feed-forward network consisting 
    of two linear layers with a ReLU in between them, as per equation (2) 
    under section 3.3 of the paper. Applies dropout to the output of the 
    first layer as per ?? (@dst.cs.cmu.edu) 
    """

    def __init__(self, d_model, d_ff, dropout_prob=0.1):
        super(PositionwiseFeedForwardNetwork, self).__init__()
        self.linear_layer_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.linear_layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        linear_1_output = self.linear_layer_1(x)
        relu_output = self.relu(linear_1_output)
        dropout_output = self.dropout_layer(relu_output)
        linear_2_output = self.linear_layer_2(dropout_output)
        return linear_2_output


class LinearAndSoftmaxLayers(nn.Module):
    """
    Define standard linear + softmax generation step.
    1. project transformer output to target vocabulary size
    2. generate output log probabilities for the entire vocab-sized vector, ie,
       every token in the target vocabulary
    """

    def __init__(self, d_model, vocab_size):
        super(LinearAndSoftmaxLayers, self).__init__()
        self.linear_layer = nn.Linear(d_model, vocab_size)
        self.softmax_layer = nn.LogSoftmax(dim = -1) # Note use of log probabilities

    def forward(self, x):
        linear_layer_output = self.linear_layer(x)
        softmax_layer_output = self.softmax_layer(linear_layer_output)
        return softmax_layer_output
