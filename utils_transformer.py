### Utility modules for a transformer

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len] - this is the expected mask.shape for causal mask for nn.MultiheadAttention module
    return mask == 1  # True elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token) # mask.shape: [batch_size, max_seq_len] - this is the expected mask.shape for padding mask for nn.MultiheadAttention module
    return mask  # True elements are masked

# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.GELU()
    def forward(self, x):
        return self.w2(self.dropout( self.act_fn(self.w1(x)) ))

# class implementing multi head attention (using pytorch inbuilt nn.MultiheadAttention)
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, kdim=d_k*n_heads , vdim=d_v*n_heads , batch_first=True) # kdim is the total dim (multiplying with n_heads)
    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, mask_padding=None, mask_causal=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask=mask_padding, attn_mask=mask_causal)
        return attn_output

# class implementing residual + normalization connection - takes in any block and applies a normalization + residual connection
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return x + self.dropout(sublayer( self.norm(x) )) # note that we apply the norm first

# class implementing a single encoder layer
# each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
    def forward(self, x, mask_padding, mask_causal):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask_padding=mask_padding, mask_causal=mask_causal)) # x.shape: [batch_size, seq_len, d_model]
        x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
        return x

# class implementing the entire encoder block = stacked encoder layers
class Encoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model) # final layernorm at encoder output
    def forward(self, x, mask_padding=None, mask_causal=None):
        for layer in self.layers:
            x = layer(x, mask_padding, mask_causal)
        return self.norm(x)

# class implementing a single decoder layer
# each decoder layer has three blocks: 1. (self) (masked) multihead attention 2. (src) (unmasked) multihead attention  3. feed_forward; with sublayer connection around each
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 3) # one for self_attn block, second for src_attn block, third for feed_forward block
    def forward(self, x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal):
        m = encoder_out
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask_padding=tgt_mask_padding, mask_causal=tgt_mask_causal)) # first apply self_attn block
        x = self.sublayers[1](x, lambda x: self.src_attn(m, x, m, mask_padding=src_mask_padding)) # src_attn: (key from encoder, query from decoder, value from encoder)
        x = self.sublayers[2](x, self.feed_forward)
        return x

# class implementing the entire decoder block = stacked decoder layers
class Decoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model) # final layernorm at decoder output
    def forward(self, x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal)
        return self.norm(x)
