# From humanmodelling even more barebones version of NanoGPT.py
# Original license:
# From https://github.com/karpathy/nanoGPT/blob/master/model.py - Thanks Andrej Karpathy

# MIT License
# Copyright (c) 2022 Andrej Karpathy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelbc.models.utils.model_utils import (MLP, BackboneModel,
                                              verify_input_shape)


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, bias=True, is_causal=True):
        super().__init__()
        assert hasattr(torch.nn.functional, "scaled_dot_product_attention"), "pytorch>=2.0.0 needed for Flash Attention"
        assert embedding_dim % num_heads == 0, f"embedding dimension must be divisible by number of heads but are {embedding_dim} and {num_heads}"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.is_causal = is_causal
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def forward(self, x):
        batch_size, token_len, n_embd = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embedding_dim, dim=2)
        k = k.view(batch_size, token_len, self.num_heads, n_embd // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, token_len, self.num_heads, n_embd // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, token_len, self.num_heads, n_embd // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=self.is_causal)

        y = y.transpose(1, 2).contiguous().view(batch_size, token_len, n_embd)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class GPTMLP(nn.Module):
    def __init__(self, embedding_dim, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias)
        self.c_proj = nn.Linear(4 * embedding_dim, embedding_dim, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, bias=True, is_causal=True):
        super().__init__()
        self.ln_1 = LayerNorm(embedding_dim, bias=bias)
        self.attn = SelfAttention(embedding_dim, num_heads, bias=bias, is_causal=is_causal)
        self.ln_2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = GPTMLP(embedding_dim, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PicoGPT(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        embedding_dim,
        sequence_length=None,
        use_positional_encoding=False,
        bias=True,
        is_causal=True,
    ):
        """
        A barebones PyTorch implementation of GPT-2's Transformer Decoder.
        :param num_layers: number of transformer blocks
        :param num_heads: number of attention heads in each transformer block
        :param embedding_dim: embedding dimensionality
        :param sequence_length: maximum length of input sequence. None: no positional embeddings
        :param use_positional_encoding: if True, use learned positional embeddings. False: no positional embeddings
        :param bias: if True, bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        :param is_causal: if True, use causal attention mask in self-attention layers. False for no any masking like this.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.bias = bias
        self.is_causal = is_causal

        model_layers = []
        # Blocks + layer norm
        for _ in range(self.num_layers):
            model_layers.append(Block(embedding_dim, num_heads, bias=bias, is_causal=is_causal))
        model_layers.append(LayerNorm(embedding_dim, bias=bias))

        if use_positional_encoding:
            assert sequence_length is not None, "sequence_length must be provided if use_positional_encoding is True"
            self.positional_encoding = nn.Embedding(sequence_length, embedding_dim)
        else:
            self.positional_encoding = None

        self.model = nn.Sequential(*model_layers)

        # Initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_positional_encoding(self, x, t=None):
        if self.positional_encoding is not None:
            if t is None:
                seq_length = x.shape[1]
                t = torch.arange(0, seq_length, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.positional_encoding(t)
            x = x + pos_emb
        return x

    def forward(self, x):
        x = self._apply_positional_encoding(x)
        return self.model(x)


class GPTTransformer(nn.Module, BackboneModel):
    def __init__(
        self,
        input_size,
        output_size,
        num_layers,
        num_heads,
        embedding_dim,
        sequence_length=None,
        use_positional_encoding=False,
        bias=True,
        is_causal=True,
    ):
        """
        A barebones PyTorch implementation of GPT-2's Transformer Decoder with MLPs for input and output.
        :param input_size: input dimensionality (projected with MLP into embedding dimension)
        :param output_size: output dimensionality (projected with MLP out of embedding dimension)
        :param num_layers: number of transformer blocks
        :param num_heads: number of attention heads in each transformer block
        :param embedding_dim: embedding dimensionality
        :param sequence_length: maximum length of input sequence. None: no positional embeddings
        :param use_positional_encoding: if True, use learned positional embeddings. False: no positional embeddings
        :param bias: if True, bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        :param is_causal: if True, use causal attention mask in self-attention layers. False for no any masking like this.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.use_positional_encoding = use_positional_encoding

        self.mlp_in = MLP(input_size, embedding_dim, num_layers=1)
        self.mlp_out = MLP(embedding_dim, output_size, num_layers=1)

        self.gpt = PicoGPT(
            num_layers,
            num_heads,
            embedding_dim,
            sequence_length=sequence_length,
            use_positional_encoding=use_positional_encoding,
            bias=bias,
            is_causal=is_causal,
        )

        self.gpt_inputs = deque(maxlen=sequence_length)

    def init_for_sequence(self, batch_size):
        self.gpt_inputs.clear()

    def forward(self, x, rollout=False):
        verify_input_shape(self.input_size, x.shape)
        batch_size, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"Input shape {x.shape} does not match input size {self.input_size}"

        # project image encoding into transformer embedding dim
        x = self.mlp_in(x)

        if rollout:
            assert batch_size == 1, "Batch size must be 1 for rollout"
            # concatenate with previous inputs in sequence (up to sequence_length)
            self.gpt_inputs.append(x)
            gpt_in = torch.cat(list(self.gpt_inputs), dim=1)

            # run through transformer
            gpt_out = self.gpt(gpt_in)

            # only project last embedding to output size for action
            last_out = gpt_out[:, -1, :]
            return self.mlp_out(last_out)
        else:
            # run entire (batch_size, seq_len, embedding_dim) sequence through transformer
            gpt_out = self.gpt(x)
            # project all embeddings to output size for sequence
            return self.mlp_out(gpt_out)
