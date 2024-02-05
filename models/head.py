import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import params
from models.utils import scaled_dot_product
from models.positional import PositionalEncoding
from data.utils import process_text

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class Head(nn.Module):
    """
    Defines the Head architecture.

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary.
        d_in (int): The input dimension (embedding dimension).
        d_out (int): The output dimension (value dimension).
        d_attn (int): The attention dimension.
        num_classes (int): The number of output classes.
        vocab: The vocabulary mapping tokens to indices.
        tokenizer: The tokenizer used to preprocess text.
        cls_pos (int): The position of the [CLS] token in the text.
    """

    def __init__(self, vocab_size, d_in, d_out, d_attn, num_classes, vocab, tokenizer, cls_pos=0, scale=True, pos=True):
        super().__init__()

        self.d_in = d_in  # The input dimension (embedding dimension)
        self.d_out = d_out  # The output dimension (value dimension)
        self.num_classes = num_classes  # The number of output classes
        self.d_attn = d_attn  # The attention dimension

        self.cls_pos = cls_pos  # The position of the [CLS] token in the text
        self.scale = scale

        # Create linear layers for query, key, and value.
        self.query = nn.Linear(d_in, d_attn, bias=False)
        self.key = nn.Linear(d_in, d_attn, bias=False)
        self.value = nn.Linear(d_in, d_out, bias=False)

        # Create the classification layer.
        self.classifier = nn.Linear(d_out, num_classes, bias=False)  # Classification layer

        # Initialize the parameters.
        self._reset_parameters()

    def _reset_parameters(self):
        initrange = 0.1
        self.query.weight.data.uniform_(-initrange, initrange)
        self.key.weight.data.uniform_(-initrange, initrange)
        self.value.weight.data.uniform_(-initrange, initrange)
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, e, return_attention=False, return_g=False):
        # Apply the query, key, and value layers to the embedded input.
        q = self.query(e)
        k = self.key(e)
        v = self.value(e)

        # Apply the scaled dot-product attention mechanism to compute attention weights.
        values, attention, g = scaled_dot_product(q, k, v, return_g=return_g, scale=self.scale)

        # Extract the attention scores for the [CLS] token.
        cls_values = values[:, self.cls_pos, :]

        # Pass the attention-weighted values through the classifier layer to obtain predictions.
        output = self.classifier(cls_values)

        if return_attention:
            if return_g:
                return output, attention, g
            return output, attention
        return output

