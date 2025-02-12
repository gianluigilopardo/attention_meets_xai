import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    """Implements positional encoding for input sequences, adding a fixed sinusoidal pattern 
    to each position in the sequence to capture positional information.

    This encoding is added to the input embeddings to provide the model with information 
    about the relative or absolute position of the tokens in the sequence.

    Args:
        d_in (int): The dimensionality of the input embeddings.
        max_seq_length (int): The maximum length of the input sequences.

    Methods:
        forward(x):
            Applies the positional encoding to the input tensor.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_in).

            Returns:
                torch.Tensor: The input tensor with positional encoding applied, 
                of the same shape as the input tensor."""
    def __init__(self, d_in, max_seq_length):
        """Initializes the positional encoding matrix for a transformer model.

    This constructor creates a positional encoding matrix `pe` of shape 
    (1, max_seq_length, d_in) using sine and cosine functions. The matrix 
    is intended to be used to provide positional information to a transformer 
    model by encoding positions with unique patterns. The positional encoding 
    allows models to introduce the notion of word order without using recurrent 
    or convolutional networks.

    Args:
        d_in (int): The dimensionality of the model's input features.
        max_seq_length (int): The maximum sequence length for which the 
            positional encoding will be generated.

    Attributes:
        pe (torch.Tensor): A tensor of shape (1, max_seq_length, d_in) 
            containing the positional encodings. The tensor requires gradients."""
        super().__init__()

        pe = torch.zeros(max_seq_length, d_in)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        n = 10000.0
        div_term = torch.exp(torch.arange(0, d_in, 2).float() * -(math.log(n) / d_in))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe.requires_grad = True
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        """Applies positional encoding to the input tensor.

    Args:
        x (Tensor): The input tensor with shape (batch_size, seq_length, embedding_dim).

    Returns:
        Tensor: The input tensor with positional encodings added, 
        maintaining the same shape as the input."""
        z = x + self.pe[:, :x.size(1)]
        return z

