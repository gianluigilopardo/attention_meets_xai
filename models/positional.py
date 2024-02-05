import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_in, max_seq_length):
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
        z = x + self.pe[:, :x.size(1)]
        return z

