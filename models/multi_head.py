import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import params
from models.utils import scaled_dot_product
from models.positional import PositionalEncoding
from data.utils import process_text

from models.head import Head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class MultiHead(nn.Module):
    """
    Defines the MultiHead model architecture.

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

    def __init__(self, vocab_size, d_in, d_out, d_attn, num_classes, vocab, tokenizer, cls_pos=0, scale=True, pos=True,
                 num_heads=6):
        super().__init__()
        self.vocab_size = vocab_size  # The number of unique tokens in the vocabulary
        self.d_in = d_in  # The input dimension (embedding dimension)
        self.d_out = d_out  # The output dimension (value dimension)
        self.num_classes = num_classes  # The number of output classes
        self.d_attn = d_attn  # The attention dimension

        self.pos = pos  # if positional embedding
        self.cls_pos = cls_pos  # The position of the [CLS] token in the text
        self.scale = scale

        self.vocab = vocab
        self.tokenizer = tokenizer

        # Create an embedding layer to map tokens to vectors.
        self.embedding = nn.Embedding(vocab_size, d_in)
        self.pe = PositionalEncoding(d_in, params.MAX_LEN)

        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            Head(vocab_size, d_in, d_out, d_attn, num_classes, vocab, tokenizer, cls_pos, scale, pos) for _ in
            range(num_heads)])

    def _reset_parameters(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, return_attention=False, return_g=False):
        # Input: x: tensor of token indices

        # Apply the embedding layer to the input tensor.
        emb = self.embedding(x.to(torch.int64))

        if self.pos:
            e = self.pe(emb)
            # e = (emb + pe)
        else:
            e = emb

        outputs = [h(e) for h in self.heads]
        tensor_sum = torch.zeros(outputs[0].shape).to(device)

        for tensor in outputs:
            tensor_sum += tensor.to(device)
        return tensor_sum / len(self.heads)

    def preprocess(self, text):
        # Preprocess the input text (a string).
        return process_text(text, params.MAX_LEN, self.vocab, self.tokenizer)

    def predict_proba(self, x):
        x = torch.stack([self.preprocess(t) for t in x])  # Preprocess the input text.
        with torch.no_grad():  # Disable gradient computation during prediction.
            preds = self.forward(x)  # Forward pass through the model.
            # Apply softmax activation to obtain probability distribution over classes.
            # preds = F.softmax(preds, dim=-1)
        return preds.to('cpu').detach().numpy()

    def predict(self, x):
        preds = self.predict_proba(x)  # Obtain probability distribution over classes.
        return preds.argmax(dim=-1)  # Predict the most probable class.

    def get_attention_exp(self, text, option='avg'):
        num_heads = len(self.heads)
        tokens = self.tokenizer(text)
        x = self.preprocess(text)
        emb = self.embedding(x.to(torch.int64))
        e = self.pe(torch.unsqueeze(emb, dim=0))
        alpha, alpha_ = [], []
        for h in range(num_heads):
            _, attn = self.heads[h].forward(e, return_attention=True)
            alpha.append(attn[:, self.cls_pos, :])
            alpha_.append(attn)
        attention = torch.stack(alpha_, dim=0)
        attn_heads = torch.stack(alpha, dim=0)
        if option == 'max':
            attn_model = torch.max(attn_heads.squeeze(dim=1), dim=0)
        else:
            attn_model = torch.mean(attn_heads, dim=0)
        attention_words = {w: float(a) for w, a in zip(tokens, attn_model[0][1:params.MAX_LEN + 1])}
        # Sort the attention dictionary by attention score in descending order.
        attn_words = dict(sorted(attention_words.items(), key=lambda x: x[1], reverse=True))
        return attention, attn_heads, attn_model, attn_words

    def get_gradient_exp(self, text, option='mean'):
        tokens = self.tokenizer(text)
        x = torch.stack((self.preprocess(text), ))
        emb = self.embedding(x.to(torch.int64))
        e = self.pe(emb)
        grad = {}

        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                grad[name] = grad_output[0].detach()

            return hook

        handle_gradient_pe = self.pe.register_full_backward_hook(get_gradient('pe'))
        Z = self(x)
        Z[0, 1].backward()
        # Print output and gradients
        gradient = torch.squeeze(grad['pe'])

        if option == 'l1':
            gradient_token = torch.norm(gradient, p=1, dim=1)
        elif option == 'l2':
            gradient_token = torch.norm(gradient, p=2, dim=1)
        elif option == 'xInput':
            gradient_token = torch.diag(torch.matmul(e, gradient.T)[0])
        else:
            gradient_token = torch.mean(gradient, dim=1)

        gradient_words = {w: float(a) for w, a in zip(tokens, gradient_token[1:params.MAX_LEN + 1])}
        # Sort the attention dictionary by attention score in descending order.
        gradient_exp = dict(sorted(gradient_words.items(), key=lambda x: x[1], reverse=True))
        return gradient, gradient_exp
