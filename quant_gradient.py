import os
import pickle

import torch
import numpy as np

import pandas as pd

import math
import io

import re

# Import the IMDB dataset from torchtext
from torchtext.datasets import IMDB

import datetime

from eval.utils import get_token_pos
from models.utils import scaled_dot_product
from data.dataset import Dataset

from eval.quant import get_head
from eval.quant import get_head_params
from eval.quant import get_head_unk_params

import params


# Experiment file
exp_name = 'grad_vs_attn.csv'

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model name and path
model_name = 'multi_head_e'
model_path = os.path.join('.', 'models', 'saved', 'IMDB', model_name, 'model')

# Load the classifier model
with open(os.path.join(model_path, 'classifier.pkl'), 'rb') as inp:
    classifier = pickle.load(inp)
print(f"Model loaded successfully from: {model_path}")

# results path
ts = datetime.datetime.now().strftime("%y%m%d%H%M")
res_path = os.path.join('.', 'results', 'IMDB', ts, model_name, 'gradient')
if not os.path.exists(res_path):
    os.makedirs(res_path)

# Define the corpus size
corpus_size = params.TEST_SIZE

# Load the IMDB dataset
dataset = Dataset(IMDB)

# Get the test subset
_, _, corpus = dataset.get_subsets(0, 0, corpus_size)
print(f"Corpus size: {len(corpus)}")

# Convert corpus to a NumPy array
sample = np.array(corpus)

# Extract only the second column (text content)
texts = sample[:, 1]

n_heads = params.NUM_HEADS
mask_string = params.MASK
d_k = params.ATTN_DIM

num_tokens = params.MAX_LEN - 1

errors = []
res = pd.DataFrame(columns=['Example', 'Tokens', 'Prediction'])

# iterate over corpus
for i, doc in enumerate(texts):
    # preprocess for LIME compatibility
    doc = re.sub('[^A-Za-z0-9]+', ' ', doc)
    example = ' '.join(classifier.tokenizer(doc)[:params.MAX_LEN - 1])
    tokens = classifier.tokenizer(example)

    print(f'Example {i + 1} / {len(corpus)}')

    pred = classifier.predict_proba([example])

    x, emb, e, v, q, k, attention, g_u, W_k, W_q, W_v, W, q_cls, output = get_head_params(classifier, example)
    x_unk, emb_unk, e_unk, v_unk, q_unk, k_unk, g_unk = get_head_unk_params(classifier, q)

    token_pos = get_token_pos(classifier, example)  # occurrences position per token id

    # Gradient from the theory
    grad_theory = []
    for h in range(n_heads):
        grad_head = []

        for t in range(params.MAX_LEN):
            p1 = torch.matmul(q_cls[h], W_k[h].T) / math.sqrt(d_k) * torch.matmul(v[h][:, t, :], W[h])
            p2 = torch.matmul(W[h], W_v[h].T)
            n1 = torch.sum(torch.stack([attention[h][:, classifier.cls_pos, s] *
                                        torch.matmul(q_cls[h], W_k[h].T) / math.sqrt(d_k) *
                                        torch.matmul(v[h][:, s, :], W[h])
                                        for s in range(params.MAX_LEN)]), dim=0)

            grad_head.append((attention[h][:, classifier.cls_pos, t]) * (p1 + p2 - n1))

        grad_theory.append(torch.cat(grad_head))

    res.loc[i] = [example, tokens, pred]

    grads = torch.stack(grad_theory)
    # torch.mean(torch.stack(grad_theory), dim=0)

    # Gradient from Pytorch
    grad = {}


    def get_gradient(name):
        def hook(model, grad_input, grad_output):
            grad[name] = grad_output[0].detach()

        return hook


    handle_gradient_pe = classifier.pe.register_full_backward_hook(get_gradient('pe'))
    Z = classifier(x)
    Z[0, 1].backward()
    # Print output and gradients
    gradient_e = torch.squeeze(grad['pe'])

    # attention
    att_path = os.path.join(res_path, 'attention')
    if not os.path.exists(att_path):
        os.makedirs(att_path)
    torch.save(torch.stack(attention), os.path.join(att_path, f'{i}.pt'))

    # grad_theory
    g_theory_path = os.path.join(res_path, 'grad_theory')
    if not os.path.exists(g_theory_path):
        os.makedirs(g_theory_path)
    torch.save(torch.stack(grad_theory), os.path.join(g_theory_path, f'{i}.pt'))

    # gradient_e
    g_e_path = os.path.join(res_path, 'gradient_e')
    if not os.path.exists(g_e_path):
        os.makedirs(g_e_path)
    torch.save(gradient_e, os.path.join(g_e_path, f'{i}.pt'))

    if i % 10 == 0:
        res.to_csv(os.path.join(res_path, exp_name))
        print(res)

res.to_csv(os.path.join(res_path, exp_name))
