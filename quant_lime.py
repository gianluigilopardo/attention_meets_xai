import os
import pickle

import torch
import numpy as np

import pandas as pd

from lime import lime_text

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
exp_name = 'lime_vs_attn.csv'
lime_rep = 5

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
res_path = os.path.join('.', 'results', 'IMDB', ts, model_name)
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

# LIME initialization
lime_explainer = lime_text.LimeTextExplainer(class_names=['negative', 'positive'], mask_string=mask_string)

num_tokens = params.MAX_LEN - 1

errors = []
res = pd.DataFrame(columns=['Example', 'Tokens', 'Prediction',
                            'LIME', 'Approx', 'Error'])

# iterate over corpus
for i, doc in enumerate(texts):
    # preprocess for LIME compatibility
    doc = re.sub('[^A-Za-z0-9]+', ' ', doc)
    example = ' '.join(classifier.tokenizer(doc)[:params.MAX_LEN - 1])
    tokens = classifier.tokenizer(example)

    pred = classifier.predict_proba([example])

    x, emb, e, v, q, k, attention, g_u, W_k, W_q, W_v, W, q_cls, output = get_head_params(classifier, example)
    x_unk, emb_unk, e_unk, v_unk, q_unk, k_unk, g_unk = get_head_unk_params(classifier, q)

    token_pos = get_token_pos(classifier, example)  # occurrences position per token id

    lime_approx = {}  # Dictionary to store approximate LIME scores for tokens
    for token in token_pos.keys():
        # Calculate the approximate LIME score for the current token
        approx_score = 3 / 2 * np.mean(
            [np.sum(
                [
                    float((torch.matmul((attention[h][:, classifier.cls_pos, t] * v[h][:, t, :] -
                                         g_unk[h][:, t] * v_unk[h][:, t, :] / torch.sum(g_unk[h])),
                                        W[h])))
                    for t in token_pos[token]
                ])
                for h in range(params.NUM_HEADS)
            ])

        # Store the approximate LIME score for the token in the dictionary
        lime_approx[token] = approx_score

    # Get LIME explanations
    lime_exp_dict = {}  # Dictionary to store aggregated LIME explanations
    lime_avg_dict = {}  # Dictionary to store averages

    # Generate multiple LIME explanations for the example sentence
    for rep in range(lime_rep):

        # Generate LIME explanation for the current repetition
        lime_exp = lime_explainer.explain_instance(example, classifier.predict_proba, num_features=params.MAX_LEN - 1)

        # Convert the LIME explanation to a dictionary
        lime_exp_dict_rep = dict(lime_exp.as_list())

        # Iterate over the explanation dictionary and aggregate values
        for key, value in lime_exp_dict_rep.items():
            # Check if the key already exists in the aggregated dictionary
            if key not in lime_exp_dict:
                lime_exp_dict[key] = []  # Initialize the value list if it doesn't exist

            # Append the current value to the corresponding key's value list
            lime_exp_dict[key].append(value)

    # Calculate and store the average value for each key
    for key, values in lime_exp_dict.items():
        if key not in lime_avg_dict:
            lime_avg_dict[key] = sum(values) / lime_rep
        else:
            lime_avg_dict[key] += sum(values) / lime_rep

    labels = list(lime_exp_dict.keys())  # [:num_tokens]

    lime_approx_dict = {k: lime_approx[k] for k in labels}

    print(f'Example {i + 1} / {len(corpus)}')
    print(f'LIME: \n{lime_avg_dict}')
    print(f'Approx: \n{lime_approx_dict}')

    error = np.linalg.norm(
        np.fromiter(lime_avg_dict.values(), dtype=float) - np.fromiter(lime_approx_dict.values(), dtype=float))
    print(f'Norm 2 error: {error}')
    print()
    errors.append(error)

    res.loc[i] = [example, tokens, pred,
                  lime_exp_dict, lime_approx_dict, error]

    if i % 10 == 0:
        res.to_csv(os.path.join(res_path, exp_name))
        print(res)

res.to_csv(os.path.join(res_path, exp_name))




