import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB

import os
import datetime
import json
import importlib

from data.dataset import Dataset
from data.utils import get_batch_fn

import params

import pickle

from models.multi_head import MultiHead

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For save & loads
dataset_name = 'IMDB'
model_name = f'multi_head_e'

# run = datetime.datetime.now().strftime("%d/%m/%y - %H:%M")
ts = datetime.datetime.now().strftime("%y%m%d%H%M")

model_path = os.path.join('.', 'models', 'saved', dataset_name, model_name, ts)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Loading the tokenizer for text preprocessing
tokenizer = get_tokenizer('basic_english')

# Loading the IMDB dataset
dataset = Dataset(IMDB)

# Splitting the dataset into training, validation, and test subsets
train_subset, val_subset, test_subset = dataset.get_subsets(params.TRAIN_SIZE, params.VAL_SIZE, params.TEST_SIZE)

# Building the vocabulary from the training data
vocab = build_vocab_from_iterator(
    map(lambda x: tokenizer(x[1]), dataset.vocab_iter), specials=['[cls]', params.MASK]
)
vocab.set_default_index(vocab['[cls]'])
vocab.set_default_index(vocab[params.MASK])

# Creating a collate_batch function to pad and package text sequences
collate_batch = get_batch_fn(params.MAX_LEN, vocab, tokenizer, device)

train_dl, val_dl, test_dl = dataset.get_dataloaders(train_subset, val_subset, test_subset, params.BATCH_SIZE,
                                                    collate_batch)


def train_step(model, x, y, optim):
    """
    Performs a single training step for the sentiment classifier model.

    Args:
        model: Sentiment classifier model.
        x (Tensor): Input text tensors.
        y (Tensor): Target labels tensors.
        optim (Optimizer): Optimizer for updating model parameters.

    Returns:
        loss (Tensor): Training loss.
        acc (Tensor): Training accuracy.
    """

    model.train()

    # Convert to LongTensor as required for embedding layers
    x = x.type(torch.LongTensor)

    # Forward pass: Compute model predictions
    preds = model(x)

    # Calculate cross-entropy loss
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))

    # Calculate accuracy
    acc = (preds.argmax(dim=-1) == y).float().mean()

    # Backward pass: Compute gradients and update model parameters
    loss.backward()
    optim.step()
    optim.zero_grad()

    return loss, acc


def eval_step(model, x, y):
    """
    Performs a single evaluation step for the sentiment classifier model.

    Args:
        model: Sentiment classifier model.
        x (Tensor): Input text tensors.
        y (Tensor): Target labels tensors.

    Returns:
        loss (Tensor): Evaluation loss.
        acc (Tensor): Evaluation accuracy.
    """

    with torch.no_grad():
        model.eval()

        # Convert to LongTensor as required for embedding layers
        x = x.type(torch.LongTensor)

        # Forward pass: Compute model predictions
        preds = model(x)

        # Calculate cross-entropy loss
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))

        # Calculate accuracy
        acc = (preds.argmax(dim=-1) == y).float().mean()

    return loss, acc


def train_epoch(model, train_loader, optim):
    """
    Performs a single training epoch for the sentiment classifier model.

    Args:
        model (SentimentClassifier): Sentiment classifier model.
        train_loader (DataLoader): Training data loader.
        optim (Optimizer): Optimizer for updating model parameters.

    Returns:
        loss (Tensor): Average training loss.
        acc (Tensor): Average training accuracy.
    """

    model.train()

    train_loss = 0.
    train_acc = 0.

    for idx, (y, x) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        loss, acc = train_step(model, x, y, optim)
        train_loss += loss
        train_acc += acc

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    return avg_train_loss, avg_train_acc


def eval_epoch(model, val_loader):
    """
    Performs a single evaluation epoch for the sentiment classifier model.

    Args:
        model (SentimentClassifier): Sentiment classifier model.
        val_loader (DataLoader): Validation data loader.

    Returns:
        loss (Tensor): Average validation loss.
        acc (Tensor): Average validation accuracy.
    """

    model.eval()

    val_loss = 0.
    val_acc = 0.

    for idx, (y, x) in enumerate(val_loader):
        loss, acc = eval_step(model, x, y)
        val_loss += loss
        val_acc += acc

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    return avg_val_loss, avg_val_acc


def train_model(model, train_loader, val_loader, test_loader, optim, epochs):
    """
    Trains the sentiment classifier model.

    Args:
        model: Sentiment classifier model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        optim (Optimizer): Optimizer for updating model parameters.
        epochs (int): Number of training epochs.

    Returns:
       : Trained sentiment classifier model.
    """

    best_acc = 0.

    for e in range(epochs):
        print(f"\nEpoch: {e + 1}/{epochs}")

        # Train the model for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optim)

        # Evaluate the model on the validation set
        val_loss, val_acc = eval_epoch(model, val_loader)

        # Save the model if it achieves the best validation accuracy so far
        if val_acc / len(val_loader) > best_acc:
            best_acc = val_acc / len(val_loader)
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pt'))

        # Print training and validation metrics
        print(
            f'Train accuracy: {train_acc * 100:.2f} | Train Loss: {train_loss:.2f}'
            f'\nVal. accuracy: {val_acc * 100:.2f} | Val. loss: {val_loss:.2f}'
        )

    # Load the best-performing model from the saved checkpoint
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))

    # Evaluate the model on the test set
    # test_loss, test_acc = eval_epoch(model, test_loader)
    # print(f'\nTest accuracy: {test_acc * 100:.3f}')

    return model


classifier = MultiHead(vocab_size=len(vocab), d_in=params.EMBED_DIM, d_out=params.d_out,
                       d_attn=params.ATTN_DIM, num_classes=2,
                       vocab=vocab, tokenizer=tokenizer, scale=params.SCALE, pos=params.POS,
                       num_heads=params.NUM_HEADS)

optimizer = optim.AdamW(classifier.parameters(), lr=params.LR)

classifier = train_model(classifier, train_dl, val_dl, test_dl, optimizer, epochs=params.EPOCHS)

# Evaluate the model on the test set
test_loss, test_acc = eval_epoch(classifier, test_dl)
print(f'\nTest accuracy: {test_acc * 100:.3f}')

#######################################################################################################################

# Save info and model

params = importlib.import_module('params')  # Import the 'params' module
dataset_name = 'IMDB'  # Set the dataset name

# Generate a timestamp for the experiment run
run = datetime.datetime.now().strftime("%d/%m/%y - %H:%M")

# Create a dictionary to store experiment information
info = {
    'Date': run,  # Experiment run date
    'Model': model_name,  # Model name
    'Dataset': dataset_name,  # Dataset name
    'Test accuracy': str(f'{test_acc * 100:.3f}'),  # Test accuracy rounded to three decimal places
}

# Iterate over the attributes of the 'params' module and add them to the 'info' dictionary
for attr in dir(params):
    if attr.isupper():
        info[attr] = getattr(params, attr)  # Extract parameter values from the 'params' module

# Convert the 'info' dictionary to JSON format with indentation
json_info = json.dumps(info, indent=4)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Save the model information as a JSON file
with open(os.path.join(model_path, 'info.json'), 'w') as file:
    file.write(json_info)

# Save the trained classifier model
with open(os.path.join(model_path, 'classifier.pkl'), 'wb') as outp:
    pickle.dump(classifier, outp, pickle.HIGHEST_PROTOCOL)  # Serialize the classifier object using Pickle
