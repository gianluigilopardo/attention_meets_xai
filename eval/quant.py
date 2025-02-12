import math
import torch

import params

from models.utils import scaled_dot_product


def get_head(classifier, h):
    """Creates a function that performs classification using a specified head of a classifier.

    This function generates a classifier function that processes input data using the specified
    head of the given classifier. The input data is preprocessed and passed through the classifier's
    embedding and positional encoding layers before being forwarded through the selected head.

    Args:
        classifier: An object representing the classifier with multiple heads. It must have
            attributes `heads`, `preprocess`, `embedding`, and `pe`.
        h: An integer index representing the specific head of the classifier to use for predictions.

    Returns:
        A function `head_classifier` that takes input data `x`, preprocesses it, and returns the
        predictions from the specified head of the classifier."""
    def head_classifier(x):
        """Classifies input text data using a specified model head.

    This function preprocesses the input text, converts it to embeddings,
    applies positional encoding, and performs a forward pass through
    the specified model head to produce predictions.

    Args:
        x (list of str): A list of input text data to be classified.

    Returns:
        torch.Tensor: The model's predictions for the input text."""
        head = classifier.heads[h]
        x = torch.stack([classifier.preprocess(t) for t in x])  # Preprocess the input text.
        with torch.no_grad():  # Disable gradient computation during prediction.
            emb = classifier.embedding(x.to(torch.int64))
            e = classifier.pe(emb)
            preds = head(e)  # Forward pass through the model.
        return preds

    return head_classifier


def get_head_params(classifier, example):
    """Extracts and computes parameters from a multi-head attention classifier.

    This function processes an input example through a classifier with multiple attention heads.
    It extracts query, key, and value representations, computes attention scores, and retrieves
    classifier outputs and transformation matrices for each attention head.

    Args:
        classifier: An object representing the multi-head attention classifier with methods for 
            preprocessing, embedding, and attention heads.
        example: An input example to be processed by the classifier.

    Returns:
        tuple: A tuple containing the following elements:
            - x: Tensor representation of the preprocessed input example.
            - emb: Embedding of the input example.
            - e: Positional encoding added to the embedding.
            - v: List of value representations for each attention head.
            - q: List of query representations for each attention head.
            - k: List of key representations for each attention head.
            - attention: List of attention scores for each attention head.
            - g_u: List of attention gradients for the classifier's position for each head.
            - W_k: List of key transformation matrices for each attention head.
            - W_q: List of query transformation matrices for each attention head.
            - W_v: List of value transformation matrices for each attention head.
            - W: List of classifier transformation matrices for each attention head.
            - q_cls: List of query representations for the classifier's position for each head.
            - output: List of classifier outputs for the attention-weighted values of each head."""
    v, q, k, attention, g_u = [], [], [], [], []
    W_k, W_q, W_v, W, q_cls = [], [], [], [], []
    output = []

    x = torch.stack([classifier.preprocess(example)])

    emb = classifier.embedding(x.to(torch.int64))
    e = classifier.pe(emb)

    for h in range(params.NUM_HEADS):
        # Transform the sentence embedding using the embedding layer
        head = classifier.heads[h]

        # Extract query, key, and value representations from the embedding
        q.append(head.query(e))
        k.append(head.key(e))
        v.append(head.value(e))

        W_k.append(head.key(torch.eye(e.shape[-1])))
        W_q.append(head.query(torch.eye(e.shape[-1])))
        W_v.append(head.value(torch.eye(e.shape[-1])))
        W.append(head.classifier(torch.eye(v[h].shape[-1]))[:, 1])
        q_cls.append(q[h][:, classifier.cls_pos, :])

        # Calculate scaled dot-product attention
        values, alpha, g = scaled_dot_product(q[h], k[h], v[h], return_g=True, scale=True)

        attention.append(alpha)
        g_u.append(g[:, classifier.cls_pos, :])

        # Extract attention-weighted values for the classifier layer
        cls_values = values[:, classifier.cls_pos, :]

        # Pass the attention-weighted values through the classifier to obtain predictions
        output.append(head.classifier(cls_values))

    return x, emb, e, v, q, k, attention, g_u, W_k, W_q, W_v, W, q_cls, output


def get_head_unk_params(classifier, q):
    """Generate embeddings and attention parameters for an "unknown" token.

    This function creates a sequence of "unknown" tokens and processes it through
    the given classifier to obtain its embedding and attention parameters for each head.
    It calculates the value, query, key, and attention (g) parameters for the "unknown" token.

    Args:
        classifier: An object with methods and attributes for processing tokens, including
            embedding, positional encoding, and heads for multi-head attention.
        q: A tensor representing the query component used in attention mechanisms.

    Returns:
        A tuple containing the following elements:
        - x_unk: Tensor representing the processed "unknown" sequence.
        - emb_unk: Embedding tensor for the "unknown" sequence.
        - e_unk: Positional encoded embedding tensor for the "unknown" sequence.
        - v_unk: List of value tensors from each attention head.
        - q_unk: List of query tensors from each attention head.
        - k_unk: List of key tensors from each attention head.
        - g_unk: List of attention parameters for each head, representing
          the attention scores for the "unknown" token."""
    unk_string = ' '.join([params.MASK for _ in range(params.MAX_LEN - 1)])
    # Preprocess the "UNK" token
    x_unk = torch.stack([classifier.preprocess(unk_string)])

    v_unk, q_unk, k_unk, g_unk = [], [], [], []

    # Convert the "UNK" token to embedding representation
    emb_unk = classifier.embedding(x_unk.to(torch.int64))
    e_unk = classifier.pe(emb_unk)

    for h in range(params.NUM_HEADS):
        # Extract the hidden state (h) from the "UNK" token embedding
        head = classifier.heads[h]

        v_unk.append(head.value(e_unk))  # + classifier.value(classifier.pe(x_unk.to(torch.int64)))
        q_unk.append(head.query(e_unk))  # + classifier.query(classifier.pe(x_unk.to(torch.int64)))
        k_unk.append(head.key(e_unk))  # + classifier.key(classifier.pe(x_unk.to(torch.int64)))

        d_k = q_unk[h].size()[-1]
        attn_logits = torch.matmul(q[h], k_unk[h].transpose(-2, -1)) / math.sqrt(d_k)
        g_unk.append(torch.exp(attn_logits)[:, classifier.cls_pos, :])

    return x_unk, emb_unk, e_unk, v_unk, q_unk, k_unk, g_unk


