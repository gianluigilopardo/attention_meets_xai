import torch
import torch.nn.functional as F
import math


def scaled_dot_product(q, k, v, mask=None, return_g=False, scale=False):
    """
    Computes the scaled dot product attention.

    Args:
        q (Tensor): Queries of shape (batch_size, query_length, dim_q).
        k (Tensor): Keys of shape (batch_size, key_length, dim_k).
        v (Tensor): Values of shape (batch_size, key_length, dim_v).
        mask (Tensor, optional): Mask of shape (batch_size, query_length, key_length).

    Returns:
        output_values (Tensor): Output values of shape (batch_size, query_length, dim_v).
        attention (Tensor): Attention probabilities of shape (batch_size, query_length, key_length).
    """

    # Calculate attention logits using the scaled dot product
    attn_logits = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, query_length, key_length)

    if scale:
        d_k = q.size()[-1]  # Extract the dimension of the key (d_k)
        attn_logits = attn_logits / math.sqrt(d_k)  # Scale the attn_logits

    # Apply the mask if provided
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -10e14)  # Mask out invalid logits

    # Compute the attention probabilities using softmax
    attention = F.softmax(attn_logits, dim=-1)  # Shape: (batch_size, query_length, key_length)

    # Compute the weighted sum of values using the attention probabilities
    output_values = torch.matmul(attention, v)  # Shape: (batch_size, query_length, dim_v)

    if return_g:
        return output_values, attention, torch.exp(attn_logits)
    return output_values, attention, None


def get_g_values(model, x):
    x = torch.stack([model.preprocess(t) for t in x])  # Preprocess the input text.
    with torch.no_grad():  # Disable gradient computation during attention computation.
        # Forward pass through the model, including attention calculation.
        _, _, g_matrices = model.forward(x, return_attention=True, return_g=True)
    # Extract the attention score for the [CLS] token and the remaining words.
    g_scores = g_matrices[:, model.cls_pos, :]
    return g_matrices, g_scores

