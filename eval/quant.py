import math
import torch

import params

from models.utils import scaled_dot_product


def get_head(classifier, h):
    def head_classifier(x):
        head = classifier.heads[h]
        x = torch.stack([classifier.preprocess(t) for t in x])  # Preprocess the input text.
        with torch.no_grad():  # Disable gradient computation during prediction.
            emb = classifier.embedding(x.to(torch.int64))
            e = classifier.pe(emb)
            preds = head(e)  # Forward pass through the model.
        return preds

    return head_classifier


def get_head_params(classifier, example):
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


