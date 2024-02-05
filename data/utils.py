import torch

# Define text and label pipelines
label_pipeline = lambda x: 1 if x == 'pos' else 0  # Encode labels as 1 for 'pos' and 0 for 'neg'


def process_text(text, max_len, vocab, tokenizer):
    """
    Preprocesses and pads the input text to a maximum length.

    Args:
        text (str): Input text.
        max_len (int): Maximum length of the padded text.

    Returns:
        processed_text (Tensor): Padded text of shape (max_len,).
    """

    text_pipeline = lambda x: vocab(tokenizer('[cls] ' + x))  # Convert text to token IDs using the vocabulary

    # Convert the text to tokens using the text pipeline
    processed_text = torch.tensor(text_pipeline(text)[:max_len], dtype=torch.long)

    # Pad the text with zeros to reach the maximum length
    processed_text = torch.cat([processed_text, vocab(tokenizer('[unk]'))[0] * torch.ones(max_len - len(processed_text))])

    return processed_text


def get_batch_fn(max_len, vocab, tokenizer, device):
    """
    Creates a batch collate function for text classification.

    Args:
        max_len (int): Maximum length of the padded text.

    Returns:
        collate_batch (function): Batch collate function.
    """

    def collate_batch(batch):
        """
        Batch collate function for text classification.

        Args:
            batch (list): A list of training examples.

        Returns:
            labels (Tensor): Batch of labels of shape (batch_size,).
            texts (Tensor): Batch of padded texts of shape (batch_size, max_len).
        """

        # Extract labels and texts from the batch
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = process_text(_text, max_len=max_len, vocab=vocab, tokenizer=tokenizer)
            text_list.append(processed_text)

        # Convert labels to LongTensor and move them to the device
        labels = torch.tensor(label_list, dtype=torch.long)
        labels = labels.to(device)

        # Stack texts into a Tensor and move them to the device
        texts = torch.stack(text_list)
        texts = texts.to(device)

        return labels, texts

    return collate_batch

