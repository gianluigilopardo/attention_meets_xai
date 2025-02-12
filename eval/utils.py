import params


def get_token_pos(classifier, example):
    """Extracts token positions from a given example using a tokenizer.

    This function tokenizes the input text example and records the positions
    of each token in a dictionary. The positions are adjusted to start from 1
    and are limited by a maximum length parameter.

    Args:
        classifier: An object with a 'tokenizer' method that tokenizes the input example.
        example: A string representing the text input to be tokenized.

    Returns:
        dict: A dictionary where keys are tokens and values are lists of positions
              (1-indexed) where the tokens appear in the tokenized sequence, sorted by token."""
    tokens = classifier.tokenizer(example)
    token_pos = {}  # Dictionary to store token positions
    for i, token in enumerate(tokens):
        if i < params.MAX_LEN - 1:
            if token not in token_pos:  # Check if token exists in the dictionary
                token_pos[token] = []  # Initialize an empty list for the token
            token_pos[token].append(i + 1)  # Append the current position to the token's list

    return dict(sorted(token_pos.items()))

