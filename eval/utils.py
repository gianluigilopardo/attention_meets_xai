import params


def get_token_pos(classifier, example):
    tokens = classifier.tokenizer(example)
    token_pos = {}  # Dictionary to store token positions
    for i, token in enumerate(tokens):
        if i < params.MAX_LEN - 1:
            if token not in token_pos:  # Check if token exists in the dictionary
                token_pos[token] = []  # Initialize an empty list for the token
            token_pos[token].append(i + 1)  # Append the current position to the token's list

    return dict(sorted(token_pos.items()))

