TRAIN_SIZE = 20000  # Number of training examples
VAL_SIZE = 5000  # Number of validation examples
TEST_SIZE = 25000  # Number of test examples

EPOCHS = 10  # Number of training epochs
LR = 0.0001  # Learning rate

SCALE = True  # / \sqrt(d_k)
MAX_LEN = 256  # Maximum length of a sentence
BATCH_SIZE = 16  # Batch size for training and evaluation
EMBED_DIM = 128  # Dimension of word embeddings (d_in)
ATTN_DIM = 64  # Dimension of attention layer (d_attn)
d_out = 64  # Dimension of the output layer (d_out)
POS = True  # Positional encoding

MASK = 'UNK'
NUM_HEADS = 6

