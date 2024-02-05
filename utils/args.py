import argparse

DATASETS = ['imdb']
MODELS = ['nano_transformer']


def parse_args():
    """Parses the command-line arguments.

  Returns:
    A namespace object containing the parsed arguments.
  """

    parser = argparse.ArgumentParser(description='Evaluate attention-based explanations')

    # Required arguments
    parser.add_argument('--dataset',
                        type=str,
                        choices=DATASETS,
                        required=True,
                        help='Name of the dataset.')
    parser.add_argument('--model',
                        type=str,
                        choices=MODELS,
                        required=True,
                        help='Name of the model.')

    # Optional arguments
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed for reproducibility.')

    return parser.parse_args()
