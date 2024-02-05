import torchtext.datasets as datasets
import torchtext.data.functional as F
from torch.utils.data import DataLoader, Subset
import torch


class Dataset:
    """
    Class for loading and splitting the dataset for text classification.

    Args:
        dataset: A torchtext.datasets object

    Returns:
        Dataset object with methods for:
            - Getting train, validation, and test subsets
            - Creating dataloaders for train, validation, and test sets
    """

    def __init__(self, dataset):
        """
        Load the dataset using torchtext.
        """

        # Load the dataset from torchtext
        self.dataset = dataset
        self.vocab_iter = dataset(split='train')
        self.train_iter, self.test_iter = dataset(split=('train', 'test'))

        # Convert the dataset to map-style for easier processing
        self.train_dataset = F.to_map_style_dataset(self.train_iter)
        self.test_dataset = F.to_map_style_dataset(self.test_iter)

    def get_subsets(self, train_size, val_size, test_size):
        """
        Create train, validation, and test subsets from the loaded dataset.

        Args:
            train_size (int): Number of examples for the training subset
            val_size (int): Number of examples for the validation subset
            test_size (int): Number of examples for the test subset

        Returns:
            train_subset (torch.utils.data.Subset): Training subset
            val_subset (torch.utils.data.Subset): Validation subset
            test_subset (torch.utils.data.Subset): Test subset
        """

        # Create a random permutation of indices for the training data
        train_indices = torch.randperm(len(self.train_dataset))[:int(train_size + val_size)]

        # Split the training indices into train and validation subsets
        train_subset_indices = train_indices[:train_size]
        val_subset_indices = train_indices[train_size:]

        # Create train and validation subsets
        train_subset = Subset(self.train_dataset, train_subset_indices)
        val_subset = Subset(self.train_dataset, val_subset_indices)

        # Create a random permutation of indices for the test data
        test_indices = torch.randperm(len(self.test_dataset))[:test_size]

        # Create the test subset
        test_subset = Subset(self.test_dataset, test_indices)

        return train_subset, val_subset, test_subset

    def get_dataloaders(self, train_subset, val_subset, test_subset, batch_size, collate_batch):
        """
        Create dataloaders for train, validation, and test sets.

        Args:
            train_subset (torch.utils.data.Subset): Training subset
            val_subset (torch.utils.data.Subset): Validation subset
            test_subset (torch.utils.data.Subset): Test subset
            batch_size (int): Batch size for dataloaders
            collate_batch (function): Custom collate function for batching

        Returns:
            train_dl (torch.utils.data.DataLoader): Training dataloader
            val_dl (torch.utils.data.DataLoader): Validation dataloader
            test_dl (torch.utils.data.DataLoader): Test dataloader
        """

        # Create dataloaders for train, validation, and test sets with the specified batch size and collate function
        train_dl = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, drop_last=True
        )

        val_dl = DataLoader(
            val_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, drop_last=True
        )

        test_dl = DataLoader(
            test_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
        )

        return train_dl, val_dl, test_dl


