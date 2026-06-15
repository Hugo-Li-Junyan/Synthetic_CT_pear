import torch
from torch.utils.data import random_split


def _split_count(dataset_size, split, name):
    if not 0 < split < 1:
        raise ValueError(f"{name} must be between 0 and 1, got {split}")
    count = int(dataset_size * split)
    if count == 0:
        raise ValueError(f"{name} produced 0 samples; increase {name} or dataset size")
    return count


def split_train_val(dataset, val_split, random_state):
    val_size = _split_count(len(dataset), val_split, "val_split")
    train_size = len(dataset) - val_size
    if train_size == 0:
        raise ValueError("Training split produced 0 samples; decrease val_split")

    generator = torch.Generator().manual_seed(random_state)
    return random_split(dataset, [train_size, val_size], generator=generator)


def split_train_val_test(dataset, val_split, test_split, random_state):
    val_size = _split_count(len(dataset), val_split, "val_split")
    test_size = _split_count(len(dataset), test_split, "test_split")
    train_size = len(dataset) - val_size - test_size
    if train_size == 0:
        raise ValueError("Training split produced 0 samples; decrease val_split or test_split")
    if train_size < 0:
        raise ValueError("val_split and test_split leave no room for training samples")

    generator = torch.Generator().manual_seed(random_state)
    train_dataset, val_test_dataset = random_split(
        dataset,
        [train_size, val_size + test_size],
        generator=generator,
    )
    val_dataset, test_dataset = random_split(
        val_test_dataset,
        [val_size, test_size],
        generator=generator,
    )
    return train_dataset, val_dataset, test_dataset
