"""
la_dataset.py

Core class definition for datasets and preprocessing, and other necessary startup steps to run before training the
vanilla auto-encoding (Latent Actions) models.
"""
import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset


# Nest Overwatch under root `lila` logger, inheriting formatting!
overwatch = logging.getLogger("lila.preprocessing.la_dataset")


class SingleDemo(Dataset):
    """ Dataset Wrapper around a Single Demonstration in the Dataset. """

    def __init__(
        self, demo: List[List[Any]], window: int = 1, noise: bool = False, noise_std: float = 0.01, seed: int = 21
    ):
        self.demo = demo

        # Filter out "<GRIP>" actions -- User takes care of this for LA Models
        self.demo = [d for d in self.demo if not isinstance(d, str)]

        # Full Reproducibility
        self.noise, self.window = noise, window
        if self.noise:
            torch.manual_seed(seed)
            self.noise_vec = torch.normal(torch.zeros(len(self.demo), 7), std=noise_std)  # 7 = 7-DoF Joint States!

    def __len__(self) -> int:
        return len(self.demo) - self.window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # State is the first 7 Features (7-DoF) of the demonstration iteration (Velocity added at Runtime)
        state = torch.FloatTensor(self.demo[idx][:7])

        # Augmentation --> Add Noise (Fixed across Epochs) to Initial State
        if self.noise:
            state += self.noise_vec[idx]

        # Compute Next State with Fixed Window (Outer Loop iterates over _ALL_ window lengths)
        next_state = self.demo[idx + self.window][:7]

        # Compute Action as the Difference between the Two States (poor man's derivative = velocity)
        action = (torch.from_numpy(next_state) - state).float()

        return state, action


class ZeroDemo(Dataset):
    """ Dataset Wrapper around a `zero-augmented` (s, z=0, a=0) Demonstration in the Dataset. """

    def __init__(
        self, demo: List[List[Any]], window: int = 1, noise: bool = False, noise_std: float = 0.01, seed: int = 21
    ):
        self.demo = demo

        # Filter out "<GRIP>" actions -- User takes care of this for LA Models
        self.demo = [d for d in self.demo if not isinstance(d, str)]

        # Full Reproducibility
        self.noise, self.window = noise, window
        if self.noise:
            torch.manual_seed(seed)
            self.noise_vec = torch.normal(torch.zeros(len(self.demo), 7), std=noise_std)  # 7 = 7-DoF Joint States!

    def __len__(self) -> int:
        return len(self.demo) - self.window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # State is the first 7 Features (7-DoF) of the demonstration iteration (Velocity added at Runtime)
        state = torch.FloatTensor(self.demo[idx][:7])

        # Augmentation --> Add Noise (Fixed across Epochs) to Initial State
        if self.noise:
            state += self.noise_vec[idx]

        # Action = 0
        action = torch.zeros(7).float()

        return state, action


class AugmentedDataset(Dataset):
    """ Return k batches, with one batch from each of k datasets! """

    def __init__(self, *datasets: List[Dataset]):
        self.datasets = datasets

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return tuple(d[idx] for d in self.datasets)

    def __len__(self) -> int:
        return min(len(d) for d in self.datasets)


def get_la_dataset(
    demonstrations: Path,
    zaug: bool = True,
    val_split: float = 0.1,
    augment: bool = True,
    augment_factor: int = 1,
    window: int = 10,
    seed: int = 21,
) -> Tuple[Dataset, Dataset]:
    overwatch.info("Preparing LA Demonstration Dataset...")

    train, augmented, val = [], [], []
    for object_demo in [x for x in Path(demonstrations).iterdir() if ".pkl" in str(x)]:
        # Retrieve Data
        with open(object_demo, "rb") as f:
            demos = pickle.load(f)

        # Iterate through Demonstrations, allocate (1 - val_split) fraction to train, and the rest to val
        n_train = int(len(demos) * (1 - val_split))
        for offset, d in enumerate(demos[:n_train]):
            # No Window Augmentation...
            if not augment:
                # Add Demo to Train
                train.append(SingleDemo(d))

            # Consistent Actions for States that are Close Together
            else:
                # Iterate through `augment_factor`
                for af in range(augment_factor):
                    # Iterate through Window (Exhaustively!)
                    for w in range(1, window + 1):
                        train.append(SingleDemo(d, window=w, noise=True, seed=(af * 1000) + seed + offset))

                        # Add Augmented Data (z = 0, a = 0) to Augmentation Dataset
                        if zaug:
                            augmented.append(ZeroDemo(d, window=w, noise=True, seed=(af * 1000) + seed + offset))

        # Allocate the Remainder to Validation
        for offset, d in enumerate(demos[-max(1, len(demos) - n_train) :]):
            # No Window Augmentation...
            if not augment:
                # Add Demo to Validation
                val.append(SingleDemo(d))

            # Consistent Actions for States that are Close Together
            else:
                # Iterate through `augment_factor`
                for af in range(augment_factor):
                    # Iterate through Window (Exhaustively!)
                    for w in range(1, window + 1):
                        val.append(SingleDemo(d, window=w, noise=True, seed=(af * 1000) + seed + offset))

    # Create Datasets
    train, val = ConcatDataset(train), ConcatDataset(val)
    if zaug:
        augmented = ConcatDataset(augmented)

        # Create "new" Training Dataset
        train = AugmentedDataset(train, augmented)

    return train, val
