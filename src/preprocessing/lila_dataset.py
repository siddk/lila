""""
lila_dataset.py

Core class definition for datasets and preprocessing, and other necessary startup steps to run before training
language-informed latent actions (LILA) models.
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import torch
from annoy import AnnoyIndex
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoModel, AutoTokenizer


# Nest Overwatch under root `lila` logger, inheriting formatting!
overwatch = logging.getLogger("lila.preprocessing.lila_dataset")


# Utility Function -- Mean Pooling for Obtaining BERT "Sentence" Embeddings
def pool(output, attention_mask):
    embeddings = output[0]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embedding_sum = torch.sum(embeddings * mask, dim=1)
    mask_sum = torch.clamp(mask.sum(1), min=1e-9)
    return (embedding_sum / mask_sum).squeeze()


class SingleLanguageDemo(Dataset):
    """ Dataset Wrapper around a Single (Demonstration, Language) Pair in the Dataset. """

    def __init__(
        self,
        demo: List[List[Any]],
        embedding: torch.Tensor,
        window: int = 1,
        noise: bool = False,
        noise_std: float = 0.01,
        seed: int = 21,
    ):
        self.demo, self.embedding = demo, embedding

        # Filter out "<GRIP>" actions -- User takes care of this for LILA Models
        self.demo = [d for d in self.demo if not isinstance(d, str)]

        # Full Reproducibility
        self.noise, self.window = noise, window
        if self.noise:
            torch.manual_seed(seed)
            self.noise_vec = torch.normal(torch.zeros(len(self.demo), 7), std=noise_std)

    def __len__(self) -> int:
        return len(self.demo) - self.window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # State is the first 7 Features (7-DoF) of the demonstration iteration (Velocity added at Runtime)
        state = torch.FloatTensor(self.demo[idx][:7])

        # Augmentation --> Add Noise (Fixed across Epochs) to Initial State
        if self.noise:
            state += self.noise_vec[idx]

        # Compute Next State with Fixed Window (Outer Loop iterates over _ALL_ window lengths)
        next_state = self.demo[idx + self.window][:7]

        # Compute Action as the Difference between the Two States (poor man's derivative = velocity)
        action = (torch.from_numpy(next_state) - state).float()

        return state, self.embedding, action


class ZeroLanguageDemo(Dataset):
    """ Dataset Wrapper around a `zero-language-augmented` (s, l, z=0, a=0) Demonstration in the Dataset. """

    def __init__(
        self,
        demo: List[List[Any]],
        embedding: torch.Tensor,
        window: int = 1,
        noise: bool = False,
        noise_std: float = 0.01,
        seed: int = 21,
    ):
        self.demo, self.embedding = demo, embedding

        # Filter out "<GRIP>" actions -- User takes care of this for LILA Models
        self.demo = [d for d in self.demo if not isinstance(d, str)]

        # Full Reproducibility
        self.noise, self.window = noise, window
        if self.noise:
            torch.manual_seed(seed)
            self.noise_vec = torch.normal(torch.zeros(len(self.demo), 7), std=noise_std)

    def __len__(self) -> int:
        return len(self.demo) - self.window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # State is the first 7 Features (7-DoF) of the demonstration iteration (Velocity added at Runtime)
        state = torch.FloatTensor(self.demo[idx][:7])

        # Augmentation --> Add Noise (Fixed across Epochs) to Initial State
        if self.noise:
            state += self.noise_vec[idx]

        # Action = 0
        action = torch.zeros(7).float()

        return state, self.embedding, action


class AugmentedDataset(Dataset):
    """ Return k batches, with one batch from each of k datasets! """

    def __init__(self, *datasets: List[Dataset]):
        self.datasets = datasets

    def __getitem__(
        self, idx
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return tuple(d[idx] for d in self.datasets)

    def __len__(self) -> int:
        return min(len(d) for d in self.datasets)


def get_lila_dataset(
    demonstrations: Path,
    run_dir: Path,
    zaug: bool = True,
    retrieval: bool = True,
    val_split: float = 0.1,
    augment: bool = True,
    augment_factor: int = 1,
    window: int = 10,
    seed: int = 21,
) -> Tuple[Dataset, Dataset]:
    overwatch.info("Preparing LILA Demonstration Dataset...")

    # Initialize *RoBERTa* Model!
    os.makedirs("cache/paraphrase", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
    )
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1", cache_dir="cache/paraphrase"
    )

    # If `retrieval` --> create lang2idx, ANN index
    if retrieval:
        lang2idx, index = {}, AnnoyIndex(model.config.hidden_size, "angular")

    train, augmented, val = [], [], []
    for object_demo in [x for x in Path(demonstrations).iterdir() if ".pkl" in str(x)]:
        # Retrieve Data
        with open(object_demo, "rb") as f:
            demos = pickle.load(f)

        # Fetch Language File
        lfile = Path(f"{str(object_demo)[:-4]}.jsonl")
        with open(lfile, "r") as f:
            language = [json.loads(x) for x in f.read().splitlines()]

        # Assert that # demos == # language examples
        #   Language Schema :: {id: <idx>, "language": List[str] of language instructions for demo}
        assert len(demos) == len(language), f"# Demonstrations in {object_demo} != # language instructions!"

        # Iterate through Demonstrations, allocate (1 - val_split) fraction to train, and the rest to val
        n_train = int(len(demos) * (1 - val_split))
        for offset, d in enumerate(demos[:n_train]):
            # Iterate through Language Instructions
            for lang in language[offset]["language"]:
                # Compute Language/Sentence Embedding
                encoding = tokenizer(lang, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    output = model(**encoding)

                # Mean Pooling
                embedding = pool(output, encoding["attention_mask"])

                # If `retrieval`, add to Index...
                if retrieval:
                    if lang not in lang2idx:
                        i = len(lang2idx)
                        lang2idx[lang] = i
                        index.add_item(i, embedding)

                # No Window Augmentation...
                if not augment:
                    # Add Demo to Train
                    train.append(SingleLanguageDemo(d, embedding))

                # Consistent Actions for States that are Close Together
                else:
                    # Iterate through `augment_factor`
                    for af in range(augment_factor):
                        # Iterate through Window (Exhaustively!)
                        for w in range(1, window + 1):
                            train.append(
                                SingleLanguageDemo(d, embedding, window=w, noise=True, seed=(af * 1000) + seed + offset)
                            )

                            # Add Augmented Data (z = 0, a = 0) to Augmentation Dataset
                            if zaug:
                                augmented.append(
                                    ZeroLanguageDemo(
                                        d, embedding, window=w, noise=True, seed=(af * 1000) + seed + offset
                                    )
                                )

        # Allocate the Remainder to Validation
        for offset, d in enumerate(demos[-max(1, len(demos) - n_train) :]):
            # Iterate through Language Instructions
            for lang in language[min(len(language), n_train + offset)]["language"]:
                # Compute Language/Sentence Embedding
                encoding = tokenizer(lang, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    output = model(**encoding)

                # Mean Pooling
                embedding = pool(output, encoding["attention_mask"])

                # No Window Augmentation...
                if not augment:
                    # Add Demo to Validation
                    val.append(SingleLanguageDemo(d, embedding))

                # Consistent Actions for States that are Close Together
                else:
                    # Iterate through `augment_factor`
                    for af in range(augment_factor):
                        # Iterate through Window (Exhaustively!)
                        for w in range(1, window + 1):
                            val.append(
                                SingleLanguageDemo(d, embedding, window=w, noise=True, seed=(af * 1000) + seed + offset)
                            )

    # If `retrieval` build trees, save metadata!
    if retrieval:
        overwatch.info("Building ANN Index w/ 10 Trees...")
        index.build(10)

        overwatch.info(f"Saving ANN Index and Language Metadata to {run_dir}...")
        index.save(str(Path(run_dir) / "retrieval.ann"))
        with open(Path(run_dir) / "retrieval.pik", "wb") as f:
            pickle.dump({idx: lang for lang, idx in lang2idx.items()}, f)

    # Create Datasets
    train, val = ConcatDataset(train), ConcatDataset(val)
    if zaug:
        augmented = ConcatDataset(augmented)

        # Create "new" Training Dataset
        train = AugmentedDataset(train, augmented)

    return train, val
