"""
train.py

Core training script -- loads and preprocesses collected Panda demonstrations for a given task (or set of tasks),
instantiates a Lightning Module, and runs LILA or Imitation Learning training!

To be extended with support for multiple tasks, baselines, and eventually, support for language instructions!

Run with: `python train.py --config conf/lila-mturk-all-config.yaml`
"""
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from quinine import QuinineArgumentParser
from torch.utils.data import DataLoader

from src.models import GCAE, FiLM, Imitation
from src.overwatch import MetricsLogger, get_overwatch
from src.preprocessing import get_imitation_dataset, get_la_dataset, get_lila_dataset
from src.util import create_paths


# Disable Tokenizers Parallelism to Avoid Deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train() -> None:
    # Parse Quinfig (via Quinine Argparse Binding)
    print("[*] LILA :: Training Latent Actions =>>>")
    quinfig = QuinineArgumentParser().parse_quinfig()

    # Create Unique Run Name (for Logging, Checkpointing) :: Initialize all Directories
    run_id = quinfig.run_id
    if run_id is None:
        run_id = f"lila-{quinfig.mode}-{quinfig.arch}+{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    paths = create_paths(run_id, quinfig.run_dir)

    # Overwatch :: Setup & Configure Console/File Logger --> Handle Process 0 vs. other Process Logging!
    overwatch = get_overwatch(paths["runs"], run_id, quinfig.log_level)
    overwatch.info(f"Starting Run: {run_id}...")

    # Set Randomness
    overwatch.info(f"Setting Random Seed to {quinfig.seed}!")
    np.random.seed(quinfig.seed)
    torch.manual_seed(quinfig.seed)

    # Build Dataset & DataLoaders --> may want to play with the default parameters to this function!
    overwatch.info(f"Building Dataset using Demonstrations located at `{quinfig.demonstrations}`...")

    # Obtain Demonstrations based on "modality" of architectures
    if "lila" in quinfig.arch:
        train, validation = get_lila_dataset(
            quinfig.demonstrations,
            paths["runs"],
            augment=quinfig.augment,
            augment_factor=quinfig.augmentation_factor,
            seed=quinfig.seed,
        )
    elif "no-lang" in quinfig.arch:
        train, validation = get_la_dataset(
            quinfig.demonstrations, augment=quinfig.augment, augment_factor=quinfig.augment_factor, seed=quinfig.seed
        )

    elif "imitation" in quinfig.arch:
        train, validation = get_imitation_dataset(
            quinfig.demonstrations,
            run_dir=paths["runs"],
            augment=quinfig.augment,
            augment_factor=quinfig.augmentation_factor,
            seed=quinfig.seed,
        )
    else:
        raise NotImplementedError(f"Dataset Loading for Architecture {quinfig.arch} isn't supported...")

    # Create DataLoaders
    train_loader = DataLoader(train, batch_size=quinfig.bsz, shuffle=True)
    val_loader = DataLoader(validation, batch_size=quinfig.bsz, shuffle=False)

    # Create Model (one of multiple architectures)
    if quinfig.arch in ["lila"]:
        overwatch.info("Initializing LILA :: FiLM-GeLU Conditional Autoencoder...")
        nn = FiLM(
            quinfig.state_dim,
            quinfig.action_dim,
            768,
            quinfig.latent_dim,
            hidden_dim=quinfig.hidden_dim,
            lr=quinfig.lr,
            lr_step_size=quinfig.lr_step_size,
            lr_gamma=quinfig.lr_gamma,
            zaug=True,
            zaug_lambda=quinfig.zaug_lambda,
            retrieval=True,
            run_dir=paths["runs"],
        )
    elif quinfig.arch in ["no-lang"]:
        overwatch.info("Initializing No-Language Latent Actions :: GeLU Conditional Autoencoder...")
        nn = GCAE(
            quinfig.state_dim,
            quinfig.action_dim,
            quinfig.latent_dim,
            hidden_dim=quinfig.hidden_dim,
            lr=quinfig.lr,
            lr_step_size=quinfig.lr_step_size,
            lr_gamme=quinfig.lr_gamma,
            zaug=True,
            zaug_lambda=quinfig.zaug_lambda,
            run_dir=paths["runs"],
        )

    elif quinfig.arch in ["imitation"]:
        overwatch.info("Initializing Imitation Learning :: FiLM-GELU Behavioral Cloning...")
        nn = Imitation(
            quinfig.state_dim,
            quinfig.action_dim,
            768,
            hidden_dim=quinfig.hidden_dim,
            lr=quinfig.lr,
            lr_step_size=quinfig.lr_step_size,
            lr_gamma=quinfig.lr_gamma,
            unnatural=True,
            run_dir=paths["runs"],
        )
    else:
        raise NotImplementedError(f"Model `{quinfig.arch}` not implemented!")

    # Create Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["runs"],
        filename=f"{run_id}+" + "{train_loss:.2f}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss")
    logger = MetricsLogger(name=run_id, save_dir=paths["runs"])

    # Train!
    overwatch.info("Training...")
    trainer = pl.Trainer(
        max_epochs=quinfig.epochs, gpus=quinfig.gpus, logger=logger, callbacks=[checkpoint_callback, early_stopping]
    )
    trainer.fit(nn, train_loader, val_loader)


if __name__ == "__main__":
    train()
