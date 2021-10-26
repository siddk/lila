"""
imitation.py

PyTorch-Lightning Module Definition for the FiLM-GeLU Behavioral Cloning Model for predicting continuous joint
velocities.
"""
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from annoy import AnnoyIndex


class Imitation(pl.LightningModule):
    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        lang_dim: int = 768,
        hidden_dim: int = 30,
        lr: float = 0.01,
        lr_step_size: int = 200,
        lr_gamma: float = 0.1,
        retrieval: bool = True,
        run_dir: Path = None,
    ):
        super(Imitation, self).__init__()

        # Save Hyperparameters
        self.state_dim, self.action_dim, self.lang_dim, self.hidden_dim = state_dim, action_dim, lang_dim, hidden_dim
        self.lr, self.lr_step_size, self.lr_gamma = lr, lr_step_size, lr_gamma

        # Pointer to Run Directory (just in case)
        self.run_dir = run_dir

        # Retrieval --> Load Index, idx2language
        if retrieval:
            self.index = AnnoyIndex(self.lang_dim, "angular")
            self.index.load(str(Path(self.run_dir) / "retrieval.ann"))

            with open(Path(self.run_dir) / "retrieval.pik", "rb") as f:
                self.idx2lang = pickle.load(f)

        # Build Model
        self.build_model()

    def build_model(self) -> None:
        # FiLM Generator --> Takes (Language) --> Encodes to enc_gamma, enc_beta
        self.film_gen = nn.Sequential(
            nn.Linear(self.lang_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.fg, self.fb = nn.Linear(self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim)

        # Pre-FiLM Encoder :: Takes (State, Action) --> Encodes to Hidden
        self.enc = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Post-FiLM Decoder :: Takes (Hidden) --> Decodes to Hidden
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        return [optimizer], [scheduler]

    def forward(self, s: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # Encode State
        to_film = self.enc(s)

        # Generate FiLM Gamma & Beta
        film_emb = self.film_gen(emb)
        gamma, beta = self.fg(film_emb), self.fb(film_emb)

        # FiLM!
        filmed = (gamma * to_film) + beta

        # Predict Action
        return self.dec(filmed)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Extract Batch
        state, emb, action = batch

        # Get Predicted Action
        predicted_action = self.forward(state, emb)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action)

        # Log Loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # Extract Batch
        state, emb, action = batch

        # Get Predicted Action
        predicted_action = self.forward(state, emb)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action)

        # Log Loss
        self.log("val_loss", loss, prog_bar=True)
