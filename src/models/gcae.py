"""
gcae.py

PyTorch-Lightning Module Definition for the No-Language Latent Actions GELU Conditional Auto-Encoding (GCAE) Model.
"""
from pathlib import Path
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GCAE(pl.LightningModule):
    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        latent_dim: int = 2,
        hidden_dim: int = 30,
        lr: float = 0.01,
        lr_step_size: int = 200,
        lr_gamma: float = 0.1,
        zaug: bool = True,
        zaug_lambda: float = 10.0,
        run_dir: Path = None,
    ):
        super(GCAE, self).__init__()

        # Save Hyperparameters
        self.state_dim, self.action_dim = state_dim, action_dim
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.lr, self.lr_step_size, self.lr_gamma = lr, lr_step_size, lr_gamma

        # If True, Train Dataset will have augmented data batch --> combine losses!
        self.zaug, self.zaug_lambda = zaug, zaug_lambda

        # Pointer to Run Directory (just in case)
        self.run_dir = run_dir

        # Build Model
        self.build_model()

    def build_model(self) -> None:
        # Encoder --> Takes (State, Action) --> Encodes to `z` latent space
        self.enc = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        # Decoder --> Takes State + Latent Action --> Decodes to Action Space
        self.dec = nn.Sequential(
            nn.Linear(self.state_dim + self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        return [optimizer], [scheduler]

    def decoder(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Create Input to Decoder --> (s, z)
        y = torch.cat([s, z], 1)

        # Return Predicted Action
        return self.dec(y)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ Default forward pass --> encode (s, a) --> z; decode (s, z) --> a. """
        x = torch.cat([s, a], 1)
        z = self.enc(x)

        # Return Predicted Action via Decoder
        return self.decoder(s, z)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Regular Pipeline
        if not self.zaug:
            # Extract Batch
            state, action = batch

            # Get Predicted Action
            predicted_action = self.forward(state, action)

            # Measure MSE Loss
            loss = F.mse_loss(predicted_action, action)

            # Log Loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

        # Augmentation Pipeline
        else:
            # Extract Batches
            (state, action), (aug_state, zero_action) = batch

            # First, "regular" pipeline
            predicted_action = self.forward(state, action)
            loss = F.mse_loss(predicted_action, action)

            # Next, "augmented" (decoder-only) pipeline
            predicted_zero_action = self.decoder(aug_state, torch.zeros_like(aug_state)[:, : self.latent_dim])
            loss += self.zaug_lambda * F.mse_loss(predicted_zero_action, zero_action)

            # Log Loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # Extract Batch
        state, action = batch

        # Get Predicted Action
        predicted_action = self.forward(state, action)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action)

        # Log Loss
        self.log("val_loss", loss, prog_bar=True)
