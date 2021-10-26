"""
film.py

PyTorch-Lightning Module Definition for the FiLM-GeLU Conditional Auto-Encoding (FiLM) Model.
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


class FiLM(pl.LightningModule):
    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        lang_dim: int = 768,
        latent_dim: int = 2,
        hidden_dim: int = 30,
        lr: float = 0.01,
        lr_step_size: int = 200,
        lr_gamma: float = 0.1,
        zaug: bool = True,
        zaug_lambda: float = 10.0,
        retrieval: bool = True,
        run_dir: Path = None,
    ):
        super(FiLM, self).__init__()

        # Save Hyperparameters
        self.state_dim, self.action_dim, self.lang_dim = state_dim, action_dim, lang_dim
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.lr, self.lr_step_size, self.lr_gamma = lr, lr_step_size, lr_gamma

        # If True, Train Dataset will have augmented data batch --> combine losses!
        self.zaug, self.zaug_lambda = zaug, zaug_lambda

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
        # Encoder FiLM Generator --> Takes (Language) --> Encodes to enc_gamma, enc_beta
        self.enc_film_gen = nn.Sequential(
            nn.Linear(self.lang_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.efg, self.efb = nn.Linear(self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim)

        # Encoder --> Takes (State, Action) --> Encodes to `z` latent space
        self.enc2film = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.film2latent = nn.Sequential(nn.GELU(), nn.Linear(self.hidden_dim, self.latent_dim))

        # Decoder FiLM Generator --> Takes (Language) --> Encodes to dec_gamma, dec_beta
        self.dec_film_gen = nn.Sequential(
            nn.Linear(self.lang_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.dfg, self.dfb = nn.Linear(self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim)

        # Decoder --> Takes State + Latent Action --> Decodes to Action Space
        self.dec2film = nn.Sequential(
            nn.Linear(self.state_dim + self.latent_dim, self.hidden_dim, bias=not self.constrain_decoder),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=not self.constrain_decoder),
        )
        self.film2action = nn.Sequential(
            nn.GELU(), nn.Linear(self.hidden_dim, self.action_dim, bias=not self.constrain_decoder)
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        return [optimizer], [scheduler]

    def decoder(self, s: torch.Tensor, emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Create Input to Decoder --> (s, z)
        y = torch.cat([s, z], 1)

        # Project up to "FiLMing" Space
        to_film = self.dec2film(y)

        # Generate Decoder FiLM Gamma & Beta
        film_emb = self.dec_film_gen(emb)
        gamma, beta = self.dfg(film_emb), self.dfb(film_emb)

        # FiLM!
        dec_filmed = (gamma * to_film) + beta

        # Return predicted action
        return self.film2action(dec_filmed)

    def forward(self, s: torch.Tensor, emb: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Create Input to Encoder --> (s, a)
        x = torch.cat([s, a], 1)

        # Project up to "FiLMing" Space
        to_film = self.enc2film(x)

        # Generate Encoder FiLM Gamma & Beta
        film_emb = self.enc_film_gen(emb)
        gamma, beta = self.efg(film_emb), self.dfb(film_emb)

        # FiLM!
        enc_filmed = (gamma * to_film) + beta

        # Get Z
        z = self.film2latent(enc_filmed)

        # Return Predicted Action via Decoder
        return self.decoder(s, emb, z)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Regular Pipeline
        if not self.zaug:
            # Extract Batch
            state, emb, action = batch

            # Get Predicted Action
            predicted_action = self.forward(state, emb, action)

            # Measure MSE Loss
            loss = F.mse_loss(predicted_action, action)

            # Log Loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

        # Augmentation Pipeline
        else:
            # Extract Batches
            (state, emb, action), (aug_state, aug_emb, zero_action) = batch

            # First, "regular" pipeline
            predicted_action = self.forward(state, emb, action)
            loss = F.mse_loss(predicted_action, action)

            # Next, "augmented" (decoder-only) pipeline
            predicted_zero_action = self.decoder(aug_state, aug_emb, torch.zeros_like(aug_state)[:, : self.latent_dim])
            loss += self.zaug_lambda * F.mse_loss(predicted_zero_action, zero_action)

            # Log Loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # Extract Batch
        state, emb, action = batch

        # Get Predicted Action
        predicted_action = self.forward(state, emb, action)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action)

        # Log Loss
        self.log("val_loss", loss, prog_bar=True)
