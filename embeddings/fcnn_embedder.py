"""embeddings/fcnn_embedder.py – Extract FCNN hidden-layer representations.

Implements Phase 1 (FCNN Embeddings) and the FCNN simulation from Soto et al.
Uses a fine-tuned MobileNetV2 backbone (most informative per paper) with
adaptive pooling + a configurable hidden layer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config.settings import Settings

logger = logging.getLogger(__name__)


# ── Minimal image dataset for inference ─────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: T.Compose) -> None:
        self._paths = image_paths
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        img = Image.open(self._paths[idx]).convert("L")   # greyscale as in paper
        img = img.convert("RGB")                          # model expects 3-channel
        return self._transform(img), idx


# ── FCNN model wrapper ───────────────────────────────────────────────────────

class _FCNNModel(nn.Module):
    """
    MobileNetV2 backbone + adaptive pooling + hidden FC layer.
    Matches the architecture described in Soto et al. Methods:
        - Pre-trained convolutional weights frozen
        - New fully connected hidden layer (configurable units)
        - Classification layer (2-class softmax)
    """

    def __init__(
        self,
        n_hidden_units: int = 300,
        dropout_rate: float = 0.0,
        hidden_activation: str = "relu",
    ) -> None:
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Freeze convolutional weights
        for param in backbone.features.parameters():
            param.requires_grad = False

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        feature_dim = 1280  # MobileNetV2 output channels

        self.hidden = nn.Linear(feature_dim, n_hidden_units)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = self._build_activation(hidden_activation)
        self.classifier = nn.Linear(n_hidden_units, 2)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        mapping = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "linear": nn.Identity(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return mapping.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden_repr, logits)."""
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        h = self.activation(self.dropout(self.hidden(x)))
        logits = self.classifier(h)
        return h, logits


# ── Main embedder class ──────────────────────────────────────────────────────

class FCNNEmbedder:
    """
    Extracts hidden-layer representations from a MobileNetV2-based FCNN
    for the 96 experimental stimuli at two noise levels:
        * clear  (noise_sigma² = 0)    → conscious-analog
        * noisy  (noise_sigma² = 300)  → unconscious-analog

    Fine-tuning is performed once and the checkpoint is persisted to disk.
    On subsequent runs the checkpoint is loaded automatically, skipping training.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, settings: Settings) -> None:
        cfg = settings.fcnn
        self._device = torch.device(cfg.get("device", "cpu"))
        self._noise_levels: dict[str, float] = cfg["noise_levels"]
        self._n_sessions: int = cfg.get("n_noise_sessions", 20)
        self._batch_size: int = cfg.get("batch_size", 8)

        # Checkpoint path: from config if provided, else default location
        checkpoint_rel = cfg.get("checkpoint_path", "checkpoints/fcnn_finetuned.pt")
        self._checkpoint_path = Path(checkpoint_rel)
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self._model = _FCNNModel(
            n_hidden_units=cfg.get("n_hidden_units", 300),
            dropout_rate=cfg.get("dropout_rate", 0.0),
            hidden_activation=cfg.get("hidden_activation", "relu"),
        ).to(self._device)

        # Auto-load checkpoint if it already exists
        if self._checkpoint_path.exists():
            self._load_checkpoint(self._checkpoint_path)
        else:
            logger.info(
                "No FCNN checkpoint found at %s – model will use ImageNet init "
                "until finetune() is called.", self._checkpoint_path
            )

        self._model.eval()

        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])

    # ── Public API ───────────────────────────────────────────────────────────

    def is_finetuned(self) -> bool:
        """Return True if a fine-tuned checkpoint already exists on disk."""
        return self._checkpoint_path.exists()

    def load_weights(self, checkpoint_path: str | Path) -> None:
        """Load fine-tuned weights from a .pt/.pth checkpoint (manual override)."""
        self._load_checkpoint(Path(checkpoint_path))

    def extract_embeddings(
        self,
        image_paths: list[Path],
        noise_state: str = "clear",
    ) -> np.ndarray:
        """
        Extract hidden-layer representations for a list of images.

        Parameters
        ----------
        image_paths : List of paths to stimulus images
        noise_state : "clear" (σ²=0) or "chance" (σ²≈300)

        Returns
        -------
        np.ndarray of shape (n_stimuli, n_hidden_units)
            Averaged across ``n_noise_sessions`` repetitions.
        """
        sigma2 = float(self._noise_levels.get(noise_state, 0))
        sigma = np.sqrt(sigma2)

        dataset = _ImageDataset(image_paths, self._transform)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

        all_hidden: list[np.ndarray] = []

        for _session in range(self._n_sessions):
            session_hidden = self._run_inference(loader, sigma)
            all_hidden.append(session_hidden)

        # Average across sessions (mimicking the 20-session protocol in paper)
        embeddings = np.stack(all_hidden, axis=0).mean(axis=0)
        logger.info(
            "Extracted FCNN embeddings (noise_state=%s, shape=%s)",
            noise_state, embeddings.shape,
        )
        return embeddings

    def finetune(
            self,
            train_paths: list[Path],
            train_labels: list[int],
            val_paths: list[Path] = None,
            val_labels: list[int] = None,
            max_epochs: int = 200,
            lr: float = 1e-4,
            checkpoint_path: str | Path | None = None,
    ) -> None:
        """
        Fine-tune the hidden + classifier layers on clear images.
        Convolutional backbone remains frozen.

        Implements early stopping (every 10 epochs, 5 non-improving checks)
        and batch-wise noise augmentation based on the paper.
        """
        save_path = Path(checkpoint_path) if checkpoint_path is not None else self._checkpoint_path

        # ── Idempotency guard: skip if checkpoint already present ─────────
        if save_path.exists():
            logger.info(
                "FCNN checkpoint already exists at %s – skipping fine-tuning.", save_path
            )
            self._load_checkpoint(save_path)
            self._model.eval()
            return

        logger.info("Starting FCNN fine-tuning (max %d epochs, lr=%.1e)…", max_epochs, lr)

        self._model.train()
        for p in list(self._model.hidden.parameters()) + list(self._model.classifier.parameters()):
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            list(self._model.hidden.parameters()) + list(self._model.classifier.parameters()),
            lr=lr,
        )
        criterion = nn.CrossEntropyLoss()

        train_dataset = _ImageDataset(train_paths, self._transform)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)

        val_loader = None
        if val_paths and val_labels:
            val_dataset = _ImageDataset(val_paths, self._transform)
            val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            self._model.train()

            for imgs, indices in train_loader:
                imgs = imgs.to(self._device)
                labels_batch = torch.tensor(
                    [train_labels[i] for i in indices], dtype=torch.long
                ).to(self._device)

                # --- FIX 2: Noise-Augmented Training Batches ---
                # Calculate batch mean and std per channel
                b_mean = imgs.mean(dim=[0, 2, 3], keepdim=True)
                b_std = imgs.std(dim=[0, 2, 3], keepdim=True)

                # Sample 1 noise image using the batch's distribution
                noise_img = torch.randn(1, imgs.size(1), imgs.size(2), imgs.size(3), device=self._device)
                noise_img = (noise_img * b_std) + b_mean

                # Append the noise image and assign a dummy label (e.g., 0)
                imgs = torch.cat([imgs, noise_img], dim=0)
                dummy_label = torch.tensor([0], dtype=torch.long, device=self._device)
                labels_batch = torch.cat([labels_batch, dummy_label], dim=0)
                # -----------------------------------------------

                optimizer.zero_grad()
                _, logits = self._model(imgs)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # --- FIX 1: Early Stopping every 10 epochs ---
            if val_loader and (epoch + 1) % 10 == 0:
                self._model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for v_imgs, v_indices in val_loader:
                        v_imgs = v_imgs.to(self._device)
                        v_labels_batch = torch.tensor(
                            [val_labels[i] for i in v_indices], dtype=torch.long
                        ).to(self._device)

                        _, v_logits = self._model(v_imgs)
                        batch_loss = criterion(v_logits, v_labels_batch)
                        val_loss += batch_loss.item()

                val_loss /= len(val_loader)
                logger.info("Epoch %d/%d – Train Loss: %.4f | Val Loss: %.4f",
                            epoch + 1, max_epochs, epoch_loss / len(train_loader), val_loss)

                # Check stopping criterion
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self._model.state_dict())
                else:
                    patience_counter += 1
                    logger.info("Validation loss did not decrease. Patience: %d/5", patience_counter)

                if patience_counter >= 5:
                    logger.info("Early stopping triggered at epoch %d. Restoring best weights.", epoch + 1)
                    if best_model_state:
                        self._model.load_state_dict(best_model_state)
                    break
            elif not val_loader and (epoch + 1) % 10 == 0:
                # Fallback if no val set is provided
                logger.info("Epoch %d/%d – Train Loss: %.4f", epoch + 1, max_epochs, epoch_loss / len(train_loader))

        self._model.eval()

        # ── Save checkpoint ───────────────────────────────────────────────
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the best model state if validation was used, otherwise save current state
        state_to_save = best_model_state if best_model_state else self._model.state_dict()
        torch.save(state_to_save, str(save_path))

        # Ensure the model in memory has the best weights loaded
        if best_model_state:
            self._model.load_state_dict(best_model_state)

        logger.info("FCNN checkpoint saved → %s", save_path)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_checkpoint(self, path: Path) -> None:
        state = torch.load(str(path), map_location=self._device)
        self._model.load_state_dict(state)
        logger.info("Loaded FCNN weights from %s", path)

    def _run_inference(
        self,
        loader: DataLoader,
        sigma: float,
    ) -> np.ndarray:
        """Run one inference pass over all images, optionally with Gaussian noise."""
        all_hidden = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self._device)
                if sigma > 0:
                    noise = torch.randn_like(imgs) * (sigma / 255.0)
                    imgs = (imgs + noise).clamp(0, 1)
                hidden, _ = self._model(imgs)
                all_hidden.append(hidden.cpu().numpy())
        return np.concatenate(all_hidden, axis=0)
