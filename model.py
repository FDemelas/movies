"""
model.py
========
Architecture image-only pour la prédiction de genre (multi-label).

    EfficientNet-B0 (pré-entraîné ImageNet, backbone gelé en phase 1)
         ↓ avgpool → (1280,)
    MLP tête de classification
         ↓
    logits (num_genres,)  — Sigmoid appliqué à l'inférence
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class PosterGenreClassifier(nn.Module):
    """
    EfficientNet-B0 pré-entraîné + tête MLP pour classification multi-label.

    Args:
        num_genres   : nombre de genres (taille de la sortie)
        dropout      : taux de dropout dans la tête
        freeze       : si True, le backbone est gelé (phase 1)
    """

    def __init__(self, num_genres: int, dropout: float = 0.4, freeze: bool = True):
        super().__init__()

        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features   # CNN  → (B, 1280, 7, 7)
        self.avgpool  = backbone.avgpool    # pool → (B, 1280, 1, 1)

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

        # Tête de classification
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_genres),    # logits bruts — pas de Sigmoid ici
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)               # (B, 1280, 7, 7)
        x = self.avgpool(x)                # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)           # (B, 1280)
        return self.head(x)                # (B, num_genres)

    def unfreeze(self, blocks_from_end: int = 3):
        """
        Dégèle les derniers blocs du backbone pour le fine-tuning.
        blocks_from_end=3 : dégèle uniquement les 3 derniers blocs MBConv
        (bon compromis entre stabilité et adaptation).
        """
        blocks = list(self.features.children())
        for block in blocks[-blocks_from_end:]:
            for p in block.parameters():
                p.requires_grad = True
