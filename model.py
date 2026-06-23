"""
2_model.py
==========
Architecture multimodale pour la prédiction de genre (multi-label).

    TextBranch   (sentence-transformer embeddings → MLP)
    ImageBranch  (EfficientNet-B0 pré-entraîné → MLP)
    NumericBranch(features normalisées → MLP)
         ↓ concat
    FusionHead   (Dense → Dropout → Sigmoid)

Usage :
    python 2_model.py              # entraînement complet
    python 2_model.py --eval       # évaluation sur val set uniquement
"""

import argparse
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import f1_score, classification_report

# Importer le Dataset défini dans le script précédent
from preprocessing import MovieDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

DATA_DIR    = Path("data")
CHECKPOINT  = DATA_DIR / "best_model.pt"

# ──────────────────────────────────────────────
# HYPERPARAMÈTRES
# ──────────────────────────────────────────────
CFG = {
    "text_dim":      384,    # sortie de all-MiniLM-L6-v2
    "image_dim":     1280,   # sortie de EfficientNet-B0 (avgpool)
    "fusion_hidden": 512,
    "proj_dim":      256,    # dimension commune après chaque branche
    "dropout":       0.4,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "epochs":        30,
    "batch_size":    32,
    "patience":      5,      # early stopping
    "threshold":     0.5,    # seuil sigmoid → genre présent ou non
}


# ──────────────────────────────────────────────
# BLOCS DE L'ARCHITECTURE
# ──────────────────────────────────────────────
def mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Projection linéaire standard : Linear → BN → ReLU → Dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class TextBranch(nn.Module):
    """
    Reçoit un embedding sentence-transformer déjà calculé (384 dim).
    Projette vers proj_dim avec un MLP à 2 couches.
    """
    def __init__(self, in_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            mlp_block(in_dim, in_dim // 2, dropout),
            mlp_block(in_dim // 2, proj_dim, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, proj_dim)


class ImageBranch(nn.Module):
    """
    EfficientNet-B0 pré-entraîné (ImageNet).
    On remplace la tête de classification par un MLP de projection.
    Deux modes :
      - freeze=True  : on n'entraîne que la tête (phase 1)
      - freeze=False : fine-tuning complet (phase 2, optionnel)
    """
    def __init__(self, proj_dim: int, dropout: float, freeze: bool = True):
        super().__init__()
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Conserver le backbone sans la tête de classification
        self.features = backbone.features      # CNN
        self.avgpool  = backbone.avgpool        # AdaptiveAvgPool2d → (B, 1280, 1, 1)

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            mlp_block(1280, 512, dropout),
            mlp_block(512, proj_dim, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                   # (B, 1280, 7, 7)
        x = self.avgpool(x)                    # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)               # (B, 1280)
        return self.head(x)                    # (B, proj_dim)

    def unfreeze(self):
        """Déverrouille le backbone pour un fine-tuning de fin d'entraînement."""
        for p in self.features.parameters():
            p.requires_grad = True


class NumericBranch(nn.Module):
    """MLP pour les features numériques (popularité, votes, langue…)."""
    def __init__(self, in_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            mlp_block(in_dim, 64, dropout),
            mlp_block(64, proj_dim, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, proj_dim)


class MultiModalGenreClassifier(nn.Module):
    """
    Modèle complet.
    text   : (B, text_dim)
    image  : (B, 3, 224, 224)
    numeric: (B, num_dim)
    → logits: (B, num_genres)   — pas de sigmoid ici, inclus dans BCEWithLogitsLoss
    """
    def __init__(
        self,
        text_dim:   int,
        num_dim:    int,
        num_genres: int,
        proj_dim:   int   = 256,
        fusion_hidden: int = 512,
        dropout:    float = 0.4,
        freeze_image: bool = True,
    ):
        super().__init__()
        self.text_branch    = TextBranch(text_dim, proj_dim, dropout)
        self.image_branch   = ImageBranch(proj_dim, dropout, freeze=freeze_image)
        self.numeric_branch = NumericBranch(num_dim, proj_dim, dropout)

        # Fusion : concaténation des 3 vecteurs projetés
        fusion_in = proj_dim * 3
        self.fusion = nn.Sequential(
            mlp_block(fusion_in, fusion_hidden, dropout),
            mlp_block(fusion_hidden, fusion_hidden // 2, dropout),
            nn.Linear(fusion_hidden // 2, num_genres),
        )

    def forward(
        self,
        text:    torch.Tensor,
        image:   torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        h_text    = self.text_branch(text)       # (B, proj_dim)
        h_image   = self.image_branch(image)     # (B, proj_dim)
        h_numeric = self.numeric_branch(numeric) # (B, proj_dim)

        fused  = torch.cat([h_text, h_image, h_numeric], dim=1)  # (B, proj_dim*3)
        logits = self.fusion(fused)                               # (B, num_genres)
        return logits


# ──────────────────────────────────────────────
# BOUCLES D'ENTRAÎNEMENT / ÉVALUATION
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler_amp):
    model.train()
    total_loss = 0.0
    for batch in loader:
        text    = batch["text"].to(device)
        image   = batch["image"].to(device)
        numeric = batch["numeric"].to(device)
        labels  = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(text, image, numeric)
            loss   = criterion(logits, labels)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        text    = batch["text"].to(device)
        image   = batch["image"].to(device)
        numeric = batch["numeric"].to(device)
        labels  = batch["label"].to(device)

        logits = model(text, image, numeric)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(loader), f1_micro, f1_macro, all_preds, all_labels


# ──────────────────────────────────────────────
# PIPELINE D'ENTRAÎNEMENT
# ──────────────────────────────────────────────
def train(eval_only: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    # ── Chargement des données ──
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        data = pickle.load(f)
    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    df             = data["df"]
    text_embeds    = data["text_embeddings"]
    num_features   = data["num_features"]
    labels         = data["labels"]
    num_genres     = labels.shape[1]
    num_dim        = num_features.shape[1]

    log.info(f"Films : {len(df)}  |  Genres : {num_genres}  |  Num features : {num_dim}")

    # ── Split train / val ──
    idx = list(range(len(df)))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    image_paths = df["image_path"].tolist()
    train_ds = MovieDataset(
        [image_paths[i] for i in train_idx], text_embeds[train_idx],
        num_features[train_idx], labels[train_idx], train=True,
    )
    val_ds = MovieDataset(
        [image_paths[i] for i in val_idx], text_embeds[val_idx],
        num_features[val_idx], labels[val_idx], train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Modèle ──
    model = MultiModalGenreClassifier(
        text_dim      = CFG["text_dim"],
        num_dim       = num_dim,
        num_genres    = num_genres,
        proj_dim      = CFG["proj_dim"],
        fusion_hidden = CFG["fusion_hidden"],
        dropout       = CFG["dropout"],
        freeze_image  = True,   # Phase 1 : backbone CNN gelé
    ).to(device)

    log.info(f"Paramètres entraînables : "
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss & optimizer ──
    # Pondération par fréquence inverse pour corriger le déséquilibre des classes
    genre_freq  = labels.mean(axis=0)
    pos_weights = torch.tensor(
        (1 - genre_freq) / (genre_freq + 1e-6), dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if eval_only:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        _, f1_micro, f1_macro, preds, gt = evaluate(
            model, val_loader, criterion, device, CFG["threshold"]
        )
        log.info(f"F1 micro : {f1_micro:.4f}  |  F1 macro : {f1_macro:.4f}")
        print(classification_report(gt, preds, target_names=mlb.classes_, zero_division=0))
        return

    # ── Boucle d'entraînement ──
    best_f1    = 0.0
    no_improve = 0

    for epoch in range(1, CFG["epochs"] + 1):

        # Phase 2 : dégeler le backbone à mi-entraînement
        if epoch == CFG["epochs"] // 2 + 1:
            model.image_branch.unfreeze()
            # Réduire le LR pour le fine-tuning du CNN
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
            log.info("Phase 2 : backbone EfficientNet dégelé (LR ×0.1)")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler
        )
        val_loss, f1_micro, f1_macro, _, _ = evaluate(
            model, val_loader, criterion, device, CFG["threshold"]
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:02d}/{CFG['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"F1_micro={f1_micro:.4f}  F1_macro={f1_macro:.4f}"
        )

        # Sauvegarde du meilleur modèle (F1 micro)
        if f1_micro > best_f1:
            best_f1 = f1_micro
            torch.save(model.state_dict(), CHECKPOINT)
            log.info(f"  ✓ Nouveau meilleur modèle sauvegardé (F1={best_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                log.info(f"Early stopping à l'epoch {epoch}")
                break

    # ── Rapport final ──
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    _, f1_micro, f1_macro, preds, gt = evaluate(
        model, val_loader, criterion, device, CFG["threshold"]
    )
    log.info(f"\nMeilleur modèle — F1 micro : {f1_micro:.4f}  F1 macro : {f1_macro:.4f}")
    print(classification_report(gt, preds, target_names=mlb.classes_, zero_division=0))


# ──────────────────────────────────────────────
# INFÉRENCE SUR UN FILM UNIQUE
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_single(
    title: str,
    overview: str,
    poster_url: str,
    popularity: float,
    vote_avg: float,
    vote_count: int,
    language: str,
):
    """Prédit les genres d'un film à partir de ses données brutes."""
    import requests, pickle
    from PIL import Image
    from io import BytesIO
    from sentence_transformers import SentenceTransformer
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)
    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        data = pickle.load(f)

    num_dim    = data["num_features"].shape[1]
    num_genres = data["labels"].shape[1]

    # Texte
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_emb   = text_model.encode([f"{title}. {overview}"], normalize_embeddings=True)
    text_tensor = torch.tensor(text_emb, dtype=torch.float32).to(device)

    # Image
    resp = requests.get(poster_url, timeout=10)
    img  = Image.open(BytesIO(resp.content)).convert("RGB")
    tfm  = MovieDataset.EVAL_TRANSFORMS
    img_tensor = tfm(img).unsqueeze(0).to(device)

    # Numérique (approximation : on suppose la langue est connue)
    num_raw    = np.array([[popularity, vote_avg, vote_count]], dtype=np.float32)
    num_scaled = scaler.transform(num_raw)
    # Lang one-hot (rempli à zéro si inconnue)
    lang_cols  = data["lang_cols"]
    lang_vec   = np.zeros((1, len(lang_cols)), dtype=np.float32)
    col_name   = f"lang_{language}"
    if col_name in lang_cols:
        lang_vec[0, lang_cols.index(col_name)] = 1.0
    num_tensor = torch.tensor(
        np.concatenate([num_scaled, lang_vec], axis=1), dtype=torch.float32
    ).to(device)

    # Modèle
    model = MultiModalGenreClassifier(
        text_dim=384, num_dim=num_dim, num_genres=num_genres,
        proj_dim=CFG["proj_dim"], fusion_hidden=CFG["fusion_hidden"],
        dropout=CFG["dropout"], freeze_image=False,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    logits = model(text_tensor, img_tensor, num_tensor)
    probs  = torch.sigmoid(logits).cpu().numpy()[0]

    results = sorted(zip(mlb.classes_, probs), key=lambda x: -x[1])
    print(f"\nPrédictions pour « {title} »:")
    for genre, prob in results:
        bar = "█" * int(prob * 20)
        print(f"  {genre:<25} {prob:.2f}  {bar}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Évaluation uniquement (charge best_model.pt)")
    args = parser.parse_args()
    train(eval_only=args.eval)
