"""
train.py
==========
Entraînement du modèle multimodal avec :
  - early stopping + checkpoint du meilleur modèle
  - LR scheduler cosine
  - fine-tuning progressif du CNN (2 phases)
  - pondération des classes déséquilibrées
  - rapport de classification détaillé par genre

Usage :
    python train.py                        # entraînement complet
    python train.py --epochs 10            # entraînement court (debug)
    python train.py --eval                 # évaluation uniquement
    python train.py --predict              # inférence sur un exemple
"""

import argparse
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing import MovieDataset
from model import MultiModalGenreClassifier

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR   = Path("data")
CHECKPOINT = DATA_DIR / "best_model.pt"
LOG_FILE   = DATA_DIR / "training.log"

CFG = {
    "text_dim":      384,
    "proj_dim":      256,
    "fusion_hidden": 512,
    "dropout":       0.4,
    "lr":            3e-4,
    "lr_finetune":   3e-5,    # LR réduit pour la phase 2 (fine-tuning CNN)
    "weight_decay":  1e-4,
    "epochs":        30,
    "batch_size":    32,
    "patience":      5,
    "threshold":     0.5,
    "val_split":     0.2,
    "num_workers":   4,
    "phase2_epoch":  None,    # None = milieu automatique
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s │ %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE if DATA_DIR.exists() else "/tmp/training.log"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────
def load_data():
    log.info("Chargement de data/dataset.pkl ...")
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        data = pickle.load(f)
    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    df             = data["df"]
    text_embeds    = data["text_embeddings"]
    num_features   = data["num_features"]
    labels         = data["labels"]
    feature_names  = data.get("feature_names", [])

    log.info(f"  Films              : {len(df)}")
    log.info(f"  Texte dim          : {text_embeds.shape[1]}")
    log.info(f"  Features num dim   : {num_features.shape[1]}")
    log.info(f"  Genres             : {len(mlb.classes_)} → {list(mlb.classes_)}")
    if feature_names:
        log.info(f"  Features : {feature_names}")

    return df, text_embeds, num_features, labels, mlb


def build_loaders(df, text_embeds, num_features, labels):
    idx = list(range(len(df)))
    train_idx, val_idx = train_test_split(
        idx, test_size=CFG["val_split"], random_state=42
    )

    paths = df["image_path"].tolist()

    train_ds = MovieDataset(
        image_paths  = [paths[i] for i in train_idx],
        text_embeds  = text_embeds[train_idx],
        num_features = num_features[train_idx],
        labels       = labels[train_idx],
        train        = True,
    )
    val_ds = MovieDataset(
        image_paths  = [paths[i] for i in val_idx],
        text_embeds  = text_embeds[val_idx],
        num_features = num_features[val_idx],
        labels       = labels[val_idx],
        train        = False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=CFG["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=CFG["num_workers"], pin_memory=True,
    )

    log.info(f"  Train : {len(train_ds)}  |  Val : {len(val_ds)}")
    return train_loader, val_loader, train_idx, val_idx


# ──────────────────────────────────────────────
# LOSS AVEC PONDÉRATION DES CLASSES
# ──────────────────────────────────────────────
def weighted_bce(labels: np.ndarray, device: torch.device) -> nn.BCEWithLogitsLoss:
    """
    Pondération par fréquence inverse pour corriger le déséquilibre.
    Un genre présent dans 5% des films a un poids ~19x plus élevé
    qu'un genre présent dans 95% des films.
    """
    freq        = labels.mean(axis=0).clip(1e-3, 1 - 1e-3)
    pos_weights = torch.tensor((1 - freq) / freq, dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# ──────────────────────────────────────────────
# BOUCLE TRAIN
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, amp_scaler):
    model.train()
    total_loss = 0.0

    for batch in loader:
        text    = batch["text"].to(device, non_blocking=True)
        image   = batch["image"].to(device, non_blocking=True)
        numeric = batch["numeric"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(text, image, numeric)
            loss   = criterion(logits, labels)

        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# BOUCLE ÉVALUATION
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        text    = batch["text"].to(device, non_blocking=True)
        image   = batch["image"].to(device, non_blocking=True)
        numeric = batch["numeric"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)

        logits = model(text, image, numeric)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= CFG["threshold"]).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(loader), f1_micro, f1_macro, all_preds, all_labels


# ──────────────────────────────────────────────
# ENTRAÎNEMENT COMPLET
# ──────────────────────────────────────────────
def train(n_epochs: int = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    df, text_embeds, num_features, labels, mlb = load_data()
    train_loader, val_loader, _, _ = build_loaders(df, text_embeds, num_features, labels)

    num_genres = labels.shape[1]
    num_dim    = num_features.shape[1]
    epochs     = n_epochs or CFG["epochs"]
    phase2_ep  = CFG["phase2_epoch"] or (epochs // 2 + 1)

    # ── Modèle ──
    model = MultiModalGenreClassifier(
        text_dim      = CFG["text_dim"],
        num_dim       = num_dim,
        num_genres    = num_genres,
        proj_dim      = CFG["proj_dim"],
        fusion_hidden = CFG["fusion_hidden"],
        dropout       = CFG["dropout"],
        freeze_image  = True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Paramètres entraînables (phase 1) : {n_params:,}")

    # ── Loss ──
    criterion = weighted_bce(labels, device)

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history    = []
    best_f1    = 0.0
    no_improve = 0

    log.info("─" * 60)
    log.info(f"Démarrage : {epochs} epochs  |  batch={CFG['batch_size']}  |  patience={CFG['patience']}")
    log.info("─" * 60)

    for epoch in range(1, epochs + 1):

        # ── Phase 2 : dégeler le CNN ──
        if epoch == phase2_ep:
            model.image_branch.unfreeze()
            # Reconstruire l'optimizer pour inclure les nouveaux paramètres
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CFG["lr_finetune"],
                weight_decay=CFG["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1
            )
            n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Phase 2 : backbone dégelé — {n_total:,} params — LR={CFG['lr_finetune']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_loss, f1_micro, f1_macro, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"F1_micro={f1_micro:.4f}  F1_macro={f1_macro:.4f}  "
            f"lr={lr_now:.2e}"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "f1_micro": f1_micro, "f1_macro": f1_macro,
        })

        # ── Checkpoint ──
        if f1_micro > best_f1:
            best_f1 = f1_micro
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "cfg":         CFG,
                "num_dim":     num_dim,
                "num_genres":  num_genres,
                "f1_micro":    f1_micro,
            }, CHECKPOINT)
            log.info(f"  ✓ Checkpoint sauvegardé (F1={best_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                log.info(f"Early stopping à l'epoch {epoch}")
                break

    # ── Sauvegarder l'historique ──
    with open(DATA_DIR / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    # ── Rapport final ──
    log.info("─" * 60)
    log.info("Rapport final (meilleur checkpoint) :")
    eval_best(mlb)


# ──────────────────────────────────────────────
# ÉVALUATION DU MEILLEUR CHECKPOINT
# ──────────────────────────────────────────────
def eval_best(mlb=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    log.info(f"Checkpoint epoch {ckpt['epoch']}  F1_micro={ckpt['f1_micro']:.4f}")

    if mlb is None:
        with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
            mlb = pickle.load(f)

    df, text_embeds, num_features, labels, _ = load_data()
    _, val_loader, _, _ = build_loaders(df, text_embeds, num_features, labels)

    model = MultiModalGenreClassifier(
        text_dim      = CFG["text_dim"],
        num_dim       = ckpt["num_dim"],
        num_genres    = ckpt["num_genres"],
        proj_dim      = CFG["proj_dim"],
        fusion_hidden = CFG["fusion_hidden"],
        dropout       = CFG["dropout"],
        freeze_image  = False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    criterion = weighted_bce(labels, device)
    _, f1_micro, f1_macro, preds, gt = evaluate(model, val_loader, criterion, device)

    log.info(f"F1 micro : {f1_micro:.4f}  |  F1 macro : {f1_macro:.4f}")
    print("\n" + classification_report(
        gt, preds, target_names=mlb.classes_, zero_division=0, digits=3
    ))


# ──────────────────────────────────────────────
# INFÉRENCE SUR UN FILM UNIQUE
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_single(
    title:       str,
    overview:    str,
    poster_url:  str,
    popularity:  float,
    vote_avg:    float,
    vote_count:  int,
    language:    str,
    release_date: str = "2020-01-01",
):
    import requests as req
    from PIL import Image
    from io import BytesIO
    from sentence_transformers import SentenceTransformer
    from preprocessing import MovieDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CHECKPOINT, map_location=device)

    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)
    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(DATA_DIR / "feature_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    feature_names = meta["feature_names"]

    # ── Texte ──
    text_model  = SentenceTransformer("all-MiniLM-L6-v2")
    text_emb    = text_model.encode(
        [f"{title}. {overview}"], normalize_embeddings=True
    ).astype(np.float32)

    # ── Image ──
    resp       = req.get(poster_url, timeout=10)
    img        = Image.open(BytesIO(resp.content)).convert("RGB")
    img_tensor = MovieDataset.EVAL_TRANSFORMS(img).unsqueeze(0).to(device)

    # ── Features numériques ──
    import pandas as pd
    date  = pd.Timestamp(release_date)
    month = date.month
    year  = date.year

    # Colonnes dans l'ordre attendu par le scaler
    scale_cols = [
        "Popularity", "Vote_Average", "Vote_Count",
        "year", "decade",
        "title_length", "overview_length",
        "vote_pop_ratio",
    ]
    row_scale = np.array([[
        popularity,
        vote_avg,
        vote_count,
        year,
        year // 10 * 10,
        len(title.split()),
        len(overview.split()),
        vote_count / (popularity + 1e-6),
    ]], dtype=np.float32)
    scaled = scaler.transform(row_scale)

    # Encodage cyclique du mois
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # One-hot langue
    lang_cols = [n for n in feature_names if n.startswith("lang_")]
    lang_vec  = np.zeros((1, len(lang_cols)), dtype=np.float32)
    col_name  = f"lang_{language}"
    if col_name in lang_cols:
        lang_vec[0, lang_cols.index(col_name)] = 1.0

    num_tensor = torch.tensor(
        np.concatenate([scaled, [[month_sin, month_cos]], lang_vec], axis=1),
        dtype=torch.float32,
    ).to(device)

    # ── Modèle ──
    model = MultiModalGenreClassifier(
        text_dim      = CFG["text_dim"],
        num_dim       = ckpt["num_dim"],
        num_genres    = ckpt["num_genres"],
        proj_dim      = CFG["proj_dim"],
        fusion_hidden = CFG["fusion_hidden"],
        dropout       = CFG["dropout"],
        freeze_image  = False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logits = model(
        torch.tensor(text_emb).to(device),
        img_tensor,
        num_tensor,
    )
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    print(f"\nPrédictions pour « {title} »")
    print("─" * 45)
    for genre, p in sorted(zip(mlb.classes_, probs), key=lambda x: -x[1]):
        bar   = "█" * int(p * 30)
        flag  = " ✓" if p >= CFG["threshold"] else ""
        print(f"  {genre:<25} {p:.2f}  {bar}{flag}")


# ──────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement multimodal genre prédiction")
    parser.add_argument("--eval",    action="store_true", help="Évaluation uniquement")
    parser.add_argument("--predict", action="store_true", help="Inférence exemple")
    parser.add_argument("--epochs",  type=int, default=None, help="Nb d'epochs")
    args = parser.parse_args()

    if args.eval:
        eval_best()
    elif args.predict:
        predict_single(
            title        = "Inception",
            overview     = "A thief who steals corporate secrets through the use of dream-sharing technology.",
            poster_url   = "https://image.tmdb.org/t/p/original/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
            popularity   = 87.5,
            vote_avg     = 8.4,
            vote_count   = 34000,
            language     = "en",
            release_date = "2010-07-16",
        )
    else:
        train(n_epochs=args.epochs)
