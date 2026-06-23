"""
train.py
========
Entraînement du classifieur de genre à partir des affiches.

  - Phase 1 : backbone gelé, seule la tête s'entraîne (rapide)
  - Phase 2 : fine-tuning des derniers blocs EfficientNet (LR réduit)
  - BCEWithLogitsLoss avec pondération par fréquence inverse
  - Early stopping + checkpoint du meilleur modèle (F1 micro)
  - Rapport détaillé par genre à la fin

Usage :
    python train.py                  # entraînement complet
    python train.py --epochs 5       # test rapide
    python train.py --eval           # rapport sur val set
    python train.py --predict URL    # inférence sur une affiche
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

from preprocessing import PosterDataset
from model import PosterGenreClassifier

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR   = Path("data")
CHECKPOINT = DATA_DIR / "best_model.pt"

CFG = {
    "dropout":        0.4,
    "lr":             3e-4,    # phase 1 : tête seule
    "lr_finetune":    3e-5,    # phase 2 : fine-tuning CNN
    "weight_decay":   1e-4,
    "epochs":         30,
    "batch_size":     32,
    "patience":       5,
    "threshold":      0.5,
    "val_split":      0.2,
    "num_workers":    4,
    "unfreeze_blocks": 3,      # nb de blocs EfficientNet dégelés en phase 2
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s │ %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "training.log"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# DONNÉES
# ──────────────────────────────────────────────
def load_data():
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        data = pickle.load(f)
    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    df     = data["df"]
    labels = data["labels"]

    log.info(f"Films   : {len(df)}")
    log.info(f"Genres  : {len(mlb.classes_)} → {list(mlb.classes_)}")
    return df, labels, mlb


def build_loaders(df, labels):
    idx = list(range(len(df)))
    train_idx, val_idx = train_test_split(idx, test_size=CFG["val_split"], random_state=42)

    paths = df["image_path"].tolist()

    train_ds = PosterDataset([paths[i] for i in train_idx], labels[train_idx], train=True)
    val_ds   = PosterDataset([paths[i] for i in val_idx],   labels[val_idx],   train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=CFG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

    log.info(f"Train : {len(train_ds)}  |  Val : {len(val_ds)}")
    return train_loader, val_loader


# ──────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────
def weighted_bce(labels: np.ndarray, device: torch.device) -> nn.BCEWithLogitsLoss:
    freq        = labels.mean(axis=0).clip(1e-3, 1 - 1e-3)
    pos_weights = torch.tensor((1 - freq) / freq, dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# ──────────────────────────────────────────────
# BOUCLES
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, amp_scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        total_loss += criterion(logits, labels).item()

        preds = (torch.sigmoid(logits) >= CFG["threshold"]).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_micro   = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(loader), f1_micro, f1_macro, all_preds, all_labels


# ──────────────────────────────────────────────
# ENTRAÎNEMENT
# ──────────────────────────────────────────────
def train(n_epochs: int = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    df, labels, mlb = load_data()
    train_loader, val_loader = build_loaders(df, labels)

    num_genres = labels.shape[1]
    epochs     = n_epochs or CFG["epochs"]
    phase2_ep  = epochs // 2 + 1

    model = PosterGenreClassifier(
        num_genres = num_genres,
        dropout    = CFG["dropout"],
        freeze     = True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Paramètres entraînables (phase 1) : {n_params:,}")

    criterion  = weighted_bce(labels, device)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1, no_improve = 0.0, 0
    history = []

    log.info("─" * 60)
    log.info(f"Démarrage : {epochs} epochs | batch={CFG['batch_size']} | patience={CFG['patience']}")
    log.info("─" * 60)

    for epoch in range(1, epochs + 1):

        # Phase 2 : dégeler les derniers blocs
        if epoch == phase2_ep:
            model.unfreeze(blocks_from_end=CFG["unfreeze_blocks"])
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CFG["lr_finetune"], weight_decay=CFG["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1
            )
            n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Phase 2 : {CFG['unfreeze_blocks']} blocs dégelés — {n_total:,} params — LR={CFG['lr_finetune']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_loss, f1_micro, f1_macro, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"F1_micro={f1_micro:.4f}  F1_macro={f1_macro:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "f1_micro": f1_micro, "f1_macro": f1_macro})

        if f1_micro > best_f1:
            best_f1 = f1_micro
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "num_genres": num_genres, "f1_micro": f1_micro,
            }, CHECKPOINT)
            log.info(f"  ✓ Checkpoint sauvegardé (F1={best_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                log.info(f"Early stopping à l'epoch {epoch}")
                break

    with open(DATA_DIR / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    log.info("─" * 60)
    eval_best(mlb, labels)


# ──────────────────────────────────────────────
# ÉVALUATION
# ──────────────────────────────────────────────
def eval_best(mlb=None, labels=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CHECKPOINT, map_location=device)

    if mlb is None:
        with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
            mlb = pickle.load(f)

    df, labels, _ = load_data()
    _, val_loader = build_loaders(df, labels)

    model = PosterGenreClassifier(
        num_genres=ckpt["num_genres"], dropout=CFG["dropout"], freeze=False
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    criterion = weighted_bce(labels, device)
    _, f1_micro, f1_macro, preds, gt = evaluate(model, val_loader, criterion, device)

    log.info(f"F1 micro : {f1_micro:.4f}  |  F1 macro : {f1_macro:.4f}")
    print("\n" + classification_report(gt, preds, target_names=mlb.classes_,
                                       zero_division=0, digits=3))


# ──────────────────────────────────────────────
# INFÉRENCE SUR UNE AFFICHE
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_from_url(poster_url: str):
    import requests as req
    from PIL import Image
    from io import BytesIO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CHECKPOINT, map_location=device)

    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    model = PosterGenreClassifier(
        num_genres=ckpt["num_genres"], dropout=CFG["dropout"], freeze=False
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img    = Image.open(BytesIO(req.get(poster_url, timeout=10).content)).convert("RGB")
    tensor = PosterDataset.EVAL_TRANSFORMS(img).unsqueeze(0).to(device)

    probs  = torch.sigmoid(model(tensor)).cpu().numpy()[0]

    print(f"\nPrédictions — {poster_url}")
    print("─" * 50)
    for genre, p in sorted(zip(mlb.classes_, probs), key=lambda x: -x[1]):
        bar  = "█" * int(p * 30)
        flag = " ✓" if p >= CFG["threshold"] else ""
        print(f"  {genre:<25} {p:.2f}  {bar}{flag}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--predict", type=str, default=None,
                        metavar="URL", help="URL d'une affiche à prédire")
    args = parser.parse_args()

    if args.eval:
        eval_best()
    elif args.predict:
        predict_from_url(args.predict)
    else:
        train(n_epochs=args.epochs)
