"""
preprocessing.py
================
Pipeline image-only pour la prédiction de genre à partir des affiches.

Structure de sortie :
    data/
        images/              ← images .jpg téléchargées
        dataset.pkl          ← chemins images + labels
        label_binarizer.pkl  ← MultiLabelBinarizer fitté sur les genres
"""

import time
import pickle
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CSV_PATH       = "movies.csv"
OUTPUT_DIR     = Path("data")
IMAGES_DIR     = OUTPUT_DIR / "images"
IMG_SIZE       = 224
DOWNLOAD_DELAY = 0.2
MAX_RETRIES    = 3

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ÉTAPE 1 : Chargement et nettoyage
# ──────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        engine="python",
        quoting=0,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )
    log.info(f"Dataset chargé : {len(df)} films")

    df = df.dropna(subset=["Genre", "Poster_Url"])

    df["genre_list"] = df["Genre"].apply(
        lambda g: [x.strip() for x in g.replace('"', "").split(",")]
    )

    log.info(f"Après nettoyage : {len(df)} films")
    return df


# ──────────────────────────────────────────────
# ÉTAPE 2 : Images (téléchargement ou chargement)
# ──────────────────────────────────────────────
def download_image(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(dest, format="JPEG", quality=90)
            return True
        except (requests.RequestException, UnidentifiedImageError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 ** attempt)
            else:
                log.warning(f"Échec : {url} — {e}")
    return False


def download_all_images(df: pd.DataFrame) -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    paths, flags = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Téléchargement images"):
        dest = IMAGES_DIR / f"{row.name:05d}.jpg"
        ok   = download_image(row["Poster_Url"], dest)
        paths.append(str(dest) if ok else None)
        flags.append(ok)
        time.sleep(DOWNLOAD_DELAY)
    df["image_path"] = paths
    df["image_ok"]   = flags
    n_ok = sum(flags)
    log.info(f"Images : {n_ok}/{len(df)} ({n_ok/len(df)*100:.1f}%)")
    return df[df["image_ok"]].reset_index(drop=True)


def load_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les chemins sans re-télécharger."""
    paths, flags = [], []
    for idx in df.index:
        dest   = IMAGES_DIR / f"{idx:05d}.jpg"
        exists = dest.exists()
        paths.append(str(dest) if exists else None)
        flags.append(exists)
    df["image_path"] = paths
    df["image_ok"]   = flags
    n_ok = sum(flags)
    log.info(f"Images trouvées : {n_ok}/{len(df)} ({n_ok/len(df)*100:.1f}%)")
    return df[df["image_ok"]].reset_index(drop=True)


# ──────────────────────────────────────────────
# ÉTAPE 3 : Labels
# ──────────────────────────────────────────────
def binarize_labels(df: pd.DataFrame):
    mlb    = MultiLabelBinarizer()
    labels = mlb.fit_transform(df["genre_list"]).astype(np.float32)
    log.info(f"Genres ({len(mlb.classes_)}) : {list(mlb.classes_)}")
    log.info(f"Labels : {labels.shape}")
    return labels, mlb


# ──────────────────────────────────────────────
# DATASET PYTORCH
# ──────────────────────────────────────────────
class PosterDataset(Dataset):
    """
    Dataset image-only.
    Retourne (image_tensor, label_vector).
    """

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    EVAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, image_paths: list, labels: np.ndarray, train: bool = True):
        self.image_paths = image_paths
        self.labels      = torch.tensor(labels, dtype=torch.float32)
        self.transform   = self.TRAIN_TRANSFORMS if train else self.EVAL_TRANSFORMS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "image": self.transform(img),        # (3, 224, 224)
            "label": self.labels[idx],           # (num_genres,)
        }


# ──────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ──────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(CSV_PATH)

    # Commenter/décommenter selon le besoin :
    # df = download_all_images(df)   # premier lancement
    df = load_existing_images(df)    # images déjà téléchargées

    labels, mlb = binarize_labels(df)

    with open(OUTPUT_DIR / "dataset.pkl", "wb") as f:
        pickle.dump({"df": df, "labels": labels}, f)

    with open(OUTPUT_DIR / "label_binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)

    log.info("── Résumé ────────────────────────────────")
    log.info(f"  Films   : {len(df)}")
    log.info(f"  Genres  : {len(mlb.classes_)}")
    log.info("──────────────────────────────────────────")

    # Vérification rapide
    ds     = PosterDataset(df["image_path"].tolist()[:4], labels[:4], train=True)
    sample = ds[0]
    log.info(f"  Sample — image:{sample['image'].shape}  label:{sample['label'].shape}")
    log.info("Preprocessing terminé.")


if __name__ == "__main__":
    main()
