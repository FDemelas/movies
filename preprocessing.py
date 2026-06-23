"""
1_prepare_dataset.py
====================
Télécharge les images depuis les URLs TMDB, applique les transformations,
et construit un dataset PyTorch prêt pour l'entraînement.

Structure de sortie :
    data/
        images/          ← images .jpg téléchargées
        dataset.pkl      ← DataFrame enrichi (embeddings texte + chemin image)
        label_binarizer.pkl  ← MultiLabelBinarizer fitté sur les genres
"""

import os
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

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sentence_transformers import SentenceTransformer

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CSV_PATH       = "movies.csv"          # votre fichier CSV
OUTPUT_DIR     = Path("data")
IMAGES_DIR     = OUTPUT_DIR / "images"
IMG_SIZE       = 224                   # taille cible pour le CNN
TEXT_MODEL     = "all-MiniLM-L6-v2"   # sentence-transformer léger (384 dim)
DOWNLOAD_DELAY = 0.2                   # secondes entre requêtes (respecter TMDB)
MAX_RETRIES    = 3

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# REMPLACER download_all_images par :
def load_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les chemins sans re-télécharger."""
    paths, flags = [], []
    for idx in df.index:
        filename = IMAGES_DIR / f"{idx:05d}.jpg"
        exists = filename.exists()
        paths.append(str(filename) if exists else None)
        flags.append(exists)

    df["image_path"] = paths
    df["image_ok"]   = flags

    n_ok = sum(flags)
    log.info(f"Images trouvées : {n_ok}/{len(df)} ({n_ok/len(df)*100:.1f}%)")
    df = df[df["image_ok"]].reset_index(drop=True)
    return df
    
# ──────────────────────────────────────────────
# ÉTAPE 1 : Chargement et nettoyage du CSV
# ──────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(
     csv_path,
     engine="python",          # parser plus tolérant que le parser C
     quoting=0,                # QUOTE_MINIMAL — respecte les guillemets doubles
     on_bad_lines="skip",      # ignore les lignes vraiment malformées
     encoding="utf-8",
     encoding_errors="replace", # remplace les caractères invalides plutôt que crasher
    )

    log.info(f"Dataset chargé : {len(df)} films")

    # Supprimer les lignes sans genre ni URL
    df = df.dropna(subset=["Genre", "Poster_Url"])

    # Genres : "Action, Adventure" → ["Action", "Adventure"]
    df["genre_list"] = df["Genre"].apply(
        lambda g: [x.strip() for x in g.replace('"', "").split(",")]
    )

    # Texte combiné titre + description
    df["text_input"] = (
        df["Title"].fillna("") + ". " + df["Overview"].fillna("")
    ).str.strip()

    # Features numériques — remplir les NaN par la médiane
    num_cols = ["Popularity", "Vote_Average", "Vote_Count"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Langue → one-hot (garder les 10 plus fréquentes)
    top_langs = df["Original_Language"].value_counts().head(10).index.tolist()
    df["lang_clean"] = df["Original_Language"].apply(
        lambda x: x if x in top_langs else "other"
    )
    lang_dummies = pd.get_dummies(df["lang_clean"], prefix="lang")
    df = pd.concat([df, lang_dummies], axis=1)

    log.info(f"Après nettoyage : {len(df)} films")
    return df


# ──────────────────────────────────────────────
# ÉTAPE 2 : Téléchargement des images
# ──────────────────────────────────────────────
def download_image(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Télécharge une image et la sauvegarde. Retourne True si succès."""
    if dest.exists():
        return True  # déjà téléchargée
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(dest, format="JPEG", quality=90)
            return True
        except (requests.RequestException, UnidentifiedImageError) as e:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                log.warning(f"Échec téléchargement {url} : {e}")
    return False


def download_all_images(df: pd.DataFrame) -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    paths, success_flags = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Téléchargement images"):
        # Nom de fichier basé sur l'index pour éviter les collisions
        filename = IMAGES_DIR / f"{row.name:05d}.jpg"
        ok = download_image(row["Poster_Url"], filename)
        paths.append(str(filename) if ok else None)
        success_flags.append(ok)
        time.sleep(DOWNLOAD_DELAY)

    df["image_path"] = paths
    df["image_ok"]   = success_flags

    n_ok = sum(success_flags)
    log.info(f"Images téléchargées : {n_ok}/{len(df)} ({n_ok/len(df)*100:.1f}%)")

    # Conserver uniquement les films avec image valide
    df = df[df["image_ok"]].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# ÉTAPE 3 : Encodage texte (sentence-transformers)
# ──────────────────────────────────────────────
def encode_texts(df: pd.DataFrame) -> np.ndarray:
    log.info(f"Chargement du modèle texte : {TEXT_MODEL}")
    model = SentenceTransformer(TEXT_MODEL)

    log.info("Encodage des textes (titre + description)...")
    embeddings = model.encode(
        df["text_input"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-norm → cosine similarity
    )
    log.info(f"Embeddings texte : shape {embeddings.shape}")
    return embeddings  # (N, 384)


# ──────────────────────────────────────────────
# ÉTAPE 4 : Normalisation des features numériques
# ──────────────────────────────────────────────
def build_numerical_features(df: pd.DataFrame):
    num_cols  = ["Popularity", "Vote_Average", "Vote_Count"]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])

    lang_cols = [c for c in df.columns if c.startswith("lang_") and c != "lang_clean"]
    lang_matrix = df[lang_cols].astype(float).values.astype(np.float32)

    num_features = np.concatenate([scaled, lang_matrix], axis=1).astype(np.float32)

    log.info(f"Features numériques : shape {num_features.shape}")
    return num_features, scaler, lang_cols


# ──────────────────────────────────────────────
# ÉTAPE 5 : Binarisation des labels (multi-label)
# ──────────────────────────────────────────────
def binarize_labels(df: pd.DataFrame):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df["genre_list"])
    log.info(f"Genres détectés ({len(mlb.classes_)}) : {list(mlb.classes_)}")
    log.info(f"Labels shape : {labels.shape}")
    return labels, mlb


# ──────────────────────────────────────────────
# DATASET PYTORCH
# ──────────────────────────────────────────────
class MovieDataset(Dataset):
    """
    Dataset multimodal.
    Retourne (text_emb, image_tensor, num_features, label_vector)
    """

    # Transformations image : entraînement (avec augmentation)
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                             std=[0.229, 0.224, 0.225]),
    ])

    # Transformations image : validation / test (sans augmentation)
    EVAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        image_paths:   list,
        text_embeds:   np.ndarray,
        num_features:  np.ndarray,
        labels:        np.ndarray,
        train:         bool = True,
    ):
        self.image_paths  = image_paths
        self.text_embeds  = torch.tensor(text_embeds,  dtype=torch.float32)
        self.num_features = torch.tensor(num_features, dtype=torch.float32)
        self.labels       = torch.tensor(labels,       dtype=torch.float32)
        self.transform    = self.TRAIN_TRANSFORMS if train else self.EVAL_TRANSFORMS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.transform(img)

        return {
            "text":    self.text_embeds[idx],      # (384,)
            "image":   img_tensor,                  # (3, 224, 224)
            "numeric": self.num_features[idx],      # (num_features_dim,)
            "label":   self.labels[idx],            # (num_genres,)
        }


# ──────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ──────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    df = load_and_clean(CSV_PATH)

    # 2. Images
    df = download_all_images(df)
    #df = load_existing_images(df)

    # 3. Encodage texte
    text_embeddings = encode_texts(df)

    # 4. Features numériques
    
    
   
    num_features, scaler, lang_cols = build_numerical_features(df)
    
    # S'assurer qu'on ne prend que les colonnes one-hot, pas lang_clean
    
    lang_cols = [c for c in df.columns if c.startswith("lang_") and c != "lang_clean"]

    # 5. Labels
    labels, mlb = binarize_labels(df)

    # 6. Sauvegarde
    dataset_payload = {
        "df":              df,
        "text_embeddings": text_embeddings,
        "num_features":    num_features,
        "labels":          labels,
        "lang_cols":       lang_cols,
    }
    with open(OUTPUT_DIR / "dataset.pkl", "wb") as f:
        pickle.dump(dataset_payload, f)

    with open(OUTPUT_DIR / "label_binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)

    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    log.info("Fichiers sauvegardés dans data/")
    log.info(f"  dataset.pkl         — {len(df)} films")
    log.info(f"  label_binarizer.pkl — {len(mlb.classes_)} genres")
    log.info(f"  scaler.pkl          — StandardScaler pour les features numériques")

    # 7. Vérification rapide : instancier le Dataset
    from sklearn.model_selection import train_test_split
    idx = list(range(len(df)))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_ds = MovieDataset(
        image_paths  = df.iloc[train_idx]["image_path"].tolist(),
        text_embeds  = text_embeddings[train_idx],
        num_features = num_features[train_idx],
        labels       = labels[train_idx],
        train        = True,
    )
    val_ds = MovieDataset(
        image_paths  = df.iloc[val_idx]["image_path"].tolist(),
        text_embeds  = text_embeddings[val_idx],
        num_features = num_features[val_idx],
        labels       = labels[val_idx],
        train        = False,
    )

    log.info(f"Train : {len(train_ds)} films  |  Val : {len(val_ds)} films")
    sample = train_ds[0]
    log.info(
        f"Sample shapes — text: {sample['text'].shape}  "
        f"image: {sample['image'].shape}  "
        f"numeric: {sample['numeric'].shape}  "
        f"label: {sample['label'].shape}"
    )
    log.info("Preprocessing terminé.")


if __name__ == "__main__":
    main()
