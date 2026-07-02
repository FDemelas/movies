"""
plot_results.py
===============
Génère les figures d'évaluation depuis le checkpoint entraîné.

Usage :
    python plot_results.py          # génère toutes les figures
    python plot_results.py --history  # ajoute aussi la courbe d'apprentissage

Sorties dans data/figures/ :
    fig1_metrics_par_genre.jpg
    fig2_precision_recall_scatter.jpg
    fig3_f1_vs_support.jpg
    fig4_heatmap.jpg
    fig5_learning_curves.jpg  (si --history)
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from preprocessing import PosterDataset
from model import PosterGenreClassifier

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR    = Path("data")
FIGURES_DIR = DATA_DIR / "figures"
CHECKPOINT  = DATA_DIR / "best_model.pt"
THRESHOLD   = 0.5
BATCH_SIZE  = 32
NUM_WORKERS = 4

STYLE = {
    "prec":  "#3266AD",
    "rec":   "#1D9E75",
    "f1":    "#D85A30",
    "light": "#F5F5F3",
    "grid":  "#E0DED8",
    "text":  "#2C2C2A",
}
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         True,
    "grid.color":        STYLE["grid"],
    "grid.linewidth":    0.5,
    "text.color":        STYLE["text"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["text"],
    "ytick.color":       STYLE["text"],
})


# ──────────────────────────────────────────────
# ÉTAPE 1 : charger le modèle et calculer les métriques
# ──────────────────────────────────────────────
def compute_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Données — on évite de charger le DataFrame pour contourner
    # les incompatibilités de version pandas entre machines
    with open(DATA_DIR / "dataset.pkl", "rb") as f:
        data = pickle.load(f)
    with open(DATA_DIR / "label_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    # Supporte les deux formats de pickle :
    #   - ancien : {"df": DataFrame, "labels": array}
    #   - nouveau : {"image_paths": list, "labels": array}
    if "image_paths" in data:
        paths  = data["image_paths"]
        labels = data["labels"]
    else:
        df     = data["df"]
        labels = data["labels"]
        paths  = df["image_path"].tolist()

    idx = list(range(len(paths)))
    _, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    val_ds = PosterDataset(
        [paths[i] for i in val_idx],
        labels[val_idx],
        train=False,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Modèle
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = PosterGenreClassifier(
        num_genres=ckpt["num_genres"], dropout=0.4, freeze=False
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            preds  = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["label"].numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )

    print(classification_report(
        all_labels, all_preds,
        target_names=mlb.classes_, zero_division=0, digits=3
    ))

    return mlb.classes_, prec, rec, f1, sup.astype(int)


# ──────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────
def fig1_barres(genres, prec, rec, f1, sup):
    order   = np.argsort(f1)[::-1]
    gs, ps, rs, fs = ([x[i] for i in order]
                      for x in [genres, prec, rec, f1])
    n = len(gs)
    y = np.arange(n)
    h = 0.25

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(STYLE["light"])

    ax.barh(y + h, ps, h, label="Précision", color=STYLE["prec"], alpha=0.9)
    ax.barh(y,     rs, h, label="Rappel",    color=STYLE["rec"],  alpha=0.9)
    ax.barh(y - h, fs, h, label="F1",        color=STYLE["f1"],   alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(gs, fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score", fontsize=11)
    ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.set_title("Précision, Rappel et F1 par genre\n(modèle image-only, EfficientNet-B0)",
                 fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_metrics_par_genre.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {path}")


def fig2_scatter(genres, prec, rec, f1, sup):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(STYLE["light"])

    sizes = [max(s / max(sup) * 800, 40) for s in sup]
    sc = ax.scatter(prec, rec, s=sizes, c=f1, cmap="RdYlGn",
                    vmin=0, vmax=0.9, alpha=0.85,
                    edgecolors="white", linewidths=0.8)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("F1", fontsize=10)

    ax.axhline(0.5, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)

    for i, g in enumerate(genres):
        if f1[i] >= 0.5 or f1[i] <= 0.15:
            ax.annotate(g, (prec[i], rec[i]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color=STYLE["text"], alpha=0.9)

    ax.set_xlabel("Précision", fontsize=11)
    ax.set_ylabel("Rappel", fontsize=11)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Précision vs Rappel par genre\n(taille du point = nombre d'exemples)",
                 fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_precision_recall_scatter.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {path}")


def fig3_f1_support(genres, f1, sup):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(STYLE["light"])

    ax.scatter(sup, f1, s=60, color=STYLE["f1"], alpha=0.85,
               edgecolors="white", linewidths=0.8)

    z  = np.polyfit(np.log(sup + 1), f1, 1)
    xs = np.linspace(min(sup), max(sup), 200)
    ax.plot(xs, z[0] * np.log(xs + 1) + z[1],
            color=STYLE["prec"], linewidth=1.5, linestyle="--",
            alpha=0.7, label="Tendance log")

    for i, g in enumerate(genres):
        ax.annotate(g, (sup[i], f1[i]),
                    textcoords="offset points", xytext=(5, 2),
                    fontsize=8, color=STYLE["text"], alpha=0.85)

    ax.set_xlabel("Nombre d'exemples (support)", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("F1 en fonction du nombre d'exemples par genre",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_f1_vs_support.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {path}")


def fig4_heatmap(genres, prec, rec, f1):
    order  = np.argsort(f1)[::-1]
    gs     = [genres[i] for i in order]
    matrix = np.array([[prec[i], rec[i], f1[i]] for i in order])
    n      = len(gs)

    fig, ax = plt.subplots(figsize=(7, 8))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", extent=[0, 3, 0, n])

    for i in range(n):
        for j, val in enumerate(matrix[i]):
            fc = "white" if val < 0.35 or val > 0.75 else STYLE["text"]
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=fc)

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Précision", "Rappel", "F1"], fontsize=11)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(gs, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax.set_title("Heatmap des métriques par genre",
                 fontsize=13, fontweight="bold", pad=28)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_heatmap.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {path}")


def fig5_learning_curves():
    history_path = DATA_DIR / "history.pkl"
    if not history_path.exists():
        print("data/history.pkl introuvable — fig5 ignorée.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    f1_micro   = [h["f1_micro"]   for h in history]
    f1_macro   = [h["f1_macro"]   for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    for ax in (ax1, ax2):
        ax.set_facecolor(STYLE["light"])

    # Loss
    ax1.plot(epochs, train_loss, color=STYLE["prec"], linewidth=2, label="Train loss")
    ax1.plot(epochs, val_loss,   color=STYLE["f1"],   linewidth=2,
             linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("BCE Loss", fontsize=11)
    ax1.set_title("Courbes de loss", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(fontsize=10)

    # F1
    ax2.plot(epochs, f1_micro, color=STYLE["prec"], linewidth=2, label="F1 micro")
    ax2.plot(epochs, f1_macro, color=STYLE["rec"],  linewidth=2,
             linestyle="--", label="F1 macro")
    best_ep = epochs[int(np.argmax(f1_micro))]
    ax2.axvline(best_ep, color="#888", linewidth=0.8, linestyle=":", alpha=0.7,
                label=f"Meilleur checkpoint (ep {best_ep})")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("F1", fontsize=11)
    ax2.set_title("F1 micro et macro", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(fontsize=10)

    plt.suptitle("Courbes d'apprentissage — EfficientNet-B0",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig5_learning_curves.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {path}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true",
                        help="Générer aussi la courbe d'apprentissage (nécessite data/history.pkl)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    genres, prec, rec, f1, sup = compute_metrics()

    fig1_barres(genres, prec, rec, f1, sup)
    fig2_scatter(genres, prec, rec, f1, sup)
    fig3_f1_support(genres, f1, sup)
    fig4_heatmap(genres, prec, rec, f1)

    if args.history:
        fig5_learning_curves()

    print(f"\nToutes les figures sont dans {FIGURES_DIR}/")
