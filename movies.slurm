#!/bin/bash
#SBATCH --job-name=movies-genre-prediction
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu_t4
#SBATCH --time=04:00:00
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/error_%j.txt

# Arrêter le script en cas d'erreur
set -e

# Charger les modules (adapter selon le cluster)
module purge
module load python/3.11.5
module load cuda/12.1.0

printf "\n========================================\n"
printf "  Job ID      : %s\n" "$SLURM_JOB_ID"
printf "  Node        : %s\n" "$SLURMD_NODENAME"
printf "  Démarré le  : %s\n" "$(date)"
printf "========================================\n\n"

# --- 1. Environnement virtuel ---
printf "[1/5] Création / activation du venv...\n"
[ -d ~/movies/venv ] || python3 -m venv ~/movies/venv
source ~/movies/venv/bin/activate

# --- 2. Dépendances ---
printf "[2/5] Installation des dépendances...\n"
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet pandas scikit-learn Pillow requests tqdm sentence-transformers

# --- 3. Preprocessing ---
printf "[3/5] Construction du dataset...\n"
python3 preprocessing.py

# --- 4. Entraînement ---
printf "[4/5] Entraînement du modèle...\n"
python3 train.py

# --- 5. Évaluation + plots ---
printf "[5/5] Évaluation et génération des graphiques...\n"
python3 train.py --eval
python3 plot.py

printf "\n========================================\n"
printf "  Terminé le  : %s\n" "$(date)"
printf "========================================\n"
