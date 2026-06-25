#!/usr/bin/env bash

#SBATCH --job-name=MOVIES
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=1     
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1       
#SBATCH --qos=qos_gpu_t4        
#SBATCH --output=./Output_movies.txt    
#SBATCH --error=./Error_movies.txt 

echo "\n\n Installing Requirement \n\n"

# Créer le venv
python3 -m venv ~/movies/venv

# L'activer
source ~/movies/venv/bin/activate

# Installer les dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn Pillow requests tqdm sentence-transformers

#pip install -r requirements.txt

echo "\n\n Preprocessing Dataset \n\n"

python3 preprocessing.py          # construit data/

echo "\n\n Training the model\n\n"

python3 train.py                    # entraînement complet

#python3 train.py --epochs 5         # test rapide

echo "\n\n Evaluation\n\n"

python3 train.py --eval             # rapport par genre


