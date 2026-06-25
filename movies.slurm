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

echo "\n\n Preprocessing Dataset \n\n"

# construction du data/
python3 preprocessing.py

echo "\n\n Training the model\n\n"

# Entraînement complet
python3 train.py

echo "\n\n Evaluation\n\n"

# Rapport par genre
python3 train.py --eval             


echo "\n\n Creation des images\n\n"

# Plot des images
python3 plot.py

