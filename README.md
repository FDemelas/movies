# Prédiction du genre de films

L'objectif de ce projet est de prédire le genre d'un film uniquement à partir de son affiche.
Il s'agit d'un problème de **classification multi-label** car un film peut appartenir à plusieurs genres simultanément.

---

## Prérequis

- Python 3.8+
- pip

---

## Instructions d'exécution locale

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Construire le dataset

```bash
python3 preprocessing.py
```

### 3. Entraîner le modèle

```bash
python3 train.py
```

### 4. Évaluer le modèle entraîné

Évalue le modèle et affiche les résultats par genre :

```bash
python3 train.py --eval
```

### 5. Générer les graphiques

Produit 4 graphiques illustrant les résultats de l'analyse :

```bash
python3 plot.py
```

---

## Exécution sur un cluster avec Slurm

Ce projet peut être soumis sur un cluster de calcul géré par [Slurm](https://slurm.schedmd.com/) via le script `movies.slurm`.

### Préparer les logs

Créer le dossier de logs avant la première soumission :

```bash
mkdir -p logs
```

### Soumettre le job

```bash
sbatch movies.slurm
```

### Contenu du script `movies.slurm`

Le script enchaîne automatiquement les 5 étapes suivantes :

| Étape | Commande | Description |
|-------|----------|-------------|
| 1 | `python3 -m venv` | Création du venv (si absent) et installation des dépendances |
| 2 | `preprocessing.py` | Construction du dataset |
| 3 | `train.py` | Entraînement complet du modèle |
| 4 | `train.py --eval` | Évaluation et rapport par genre |
| 5 | `plot.py` | Génération des 4 graphiques |

Les directives SBATCH utilisées :

| Directive | Valeur | Description |
|-----------|--------|-------------|
| `--cpus-per-task` | 1 | Nombre de cœurs CPU |
| `--ntasks` | 1 | Une seule tâche |
| `--gres` | `gpu:1` | 1 GPU |
| `--qos` | `qos_gpu_t4` | Qualité de service GPU T4 |
| `--time` | `04:00:00` | Durée maximale du job |
| `--output` | `./logs/Output_movies.txt` | Logs de sortie (remplace `%j` par le job ID) |
| `--error` | `./logs/Error_movies.txt` | Logs d'erreur |

### Commandes Slurm utiles

| Commande | Description |
|----------|-------------|
| `sbatch movies.slurm` | Soumettre le job |
| `squeue -u $USER` | Voir l'état de ses jobs |
| `scancel <job_id>` | Annuler un job |
| `sinfo` | Afficher les partitions disponibles |
| `sacct -j <job_id>` | Consulter l'historique d'un job terminé |
