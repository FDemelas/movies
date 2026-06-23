
pip install -r requirements.txt
python3 preprocessing.py          # construit data/
python3 train.py                    # entraînement complet
python3 train.py --epochs 5         # test rapide
python3 train.py --eval             # rapport par genre
python3 train.py --predict          # exemple Inception

