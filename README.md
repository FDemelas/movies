Install dependencies:
```python
pip install -r requirements.txt
```

Construct the dataset:
```python
python3 preprocessing.py
```

Train the model
```python
python3 train.py                    # entraînement complet
```

Evaluate the model and print all the results:
```python
python3 train.py --eval             # rapport par genre
```

Plot 4 figures summarizing the resullts:
```python
python3 plot.py
```
