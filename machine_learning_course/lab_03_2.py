from sklearn import datasets
import pandas as pd
import numpy as np

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
print(X.describe())  # Opis

print(X.head())  # 5 pierwszych próbek
print(X.tail())  # 5 ostatnich próbek
